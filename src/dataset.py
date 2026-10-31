"""PyTorch datasets of procedurally generated mazes.

Each sample is an (input, target) pair of float tensors for a size x size maze
plus a 1-pixel wall around it, so H = W = size + 2:

    input:  (3, H, W)  channel 0 = maze (1 = passage, 0 = wall),
                       channel 1 = start (1 at one pixel), channel 2 = exit
    target: (1, H, W)  1 on the shortest path from start to exit, inclusive

Start and exit are two distinct random cells. Mazes are generated lazily and
deterministically from (seed, index), so a dataset is reproducible without
storing anything on disk and DataLoader workers need no shared state.

See usage example in dataset_generation_ex.ipynb
"""

from collections import deque
from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from generators import PASSAGE, get_generator, remove_walls


Point = tuple[int, int]


def random_endpoints(maze: np.ndarray, rng: np.random.Generator) -> tuple[Point, Point]:
    """Two distinct random cells (odd pixel coordinates) to use as start and exit."""
    n_cells = (maze.shape[0] - 1) // 2
    a, b = rng.choice(n_cells * n_cells, size=2, replace=False).tolist()
    (ar, ac), (br, bc) = divmod(a, n_cells), divmod(b, n_cells)

    return (2 * ar + 1, 2 * ac + 1), (2 * br + 1, 2 * bc + 1)


def shortest_path(maze: np.ndarray, start: Point, goal: Point) -> np.ndarray:
    """0/1 mask of the BFS shortest path from start to goal through passage pixels."""
    h, w = maze.shape
    open_ = (maze == PASSAGE).ravel()
    parent = np.full(h * w, -1, dtype=np.int64)
    s, g = start[0] * w + start[1], goal[0] * w + goal[1]
    parent[s] = s

    queue = deque([s])
    while queue:
        cur = queue.popleft()
        if cur == g:
            break

        r, c = divmod(cur, w)
        for nxt in (cur - w if r > 0 else -1, cur + w if r < h - 1 else -1,
                    cur - 1 if c > 0 else -1, cur + 1 if c < w - 1 else -1):
            if nxt >= 0 and open_[nxt] and parent[nxt] < 0:
                parent[nxt] = cur
                queue.append(nxt)

    if parent[g] < 0:
        raise ValueError(f"No path from {start} to {goal}")

    path = np.zeros(h * w, dtype=np.uint8)
    cur = g
    while cur != s:
        path[cur] = 1
        cur = parent[cur]
    path[s] = 1

    return path.reshape(h, w)


class MazeDataset(Dataset):
    """
    Mazes split evenly across `generators`.

    Sample i is produced by generators[i % len(generators)], so the generator
    mix is balanced to within one sample (e.g. 1/3 each for three generators).

    Args:
        generators: Generator names (see generators.GENERATORS).
        size: Odd maze width inside the border; 63 gives a 65 x 65 image.
        num_samples: Number of mazes.
        seed: Base seed; sample i is seeded with (seed, i).
        wall_removal: Fraction of interior walls to open after generation,
            turning perfect mazes into mazes with loops. 0 disables.
        exclude_generators: Generators that must not appear in `generators`
            (used to keep test generators disjoint from training ones).
    """

    def __init__(
        self,
        generators: Sequence[str],
        size: int,
        num_samples: int,
        seed: int = 0,
        wall_removal: float = 0.0,
        exclude_generators: Sequence[str] = (),
    ):
        if not generators:
            raise ValueError("At least one generator is required")
        if len(set(generators)) != len(generators):
            raise ValueError(f"Duplicate generators: {list(generators)}")

        overlap = set(generators) & set(exclude_generators)

        if overlap:
            raise ValueError(
                f"Generators {sorted(overlap)} are also used for training; "
                "test mazes must come from unseen generators"
            )

        if size < 3:
            raise ValueError(f"Maze size must be >= 3 to fit a start and an exit, got {size}")

        if not 0.0 <= wall_removal <= 1.0:
            raise ValueError(f"wall_removal must be in [0, 1], got {wall_removal}")

        self.generator_names = list(generators)
        self._generators = [get_generator(name) for name in generators]
        self.size = size
        self.num_samples = num_samples
        self.seed = seed
        self.wall_removal = wall_removal

    def __len__(self) -> int:
        return self.num_samples

    def generator_index(self, idx: int) -> int:
        return idx % len(self._generators)

    @property
    def labels(self) -> np.ndarray:
        """Generator index of every sample, aligned with generator_names."""
        return np.arange(self.num_samples) % len(self._generators)

    def generate(self, idx: int) -> tuple[np.ndarray, Point, Point]:
        """Maze (H x W, 1 = passage), start and exit pixel coordinates for sample idx."""
        if not 0 <= idx < self.num_samples:
            raise IndexError(idx)

        rng = np.random.default_rng([self.seed, idx])
        maze = self._generators[self.generator_index(idx)](self.size, rng)

        # Endpoints come before wall removal so the same maze keeps the same
        # start and exit with or without it.
        start, goal = random_endpoints(maze, rng)

        if self.wall_removal > 0:
            maze = remove_walls(maze, self.wall_removal, rng)

        return maze, start, goal

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        maze, start, goal = self.generate(idx)

        inputs = np.zeros((3, *maze.shape), dtype=np.float32)
        inputs[0] = maze
        inputs[1][start] = 1
        inputs[2][goal] = 1
        target = shortest_path(maze, start, goal)[None].astype(np.float32)

        return torch.from_numpy(inputs), torch.from_numpy(target)


def make_train_dataset(
    generators: str | Sequence[str] = ("dfs", "prim", "kruskal"),
    size: int = 63,
    num_samples: int = 10_000,
    seed: int = 0,
) -> MazeDataset:
    """
    Training mazes, evenly split across the given generator(s).

    Pass a single name (e.g. "dfs") to train on one generator only.
    """
    if isinstance(generators, str):
        generators = [generators]

    return MazeDataset(generators, size, num_samples, seed=seed)


def make_test_datasets(
    generators: str | Sequence[str],
    sizes: int | Sequence[int],
    num_samples: int,
    train_generators: Sequence[str],
    seed: int = 1,
    wall_removal: float = 0.0,
) -> dict[int, MazeDataset]:
    """
    One test dataset per maze size, keyed by size.

    Sizes are kept in separate datasets so each one batches with the default
    collate function. Raises if any test generator was used for training.
    Each size gets its own seed offset so mazes are not correlated across sizes.
    """
    if isinstance(generators, str):
        generators = [generators]
    if isinstance(sizes, int):
        sizes = [sizes]
    return {
        size: MazeDataset(
            generators,
            size,
            num_samples,
            seed=seed * 1_000_003 + size,
            wall_removal=wall_removal,
            exclude_generators=train_generators,
        )
        for size in sizes
    }


def stratified_split(
    dataset: MazeDataset, val_fraction: float, seed: int = 0
) -> tuple[Subset, Subset]:
    """
    Split into (train, val) keeping the per-generator proportions identical.

    Each generator's samples are shuffled and split independently, so both
    subsets keep the same generator mix as the full dataset.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    rng = np.random.default_rng(seed)
    labels = dataset.labels
    train_idx, val_idx = [], []

    for g in range(len(dataset.generator_names)):
        idx = rng.permutation(np.flatnonzero(labels == g))
        n_val = int(round(val_fraction * len(idx)))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    
    return Subset(dataset, sorted(train_idx)), Subset(dataset, sorted(val_idx))
