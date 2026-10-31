import numpy as np

from .base import MazeGenerator, carve


class EllerGenerator(MazeGenerator):
    """Eller's algorithm. Builds the maze one row at a time, tracking connected sets."""

    name = "eller"

    def __init__(self, join_prob: float = 0.5, down_prob: float = 0.5):
        self.join_prob = join_prob
        self.down_prob = down_prob

    def _carve(self, grid: np.ndarray, n_cells: int, rng: np.random.Generator) -> None:
        sets = np.arange(n_cells)
        next_label = n_cells

        for r in range(n_cells):
            last_row = r == n_cells - 1

            for c in range(n_cells):
                carve(grid, (r, c), (r, c))

            # Join horizontally adjacent cells from different sets.
            for c in range(n_cells - 1):
                if sets[c] != sets[c + 1] and (last_row or rng.random() < self.join_prob):
                    sets[sets == sets[c + 1]] = sets[c]
                    carve(grid, (r, c), (r, c + 1))

            if last_row:
                break

            # Each set extends downward at least once; other cells start new sets.
            next_sets = np.empty_like(sets)
            for label in np.unique(sets):
                members = np.flatnonzero(sets == label)
                down = members[rng.random(len(members)) < self.down_prob]
                if len(down) == 0:
                    down = [members[rng.integers(len(members))]]

                for c in down:
                    carve(grid, (r, c), (r + 1, c))

                fresh = np.setdiff1d(members, down)
                next_sets[down] = label
                next_sets[fresh] = np.arange(next_label, next_label + len(fresh))
                next_label += len(fresh)

            sets = next_sets
