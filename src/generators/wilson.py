import numpy as np

from .base import MazeGenerator, carve, neighbours


class WilsonGenerator(MazeGenerator):
    """Wilson's algorithm. Loop-erased random walks; produces uniform spanning trees."""

    name = "wilson"

    def _carve(self, grid: np.ndarray, n_cells: int, rng: np.random.Generator) -> None:
        in_maze = np.zeros((n_cells, n_cells), dtype=bool)
        root = (int(rng.integers(n_cells)), int(rng.integers(n_cells)))
        in_maze[root] = True

        carve(grid, root, root)

        # Visit start cells in random order; each walk ends on the existing maze.
        for flat in rng.permutation(n_cells * n_cells).tolist():
            start = divmod(flat, n_cells)
            if in_maze[start]:
                continue

            # Remembering only the last exit from each cell erases loops implicitly.
            exit_to: dict[tuple[int, int], tuple[int, int]] = {}
            cell = start
            while not in_maze[cell]:
                options = list(neighbours(*cell, n_cells))
                nxt = options[rng.integers(len(options))]
                exit_to[cell] = nxt
                cell = nxt

            cell = start
            while not in_maze[cell]:
                nxt = exit_to[cell]
                carve(grid, cell, nxt)
                in_maze[cell] = True
                cell = nxt
