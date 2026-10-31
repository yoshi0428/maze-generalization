import numpy as np

from .base import MazeGenerator, carve, neighbours


class DFSGenerator(MazeGenerator):
    """Randomized depth-first search (recursive backtracker). Long, winding corridors."""

    name = "dfs"

    def _carve(self, grid: np.ndarray, n_cells: int, rng: np.random.Generator) -> None:
        visited = np.zeros((n_cells, n_cells), dtype=bool)
        start = (int(rng.integers(n_cells)), int(rng.integers(n_cells)))
        visited[start] = True

        carve(grid, start, start)

        stack = [start]

        while stack:
            cell = stack[-1]
            options = [nb for nb in neighbours(*cell, n_cells) if not visited[nb]]

            if not options:
                stack.pop()
                continue

            nxt = options[rng.integers(len(options))]
            visited[nxt] = True

            carve(grid, cell, nxt)

            stack.append(nxt)
