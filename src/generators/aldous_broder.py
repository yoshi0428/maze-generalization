import numpy as np

from .base import DIRECTIONS, MazeGenerator, carve


class AldousBroderGenerator(MazeGenerator):
    """
    Aldous-Broder algorithm. A random walk that carves into every unvisited cell it enters.

    Produces uniform spanning trees, but needs a full cover time of the grid,
    so it is the slowest generator on large mazes.
    """

    name = "aldous_broder"

    def _carve(self, grid: np.ndarray, n_cells: int, rng: np.random.Generator) -> None:
        visited = np.zeros((n_cells, n_cells), dtype=bool)
        r, c = int(rng.integers(n_cells)), int(rng.integers(n_cells))
        visited[r, c] = True
        remaining = n_cells * n_cells - 1

        carve(grid, (r, c), (r, c))

        while remaining:
            # Draw directions in chunks to keep the per-step cost down.
            for d in rng.integers(4, size=4096).tolist():
                dr, dc = DIRECTIONS[d]
                nr, nc = r + dr, c + dc
                if not (0 <= nr < n_cells and 0 <= nc < n_cells):
                    continue

                if not visited[nr, nc]:
                    visited[nr, nc] = True
                    carve(grid, (r, c), (nr, nc))
                    remaining -= 1
                    if not remaining:
                        break

                r, c = nr, nc
