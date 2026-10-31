import numpy as np

from .base import MazeGenerator, carve, neighbours


class HuntAndKillGenerator(MazeGenerator):
    """
    Hunt-and-kill. Random walk until stuck, then scan for an unvisited cell next to the maze.

    Similar to DFS (long corridors) but restarts from scanned cells instead of
    backtracking, which leaves a different dead-end pattern.
    """

    name = "hunt_and_kill"

    def _carve(self, grid: np.ndarray, n_cells: int, rng: np.random.Generator) -> None:
        visited = np.zeros((n_cells, n_cells), dtype=bool)
        cell = (int(rng.integers(n_cells)), int(rng.integers(n_cells)))
        visited[cell] = True

        carve(grid, cell, cell)

        # Rows above this one are fully visited, so the hunt can skip them.
        hunt_row = 0

        while cell is not None:
            # Kill: walk to random unvisited neighbours.
            while True:
                options = [nb for nb in neighbours(*cell, n_cells) if not visited[nb]]
                if not options:
                    break

                nxt = options[rng.integers(len(options))]
                visited[nxt] = True
                carve(grid, cell, nxt)
                cell = nxt

            # Hunt: first unvisited cell (row-major) with a visited neighbour.
            cell = None
            while hunt_row < n_cells and visited[hunt_row].all():
                hunt_row += 1

            for r in range(hunt_row, n_cells):
                adjacent = np.zeros(n_cells, dtype=bool)

                if r > 0:
                    adjacent |= visited[r - 1]

                if r < n_cells - 1:
                    adjacent |= visited[r + 1]

                adjacent[1:] |= visited[r, :-1]
                adjacent[:-1] |= visited[r, 1:]

                candidates = np.flatnonzero(adjacent & ~visited[r])
                if len(candidates):
                    cell = (r, int(candidates[0]))
                    attached = [nb for nb in neighbours(*cell, n_cells) if visited[nb]]
                    carve(grid, attached[rng.integers(len(attached))], cell)
                    visited[cell] = True
                    break
