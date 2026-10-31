import numpy as np

from .base import MazeGenerator, carve, neighbours


class PrimGenerator(MazeGenerator):
    """Randomized Prim's algorithm. Short dead ends branching from many junctions."""

    name = "prim"

    def _carve(self, grid: np.ndarray, n_cells: int, rng: np.random.Generator) -> None:
        in_maze = np.zeros((n_cells, n_cells), dtype=bool)
        start = (int(rng.integers(n_cells)), int(rng.integers(n_cells)))
        in_maze[start] = True

        carve(grid, start, start)

        frontier: list[tuple[int, int]] = []
        in_frontier = np.zeros_like(in_maze)

        def add_frontier(cell):
            for nb in neighbours(*cell, n_cells):
                if not in_maze[nb] and not in_frontier[nb]:
                    in_frontier[nb] = True
                    frontier.append(nb)

        add_frontier(start)

        while frontier:
            # Swap-remove a random frontier cell for O(1) deletion.
            i = int(rng.integers(len(frontier)))
            frontier[i], frontier[-1] = frontier[-1], frontier[i]
            cell = frontier.pop()
            attached = [nb for nb in neighbours(*cell, n_cells) if in_maze[nb]]

            carve(grid, attached[rng.integers(len(attached))], cell)

            in_maze[cell] = True

            add_frontier(cell)
