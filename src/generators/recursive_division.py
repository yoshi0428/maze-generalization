import numpy as np

from .base import PASSAGE, WALL, MazeGenerator


class RecursiveDivisionGenerator(MazeGenerator):
    """
    Recursive division. Starts from an open room and adds walls with one gap each.

    Unlike the passage-carving generators, this produces long straight walls
    and a visibly rectangular, hierarchical structure.
    """

    name = "recursive_division"

    def _carve(self, grid: np.ndarray, n_cells: int, rng: np.random.Generator) -> None:
        grid[1:-1, 1:-1] = PASSAGE

        # Chambers in cell coordinates: (row0, col0, height, width).
        stack = [(0, 0, n_cells, n_cells)]

        while stack:
            r0, c0, h, w = stack.pop()
            if h < 2 and w < 2:
                continue

            horizontal = h > w if h != w else bool(rng.integers(2))
            if w < 2:
                horizontal = True
            elif h < 2:
                horizontal = False

            if horizontal:
                # Wall between cell rows k and k + 1, with a gap at one cell column.
                k = r0 + int(rng.integers(h - 1))
                gap = c0 + int(rng.integers(w))
                row = 2 * k + 2
                grid[row, 2 * c0 : 2 * (c0 + w) + 1] = WALL
                grid[row, 2 * gap + 1] = PASSAGE

                stack.append((r0, c0, k - r0 + 1, w))
                stack.append((k + 1, c0, r0 + h - k - 1, w))
            else:
                k = c0 + int(rng.integers(w - 1))
                gap = r0 + int(rng.integers(h))
                col = 2 * k + 2
                grid[2 * r0 : 2 * (r0 + h) + 1, col] = WALL
                grid[2 * gap + 1, col] = PASSAGE

                stack.append((r0, c0, h, k - c0 + 1))
                stack.append((r0, k + 1, h, c0 + w - k - 1))

        # Pillars (even, even) are always wall, matching the other generators.
        grid[::2, ::2] = WALL
