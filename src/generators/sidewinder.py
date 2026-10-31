import numpy as np

from .base import MazeGenerator, carve


class SidewinderGenerator(MazeGenerator):
    """
    Sidewinder. Row by row, carves east-west runs that each open once to the row above.

    Strongly biased: the top row is always one straight corridor.
    """

    name = "sidewinder"

    def __init__(self, east_prob: float = 0.5):
        self.east_prob = east_prob

    def _carve(self, grid: np.ndarray, n_cells: int, rng: np.random.Generator) -> None:
        for c in range(n_cells):
            carve(grid, (0, c), (0, c + 1) if c < n_cells - 1 else (0, c))

        for r in range(1, n_cells):
            run_start = 0

            for c in range(n_cells):
                at_east_edge = c == n_cells - 1
                if not at_east_edge and rng.random() < self.east_prob:
                    carve(grid, (r, c), (r, c + 1))
                    continue

                # Close the run: open one of its cells to the row above.
                up = run_start + int(rng.integers(c - run_start + 1))
                carve(grid, (r, up), (r - 1, up))
                carve(grid, (r, c), (r, c))
                run_start = c + 1
