"""Shared grid conventions for maze generators.

A maze of odd size N (e.g. 63 or 127) is an N x N maze wrapped in a 1-pixel
wall, returned as an (N + 2) x (N + 2) uint8 array where 1 = open passage and
0 = wall. It has (N + 1) / 2 x (N + 1) / 2 cells at odd (row, col) coordinates.
Opening the pixel between two adjacent cells carves a passage between them.
"""

from abc import ABC, abstractmethod

import numpy as np

PASSAGE = 1
WALL = 0

# Neighbour offsets in cell coordinates.
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def empty_grid(size: int) -> np.ndarray:
    return np.full((size, size), WALL, dtype=np.uint8)


def cell_to_pixel(r: int, c: int) -> tuple[int, int]:
    return 2 * r + 1, 2 * c + 1


def carve(grid: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> None:
    """Open cells a and b (cell coordinates) and the wall between them."""
    (ar, ac), (br, bc) = cell_to_pixel(*a), cell_to_pixel(*b)
    grid[ar, ac] = PASSAGE
    grid[br, bc] = PASSAGE
    grid[(ar + br) // 2, (ac + bc) // 2] = PASSAGE


def neighbours(r: int, c: int, n_cells: int):
    for dr, dc in DIRECTIONS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < n_cells and 0 <= nc < n_cells:
            yield nr, nc


class MazeGenerator(ABC):
    """Produces perfect mazes (spanning trees over the cell grid)."""

    name: str

    def __call__(self, size: int, rng: np.random.Generator) -> np.ndarray:
        if size < 1 or size % 2 == 0:
            raise ValueError(f"Maze size must be an odd integer >= 1, got {size}")

        n_cells = (size + 1) // 2
        grid = empty_grid(size + 2)
        self._carve(grid, n_cells, rng)
        return grid

    @abstractmethod
    def _carve(self, grid: np.ndarray, n_cells: int, rng: np.random.Generator) -> None:
        ...


def remove_walls(grid: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Return a copy of grid with `fraction` of the remaining interior walls opened.

    Only walls that separate two cells are candidates (pixels with exactly one
    odd coordinate, away from the border), so pillars and the outer border stay
    intact and the result is a maze with loops rather than open blobs.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")

    out = grid.copy()
    size = grid.shape[0]

    rr, cc = np.meshgrid(np.arange(1, size - 1), np.arange(1, size - 1), indexing="ij")
    is_separator = (rr % 2) != (cc % 2)

    candidates = np.argwhere(is_separator & (out[1:-1, 1:-1] == WALL)) + 1

    k = int(round(fraction * len(candidates)))
    if k:
        chosen = candidates[rng.choice(len(candidates), size=k, replace=False)]
        out[chosen[:, 0], chosen[:, 1]] = PASSAGE

    return out
