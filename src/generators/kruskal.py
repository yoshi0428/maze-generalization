import numpy as np

from .base import MazeGenerator, carve


class KruskalGenerator(MazeGenerator):
    """Randomized Kruskal's algorithm. Uniform-looking texture with many short dead ends."""

    name = "kruskal"

    def _carve(self, grid: np.ndarray, n_cells: int, rng: np.random.Generator) -> None:
        parent = list(range(n_cells * n_cells))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        edges = [((r, c), (r, c + 1)) for r in range(n_cells) for c in range(n_cells - 1)]
        edges += [((r, c), (r + 1, c)) for r in range(n_cells - 1) for c in range(n_cells)]

        if n_cells == 1:
            carve(grid, (0, 0), (0, 0))

        for i in rng.permutation(len(edges)):
            a, b = edges[i]
            ra, rb = find(a[0] * n_cells + a[1]), find(b[0] * n_cells + b[1])
            if ra != rb:
                parent[ra] = rb
                carve(grid, a, b)
