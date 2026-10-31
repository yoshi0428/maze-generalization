"""Maze generator registry.

To add a generator, subclass MazeGenerator in a new module, give it a unique
`name`, and add it to GENERATORS below.
"""

from .aldous_broder import AldousBroderGenerator
from .base import PASSAGE, WALL, MazeGenerator, remove_walls
from .dfs import DFSGenerator
from .eller import EllerGenerator
from .hunt_and_kill import HuntAndKillGenerator
from .kruskal import KruskalGenerator
from .prim import PrimGenerator
from .recursive_division import RecursiveDivisionGenerator
from .sidewinder import SidewinderGenerator
from .wilson import WilsonGenerator

GENERATORS: dict[str, MazeGenerator] = {
    g.name: g
    for g in (
        DFSGenerator(),
        PrimGenerator(),
        KruskalGenerator(),
        EllerGenerator(),
        AldousBroderGenerator(),
        WilsonGenerator(),
        RecursiveDivisionGenerator(),
        HuntAndKillGenerator(),
        SidewinderGenerator(),
    )
}


def get_generator(name: str) -> MazeGenerator:
    try:
        return GENERATORS[name]
    except KeyError:
        raise ValueError(f"Unknown generator {name!r}; available: {sorted(GENERATORS)}") from None


__all__ = ["GENERATORS", "PASSAGE", "WALL", "MazeGenerator", "get_generator", "remove_walls"]
