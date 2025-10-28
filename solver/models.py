from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class TileType:
    name: str
    width_ft: float
    height_ft: float

    @property
    def area_ft2(self) -> float:
        return self.width_ft * self.height_ft

    def as_cells(self, unit_ft: float) -> Tuple[int, int]:
        return int(round(self.width_ft / unit_ft)), int(round(self.height_ft / unit_ft))


@dataclass
class TileInstance:
    type: TileType
    identifier: str
    allow_rotation: bool


@dataclass
class Placement:
    tile: TileInstance
    x: int
    y: int
    width: int
    height: int

    def to_top_left_ft(self, unit_ft: float) -> Tuple[float, float]:
        return self.x * unit_ft, self.y * unit_ft


@dataclass
class SolveRequest:
    tile_quantities: Dict[TileType, int]
    board_width_cells: int
    board_height_cells: int
    allow_rotation: bool
    allow_pop_outs: bool
    allow_discards: bool
    pop_out_mask: Optional[frozenset[Tuple[int, int]]]


@dataclass
class SolveResult:
    placements: List[Placement]
    board_width_cells: int
    board_height_cells: int
    phase_name: str
    board_width_ft: float
    board_height_ft: float
    discarded_tiles: List[TileInstance]


@dataclass
class SolverStats:
    boards_attempted: int = 0
    backtracks: int = 0
    elapsed: float = 0.0

