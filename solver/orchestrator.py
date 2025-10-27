from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from config import SETTINGS
from .backtracking_solver import BacktrackingSolver, SolverOptions
from .models import SolveRequest, SolveResult, TileType


@dataclass
class PhaseAttempt:
    phase_name: str
    board_size_ft: Tuple[float, float]
    board_size_cells: Tuple[int, int]
    elapsed: float
    backtracks: int
    success: bool


@dataclass
class PhaseLog:
    phase_name: str
    attempts: List[PhaseAttempt]
    total_elapsed: float
    result: Optional[SolveResult]


class TileSolverOrchestrator:
    def __init__(self) -> None:
        self.unit_ft = SETTINGS.GRID_UNIT_FT
        self.tile_types = {
            name: TileType(name, dims[0], dims[1]) for name, dims in SETTINGS.TILE_OPTIONS.items()
        }

    def solve(self, selection: Dict[str, int]) -> Tuple[Optional[SolveResult], List[PhaseLog]]:
        tile_quantities = self._build_quantities(selection)
        total_area_ft = sum(tile.area_ft2 * qty for tile, qty in tile_quantities.items())
        if total_area_ft <= 0:
            raise ValueError("No tiles selected")
        phases = self._select_phases(total_area_ft)
        logs: List[PhaseLog] = []
        for phase in phases:
            phase_start = time.time()
            attempts: List[PhaseAttempt] = []
            result: Optional[SolveResult] = None
            for board_w, board_h in self._candidate_boards(total_area_ft):
                request = SolveRequest(tile_quantities, board_w, board_h, allow_rotation=phase.allow_rotation)
                options = SolverOptions(
                    max_edge_cells=int(SETTINGS.MAX_EDGE_FT / self.unit_ft),
                    same_shape_limit=SETTINGS.SAME_SHAPE_LIMIT,
                    enforce_plus_rule=SETTINGS.PLUS_TOGGLE,
                    time_limit_sec=phase.time_limit_sec,
                )
                solver = BacktrackingSolver(request, options, self.unit_ft, phase.name)
                solve_start = time.time()
                solve_result = solver.solve()
                elapsed = time.time() - solve_start
                attempt = PhaseAttempt(
                    phase_name=phase.name,
                    board_size_ft=(board_w * self.unit_ft, board_h * self.unit_ft),
                    board_size_cells=(board_w, board_h),
                    elapsed=elapsed,
                    backtracks=solver.stats.backtracks,
                    success=solve_result is not None,
                )
                attempts.append(attempt)
                if solve_result:
                    result = solve_result
                    break
            total_elapsed = time.time() - phase_start
            logs.append(PhaseLog(phase.name, attempts, total_elapsed, result))
            if result:
                return result, logs
        return None, logs

    def _select_phases(self, total_area_ft: float):
        if total_area_ft < 100:
            return [SETTINGS.PHASE_A, SETTINGS.PHASE_B]
        return [SETTINGS.PHASE_C, SETTINGS.PHASE_D]

    def _build_quantities(self, selection: Dict[str, int]):
        quantities: Dict[TileType, int] = {}
        for name, count in selection.items():
            tile_type = self.tile_types.get(name)
            if tile_type is None:
                raise ValueError(f"Unknown tile: {name}")
            if count < 0:
                raise ValueError("Tile quantities must be non-negative")
            if count:
                quantities[tile_type] = count
        return quantities

    def _candidate_boards(self, total_area_ft: float) -> Iterable[Tuple[int, int]]:
        area_cells = int(round(total_area_ft / (self.unit_ft ** 2)))
        if area_cells <= 0:
            return []
        candidates = set()
        for width in range(1, int(math.sqrt(area_cells)) + 1):
            if area_cells % width == 0:
                height = area_cells // width
                candidates.add((width, height))
                candidates.add((height, width))
        start_leg = math.ceil(math.sqrt(total_area_ft))
        target_dim = int(math.ceil((start_leg + 2) / self.unit_ft))
        def sort_key(dim: Tuple[int, int]):
            w, h = dim
            return (abs(w - h), abs(w - target_dim) + abs(h - target_dim), -min(w, h))
        sorted_candidates = sorted(candidates, key=sort_key)
        return sorted_candidates


__all__ = ["TileSolverOrchestrator", "PhaseLog", "PhaseAttempt"]
