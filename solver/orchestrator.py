from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

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

    def solve(
        self,
        selection: Dict[str, int],
        progress_callback: Optional[Callable[[Dict[str, object]], None]] = None,
    ) -> Tuple[Optional[SolveResult], List[PhaseLog]]:
        def emit(event_type: str, **payload: object) -> None:
            if progress_callback:
                event = {"type": event_type}
                event.update(payload)
                progress_callback(event)

        tile_quantities = self._build_quantities(selection)
        total_area_ft = sum(tile.area_ft2 * qty for tile, qty in tile_quantities.items())
        if total_area_ft <= 0:
            raise ValueError("No tiles selected")
        phases = self._select_phases(total_area_ft)
        emit(
            "run_started",
            phases=[
                {
                    "name": phase.name,
                    "time_limit_sec": phase.time_limit_sec,
                    "allow_rotation": phase.allow_rotation,
                    "allow_discards": phase.allow_discards,
                }
                for phase in phases
            ],
            total_allotment=sum(phase.time_limit_sec for phase in phases),
        )
        logs: List[PhaseLog] = []
        overall_start = time.time()
        total_allotment = sum(phase.time_limit_sec for phase in phases)
        candidate_boards = list(self._candidate_boards(total_area_ft))
        for phase_index, phase in enumerate(phases):
            phase_start = time.time()
            attempts: List[PhaseAttempt] = []
            result: Optional[SolveResult] = None
            emit(
                "phase_started",
                phase=phase.name,
                time_limit_sec=phase.time_limit_sec,
                allow_rotation=phase.allow_rotation,
                allow_discards=phase.allow_discards,
                phase_index=phase_index,
                phase_count=len(phases),
                overall_elapsed=time.time() - overall_start,
            )
            for attempt_index, (board_w, board_h) in enumerate(candidate_boards, start=1):
                request = SolveRequest(tile_quantities, board_w, board_h, allow_rotation=phase.allow_rotation)
                options = SolverOptions(
                    max_edge_cells=int(SETTINGS.MAX_EDGE_FT / self.unit_ft),
                    same_shape_limit=SETTINGS.SAME_SHAPE_LIMIT,
                    enforce_plus_rule=SETTINGS.PLUS_TOGGLE,
                    time_limit_sec=phase.time_limit_sec,
                )
                solver = BacktrackingSolver(request, options, self.unit_ft, phase.name)
                emit(
                    "attempt_started",
                    phase=phase.name,
                    attempt_index=attempt_index,
                    total_attempts=len(candidate_boards),
                    board_size_ft=(board_w * self.unit_ft, board_h * self.unit_ft),
                    board_size_cells=(board_w, board_h),
                    phase_index=phase_index,
                    time_limit_sec=phase.time_limit_sec,
                    overall_elapsed=time.time() - overall_start,
                    phase_elapsed=time.time() - phase_start,
                )
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
                phase_elapsed = time.time() - phase_start
                prev_allotment = sum(p.time_limit_sec for p in phases[:phase_index])
                overall_progress = 0.0
                if total_allotment > 0:
                    overall_progress = min(
                        (prev_allotment + min(phase_elapsed, phase.time_limit_sec)) / total_allotment,
                        1.0,
                    )
                emit(
                    "attempt_completed",
                    phase=phase.name,
                    attempt_index=attempt_index,
                    total_attempts=len(candidate_boards),
                    board_size_ft=(board_w * self.unit_ft, board_h * self.unit_ft),
                    board_size_cells=(board_w, board_h),
                    elapsed=elapsed,
                    success=solve_result is not None,
                    backtracks=solver.stats.backtracks,
                    phase_index=phase_index,
                    time_limit_sec=phase.time_limit_sec,
                    overall_elapsed=time.time() - overall_start,
                    phase_elapsed=phase_elapsed,
                    phase_progress=min(phase_elapsed / phase.time_limit_sec, 1.0)
                    if phase.time_limit_sec
                    else 1.0,
                    overall_progress=overall_progress,
                )
                if solve_result:
                    result = solve_result
                    break
            total_elapsed = time.time() - phase_start
            logs.append(PhaseLog(phase.name, attempts, total_elapsed, result))
            emit(
                "phase_completed",
                phase=phase.name,
                total_elapsed=total_elapsed,
                success=result is not None,
                phase_index=phase_index,
                phase_count=len(phases),
                overall_elapsed=time.time() - overall_start,
            )
            if result:
                emit(
                    "run_completed",
                    success=True,
                    overall_elapsed=time.time() - overall_start,
                    overall_progress=1.0,
                )
                return result, logs
        emit(
            "run_completed",
            success=False,
            overall_elapsed=time.time() - overall_start,
            overall_progress=1.0,
        )
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
        cell_area = total_area_ft / (self.unit_ft ** 2)
        area_cells = int(round(cell_area))
        if area_cells <= 0:
            return []
        if not math.isclose(cell_area, area_cells, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                "Total tile coverage must align to the grid size. Adjust tile quantities to form a square grid."
            )
        candidates: List[Tuple[int, int]] = []
        limit = int(math.isqrt(area_cells))
        for width in range(1, limit + 1):
            if area_cells % width != 0:
                continue
            height = area_cells // width
            candidates.append((width, height))
            if width != height:
                candidates.append((height, width))
        if not candidates:
            candidates.append((area_cells, 1))
        candidates.sort(key=lambda dims: abs(dims[0] - dims[1]))
        return candidates


__all__ = ["TileSolverOrchestrator", "PhaseLog", "PhaseAttempt"]
