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
                    "allow_pop_outs": phase.allow_pop_outs,
                }
                for phase in phases
            ],
            total_allotment=sum(
                phase.time_limit_sec or 0.0 for phase in phases if phase.time_limit_sec is not None
            ),
        )
        logs: List[PhaseLog] = []
        overall_start = time.time()
        total_allotment = sum(
            phase.time_limit_sec or 0.0 for phase in phases if phase.time_limit_sec is not None
        )
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
                allow_pop_outs=phase.allow_pop_outs,
                phase_index=phase_index,
                phase_count=len(phases),
                overall_elapsed=time.time() - overall_start,
            )
            for attempt_index, (board_w, board_h) in enumerate(candidate_boards, start=1):
                phase_limit = phase.time_limit_sec
                if phase_limit is not None:
                    elapsed_before_attempt = time.time() - phase_start
                    remaining_time = phase_limit - elapsed_before_attempt
                    if remaining_time <= 0:
                        break
                else:
                    remaining_time = None
                request = SolveRequest(
                    tile_quantities,
                    board_w,
                    board_h,
                    allow_rotation=phase.allow_rotation,
                    allow_pop_outs=phase.allow_pop_outs,
                )
                options = SolverOptions(
                    max_edge_cells=int(SETTINGS.MAX_EDGE_FT / self.unit_ft),
                    same_shape_limit=SETTINGS.SAME_SHAPE_LIMIT,
                    enforce_plus_rule=SETTINGS.PLUS_TOGGLE,
                    time_limit_sec=remaining_time,
                )
                def report_solver_progress(solver_instance: BacktrackingSolver) -> None:
                    phase_elapsed = time.time() - phase_start
                    prev_allotment = sum(
                        p.time_limit_sec or 0.0
                        for p in phases[:phase_index]
                        if p.time_limit_sec is not None
                    )
                    overall_progress = 0.0
                    if total_allotment > 0:
                        phase_cap = phase.time_limit_sec or 0.0
                        overall_progress = min(
                            (prev_allotment + min(phase_elapsed, phase_cap)) / total_allotment,
                            1.0,
                        )
                    emit(
                        "attempt_progress",
                        phase=phase.name,
                        attempt_index=attempt_index,
                        total_attempts=len(candidate_boards),
                        board_size_ft=(board_w * self.unit_ft, board_h * self.unit_ft),
                        board_size_cells=(board_w, board_h),
                        phase_index=phase_index,
                        time_limit_sec=attempt_limit,
                        overall_elapsed=time.time() - overall_start,
                        phase_elapsed=phase_elapsed,
                        phase_progress=(
                            min(phase_elapsed / phase.time_limit_sec, 1.0)
                            if phase.time_limit_sec
                            else 0.0
                        ),
                        overall_progress=overall_progress,
                        attempt_elapsed=solver_instance.stats.elapsed,
                        backtracks=solver_instance.stats.backtracks,
                    )

                solver = BacktrackingSolver(
                    request,
                    options,
                    self.unit_ft,
                    phase.name,
                    progress_callback=report_solver_progress,
                )
                attempt_limit = remaining_time if remaining_time is not None else phase.time_limit_sec
                emit(
                    "attempt_started",
                    phase=phase.name,
                    attempt_index=attempt_index,
                    total_attempts=len(candidate_boards),
                    board_size_ft=(board_w * self.unit_ft, board_h * self.unit_ft),
                    board_size_cells=(board_w, board_h),
                    phase_index=phase_index,
                    time_limit_sec=attempt_limit,
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
                prev_allotment = sum(
                    p.time_limit_sec or 0.0
                    for p in phases[:phase_index]
                    if p.time_limit_sec is not None
                )
                overall_progress = 0.0
                if total_allotment > 0:
                    phase_cap = phase.time_limit_sec or 0.0
                    overall_progress = min(
                        (prev_allotment + min(phase_elapsed, phase_cap)) / total_allotment,
                        1.0,
                    )
                remaining_after_attempt = None
                if phase_limit is not None:
                    remaining_after_attempt = max(
                        phase_limit - (time.time() - phase_start),
                        0.0,
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
                    time_limit_sec=(
                        remaining_after_attempt
                        if phase_limit is not None
                        else phase.time_limit_sec
                    ),
                    overall_elapsed=time.time() - overall_start,
                    phase_elapsed=phase_elapsed,
                    phase_progress=(
                        min(phase_elapsed / phase.time_limit_sec, 1.0)
                        if phase.time_limit_sec
                        else 0.0
                    ),
                    overall_progress=overall_progress,
                )
                if solve_result:
                    result = solve_result
                    break
                if (
                    phase_limit is not None
                    and time.time() - phase_start >= phase_limit
                ):
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

    def _candidate_boards(self, total_area_ft: float) -> List[Tuple[int, int]]:
        if total_area_ft <= 0:
            return []

        padded_area_ft = math.ceil(total_area_ft)
        if padded_area_ft <= 0:
            return []

        unit_area = self.unit_ft ** 2
        cells_area = padded_area_ft / unit_area
        area_cells = int(round(cells_area))
        if area_cells <= 0:
            return []
        if not math.isclose(cells_area, area_cells, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                "Total tile coverage must align to the grid size. Adjust tile quantities to form a square grid."
            )

        starting_side_ft = max(1, int(math.ceil(math.sqrt(padded_area_ft))))

        def ft_to_cells(feet: int) -> int:
            cells = feet / self.unit_ft
            rounded = int(round(cells))
            if rounded <= 0:
                return 1
            if not math.isclose(cells, rounded, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(
                    "Board dimensions must align with the grid size defined by GRID_UNIT_FT."
                )
            return rounded

        boards: List[Tuple[int, int]] = []
        for reduction in range(6):
            side_ft = max(1, starting_side_ft - reduction)
            cells = ft_to_cells(side_ft)
            boards.append((cells, cells))

        return boards


__all__ = ["TileSolverOrchestrator", "PhaseLog", "PhaseAttempt"]
