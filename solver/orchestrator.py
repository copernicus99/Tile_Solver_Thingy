from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from config import SETTINGS, PhaseConfig
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


@dataclass(frozen=True)
class BoardCandidate:
    width_cells: int
    height_cells: int
    pop_out_mask: Optional[frozenset[Tuple[int, int]]]


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
            phase_candidates = [
                board
                for board in candidate_boards
                if phase.allow_pop_outs or board.pop_out_mask is None
            ]
            total_attempts = len(phase_candidates)
            for attempt_index, candidate in enumerate(phase_candidates, start=1):
                board_w = candidate.width_cells
                board_h = candidate.height_cells
                mask = candidate.pop_out_mask if phase.allow_pop_outs else None
                phase_limit = phase.time_limit_sec
                if phase_limit is not None:
                    elapsed_before_attempt = time.time() - phase_start
                    remaining_time = phase_limit - elapsed_before_attempt
                    if remaining_time <= 0:
                        break
                    attempt_limit = self._attempt_time_limit(
                        phase,
                        attempt_index,
                        total_attempts,
                        remaining_time,
                    )
                    if attempt_limit <= 0:
                        break
                else:
                    remaining_time = None
                    attempt_limit = None
                request = SolveRequest(
                    tile_quantities,
                    board_w,
                    board_h,
                    allow_rotation=phase.allow_rotation,
                    allow_pop_outs=phase.allow_pop_outs,
                    allow_discards=phase.allow_discards,
                    pop_out_mask=mask,
                )
                options = SolverOptions(
                    max_edge_cells_horizontal=self._max_edge_for_dimension(board_w),
                    max_edge_cells_vertical=self._max_edge_for_dimension(board_h),
                    max_edge_include_perimeter=not getattr(
                        SETTINGS, "MAX_EDGE_INSIDE_ONLY", True
                    ),
                    same_shape_limit=SETTINGS.SAME_SHAPE_LIMIT,
                    enforce_plus_rule=SETTINGS.PLUS_TOGGLE,
                    time_limit_sec=attempt_limit,
                )
                attempt_limit_value = attempt_limit
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
                        total_attempts=total_attempts,
                        board_size_ft=(board_w * self.unit_ft, board_h * self.unit_ft),
                        board_size_cells=(board_w, board_h),
                        phase_index=phase_index,
                        time_limit_sec=attempt_limit_value,
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
                emit(
                    "attempt_started",
                    phase=phase.name,
                    attempt_index=attempt_index,
                    total_attempts=total_attempts,
                    board_size_ft=(board_w * self.unit_ft, board_h * self.unit_ft),
                    board_size_cells=(board_w, board_h),
                    pop_out_mask_cells=len(mask) if mask else 0,
                    phase_index=phase_index,
                    time_limit_sec=attempt_limit_value,
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
                    total_attempts=total_attempts,
                    board_size_ft=(board_w * self.unit_ft, board_h * self.unit_ft),
                    board_size_cells=(board_w, board_h),
                    pop_out_mask_cells=len(mask) if mask else 0,
                    elapsed=elapsed,
                    success=solve_result is not None,
                    backtracks=solver.stats.backtracks,
                    phase_index=phase_index,
                    time_limit_sec=attempt_limit_value,
                    remaining_phase_time_sec=remaining_after_attempt,
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

    def _attempt_time_limit(
        self,
        phase: PhaseConfig,
        attempt_index: int,
        total_attempts: int,
        remaining_time: float,
    ) -> float:
        phase_limit = phase.time_limit_sec
        if phase_limit is None:
            return remaining_time
        first_share = min(max(phase.first_board_time_share, 0.0), 1.0)
        additional_slots = min(5, max(total_attempts - 1, 0))
        if additional_slots == 0 and total_attempts <= 1:
            # Allow the lone attempt to consume the entire phase allotment.
            first_share = 1.0
        remainder_share = max(0.0, 1.0 - first_share)
        if attempt_index == 1:
            share = first_share
        elif additional_slots > 0 and attempt_index <= 1 + additional_slots:
            share = remainder_share / additional_slots if additional_slots else 0.0
        else:
            share = 0.0
        attempt_cap = phase_limit * share
        limit = min(remaining_time, attempt_cap)
        return max(0.0, limit)

    def _max_edge_for_dimension(self, length_cells: int) -> int:
        limits: List[int] = []

        ratio = getattr(SETTINGS, "MAX_EDGE_RATIO", None)
        if ratio is not None and ratio > 0:
            ratio_limit_cells = int(math.floor(length_cells * ratio + 1e-9))
            if length_cells > 0 and ratio_limit_cells <= 0:
                ratio_limit_cells = 1
            if ratio_limit_cells > 0:
                ratio_limit_cells = min(length_cells, ratio_limit_cells)
                limits.append(ratio_limit_cells)

        absolute_ft = getattr(SETTINGS, "MAX_EDGE_FT", 0.0)
        if absolute_ft and absolute_ft > 0:
            limit_cells = int(math.floor(absolute_ft / self.unit_ft + 1e-9))
            if limit_cells > 0:
                limits.append(limit_cells)

        if not limits:
            return 0

        return min(limits)

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

    def _candidate_boards(self, total_area_ft: float) -> List[BoardCandidate]:
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

        boards: List[BoardCandidate] = []
        for reduction in range(6):
            side_ft = max(1, starting_side_ft - reduction)
            cells = ft_to_cells(side_ft)
            boards.append(BoardCandidate(cells, cells, None))
            for mask in self._pop_out_masks(cells, cells, area_cells):
                boards.append(BoardCandidate(cells, cells, mask))

        return boards

    def _pop_out_masks(
        self, width_cells: int, height_cells: int, required_cells: int
    ) -> Iterable[frozenset[Tuple[int, int]]]:
        extra_capacity = width_cells * height_cells - required_cells
        if extra_capacity <= 0:
            return []

        max_variations = getattr(SETTINGS, "POP_OUT_VARIATIONS_PER_BOARD", 0)
        if max_variations <= 0:
            return []

        max_depth = getattr(SETTINGS, "POP_OUT_MAX_NOTCH_DEPTH", 0)
        span_limit = getattr(SETTINGS, "POP_OUT_MAX_NOTCH_SPAN", 0)
        if max_depth <= 0 or span_limit <= 0:
            return []

        masks: List[frozenset[Tuple[int, int]]] = []
        for variant in range(max_variations):
            mask = self._generate_pop_out_mask(
                width_cells,
                height_cells,
                extra_capacity,
                max_depth,
                span_limit,
                variant,
            )
            if mask:
                masks.append(frozenset(mask))
        return masks

    def _generate_pop_out_mask(
        self,
        width_cells: int,
        height_cells: int,
        target_cells: int,
        max_depth: int,
        span_limit: int,
        variant: int,
    ) -> Optional[set[Tuple[int, int]]]:
        if target_cells <= 0:
            return None
        mask: set[Tuple[int, int]] = set()
        corners = ("top_left", "top_right", "bottom_left", "bottom_right")
        start_index = variant % len(corners)
        base_offset = (variant // len(corners)) * span_limit
        offsets = [base_offset for _ in range(len(corners))]
        remaining = target_cells
        passes = 0
        while remaining > 0 and passes < (width_cells + height_cells):
            for step in range(len(corners)):
                corner_index = (start_index + step) % len(corners)
                corner = corners[corner_index]
                offset = offsets[corner_index]
                removed = self._carve_corner_notch(
                    corner,
                    offset,
                    width_cells,
                    height_cells,
                    remaining,
                    max_depth,
                    span_limit,
                    mask,
                )
                offsets[corner_index] += span_limit
                remaining -= removed
                if remaining <= 0:
                    break
            passes += 1
        if remaining > 0:
            return None
        return mask

    def _carve_corner_notch(
        self,
        corner: str,
        offset: int,
        width_cells: int,
        height_cells: int,
        remaining: int,
        max_depth: int,
        span_limit: int,
        mask: set[Tuple[int, int]],
    ) -> int:
        removed = 0
        if corner == "top_left":
            x_fn = lambda span: offset + span
            y_fn = lambda depth: depth
        elif corner == "top_right":
            x_fn = lambda span: width_cells - 1 - (offset + span)
            y_fn = lambda depth: depth
        elif corner == "bottom_left":
            x_fn = lambda span: offset + span
            y_fn = lambda depth: height_cells - 1 - depth
        elif corner == "bottom_right":
            x_fn = lambda span: width_cells - 1 - (offset + span)
            y_fn = lambda depth: height_cells - 1 - depth
        else:
            return 0

        for depth in range(max_depth):
            y = y_fn(depth)
            if y < 0 or y >= height_cells:
                break
            for span in range(span_limit):
                if removed >= remaining:
                    break
                x = x_fn(span)
                if x < 0 or x >= width_cells:
                    break
                cell = (x, y)
                if cell in mask:
                    continue
                mask.add(cell)
                removed += 1
            if removed >= remaining:
                break
        return removed


__all__ = ["TileSolverOrchestrator", "PhaseLog", "PhaseAttempt"]
