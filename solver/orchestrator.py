from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from config import SETTINGS, PhaseConfig
try:  # pragma: no cover - prefer package-relative import
    from .backtracking_solver import BacktrackingSolver, SolverOptions
    from .models import SolveRequest, SolveResult, TileType
except ImportError:  # pragma: no cover - allow running as a script
    from solver.backtracking_solver import BacktrackingSolver, SolverOptions  # type: ignore[no-redef]
    from solver.models import SolveRequest, SolveResult, TileType  # type: ignore[no-redef]


@dataclass(frozen=True)
class BoardCandidate:
    width: int
    height: int
    pop_out_masks: Tuple[Tuple[Tuple[bool, ...], ...], ...]


@dataclass
class PhaseAttempt:
    phase_name: str
    board_size_ft: Tuple[float, float]
    board_size_cells: Tuple[int, int]
    elapsed: float
    backtracks: int
    success: bool
    variant_kind: str = "initial"
    variant_index: Optional[int] = None

    @property
    def variant_label(self) -> str:
        if self.variant_kind == "mask":
            if self.variant_index is not None:
                return f"Mask {self.variant_index}"
            return "Mask"
        return "Initial"


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
        max_pop_out_depth = self._derive_pop_out_depth_limit(tile_quantities)
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
        candidate_boards = list(self._candidate_boards(total_area_ft, max_pop_out_depth))
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
            phase_candidates = list(
                self._phase_board_attempts(candidate_boards, phase.allow_pop_outs)
            )
            total_attempts = len(phase_candidates)
            for attempt_index, (board_w, board_h, mask, mask_index) in enumerate(
                phase_candidates, start=1
            ):
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
                variant_kind = "mask" if mask is not None else "initial"
                variant_label = (
                    "Initial"
                    if variant_kind == "initial"
                    else (f"Mask {mask_index}" if mask_index is not None else "Mask")
                )
                request = SolveRequest(
                    tile_quantities,
                    board_w,
                    board_h,
                    allow_rotation=phase.allow_rotation,
                    allow_pop_outs=phase.allow_pop_outs,
                    allow_discards=phase.allow_discards,
                    board_mask=[list(row) for row in mask] if mask is not None else None,
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
                        variant_kind=variant_kind,
                        variant_index=mask_index,
                        variant_label=variant_label,
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
                    phase_index=phase_index,
                    time_limit_sec=attempt_limit_value,
                    overall_elapsed=time.time() - overall_start,
                    phase_elapsed=time.time() - phase_start,
                    variant_kind=variant_kind,
                    variant_index=mask_index,
                    variant_label=variant_label,
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
                    variant_kind=variant_kind,
                    variant_index=mask_index,
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
                    variant_kind=variant_kind,
                    variant_index=mask_index,
                    variant_label=variant_label,
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
        remaining_attempts = max(total_attempts - 1, 0)
        if remaining_attempts == 0 and total_attempts <= 1:
            # Allow the lone attempt to consume the entire phase allotment.
            first_share = 1.0
        remainder_share = max(0.0, 1.0 - first_share)
        if attempt_index == 1:
            share = first_share
        elif remaining_attempts > 0:
            share = remainder_share / remaining_attempts if remaining_attempts else 0.0
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

    def _derive_pop_out_depth_limit(self, tile_quantities: Dict[TileType, int]) -> int:
        longest_leg_ft = 0.0
        for tile in tile_quantities.keys():
            longest_leg_ft = max(longest_leg_ft, tile.width_ft, tile.height_ft)

        if longest_leg_ft <= 0:
            return max(getattr(SETTINGS, "MAX_POP_OUT_DEPTH", 2), 1)

        depth_ft = longest_leg_ft - 1.0
        if depth_ft <= 0:
            return 0

        depth_cells = int(math.floor(depth_ft / self.unit_ft + 1e-9))
        return max(depth_cells, 0)

    def _phase_board_attempts(
        self, candidates: Sequence[BoardCandidate], allow_pop_outs: bool
    ) -> Iterable[Tuple[int, int, Optional[Tuple[Tuple[bool, ...], ...]], Optional[int]]]:
        max_variants = max(getattr(SETTINGS, "MAX_POP_OUT_VARIANTS_PER_BOARD", 0), 0)
        for candidate in candidates:
            yield candidate.width, candidate.height, None, None
            if not allow_pop_outs or max_variants <= 0:
                continue
            for index, mask in enumerate(candidate.pop_out_masks[:max_variants], start=1):
                yield candidate.width, candidate.height, mask, index

    def _candidate_boards(self, total_area_ft: float, max_pop_out_depth: int) -> List[BoardCandidate]:
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
            pop_out_masks = self._generate_pop_out_masks(cells, cells, area_cells, max_pop_out_depth)
            boards.append(BoardCandidate(cells, cells, pop_out_masks))

        return boards

    def _generate_pop_out_masks(
        self, width: int, height: int, target_cells: int, max_depth: int
    ) -> Tuple[Tuple[Tuple[bool, ...], ...], ...]:
        slack = width * height - target_cells
        if slack <= 0:
            # When the requested tile coverage exceeds the base board area we still
            # want to explore pop-out variants. Fallback to removing a single strip
            # of cells so that downstream code can construct up to the configured
            # number of masks for the board.
            slack = min(width, height)

        max_variants = max(getattr(SETTINGS, "MAX_POP_OUT_VARIANTS_PER_BOARD", 0), 0)
        if max_variants <= 0:
            return ()

        if max_depth <= 0:
            return ()
        masks: List[Tuple[Tuple[bool, ...], ...]] = []

        def build_mask(notches: Sequence[Tuple[str, int, int]]) -> None:
            if len(masks) >= max_variants:
                return
            mask = [[True for _ in range(width)] for _ in range(height)]
            removed = 0
            for orientation, depth, length in notches:
                if orientation in ("top", "bottom"):
                    if depth > height:
                        return
                    if length > width:
                        return
                    for offset in range(depth):
                        row_idx = offset if orientation == "top" else height - 1 - offset
                        for col in range(length):
                            col_idx = col if orientation == "top" else width - 1 - col
                            if not mask[row_idx][col_idx]:
                                return
                            mask[row_idx][col_idx] = False
                            removed += 1
                else:
                    if depth > width:
                        return
                    if length > height:
                        return
                    for offset in range(depth):
                        col_idx = offset if orientation == "left" else width - 1 - offset
                        for row in range(length):
                            row_idx = row if orientation == "left" else height - 1 - row
                            if not mask[row_idx][col_idx]:
                                return
                            mask[row_idx][col_idx] = False
                            removed += 1
            if removed != slack:
                return
            masks.append(tuple(tuple(row) for row in mask))

        orientations = ("top", "bottom", "left", "right")
        orientation_dims = {
            "top": width,
            "bottom": width,
            "left": height,
            "right": height,
        }
        orientation_depth_limits = {
            "top": height,
            "bottom": height,
            "left": width,
            "right": width,
        }

        for orientation in orientations:
            dim = orientation_dims[orientation]
            depth_limit = min(max_depth, orientation_depth_limits[orientation])
            for depth in range(1, depth_limit + 1):
                if slack % depth != 0:
                    continue
                length = slack // depth
                if 0 < length <= dim:
                    build_mask(((orientation, depth, length),))
                if len(masks) >= max_variants:
                    return tuple(masks)

        opposite_pairs = (("top", "bottom"), ("left", "right"))
        for first, second in opposite_pairs:
            first_dim = orientation_dims[first]
            second_dim = orientation_dims[second]
            first_depth_limit = min(max_depth, orientation_depth_limits[first])
            second_depth_limit = min(max_depth, orientation_depth_limits[second])
            for depth1 in range(1, first_depth_limit + 1):
                for depth2 in range(1, second_depth_limit + 1):
                    max_length1 = min(first_dim, slack // depth1)
                    for length1 in range(1, max_length1 + 1):
                        remaining = slack - depth1 * length1
                        if remaining <= 0:
                            continue
                        if remaining % depth2 != 0:
                            continue
                        length2 = remaining // depth2
                        if 0 < length2 <= second_dim:
                            build_mask(((first, depth1, length1), (second, depth2, length2)))
                        if len(masks) >= max_variants:
                            return tuple(masks)

        return tuple(masks)


__all__ = ["TileSolverOrchestrator", "PhaseLog", "PhaseAttempt", "BoardCandidate"]
