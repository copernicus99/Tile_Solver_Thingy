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
    target_cells: int
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
        min_notch_span = self._derive_min_notch_span(tile_quantities)
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
        candidate_boards = list(
            self._candidate_boards(total_area_ft, max_pop_out_depth, min_notch_span)
        )
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
                self._phase_board_attempts(
                    candidate_boards,
                    phase.allow_pop_outs,
                    phase.allow_discards,
                )
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

    def _derive_min_notch_span(self, tile_quantities: Dict[TileType, int]) -> int:
        min_side_ft = None
        for tile in tile_quantities.keys():
            candidate = min(tile.width_ft, tile.height_ft)
            if min_side_ft is None or candidate < min_side_ft:
                min_side_ft = candidate

        if not min_side_ft or min_side_ft <= 0:
            return 1

        span_cells = int(round(min_side_ft / self.unit_ft))
        if span_cells <= 0:
            return 1
        if not math.isclose(min_side_ft / self.unit_ft, span_cells, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("Tile dimensions must align with the grid size defined by GRID_UNIT_FT.")
        return span_cells

    def _phase_board_attempts(
        self,
        candidates: Sequence[BoardCandidate],
        allow_pop_outs: bool,
        allow_discards: bool,
    ) -> Iterable[Tuple[int, int, Optional[Tuple[Tuple[bool, ...], ...]], Optional[int]]]:
        max_variants = max(getattr(SETTINGS, "MAX_POP_OUT_VARIANTS_PER_BOARD", 0), 0)
        for candidate in candidates:
            board_area = candidate.width * candidate.height
            target = candidate.target_cells
            if not allow_discards and board_area < target:
                continue
            if not allow_pop_outs and board_area > target:
                continue
            yield candidate.width, candidate.height, None, None
            if not allow_pop_outs or max_variants <= 0:
                continue
            for index, mask in enumerate(candidate.pop_out_masks[:max_variants], start=1):
                yield candidate.width, candidate.height, mask, index

    def _candidate_boards(
        self, total_area_ft: float, max_pop_out_depth: int, min_notch_span: int
    ) -> List[BoardCandidate]:
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
            pop_out_masks = self._generate_pop_out_masks(
                cells, cells, area_cells, max_pop_out_depth, min_notch_span
            )
            boards.append(BoardCandidate(cells, cells, area_cells, pop_out_masks))

        return boards

    def _generate_pop_out_masks(
        self,
        width: int,
        height: int,
        target_cells: int,
        max_depth: int,
        min_notch_span: int,
    ) -> Tuple[Tuple[Tuple[bool, ...], ...], ...]:
        slack = width * height - target_cells
        if slack <= 0:
            slack = self._resolve_slack_for_masks(
                width, height, max_depth, min_notch_span
            )
        if slack <= 0:
            return ()

        max_variants = max(getattr(SETTINGS, "MAX_POP_OUT_VARIANTS_PER_BOARD", 0), 0)
        if max_variants <= 0:
            return ()

        if max_depth <= 0:
            return ()
        min_notch_span = max(1, min_notch_span)

        masks: List[Tuple[Tuple[bool, ...], ...]] = []
        seen: set[Tuple[Tuple[bool, ...], ...]] = set()

        def build_mask(notches: Sequence[Tuple[str, int, int]]) -> None:
            if len(masks) >= max_variants:
                return
            mask_tuple = self._render_mask_from_notches(
                width, height, slack, notches
            )
            if mask_tuple is None or mask_tuple in seen:
                return
            seen.add(mask_tuple)
            masks.append(mask_tuple)

        pair_options = self._enumerate_mirrored_notch_options(
            width, height, slack, max_depth, min_notch_span
        )

        opposite_pairs = (("top", "bottom"), ("left", "right"))

        for pair in opposite_pairs:
            for first_notch, second_notch, removed in pair_options[pair]:
                if removed == slack:
                    build_mask((first_notch, second_notch))
                if len(masks) >= max_variants:
                    return tuple(masks)

        horizontal_pair = opposite_pairs[0]
        vertical_pair = opposite_pairs[1]
        for h_first, h_second, h_removed in pair_options[horizontal_pair]:
            for v_first, v_second, v_removed in pair_options[vertical_pair]:
                if h_removed + v_removed != slack:
                    continue
                build_mask((h_first, h_second, v_first, v_second))
                if len(masks) >= max_variants:
                    return tuple(masks)

        return tuple(masks)

    def _resolve_slack_for_masks(
        self,
        width: int,
        height: int,
        max_depth: int,
        min_notch_span: int,
    ) -> int:
        """Find the smallest slack that can yield a mirrored mask variant."""

        max_cells = width * height
        if max_cells <= 0:
            return 0

        start = max(2, 2 * min_notch_span)
        if start % 2 != 0:
            start += 1

        for candidate in range(start, max_cells + 1, 2):
            pair_options = self._enumerate_mirrored_notch_options(
                width, height, candidate, max_depth, min_notch_span
            )
            if not pair_options:
                continue
            if self._has_mask_for_slack(pair_options, candidate):
                return candidate

        return 0

    def _has_mask_for_slack(
        self,
        pair_options: Dict[
            Tuple[str, str], List[Tuple[Tuple[str, int, int], Tuple[str, int, int], int]]
        ],
        slack: int,
    ) -> bool:
        horizontal_pair = ("top", "bottom")
        vertical_pair = ("left", "right")

        horizontal_entries = pair_options.get(horizontal_pair, [])
        vertical_entries = pair_options.get(vertical_pair, [])

        for _, _, removed in horizontal_entries:
            if removed == slack:
                return True
        for _, _, removed in vertical_entries:
            if removed == slack:
                return True

        for h_first, h_second, h_removed in horizontal_entries:
            for v_first, v_second, v_removed in vertical_entries:
                if h_removed + v_removed == slack:
                    return True

        return False

    def _render_mask_from_notches(
        self,
        width: int,
        height: int,
        slack: int,
        notches: Sequence[Tuple[str, int, int]],
    ) -> Optional[Tuple[Tuple[bool, ...], ...]]:
        mask = [[True for _ in range(width)] for _ in range(height)]
        removed_cells = set()

        def emit_cell(row: int, col: int) -> bool:
            if row < 0 or row >= height or col < 0 or col >= width:
                return False
            if (row, col) in removed_cells:
                return False
            removed_cells.add((row, col))
            mask[row][col] = False
            return True

        for orientation, depth, length in notches:
            if orientation in ("top", "bottom"):
                limit_depth = height
                limit_length = width
            else:
                limit_depth = width
                limit_length = height

            if depth <= 0 or length <= 0:
                return None
            if depth > limit_depth or length > limit_length:
                return None

            if orientation == "top":
                for offset in range(depth):
                    row_idx = offset
                    for col in range(length):
                        if not emit_cell(row_idx, col):
                            return None
            elif orientation == "bottom":
                for offset in range(depth):
                    row_idx = height - 1 - offset
                    for col in range(length):
                        if not emit_cell(row_idx, width - 1 - col):
                            return None
            elif orientation == "left":
                for offset in range(depth):
                    col_idx = offset
                    for row in range(length):
                        if not emit_cell(row, col_idx):
                            return None
            elif orientation == "right":
                for offset in range(depth):
                    col_idx = width - 1 - offset
                    for row in range(length):
                        if not emit_cell(height - 1 - row, col_idx):
                            return None
            else:
                return None

        if len(removed_cells) != slack:
            return None

        return tuple(tuple(row) for row in mask)

    def _enumerate_mirrored_notch_options(
        self,
        width: int,
        height: int,
        slack: int,
        max_depth: int,
        min_notch_span: int,
    ) -> Dict[
        Tuple[str, str], List[Tuple[Tuple[str, int, int], Tuple[str, int, int], int]]
    ]:
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

        opposite_pairs = (("top", "bottom"), ("left", "right"))

        pair_options: Dict[
            Tuple[str, str], List[Tuple[Tuple[str, int, int], Tuple[str, int, int], int]]
        ] = {}

        for pair in opposite_pairs:
            first, second = pair
            depth_limit = min(
                max_depth,
                orientation_depth_limits[first],
                orientation_depth_limits[second],
            )
            dim = min(orientation_dims[first], orientation_dims[second])
            options: List[Tuple[Tuple[str, int, int], Tuple[str, int, int], int]] = []
            for depth in range(1, depth_limit + 1):
                max_length = min(dim, slack // (2 * depth))
                if max_length < min_notch_span:
                    continue
                for length in range(min_notch_span, max_length + 1):
                    removed = 2 * depth * length
                    if removed <= 0 or removed > slack:
                        continue
                    options.append(((first, depth, length), (second, depth, length), removed))
            pair_options[pair] = options

        return pair_options


__all__ = ["TileSolverOrchestrator", "PhaseLog", "PhaseAttempt", "BoardCandidate"]
