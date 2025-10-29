from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

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
        self._min_tile_edge_cells = self._compute_min_tile_edge()
        self._mask_cache: Dict[Tuple[int, int, int, int], Tuple[Tuple[Tuple[bool, ...], ...], ...]] = {}

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
            allow_overage = phase in (SETTINGS.PHASE_A, SETTINGS.PHASE_C)
            phase_candidates = list(
                self._phase_board_attempts(
                    candidate_boards,
                    phase.allow_pop_outs,
                    phase.allow_discards,
                    allow_overage_without_popouts=allow_overage,
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

    def _phase_board_attempts(
        self,
        candidates: Sequence[BoardCandidate],
        allow_pop_outs: bool,
        allow_discards: bool,
        *,
        allow_overage_without_popouts: bool = False,
    ) -> Iterable[Tuple[int, int, Optional[Tuple[Tuple[bool, ...], ...]], Optional[int]]]:
        max_variants = max(getattr(SETTINGS, "MAX_POP_OUT_VARIANTS_PER_BOARD", 0), 0)
        for candidate in candidates:
            board_area = candidate.width * candidate.height
            target = candidate.target_cells
            if not allow_discards and board_area < target:
                continue
            if (
                not allow_pop_outs
                and board_area > target
                and not allow_overage_without_popouts
            ):
                continue
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
            boards.append(BoardCandidate(cells, cells, area_cells, pop_out_masks))

        return boards

    def _generate_pop_out_masks(
        self, width: int, height: int, target_cells: int, max_depth: int
    ) -> Tuple[Tuple[Tuple[bool, ...], ...], ...]:
        cache_key = (width, height, target_cells, max_depth)
        cached = self._mask_cache.get(cache_key)
        if cached is not None:
            return cached
        slack = self._resolve_slack_for_mask(width, height, target_cells, max_depth)
        if slack <= 0:
            self._mask_cache[cache_key] = ()
            return ()

        max_variants = max(getattr(SETTINGS, "MAX_POP_OUT_VARIANTS_PER_BOARD", 0), 0)
        if max_variants <= 0:
            return ()

        if max_depth <= 0:
            return ()

        masks: List[Tuple[Tuple[bool, ...], ...]] = []
        seen: Set[Tuple[Tuple[bool, ...], ...]] = set()

        def build_mask(notches: Sequence[Tuple[str, int, int, int]]) -> None:
            if len(masks) >= max_variants:
                return
            if not notches:
                return
            if not self._has_mirrored_notch(notches):
                return
            rendered = self._render_mask_from_notches(width, height, notches)
            if rendered is None:
                return
            mask_tuple, removed = rendered
            if removed != slack:
                return
            if mask_tuple in seen:
                return
            seen.add(mask_tuple)
            masks.append(mask_tuple)

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

        def sort_notches(notches: Sequence[Tuple[str, int, int, int]]) -> Tuple[Tuple[str, int, int, int], ...]:
            order = {"top": 0, "bottom": 1, "left": 2, "right": 3}
            return tuple(sorted(notches, key=lambda item: (order[item[0]], item[3], item[1], item[2])))

        def enumerate_orientation_layouts(
            orientation: str,
        ) -> Dict[int, List[Tuple[Tuple[str, int, int, int], ...]]]:
            depth_limit = min(max_depth, orientation_depth_limits[orientation])
            dim = orientation_dims[orientation]
            layouts: Dict[int, List[Tuple[Tuple[str, int, int, int], ...]]] = defaultdict(list)
            layout_cap = max(1, max_variants)
            if depth_limit <= 0 or dim <= 0:
                layouts[0].append(tuple())
                return layouts

            max_notches = max(1, min(slack if slack > 0 else 0, dim, 8))

            def backtrack(
                start_index: int,
                used_mask: int,
                removed: int,
                placements: Tuple[Tuple[str, int, int, int], ...],
                notch_count: int,
            ) -> None:
                if notch_count > max_notches:
                    return
                options = layouts[removed]
                if len(options) < layout_cap:
                    options.append(placements)
                if removed >= slack:
                    return
                for start in range(start_index, dim):
                    if used_mask & (1 << start):
                        continue
                    max_length = 0
                    while (
                        start + max_length < dim
                        and not (used_mask & (1 << (start + max_length)))
                    ):
                        max_length += 1
                    if max_length <= 0:
                        continue
                    for depth in range(1, depth_limit + 1):
                        remaining_slack = slack - removed
                        max_length_for_depth = min(
                            max_length,
                            remaining_slack // depth if remaining_slack > 0 else 0,
                        )
                        if max_length_for_depth <= 0:
                            continue
                        for length in range(1, max_length_for_depth + 1):
                            segment_mask = ((1 << length) - 1) << start
                            if used_mask & segment_mask:
                                continue
                            removed_cells = depth * length
                            new_removed = removed + removed_cells
                            new_notches = placements + ((orientation, depth, length, start),)
                            backtrack(
                                start + length,
                                used_mask | segment_mask,
                                new_removed,
                                new_notches,
                                notch_count + 1,
                            )

            backtrack(0, 0, 0, tuple(), 0)
            return layouts

        top_layouts = enumerate_orientation_layouts("top")
        bottom_layouts = enumerate_orientation_layouts("bottom")
        left_layouts = enumerate_orientation_layouts("left")
        right_layouts = enumerate_orientation_layouts("right")

        for top_removed, top_options in top_layouts.items():
            for bottom_removed, bottom_options in bottom_layouts.items():
                horizontal_removed = top_removed + bottom_removed
                if horizontal_removed > slack:
                    continue
                for left_removed, left_options in left_layouts.items():
                    total_horizontal_vertical = horizontal_removed + left_removed
                    if total_horizontal_vertical > slack:
                        continue
                    for right_removed, right_options in right_layouts.items():
                        total_removed = total_horizontal_vertical + right_removed
                        if total_removed != slack:
                            continue
                        for top_notches in top_options:
                            for bottom_notches in bottom_options:
                                for left_notches in left_options:
                                    for right_notches in right_options:
                                        combined = sort_notches(
                                            top_notches
                                            + bottom_notches
                                            + left_notches
                                            + right_notches
                                        )
                                        build_mask(combined)
                                        if len(masks) >= max_variants:
                                            return tuple(masks)

        result = tuple(masks)
        self._mask_cache[cache_key] = result
        return result

    def _render_mask_from_notches(
        self, width: int, height: int, notches: Sequence[Tuple[str, int, int, int]]
    ) -> Optional[Tuple[Tuple[Tuple[bool, ...], ...], int]]:
        mask = [[True for _ in range(width)] for _ in range(height)]
        removed = 0
        for orientation, depth, length, start in notches:
            if orientation in ("top", "bottom"):
                if depth > height or length > width:
                    return None
                if start < 0 or start + length > width:
                    return None
                for offset in range(depth):
                    row_idx = offset if orientation == "top" else height - 1 - offset
                    for col in range(length):
                        col_idx = start + col
                        if not mask[row_idx][col_idx]:
                            return None
                        mask[row_idx][col_idx] = False
                        removed += 1
            else:
                if depth > width or length > height:
                    return None
                if start < 0 or start + length > height:
                    return None
                for offset in range(depth):
                    col_idx = offset if orientation == "left" else width - 1 - offset
                    for row in range(length):
                        row_idx = start + row
                        if not mask[row_idx][col_idx]:
                            return None
                        mask[row_idx][col_idx] = False
                        removed += 1
        return tuple(tuple(row) for row in mask), removed

    @staticmethod
    def _has_mirrored_notch(notches: Sequence[Tuple[str, int, int, int]]) -> bool:
        if not notches:
            return False
        counts = Counter(orientation for orientation, _, _, _ in notches)
        top = counts.get("top", 0)
        bottom = counts.get("bottom", 0)
        left = counts.get("left", 0)
        right = counts.get("right", 0)
        if (top > 0) != (bottom > 0):
            return False
        if (left > 0) != (right > 0):
            return False
        return top > 0 or left > 0

    def _compute_min_tile_edge(self) -> int:
        edge_lengths: List[int] = []
        for tile in self.tile_types.values():
            width_cells, height_cells = tile.as_cells(self.unit_ft)
            edge_lengths.append(width_cells)
            edge_lengths.append(height_cells)
        edge_lengths = [edge for edge in edge_lengths if edge > 0]
        if not edge_lengths:
            return 0
        return min(edge_lengths)

    def _minimum_mask_span(self, width: int, height: int) -> int:
        if self._min_tile_edge_cells <= 0:
            return 0
        orientation_lengths = []
        if width > 0 and height > 0:
            orientation_lengths.append(min(self._min_tile_edge_cells, width))
            orientation_lengths.append(min(self._min_tile_edge_cells, height))
        orientation_lengths = [length for length in orientation_lengths if length > 0]
        if not orientation_lengths:
            return 0
        length = max(1, min(orientation_lengths))
        return 2 * length

    def _resolve_slack_for_mask(
        self, width: int, height: int, target_cells: int, max_depth: int
    ) -> int:
        slack = width * height - target_cells
        if slack == 0:
            return 0
        if max_depth <= 0:
            return 0
        minimum = self._minimum_mask_span(width, height)
        required = abs(slack)
        if minimum > 0:
            required = max(required, minimum)
        return required


__all__ = ["TileSolverOrchestrator", "PhaseLog", "PhaseAttempt", "BoardCandidate"]
