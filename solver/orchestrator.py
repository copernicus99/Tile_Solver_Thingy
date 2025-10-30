from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from config import SETTINGS, PhaseConfig
try:  # pragma: no cover - prefer package-relative import
    from .backtracking_solver import BacktrackingSolver, SolverOptions
    from .models import SolveRequest, SolveResult, TileType
    from .mask_builder import generate_mask
except ImportError:  # pragma: no cover - allow running as a script
    from solver.backtracking_solver import BacktrackingSolver, SolverOptions  # type: ignore[no-redef]
    from solver.models import SolveRequest, SolveResult, TileType  # type: ignore[no-redef]
    from solver.mask_builder import generate_mask  # type: ignore[no-redef]


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
        return "Square Fit"


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
        self._mask_cache: Dict[
            Tuple[int, int, int, int, Tuple[Tuple[str, int], ...]],
            Tuple[Tuple[Tuple[bool, ...], ...], ...],
        ] = {}

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
        candidate_boards = list(
            self._candidate_boards(total_area_ft, max_pop_out_depth, tile_quantities)
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
                    "Square Fit"
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
        yielded_initial_boards: Set[Tuple[int, int]] = set()
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
            if (
                allow_overage_without_popouts
                and not allow_pop_outs
                and board_area != target
            ):
                continue
            board_key = (candidate.width, candidate.height)
            if board_key in yielded_initial_boards:
                continue
            yielded_initial_boards.add(board_key)
            yield candidate.width, candidate.height, None, None
            if not allow_pop_outs:
                continue
            for index, mask in enumerate(candidate.pop_out_masks, start=1):
                yield candidate.width, candidate.height, mask, index

    def _candidate_boards(
        self,
        total_area_ft: float,
        max_pop_out_depth: int,
        tile_quantities: Dict[TileType, int],
    ) -> List[BoardCandidate]:
        if total_area_ft <= 0:
            return []

        unit_area = self.unit_ft ** 2
        cells_area = total_area_ft / unit_area
        target_cells = int(round(cells_area))
        if target_cells <= 0:
            return []
        if not math.isclose(cells_area, target_cells, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                "Total tile coverage must align to the grid size. Adjust tile quantities to form a square grid."
            )

        inverse_unit = 1.0 / self.unit_ft
        cells_per_foot = int(round(inverse_unit))
        if cells_per_foot <= 0 or not math.isclose(inverse_unit, cells_per_foot, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("GRID_UNIT_FT must evenly divide one foot to build boards.")

        min_side = max(1, cells_per_foot)
        starting_side_cells = max(min_side, int(math.ceil(math.sqrt(target_cells))))

        boards: List[BoardCandidate] = []
        seen_rectangles: Set[Tuple[int, int]] = set()

        def add_candidate(width_cells: int, height_cells: int) -> None:
            pop_out_masks = self._generate_pop_out_masks(
                width_cells,
                height_cells,
                target_cells,
                max_pop_out_depth,
                tile_quantities,
            )
            boards.append(BoardCandidate(width_cells, height_cells, target_cells, pop_out_masks))

        for reduction in range(6):
            side_cells = starting_side_cells - reduction * cells_per_foot
            if side_cells < min_side:
                side_cells = min_side
            add_candidate(side_cells, side_cells)

        if getattr(SETTINGS, "ALLOW_RECTANGLES", False):
            max_height = int(math.sqrt(target_cells))
            for height_cells in range(min_side, max_height + 1):
                if target_cells % height_cells != 0:
                    continue
                width_cells = target_cells // height_cells
                if width_cells < min_side or width_cells == height_cells:
                    continue
                key = (width_cells, height_cells)
                if key in seen_rectangles:
                    continue
                seen_rectangles.add(key)
                add_candidate(width_cells, height_cells)

        return boards

    def _generate_pop_out_masks(
        self,
        width: int,
        height: int,
        target_cells: int,
        max_depth: int,
        tile_quantities: Dict[TileType, int],
    ) -> Tuple[Tuple[Tuple[bool, ...], ...], ...]:
        signature = self._mask_tile_signature(tile_quantities)
        cache_key = (width, height, target_cells, max_depth, signature)
        cached = self._mask_cache.get(cache_key)
        if cached is not None:
            return cached

        slack = self._resolve_slack_for_mask(width, height, target_cells, max_depth)
        if slack <= 0 or max_depth <= 0:
            self._mask_cache[cache_key] = ()
            return ()

        min_span = self._minimum_mask_span(width, height)
        horizontal_limit = self._max_edge_for_dimension(width)
        vertical_limit = self._max_edge_for_dimension(height)
        attempts = max(getattr(SETTINGS, "MASK_GENERATION_ATTEMPTS", 8), 1)
        validation_attempts = max(getattr(SETTINGS, "MASK_VALIDATION_ATTEMPTS", 4), 1)
        seen: Set[Tuple[Tuple[bool, ...], ...]] = set()

        for attempt in range(attempts):
            seed = hash((width, height, target_cells, max_depth, signature, attempt))
            rng = random.Random(seed)
            mask = generate_mask(
                width,
                height,
                target_cells,
                max_depth,
                min_span,
                horizontal_limit,
                vertical_limit,
                rng=rng,
                attempts=validation_attempts * 2,
            )
            if mask is None:
                continue
            mask_tuple = tuple(tuple(row) for row in mask)
            if mask_tuple in seen:
                continue
            seen.add(mask_tuple)
            if self._mask_is_valid(
                tile_quantities,
                width,
                height,
                target_cells,
                mask_tuple,
            ):
                result = (mask_tuple,)
                self._mask_cache[cache_key] = result
                return result

        self._mask_cache[cache_key] = ()
        return ()

    def _mask_tile_signature(self, tile_quantities: Dict[TileType, int]) -> Tuple[Tuple[str, int], ...]:
        pairs = [
            (tile.name, qty)
            for tile, qty in tile_quantities.items()
            if qty > 0
        ]
        return tuple(sorted(pairs))

    def _mask_is_valid(
        self,
        tile_quantities: Dict[TileType, int],
        width: int,
        height: int,
        target_cells: int,
        mask: Tuple[Tuple[bool, ...], ...],
    ) -> bool:
        available = sum(1 for row in mask for cell in row if cell)
        if available != target_cells:
            return False

        request = SolveRequest(
            tile_quantities,
            width,
            height,
            allow_rotation=True,
            allow_pop_outs=True,
            allow_discards=False,
            board_mask=[list(row) for row in mask],
        )
        validation_limit = getattr(SETTINGS, "MASK_VALIDATION_TIME_LIMIT", 5.0)
        options = SolverOptions(
            max_edge_cells_horizontal=self._max_edge_for_dimension(width),
            max_edge_cells_vertical=self._max_edge_for_dimension(height),
            max_edge_include_perimeter=not getattr(SETTINGS, "MAX_EDGE_INSIDE_ONLY", True),
            same_shape_limit=SETTINGS.SAME_SHAPE_LIMIT,
            enforce_plus_rule=SETTINGS.PLUS_TOGGLE,
            time_limit_sec=validation_limit,
        )
        solver = BacktrackingSolver(
            request,
            options,
            self.unit_ft,
            "Mask Validation",
        )
        return solver.solve() is not None

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
