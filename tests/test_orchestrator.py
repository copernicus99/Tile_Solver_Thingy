import math
import unittest
from collections import Counter, OrderedDict
from contextlib import ExitStack
from typing import Dict, List, Optional, Tuple
from unittest import mock

from config import SETTINGS

from solver.backtracking_solver import BacktrackingSolver, SolverOptions
from solver.orchestrator import BoardCandidate, TileSolverOrchestrator
from solver.models import Placement, SolveRequest, SolveResult, SolverStats, TileInstance, TileType


class CandidateBoardTests(unittest.TestCase):
    def setUp(self):
        self.orchestrator = TileSolverOrchestrator()

    def test_returns_six_boards_reducing_by_one_foot(self):
        total_area_ft = 30.0
        default_depth = max(getattr(SETTINGS, "MAX_POP_OUT_DEPTH", 2), 1)
        candidates = self.orchestrator._candidate_boards(total_area_ft, default_depth)
        expected = [
            (12, 12),
            (10, 10),
            (8, 8),
            (6, 6),
            (4, 4),
            (2, 2),
        ]
        self.assertEqual(expected, [(c.width, c.height) for c in candidates])

    def test_clamps_minimum_board_size_to_one_foot(self):
        unit_area = self.orchestrator.unit_ft ** 2
        total_area_ft = unit_area * 1  # corresponds to one cell squared
        default_depth = max(getattr(SETTINGS, "MAX_POP_OUT_DEPTH", 2), 1)
        candidates = self.orchestrator._candidate_boards(total_area_ft, default_depth)
        expected = [(2, 2)] * 6
        self.assertEqual(expected, [(c.width, c.height) for c in candidates])

    def test_small_fractional_area_pads_up_to_one_foot(self):
        default_depth = max(getattr(SETTINGS, "MAX_POP_OUT_DEPTH", 2), 1)
        candidates = self.orchestrator._candidate_boards(0.3, default_depth)
        expected = [(2, 2)] * 6
        self.assertEqual(expected, [(c.width, c.height) for c in candidates])


class PhaseBoardAttemptTests(unittest.TestCase):
    def setUp(self):
        self.orchestrator = TileSolverOrchestrator()

    def test_non_pop_out_phases_skip_oversized_boards_by_default(self):
        candidate = BoardCandidate(width=10, height=10, target_cells=80, pop_out_masks=())
        attempts = list(
            self.orchestrator._phase_board_attempts(
                [candidate],
                allow_pop_outs=False,
                allow_discards=False,
            )
        )
        self.assertFalse(attempts, "Oversized boards should be skipped when overage is disallowed")

    def test_phases_a_and_c_allow_oversized_boards_without_pop_outs(self):
        candidate = BoardCandidate(width=10, height=10, target_cells=80, pop_out_masks=())
        attempts = list(
            self.orchestrator._phase_board_attempts(
                [candidate],
                allow_pop_outs=False,
                allow_discards=False,
                allow_overage_without_popouts=True,
            )
        )
        self.assertEqual(
            [(candidate.width, candidate.height, None, None)],
            attempts,
            "Oversized boards should be attempted when overage is allowed",
        )

    def test_solve_marks_phases_a_and_c_for_oversized_board_attempts(self):
        orchestrator = self.orchestrator
        selection = {"1x1": 4}

        with mock.patch.object(
            TileSolverOrchestrator,
            "_phase_board_attempts",
            autospec=True,
            return_value=[],
        ) as patched:
            orchestrator.solve(selection)

        self.assertGreaterEqual(len(patched.call_args_list), 2)
        phase_a_args = patched.call_args_list[0].kwargs
        self.assertTrue(
            phase_a_args.get("allow_overage_without_popouts"),
            "Phase A should allow oversized boards without pop-outs",
        )
        phase_b_args = patched.call_args_list[1].kwargs
        self.assertFalse(
            phase_b_args.get("allow_overage_without_popouts"),
            "Phase B should not allow oversized boards without pop-outs",
        )


class PopOutBoardTests(unittest.TestCase):
    def setUp(self):
        self.orchestrator = TileSolverOrchestrator()

    def test_depth_limit_scales_with_longest_leg(self):
        tile = self.orchestrator.tile_types["1.5x3"]
        limit = self.orchestrator._derive_pop_out_depth_limit({tile: 1})
        longest_leg = max(tile.width_ft, tile.height_ft)
        expected = int(math.floor((longest_leg - 1.0) / self.orchestrator.unit_ft + 1e-9))
        self.assertEqual(expected, limit)

    def test_depth_limit_zero_when_tiles_are_one_foot(self):
        tile = self.orchestrator.tile_types["1x1"]
        limit = self.orchestrator._derive_pop_out_depth_limit({tile: 4})
        self.assertEqual(0, limit)

    def test_solve_passes_dynamic_depth_to_candidate_boards(self):
        tile_name = "1.5x3"
        tile = self.orchestrator.tile_types[tile_name]
        expected_depth = self.orchestrator._derive_pop_out_depth_limit({tile: 1})

        with mock.patch.object(TileSolverOrchestrator, "_candidate_boards", return_value=[]) as mock_boards:
            self.orchestrator.solve({tile_name: 1})

        mock_boards.assert_called()
        args = mock_boards.call_args[0]
        self.assertGreaterEqual(len(args), 2)
        self.assertEqual(expected_depth, args[1])

    def test_candidate_masks_preserve_target_area(self):
        total_area_ft = 30.0
        default_depth = max(getattr(SETTINGS, "MAX_POP_OUT_DEPTH", 2), 1)
        candidates = self.orchestrator._candidate_boards(total_area_ft, default_depth)
        self.assertTrue(candidates, "Expected at least one candidate board")
        first = candidates[0]
        self.assertTrue(first.pop_out_masks, "Pop-out masks should be generated when slack allows")
        mask = first.pop_out_masks[0]
        available_cells = sum(1 for row in mask for cell in row if cell)
        unit_area = self.orchestrator.unit_ft ** 2
        padded_area_ft = math.ceil(total_area_ft)
        target_cells = int(round(padded_area_ft / unit_area))
        self.assertEqual(target_cells, available_cells)

    def test_masks_generated_even_when_tiles_exceed_board_area(self):
        total_area_ft = 150.0
        default_depth = max(getattr(SETTINGS, "MAX_POP_OUT_DEPTH", 2), 1)
        candidates = self.orchestrator._candidate_boards(total_area_ft, default_depth)
        unit_area = self.orchestrator.unit_ft ** 2
        padded_area_ft = math.ceil(total_area_ft)
        target_cells = int(round(padded_area_ft / unit_area))
        max_variants = max(getattr(SETTINGS, "MAX_POP_OUT_VARIANTS_PER_BOARD", 0), 0)
        self.assertGreater(max_variants, 0)
        insufficient = [
            c for c in candidates if (c.width * c.height) < target_cells
        ]
        self.assertTrue(
            insufficient,
            "Expected at least one board candidate smaller than the tile coverage",
        )
        for candidate in insufficient:
            mask_count = len(candidate.pop_out_masks)
            self.assertGreater(
                mask_count,
                0,
                "Boards with insufficient area should still receive at least one mirrored variant",
            )
            self.assertLessEqual(
                mask_count,
                max_variants,
                "Boards should never exceed the configured pop-out variant cap",
            )

    def test_generated_masks_are_mirrored(self):
        width = 6
        height = 6
        total_cells = width * height
        slack = 4  # ensure at least one mirrored pair is required
        default_depth = max(getattr(SETTINGS, "MAX_POP_OUT_DEPTH", 2), 1)
        self.orchestrator._mask_cache.clear()
        captured = {}
        original = TileSolverOrchestrator._render_mask_from_notches

        def wrapper(self_obj, w, h, notches):
            result = original(self_obj, w, h, notches)
            if result is not None:
                mask_tuple, _ = result
                captured[mask_tuple] = tuple(sorted(notches))
            return result

        with mock.patch.object(TileSolverOrchestrator, "_render_mask_from_notches", new=wrapper):
            masks = self.orchestrator._generate_pop_out_masks(
                width, height, total_cells - slack, default_depth
            )
        self.assertTrue(masks, "Expected mirrored pop-out masks to be generated")
        for mask in masks:
            notches = captured.get(mask)
            self.assertIsNotNone(notches, "Expected to capture notches for each mask")
            counts = Counter(orientation for orientation, _, _, _ in notches)
            top = counts.get("top", 0)
            bottom = counts.get("bottom", 0)
            left = counts.get("left", 0)
            right = counts.get("right", 0)
            self.assertEqual(
                top > 0,
                bottom > 0,
                "Horizontal pop-outs must appear on both top and bottom edges",
            )
            self.assertEqual(
                left > 0,
                right > 0,
                "Vertical pop-outs must appear on both left and right edges",
            )
            self.assertTrue(
                top > 0 or left > 0,
                "Each mask should include at least one mirrored notch pair",
            )

    def test_allows_offset_mirrored_masks(self):
        width = 10
        height = 10
        total_cells = width * height
        slack = 24
        default_depth = max(getattr(SETTINGS, "MAX_POP_OUT_DEPTH", 2), 1)
        masks = self.orchestrator._generate_pop_out_masks(
            width, height, total_cells - slack, default_depth
        )
        self.assertTrue(masks, "Expected pop-out masks to be generated for offset checks")
        found_offset = False
        for mask in masks:
            top_columns = {idx for idx, cell in enumerate(mask[0]) if not cell}
            bottom_columns = {idx for idx, cell in enumerate(mask[-1]) if not cell}
            if top_columns and bottom_columns and top_columns != bottom_columns:
                found_offset = True
                break
        self.assertTrue(
            found_offset,
            "Generator should emit masks where mirrored edges remove different segments",
        )

    def test_generated_masks_are_unique(self):
        width = 8
        height = 8
        total_cells = width * height
        slack = 12
        default_depth = max(getattr(SETTINGS, "MAX_POP_OUT_DEPTH", 2), 1)
        masks = self.orchestrator._generate_pop_out_masks(
            width, height, total_cells - slack, default_depth
        )
        self.assertGreater(len(masks), 1, "Expected multiple pop-out masks to be generated")
        unique_masks = {mask for mask in masks}
        self.assertEqual(
            len(masks),
            len(unique_masks),
            "Pop-out mask variants must be unique",
        )

    def test_negative_slack_masks_remove_full_deficit(self):
        width = 10
        height = 10
        total_cells = width * height
        target_cells = total_cells + 20
        default_depth = max(getattr(SETTINGS, "MAX_POP_OUT_DEPTH", 2), 1)
        masks = self.orchestrator._generate_pop_out_masks(
            width, height, target_cells, default_depth
        )
        self.assertTrue(
            masks,
            "Pop-out masks should be generated when the board is smaller than the tile coverage",
        )
        raw_deficit = abs(total_cells - target_cells)
        minimum_span = self.orchestrator._minimum_mask_span(width, height)
        expected_removed = max(raw_deficit, minimum_span)
        for mask in masks:
            available = sum(1 for row in mask for cell in row if cell)
            removed = total_cells - available
            self.assertEqual(
                expected_removed,
                removed,
                "Pop-out masks should remove the larger of the true deficit or the minimum mirrored span",
            )

    def test_render_mask_accepts_offset_mirrored_notches(self):
        width = 10
        height = 10
        notches = (
            ("top", 1, 2, 3),
            ("bottom", 1, 2, 3),
            ("left", 1, 2, 4),
            ("right", 1, 2, 4),
        )
        rendered = self.orchestrator._render_mask_from_notches(width, height, notches)
        self.assertIsNotNone(rendered, "Offset mirrored notches should render successfully")
        mask, removed = rendered  # type: ignore[misc]
        self.assertEqual(removed, 8)
        # Ensure the removed cells include interior offsets rather than just corners.
        removed_cells = {
            (r, c)
            for r, row in enumerate(mask)
            for c, cell in enumerate(row)
            if not cell
        }
        self.assertIn((0, 3), removed_cells)
        self.assertIn((height - 1, 3), removed_cells)
        self.assertIn((4, 0), removed_cells)
        self.assertIn((4, width - 1), removed_cells)


class SolverOptionTests(unittest.TestCase):
    def test_max_edge_limit_scales_with_board_size(self):
        orchestrator = TileSolverOrchestrator()
        board_ft = 8
        board_length_cells = int(round(board_ft / orchestrator.unit_ft))

        limit = orchestrator._max_edge_for_dimension(board_length_cells)
        expected = int(math.floor(board_length_cells * SETTINGS.MAX_EDGE_RATIO + 1e-9))
        expected = max(1, min(board_length_cells, expected))

        self.assertEqual(expected, limit)

    def test_absolute_cap_still_limits_large_boards(self):
        orchestrator = TileSolverOrchestrator()
        board_length_cells = int(round(12 / orchestrator.unit_ft))

        limit = orchestrator._max_edge_for_dimension(board_length_cells)
        expected = int(round(SETTINGS.MAX_EDGE_FT / orchestrator.unit_ft))

        self.assertEqual(expected, limit)

    def test_perimeter_edges_are_included_in_straight_edge_limit(self):
        orchestrator = TileSolverOrchestrator()
        captured_options = []

        class FakeSolver:
            def __init__(self, request, options, unit_ft, phase_name, progress_callback=None):
                self.request = request
                self.options = options
                self.unit_ft = unit_ft
                self.phase_name = phase_name
                self.stats = SolverStats()
                captured_options.append(options)

            def solve(self):
                return None

        with mock.patch.object(
            TileSolverOrchestrator,
            "_candidate_boards",
            return_value=[BoardCandidate(14, 14, 14 * 14, tuple())],
        ), mock.patch(
            "solver.orchestrator.BacktrackingSolver", FakeSolver
        ):
            orchestrator.solve({"1x1": 1})

        self.assertTrue(captured_options, "Solver should have been invoked at least once")
        self.assertTrue(
            captured_options[0].max_edge_include_perimeter,
            "Perimeter seams must be included when enforcing straight-edge limits.",
        )


class PhasePopOutIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.orchestrator = TileSolverOrchestrator()
        self.mask = (
            (False, True, True, True),
            (False, True, True, True),
            (True, True, True, True),
            (True, True, True, True),
        )
        self.candidate = [BoardCandidate(4, 4, 14, (self.mask,))]

    def _run_with_overrides(self, selection, phase_sequence=None):
        records: List[Tuple[str, Optional[List[List[bool]]]]] = []

        class RecordingSolver:
            def __init__(self, request, options, unit_ft, phase_name, progress_callback=None):
                self.request = request
                self.options = options
                self.unit_ft = unit_ft
                self.phase_name = phase_name
                self.stats = SolverStats()
                records.append((phase_name, request.board_mask))

            def solve(self):
                return None

        candidate_patch = mock.patch.object(
            TileSolverOrchestrator, "_candidate_boards", return_value=self.candidate
        )
        solver_patch = mock.patch("solver.orchestrator.BacktrackingSolver", RecordingSolver)
        patches = [candidate_patch, solver_patch]
        if phase_sequence is not None:
            patches.append(
                mock.patch.object(
                    TileSolverOrchestrator, "_select_phases", return_value=phase_sequence
                )
            )
        with ExitStack() as stack:
            for patcher in patches:
                stack.enter_context(patcher)
            self.orchestrator.solve(selection)
        return records

    def test_phase_b_attempts_masked_board(self):
        records = self._run_with_overrides({"1x1": 1})
        masks_by_phase: Dict[str, List[Optional[List[List[bool]]]]] = {}
        for phase, mask in records:
            masks_by_phase.setdefault(phase, []).append(mask)
        self.assertIn(SETTINGS.PHASE_B.name, masks_by_phase)
        self.assertTrue(
            any(mask is not None for mask in masks_by_phase[SETTINGS.PHASE_B.name]),
            "Phase B should attempt at least one masked board when pop-outs are enabled.",
        )
        expected_mask = [list(row) for row in self.mask]
        self.assertIn(
            expected_mask,
            masks_by_phase[SETTINGS.PHASE_B.name],
        )
        self.assertTrue(
            all(mask is None for mask in masks_by_phase.get(SETTINGS.PHASE_A.name, [])),
            "Phase A must not receive pop-out masks when disabled.",
        )

    def test_phase_d_attempts_masked_board(self):
        phase_sequence = [SETTINGS.PHASE_C, SETTINGS.PHASE_D]
        records = self._run_with_overrides({"1x1": 250}, phase_sequence=phase_sequence)
        masks_by_phase: Dict[str, List[Optional[List[List[bool]]]]] = {}
        for phase, mask in records:
            masks_by_phase.setdefault(phase, []).append(mask)
        self.assertIn(SETTINGS.PHASE_D.name, masks_by_phase)
        self.assertTrue(
            any(mask is not None for mask in masks_by_phase[SETTINGS.PHASE_D.name]),
            "Phase D should attempt at least one masked board when pop-outs are enabled.",
        )
        expected_mask = [list(row) for row in self.mask]
        self.assertIn(
            expected_mask,
            masks_by_phase[SETTINGS.PHASE_D.name],
        )

    def test_disabling_pop_out_variants_skips_masks(self):
        original_variants = getattr(SETTINGS, "MAX_POP_OUT_VARIANTS_PER_BOARD", None)
        try:
            setattr(SETTINGS, "MAX_POP_OUT_VARIANTS_PER_BOARD", 0)
            records = self._run_with_overrides({"1x1": 1})
        finally:
            setattr(SETTINGS, "MAX_POP_OUT_VARIANTS_PER_BOARD", original_variants)
        self.assertTrue(records, "Solver should still attempt boards")
        self.assertTrue(all(mask is None for _, mask in records))


class DiscardHandlingTests(unittest.TestCase):
    def setUp(self):
        self.unit = SETTINGS.GRID_UNIT_FT
        self.square_tile = TileType("2x2", 2.0, 2.0)
        self.extra_tile = TileType("1x1", 1.0, 1.0)
        cells = int(round(2.0 / self.unit))
        self.board_cells = (cells, cells)
        self.options = SolverOptions(
            max_edge_cells_horizontal=10,
            max_edge_cells_vertical=10,
            max_edge_include_perimeter=True,
            same_shape_limit=SETTINGS.SAME_SHAPE_LIMIT,
            enforce_plus_rule=SETTINGS.PLUS_TOGGLE,
            time_limit_sec=None,
        )

    def _request(self, allow_discards: bool) -> SolveRequest:
        quantities = OrderedDict(
            (
                (self.square_tile, 1),
                (self.extra_tile, 1),
            )
        )
        return SolveRequest(
            tile_quantities=quantities,
            board_width_cells=self.board_cells[0],
            board_height_cells=self.board_cells[1],
            allow_rotation=True,
            allow_pop_outs=False,
            allow_discards=allow_discards,
        )

    def test_solver_requires_all_tiles_when_discards_disallowed(self):
        solver = BacktrackingSolver(
            self._request(allow_discards=False),
            self.options,
            self.unit,
            "Phase Test",
        )
        self.assertIsNone(
            solver.solve(),
            "Solver should not succeed when required tiles cannot all be placed",
        )

    def test_solver_records_discards_when_allowed(self):
        solver = BacktrackingSolver(
            self._request(allow_discards=True),
            self.options,
            self.unit,
            "Phase Test",
        )
        result = solver.solve()
        self.assertIsNotNone(result, "Solver should return a layout when discards are allowed")
        self.assertEqual(len(result.placements), 1)
        self.assertEqual(len(result.discarded_tiles), 1)
        self.assertEqual(result.discarded_tiles[0].type.name, "1x1")

    def test_phase_log_records_discards_for_successful_phase(self):
        orchestrator = TileSolverOrchestrator()
        unit = orchestrator.unit_ft
        tile_type = orchestrator.tile_types["1x1"]
        tile_cells = tile_type.as_cells(unit)
        placed_tile = TileInstance(tile_type, "tile_1", allow_rotation=True)
        discarded_tile = TileInstance(tile_type, "tile_2", allow_rotation=True)
        placement = Placement(placed_tile, 0, 0, *tile_cells)
        fake_result = SolveResult(
            placements=[placement],
            board_width_cells=4,
            board_height_cells=4,
            phase_name="Phase Test",
            board_width_ft=4 * unit,
            board_height_ft=4 * unit,
            discarded_tiles=[discarded_tile],
        )

        class FakeSolver:
            def __init__(self, request, options, unit_ft, phase_name, progress_callback=None):
                self.request = request
                self.options = options
                self.unit_ft = unit_ft
                self.phase_name = phase_name
                self.stats = SolverStats()

            def solve(self):
                return fake_result

        with mock.patch.object(
            TileSolverOrchestrator,
            "_candidate_boards",
            return_value=[BoardCandidate(4, 4, 16, tuple())],
        ), mock.patch(
            "solver.orchestrator.BacktrackingSolver", FakeSolver
        ):
            result, logs = orchestrator.solve({"1x1": 1})

        self.assertIs(result, fake_result)
        phase_with_result = next((log for log in logs if log.result is not None), None)

        self.assertIsNotNone(phase_with_result, "A phase log should capture the successful solver result")
        self.assertIs(phase_with_result.result, result)
        self.assertTrue(result.discarded_tiles, "The solver should report discarded tiles for this scenario")
        self.assertEqual(
            [tile.identifier for tile in result.discarded_tiles],
            [tile.identifier for tile in phase_with_result.result.discarded_tiles],
        )


if __name__ == "__main__":
    unittest.main()
