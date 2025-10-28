import math
import unittest
from collections import OrderedDict
from unittest import mock

from config import SETTINGS

from solver.backtracking_solver import BacktrackingSolver, SolverOptions
from solver.orchestrator import TileSolverOrchestrator
from solver.models import Placement, SolveRequest, SolveResult, SolverStats, TileInstance, TileType


class CandidateBoardTests(unittest.TestCase):
    def setUp(self):
        self.orchestrator = TileSolverOrchestrator()

    def test_returns_six_boards_reducing_by_one_foot(self):
        total_area_ft = 30.0
        candidates = self.orchestrator._candidate_boards(total_area_ft)
        expected = [
            (12, 12),
            (10, 10),
            (8, 8),
            (6, 6),
            (4, 4),
            (2, 2),
        ]
        self.assertEqual(expected, candidates)

    def test_clamps_minimum_board_size_to_one_foot(self):
        unit_area = self.orchestrator.unit_ft ** 2
        total_area_ft = unit_area * 1  # corresponds to one cell squared
        candidates = self.orchestrator._candidate_boards(total_area_ft)
        expected = [(2, 2)] * 6
        self.assertEqual(expected, candidates)

    def test_small_fractional_area_pads_up_to_one_foot(self):
        candidates = self.orchestrator._candidate_boards(0.3)
        expected = [(2, 2)] * 6
        self.assertEqual(expected, candidates)


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

        with mock.patch.object(TileSolverOrchestrator, "_candidate_boards", return_value=[(14, 14)]), mock.patch(
            "solver.orchestrator.BacktrackingSolver", FakeSolver
        ):
            orchestrator.solve({"1x1": 1})

        self.assertTrue(captured_options, "Solver should have been invoked at least once")
        self.assertTrue(
            captured_options[0].max_edge_include_perimeter,
            "Perimeter seams must be included when enforcing straight-edge limits.",
        )


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

        with mock.patch.object(TileSolverOrchestrator, "_candidate_boards", return_value=[(4, 4)]), mock.patch(
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
