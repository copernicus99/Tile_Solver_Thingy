import math
import unittest
from unittest import mock

from config import SETTINGS

from solver.orchestrator import TileSolverOrchestrator
from solver.models import SolverStats


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
        expected_ft = math.ceil(board_ft * SETTINGS.MAX_EDGE_RATIO)
        expected = int(round(expected_ft / orchestrator.unit_ft))

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


if __name__ == "__main__":
    unittest.main()
