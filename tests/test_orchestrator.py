import unittest

from solver.orchestrator import TileSolverOrchestrator


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


if __name__ == "__main__":
    unittest.main()
