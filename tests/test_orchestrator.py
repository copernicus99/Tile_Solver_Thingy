import unittest

from solver.orchestrator import TileSolverOrchestrator


class CandidateBoardTests(unittest.TestCase):
    def setUp(self):
        self.orchestrator = TileSolverOrchestrator()

    def test_returns_single_square_candidate(self):
        unit_area = self.orchestrator.unit_ft ** 2
        total_area_ft = unit_area * (4 * 4)
        candidates = self.orchestrator._candidate_boards(total_area_ft)
        self.assertEqual([(4, 4)], candidates)

    def test_returns_largest_square_not_exceeding_total_area(self):
        unit_area = self.orchestrator.unit_ft ** 2
        total_area_ft = unit_area * 12  # 12 cells cannot form a perfect square board
        candidates = self.orchestrator._candidate_boards(total_area_ft)
        self.assertEqual([(3, 3)], candidates)


if __name__ == "__main__":
    unittest.main()
