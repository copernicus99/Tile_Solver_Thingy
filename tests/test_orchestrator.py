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

    def test_raises_for_non_square_boards(self):
        unit_area = self.orchestrator.unit_ft ** 2
        total_area_ft = unit_area * 12  # 12 cells cannot form a square board
        with self.assertRaises(ValueError) as ctx:
            self.orchestrator._candidate_boards(total_area_ft)
        self.assertIn("square board", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
