import unittest

from solver.backtracking_solver import BacktrackingSolver, SolverOptions
from solver.models import SolveRequest, TileType


UNIT = 0.5


def build_solver(width_cells: int, height_cells: int, limit: int, include_perimeter: bool = True) -> BacktrackingSolver:
    cell_tile = TileType("cell", UNIT, UNIT)
    quantities = {cell_tile: width_cells * height_cells}
    request = SolveRequest(
        tile_quantities=quantities,
        board_width_cells=width_cells,
        board_height_cells=height_cells,
        allow_rotation=True,
        allow_pop_outs=False,
        allow_discards=False,
    )
    options = SolverOptions(
        max_edge_cells_horizontal=limit,
        max_edge_cells_vertical=limit,
        max_edge_include_perimeter=include_perimeter,
        same_shape_limit=0,
        enforce_plus_rule=False,
        time_limit_sec=None,
    )
    return BacktrackingSolver(request, options, UNIT, "Test Phase")


class EdgeLimitValidationTests(unittest.TestCase):
    def test_interior_horizontal_run_exceeding_limit_is_rejected(self):
        solver = build_solver(width_cells=10, height_cells=2, limit=6)
        solver.grid = [
            [0] * 10,
            [1] * 10,
        ]

        self.assertFalse(solver._validate_edge_lengths())

    def test_perimeter_vertical_run_exceeding_limit_is_rejected(self):
        solver = build_solver(width_cells=2, height_cells=10, limit=6)
        solver.grid = [[0, row] for row in range(10)]

        self.assertFalse(solver._validate_edge_lengths())

    def test_perimeter_runs_span_multiple_tiles(self):
        solver = build_solver(width_cells=10, height_cells=3, limit=6)
        solver.grid = [
            list(range(10)),
            [10] * 10,
            [11] * 10,
        ]

        self.assertFalse(solver._validate_edge_lengths())

    def test_interior_runs_span_multiple_tiles(self):
        solver = build_solver(width_cells=3, height_cells=10, limit=6)
        solver.grid = [
            [row, 100, 200] for row in range(10)
        ]

        self.assertFalse(solver._validate_edge_lengths())

    def test_run_equal_to_limit_is_allowed(self):
        solver = build_solver(width_cells=5, height_cells=5, limit=5)
        solver.grid = [
            [0, 0, 0, 1, 1],
            [2, 2, 2, 1, 1],
            [3, 3, 4, 4, 4],
            [5, 5, 4, 4, 4],
            [6, 6, 7, 7, 7],
        ]

        self.assertTrue(solver._validate_edge_lengths())


if __name__ == "__main__":
    unittest.main()
