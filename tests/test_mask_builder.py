import random
import unittest

from solver.mask_builder import generate_mask


class MaskBuilderTests(unittest.TestCase):
    def test_generate_mask_allows_single_corner_cell(self) -> None:
        rng = random.Random(1234)
        width = height = 5
        total_cells = width * height
        target_cells = total_cells - 1

        mask = generate_mask(
            width,
            height,
            target_cells,
            max_depth=2,
            min_span_cells=0,
            max_horizontal_edge=width,
            max_vertical_edge=height,
            rng=rng,
            attempts=20,
        )

        self.assertIsNotNone(mask, "Mask generation should succeed for a single corner cell")

        available = sum(1 for row in mask for cell in row if cell)  # type: ignore[arg-type]
        self.assertEqual(target_cells, available, "Mask should expose exactly the requested cells")

        removed_cells = {
            (r, c)
            for r, row in enumerate(mask)  # type: ignore[arg-type]
            for c, cell in enumerate(row)
            if not cell
        }
        corners = {(0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)}
        self.assertTrue(
            removed_cells.issubset(corners),
            "Single-cell removal should occur at a board corner",
        )

    def test_generate_mask_handles_odd_slack(self) -> None:
        rng = random.Random(4321)
        width = height = 22
        total_cells = width * height
        slack = 33
        target_cells = total_cells - slack

        mask = generate_mask(
            width,
            height,
            target_cells,
            max_depth=2,
            min_span_cells=4,
            max_horizontal_edge=width,
            max_vertical_edge=height,
            rng=rng,
            attempts=40,
        )

        self.assertIsNotNone(mask, "Mask generation should succeed even with odd slack")

        available = sum(1 for row in mask for cell in row if cell)  # type: ignore[arg-type]
        self.assertEqual(target_cells, available)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
