from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

Mask = Tuple[Tuple[bool, ...], ...]


@dataclass
class MaskBuilderConfig:
    width: int
    height: int
    target_cells: int
    max_depth: int
    min_span_cells: int
    max_horizontal_edge: int
    max_vertical_edge: int
    max_attempts: int = 12
    max_pair_failures: int = 24


class MaskBuilder:
    def __init__(self, config: MaskBuilderConfig, rng: Optional[random.Random] = None) -> None:
        self.config = config
        self.rng = rng or random.Random()
        self.width = config.width
        self.height = config.height
        self.slack = self.width * self.height - config.target_cells
        self.max_depth = max(config.max_depth, 0)
        self.min_span = max(config.min_span_cells, 1)
        self.max_horizontal_edge = max(config.max_horizontal_edge, 0)
        self.max_vertical_edge = max(config.max_vertical_edge, 0)

    def generate(self) -> Optional[Mask]:
        if self.slack <= 0:
            return None
        if self.max_depth <= 0:
            return None
        for _ in range(self.config.max_attempts):
            mask = [[True for _ in range(self.width)] for _ in range(self.height)]
            slack_remaining = self.slack
            orientations = ["horizontal", "vertical"]
            self.rng.shuffle(orientations)
            failures = 0
            while slack_remaining > 0 and failures < self.config.max_pair_failures:
                orientation = orientations[0]
                removed = (
                    self._place_horizontal_pair(mask, slack_remaining)
                    if orientation == "horizontal"
                    else self._place_vertical_pair(mask, slack_remaining)
                )
                if removed == 0:
                    failures += 1
                    orientations.append(orientations.pop(0))
                    continue
                slack_remaining -= removed
                failures = 0
            if slack_remaining == 0:
                return tuple(tuple(row) for row in mask)
        return None

    def _place_horizontal_pair(self, mask: List[List[bool]], slack_remaining: int) -> int:
        if self.width <= 0 or self.height <= 0:
            return 0
        max_depth = min(self.max_depth, self.height // 2)
        if max_depth <= 0:
            return 0
        min_length = min(max(1, self.min_span // 2), self.width)
        if slack_remaining < 2 * min_length:
            min_length = max(1, slack_remaining // 2)
        if min_length <= 0:
            return 0
        length_options = list(range(min_length, self.width + 1))
        self.rng.shuffle(length_options)
        for length in length_options:
            max_depth_for_slack = min(max_depth, slack_remaining // (2 * length))
            if max_depth_for_slack <= 0:
                continue
            depth_choices = list(range(1, max_depth_for_slack + 1))
            depth_choices.sort(reverse=True)
            for depth in depth_choices:
                start_positions = list(range(0, self.width - length + 1))
                self.rng.shuffle(start_positions)
                for start in start_positions:
                    cells = self._collect_horizontal_cells(start, length, depth)
                    if not self._cells_clear(mask, cells):
                        continue
                    self._set_cells(mask, cells, False)
                    if self._respects_edge_limits(mask):
                        return len(cells)
                    self._set_cells(mask, cells, True)
        return 0

    def _place_vertical_pair(self, mask: List[List[bool]], slack_remaining: int) -> int:
        if self.width <= 0 or self.height <= 0:
            return 0
        max_depth = min(self.max_depth, self.width // 2)
        if max_depth <= 0:
            return 0
        min_length = min(max(1, self.min_span // 2), self.height)
        if slack_remaining < 2 * min_length:
            min_length = max(1, slack_remaining // 2)
        if min_length <= 0:
            return 0
        length_options = list(range(min_length, self.height + 1))
        self.rng.shuffle(length_options)
        for length in length_options:
            max_depth_for_slack = min(max_depth, slack_remaining // (2 * length))
            if max_depth_for_slack <= 0:
                continue
            depth_choices = list(range(1, max_depth_for_slack + 1))
            depth_choices.sort(reverse=True)
            for depth in depth_choices:
                start_positions = list(range(0, self.height - length + 1))
                self.rng.shuffle(start_positions)
                for start in start_positions:
                    cells = self._collect_vertical_cells(start, length, depth)
                    if not self._cells_clear(mask, cells):
                        continue
                    self._set_cells(mask, cells, False)
                    if self._respects_edge_limits(mask):
                        return len(cells)
                    self._set_cells(mask, cells, True)
        return 0

    def _collect_horizontal_cells(
        self, start: int, length: int, depth: int
    ) -> List[Tuple[int, int]]:
        cells: List[Tuple[int, int]] = []
        for offset in range(depth):
            top_row = offset
            bottom_row = self.height - 1 - offset
            for col in range(start, start + length):
                cells.append((top_row, col))
                if bottom_row != top_row:
                    cells.append((bottom_row, col))
        return cells

    def _collect_vertical_cells(
        self, start: int, length: int, depth: int
    ) -> List[Tuple[int, int]]:
        cells: List[Tuple[int, int]] = []
        for offset in range(depth):
            left_col = offset
            right_col = self.width - 1 - offset
            for row in range(start, start + length):
                cells.append((row, left_col))
                if right_col != left_col:
                    cells.append((row, right_col))
        return cells

    @staticmethod
    def _cells_clear(mask: List[List[bool]], cells: Sequence[Tuple[int, int]]) -> bool:
        for row, col in cells:
            if not mask[row][col]:
                return False
        return True

    @staticmethod
    def _set_cells(mask: List[List[bool]], cells: Sequence[Tuple[int, int]], value: bool) -> None:
        for row, col in cells:
            mask[row][col] = value

    def _respects_edge_limits(self, mask: List[List[bool]]) -> bool:
        if self.max_horizontal_edge:
            if self._max_row_run(mask[0]) > self.max_horizontal_edge:
                return False
            if self._max_row_run(mask[-1]) > self.max_horizontal_edge:
                return False
        if self.max_vertical_edge:
            if self._max_column_run(mask, 0) > self.max_vertical_edge:
                return False
            if self._max_column_run(mask, -1) > self.max_vertical_edge:
                return False
        return True

    @staticmethod
    def _max_row_run(row: Sequence[bool]) -> int:
        run = 0
        best = 0
        for cell in row:
            if cell:
                run += 1
                best = max(best, run)
            else:
                run = 0
        return best

    @staticmethod
    def _max_column_run(mask: List[List[bool]], index: int) -> int:
        run = 0
        best = 0
        for row in mask:
            cell = row[index]
            if cell:
                run += 1
                best = max(best, run)
            else:
                run = 0
        return best


def generate_mask(
    width: int,
    height: int,
    target_cells: int,
    max_depth: int,
    min_span_cells: int,
    max_horizontal_edge: int,
    max_vertical_edge: int,
    *,
    rng: Optional[random.Random] = None,
    attempts: int = 12,
) -> Optional[Mask]:
    config = MaskBuilderConfig(
        width=width,
        height=height,
        target_cells=target_cells,
        max_depth=max_depth,
        min_span_cells=min_span_cells,
        max_horizontal_edge=max_horizontal_edge,
        max_vertical_edge=max_vertical_edge,
        max_attempts=attempts,
    )
    builder = MaskBuilder(config, rng=rng)
    return builder.generate()
