from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .models import Placement, SolveRequest, SolveResult, SolverStats, TileInstance, TileType


@dataclass
class SolverOptions:
    max_edge_cells: int
    same_shape_limit: int
    enforce_plus_rule: bool
    time_limit_sec: Optional[float]


class BacktrackingSolver:
    def __init__(self, request: SolveRequest, options: SolverOptions, unit_ft: float, phase_name: str) -> None:
        self.request = request
        self.options = options
        self.unit_ft = unit_ft
        self.phase_name = phase_name
        self.width = request.board_width_cells
        self.height = request.board_height_cells
        self.grid: List[List[int]] = [[-1 for _ in range(self.width)] for _ in range(self.height)]
        self.tiles: List[TileInstance] = []
        self.placements: Dict[int, Placement] = {}
        self.stats = SolverStats()
        self._start_time = 0.0
        self._tile_shapes: Dict[int, TileType] = {}
        self._build_tiles()

    def _build_tiles(self) -> None:
        for tile_type, qty in self.request.tile_quantities.items():
            for index in range(qty):
                identifier = f"{tile_type.name}_{index+1}"
                tile = TileInstance(tile_type, identifier, self.request.allow_rotation)
                self._tile_shapes[len(self.tiles)] = tile_type
                self.tiles.append(tile)
        self._tile_order = sorted(range(len(self.tiles)), key=lambda idx: self._tile_shapes[idx].area_ft2, reverse=True)
        self._used = [False] * len(self.tiles)

    def solve(self) -> Optional[SolveResult]:
        if not self.tiles:
            return None
        self._start_time = time.time()
        self.stats.boards_attempted = 1
        success = self._search()
        if not success:
            return None
        placements = list(self.placements.values())
        discarded: List[TileInstance] = []
        return SolveResult(
            placements=placements,
            board_width_cells=self.width,
            board_height_cells=self.height,
            phase_name=self.phase_name,
            board_width_ft=self.width * self.unit_ft,
            board_height_ft=self.height * self.unit_ft,
            discarded_tiles=discarded,
        )

    def _time_remaining(self) -> bool:
        elapsed = time.time() - self._start_time
        self.stats.elapsed = elapsed
        limit = self.options.time_limit_sec
        if limit is None:
            return True
        return elapsed <= limit

    def _search(self) -> bool:
        if not self._time_remaining():
            return False
        next_cell = self._find_next_empty()
        if next_cell is None:
            if self._validate_completed_layout():
                return True
            self.stats.backtracks += 1
            return False
        x, y = next_cell
        for tile_idx in self._tile_order:
            if self._used[tile_idx]:
                continue
            tile = self.tiles[tile_idx]
            for width, height in self._orientations(tile):
                if self._can_place(tile_idx, x, y, width, height):
                    placement = Placement(tile, x, y, width, height)
                    self._apply(tile_idx, placement)
                    if self._search():
                        return True
                    self._remove(tile_idx, placement)
        self.stats.backtracks += 1
        return False

    def _find_next_empty(self) -> Optional[Tuple[int, int]]:
        for y in range(self.height):
            row = self.grid[y]
            for x in range(self.width):
                if row[x] == -1:
                    return x, y
        return None

    def _orientations(self, tile: TileInstance) -> Iterable[Tuple[int, int]]:
        w_cells, h_cells = tile.type.as_cells(self.unit_ft)
        orientations = [(w_cells, h_cells)]
        if tile.allow_rotation and w_cells != h_cells:
            orientations.append((h_cells, w_cells))
        return orientations

    def _can_place(self, tile_idx: int, x: int, y: int, width: int, height: int) -> bool:
        if x + width > self.width or y + height > self.height:
            return False
        # Check occupancy
        for dy in range(height):
            row = self.grid[y + dy]
            for dx in range(width):
                if row[x + dx] != -1:
                    return False
        # Tentatively fill to run constraint checks
        self._fill_cells(tile_idx, x, y, width, height)
        try:
            if not self._check_plus_rule(x, y, width, height):
                return False
            if not self._check_same_shape_limits(tile_idx, x, y, width, height):
                return False
        finally:
            self._clear_cells(tile_idx, x, y, width, height)
        return True

    def _apply(self, tile_idx: int, placement: Placement) -> None:
        self._fill_cells(tile_idx, placement.x, placement.y, placement.width, placement.height)
        self._used[tile_idx] = True
        self.placements[tile_idx] = placement

    def _remove(self, tile_idx: int, placement: Placement) -> None:
        self._clear_cells(tile_idx, placement.x, placement.y, placement.width, placement.height)
        self._used[tile_idx] = False
        if tile_idx in self.placements:
            del self.placements[tile_idx]

    def _fill_cells(self, tile_idx: int, x: int, y: int, width: int, height: int) -> None:
        for dy in range(height):
            row = self.grid[y + dy]
            for dx in range(width):
                row[x + dx] = tile_idx

    def _clear_cells(self, tile_idx: int, x: int, y: int, width: int, height: int) -> None:
        for dy in range(height):
            row = self.grid[y + dy]
            for dx in range(width):
                if row[x + dx] == tile_idx:
                    row[x + dx] = -1

    def _check_plus_rule(self, x: int, y: int, width: int, height: int) -> bool:
        if not self.options.enforce_plus_rule:
            return True
        for iy in range(y - 1, y + height):
            if iy < 0 or iy + 1 >= self.height:
                continue
            for ix in range(x - 1, x + width):
                if ix < 0 or ix + 1 >= self.width:
                    continue
                cells = {
                    self.grid[iy][ix],
                    self.grid[iy][ix + 1],
                    self.grid[iy + 1][ix],
                    self.grid[iy + 1][ix + 1],
                }
                if -1 in cells:
                    continue
                if len(cells) == 4:
                    return False
        return True

    def _check_same_shape_limits(self, tile_idx: int, x: int, y: int, width: int, height: int) -> bool:
        def neighbor_shapes(for_tile: int) -> Dict[str, set[int]]:
            placement = self.placements.get(for_tile)
            if placement is None and for_tile != tile_idx:
                return {}
            if placement is None:
                placement = Placement(self.tiles[for_tile], x, y, width, height)
            neighbors: Dict[str, set[int]] = {}
            for nx, ny in self._edge_neighbors(placement):
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    neighbor_tile = self.grid[ny][nx]
                    if neighbor_tile == -1 or neighbor_tile == for_tile:
                        continue
                    shape = self.tiles[neighbor_tile].type.name
                    neighbors.setdefault(shape, set()).add(neighbor_tile)
            return neighbors

        limit = self.options.same_shape_limit
        if limit <= 0:
            return True
        # Check the tile being placed
        counts = neighbor_shapes(tile_idx)
        if any(len(ids) > limit for ids in counts.values()):
            return False
        # Check affected existing neighbors
        for nx in range(x, x + width):
            for ny in range(y, y + height):
                for adj in ((nx - 1, ny), (nx + 1, ny), (nx, ny - 1), (nx, ny + 1)):
                    ax, ay = adj
                    if 0 <= ax < self.width and 0 <= ay < self.height:
                        neighbor_tile = self.grid[ay][ax]
                        if neighbor_tile != -1 and neighbor_tile != tile_idx:
                            counts = neighbor_shapes(neighbor_tile)
                            if any(len(ids) > limit for ids in counts.values()):
                                return False
        return True

    def _edge_neighbors(self, placement: Placement) -> Iterable[Tuple[int, int]]:
        x, y, width, height = placement.x, placement.y, placement.width, placement.height
        for dx in range(width):
            yield x + dx, y - 1
            yield x + dx, y + height
        for dy in range(height):
            yield x - 1, y + dy
            yield x + width, y + dy

    def _validate_completed_layout(self) -> bool:
        if any(-1 in row for row in self.grid):
            return False
        if not self._validate_edge_lengths():
            return False
        if self.options.enforce_plus_rule and not self._check_plus_rule(0, 0, self.width, self.height):
            return False
        if not self._validate_same_shape_on_completion():
            return False
        return True

    def _validate_same_shape_on_completion(self) -> bool:
        limit = self.options.same_shape_limit
        if limit <= 0:
            return True
        for tile_idx, placement in self.placements.items():
            neighbors: Dict[str, set[int]] = {}
            for nx, ny in self._edge_neighbors(placement):
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    neighbor_tile = self.grid[ny][nx]
                    if neighbor_tile == -1 or neighbor_tile == tile_idx:
                        continue
                    shape = self.tiles[neighbor_tile].type.name
                    neighbors.setdefault(shape, set()).add(neighbor_tile)
            if any(len(ids) > limit for ids in neighbors.values()):
                return False
        return True

    def _validate_edge_lengths(self) -> bool:
        max_length = self.options.max_edge_cells
        if max_length <= 0:
            return True
        # Horizontal boundaries (between rows)
        for y in range(self.height + 1):
            run = 0
            current_boundary = None
            for x in range(self.width):
                upper = self.grid[y - 1][x] if y > 0 else -2
                lower = self.grid[y][x] if y < self.height else -2
                if upper == lower:
                    run = 0
                    current_boundary = None
                    continue
                boundary_key = (min(upper, lower), max(upper, lower), 'h')
                if boundary_key == current_boundary:
                    run += 1
                else:
                    run = 1
                    current_boundary = boundary_key
                if run > max_length:
                    return False
            run = 0
        # Vertical boundaries (between columns)
        for x in range(self.width + 1):
            run = 0
            current_boundary = None
            for y in range(self.height):
                left = self.grid[y][x - 1] if x > 0 else -2
                right = self.grid[y][x] if x < self.width else -2
                if left == right:
                    run = 0
                    current_boundary = None
                    continue
                boundary_key = (min(left, right), max(left, right), 'v')
                if boundary_key == current_boundary:
                    run += 1
                else:
                    run = 1
                    current_boundary = boundary_key
                if run > max_length:
                    return False
            run = 0
        return True

