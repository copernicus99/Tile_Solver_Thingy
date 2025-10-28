from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .models import Placement, SolveRequest, SolveResult, SolverStats, TileInstance, TileType


@dataclass
class SolverOptions:
    max_edge_cells_horizontal: int
    max_edge_cells_vertical: int
    max_edge_include_perimeter: bool
    same_shape_limit: int
    enforce_plus_rule: bool
    time_limit_sec: Optional[float]


ProgressCallback = Callable[["BacktrackingSolver"], None]


class BacktrackingSolver:
    def __init__(
        self,
        request: SolveRequest,
        options: SolverOptions,
        unit_ft: float,
        phase_name: str,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        self.request = request
        self.options = options
        self.unit_ft = unit_ft
        self.phase_name = phase_name
        self.width = request.board_width_cells
        self.height = request.board_height_cells
        self.allow_pop_outs = request.allow_pop_outs
        self.allow_discards = request.allow_discards
        self.grid: List[List[int]] = [[-1 for _ in range(self.width)] for _ in range(self.height)]
        self.tiles: List[TileInstance] = []
        self.placements: Dict[int, Placement] = {}
        self.stats = SolverStats()
        self._start_time = 0.0
        self._last_progress_report = 0.0
        self._tile_shapes: Dict[int, TileType] = {}
        self._width_cache: Dict[Tuple[frozenset[int], int], bool] = {}
        self._height_cache: Dict[Tuple[frozenset[int], int], bool] = {}
        self._progress_callback = progress_callback
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
        discarded: List[TileInstance] = [
            self.tiles[idx]
            for idx, used in enumerate(self._used)
            if not used
        ]
        if discarded and not self.allow_discards:
            return None
        placements = list(self.placements.values())
        return SolveResult(
            placements=placements,
            board_width_cells=self.width,
            board_height_cells=self.height,
            phase_name=self.phase_name,
            board_width_ft=self.width * self.unit_ft,
            board_height_ft=self.height * self.unit_ft,
            discarded_tiles=discarded,
        )

    def _report_progress(self) -> None:
        if not self._progress_callback:
            return
        now = time.time()
        if now - self._last_progress_report < 0.25:
            return
        self._last_progress_report = now
        self._progress_callback(self)

    def _time_remaining(self) -> bool:
        elapsed = time.time() - self._start_time
        self.stats.elapsed = elapsed
        self._report_progress()
        limit = self.options.time_limit_sec
        if limit is None:
            return True
        return elapsed <= limit

    def _search(self) -> bool:
        if not self._time_remaining():
            return False
        selection = self._select_cell_with_candidates()
        if selection is None:
            if self._validate_completed_layout():
                return True
            self.stats.backtracks += 1
            return False
        x, y, candidates = selection
        if not candidates:
            self.stats.backtracks += 1
            return False
        for tile_idx, width, height in candidates:
            if self._used[tile_idx]:
                continue
            tile = self.tiles[tile_idx]
            placement = Placement(tile, x, y, width, height)
            self._apply(tile_idx, placement)
            if not self._creates_unfillable_gap() and self._search():
                return True
            self._remove(tile_idx, placement)
        self.stats.backtracks += 1
        return False

    def _select_cell_with_candidates(self) -> Optional[Tuple[int, int, List[Tuple[int, int, int]]]]:
        next_cell = self._find_next_empty()
        if next_cell is None:
            return None
        x, y = next_cell
        candidates: List[Tuple[int, int, int]] = []
        seen_types: set[TileType] = set()
        for tile_idx in self._tile_order:
            if self._used[tile_idx]:
                continue
            tile = self.tiles[tile_idx]
            if tile.type in seen_types:
                continue
            seen_types.add(tile.type)
            for width, height in self._orientations(tile):
                if self._can_place(tile_idx, x, y, width, height):
                    candidates.append((tile_idx, width, height))
        return x, y, candidates

    def _available_metrics(self) -> Tuple[int, int, int, frozenset[int], frozenset[int]]:
        min_width: Optional[int] = None
        min_height: Optional[int] = None
        min_area: Optional[int] = None
        width_options: set[int] = set()
        height_options: set[int] = set()
        seen_types: set[TileType] = set()
        for tile_idx in self._tile_order:
            if self._used[tile_idx]:
                continue
            tile = self.tiles[tile_idx]
            if tile.type in seen_types:
                continue
            seen_types.add(tile.type)
            for width, height in self._orientations(tile):
                if min_width is None or width < min_width:
                    min_width = width
                if min_height is None or height < min_height:
                    min_height = height
                width_options.add(width)
                height_options.add(height)
                area = width * height
                if min_area is None or area < min_area:
                    min_area = area
        return (
            (min_width or 0),
            (min_height or 0),
            (min_area or 0),
            frozenset(width_options),
            frozenset(height_options),
        )

    def _length_fillable(
        self,
        length: int,
        options: frozenset[int],
        cache: Dict[Tuple[frozenset[int], int], bool],
    ) -> bool:
        if length == 0:
            return True
        if not options:
            return False
        key = (options, length)
        cached = cache.get(key)
        if cached is not None:
            return cached
        reachable = [False] * (length + 1)
        reachable[0] = True
        dims = sorted(options)
        for dim in dims:
            for value in range(dim, length + 1):
                if reachable[value - dim]:
                    reachable[value] = True
        cache[key] = reachable[length]
        return reachable[length]

    def _segment_fillable(
        self,
        run: int,
        min_dim: int,
        options: frozenset[int],
        cache: Dict[Tuple[frozenset[int], int], bool],
    ) -> bool:
        if run == 0:
            return True
        if not options:
            return False
        if min_dim > 0 and run < min_dim:
            return False
        return self._length_fillable(run, options, cache)

    def _creates_unfillable_gap(self) -> bool:
        min_width, min_height, min_area, width_options, height_options = self._available_metrics()
        if width_options:
            for y in range(self.height):
                run = 0
                for x in range(self.width):
                    if self.grid[y][x] == -1:
                        run += 1
                    else:
                        if not self._segment_fillable(run, min_width, width_options, self._width_cache):
                            return True
                        run = 0
                if not self._segment_fillable(run, min_width, width_options, self._width_cache):
                    return True
        if height_options:
            for x in range(self.width):
                run = 0
                for y in range(self.height):
                    if self.grid[y][x] == -1:
                        run += 1
                    else:
                        if not self._segment_fillable(run, min_height, height_options, self._height_cache):
                            return True
                        run = 0
                if not self._segment_fillable(run, min_height, height_options, self._height_cache):
                    return True
        if min_area > 1:
            visited = [[False for _ in range(self.width)] for _ in range(self.height)]
            for y in range(self.height):
                for x in range(self.width):
                    if self.grid[y][x] != -1 or visited[y][x]:
                        continue
                    stack = [(x, y)]
                    visited[y][x] = True
                    area = 0
                    while stack:
                        cx, cy = stack.pop()
                        area += 1
                        for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                            if 0 <= nx < self.width and 0 <= ny < self.height:
                                if not visited[ny][nx] and self.grid[ny][nx] == -1:
                                    visited[ny][nx] = True
                                    stack.append((nx, ny))
                    if area < min_area:
                        return True
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
        if not self.allow_discards and not all(self._used):
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
        include_perimeter = self.options.max_edge_include_perimeter
        max_horizontal = self.options.max_edge_cells_horizontal
        max_vertical = self.options.max_edge_cells_vertical
        if max_horizontal > 0:
            for y in range(self.height + 1):
                run = 0
                for x in range(self.width):
                    upper = self.grid[y - 1][x] if y > 0 else -2
                    lower = self.grid[y][x] if y < self.height else -2
                    if upper == lower:
                        run = 0
                        continue
                    if not include_perimeter and (upper == -2 or lower == -2):
                        run = 0
                        continue
                    run += 1
                    if run > max_horizontal:
                        return False
        if max_vertical > 0:
            for x in range(self.width + 1):
                run = 0
                for y in range(self.height):
                    left = self.grid[y][x - 1] if x > 0 else -2
                    right = self.grid[y][x] if x < self.width else -2
                    if left == right:
                        run = 0
                        continue
                    if not include_perimeter and (left == -2 or right == -2):
                        run = 0
                        continue
                    run += 1
                    if run > max_vertical:
                        return False
        return True

