from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional

from config import SETTINGS


@dataclass
class PredictionFactor:
    """Breakdown component that contributes to the final probability."""

    key: str
    label: str
    score: float
    weight: float
    contribution: float
    description: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "key": self.key,
            "label": self.label,
            "score": round(self.score, 4),
            "weight": round(self.weight, 2),
            "contribution": round(self.contribution, 2),
            "description": self.description,
        }


@dataclass
class PredictionResult:
    probability: float
    tolerance: float
    confidence: str
    summary: str
    factors: List[PredictionFactor]
    effective_config: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return {
            "probability": round(self.probability, 2),
            "tolerance": round(self.tolerance, 2),
            "confidence": self.confidence,
            "summary": self.summary,
            "factors": [factor.to_dict() for factor in self.factors],
            "effective_config": self.effective_config,
        }


def estimate_solve_probability(
    selection: Mapping[str, int],
    overrides: Optional[Mapping[str, object]] = None,
) -> PredictionResult:
    """Return a heuristic probability that the solver will find a solution."""

    effective_config = _build_effective_config(overrides or {})
    total_tiles = sum(max(0, int(selection.get(name, 0))) for name in SETTINGS.TILE_OPTIONS)

    factors: List[PredictionFactor] = []
    base_probability = 2.0
    if total_tiles <= 0:
        factors.append(
            PredictionFactor(
                key="inventory",
                label="Tile inventory",
                score=0.0,
                weight=25.0,
                contribution=0.0,
                description="No tiles selected. The solver requires at least one tile to attempt a layout.",
            )
        )
        probability = 0.0
    else:
        layout_contribution = _score_layout(selection, factors)
        config_contribution = _score_config(effective_config, factors)
        probability = base_probability + layout_contribution + config_contribution

    probability = max(0.0, min(100.0, probability))
    tolerance = _estimate_tolerance(probability, factors)
    confidence = _confidence_label(probability, tolerance)
    summary = _build_summary(probability, tolerance)

    return PredictionResult(
        probability=probability,
        tolerance=tolerance,
        confidence=confidence,
        summary=summary,
        factors=factors,
        effective_config=effective_config,
    )


def _score_layout(selection: Mapping[str, int], factors: List[PredictionFactor]) -> float:
    grid_unit = SETTINGS.GRID_UNIT_FT
    unit_area = grid_unit * grid_unit if grid_unit > 0 else 1.0

    total_area = 0.0
    total_tiles = 0
    distinct_tiles = 0
    tile_areas: List[float] = []
    for tile_name, dims in SETTINGS.TILE_OPTIONS.items():
        count = max(0, int(selection.get(tile_name, 0)))
        if count <= 0:
            continue
        distinct_tiles += 1
        total_tiles += count
        area = float(dims[0]) * float(dims[1])
        tile_areas.append(area)
        total_area += area * count

    if total_tiles <= 0 or total_area <= 0:
        return 0.0

    cells_exact = total_area / unit_area
    rounded_cells = round(cells_exact)
    alignment_error = abs(cells_exact - rounded_cells)
    alignment_score = max(0.0, 1.0 - min(1.0, alignment_error * 4.0))

    square_cells = max(1, rounded_cells)
    root_exact = math.sqrt(square_cells)
    square_error = abs(root_exact - round(root_exact))
    square_score = max(0.0, 1.0 - min(1.0, square_error * 2.0))

    variety_score = min(1.0, distinct_tiles / 4.0)

    average_stack = total_tiles / max(1, distinct_tiles)
    inventory_score = min(1.0, average_stack / 3.0)

    if tile_areas:
        max_area = max(tile_areas)
        min_area = min(tile_areas)
        if max_area <= 0:
            balance_score = 0.0
        else:
            balance_score = max(0.0, min(1.0, min_area / max_area))
    else:
        balance_score = 0.0

    contributions = [
        ("alignment", "Grid alignment", alignment_score, 22.0, "How closely the total area aligns to the solver grid."),
        ("square", "Square potential", square_score, 14.0, "Preference for layouts that can form near-square boards."),
        ("variety", "Tile variety", variety_score, 12.0, "Benefit of having multiple tile shapes to explore."),
        ("inventory", "Inventory depth", inventory_score, 10.0, "Availability of duplicates for backtracking."),
        ("balance", "Shape balance", balance_score, 8.0, "How similar the tile areas are to each other."),
    ]

    total = 0.0
    for key, label, score, weight, description in contributions:
        contribution = score * weight
        factors.append(
            PredictionFactor(
                key=f"layout_{key}",
                label=label,
                score=score,
                weight=weight,
                contribution=contribution,
                description=description,
            )
        )
        total += contribution
    return total


def _score_config(config: Mapping[str, object], factors: List[PredictionFactor]) -> float:
    total = 0.0

    def add_factor(key: str, label: str, score: float, weight: float, description: str) -> None:
        nonlocal total
        contribution = score * weight
        factors.append(
            PredictionFactor(
                key=f"config_{key}",
                label=label,
                score=score,
                weight=weight,
                contribution=contribution,
                description=description,
            )
        )
        total += contribution

    edge_ratio = _normalize(config.get("max_edge_ratio"), 0.3, 0.9)
    add_factor(
        "max_edge_ratio",
        "Max edge ratio",
        edge_ratio,
        6.0,
        "Longer straight edges increase the solver's freedom when placing seams.",
    )

    plus_toggle = 1.0 if _truthy(config.get("plus_toggle")) else 0.0
    add_factor(
        "plus_toggle",
        "Plus intersections",
        plus_toggle,
        4.5,
        "Allowing plus-shaped intersections removes a major restriction.",
    )

    same_shape = _normalize(config.get("same_shape_limit"), 0.0, 4.0)
    add_factor(
        "same_shape_limit",
        "Same side limit",
        same_shape,
        4.5,
        "Higher limits let the solver reuse shapes on the same edge.",
    )

    pop_out_depth = _normalize(config.get("max_pop_out_depth"), 0.0, 4.0)
    add_factor(
        "max_pop_out_depth",
        "Pop-out depth",
        pop_out_depth,
        5.0,
        "Deeper pop-outs expand the search space when boards get stuck.",
    )

    mask_time = _normalize(config.get("mask_validation_time_limit"), 15.0, 240.0)
    add_factor(
        "mask_validation_time_limit",
        "Mask validation time",
        mask_time,
        3.5,
        "Longer mask validation windows catch more viable boards.",
    )

    rectangle_ratio = _normalize(config.get("max_rectangle_aspect_ratio"), 1.0, 2.0)
    add_factor(
        "max_rectangle_aspect_ratio",
        "Rectangle aspect",
        rectangle_ratio,
        3.5,
        "Higher ratios unlock more rectangular board options.",
    )

    edge_ft = _normalize(config.get("max_edge_ft"), 3.0, 12.0)
    add_factor(
        "max_edge_ft",
        "Straight-edge length",
        edge_ft,
        3.0,
        "Allowing longer straight seams reduces artificial cuts.",
    )

    grid_unit = config.get("grid_unit_ft")
    grid_score = 1.0 - _normalize(grid_unit, 0.25, 1.0)
    add_factor(
        "grid_unit_ft",
        "Grid resolution",
        max(0.0, grid_score),
        3.0,
        "Smaller grid units increase placement granularity.",
    )

    rectangles_allowed = 1.0 if _truthy(config.get("allow_rectangles")) else 0.2
    add_factor(
        "allow_rectangles",
        "Rectangular boards",
        rectangles_allowed,
        3.0,
        "Allowing rectangular boards broadens the solver's search space.",
    )

    inside_only = 1.0 if _truthy(config.get("max_edge_inside_only")) else 0.6
    add_factor(
        "max_edge_inside_only",
        "Perimeter edge rule",
        inside_only,
        2.5,
        "Applying the straight-edge limit only inside the board preserves perimeter flexibility.",
    )

    mask_attempts = _normalize(config.get("mask_generation_attempts"), 1.0, 20.0)
    add_factor(
        "mask_generation_attempts",
        "Mask attempts",
        mask_attempts,
        3.0,
        "More attempts at generating masks improve coverage for complex boards.",
    )

    validation_attempts = _normalize(config.get("mask_validation_attempts"), 1.0, 6.0)
    add_factor(
        "mask_validation_attempts",
        "Mask validation attempts",
        validation_attempts,
        2.5,
        "Repeated validation cycles catch edge cases that a single pass might miss.",
    )

    phase_score = _score_phase_configs(config.get("phases"))
    add_factor(
        "phase_windows",
        "Phase time windows",
        phase_score,
        5.5,
        "Generous per-phase time limits increase the odds of exploring difficult boards.",
    )

    return total


def _score_phase_configs(phases: Optional[Iterable[Mapping[str, object]]]) -> float:
    if not phases:
        return 0.0

    phase_list = list(phases)
    if not phase_list:
        return 0.0

    time_total = 0.0
    rotation_count = 0
    pop_out_count = 0
    discard_count = 0
    for phase in phase_list:
        try:
            time_limit = float(phase.get("time_limit_sec"))
        except (TypeError, ValueError):
            time_limit = 0.0
        time_total += max(0.0, time_limit)
        if _truthy(phase.get("allow_rotation")):
            rotation_count += 1
        if _truthy(phase.get("allow_pop_outs")):
            pop_out_count += 1
        if _truthy(phase.get("allow_discards")):
            discard_count += 1

    phase_count = max(1, len(phase_list))
    time_score = min(1.0, time_total / 3600.0)
    rotation_score = rotation_count / phase_count
    pop_out_score = pop_out_count / phase_count
    discard_score = discard_count / phase_count

    return min(1.0, (time_score * 0.5) + (rotation_score * 0.2) + (pop_out_score * 0.2) + (discard_score * 0.1))


def _estimate_tolerance(probability: float, factors: List[PredictionFactor]) -> float:
    positive = sum(max(0.0, factor.contribution) for factor in factors)
    negative = sum(-min(0.0, factor.contribution) for factor in factors)
    spread = positive + negative
    if spread <= 0:
        return 18.0
    base = 18.0 - min(10.0, positive / 6.0)
    adjusted = base + min(6.0, negative / 8.0)
    adjusted = max(5.0, min(24.0, adjusted))

    if probability >= 80.0:
        adjusted = min(adjusted, 8.0)
    elif probability >= 60.0:
        adjusted = min(adjusted, 10.0)
    return adjusted


def _confidence_label(probability: float, tolerance: float) -> str:
    if probability >= 75.0 and tolerance <= 8.0:
        return "High"
    if probability >= 45.0 and tolerance <= 12.0:
        return "Moderate"
    return "Low"


def _build_summary(probability: float, tolerance: float) -> str:
    if probability >= 75.0:
        return "Strong chance of a solve with the current settings."
    if probability >= 50.0:
        return "Solver has a reasonable opportunity to find a layout."
    if probability >= 25.0:
        return "Challenging scenario—consider relaxing constraints or adding tiles."
    return "Very low chance of success without additional tiles or configuration changes."


def _build_effective_config(overrides: Mapping[str, object]) -> Dict[str, object]:
    snapshot: Dict[str, object] = {
        "grid_unit_ft": SETTINGS.GRID_UNIT_FT,
        "max_edge_ratio": SETTINGS.MAX_EDGE_RATIO,
        "max_edge_ft": SETTINGS.MAX_EDGE_FT,
        "max_edge_inside_only": SETTINGS.MAX_EDGE_INSIDE_ONLY,
        "plus_toggle": SETTINGS.PLUS_TOGGLE,
        "same_shape_limit": SETTINGS.SAME_SHAPE_LIMIT,
        "max_pop_out_depth": SETTINGS.MAX_POP_OUT_DEPTH,
        "mask_generation_attempts": SETTINGS.MASK_GENERATION_ATTEMPTS,
        "mask_validation_attempts": SETTINGS.MASK_VALIDATION_ATTEMPTS,
        "mask_validation_time_limit": SETTINGS.MASK_VALIDATION_TIME_LIMIT,
        "allow_rectangles": SETTINGS.ALLOW_RECTANGLES,
        "max_rectangle_aspect_ratio": SETTINGS.MAX_RECTANGLE_ASPECT_RATIO,
        "phases": [
            {
                "name": phase.name,
                "allow_rotation": phase.allow_rotation,
                "allow_discards": phase.allow_discards,
                "allow_pop_outs": phase.allow_pop_outs,
                "time_limit_sec": phase.time_limit_sec,
                "first_board_time_share": phase.first_board_time_share,
            }
            for phase in (
                SETTINGS.PHASE_A,
                SETTINGS.PHASE_B,
                SETTINGS.PHASE_C,
                SETTINGS.PHASE_D,
            )
        ],
    }

    for key, value in overrides.items():
        if key not in snapshot:
            continue
        snapshot[key] = _coerce_override_value(key, value, snapshot[key])
    return snapshot


def _coerce_override_value(key: str, value: object, fallback: object) -> object:
    if isinstance(fallback, bool):
        return _truthy(value)
    if isinstance(fallback, (int, float)):
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(fallback)
    return value


def _normalize(value: object, lower: float, upper: float) -> float:
    if value is None:
        return 0.0
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isclose(upper, lower):
        return 0.0
    return max(0.0, min(1.0, (numeric - lower) / (upper - lower)))


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(value, (int, float)):
        return float(value) != 0.0
    return False
