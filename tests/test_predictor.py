import math

from config import SETTINGS
from solver.predictor import estimate_solve_probability


def build_empty_selection():
    return {name: 0 for name in SETTINGS.TILE_OPTIONS}


def build_square_selection():
    selection = build_empty_selection()
    selection["2x2"] = 4
    return selection


def test_prediction_zero_tiles_returns_zero_probability():
    result = estimate_solve_probability(build_empty_selection())
    assert result.probability == 0.0
    assert result.confidence == "Low"
    as_dict = result.to_dict()
    assert "probability" in as_dict and math.isclose(as_dict["probability"], 0.0)
    assert as_dict["effective_config"]["max_edge_ratio"] == SETTINGS.MAX_EDGE_RATIO


def test_prediction_prefers_square_inventory():
    result = estimate_solve_probability(build_square_selection())
    assert result.probability > 60.0
    assert result.tolerance <= 12.0
    assert result.confidence in {"High", "Moderate"}


def test_prediction_penalises_restrictive_overrides():
    base = estimate_solve_probability(build_square_selection())
    restricted = estimate_solve_probability(
        build_square_selection(),
        overrides={
            "max_edge_ratio": 0.3,
            "plus_toggle": False,
            "same_shape_limit": 0,
            "max_pop_out_depth": 0,
            "mask_validation_time_limit": 30,
            "max_rectangle_aspect_ratio": 1.1,
        },
    )
    assert restricted.probability < base.probability
    assert restricted.tolerance >= base.tolerance
