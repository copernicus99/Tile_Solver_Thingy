from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class PhaseConfig:
    """Configuration values that control solver behaviour for a phase.

    Attributes:
        first_board_time_share: Fraction of the phase's time limit allotted to
            the first board attempt. Expected range: 0.0–1.0.
    """
    name: str
    allow_rotation: bool
    allow_discards: bool
    allow_pop_outs: bool
    time_limit_sec: Optional[float]
    first_board_time_share: float


class Settings:
    TILE_OPTIONS = {
        "1x1": (1.0, 1.0),
        "1x1.5": (1.0, 1.5),
        "1x2": (1.0, 2.0),
        "1.5x1.5": (1.5, 1.5),
        "1.5x2": (1.5, 2.0),
        "1.5x2.5": (1.5, 2.5),
        "1.5x3": (1.5, 3.0),
        "2x2": (2.0, 2.0),
        "2x2.5": (2.0, 2.5),
        "2x3": (2.0, 3.0),
    }

    GRID_UNIT_FT = 0.5

    # Maximum allowable straight edge expressed as a fraction of the board side.
    # With the default GRID_UNIT_FT of 0.5, a 10 ft board (20 cells) and the
    # default ratio of 0.6 results in a 6 ft (12 cell) limit, matching the
    # previous fixed behaviour.
    MAX_EDGE_RATIO = 0.6

    MAX_EDGE_FT = 6.0

    # The 6 ft straight-edge limit applies to both internal seams and the board
    # perimeter.
    MAX_EDGE_INSIDE_ONLY = False

    PLUS_TOGGLE = True

    # A limit of ``0`` disables the adjacency restriction so that phases can
    # freely use duplicate tile shapes when the inventory requires it.
    SAME_SHAPE_LIMIT = 1
    MAX_POP_OUT_DEPTH = 2

    MASK_GENERATION_ATTEMPTS = 10
    MASK_VALIDATION_ATTEMPTS = 2
    MASK_VALIDATION_TIME_LIMIT = 60.0

    ALLOW_RECTANGLES = False
    MAX_RECTANGLE_ASPECT_RATIO = 4.0 / 3.0

    PHASE_A = PhaseConfig(
        "Phase A",
        allow_rotation=True,
        allow_discards=False,
        allow_pop_outs=False,
        time_limit_sec=1200.0,
        first_board_time_share=.2,
    )
    PHASE_B = PhaseConfig(
        "Phase B",
        allow_rotation=True,
        allow_discards=True,
        allow_pop_outs=True,
        time_limit_sec=1800.0,
        first_board_time_share=.2,
    )
    PHASE_C = PhaseConfig(
        "Phase C",
        allow_rotation=True,
        allow_discards=False,
        allow_pop_outs=False,
        time_limit_sec=1200.0,
        first_board_time_share=.2,
    )
    PHASE_D = PhaseConfig(
        "Phase D",
        allow_rotation=True,
        allow_discards=True,
        allow_pop_outs=True,
        time_limit_sec=1800.0,
        first_board_time_share=.2,
    )

    OUTPUT_DIR = Path("outputs")
    LOG_DIR = Path("static")


SETTINGS = Settings()