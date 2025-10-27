from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class PhaseConfig:
    name: str
    allow_rotation: bool
    allow_discards: bool
    time_limit_sec: Optional[float]


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
    MAX_EDGE_FT = 6.0
    PLUS_TOGGLE = True
    SAME_SHAPE_LIMIT = 1

    PHASE_A = PhaseConfig("Phase A", allow_rotation=False, allow_discards=False, time_limit_sec=600.0)
    PHASE_B = PhaseConfig("Phase B", allow_rotation=True, allow_discards=False, time_limit_sec=600.0)
    PHASE_C = PhaseConfig("Phase C", allow_rotation=False, allow_discards=False, time_limit_sec=600.0)
    PHASE_D = PhaseConfig("Phase D", allow_rotation=True, allow_discards=False, time_limit_sec=600.0)

    OUTPUT_DIR = Path("outputs")
    LOG_DIR = Path("logs")


SETTINGS = Settings()
