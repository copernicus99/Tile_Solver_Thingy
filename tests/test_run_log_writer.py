from pathlib import Path

import pytest

from app import RunLogWriter
from solver.models import SolveResult, TileInstance, TileType
from solver.orchestrator import PhaseAttempt, PhaseLog


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    return tmp_path / "run.log"


def _sample_result() -> SolveResult:
    tile_type = TileType("1x1", 1.0, 1.0)
    discarded_tile = TileInstance(tile_type, "1x1_1", allow_rotation=True)
    return SolveResult(
        placements=[],
        board_width_cells=2,
        board_height_cells=2,
        phase_name="Phase Test",
        board_width_ft=1.0,
        board_height_ft=1.0,
        discarded_tiles=[discarded_tile],
    )


def test_append_summary_lists_discarded_tiles(log_path: Path) -> None:
    writer = RunLogWriter(log_path)
    result = _sample_result()
    attempt = PhaseAttempt(
        phase_name=result.phase_name,
        board_size_ft=(result.board_width_ft, result.board_height_ft),
        board_size_cells=(result.board_width_cells, result.board_height_cells),
        elapsed=1.0,
        backtracks=0,
        success=True,
    )
    phase_log = PhaseLog(result.phase_name, [attempt], total_elapsed=1.0, result=result)

    writer.append_summary([phase_log], result)

    content = log_path.read_text(encoding="utf-8")
    assert "Discarded tiles:" in content
    assert "- 1x1_1 (1.0ft × 1.0ft)" in content
