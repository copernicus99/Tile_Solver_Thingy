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


def test_header_includes_timestamp_and_selection(log_path: Path) -> None:
    selection = {"1x1": 2, "1x2": 1}
    writer = RunLogWriter(log_path, selection)

    content = log_path.read_text(encoding="utf-8")
    assert "Generated at:" in content
    assert "Tile counts:" in content
    assert "- 1x1: 2 (1.0ft × 1.0ft)" in content
    assert "- 1x2: 1 (1.0ft × 2.0ft)" in content
    assert "Total tiles: 3" in content


def test_summary_includes_total_backtracks(log_path: Path) -> None:
    writer = RunLogWriter(log_path)
    attempt_one = PhaseAttempt(
        phase_name="Phase One",
        board_size_ft=(1.0, 1.0),
        board_size_cells=(1, 1),
        elapsed=1.0,
        backtracks=1234,
        success=False,
    )
    attempt_two = PhaseAttempt(
        phase_name="Phase Two",
        board_size_ft=(2.0, 2.0),
        board_size_cells=(2, 2),
        elapsed=2.5,
        backtracks=4321,
        success=True,
    )
    phase_logs = [
        PhaseLog("Phase One", [attempt_one], total_elapsed=1.0, result=None),
        PhaseLog("Phase Two", [attempt_two], total_elapsed=2.5, result=None),
    ]

    writer.append_summary(phase_logs, None)

    content = log_path.read_text(encoding="utf-8")
    assert "Total backtracks performed: 5,555" in content
