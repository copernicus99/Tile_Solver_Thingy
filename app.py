from __future__ import annotations

from typing import Dict, List

from flask import Flask, render_template, request, send_from_directory

from config import SETTINGS
from solver.models import SolveResult
from solver.orchestrator import PhaseLog, TileSolverOrchestrator

app = Flask(__name__)
app.secret_key = "tile-solver-secret"

orchestrator = TileSolverOrchestrator()


@app.route("/")
def index():
    return render_template(
        "tile_selection_form.html",
        tile_options=SETTINGS.TILE_OPTIONS,
        max_quantity=10,
    )


@app.route("/solve", methods=["POST"])
def solve_tiles():
    selection: Dict[str, int] = {}
    for name in SETTINGS.TILE_OPTIONS:
        value = request.form.get(name)
        try:
            count = int(value)
        except (TypeError, ValueError):
            count = 0
        selection[name] = max(0, min(10, count))
    try:
        result, logs = orchestrator.solve(selection)
    except ValueError as exc:
        return render_template(
            "results_form.html",
            result=None,
            logs=[],
            error=str(exc),
            outputs={},
            config=SETTINGS,
        )
    outputs = _write_outputs(result, logs) if result else {}
    return render_template(
        "results_form.html",
        result=result,
        logs=logs,
        outputs=outputs,
        error=None if result else "No solution within the provided timeframes",
        config=SETTINGS,
    )


@app.route("/outputs/<path:filename>")
def serve_output(filename: str):
    return send_from_directory(SETTINGS.OUTPUT_DIR, filename)


@app.route("/logs/<path:filename>")
def serve_log(filename: str):
    return send_from_directory(SETTINGS.LOG_DIR, filename, as_attachment=True)


def _write_outputs(result: SolveResult | None, logs: List[PhaseLog]):
    SETTINGS.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS.LOG_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {}
    if result:
        coords_path = SETTINGS.OUTPUT_DIR / "coords.txt"
        layout_path = SETTINGS.OUTPUT_DIR / "layout.html"
        _write_coords(coords_path, result)
        _write_layout(layout_path, result)
        outputs["coords"] = coords_path.name
        outputs["layout"] = layout_path.name
    log_path = SETTINGS.LOG_DIR / "run_log.txt"
    _write_log(log_path, logs)
    outputs["run_log"] = log_path.name
    return outputs


def _write_coords(path: Path, result: SolveResult) -> None:
    lines = ["TileID,Type,TopLeftX(ft),TopLeftY(ft),Width(ft),Height(ft)"]
    for placement in sorted(result.placements, key=lambda p: p.tile.identifier):
        top_x, top_y = placement.to_top_left_ft(SETTINGS.GRID_UNIT_FT)
        width_ft = placement.width * SETTINGS.GRID_UNIT_FT
        height_ft = placement.height * SETTINGS.GRID_UNIT_FT
        lines.append(
            f"{placement.tile.identifier},{placement.tile.type.name},{top_x:.2f},{top_y:.2f},{width_ft:.2f},{height_ft:.2f}"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_layout(path: Path, result: SolveResult) -> None:
    scale = 40
    board_width_px = result.board_width_cells * scale
    board_height_px = result.board_height_cells * scale
    tile_divs = []
    palette = [
        "#1abc9c",
        "#3498db",
        "#9b59b6",
        "#f1c40f",
        "#e67e22",
        "#e74c3c",
        "#2ecc71",
        "#34495e",
        "#16a085",
        "#8e44ad",
    ]
    for idx, placement in enumerate(result.placements):
        color = palette[idx % len(palette)]
        left = placement.x * scale
        top = placement.y * scale
        width = placement.width * scale
        height = placement.height * scale
        tile_divs.append(
            f'<div class="tile" style="left:{left}px;top:{top}px;width:{width}px;height:{height}px;background-color:{color}33;border-color:{color};">'
            f"<span>{placement.tile.identifier}</span></div>"
        )
    html = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<title>Tile Layout</title>
<style>
body {{ background: #0b0e11; color: #f5f7fa; font-family: 'Segoe UI', sans-serif; }}
.board {{ position: relative; width: {board_width_px}px; height: {board_height_px}px; margin: 2rem auto; border: 2px solid #f5f7fa; box-shadow: 0 0 20px rgba(0,0,0,0.6); }}
.tile {{ position: absolute; border: 1px solid; box-sizing: border-box; display:flex; align-items:center; justify-content:center; color:#f5f7fa; font-size:0.85rem; letter-spacing:0.02em; }}
</style>
</head>
<body>
<h1 style=\"text-align:center;\">Layout for {result.phase_name}</h1>
<div class=\"board\">
{''.join(tile_divs)}
</div>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def _write_log(path: Path, logs: List[PhaseLog]) -> None:
    lines: List[str] = []
    lines.append("TILE SOLVER RUN LOG")
    lines.append(f"MAX_EDGE_FT={SETTINGS.MAX_EDGE_FT}")
    lines.append(f"PLUS_TOGGLE={SETTINGS.PLUS_TOGGLE}")
    lines.append(f"SAME_SHAPE_LIMIT={SETTINGS.SAME_SHAPE_LIMIT}")
    lines.append("")
    for phase_log in logs:
        lines.append(f"{phase_log.phase_name}")
        lines.append(f"  Total elapsed: {phase_log.total_elapsed:.2f}s")
        if not phase_log.attempts:
            lines.append("  No board attempts")
        for attempt in phase_log.attempts:
            width_ft, height_ft = attempt.board_size_ft
            lines.append(
                f"  Board {width_ft:.2f}ft x {height_ft:.2f}ft | elapsed={attempt.elapsed:.2f}s | backtracks={attempt.backtracks} | success={'yes' if attempt.success else 'no'}"
            )
        if phase_log.result:
            lines.append("  Result achieved")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    app.run(debug=True)
