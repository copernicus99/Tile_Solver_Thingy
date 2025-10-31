from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    render_template,
    request,
    send_from_directory,
    stream_with_context,
)

from config import SETTINGS
from solver.models import SolveResult
from solver.orchestrator import PhaseLog, TileSolverOrchestrator

app = Flask(__name__)
app.secret_key = "tile-solver-secret"

orchestrator = TileSolverOrchestrator()
TILE_BAG_FILE = Path("tile_bags.json")

PHASE_CONFIGS = {
    phase.name: phase
    for phase in (
        SETTINGS.PHASE_A,
        SETTINGS.PHASE_B,
        SETTINGS.PHASE_C,
        SETTINGS.PHASE_D,
    )
}


def load_tile_bags() -> List[Dict[str, object]]:
    if not TILE_BAG_FILE.exists():
        return []
    try:
        raw_content = TILE_BAG_FILE.read_text(encoding="utf-8")
    except OSError:
        return []
    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError:
        return []

    if not isinstance(parsed, list):
        return []

    bags: List[Dict[str, object]] = []
    for entry in parsed:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        tiles = entry.get("tiles", {})
        if not isinstance(name, str) or not name.strip():
            continue
        sanitized_tiles: Dict[str, int] = {}
        if isinstance(tiles, dict):
            for tile_name, quantity in tiles.items():
                if not isinstance(tile_name, str):
                    continue
                try:
                    sanitized_tiles[tile_name] = int(quantity)
                except (TypeError, ValueError):
                    continue
        bags.append({"name": name.strip(), "tiles": sanitized_tiles})
    return bags


class RunLogWriter:
    def __init__(self, path: Path, selection: Optional[Dict[str, int]] = None):
        self.path = path
        self._lock = threading.Lock()
        self._summary_written = False
        SETTINGS.LOG_DIR.mkdir(parents=True, exist_ok=True)
        header = self._build_header(selection)
        with self._lock:
            with self.path.open("w", encoding="utf-8") as fh:
                for line in header:
                    fh.write(f"{line}\n")

    def _build_header(self, selection: Optional[Dict[str, int]]) -> List[str]:
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        header: List[str] = [
            "TILE SOLVER RUN LOG",
            f"Generated at: {timestamp}",
            f"MAX_EDGE_FT={SETTINGS.MAX_EDGE_FT}",
            f"PLUS_TOGGLE={SETTINGS.PLUS_TOGGLE}",
            f"SAME_SHAPE_LIMIT={SETTINGS.SAME_SHAPE_LIMIT}",
        ]
        header.extend(self._selection_lines(selection))
        header.extend(["", "Events:"])
        return header

    def _selection_lines(self, selection: Optional[Dict[str, int]]) -> List[str]:
        if not selection:
            return ["Tile counts: none selected"]

        lines: List[str] = ["Tile counts:"]
        total_tiles = 0
        total_area = 0.0
        for tile_name, dims in SETTINGS.TILE_OPTIONS.items():
            count = int(selection.get(tile_name, 0))
            if count <= 0:
                continue
            width_ft, height_ft = dims
            lines.append(
                "  - {name}: {count} ({width:.1f}ft × {height:.1f}ft)".format(
                    name=tile_name,
                    count=count,
                    width=width_ft,
                    height=height_ft,
                )
            )
            total_tiles += count
            total_area += width_ft * height_ft * count

        if total_tiles == 0:
            return ["Tile counts: none selected"]

        lines.append(f"  Total tiles: {total_tiles}")
        lines.append(f"  Total coverage: {total_area:.2f} ft²")
        return lines

    def handle_event(self, event: Dict[str, object]) -> None:
        event_type = event.get("type")
        lines: List[str] = []
        if event_type == "run_started":
            lines.append("Run started.")
            phases = event.get("phases") or []
            for idx, phase in enumerate(phases, start=1):
                limit = phase.get("time_limit_sec")
                limit_text = f"{limit:.2f}s" if isinstance(limit, (int, float)) else "no limit"
                lines.append(
                    "  Phase {idx}: {name} (time limit: {limit}, rotation: {rotation}, discards: {discards}, pop-outs: {pop_outs})".format(
                        idx=idx,
                        name=phase.get("name", "unknown"),
                        limit=limit_text,
                        rotation="yes" if phase.get("allow_rotation") else "no",
                        discards="yes" if phase.get("allow_discards") else "no",
                        pop_outs="yes" if phase.get("allow_pop_outs") else "no",
                    )
                )
            total = event.get("total_allotment")
            if isinstance(total, (int, float)):
                lines.append(f"  Total allotted time: {total:.2f}s")
        elif event_type == "phase_started":
            phase = event.get("phase")
            limit = event.get("time_limit_sec")
            limit_text = f"{limit:.2f}s" if isinstance(limit, (int, float)) else "no limit"
            index = event.get("phase_index")
            if isinstance(index, int):
                lines.append(f"Phase {index + 1} ({phase}) started. Time limit: {limit_text}.")
            else:
                lines.append(f"Phase {phase} started. Time limit: {limit_text}.")
        elif event_type == "attempt_started":
            idx = event.get("attempt_index")
            board = event.get("board_size_ft") or (None, None)
            variant_label = event.get("variant_label") or "Rectangle Fit"
            lines.append(
                "Attempt {idx} started on board {w:.2f}ft x {h:.2f}ft ({variant}).".format(
                    idx=idx,
                    w=(board[0] or 0.0),
                    h=(board[1] or 0.0),
                    variant=variant_label,
                )
            )
            mask_rows = event.get("mask_rows")
            if mask_rows:
                lines.append("  Mask layout (# = available, . = removed):")
                for row in mask_rows:
                    lines.append(f"    {row}")
        elif event_type == "mask_validation":
            board = event.get("board_size_ft") or (None, None)
            status = str(event.get("status") or "unknown").lower()
            mask_count = event.get("mask_count")
            count_value = mask_count if isinstance(mask_count, int) and mask_count > 0 else 0
            reason = event.get("reason")
            prefix = "Mask validation=Passed" if status == "passed" else "Mask validation=Failed"
            mask_text = (
                "no masks"
                if count_value == 0
                else f"{count_value} mask" + ("s" if count_value != 1 else "")
            )
            lines.append(
                "{prefix} for board {w:.2f}ft x {h:.2f}ft ({mask_text}).".format(
                    prefix=prefix,
                    w=(board[0] or 0.0),
                    h=(board[1] or 0.0),
                    mask_text=mask_text,
                )
            )
            if status != "passed" and reason:
                lines.append(f"  Reason: {reason}")
        elif event_type == "attempt_completed":
            idx = event.get("attempt_index")
            elapsed = event.get("elapsed")
            elapsed_text = f"{elapsed:.2f}s" if isinstance(elapsed, (int, float)) else "unknown"
            success = "yes" if event.get("success") else "no"
            backtracks = event.get("backtracks")
            variant_label = event.get("variant_label") or "Rectangle Fit"
            lines.append(
                "Attempt {idx} ({variant}) completed in {elapsed} (success: {success}, backtracks: {backtracks}).".format(
                    idx=idx,
                    variant=variant_label,
                    elapsed=elapsed_text,
                    success=success,
                    backtracks=backtracks if backtracks is not None else "unknown",
                )
            )
            mask_rows = event.get("mask_rows")
            if mask_rows:
                lines.append("  Mask layout (# = available, . = removed):")
                for row in mask_rows:
                    lines.append(f"    {row}")
        elif event_type == "phase_completed":
            phase = event.get("phase")
            elapsed = event.get("total_elapsed")
            elapsed_text = f"{elapsed:.2f}s" if isinstance(elapsed, (int, float)) else "unknown"
            success = "yes" if event.get("success") else "no"
            lines.append(f"Phase {phase} completed in {elapsed_text} (success: {success}).")
        elif event_type == "run_completed":
            elapsed = event.get("overall_elapsed")
            elapsed_text = f"{elapsed:.2f}s" if isinstance(elapsed, (int, float)) else "unknown"
            success = "yes" if event.get("success") else "no"
            lines.append(f"Run completed in {elapsed_text} (success: {success}).")
        elif event_type == "error":
            message = event.get("message")
            if message:
                lines.append(f"Error: {message}")

        if lines:
            self._append_lines(lines)

    def log_error(self, message: str) -> None:
        self._append_lines([f"Error: {message}"])

    def append_summary(
        self,
        logs: List[PhaseLog],
        result: Optional[SolveResult],
        error: Optional[str] = None,
    ) -> None:
        if self._summary_written:
            return
        lines: List[str] = ["", "Summary:"]
        for phase_log in logs:
            lines.append(f"{phase_log.phase_name}")
            lines.append(f"  Total elapsed: {phase_log.total_elapsed:.2f}s")
            if not phase_log.attempts:
                lines.append("  No board attempts")
            for attempt in phase_log.attempts:
                width_ft, height_ft = attempt.board_size_ft
                lines.append(
                    "  Board {w:.2f}ft x {h:.2f}ft ({variant}) | elapsed={elapsed:.2f}s | backtracks={backtracks} | success={success}".format(
                        w=width_ft,
                        h=height_ft,
                        variant=attempt.variant_label,
                        elapsed=attempt.elapsed,
                        backtracks=attempt.backtracks,
                        success="yes" if attempt.success else "no",
                    )
                )
                if attempt.notes:
                    lines.append(f"    Note: {attempt.notes}")
                if attempt.mask_rows:
                    lines.append("    Mask layout (# = available, . = removed):")
                    for row in attempt.mask_rows:
                        lines.append(f"      {row}")
            if phase_log.result:
                lines.append("  Result achieved")
                if phase_log.result.discarded_tiles:
                    lines.append("  Discarded tiles:")
                    for tile in phase_log.result.discarded_tiles:
                        lines.append(
                            "    - {identifier} ({width:.1f}ft × {height:.1f}ft)".format(
                                identifier=tile.identifier,
                                width=tile.type.width_ft,
                                height=tile.type.height_ft,
                            )
                        )
            lines.append("")
        if error:
            lines.append(f"Run ended with error: {error}")
        elif result:
            lines.append("Run ended with a successful solution.")
        else:
            lines.append("Run completed without a solution.")
        self._append_lines(lines)
        self._summary_written = True

    def _append_lines(self, lines: List[str]) -> None:
        if not lines:
            return
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                for line in lines:
                    fh.write(f"{line}\n")


@dataclass
class RunState:
    queue: "queue.Queue[Dict[str, object]]"
    result: Optional[SolveResult] = None
    logs: Optional[List[PhaseLog]] = None
    error: Optional[str] = None
    done: bool = False
    thread: Optional[threading.Thread] = None
    created_at: float = field(default_factory=time.time)
    selection: Dict[str, int] = field(default_factory=dict)
    log_path: Optional[Path] = None
    log_writer: Optional[RunLogWriter] = None


class RunManager:
    def __init__(self) -> None:
        self._runs: Dict[str, RunState] = {}
        self._lock = threading.Lock()

    def start_run(self, selection: Dict[str, int]) -> str:
        run_id = uuid.uuid4().hex
        log_path = SETTINGS.LOG_DIR / "run_log.txt"
        log_writer = RunLogWriter(log_path, selection)
        state = RunState(
            queue.Queue(),
            selection=dict(selection),
            log_path=log_path,
            log_writer=log_writer,
        )
        with self._lock:
            self._runs[run_id] = state
        thread = threading.Thread(
            target=self._worker,
            args=(run_id, selection),
            daemon=True,
        )
        state.thread = thread
        thread.start()
        return run_id

    def get_state(self, run_id: str) -> Optional[RunState]:
        with self._lock:
            return self._runs.get(run_id)

    def _worker(self, run_id: str, selection: Dict[str, int]) -> None:
        state = self.get_state(run_id)
        if state is None:
            return

        log_writer = state.log_writer

        def progress(event: Dict[str, object]) -> None:
            event.setdefault("run_id", run_id)
            state.queue.put(event)
            if log_writer:
                log_writer.handle_event(event)

        logs: List[PhaseLog] = []
        try:
            result, logs = orchestrator.solve(selection, progress_callback=progress)
            state.result = result
            state.logs = logs
        except ValueError as exc:
            state.error = str(exc)
            if log_writer:
                log_writer.log_error(state.error)
            state.queue.put({"type": "error", "message": state.error, "run_id": run_id})
        finally:
            if log_writer:
                final_logs = state.logs if state.logs is not None else logs
                log_writer.append_summary(final_logs or [], state.result, state.error)
            state.done = True
            state.queue.put(
                {
                    "type": "finished",
                    "success": state.result is not None,
                    "error": state.error,
                    "run_id": run_id,
                }
            )


run_manager = RunManager()


@app.route("/")
def index():
    return render_template(
        "tile_selection_form.html",
        tile_options=SETTINGS.TILE_OPTIONS,
        max_quantity=10,
        grid_unit=SETTINGS.GRID_UNIT_FT,
        tile_bags=load_tile_bags(),
    )


@app.route("/solve", methods=["POST"])
def solve_tiles():
    selection = _parse_selection(request.form)
    selection_summary = _selection_summary(selection)
    log_path = SETTINGS.LOG_DIR / "run_log.txt"
    log_writer = RunLogWriter(log_path, selection)

    def progress(event: Dict[str, object]) -> None:
        log_writer.handle_event(event)

    try:
        result, logs = orchestrator.solve(selection, progress_callback=progress)
    except ValueError as exc:
        message = str(exc)
        log_writer.log_error(message)
        log_writer.append_summary([], None, message)
        return render_template(
            "results_form.html",
            result=None,
            logs=[],
            error=message,
            outputs={"run_log": log_writer.path.name},
            config=SETTINGS,
            selection_summary=selection_summary,
            phase_limits=_phase_limits(),
        )
    log_writer.append_summary(logs, result, None)
    outputs = _write_outputs(result, logs, log_writer.path)
    return render_template(
        "results_form.html",
        result=result,
        logs=logs,
        outputs=outputs,
        error=None if result else "No solution within the provided timeframes",
        config=SETTINGS,
        selection_summary=selection_summary,
        phase_limits=_phase_limits(),
    )


@app.route("/runs", methods=["POST"])
def start_run():
    selection = _parse_selection(request.form)
    run_id = run_manager.start_run(selection)
    return jsonify({"run_id": run_id}), 202


@app.route("/runs/<run_id>/stream")
def stream_run(run_id: str):
    state = run_manager.get_state(run_id)
    if state is None:
        abort(404)

    def event_stream():
        while True:
            if state.done and state.queue.empty():
                break
            try:
                event = state.queue.get(timeout=1)
            except queue.Empty:
                continue
            yield f"data: {json.dumps(event)}\n\n"
        yield "event: end\ndata: {}\n\n"

    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")


@app.route("/runs/<run_id>/result")
def run_result(run_id: str):
    state = run_manager.get_state(run_id)
    if state is None:
        abort(404)
    if not state.done:
        return "", 202
    logs = state.logs or []
    selection_summary = _selection_summary(state.selection)
    log_path = state.log_path
    if state.error:
        outputs: Dict[str, str] = {}
        if log_path:
            outputs["run_log"] = log_path.name
        return render_template(
            "results_form.html",
            result=None,
            logs=logs,
            error=state.error,
            outputs=outputs,
            config=SETTINGS,
            selection_summary=selection_summary,
            phase_limits=_phase_limits(),
        )
    outputs = _write_outputs(state.result, logs, log_path)
    return render_template(
        "results_form.html",
        result=state.result,
        logs=logs,
        outputs=outputs,
        error=None if state.result else "No solution within the provided timeframes",
        config=SETTINGS,
        selection_summary=selection_summary,
        phase_limits=_phase_limits(),
    )


@app.route("/outputs/<path:filename>")
def serve_output(filename: str):
    return send_from_directory(SETTINGS.OUTPUT_DIR, filename)


@app.route("/logs/<path:filename>")
def serve_log(filename: str):
    return send_from_directory(SETTINGS.LOG_DIR, filename, as_attachment=True)


def _parse_selection(form_data) -> Dict[str, int]:
    selection: Dict[str, int] = {}
    for name in SETTINGS.TILE_OPTIONS:
        value = form_data.get(name)
        try:
            count = int(value)
        except (TypeError, ValueError):
            count = 0
        selection[name] = max(0, min(10, count))
    return selection


def _selection_summary(selection: Dict[str, int]):
    summary = []
    for tile_name, dims in SETTINGS.TILE_OPTIONS.items():
        count = selection.get(tile_name, 0)
        if count <= 0:
            continue
        width_ft, height_ft = dims
        summary.append(
            {
                "name": tile_name,
                "count": count,
                "width_ft": width_ft,
                "height_ft": height_ft,
                "display_name": f"{tile_name.replace('x', ' ft × ')} ft tile",
                "dimensions_label": f"{width_ft:.1f} ft × {height_ft:.1f} ft",
            }
        )
    return summary


def _phase_limits() -> Dict[str, Optional[float]]:
    return {name: config.time_limit_sec for name, config in PHASE_CONFIGS.items()}


def _format_seconds(seconds: Optional[float]) -> str:
    if seconds is None:
        return "No limit"
    seconds_int = int(round(seconds))
    if seconds_int < 0:
        return "0s"
    minutes, secs = divmod(seconds_int, 60)
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


@app.context_processor
def inject_helpers():
    return {"format_seconds": _format_seconds}


def _write_outputs(
    result: SolveResult | None,
    logs: List[PhaseLog],
    log_path: Optional[Path] = None,
):
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
    if log_path is None:
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
    color_map: Dict[tuple[int, int], str] = {}
    palette_index = 0
    for placement in result.placements:
        size_key = tuple(sorted((placement.width, placement.height)))
        if size_key not in color_map:
            color_map[size_key] = palette[palette_index % len(palette)]
            palette_index += 1
        color = color_map[size_key]
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
                f"  Board {width_ft:.2f}ft x {height_ft:.2f}ft ({attempt.variant_label}) | elapsed={attempt.elapsed:.2f}s | backtracks={attempt.backtracks} | success={'yes' if attempt.success else 'no'}"
            )
            if attempt.notes:
                lines.append(f"    Note: {attempt.notes}")
        if phase_log.result:
            lines.append("  Result achieved")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    app.run(debug=True)
