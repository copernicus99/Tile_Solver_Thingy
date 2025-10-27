# Tile Solver Thingy

A Flask-based application that computes an optimum tile layout given a selection of rectangular tile sizes and quantities. The solver operates on a 6-inch grid and enforces the project constraints:

- No gaps between tiles.
- Maximum continuous edge length of 6 feet (including the perimeter).
- No plus-shaped intersections (four tile corners meeting).
- A tile can share an edge with at most one tile of the same shape.

The application runs the solver through sequential phases depending on the total square footage, mirroring the requirements document.

## Getting Started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
flask --app app run
```

Open http://localhost:5000 to access the tile selection form. After submitting your configuration, the solver will produce:

- `outputs/coords.txt` – coordinates for each tile's top-left corner.
- `outputs/layout.html` – a visual rendering of the layout.
- `static/run_log.txt` – detailed log of each solver phase and board attempt.

## Project Structure

```
app.py                # Flask application entry point
config.py             # Shared configuration constants
solver/               # Backtracking solver implementation
static/styles.css     # Dark theme styling for the UI
templates/            # Jinja templates for form and results pages
```

## Tests

The repository does not include automated tests yet. You can validate the code compiles by running:

```bash
python -m compileall .
```

