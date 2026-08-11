# Agent Custom Plots

Use the review plotting API when a completed OEL run has the data needed for a
brief, report, or inspection plot that is not already in the output folder.

The API reads only completed-run evidence. It does not rerun simulations, invent
missing samples, or bypass the review-store SQL guardrails.

## Python API

```python
from sim.review import EvidencePlotter, ReviewWorkspace

workspace = ReviewWorkspace.open("outputs/my_run")
plotter = EvidencePlotter(workspace)

plotter.line(
    sql="SELECT time_s, range_km FROM relative_state ORDER BY time_s",
    x="time_s",
    y="range_km",
    title="Relative range over time",
    x_label="Time (s)",
    y_label="Range (km)",
    artifact_id="brief_relative_range",
    style="dark",
)
```

The plot is saved under `review/figures/` by default, styled with the OEL plot
theme, and recorded in `review/generated_artifacts.json` with provenance.

## Plot QA Before Handoff

Agents must inspect generated plots before handing them to the user. A plot is
not done just because the command succeeded.

Minimum QA loop:

1. Generate the plot.
2. Open or render the saved artifact.
3. Check for common presentation defects:
   - legend covers important data,
   - title, axis labels, tick labels, or footer text overlap,
   - long category labels are unreadable,
   - text is clipped at the figure edge,
   - colors or line styles are too hard to distinguish,
   - plotted data is squeezed into a tiny region by an outlier or bad axis,
   - the figure is blank, nearly blank, or missing expected series.
4. Fix the plot before handoff when any defect is visible. Prefer a clear
   layout change over cosmetic tweaks:
   - move the legend outside the axes or to an empty corner,
   - rotate or shorten long tick labels,
   - increase figure size,
   - add margins or use tighter layout,
   - split a cluttered multi-series plot into separate figures,
   - use a bar chart or table when category labels dominate the view,
   - adjust the query or axis mapping when the data does not match the request.
5. In the final response, mention the artifact path and the evidence source
   used. If a visual issue remains because the data is inherently dense, call it
   out plainly.

## CLI

```bash
.venv/bin/python -m sim.review.plot outputs/my_run \
  --sql "SELECT time_s, range_km FROM relative_state ORDER BY time_s" \
  --x time_s \
  --y range_km \
  --title "Relative range over time" \
  --artifact-id brief_relative_range
```

The same CLI is available as a `sim.review` subcommand:

```bash
.venv/bin/python -m sim.review plot outputs/my_run --recipe relative_range
```

Use `--dry-run --json` to validate the query and plot mapping without writing a
figure:

```bash
.venv/bin/python -m sim.review.plot outputs/my_run \
  --sql "SELECT time_s, range_rate_km_s FROM relative_state ORDER BY time_s" \
  --x time_s \
  --y range_rate_km_s \
  --dry-run \
  --json
```

## Built-In Recipes

List available recipes:

```bash
.venv/bin/python -m sim.review.plot --list-recipes
```

Common recipes include:

- `relative_range`
- `relative_range_rate`
- `relative_velocity_components`
- `burn_activity`
- `ground_access`

In Python:

```python
plotter.relative_velocity_components(style="light")
plotter.burn_activity(artifact_id="brief_burn_activity")
```

## Custom Plot Types

Supported plot types:

- `line`
- `scatter`
- `bar`
- `histogram`
- `heatmap`

Examples:

```bash
.venv/bin/python -m sim.review.plot outputs/my_run \
  --sql "SELECT range_km FROM relative_state" \
  --type histogram \
  --y range_km \
  --title "Relative range distribution"
```

```python
plotter.heatmap(
    sql="SELECT station_id, object_id, access_samples FROM access_summary",
    x="station_id",
    y="object_id",
    value="access_samples",
    title="Access samples by station and object",
)
```

## Agent Rules

- Inspect available tables before inventing a query.
- Use only `SELECT` or `WITH` SQL.
- Do not run a new simulation to create data for a requested post-run plot
  unless the user explicitly asks for a new study.
- If a needed table or column is absent, say what evidence is missing.
- Prefer recipes for common OEL plots, then use custom SQL for the specific
  question.
- Include the SQL query or recipe name when summarizing what a generated plot
  shows.
- Visually QA every generated plot before handoff and revise unprofessional
  artifacts instead of presenting them as finished.

For table inspection without plotting, use `.venv/bin/python -m sim.review`.
