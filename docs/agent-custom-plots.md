# Agent Custom Plots

Use the review plotting API when a completed OEL run has the data needed for a
brief, report, or inspection plot that is not already in the output folder.

The API reads only completed-run evidence. It does not rerun simulations, invent
missing samples, or bypass the review-store SQL guardrails.

For any figure derived from OEL review evidence, this is the authoritative
plotting surface. Connected agents should use OEL plot recipes or the typed MCP
plan/render workflow before host-native visualization tools.

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
Static review plots also apply the supported
[OEL Plot Quality Contract](plot-quality-contract.md), including stable
axis-wide numeric formatting and renderer-level overlap, clipping, font, and
legend checks. Its automated receipt does not replace the visual QA below.

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

- `object_eci_radius`
- `relative_range`
- `relative_range_rate`
- `relative_velocity_components`
- `relative_position_ric_2d`
- `burn_activity`
- `ground_access`

In Python:

```python
plotter.relative_velocity_components(style="light")
plotter.relative_position_ric_2d(style="light")
plotter.burn_activity(artifact_id="brief_burn_activity")
```

`relative_position_ric_2d` creates equal-aspect I-R, I-C, and C-R panels from
the rectangular-RIC columns already recorded in `relative_state`. It marks the
chief origin and the first and last deputy samples. It does not reconstruct or
approximate missing RIC evidence.

## MCP Workflow

Read `oel://review/plot-recipes/v1` to discover the same authoritative recipe
registry used by the CLI, Python API, agent-task runner, and MCP.

Use `oel.plot_evidence.v1` when a supported recipe matches. When it does not,
call `oel.plan_review_plot.v1` with a typed read-only query and chart mapping.
The plan returns a content-bound `plot_plan_id` and performs no write. Echo the
same specification and ID to `oel.render_review_plot.v2` with an operator-
approved write reference. A changed query, chart mapping, or review store makes
the plan ID stale and rendering fails closed.

For motion rather than a static figure, read
`oel://review/animation-recipes/v1`, call `oel.plan_review_animation.v1`, and
render the unchanged content-bound plan with
`oel.render_review_animation.v1`. Version 1 supports the
`relative_position_ric_2d` recipe and produces an MP4 or GIF, a structured
quality receipt, and a deterministic contact sheet. Inspect both visual
artifacts before handoff; see the
[OEL Animation Quality Contract](animation-quality-contract.md).

The render result includes the artifact, query and review-store provenance,
automated QA checks, and `visual_qa_status: pending_agent_review`. PNG and SVG
results are also returned as MCP image content so the agent can perform the
required visual inspection in the same workflow.

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
    sql=(
        "SELECT station_id, object_id, SUM(access) AS access_samples "
        "FROM ground_access GROUP BY station_id, object_id"
    ),
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
