# Review Store Contract

Status: initial implementation contract for the review-store data layer.

Current recommendation: use the `sim.review` Python/CLI query API for agent and
scripted review. The desktop Output Review Workbench is an experimental preview
and is not currently recommended for routine review workflows.

The review store is the durable data layer for inspecting completed OEL
single-run outputs. It is intended to make simulation outputs queryable after a
run finishes, without requiring the simulator to keep full in-memory state
alive.

The initial implementation writes an opt-in SQLite store for single-run
outputs. Campaign, controller-benchmark, sensitivity, and cross-run review
stores can build on the same conventions later.

## Product Goal

OEL should support this workflow:

1. A user or agent creates or edits scenario YAML.
2. The simulator validates and runs the scenario through the deterministic
   engine.
3. The output writer saves normal artifacts plus a review store.
4. Agents or scripts query the review store through the SELECT-only review API.
5. Users or agents create custom figures, tables, and saved insights from the
   same structured evidence.

The review store is not a replacement for the physics engine. It is a
post-processing and review surface over saved simulation evidence.

## Output Layout

When `outputs.review.enabled: true`, single-run outputs include:

```text
outputs/<scenario_name>/
  index.md
  master_run_summary.json
  master_run_log.json
  review/
    run.sqlite
    schema.json
    saved_views.json
    generated_artifacts.json
    figures/
```

Required paths for the first implementation:

- `review/run.sqlite`: SQLite database with normalized run data.
- `review/schema.json`: schema version, table inventory, units, and column
  descriptions.

Optional paths:

- `review/saved_views.json`: named queries, filters, and chart definitions.
- `review/generated_artifacts.json`: custom figures/tables saved from the
  workbench or agent review API.
- `review/figures/`: OEL-styled custom figures generated from review queries.

Legacy output folders without `review/run.sqlite` should still open in limited
mode by reading `master_run_summary.json`, `master_run_log.json`, CSV files, and
plot artifacts when available.

## Detail Levels

The scenario config exposes a review output section:

```yaml
outputs:
  review:
    enabled: true
    detail: "standard"
```

Recommended detail levels:

- `compact`: metadata, summary metrics, artifact inventory, events, and
  downsampled or derived histories.
- `standard`: compact plus per-sample state, relative state, thrust, attitude
  errors, and configured access/mission diagnostics.
- `full`: standard plus lower-stability debug histories that are useful for
  engineering review but may be large.

The current default is disabled so existing quickstart and smoke-test runs do
not grow extra artifacts unless a scenario opts in. When enabled, the initial
writer accepts `compact`, `standard`, and `full`; the first table set focuses on
standard single-run review data.

## Database Rules

SQLite is the preferred initial format because it is portable, inspectable,
available in the Python standard library, and directly supports SQL-like review
queries.

Rules:

- The database must be written under the run output directory.
- Table and column names must be stable, lowercase, and snake_case.
- Values should use SI or OEL-documented units, with unit names in column names
  where practical.
- Time-series tables should include `sample_index` and `time_s`.
- Object-specific tables should include `object_id`.
- Pair-specific tables should include `deputy_id` and `chief_id`.
- The writer should use transactions so partial stores are not mistaken for
  complete review evidence.
- The store should include enough metadata to detect the OEL version and schema
  version that wrote it.

The review store should be treated as derived evidence. The authoritative
simulation is still the scenario config plus deterministic simulator behavior.

## Initial Schema

The initial single-run schema should include the following tables.

### `run_metadata`

One row per run.

Recommended columns:

- `run_id`
- `scenario_name`
- `scenario_description`
- `oel_version`
- `review_schema_version`
- `generated_utc`
- `duration_s`
- `dt_s`
- `samples`
- `output_dir`
- `config_path`
- `summary_json_path`
- `run_log_json_path`

### `objects`

One row per active simulation object.

Recommended columns:

- `object_id`
- `object_type`
- `enabled`
- `mass_initial_kg`
- `role`

`role` is optional and should not be treated as a fixed engine role. Object IDs
remain the stable identifiers.

### `time_samples`

One row per retained time sample.

Recommended columns:

- `sample_index`
- `time_s`

### `object_state`

One row per object per retained time sample.

Recommended columns:

- `sample_index`
- `time_s`
- `object_id`
- `pos_x_eci_km`
- `pos_y_eci_km`
- `pos_z_eci_km`
- `vel_x_eci_km_s`
- `vel_y_eci_km_s`
- `vel_z_eci_km_s`
- `quat_w`
- `quat_x`
- `quat_y`
- `quat_z`
- `omega_x_rad_s`
- `omega_y_rad_s`
- `omega_z_rad_s`
- `mass_kg`

Columns may be null when a state component is unavailable for an object type.

### `relative_state`

One row per object pair per retained time sample for configured/default review
pairs.

Recommended columns:

- `sample_index`
- `time_s`
- `deputy_id`
- `chief_id`
- `r_radial_km`
- `i_intrack_km`
- `c_crosstrack_km`
- `v_radial_km_s`
- `v_intrack_km_s`
- `v_crosstrack_km_s`
- `range_km`
- `range_rate_km_s`

The default pair should come from `summary.primary_object_pair` when available.
Future implementations may allow additional configured review pairs.

### `thrust`

One row per object per retained time sample when thrust history is available.

Recommended columns:

- `sample_index`
- `time_s`
- `object_id`
- `accel_x_eci_km_s2`
- `accel_y_eci_km_s2`
- `accel_z_eci_km_s2`
- `accel_norm_km_s2`
- `burn_active`

Future versions may add RIC components for pair-aware review. If both ECI and
RIC components are present, column names must make the frame explicit.

### `attitude_error`

One row per object per retained time sample when desired attitude history is
available.

Recommended columns:

- `sample_index`
- `time_s`
- `object_id`
- `pointing_error_deg`
- `quat_error_angle_deg`

The first implementation may omit this table if robust attitude error
reconstruction is not yet centralized.

### `ground_access`

One row per station/object/time sample when ground-station access histories are
available.

Recommended columns:

- `sample_index`
- `time_s`
- `station_id`
- `object_id`
- `access`
- `line_of_sight`
- `range_km`
- `elevation_deg`
- `reason`

### `events`

One row per review event.

Recommended columns:

- `event_id`
- `time_s`
- `sample_index`
- `object_id`
- `event_type`
- `severity`
- `message`
- `source`

Examples include termination, threshold crossings, safety-zone violations,
guardrail events, burn interval start/stop, or review-derived observations.

### `metrics`

One row per scalar metric.

Recommended columns:

- `metric_id`
- `metric_name`
- `object_id`
- `deputy_id`
- `chief_id`
- `value`
- `units`
- `source`

Examples include closest approach, final range, total delta-v, peak pointing
error, access duration, and termination time.

### `artifacts`

One row per known output artifact.

Recommended columns:

- `artifact_id`
- `artifact_type`
- `path`
- `title`
- `source`
- `created_utc`

Artifact paths should be relative to the output directory when possible.

## Query API Requirements

The review API exposes a small safe interface for agents, notebooks, and future
review surfaces:

```python
from sim.review import ReviewWorkspace

workspace = ReviewWorkspace.open("outputs/my_run")
workspace.tables()
workspace.schema()
result = workspace.query("SELECT time_s, range_km FROM relative_state", max_rows=10)
workspace.saved_views()
```

The same safe query surface is available from the CLI for smoke checks and
agent workflows:

```bash
python -m sim.review outputs/my_run --query "SELECT time_s, range_km FROM relative_state LIMIT 10"
python -m sim.review outputs/my_run --query "SELECT scenario_name FROM run_metadata" --json
```

The experimental ORW preview can open a completed run directly, but it is not
currently recommended for routine review. Prefer `python -m sim.review`:

```bash
python run_orw.py --output outputs/my_run
```

Safety requirements:

- User queries must run through a read-only database connection.
- The public query surface allows only statements that start with `SELECT` or
  `WITH`.
- A SQLite authorizer denies mutation, schema changes, PRAGMA statements,
  attaches/detaches, transactions, and extension loading even if a query passes
  the first-token check.
- Queries have a configurable row limit.
- Query errors should be returned as review-friendly messages, not raw ORW
  crashes.
- SQL must not be able to write files, mutate tables, attach external
  databases, or load extensions.

Agents should use this API instead of ad hoc parsing of large JSON logs when a
review store is present.

## Saved Views And Generated Artifacts

A saved view should record:

- `view_id`
- `title`
- `description`
- `query`
- `created_utc`
- `chart_type`
- `x_column`
- `y_columns`
- `filters`

A generated artifact should record:

- artifact path
- source query or saved view ID
- chart/table options
- generated UTC
- OEL version
- review schema version
- style name

Custom figures should use the OEL plotting style helpers so workbench-generated
figures visually match simulator-generated artifacts.

The experimental ORW preview supports saving a query result as an OEL-styled
figure when the result contains at least two numeric columns. Saved figures are
written under `review/figures/`, and provenance for ORW-generated figures is
appended to `review/generated_artifacts.json`.

## Built-In Insight Recipes

The first workbench should include named recipes that compile to SQL or review
API calls:

- closest approach
- first range threshold crossing
- final relative state
- burn intervals
- total delta-v by object
- peak acceleration by object
- ground-station access windows
- safety or keepout violations
- termination summary
- artifact inventory

Recipes should be transparent: users should be able to inspect the generated
query or the documented calculation.

## Compatibility With Existing Outputs

The review store must not break current output artifacts.

Compatibility rules:

- `index.md`, `master_run_summary.json`, and `master_run_log.json` remain valid
  entry points.
- Existing plotting and custom-analysis scripts should keep working.
- If the review store fails to write, normal simulation artifacts should still
  be preserved unless strict review output is explicitly requested.
- Legacy folders without a review store open in limited mode.
- The review schema version should increment when tables or semantics change in
  a compatibility-sensitive way.

## Public And Pro Boundary

The public core should support single-run review stores and local output-folder
inspection.

Likely Pro extensions:

- campaign-level review databases,
- cross-run comparison stores,
- controller-benchmark stores,
- sensitivity-study stores,
- report assembly from saved views,
- richer agent-assisted insight recipes,
- large-run storage controls and compression.

Public review stores should not include private/customer data beyond what the
user's scenario and deterministic outputs already contain. Public export rules
should continue to exclude generated `outputs/` directories.
