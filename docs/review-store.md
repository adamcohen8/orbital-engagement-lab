# Review Store Contract

Status: implementation contract for the single-run review store and common
workflow review evidence layer.

Current recommendation: use the `sim.review` Python/CLI query API for plain
scripted and routine table review, and use `sim.review.EvidencePlotter` or
`.venv/bin/python -m sim.review plot` for custom OEL-styled figures from
completed runs.

The review store is the durable data layer for inspecting completed OEL outputs.
It is intended to make simulation outputs queryable after a run finishes,
without requiring the simulator to keep full in-memory state alive.

Single-run scenarios can opt into a detailed SQLite review store. Workflow
outputs for Monte Carlo campaigns, controller bench, sensitivity, and
validation now write a common `review/workflow_manifest.json`; the supported
workflow reporters also write table-backed `review/run.sqlite` stores for
run-level review evidence.

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

## Review Evidence Tiers

OEL uses three review evidence tiers:

- `single_run_review`: opt-in detailed SQLite evidence for a deterministic
  single-run scenario.
- `workflow_review_manifest`: a common manifest doorway for completed workflow
  outputs, including Monte Carlo, controller bench, sensitivity, and validation.
- `workflow_review_tables`: normalized workflow-specific SQLite tables when a
  workflow has run-level evidence worth querying.

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

Supported workflow outputs include:

```text
outputs/<workflow_output>/
  index.md
  <workflow summary artifacts>
  review/
    workflow_manifest.json
    run.sqlite
    schema.json
    saved_views.json
    generated_artifacts.json
    figures/
```

Required review paths:

- `review/run.sqlite`: SQLite database with normalized run data.
- `review/schema.json`: schema version, table inventory, units, and column
  descriptions.

Optional paths:

- `review/workflow_manifest.json`: common workflow review doorway for campaign,
  controller-bench, sensitivity, and validation outputs.
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
not grow extra artifacts unless a scenario opts in. When enabled, the writer
accepts `compact`, `standard`, and `full`; the core table set focuses on
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

## Query And Schema Discovery

Use `.venv/bin/python -m sim.review` for routine agent and scripted inspection. Keep
queries read-only and use only `SELECT` or `WITH` statements:

```bash
.venv/bin/python -m sim.review outputs/my_run --query "SELECT scenario_name, duration_s, samples FROM run_metadata"
.venv/bin/python -m sim.review outputs/my_workflow --manifest
.venv/bin/python -m sim.review outputs/my_workflow --list-artifacts
```

Before writing custom queries against an unfamiliar table, inspect the run's
schema or sample one row instead of guessing column names:

```bash
.venv/bin/python -m sim.review outputs/my_run --query "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name" --json
.venv/bin/python -m sim.review outputs/my_run --query "SELECT * FROM object_state LIMIT 1" --json
```

The saved `review/schema.json` lists the tables available for that run. Common
agent query recipes and column names are maintained in
[`agent-review-queries.md`](agent-review-queries.md).

Workflow review stores expose common tables such as `workflow_metadata`,
`workflow_summary`, and `workflow_artifacts`, plus workflow-specific tables:

- controller bench: `bench_runs`, `bench_variant_summaries`,
  `bench_leaderboard`, and `bench_failures`,
- sensitivity: `sensitivity_runs` and `sensitivity_rankings`,
- Monte Carlo: `campaign_runs` and `campaign_metrics`,
- validation: `validation_benchmarks` and `validation_artifacts`.

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
- `config_sha256`
- `config_json`
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

### `mission_recovery_summary`

One row for a configured `analysis.mission_recovery` post-run estimate.

Recommended columns:

- `object_id`
- `goal`
- `method`
- `assessment_time_s`
- `assessment_sample_index`
- `recovery_available`
- `recovery_delta_v_m_s`
- `recovery_time_s`
- `recovery_time_basis`
- `propellant_kg`
- `propellant_fraction`
- `disturbance_delta_v_m_s`
- `disturbance_apsis`
- `slot_recovery_found`
- `slot_recovery_orbits`
- `slot_recovery_time_s`
- `slot_recovery_phase_error_deg`
- `best_slot_orbits`
- `best_slot_time_s`
- `best_slot_phase_error_deg`
- `local_orbit_shape_delta_v_m_s`
- `local_orbit_shape_position_error_km`
- `notes_json`

### `mission_recovery_elements`

Initial and assessment-state classical orbital elements used by the mission
recovery estimate.

Recommended columns:

- `object_id`
- `state_label`
- `a_km`
- `ecc`
- `inc_deg`
- `raan_deg`
- `argp_deg`
- `true_anomaly_deg`

`state_label` is `initial`, `target`, or `final`. When no explicit target orbit
is configured, the target row matches the initial orbit; retaining the row
makes the comparison basis explicit for downstream review queries.

### `mission_recovery_candidates`

Planner candidate rows for configured mission recovery and Orbit Transfer
Planner trade-space analysis.

Recommended columns:

- `candidate_id`
- `object_id`
- `goal`
- `source`
- `source_family` (`analytic_reconstitution` or `orbit_transfer`)
- `target_basis` (`initial_orbit` or `configured_target_orbit`)
- `description`
- `planned_delta_v_m_s`
- `simulated_delta_v_m_s`
- `planned_time_s`
- `simulated_recovery_time_s`
- `propellant_kg`
- `propellant_fraction`
- `feasible`
- `verified`
- `within_tolerances`
- `score`
- `recommended_modes_json`
- `transfer_type` (`zero_impulse`, `one_impulse_departure`,
  `one_impulse_arrival`, or `two_impulse_lambert` for Orbit Transfer Planner
  candidates)
- `departure_wait_s`
- `time_of_flight_s`
- `arrival_time_s`
- `target_phase_deg`
- `lambert_short_way`
- `lambert_revolutions`
- `solver_iterations`
- `solver_residual_s`
- `position_residual_km`
- `velocity_residual_m_s`
- `notes_json`

### `mission_recovery_burns`

Burn sequence rows for each planner candidate. Orbit Transfer Planner
candidates omit Lambert impulses whose delta-v is at or below the configured
`impulse_epsilon_m_s`, so a valid candidate can have zero, one, or two burn
rows.

Recommended columns:

- `candidate_id`
- `burn_index`
- `start_time_s`
- `duration_s`
- `frame`
- `axis`
- `delta_v_m_s`
- `delta_v_eci_m_s_json`

### `mission_recovery_candidate_elements`

Expected candidate final classical orbital elements and element-error JSON. For
`orbit_slot` planner candidates, `element_errors_json` includes
`slot_phase_deg`, the direct position-angle phase error used for slot
verification.

Recommended columns:

- `candidate_id`
- `object_id`
- `a_km`
- `ecc`
- `inc_deg`
- `raan_deg`
- `argp_deg`
- `true_anomaly_deg`
- `element_errors_json`

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
.venv/bin/python -m sim.review outputs/my_run --query "SELECT time_s, range_km FROM relative_state LIMIT 10"
.venv/bin/python -m sim.review outputs/my_run --query "SELECT scenario_name FROM run_metadata" --json
```

Use the custom review plotting API when a completed output folder has the
review evidence needed for a plot that was not generated during the simulation:

```bash
.venv/bin/python -m sim.review plot outputs/my_run --recipe relative_velocity_components --style light
.venv/bin/python -m sim.review.plot outputs/my_run \
  --sql "SELECT time_s, range_km FROM relative_state ORDER BY time_s" \
  --x time_s --y range_km \
  --title "Relative range over time" \
  --dry-run --json
```

Python callers use the same implementation:

```python
from sim.review import EvidencePlotter, ReviewWorkspace

plotter = EvidencePlotter(ReviewWorkspace.open("outputs/my_run"))
plotter.line(
    sql="SELECT time_s, range_km FROM relative_state ORDER BY time_s",
    x="time_s",
    y="range_km",
    title="Relative range over time",
)
```

Prefer `.venv/bin/python -m sim.review` when you only need scripted or
agent-friendly tabular inspection. See `docs/agent-custom-plots.md` for custom
plot examples and agent rules.

Safety requirements:

- User queries must run through a read-only database connection.
- The public query surface allows only statements that start with `SELECT` or
  `WITH`.
- A SQLite authorizer denies mutation, schema changes, PRAGMA statements,
  attaches/detaches, transactions, and extension loading even if a query passes
  the first-token check.
- Queries have a configurable row limit.
- Query errors should be returned as review-friendly messages.
- SQL must not be able to write files, mutate tables, attach external
  databases, or load extensions.
- Custom plot creation must go through the review plotting API so SQL remains
  read-only, generated figures retain OEL style, and provenance is recorded.

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

Custom figures should use the OEL plotting style helpers so generated figures
visually match simulator-generated artifacts. Saved figures are written under
`review/figures/`, and generated-figure provenance is appended to
`review/generated_artifacts.json`.

## Built-In Insight Recipes

Built-in recipes compile to SQL or review API calls:

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
- `review/schema.json` includes a `compatibility` block. The current policy is
  `pre_1_0_additive`: stable core tables should remain queryable, new tables or
  nullable columns may be added, and compatibility-sensitive table/semantic
  changes require a schema-version bump.
- Stable core tables are `run_metadata`, `objects`, `time_samples`,
  `object_state`, `relative_state`, `thrust`, `metrics`, and `artifacts`.

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
