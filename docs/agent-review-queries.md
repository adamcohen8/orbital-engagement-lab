# Agent Review Query Recipes

When a scenario enables:

```yaml
outputs:
  review:
    enabled: true
    detail: standard
```

OEL writes `review/run.sqlite` under the run output directory. Agents should use
the SELECT-only review CLI/API for evidence-backed answers:

```bash
.venv/bin/python -m sim.review outputs/<scenario_name> --query "SELECT scenario_name, duration_s, samples FROM run_metadata"
```

For common agent workflows, built-in saved query names are available:

```bash
.venv/bin/python -m sim.review outputs/<scenario_name> --list-saved-queries
.venv/bin/python -m sim.review outputs/<scenario_name> --saved-query run_metadata
.venv/bin/python -m sim.review outputs/<scenario_name> --saved-query object_state_first_last
.venv/bin/python -m sim.review outputs/<scenario_name> --saved-query rendezvous_metrics
.venv/bin/python -m sim.review outputs/<scenario_name> --saved-query actuator_command_chain
.venv/bin/python -m sim.review outputs/<scenario_name> --saved-query safety_requirement_status
.venv/bin/python -m sim.review outputs/<scenario_name> --saved-query event_log
```

Use only `SELECT` or `WITH` queries. If `review/run.sqlite` is missing, inspect
`index.md`, `master_run_summary.json`, CSV histories, and plots instead. Do not
claim structured review evidence exists when the review store was not written.

For completed Monte Carlo, controller-bench, sensitivity, and validation
workflows, inspect the common workflow manifest first:

```bash
.venv/bin/python -m sim.review outputs/<workflow_output> --manifest
.venv/bin/python -m sim.review outputs/<workflow_output> --list-artifacts
.venv/bin/python -m sim.review outputs/<workflow_output> --saved-query workflow_metadata
```

When a workflow writes table-backed review evidence, use the workflow saved
queries before parsing JSON/CSV artifacts by hand:

```bash
.venv/bin/python -m sim.review outputs/<controller_bench_output> --saved-query controller_bench_runs
.venv/bin/python -m sim.review outputs/<sensitivity_output> --saved-query sensitivity_rankings
.venv/bin/python -m sim.review outputs/<monte_carlo_output> --saved-query campaign_runs
.venv/bin/python -m sim.review outputs/<validation_output> --saved-query validation_benchmarks
```

Saved query names are conveniences for common public agent tasks. When a result
matters, state either the saved query name or the SQL query used.
Each saved query carries machine-readable metadata in `sim.review.queries`:

- `source_tables`: the review tables the query reads,
- `maturity`: `supported`, `prototype`, or `experimental`,
- `allow_empty`: whether zero rows can be normal for that query.

Treat zero rows from a query with `allow_empty: false` as evidence to investigate
the scenario or review store before making a claim.

For first-run propagation, rendezvous, and mission-recovery workflows, start
with [`agent-capability-routing.md`](agent-capability-routing.md). It names the exact
configs, output directories, and saved queries that should be run before an
agent writes a custom SQL query.

## Common Review Columns

Use these column names when writing custom review-store queries. If a table is
missing, the scenario did not record that evidence path.

| Table | Common columns for agents |
| --- | --- |
| `run_metadata` | `scenario_name`, `duration_s`, `dt_s`, `samples`, `oel_version`, `review_schema_version` |
| `objects` | `object_id`, `object_type`, `role`, `mass_initial_kg`, `runtime_profile`, `flight_software_stack` |
| `object_state` | `sample_index`, `time_s`, `object_id`, ECI position/velocity, attitude quaternion/body rate, `mass_kg` |
| `object_propagation` | `object_id`, `propagation_method`, `general_model`, `native_frame`, `output_frame`, `frame_transform`, `tle_epoch_jd_utc`, `tle_age_start_days`, `tle_age_end_days` |
| `object_state_frame` | `object_id`, `state_frame` |
| `object_state_covariance` | `sample_index`, `time_s`, `object_id`, `frame`, `component_order_json`, `units_json`, `covariance_json`, `mathematically_valid`, `calibrated`, `calibration_scope`, `source` |
| `object_state` attitude fields | `quat_w`, `quat_x`, `quat_y`, `quat_z`, `omega_x_rad_s`, `omega_y_rad_s`, `omega_z_rad_s` |
| `relative_state` | `time_s`, `deputy_id`, `chief_id`, `r_radial_km`, `i_intrack_km`, `c_crosstrack_km`, `range_km`, `range_rate_km_s` |
| `thrust` | `time_s`, `object_id`, `burn_active`, `accel_norm_km_s2` |
| `controller_decisions` | `time_s`, `object_id`, controller/mission identities, requested/applied command norms, `burn_requested`, `burn_applied`, `saturated`, `deadline_missed` |
| `mission_modes` | `time_s`, `object_id`, `mission_strategy`, `mission_execution`, `mission_phase`, `executive_mode` |
| `mission_transitions` | `time_s`, `object_id`, `from_mode`, `to_mode`, `trigger`, `reason` |
| `command_gates` | `time_s`, `object_id`, burn state, alignment, fuel/actuator/deadline flags, `gate_reason` |
| `fsw_invocations` | `object_id`, `invocation_id`, `invocation_time_ns`, `stack_id`, `profile_id`, input/command/telemetry counts |
| `fsw_input_events` | packet identity, `invocation_id`, `kind`, source/delivery time, `schema` |
| `fsw_task_timing` | `object_id`, `invocation_id`, `task_id`, release time, modeled duration, budget, `deadline_missed` |
| `actuator_commands` | command identity, `actuator_id`, issue/not-before/expiry times, command schema |
| `actuator_command_receipts` | command identity, receive time, `disposition`, status codes |
| `actuator_realization` | command identity, realization interval, demand mode, `saturated`, physical detail |
| `safety_requirement_evidence` | `object_id`, `invocation_id`, `requirement_id`, `satisfied`, `source`, detail |
| `fsw_snapshots` | `object_id`, `invocation_id`, stack identity and state hash; opaque restart state remains in `detail_json` |
| `ground_access` | `time_s`, `station_id`, `object_id`, `access`, `range_km`, `elevation_deg`, `reason` |
| `events` | `time_s`, `object_id`, `event_type`, `severity`, `message` |
| `metrics` | `metric_name`, `value`, `units`, `object_id`, `deputy_id`, `chief_id` |
| `mission_recovery_summary` | `object_id`, `goal`, `method`, `recovery_delta_v_m_s`, `recovery_time_s`, `propellant_kg`, `slot_recovery_time_s` |
| `mission_recovery_elements` | `object_id`, `state_label`, `a_km`, `ecc`, `inc_deg`, `raan_deg`, `argp_deg`, `true_anomaly_deg` |
| `mission_recovery_candidates` | `candidate_id`, `object_id`, `goal`, `source`, `planned_delta_v_m_s`, `planned_time_s`, `feasible`, `verified` |
| `mission_recovery_burns` | `candidate_id`, `burn_index`, `start_time_s`, `frame`, `axis`, `delta_v_m_s` |
| `artifacts` | `artifact_type`, `artifact_id`, `path` |
| `workflow_metadata` | `workflow_type`, `scenario_name`, `title`, `status`, `generated_utc`, `review_schema_version`, `source_config` |
| `workflow_artifacts` | `artifact_key`, `artifact_type`, `path` |
| `bench_runs` | `variant_name`, `case_name`, `passed`, `failure_count`, `output_dir` |
| `bench_leaderboard` | `kind`, `objective`, `metric`, `rank`, `variant_name`, `value`, `samples` |
| `bench_failures` | `variant_name`, `case_name`, `objective`, `metric`, `reason`, `failure_mode`, `suggestion` |
| `sensitivity_runs` | `run_id`, `status`, `parameter_path`, `parameter_value`, `output_dir` |
| `sensitivity_rankings` | `rank`, `parameter_path`, `metric_path`, `method`, `effect_size` |
| `campaign_runs` | `iteration`, `passed`, `closest_approach_km`, `duration_s`, `total_dv_m_s`, `output_dir` |
| `campaign_metrics` | `iteration`, `metric_name`, `metric_value` |
| `validation_benchmarks` | `benchmark_name`, `kind`, `passed`, `duration_s`, `output_dir` |

## Schema Discovery

Do not guess review-store column names. When a query fails or you need a table
you have not used before, inspect the saved schema or ask SQLite for one row:

```bash
.venv/bin/python -m sim.review outputs/<scenario_name> --query "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name" --json
.venv/bin/python -m sim.review outputs/<scenario_name> --query "SELECT * FROM object_state LIMIT 1" --json
```

You can also inspect the schema file saved with the run:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path

schema = json.loads(Path("outputs/<scenario_name>/review/schema.json").read_text())
print(json.dumps(schema, indent=2))
PY
```

For Python workflows, use `ReviewWorkspace`:

```python
from sim.review import ReviewWorkspace

workspace = ReviewWorkspace.open("outputs/<scenario_name>")
print(workspace.tables())
print(workspace.query("SELECT * FROM object_state LIMIT 1").columns)
```

For first/final object-state checks, prefer this pattern over guessing field
names:

```bash
.venv/bin/python -m sim.review outputs/<scenario_name> --query "WITH bounds AS (SELECT object_id, MIN(sample_index) AS first_i, MAX(sample_index) AS last_i FROM object_state WHERE object_id = 'target' GROUP BY object_id) SELECT s.sample_index, s.time_s, s.object_id, s.pos_x_eci_km, s.pos_y_eci_km, s.pos_z_eci_km, s.vel_x_eci_km_s, s.vel_y_eci_km_s, s.vel_z_eci_km_s FROM object_state s JOIN bounds b ON s.object_id = b.object_id AND s.sample_index IN (b.first_i, b.last_i) ORDER BY s.sample_index"
```

## First Checks

Run metadata:

```bash
.venv/bin/python -m sim.review outputs/<scenario_name> --query "SELECT scenario_name, duration_s, dt_s, samples, oel_version, review_schema_version FROM run_metadata"
```

Active objects:

```bash
.venv/bin/python -m sim.review outputs/<scenario_name> --query "SELECT object_id, object_type, role, mass_initial_kg, runtime_profile, flight_software_stack FROM objects ORDER BY object_id"
```

Artifact inventory:

```bash
.venv/bin/python -m sim.review outputs/<scenario_name> --query "SELECT artifact_type, artifact_id, path FROM artifacts ORDER BY artifact_type, artifact_id"
```

Events:

```bash
.venv/bin/python -m sim.review outputs/<scenario_name> --query "SELECT time_s, object_id, event_type, severity, message FROM events ORDER BY time_s, event_id"
```

## Passive Propagation

Continuous OGP propagation and frame contract:

```bash
.venv/bin/python -m sim.review outputs/<scenario_name> --saved-query ogp_propagation_contract --json
```

The OGP product's native/output frame and the canonical review-state frame are
separate fields. Do not infer continuous OGP behavior from a TLE alone; require
`propagation_method: general` and the expected `general_model` row.

Final object state:

```bash
.venv/bin/python -m sim.review outputs/<scenario_name> --query "SELECT object_id, time_s, pos_x_eci_km, pos_y_eci_km, pos_z_eci_km, vel_x_eci_km_s, vel_y_eci_km_s, vel_z_eci_km_s FROM object_state WHERE sample_index = (SELECT MAX(sample_index) FROM object_state) ORDER BY object_id"
```

First and last state for one object:

```bash
.venv/bin/python -m sim.review outputs/<scenario_name> --query "WITH bounds AS (SELECT MIN(sample_index) AS first_i, MAX(sample_index) AS last_i FROM object_state WHERE object_id = 'target') SELECT object_id, time_s, pos_x_eci_km, pos_y_eci_km, pos_z_eci_km, vel_x_eci_km_s, vel_y_eci_km_s, vel_z_eci_km_s FROM object_state, bounds WHERE object_id = 'target' AND sample_index IN (first_i, last_i) ORDER BY sample_index"
```

Replace `target` with the object ID from the `objects` table.

## Rendezvous And Relative Motion

Relative range summary from metrics:

```bash
.venv/bin/python -m sim.review outputs/<scenario_name> --query "SELECT metric_name, value, units, deputy_id, chief_id FROM metrics WHERE metric_name IN ('initial_range_km', 'final_range_km', 'closest_approach_km', 'closest_approach_time_s') ORDER BY metric_name"
```

Relative state samples:

```bash
.venv/bin/python -m sim.review outputs/<scenario_name> --query "SELECT time_s, deputy_id, chief_id, r_radial_km, i_intrack_km, c_crosstrack_km, range_km, range_rate_km_s FROM relative_state ORDER BY time_s LIMIT 20" --json
```

Closest approach from time history:

```bash
.venv/bin/python -m sim.review outputs/<scenario_name> --query "SELECT time_s, deputy_id, chief_id, range_km, range_rate_km_s FROM relative_state WHERE range_km IS NOT NULL ORDER BY range_km ASC LIMIT 1"
```

Final relative state:

```bash
.venv/bin/python -m sim.review outputs/<scenario_name> --query "SELECT time_s, deputy_id, chief_id, r_radial_km, i_intrack_km, c_crosstrack_km, range_km, range_rate_km_s FROM relative_state ORDER BY time_s DESC LIMIT 1"
```

Burn activity:

```bash
.venv/bin/python -m sim.review outputs/<scenario_name> --query "SELECT object_id, COUNT(*) AS active_samples, MAX(accel_norm_km_s2) AS max_accel_km_s2 FROM thrust WHERE burn_active = 1 GROUP BY object_id ORDER BY object_id"
```

Burn intervals from events:

```bash
.venv/bin/python -m sim.review outputs/<scenario_name> --query "SELECT time_s, object_id, event_type, message FROM events WHERE event_type IN ('burn_start', 'burn_end') ORDER BY time_s, event_id"
```

## Mission Recovery

Configured mission recovery summary:

```bash
.venv/bin/python -m sim.review outputs/<scenario_name> --saved-query mission_recovery_summary
```

Initial and final orbital elements used for recovery:

```bash
.venv/bin/python -m sim.review outputs/<scenario_name> --saved-query mission_recovery_elements
```

Planner candidate trade space:

```bash
.venv/bin/python -m sim.review outputs/<scenario_name> --saved-query mission_recovery_candidates
.venv/bin/python -m sim.review outputs/<scenario_name> --saved-query mission_recovery_burns
```

## Ground Access

Access samples by station and object:

```bash
.venv/bin/python -m sim.review outputs/<scenario_name> --query "SELECT station_id, object_id, COUNT(*) AS samples, SUM(access) AS access_samples, MIN(range_km) AS min_range_km, MAX(elevation_deg) AS max_elevation_deg FROM ground_access GROUP BY station_id, object_id ORDER BY station_id, object_id"
```

Access-window edge samples:

```bash
.venv/bin/python -m sim.review outputs/<scenario_name> --query "SELECT time_s, station_id, object_id, access, range_km, elevation_deg, reason FROM ground_access WHERE access = 1 ORDER BY station_id, object_id, time_s LIMIT 20" --json
```

No-access reasons:

```bash
.venv/bin/python -m sim.review outputs/<scenario_name> --query "SELECT station_id, object_id, reason, COUNT(*) AS samples FROM ground_access WHERE access = 0 GROUP BY station_id, object_id, reason ORDER BY station_id, object_id, samples DESC"
```

## Coverage And Directed Links

Coverage summary and sampled extrema:

```bash
python -m sim.review outputs/<scenario_name> --query "SELECT s.analysis_id, s.source_object_id, COUNT(c.sample_index) AS samples, MIN(c.instantaneous_covered_fraction) AS minimum_fraction, MAX(c.instantaneous_covered_fraction) AS maximum_fraction FROM coverage_summary s LEFT JOIN coverage_samples c USING (analysis_id) GROUP BY s.analysis_id, s.source_object_id ORDER BY s.analysis_id"
```

Coverage transitions and their refinement dispositions:

```bash
python -m sim.review outputs/<scenario_name> --query "SELECT analysis_id, transition_kind, disposition, COUNT(*) AS transitions FROM coverage_transitions GROUP BY analysis_id, transition_kind, disposition ORDER BY analysis_id, transition_kind, disposition"
```

Directed-link availability, range, and margin:

```bash
python -m sim.review outputs/<scenario_name> --query "SELECT s.analysis_id, s.link_id, COUNT(l.sample_index) AS samples, SUM(l.available) AS available_samples, MIN(l.range_km) AS minimum_range_km, MIN(l.margin_db) AS minimum_margin_db, MAX(l.margin_db) AS maximum_margin_db FROM link_summary s LEFT JOIN link_samples l USING (analysis_id) GROUP BY s.analysis_id, s.link_id ORDER BY s.analysis_id"
```

Directed-link windows:

```bash
python -m sim.review outputs/<scenario_name> --query "SELECT analysis_id, interval_index, start_s, end_s, duration_s, minimum_margin_db, estimated_delivered_data_bits, acquisition_disposition, loss_disposition FROM link_windows ORDER BY analysis_id, interval_index" --json
```

## Attitude And Applied Commands

Angular-rate samples:

```bash
.venv/bin/python -m sim.review outputs/<scenario_name> --query "SELECT object_id, time_s, omega_x_rad_s, omega_y_rad_s, omega_z_rad_s FROM object_state ORDER BY object_id, time_s LIMIT 20"
```

First and final angular rates:

```bash
.venv/bin/python -m sim.review outputs/<scenario_name> --query "WITH bounds AS (SELECT object_id, MIN(sample_index) AS first_i, MAX(sample_index) AS last_i FROM object_state GROUP BY object_id) SELECT s.object_id, s.time_s, s.omega_x_rad_s, s.omega_y_rad_s, s.omega_z_rad_s FROM object_state s JOIN bounds b ON s.object_id = b.object_id AND s.sample_index IN (b.first_i, b.last_i) ORDER BY s.object_id, s.sample_index"
```

First and final quaternion plus angular rates:

```bash
.venv/bin/python -m sim.review outputs/<scenario_name> --saved-query attitude_state_first_last
```

Applied acceleration summary:

```bash
.venv/bin/python -m sim.review outputs/<scenario_name> --query "SELECT object_id, COUNT(*) AS samples, SUM(burn_active) AS active_samples, MAX(accel_norm_km_s2) AS max_accel_km_s2 FROM thrust GROUP BY object_id ORDER BY object_id"
```

For attitude-only scenarios, treat `thrust` as orbital-acceleration evidence;
it is not reaction-wheel torque, wheel speed, or full ADCS telemetry.

## Answering Rules

- State the query used when a result matters.
- Treat empty tables as evidence about what was not recorded, not as permission
  to invent a result.
- Use `relative_state` only when the scenario has a primary object pair.
- For single-object propagation, use `object_state`, `metrics`, and artifacts.
- For access questions, use `ground_access` plus the output reports when
  present.
- For coverage or link questions, use the matching summary plus sample,
  interval/window, and transition tables; report cadence, censoring, and
  refinement disposition with the result.
- For rendezvous success, compare final range, closest approach, range rate,
  time, delta-v or burn activity, and any user-provided safety constraints.
