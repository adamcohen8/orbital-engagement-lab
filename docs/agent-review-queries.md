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
python -m sim.review outputs/<scenario_name> --query "SELECT scenario_name, duration_s, samples FROM run_metadata"
```

For common agent workflows, built-in saved query names are available:

```bash
python -m sim.review outputs/<scenario_name> --list-saved-queries
python -m sim.review outputs/<scenario_name> --saved-query run_metadata
python -m sim.review outputs/<scenario_name> --saved-query rendezvous_metrics
```

Use only `SELECT` or `WITH` queries. If `review/run.sqlite` is missing, inspect
`index.md`, `master_run_summary.json`, CSV histories, and plots instead. Do not
claim structured review evidence exists when the review store was not written.

Saved query names are conveniences for common public agent tasks. When a result
matters, state either the saved query name or the SQL query used.

## Common Review Columns

Use these column names when writing custom review-store queries. If a table is
missing, the scenario did not record that evidence path.

| Table | Common columns for agents |
| --- | --- |
| `run_metadata` | `scenario_name`, `duration_s`, `dt_s`, `samples`, `oel_version`, `review_schema_version` |
| `objects` | `object_id`, `object_type`, `role`, `mass_initial_kg` |
| `object_state` | `sample_index`, `time_s`, `object_id`, `pos_x_eci_km`, `pos_y_eci_km`, `pos_z_eci_km`, `vel_x_eci_km_s`, `vel_y_eci_km_s`, `vel_z_eci_km_s` |
| `object_state` attitude fields | `quat_w`, `quat_x`, `quat_y`, `quat_z`, `omega_x_rad_s`, `omega_y_rad_s`, `omega_z_rad_s` |
| `relative_state` | `time_s`, `deputy_id`, `chief_id`, `r_radial_km`, `i_intrack_km`, `c_crosstrack_km`, `range_km`, `range_rate_km_s` |
| `thrust` | `time_s`, `object_id`, `burn_active`, `accel_norm_km_s2` |
| `ground_access` | `time_s`, `station_id`, `object_id`, `access`, `range_km`, `elevation_deg`, `reason` |
| `events` | `time_s`, `object_id`, `event_type`, `severity`, `message` |
| `metrics` | `metric_name`, `value`, `units`, `object_id`, `deputy_id`, `chief_id` |
| `artifacts` | `artifact_type`, `artifact_id`, `path` |

For first/final object-state checks, prefer this pattern over guessing field
names:

```bash
python -m sim.review outputs/<scenario_name> --query "WITH bounds AS (SELECT object_id, MIN(sample_index) AS first_i, MAX(sample_index) AS last_i FROM object_state WHERE object_id = 'target' GROUP BY object_id) SELECT s.sample_index, s.time_s, s.object_id, s.pos_x_eci_km, s.pos_y_eci_km, s.pos_z_eci_km, s.vel_x_eci_km_s, s.vel_y_eci_km_s, s.vel_z_eci_km_s FROM object_state s JOIN bounds b ON s.object_id = b.object_id AND s.sample_index IN (b.first_i, b.last_i) ORDER BY s.sample_index"
```

## First Checks

Run metadata:

```bash
python -m sim.review outputs/<scenario_name> --query "SELECT scenario_name, duration_s, dt_s, samples, oel_version, review_schema_version FROM run_metadata"
```

Active objects:

```bash
python -m sim.review outputs/<scenario_name> --query "SELECT object_id, object_type, role, mass_initial_kg FROM objects ORDER BY object_id"
```

Artifact inventory:

```bash
python -m sim.review outputs/<scenario_name> --query "SELECT artifact_type, artifact_id, path FROM artifacts ORDER BY artifact_type, artifact_id"
```

Events:

```bash
python -m sim.review outputs/<scenario_name> --query "SELECT time_s, object_id, event_type, severity, message FROM events ORDER BY time_s, event_id"
```

## Passive Propagation

Final object state:

```bash
python -m sim.review outputs/<scenario_name> --query "SELECT object_id, time_s, pos_x_eci_km, pos_y_eci_km, pos_z_eci_km, vel_x_eci_km_s, vel_y_eci_km_s, vel_z_eci_km_s FROM object_state WHERE sample_index = (SELECT MAX(sample_index) FROM object_state) ORDER BY object_id"
```

First and last state for one object:

```bash
python -m sim.review outputs/<scenario_name> --query "WITH bounds AS (SELECT MIN(sample_index) AS first_i, MAX(sample_index) AS last_i FROM object_state WHERE object_id = 'target') SELECT object_id, time_s, pos_x_eci_km, pos_y_eci_km, pos_z_eci_km, vel_x_eci_km_s, vel_y_eci_km_s, vel_z_eci_km_s FROM object_state, bounds WHERE object_id = 'target' AND sample_index IN (first_i, last_i) ORDER BY sample_index"
```

Replace `target` with the object ID from the `objects` table.

## Rendezvous And Relative Motion

Relative range summary from metrics:

```bash
python -m sim.review outputs/<scenario_name> --query "SELECT metric_name, value, units, deputy_id, chief_id FROM metrics WHERE metric_name IN ('initial_range_km', 'final_range_km', 'closest_approach_km', 'closest_approach_time_s') ORDER BY metric_name"
```

Relative state samples:

```bash
python -m sim.review outputs/<scenario_name> --query "SELECT time_s, deputy_id, chief_id, r_radial_km, i_intrack_km, c_crosstrack_km, range_km, range_rate_km_s FROM relative_state ORDER BY time_s LIMIT 20" --json
```

Closest approach from time history:

```bash
python -m sim.review outputs/<scenario_name> --query "SELECT time_s, deputy_id, chief_id, range_km, range_rate_km_s FROM relative_state ORDER BY range_km ASC LIMIT 1"
```

Final relative state:

```bash
python -m sim.review outputs/<scenario_name> --query "SELECT time_s, deputy_id, chief_id, r_radial_km, i_intrack_km, c_crosstrack_km, range_km, range_rate_km_s FROM relative_state ORDER BY time_s DESC LIMIT 1"
```

Burn activity:

```bash
python -m sim.review outputs/<scenario_name> --query "SELECT object_id, COUNT(*) AS active_samples, MAX(accel_norm_km_s2) AS max_accel_km_s2 FROM thrust WHERE burn_active = 1 GROUP BY object_id ORDER BY object_id"
```

Burn intervals from events:

```bash
python -m sim.review outputs/<scenario_name> --query "SELECT time_s, object_id, event_type, message FROM events WHERE event_type IN ('burn_start', 'burn_end') ORDER BY time_s, event_id"
```

## Ground Access

Access samples by station and object:

```bash
python -m sim.review outputs/<scenario_name> --query "SELECT station_id, object_id, COUNT(*) AS samples, SUM(access) AS access_samples, MIN(range_km) AS min_range_km, MAX(elevation_deg) AS max_elevation_deg FROM ground_access GROUP BY station_id, object_id ORDER BY station_id, object_id"
```

Access-window edge samples:

```bash
python -m sim.review outputs/<scenario_name> --query "SELECT time_s, station_id, object_id, access, range_km, elevation_deg, reason FROM ground_access WHERE access = 1 ORDER BY station_id, object_id, time_s LIMIT 20" --json
```

No-access reasons:

```bash
python -m sim.review outputs/<scenario_name> --query "SELECT station_id, object_id, reason, COUNT(*) AS samples FROM ground_access WHERE access = 0 GROUP BY station_id, object_id, reason ORDER BY station_id, object_id, samples DESC"
```

## Attitude And Applied Commands

Angular-rate samples:

```bash
python -m sim.review outputs/<scenario_name> --query "SELECT object_id, time_s, omega_x_rad_s, omega_y_rad_s, omega_z_rad_s FROM object_state ORDER BY object_id, time_s LIMIT 20"
```

First and final angular rates:

```bash
python -m sim.review outputs/<scenario_name> --query "WITH bounds AS (SELECT object_id, MIN(sample_index) AS first_i, MAX(sample_index) AS last_i FROM object_state GROUP BY object_id) SELECT s.object_id, s.time_s, s.omega_x_rad_s, s.omega_y_rad_s, s.omega_z_rad_s FROM object_state s JOIN bounds b ON s.object_id = b.object_id AND s.sample_index IN (b.first_i, b.last_i) ORDER BY s.object_id, s.sample_index"
```

First and final quaternion plus angular rates:

```bash
python -m sim.review outputs/<scenario_name> --saved-query attitude_state_first_last
```

Applied acceleration summary:

```bash
python -m sim.review outputs/<scenario_name> --query "SELECT object_id, COUNT(*) AS samples, SUM(burn_active) AS active_samples, MAX(accel_norm_km_s2) AS max_accel_km_s2 FROM thrust GROUP BY object_id ORDER BY object_id"
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
- For rendezvous success, compare final range, closest approach, range rate,
  time, delta-v or burn activity, and any user-provided safety constraints.
