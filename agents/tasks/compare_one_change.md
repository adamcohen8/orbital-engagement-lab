# Compare One Change Task Card

Task ID: `compare_one_change`

Example config: `agents/examples/public_agent_rendezvous_lqr.yaml`

Expected output directory: `outputs/agents/public_agent_rendezvous_lqr`

Answer example: `agents/tasks/examples/compare_one_change_answer.md`

## User Prompt

```text
Take the closed-loop rendezvous case and change only the initial in-track
separation. Validate both cases, run both, query the review evidence, and
compare final range, closest approach, burn activity, and limitations.
```

## Expected Agent Assumptions

- Start from `agents/examples/public_agent_rendezvous_lqr.yaml`.
- Copy the config to a new scenario path and change one initial-state value.
- Use distinct scenario names and output directories for the baseline and
  modified run.
- Treat scenario name and output directory edits as bookkeeping so outputs do
  not overwrite each other. The "one change" constraint applies to physical
  scenario parameters such as initial separation, dynamics, controller settings,
  duration, and timestep.
- Validate both configs before running.
- Compare only metrics and histories present in artifacts.
- If only the baseline example is available, treat it as a smoke fixture and
  explain that the comparison requires the second run.

## Commands

Validate the baseline:

```bash
.venv/bin/python run_simulation.py --config agents/examples/public_agent_rendezvous_lqr.yaml --validate-only
```

Run the baseline:

```bash
.venv/bin/python run_simulation.py --config agents/examples/public_agent_rendezvous_lqr.yaml
```

## Required Review Queries

Range metrics for each run:

```sql
SELECT metric_name, value, units, deputy_id, chief_id FROM metrics WHERE metric_name IN ('initial_range_km', 'final_range_km', 'closest_approach_km', 'closest_approach_time_s') ORDER BY metric_name
```

Final relative state for each run:

```sql
SELECT time_s, deputy_id, chief_id, r_radial_km, i_intrack_km, c_crosstrack_km, range_km, range_rate_km_s FROM relative_state ORDER BY time_s DESC LIMIT 1
```

Burn activity for each run:

```sql
SELECT object_id, COUNT(*) AS samples, SUM(burn_active) AS active_samples, MAX(accel_norm_km_s2) AS max_accel_km_s2 FROM thrust GROUP BY object_id ORDER BY object_id
```

## Expected Answer Shape

- Status for each run: validated, ran, or failed.
- The one parameter changed.
- The bookkeeping names/output directories used for the baseline and modified
  runs.
- Baseline vs modified final range, closest approach, final range rate, and
  burn activity.
- Whether the evidence supports a stronger, weaker, or inconclusive result.
- Limitations and one focused follow-up run.

## Pass Criteria

- Both configs validate before execution.
- Both runs use distinct output directories.
- Comparison uses review-store evidence or saved artifacts.
- Agent does not attribute differences to unmodified factors.

## Red Flags

- Changes multiple variables without saying so.
- Reuses the same output directory for both runs.
- Compares from memory or unstated assumptions.
- Claims causality beyond the one-variable comparison evidence.
