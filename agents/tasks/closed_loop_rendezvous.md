# Closed-Loop Rendezvous Task Card

Task ID: `closed_loop_rendezvous`

Example config: `agents/examples/public_agent_rendezvous_lqr.yaml`

Expected output directory: `outputs/agents/public_agent_rendezvous_lqr`

Answer example: `agents/tasks/examples/closed_loop_rendezvous_answer.md`

## User Prompt

```text
Create a short public closed-loop rendezvous scenario with a chaser starting
about 5 km behind a passive target. Use a public controller, validate it, run
it, query the review store, and tell me whether it achieved terminal rendezvous
or only partial closure.
```

## Expected Agent Assumptions

- Use the checked-in public HCW LQR rendezvous example unless the user asks for
  a new custom scenario.
- Treat "validated and ran" as different from "rendezvous succeeded."
- Define success thresholds before claiming terminal rendezvous.
- Inspect relative range, closest approach, range rate, and burn activity.
- Keep plots disabled unless trajectory-shape review is requested.

## Commands

Validate:

```bash
python run_simulation.py --config agents/examples/public_agent_rendezvous_lqr.yaml --validate-only
```

Run:

```bash
python run_simulation.py --config agents/examples/public_agent_rendezvous_lqr.yaml
```

## Required Review Queries

Relative range summary:

```sql
SELECT metric_name, value, units, deputy_id, chief_id FROM metrics WHERE metric_name IN ('initial_range_km', 'final_range_km', 'closest_approach_km', 'closest_approach_time_s') ORDER BY metric_name
```

Closest approach:

```sql
SELECT time_s, deputy_id, chief_id, range_km, range_rate_km_s FROM relative_state ORDER BY range_km ASC LIMIT 1
```

Burn activity:

```sql
SELECT object_id, COUNT(*) AS samples, SUM(burn_active) AS active_samples, MAX(accel_norm_km_s2) AS max_accel_km_s2 FROM thrust GROUP BY object_id ORDER BY object_id
```

## Expected Answer Shape

- Status: validated and ran, or explain the failure.
- Initial range, final range, closest approach, and closest approach time.
- Burn activity and max applied acceleration by object.
- Explicit statement of whether the evidence supports terminal rendezvous,
  partial closure, or neither.
- Limitations and one focused next run if the result is inconclusive.

## Pass Criteria

- Config validates.
- Scenario runs headlessly.
- Review store contains relative-state and thrust evidence.
- Agent does not claim terminal rendezvous without threshold evidence.

## Red Flags

- Treats any distance reduction as rendezvous success.
- Ignores range rate, time, delta-v, or safety constraints.
- Invents controller performance not present in artifacts.
