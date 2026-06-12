# Mission Reconstitution Trade Space Task Card

Task ID: `mission_reconstitution_trade_space`

Example config: `agents/examples/public_agent_mission_reconstitution_trade_space.yaml`

Expected output directory: `outputs/agents/public_agent_mission_reconstitution_trade_space`

Answer example: `agents/tasks/examples/mission_reconstitution_trade_space_answer.md`

## User Prompt

```text
Create a short public scenario that applies a simple +I in-track burn, then
uses the simulator-backed mission reconstitution trade space to compare min-time,
min-delta-v, and constrained recovery options. Validate, run, inspect the
review store, and summarize the tradeoffs.
```

## Expected Agent Assumptions

- Use the checked-in public mission-reconstitution example unless the user asks
  for a different public slot-recovery setup.
- Treat the result as a deterministic trade-space screen, not operational
  mission planning.
- Compare candidate time and delta-v tradeoffs from the saved review tables.
- State what the configured constraints do and do not cover.

## Commands

Validate:

```bash
.venv/bin/python run_simulation.py --config agents/examples/public_agent_mission_reconstitution_trade_space.yaml --validate-only
```

Run:

```bash
.venv/bin/python run_simulation.py --config agents/examples/public_agent_mission_reconstitution_trade_space.yaml
```

## Required Review Queries

Recovery summary:

```sql
SELECT object_id, goal, method, recovery_delta_v_m_s, recovery_time_s, propellant_kg, slot_recovery_time_s FROM mission_recovery_summary
```

Trade-space candidates:

```sql
SELECT candidate_id, object_id, goal, source, planned_delta_v_m_s, planned_time_s, feasible, verified FROM mission_recovery_candidates ORDER BY source, planned_delta_v_m_s, planned_time_s, candidate_id
```

Candidate burns:

```sql
SELECT candidate_id, burn_index, start_time_s, frame, axis, delta_v_m_s FROM mission_recovery_burns ORDER BY candidate_id, burn_index
```

Initial and final elements:

```sql
SELECT object_id, state_label, a_km, ecc, inc_deg, raan_deg, argp_deg, true_anomaly_deg FROM mission_recovery_elements ORDER BY object_id, state_label
```

## Expected Answer Shape

- Status: validated and ran, or explain the failure.
- Burn context and final-vs-initial orbit comparison.
- Min-time vs min-delta-v vs constrained candidate tradeoffs from the saved
  planner tables.
- Explicit statement that the artifact is a public deterministic trade-space
  screen, not validated operational planning.
- Limitations and one focused follow-up run.

## Pass Criteria

- Config validates before execution.
- Scenario runs headlessly.
- Review store contains mission-recovery candidate and burn tables.
- Agent compares saved candidate evidence rather than inventing tradeoffs.

## Red Flags

- Claims validated mission planning or operational feasibility.
- Ignores candidate feasibility/verification fields.
- Summarizes tradeoffs without inspecting the saved candidate tables.
