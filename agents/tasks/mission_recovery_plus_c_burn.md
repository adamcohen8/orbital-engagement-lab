# Mission Recovery +C Burn Task Card

Task ID: `mission_recovery_plus_c_burn`

Example config: `agents/examples/public_agent_mission_recovery_plus_c_burn.yaml`

Expected output directory: `outputs/agents/public_agent_mission_recovery_plus_c_burn`

Answer example: `agents/tasks/examples/mission_recovery_plus_c_burn_answer.md`

Command convention: activate OEL through [Installing OEL](../../docs/installation.md); commands below use portable `python` after activation.

## User Prompt

```text
Create a public scenario that applies a simple +C cross-track burn to one
satellite, propagates for ten minutes, validates and runs it, then inspects the
mission-recovery evidence. Tell me what changed in the orbit, what recovery the
artifact recommends, and what the result does not prove.
```

## Expected Agent Assumptions

- Use the checked-in public mission-recovery example unless the user asks for a
  different public burn geometry.
- Treat the run as a deterministic educational recovery study, not an
  operational recovery plan.
- Inspect saved mission-recovery tables instead of inferring from terminal logs.
- State the burn magnitude, final-vs-initial element comparison, and planner
  recommendation evidence.

## Commands

Validate:

```bash
python run_simulation.py --config agents/examples/public_agent_mission_recovery_plus_c_burn.yaml --validate-only
```

Run:

```bash
python run_simulation.py --config agents/examples/public_agent_mission_recovery_plus_c_burn.yaml
```

## Required Review Queries

Recovery summary:

```sql
SELECT object_id, goal, method, recovery_delta_v_m_s, recovery_time_s, propellant_kg, slot_recovery_time_s FROM mission_recovery_summary
```

Initial and final elements:

```sql
SELECT object_id, state_label, a_km, ecc, inc_deg, raan_deg, argp_deg, true_anomaly_deg FROM mission_recovery_elements ORDER BY object_id, state_label
```

Planner candidates:

```sql
SELECT candidate_id, object_id, goal, source, planned_delta_v_m_s, planned_time_s, feasible, verified FROM mission_recovery_candidates ORDER BY planned_delta_v_m_s, planned_time_s, candidate_id
```

Planned burns:

```sql
SELECT candidate_id, burn_index, start_time_s, frame, axis, delta_v_m_s FROM mission_recovery_burns ORDER BY candidate_id, burn_index
```

## Expected Answer Shape

- Status: validated and ran, or explain the failure.
- Burn context and total delta-v evidence.
- Initial-vs-final orbit comparison from `mission_recovery_elements`.
- Recovery recommendation from `mission_recovery_summary` and candidate tables.
- Explicit statement that this is a public deterministic recovery estimate, not
  validated mission planning.
- Limitations and one focused next run.

## Pass Criteria

- Config validates before execution.
- Scenario runs headlessly.
- Review store contains mission-recovery tables.
- Agent cites saved recovery evidence rather than guessing the recommendation.

## Red Flags

- Treats a two-body public recovery study as an operational plan.
- Ignores final-vs-initial orbit evidence.
- Reports a planner recommendation without querying the recovery tables.
