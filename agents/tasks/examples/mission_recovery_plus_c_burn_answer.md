# Mission Recovery +C Burn Answer Example
Status: validated and ran.

Commands:

- `.venv/bin/python run_simulation.py --config agents/examples/public_agent_mission_recovery_plus_c_burn.yaml --validate-only`
- `.venv/bin/python run_simulation.py --config agents/examples/public_agent_mission_recovery_plus_c_burn.yaml`

Review queries:

```sql
SELECT object_id, goal, method, recovery_delta_v_m_s, recovery_time_s, propellant_kg, slot_recovery_time_s FROM mission_recovery_summary
```

```sql
SELECT object_id, state_label, a_km, ecc, inc_deg, raan_deg, argp_deg, true_anomaly_deg FROM mission_recovery_elements ORDER BY object_id, state_label
```

```sql
SELECT candidate_id, object_id, goal, source, planned_delta_v_m_s, planned_time_s, feasible, verified FROM mission_recovery_candidates ORDER BY planned_delta_v_m_s, planned_time_s, candidate_id
```

```sql
SELECT candidate_id, burn_index, start_time_s, frame, axis, delta_v_m_s FROM mission_recovery_burns ORDER BY candidate_id, burn_index
```

Outputs inspected:

- `outputs/agents/public_agent_mission_recovery_plus_c_burn/index.md`
- `outputs/agents/public_agent_mission_recovery_plus_c_burn/master_run_summary.json`
- `outputs/agents/public_agent_mission_recovery_plus_c_burn/review/run.sqlite`

Evidence:

- The saved recovery summary reports the configured goal, recovery method,
  estimated recovery delta-v, and recovery time.
- The recovery-elements table shows the initial and final orbital elements used
  for the comparison after the +C burn.
- The candidate and burn tables expose the planner alternatives and the burns
  associated with each recommendation.

Conclusion:

The run supports a public deterministic mission-recovery assessment: the
artifacts show how the simple +C burn perturbed the orbit and what recovery
options the built-in planner recommends from saved evidence.

Limitations:

This is not validated mission planning. It does not prove operational
feasibility, finite-burn optimization quality, covariance robustness, or
real-world execution constraints.

Next run:

Ask for the real success metric, force-model fidelity, and planning constraints
before turning this evidence case into a more specific recovery study.
