# Mission Reconstitution Trade Space Answer Example
Status: validated and ran.

Commands:

- `.venv/bin/python run_simulation.py --config agents/examples/public_agent_mission_reconstitution_trade_space.yaml --validate-only`
- `.venv/bin/python run_simulation.py --config agents/examples/public_agent_mission_reconstitution_trade_space.yaml`

Review queries:

```sql
SELECT object_id, goal, method, recovery_delta_v_m_s, recovery_time_s, propellant_kg, slot_recovery_time_s FROM mission_recovery_summary
```

```sql
SELECT candidate_id, object_id, goal, source, planned_delta_v_m_s, planned_time_s, feasible, verified FROM mission_recovery_candidates ORDER BY source, planned_delta_v_m_s, planned_time_s, candidate_id
```

```sql
SELECT candidate_id, burn_index, start_time_s, frame, axis, delta_v_m_s FROM mission_recovery_burns ORDER BY candidate_id, burn_index
```

```sql
SELECT object_id, state_label, a_km, ecc, inc_deg, raan_deg, argp_deg, true_anomaly_deg FROM mission_recovery_elements ORDER BY object_id, state_label
```

Outputs inspected:

- `outputs/agents/public_agent_mission_reconstitution_trade_space/index.md`
- `outputs/agents/public_agent_mission_reconstitution_trade_space/master_run_summary.json`
- `outputs/agents/public_agent_mission_reconstitution_trade_space/review/run.sqlite`

Evidence:

- The summary table reports the configured slot-recovery goal and the chosen
  recovery method.
- The candidate table compares the min-time, min-delta-v, and constrained
  planner options by planned time, planned delta-v, and feasibility flags.
- The burn table shows the planned burn sequence associated with each candidate.
- The recovery-elements table shows how the simple +I burn changed the orbit
  before reconstitution planning.

Conclusion:

The run supports a public deterministic trade-space screen: it compares saved
recovery candidates and exposes the time-vs-delta-v tradeoffs for the simple
slot-recovery setup.

Limitations:

This is not validated operational reconstitution planning. It does not prove
optimization optimality, operational feasibility, or robustness to uncertainty
or finite-burn execution limits.

Next run:

Add the real slot tolerance, time budget, and force-model fidelity if the user
needs a more decision-relevant reconstitution study.
