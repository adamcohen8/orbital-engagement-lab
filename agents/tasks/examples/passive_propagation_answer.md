# Passive Propagation Answer Example
Status: validated and ran.

Commands:

- `.venv/bin/python run_simulation.py --config agents/examples/public_agent_single_satellite.yaml --validate-only`
- `.venv/bin/python run_simulation.py --config agents/examples/public_agent_single_satellite.yaml`

Review queries:

```sql
SELECT scenario_name, duration_s, dt_s, samples FROM run_metadata
```

```sql
SELECT object_id, time_s, pos_x_eci_km, pos_y_eci_km, pos_z_eci_km, vel_x_eci_km_s, vel_y_eci_km_s, vel_z_eci_km_s FROM object_state WHERE sample_index = (SELECT MAX(sample_index) FROM object_state) ORDER BY object_id
```

Outputs inspected:

- `outputs/agents/public_agent_single_satellite/index.md`
- `outputs/agents/public_agent_single_satellite/master_run_summary.json`
- `outputs/agents/public_agent_single_satellite/review/run.sqlite`

Evidence:

- The run metadata identifies `public_agent_single_satellite`, the configured
  duration, timestep, and sample count.
- The final `object_state` row gives the propagated ECI position and velocity
  for `target`.
- The scenario uses two-body orbit dynamics with zero orbit and attitude
  control.

Conclusion:

The run supports a simple passive-propagation smoke result: one satellite was
propagated through the deterministic OEL engine and saved queryable evidence.

Limitations:

This is a short public educational scenario. It does not validate mission
accuracy, high-fidelity force modeling, stationkeeping, sensing, estimation, or
operational readiness.

Next run:

Add the specific duration, force model, plots, or success metric the user needs
for a more meaningful study.
