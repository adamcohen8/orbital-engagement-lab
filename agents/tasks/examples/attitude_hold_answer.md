# Attitude Hold Answer Example

Status: validated and ran.

Commands:

- `python run_simulation.py --config agents/examples/public_agent_attitude_hold.yaml --validate-only`
- `python run_simulation.py --config agents/examples/public_agent_attitude_hold.yaml`

Review queries:

```sql
WITH bounds AS (SELECT object_id, MIN(sample_index) AS first_i, MAX(sample_index) AS last_i FROM object_state GROUP BY object_id) SELECT s.object_id, s.time_s, s.quat_w, s.quat_x, s.quat_y, s.quat_z, s.omega_x_rad_s, s.omega_y_rad_s, s.omega_z_rad_s FROM object_state s JOIN bounds b ON s.object_id = b.object_id AND s.sample_index IN (b.first_i, b.last_i) ORDER BY s.object_id, s.sample_index
```

```sql
SELECT object_id, COUNT(*) AS samples, SUM(burn_active) AS active_samples, MAX(accel_norm_km_s2) AS max_accel_km_s2 FROM thrust GROUP BY object_id ORDER BY object_id
```

Outputs inspected:

- `outputs/agents/public_agent_attitude_hold/index.md`
- `outputs/agents/public_agent_attitude_hold/master_run_summary.json`
- `outputs/agents/public_agent_attitude_hold/review/run.sqlite`

Evidence:

- The first/final `object_state` rows report quaternion and body angular-rate
  components.
- The scenario uses a public reaction-wheel PD attitude controller.
- The applied-acceleration summary is only an orbital-thrust sanity check; it is
  not reaction-wheel torque telemetry.

Conclusion:

The run supports a public attitude-control smoke assessment based on saved
state evidence. Any claim about attitude convergence should be tied to the
observed body rates, attitude fields, and configured controller target.

Limitations:

This is not ADCS qualification evidence. It does not establish flight
readiness, hardware performance, actuator margins, or high-fidelity disturbance
model validity.

Next run:

Enable plots or richer attitude diagnostics if the user needs pointing-error
history, settling time, or actuator-limit analysis.
