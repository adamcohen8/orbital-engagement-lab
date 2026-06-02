# Attitude Hold Task Card

Task ID: `attitude_hold`

Example config: `agents/examples/public_agent_attitude_hold.yaml`

Expected output directory: `outputs/agents/public_agent_attitude_hold`

Answer example: `agents/tasks/examples/attitude_hold_answer.md`

## User Prompt

```text
Create a public attitude-hold scenario with one satellite starting with a
pointing error and body rates. Validate, run, inspect artifacts, and explain
what evidence supports the attitude-control result.
```

## Expected Agent Assumptions

- Use a public attitude controller.
- Keep the orbit problem simple.
- Keep plots and animations disabled unless the user asks for visual review.
- Inspect body-rate and state evidence from the review store.
- Do not claim high-fidelity ADCS validation.

## Commands

Validate:

```bash
python run_simulation.py --config agents/examples/public_agent_attitude_hold.yaml --validate-only
```

Run:

```bash
python run_simulation.py --config agents/examples/public_agent_attitude_hold.yaml
```

## Required Review Queries

First and final quaternion and angular rates:

```sql
WITH bounds AS (SELECT object_id, MIN(sample_index) AS first_i, MAX(sample_index) AS last_i FROM object_state GROUP BY object_id) SELECT s.object_id, s.time_s, s.quat_w, s.quat_x, s.quat_y, s.quat_z, s.omega_x_rad_s, s.omega_y_rad_s, s.omega_z_rad_s FROM object_state s JOIN bounds b ON s.object_id = b.object_id AND s.sample_index IN (b.first_i, b.last_i) ORDER BY s.object_id, s.sample_index
```

Optional orbital-thrust sanity check:

```sql
SELECT object_id, COUNT(*) AS samples, SUM(burn_active) AS active_samples, MAX(accel_norm_km_s2) AS max_accel_km_s2 FROM thrust GROUP BY object_id ORDER BY object_id
```

## Expected Answer Shape

- Status: validated and ran, or explain the failure.
- Active object and attitude controller.
- Initial and final quaternion and body-rate evidence.
- Note that `thrust` is orbital-acceleration evidence, not reaction-wheel
  torque or full ADCS telemetry.
- Statement about what the run does and does not prove.
- Limitations: simple public attitude-control smoke scenario, not ADCS
  qualification.

## Pass Criteria

- Config validates.
- Scenario runs headlessly.
- Review store contains object-state evidence.
- Agent does not overclaim attitude-controller validation.

## Red Flags

- Claims flight-qualified ADCS behavior.
- Ignores body-rate or state evidence.
- Adds unrequested orbital-control complexity.
