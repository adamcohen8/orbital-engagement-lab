# Passive Propagation Task Card

Task ID: `passive_propagation`

Example config: `agents/examples/public_agent_single_satellite.yaml`

Expected output directory: `outputs/agents/public_agent_single_satellite`

Answer example: `agents/tasks/examples/passive_propagation_answer.md`

## User Prompt

```text
Create a simple public OEL scenario that propagates one passive satellite in a
7000 km circular orbit for 60 seconds. Keep it headless, validate it, run it,
and summarize the run from saved artifacts.
```

## Expected Agent Assumptions

- Use simple two-body dynamics.
- Use one enabled satellite object.
- Keep plots and animations disabled.
- Save JSON summary outputs.
- Enable standard review output.
- Do not add sensing, estimation, controllers beyond zero-control baselines, or
  higher-fidelity perturbations unless the user asks.

## Commands

Validate:

```bash
python run_simulation.py --config agents/examples/public_agent_single_satellite.yaml --validate-only
```

Run:

```bash
python run_simulation.py --config agents/examples/public_agent_single_satellite.yaml
```

## Required Review Queries

Run metadata:

```sql
SELECT scenario_name, duration_s, dt_s, samples FROM run_metadata
```

Final object state:

```sql
SELECT object_id, time_s, pos_x_eci_km, pos_y_eci_km, pos_z_eci_km, vel_x_eci_km_s, vel_y_eci_km_s, vel_z_eci_km_s FROM object_state WHERE sample_index = (SELECT MAX(sample_index) FROM object_state) ORDER BY object_id
```

## Expected Answer Shape

- Status: validated and ran, or explain the failure.
- Scenario name, duration, timestep, and active object.
- Dynamics model and control posture.
- Final state evidence from `object_state`.
- Output directory and artifacts inspected.
- Limitations: public educational smoke scenario, not mission validation.

## Pass Criteria

- Config validates.
- Scenario runs headlessly.
- `master_run_summary.json` exists.
- `review/run.sqlite` exists.
- The agent answers from artifacts or review queries.

## Red Flags

- Adds unrequested perturbations or controllers.
- Claims operational accuracy or validation beyond the run artifacts.
- Summarizes from memory without inspecting outputs.
