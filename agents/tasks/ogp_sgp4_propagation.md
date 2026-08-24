# OGP-SGP4 Propagation Task Card

Task ID: `ogp_sgp4_propagation`

Example config: `agents/examples/public_agent_ogp_sgp4_propagation.yaml`

Expected output directory: `outputs/agent_tasks/ogp_sgp4_review`

Answer example: `agents/tasks/examples/ogp_sgp4_propagation_answer.md`

Command convention: activate OEL through [Installing OEL](../../docs/installation.md); commands below use portable `python` after activation.

## User Prompt

```text
Propagate a fixed public TLE for two hours with continuous passive OGP-SGP4.
Validate the scenario, run it, query the propagation provenance and final
state, create a position-history plot, and give me a bounded conclusion.
```

## Expected Agent Assumptions

- Use continuous passive OGP-SGP4, not TLE initialization followed by ONP.
- Use the checked-in historical TLE so the workflow remains offline and deterministic.
- Keep simulation-owned plots and animations disabled; generate the requested
  plot from completed review evidence.
- Treat the OGP product's native frame and the canonical review-state frame as
  separate evidence fields.
- Do not claim current-catalog freshness or operational ephemeris accuracy.

## Commands

Validate the scenario directly:

```bash
python run_simulation.py --config agents/examples/public_agent_ogp_sgp4_propagation.yaml --validate-only
```

Run the supported recipe and create its review-store plot:

```bash
python -m sim.agent_task run ogp_sgp4_review --output-root outputs/agent_tasks --plot
```

## Required Review Queries

Run metadata:

```sql
SELECT scenario_name, duration_s, dt_s, samples FROM run_metadata
```

Propagation and frame contract:

```sql
SELECT p.object_id, p.propagation_method, p.general_model, p.native_frame, p.output_frame, p.frame_transform, f.state_frame, p.tle_epoch_jd_utc, p.tle_age_start_days, p.tle_age_end_days FROM object_propagation p LEFT JOIN object_state_frame f USING (object_id) ORDER BY p.object_id
```

Final canonical state:

```sql
SELECT object_id, time_s, pos_x_eci_km, pos_y_eci_km, pos_z_eci_km, vel_x_eci_km_s, vel_y_eci_km_s, vel_z_eci_km_s FROM object_state WHERE sample_index = (SELECT MAX(sample_index) FROM object_state) ORDER BY object_id
```

## Expected Answer Shape

- Status: validated and ran, or explain the failure.
- Scenario duration, timestep, and fixed TLE epoch.
- Explicit evidence that `propagation_method` is `general` and the model is `sgp4`.
- Native/output frame, frame transform, and canonical review-state frame.
- Final ECI state and generated position-history plot path.
- A bounded conclusion and a clear statement that this fixed historical TLE
  workflow is not current or operational ephemeris evidence.

## Pass Criteria

- Config validates and runs headlessly.
- The evidence packet reports complete review queries and plot generation.
- `ogp_propagation_contract` reports `general` plus `sgp4`.
- `passive_final_state` returns one final canonical ECI state.
- `review/figures/evidence_object_eci_radius.png` exists.
- The answer distinguishes native OGP product frame from review-state frame.

## Red Flags

- Describes this as ONP propagation.
- Infers continuous OGP merely because a TLE is present without checking provenance.
- Treats TEME and ECI as interchangeable.
- Claims current orbit knowledge, operational accuracy, or validation beyond the saved evidence.
