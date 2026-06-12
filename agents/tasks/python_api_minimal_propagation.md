# Python API Minimal Propagation Task Card

Task ID: `python_api_minimal_propagation`

Example config: `agents/examples/public_agent_python_api_minimal_propagation.yaml`

Expected output directory: `outputs/agents/public_agent_python_api_minimal_propagation`

Answer example: `agents/tasks/examples/python_api_minimal_propagation_answer.md`

## User Prompt

```text
Create a minimal single-satellite propagation scenario using the Python API,
not by hand-writing shortcut physics. Use ScenarioBuilder if available. The
scenario should run for 120 seconds with a 10 second step, one satellite at
position_eci_km [7000, 0, 0] and velocity_eci_km_s [0, 7.5, 0], review output
enabled, plots and animations off. Validate it, write the YAML artifact, run it
through the documented simulator CLI with --validate-only first, then run it,
query the review store for scenario_name, duration_s, and samples, and report
the exact commands and evidence.
```

## Expected Agent Assumptions

- Use `ScenarioBuilder` from the public Python API.
- Produce scenario YAML as an artifact, then validate and run that artifact
  through `run_simulation.py`.
- Use simple deterministic propagation with one satellite object.
- Keep plots and animations disabled.
- Enable standard review output.
- Do not invent alternate orbital mechanics or bypass OEL's simulator.

## Commands

Generate the YAML artifact:

```bash
.venv/bin/python - <<'PY'
from sim import ScenarioBuilder

artifact = (
    ScenarioBuilder("public_agent_python_api_minimal_propagation")
    .duration(120.0, dt_s=10.0)
    .target_satellite(
        mass_kg=300.0,
        position_eci_km=[7000.0, 0.0, 0.0],
        velocity_eci_km_s=[0.0, 7.5, 0.0],
    )
    .outputs(
        "outputs/agents/public_agent_python_api_minimal_propagation",
        stats={"print_summary": False, "save_json": True, "save_full_log": False},
    )
    .review(detail="standard")
    .artifact()
)

report = artifact.validate_report()
if not report.ok:
    raise SystemExit(report.to_dict())
artifact.write("agents/examples/public_agent_python_api_minimal_propagation.yaml")
PY
```

Validate:

```bash
.venv/bin/python run_simulation.py --config agents/examples/public_agent_python_api_minimal_propagation.yaml --validate-only
```

Run:

```bash
.venv/bin/python run_simulation.py --config agents/examples/public_agent_python_api_minimal_propagation.yaml
```

Query:

```bash
.venv/bin/python -m sim.review outputs/agents/public_agent_python_api_minimal_propagation --query "SELECT scenario_name, duration_s, samples FROM run_metadata"
```

## Required Review Queries

Run metadata:

```sql
SELECT scenario_name, duration_s, samples FROM run_metadata
```

Initial propagated state:

```sql
SELECT object_id, time_s, pos_x_eci_km, pos_y_eci_km, pos_z_eci_km, vel_x_eci_km_s, vel_y_eci_km_s, vel_z_eci_km_s FROM object_state WHERE sample_index = 0 ORDER BY object_id
```

Artifact inventory:

```sql
SELECT artifact_type, artifact_id, path FROM artifacts ORDER BY artifact_type, artifact_id
```

## Expected Answer Shape

- Status: generated via Python API, validated, ran, or explain the failure.
- Scenario name, duration, timestep, sample count, and active object.
- YAML artifact path and output directory.
- Exact validate, run, and review-query commands used.
- Review evidence for `scenario_name`, `duration_s`, and `samples`.
- Limitations: deterministic public smoke scenario, not operational validation.

## Pass Criteria

- Uses `ScenarioBuilder` or clearly explains why it is unavailable.
- Writes a YAML artifact and runs the CLI against that artifact.
- Validates with `--validate-only` before running.
- Produces `review/run.sqlite`.
- Reports evidence from `run_metadata`, not logs or memory.

## Red Flags

- Hand-rolls physics outside OEL.
- Skips CLI validation after writing YAML.
- Parses terminal logs instead of querying `review/run.sqlite`.
- Guesses review-store columns when `docs/agent-review-queries.md` or
  `SELECT * ... LIMIT 1` would show the available columns.
