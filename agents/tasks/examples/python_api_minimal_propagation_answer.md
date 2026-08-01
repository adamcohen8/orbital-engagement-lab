# Python API Minimal Propagation Answer Example
Status: generated through the Python API, validated, and ran.

Commands:

- `python agents/examples/build_public_agent_python_api_minimal_propagation.py`
- `python run_simulation.py --config agents/examples/public_agent_python_api_minimal_propagation.yaml --validate-only`
- `python run_simulation.py --config agents/examples/public_agent_python_api_minimal_propagation.yaml`
- `python -m sim.review outputs/agents/public_agent_python_api_minimal_propagation --query "SELECT scenario_name, duration_s, samples FROM run_metadata"`

Review queries:

```sql
SELECT scenario_name, duration_s, samples FROM run_metadata
```

```sql
SELECT object_id, time_s, pos_x_eci_km, pos_y_eci_km, pos_z_eci_km, vel_x_eci_km_s, vel_y_eci_km_s, vel_z_eci_km_s FROM object_state WHERE sample_index = 0 ORDER BY object_id
```

```sql
SELECT artifact_type, artifact_id, path FROM artifacts ORDER BY artifact_type, artifact_id
```

Outputs inspected:

- `agents/examples/public_agent_python_api_minimal_propagation.yaml`
- `outputs/agents/public_agent_python_api_minimal_propagation/index.md`
- `outputs/agents/public_agent_python_api_minimal_propagation/master_run_summary.json`
- `outputs/agents/public_agent_python_api_minimal_propagation/review/run.sqlite`

Evidence:

- The run metadata identifies `public_agent_python_api_minimal_propagation`,
  `duration_s = 300.0`, and `samples = 31`.
- The first `object_state` row records the requested ECI initial state:
  position `[7000, 0, 0]` km and velocity `[0, 7.5, 0]` km/s.
- The output index reports nominal completion, no plots, and no animations.

Conclusion:

The run supports a minimal Python API authoring smoke result: an agent created
a scenario artifact through `ScenarioBuilder`, validated that artifact, ran it
through the deterministic simulator, and backed the answer with review-store
evidence.

Limitations:

This is a short public educational scenario. It does not validate operational
ephemeris accuracy, high-fidelity force modeling, mission success, sensing,
estimation, or controller behavior.

Next run:

Add the specific duration, force model, output plots, object name, or success
metric needed for the user's actual study.
