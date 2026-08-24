# OGP-SGP4 Propagation Answer Example

Status: validated and ran.

Commands:

- `.venv/bin/python run_simulation.py --config agents/examples/public_agent_ogp_sgp4_propagation.yaml --validate-only`
- `.venv/bin/python -m sim.agent_task run ogp_sgp4_review --output-root outputs/agent_tasks --plot`

Review queries:

```sql
SELECT scenario_name, duration_s, dt_s, samples FROM run_metadata
```

```sql
SELECT p.object_id, p.propagation_method, p.general_model, p.native_frame, p.output_frame, p.frame_transform, f.state_frame, p.tle_epoch_jd_utc, p.tle_age_start_days, p.tle_age_end_days FROM object_propagation p LEFT JOIN object_state_frame f USING (object_id) ORDER BY p.object_id
```

```sql
SELECT object_id, time_s, pos_x_eci_km, pos_y_eci_km, pos_z_eci_km, vel_x_eci_km_s, vel_y_eci_km_s, vel_z_eci_km_s FROM object_state WHERE sample_index = (SELECT MAX(sample_index) FROM object_state) ORDER BY object_id
```

Outputs inspected:

- `outputs/agent_tasks/ogp_sgp4_review/agent_evidence_packet.json`
- `outputs/agent_tasks/ogp_sgp4_review/review/run.sqlite`
- `outputs/agent_tasks/ogp_sgp4_review/review/figures/evidence_object_eci_radius.png`

Evidence:

- `object_propagation` identifies continuous `general` propagation with the
  `sgp4` model and records the native/output frame plus frame transform.
- `object_state_frame` records the canonical frame used by the review history.
- `passive_final_state` records the final canonical ECI position and velocity.
- The plot is generated from the same read-only `object_state` evidence.

Conclusion:

The fixed historical TLE was propagated for the configured two-hour interval
through OEL's deterministic passive OGP-SGP4 path. The saved evidence supports
the propagation and frame-contract statement for this fixture.

Limitations:

This run does not establish current catalog freshness, operational ephemeris
accuracy, covariance, maneuver awareness, or mission suitability. The OGP
product's native frame and the canonical review-state frame must remain explicit.

Next run:

Replace the fixed fixture only when a trusted, appropriately sourced TLE and a
specific analysis epoch, duration, freshness requirement, and accuracy question
are available.
