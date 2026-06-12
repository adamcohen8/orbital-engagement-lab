# Compare One Change Answer Example

Status: baseline validated and ran; modified case should be validated and run
after changing exactly one initial-separation parameter.

Commands:

- `.venv/bin/python run_simulation.py --config agents/examples/public_agent_rendezvous_lqr.yaml --validate-only`
- `.venv/bin/python run_simulation.py --config agents/examples/public_agent_rendezvous_lqr.yaml`
- `.venv/bin/python run_simulation.py --config <modified_config>.yaml --validate-only`
- `.venv/bin/python run_simulation.py --config <modified_config>.yaml`

Review queries:

```sql
SELECT metric_name, value, units, deputy_id, chief_id FROM metrics WHERE metric_name IN ('initial_range_km', 'final_range_km', 'closest_approach_km', 'closest_approach_time_s') ORDER BY metric_name
```

```sql
SELECT time_s, deputy_id, chief_id, r_radial_km, i_intrack_km, c_crosstrack_km, range_km, range_rate_km_s FROM relative_state ORDER BY time_s DESC LIMIT 1
```

```sql
SELECT object_id, COUNT(*) AS samples, SUM(burn_active) AS active_samples, MAX(accel_norm_km_s2) AS max_accel_km_s2 FROM thrust GROUP BY object_id ORDER BY object_id
```

Outputs inspected:

- `<baseline_output>/master_run_summary.json`
- `<baseline_output>/review/run.sqlite`
- `<modified_output>/master_run_summary.json`
- `<modified_output>/review/run.sqlite`

Evidence:

- The baseline and modified configs use distinct scenario names and output
  directories so outputs do not overwrite each other.
- The only intentional change is the initial in-track separation.
- Scenario name and output directory edits are bookkeeping, not physical
  scenario changes.
- Range metrics, final relative state, and burn activity are queried from both
  review stores.

Conclusion:

The comparison can support a narrow one-variable finding only when both runs
validate, both runs complete, and the evidence is compared from saved artifacts.
Do not attribute differences to controller changes, dynamics changes, or output
settings if those were not changed.

Limitations:

One comparison does not establish robustness. It is a local what-if study, not a
Monte Carlo campaign or controller benchmark.

Next run:

If the one-change result matters, run a small controlled sweep or define a
success gate before comparing additional separations.
