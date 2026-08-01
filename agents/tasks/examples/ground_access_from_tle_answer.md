# Ground Access From TLE Answer Example
Status: validated and ran.

Commands:

- `python run_simulation.py --config agents/examples/public_agent_ground_access.yaml --validate-only`
- `python run_simulation.py --config agents/examples/public_agent_ground_access.yaml`

Review queries:

```sql
SELECT station_id, object_id, COUNT(*) AS samples, SUM(access) AS access_samples, MIN(range_km) AS min_range_km, MAX(elevation_deg) AS max_elevation_deg FROM ground_access GROUP BY station_id, object_id ORDER BY station_id, object_id
```

```sql
SELECT station_id, object_id, reason, COUNT(*) AS samples FROM ground_access WHERE access = 0 GROUP BY station_id, object_id, reason ORDER BY station_id, object_id, samples DESC
```

Outputs inspected:

- `outputs/agents/public_agent_ground_access/index.md`
- `outputs/agents/public_agent_ground_access/master_run_summary.json`
- `outputs/agents/public_agent_ground_access/review/run.sqlite`

Evidence:

- The `ground_access` table reports access samples, minimum range, and maximum
  elevation by station and object.
- No-access rows explain why samples failed the geometric access criteria.
- The scenario initializes from a TLE and then numerically integrates the
  configured OEL force model.

Conclusion:

The run supports a geometric access-window smoke assessment for the configured
station and object. It should not be described as OGP-SGP4/general-perturbations
propagation.

Limitations:

The access model is geometric. It does not model RF link budgets, weather,
station scheduling, command/telemetry availability, or operational contact
planning.

Next run:

Ask the user for the real TLE, station list, time horizon, and fidelity
expectations before making a more specific access claim.
