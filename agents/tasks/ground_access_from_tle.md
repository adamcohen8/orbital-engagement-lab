# Ground Access From TLE Task Card

Task ID: `ground_access_from_tle`

Example config: `agents/examples/public_agent_ground_access.yaml`

Expected output directory: `outputs/agents/public_agent_ground_access`

Answer example: `agents/tasks/examples/ground_access_from_tle_answer.md`

## User Prompt

```text
Create a short public scenario that initializes a satellite from a TLE and
checks visibility from Colorado Springs. Validate, run, query the review store,
and summarize access windows. Be explicit about whether this is SGP4.
```

## Expected Agent Assumptions

- Use the checked-in public ground-access example unless the user supplies a
  different public TLE or station.
- State that OEL uses the TLE to initialize an ECI state and then numerically
  integrates the configured OEL force model.
- Do not describe the run as SGP4/general-perturbations propagation.
- Inspect `ground_access` evidence.

## Commands

Validate:

```bash
.venv/bin/python run_simulation.py --config agents/examples/public_agent_ground_access.yaml --validate-only
```

Run:

```bash
.venv/bin/python run_simulation.py --config agents/examples/public_agent_ground_access.yaml
```

## Required Review Queries

Access sample summary:

```sql
SELECT station_id, object_id, COUNT(*) AS samples, SUM(access) AS access_samples, MIN(range_km) AS min_range_km, MAX(elevation_deg) AS max_elevation_deg FROM ground_access GROUP BY station_id, object_id ORDER BY station_id, object_id
```

No-access reasons:

```sql
SELECT station_id, object_id, reason, COUNT(*) AS samples FROM ground_access WHERE access = 0 GROUP BY station_id, object_id, reason ORDER BY station_id, object_id, samples DESC
```

## Expected Answer Shape

- Status: validated and ran, or explain the failure.
- Station ID, object ID, sample count, access sample count, min range, and max
  elevation.
- Explicit TLE assumption boundary: initializer only, not SGP4 propagation.
- Access/no-access interpretation from review evidence.
- Limitations: geometric access only, no RF link budget, scheduling, weather,
  or operational contact planning.

## Pass Criteria

- Config validates.
- Scenario runs headlessly.
- Review store contains ground-access rows.
- Agent avoids SGP4 claims.

## Red Flags

- Calls the result SGP4.
- Infers access windows without inspecting artifacts.
- Treats geometric visibility as communications availability.
