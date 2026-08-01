# Closed-Loop Rendezvous Answer Example
Status: validated and ran.

Commands:

- `python run_simulation.py --config agents/examples/public_agent_rendezvous_lqr.yaml --validate-only`
- `python run_simulation.py --config agents/examples/public_agent_rendezvous_lqr.yaml`

Review queries:

```sql
SELECT metric_name, value, units, deputy_id, chief_id FROM metrics WHERE metric_name IN ('initial_range_km', 'final_range_km', 'closest_approach_km', 'closest_approach_time_s') ORDER BY metric_name
```

```sql
SELECT time_s, deputy_id, chief_id, range_km, range_rate_km_s FROM relative_state ORDER BY range_km ASC LIMIT 1
```

```sql
SELECT object_id, COUNT(*) AS samples, SUM(burn_active) AS active_samples, MAX(accel_norm_km_s2) AS max_accel_km_s2 FROM thrust GROUP BY object_id ORDER BY object_id
```

Outputs inspected:

- `outputs/agents/public_agent_rendezvous_lqr/index.md`
- `outputs/agents/public_agent_rendezvous_lqr/master_run_summary.json`
- `outputs/agents/public_agent_rendezvous_lqr/review/run.sqlite`

Evidence:

- The `metrics` table reports initial range, final range, closest approach,
  and closest approach time for the primary chaser/target pair.
- The `relative_state` query identifies the minimum range sample and range
  rate at closest approach.
- The `thrust` query reports whether each object had active applied
  acceleration samples.

Conclusion:

The run supports an evidence-backed closed-loop rendezvous assessment. It
should be described as terminal rendezvous only if the final range, range rate,
time, delta-v or burn activity, and any user-defined safety constraints meet a
stated acceptance threshold. Otherwise, describe it as partial closure or an
inconclusive approach result.

Limitations:

This is a short public smoke scenario using simplified public assumptions. It
is not flight software, mission qualification evidence, or an operational RPO
assessment.

Next run:

Ask the user for terminal success thresholds or add richer plots/logging if
trajectory shape, monotonic closure, or transient behavior matters.
