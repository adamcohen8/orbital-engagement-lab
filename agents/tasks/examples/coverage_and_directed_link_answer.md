# Coverage And Directed Link Answer Example

Status: validated and ran.

Commands:

- `python run_simulation.py --config examples/configs/public_coverage_and_link_analysis.yaml --validate-only`
- `python -m sim.agent_task run coverage_link_review --output-root outputs/agent_tasks --plot`

Review queries:

```sql
SELECT scenario_name, duration_s, dt_s, samples FROM run_metadata
```

```sql
SELECT s.analysis_id, s.source_object_id, s.product_kind, MAX(c.instantaneous_covered_fraction) AS peak_covered_fraction FROM coverage_summary s LEFT JOIN coverage_samples c USING (analysis_id) GROUP BY s.analysis_id, s.source_object_id, s.product_kind ORDER BY s.analysis_id
```

```sql
SELECT s.analysis_id, s.tx_object_id, s.rx_object_id, MIN(l.margin_db) AS min_margin_db, MAX(l.margin_db) AS max_margin_db FROM link_summary s LEFT JOIN link_samples l USING (analysis_id) GROUP BY s.analysis_id, s.tx_object_id, s.rx_object_id ORDER BY s.analysis_id
```

Outputs inspected:

- `outputs/agent_tasks/coverage_link_review/agent_evidence_packet.json`
- `outputs/agent_tasks/coverage_link_review/review/run.sqlite`
- the generated coverage-fraction and directed-link-margin plots

Evidence:

The review tables record analysis identity, sampled coverage fraction,
directed endpoint identities, link margin, windows, and provenance.

Conclusion:

The deterministic run supports only the reported sampled geometric coverage
and free-space directed-link results for its declared inputs.

Limitations:

The experimental domain workflow does not establish calibrated payload
performance, exact swept footprints, weather/interference availability,
scheduling, packet delivery, probability, operational assurance, or
independent-tool parity.

Next run:

Change one declared orbit, attitude, cadence, grid, endpoint, or RF assumption
at a time and preserve the paired evidence boundary.
