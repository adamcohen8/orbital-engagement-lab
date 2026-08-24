# Coverage And Directed Link Task Card

Task ID: `coverage_and_directed_link`

Install and activate OEL first using [`docs/installation.md`](../../docs/installation.md).

Example config: `examples/configs/public_coverage_and_link_analysis.yaml`

Expected output directory: `outputs/agent_tasks/coverage_link_review`

Answer example: `agents/tasks/examples/coverage_and_directed_link_answer.md`

## User Prompt

```text
Run the public coverage and directed-link example, inspect the coverage and
link evidence, create the standard plots, and explain what it does and does not
support.
```

## Expected Agent Assumptions

- Treat the analysis domain as experimental and evidence-only even though the
  recipe wrapper is supported.
- Use the declared attitude, cadence, grid, endpoint, and RF inputs exactly.
- Do not infer calibrated sensor performance, interference/weather
  availability, scheduling, packet delivery, probability, or external parity.

## Commands

```bash
python run_simulation.py --config examples/configs/public_coverage_and_link_analysis.yaml --validate-only
python -m sim.agent_task run coverage_link_review --output-root outputs/agent_tasks --plot
```

## Required Review Queries

```sql
SELECT scenario_name, duration_s, dt_s, samples FROM run_metadata
```

```sql
SELECT s.analysis_id, s.source_object_id, s.product_kind, MAX(c.instantaneous_covered_fraction) AS peak_covered_fraction FROM coverage_summary s LEFT JOIN coverage_samples c USING (analysis_id) GROUP BY s.analysis_id, s.source_object_id, s.product_kind ORDER BY s.analysis_id
```

```sql
SELECT s.analysis_id, s.tx_object_id, s.rx_object_id, MIN(l.margin_db) AS min_margin_db, MAX(l.margin_db) AS max_margin_db FROM link_summary s LEFT JOIN link_samples l USING (analysis_id) GROUP BY s.analysis_id, s.tx_object_id, s.rx_object_id ORDER BY s.analysis_id
```

## Expected Answer Shape

- Status, commands, queries, output directory, and generated plot paths.
- Declared propagation/attitude/cadence/grid/RF assumptions.
- Coverage fraction and link-margin/window evidence.
- Experimental domain posture and all required non-claims.

## Pass Criteria

- The config validates and the supported recipe completes.
- Coverage and link queries return evidence rows.
- Both standard review plots exist and are inspected.
- The conclusion stays within the deterministic evidence-only contract.

## Red Flags

- Calls the domain analysis calibrated or operationally assured.
- Treats sampled coverage as an exact swept footprint or probability.
- Infers weather, interference, scheduling, or packet delivery.
