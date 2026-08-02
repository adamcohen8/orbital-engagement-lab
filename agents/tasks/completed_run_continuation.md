# Completed-run Continuation Task Card

Task ID: `completed_run_continuation`

Example config: `agents/examples/public_agent_ground_access.yaml`

Expected output directory: `outputs/agents/public_agent_ground_access`

Answer example: `agents/tasks/examples/completed_run_continuation_answer.md`

Command convention: activate OEL through [Installing OEL](../../docs/installation.md); commands below use portable `python` after activation.

## User Prompt

```text
Continue the final satellite state from this completed OEL run for another ten
minutes. Preserve provenance, validate the new scenario, prove the handoff did
not change the state, and do not run the continuation until I approve it.
```

## Expected Agent Assumptions

- The source output is trusted and contains a standard review store.
- Select an object explicitly if more than one eligible object exists.
- Use `--sample final` only when the user asked for that exact selector.
- Require canonical ECI state and a recorded absolute `initial_jd_utc`.
- Treat the continuation as a new passive ONP study with a new scenario name
  and output directory.
- Materialization and parity comparison do not authorize execution.
- Continue translational state and matching full covariance only; do not claim
  controller, estimator, attitude, or mission-module memory continuity.

## Commands

Validate and run the source fixture:

```bash
python run_simulation.py --config agents/examples/public_agent_ground_access.yaml --validate-only
python run_simulation.py --config agents/examples/public_agent_ground_access.yaml
```

Export, materialize, and compare without executing the continuation:

```bash
python -m sim.handoff export-state outputs/agents/public_agent_ground_access \
  --object-id iss --sample final \
  --output outputs/handoffs/public_agent_ground_access_final.json --json
python -m sim.handoff materialize-onp \
  --state-product outputs/handoffs/public_agent_ground_access_final.json \
  --scenario-name public_agent_ground_access_continuation \
  --output outputs/handoffs/public_agent_ground_access_continuation.yaml \
  --run-output-dir outputs/agents/public_agent_ground_access_continuation \
  --duration-s 600 --dt-s 30 --trust-plugins --json
python -m sim.handoff compare-handoff \
  --product outputs/handoffs/public_agent_ground_access_final.json \
  --scenario outputs/handoffs/public_agent_ground_access_continuation.yaml \
  --output outputs/handoffs/public_agent_ground_access_continuation.comparison.json --json
```

## Required Review Queries

Confirm source-run timing and absolute epoch:

```sql
SELECT rm.scenario_name, rm.duration_s, rm.dt_s, rm.samples, oi.object_id, oi.initial_jd_utc FROM run_metadata rm JOIN object_initialization oi ON oi.object_id = 'iss'
```

Confirm the exact final source state selected for export:

```sql
SELECT object_id, sample_index, time_s, pos_x_eci_km, pos_y_eci_km, pos_z_eci_km, vel_x_eci_km_s, vel_y_eci_km_s, vel_z_eci_km_s FROM object_state WHERE object_id = 'iss' ORDER BY sample_index DESC LIMIT 1
```

## Expected Answer Shape

- Source-run status and exact object/sample selector.
- Product ID, source review/config hashes, frame, and derived UTC Julian-date
  epoch.
- Scenario and manifest paths, both validation outcomes, and
  `execution_occurred: false`.
- Comparison ID and passed/failed check count.
- A clear statement that the continuation has not run.
- Limits on state, covariance, and subsystem-memory continuity.

## Pass Criteria

- The source fixture validates and creates a review store.
- One unambiguous final ECI sample is exported with source hashes.
- The new ONP scenario passes safe validation and, when trusted, ordinary
  validation without execution.
- `compare-handoff` reports `status: equivalent` and zero failed checks.
- The response does not imply that materialization or comparison executed the
  continuation.

## Red Flags

- Selecting an object or sample implicitly when the evidence is ambiguous.
- Copying state values into YAML manually.
- Ignoring a source hash, frame, or epoch failure.
- Treating the source run as mutated or resumed in place.
- Claiming controller, estimator, attitude, or mission-module memory continuity.
- Running the continuation without separate authorization.
