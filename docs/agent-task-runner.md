# Agent Task Runner

`sim.agent_task` is an optional orchestration layer for OEL-capable agents. It
packages common validate/run/inspect workflows into machine-readable recipes
and writes an `agent_evidence_packet.json` that can be cited, diffed, or handed
to another tool.

It does not replace scenario YAML, `run_simulation.py`, the deterministic
simulator, or `sim.review`. The runner prepares normal scenario YAML with
`outputs.review.enabled: true`, validates it through the public Python API,
runs it through the canonical execution service, queries the review store, and
saves the evidence it used.

## When To Use It

Use the task runner when an agent needs a repeatable, evidence-backed workflow:

- running a known public recipe and capturing the queries used;
- inspecting an existing output directory without ad hoc JSON parsing;
- comparing two scenario configs through the same review metrics;
- generating a standard plot from a completed review store.

For one-off scenario authoring, agents should still design the smallest
scenario that answers the user's question, validate it, run it, then inspect
the generated evidence. The task runner is a shortcut for common inspection and
handoff patterns, not a reason to force a user request into a canned example.

## Commands

List bundled recipes:

```bash
.venv/bin/python -m sim.agent_task list
.venv/bin/python -m sim.agent_task list --json
```

List named plot recipes:

```bash
.venv/bin/python -m sim.agent_task list --plots
```

Inspect semantic metric definitions:

```bash
.venv/bin/python -m sim.agent_task semantics
.venv/bin/python -m sim.agent_task semantics closest_approach_km --json
```

Run a recipe and write an evidence packet:

```bash
.venv/bin/python -m sim.agent_task run quickstart_review --output-root outputs/agent_tasks --json
```

Validate a recipe without executing it:

```bash
.venv/bin/python -m sim.agent_task run quickstart_review --output-root outputs/agent_tasks --dry-run
```

Inspect a completed output directory:

```bash
.venv/bin/python -m sim.agent_task inspect outputs/quickstart_5min \
  --query run_metadata \
  --query rendezvous_closest_approach \
  --json
```

When inspecting an existing directory, treat the packet as evidence for that
directory's current contents, not proof that the source scenario still produces
the same result. If a local output may be stale, rerun the recipe or scenario
before citing metrics.

Compare two configs through common review metrics:

```bash
.venv/bin/python -m sim.agent_task compare \
  --base configs/quickstart_5min.yaml \
  --candidate configs/ric_pd_10km_experiment.yaml \
  --output-dir outputs/agent_tasks/quickstart_vs_flagship \
  --metric closest_approach_km \
  --metric final_range_km \
  --json
```

Create a standard plot from a completed review store:

```bash
.venv/bin/python -m sim.agent_task plot outputs/quickstart_5min \
  --recipe relative_range \
  --style oel_light \
  --format png
```

## Evidence Packet

Each run, inspect, or compare command writes:

```text
<output-dir>/agent_evidence_packet.json
```

The packet includes:

- task id, status, generated UTC timestamp, and packet schema version;
- recipe metadata when a bundled recipe was used;
- prepared scenario config paths and output directories;
- validation reports;
- run summaries when execution occurred;
- review-store table/schema metadata;
- saved review queries, SQL, columns, row samples, and truncation status;
- semantic metric definitions used by the task;
- generated plot/artifact paths;
- structured failure hints when validation, review queries, or artifact
  inspection fail.

Agents should cite the packet path and the query name or SQL used when a claim
depends on run evidence.

## Bundled Recipes

Current bundled task recipes include:

- `quickstart_review`: public quickstart scenario with rendezvous metrics,
  closest approach, burn activity, and artifact inventory.
- `flagship_ric_pd_review`: flagship 10 km RIC PD scenario with terminal
  relative-state and burn evidence.
- `mission_reconstitution_review`: public mission-reconstitution trade-space
  example with recovery summary, candidates, and burn sequence evidence.
- `ground_access_review`: public ground-access example with station/object
  access summaries and no-access reason counts.

## Plot Recipes

Current bundled plot recipes include:

- `relative_range`
- `relative_range_rate`
- `burn_activity`
- `campaign_closest_approach`
- `sensitivity_effects`

Plot recipes run read-only review SQL and call the same review plotting service
used by OEL Evidence Studio. Generated plot provenance is recorded in
`review/generated_artifacts.json`.

## Failure Hints

The runner emits structured hints for common agent repair loops, including:

- plugin validation failures;
- missing review stores;
- review schema/query mismatches;
- YAML loading errors;
- invalid timing configuration;
- absent `review/run.sqlite` after execution.

Hints are guidance for the next deterministic repair step. They are not
simulation evidence by themselves.
