# Public OEL Agents

Use this playbook when an AI coding agent is helping with the open-source
Orbital Engagement Lab core.

## Mission

Public OEL Agents make the project easier to understand, modify, and run. They
can help users generate valid public scenario YAML, validate configs, run
approved simulations, inspect outputs, explain public models/controllers, and
support educational or game workflows.

Agents should be transparent contributors. They should cite the files or docs
they used, prefer public examples over invention, and leave deterministic
simulation and reporting to OEL itself.

Examples are rails, not boundaries. Use them when they fit the user's request,
but do not reduce OEL Agents to a fixed example catalog. The durable agent
interface is the documented workflow: scenario YAML, validation, deterministic
runs, review queries, artifacts, tests, and honest limits.

## Supported Workflows

1. Read the relevant docs and examples.
2. Draft or edit scenario YAML.
3. Validate the config before execution.
4. Run the scenario through `run_simulation.py`.
5. Inspect generated `index.md`, `master_run_summary.json`, CSV, and plot
   artifacts.
6. Use `.venv/bin/python -m sim.review` when a run includes `review/run.sqlite`.
7. Evaluate the run with `agents/public/evaluation-rubric.md`.
8. Summarize results from saved artifacts, not from memory or speculation.
9. Add tests or smoke checks for new agent-facing examples.

The preferred agent evidence loop is:

```text
ordinary-language request -> scenario YAML -> validate -> run
-> review query or artifact inspection -> evidence-backed answer
```

Use `docs/agent-evaluation-packet.md` to evaluate whether an agent follows this
loop. Use `docs/agent-capability-routing.md` to map broad user intents to
workflows, starting docs, evidence, and public-core limits. Use
`docs/agent-review-queries.md` for reusable review-store SQL. Use
`docs/agent-feedback-loop.md` when an agent finds public-safe workflow feedback
worth sending upstream. Use `docs/agent-task-cards.md` for the repeatable
public agent task-card set; task cards are evaluation fixtures, not the
boundary of what agents can help users do.

## Canonical Commands

Validate environment:

```bash
.venv/bin/python run_simulation.py --doctor
```

Validate config:

```bash
.venv/bin/python run_simulation.py --config configs/automation_smoke.yaml --validate-only
```

Run scenario:

```bash
.venv/bin/python run_simulation.py --config configs/automation_smoke.yaml
```

Run smoke test:

```bash
.venv/bin/python -m pytest sim/tests/test_oel_agents.py
```

Generate a report-like public artifact set:

```bash
.venv/bin/python run_simulation.py --quickstart
```

Inspect outputs:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path

summary = Path("outputs/quickstart_5min/master_run_summary.json")
print(json.dumps(json.loads(summary.read_text()), indent=2)[:4000])
PY
```

Public AI report generation is not a default open-source workflow. In the
public core, treat `index.md`, JSON summaries, CSV files, plots, and game
debriefs as report artifacts.

Inspect review stores when enabled:

```bash
.venv/bin/python -m sim.review outputs/my_run --query "SELECT scenario_name, duration_s FROM run_metadata"
.venv/bin/python -m sim.review outputs/my_run --query "SELECT time_s, range_km FROM relative_state LIMIT 20" --json
```

When using the review store, cite the SQL query or saved view that supports the
answer. Keep queries read-only, prefer `SELECT`/`WITH`, and inspect table names
or sample rows before writing custom SQL against an unfamiliar table:

```bash
.venv/bin/python -m sim.review outputs/my_run --query "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name" --json
.venv/bin/python -m sim.review outputs/my_run --query "SELECT * FROM object_state LIMIT 1" --json
```

You can also inspect table names with the Python API:

```python
from sim.review import ReviewWorkspace

workspace = ReviewWorkspace.open("outputs/my_run")
print(workspace.tables())
print(workspace.schema()["columns"].keys())
```

Use review tables for time-history questions, range closure, burn activity,
ground-access checks, metrics, and artifact inventories. If `review/run.sqlite`
is missing, fall back to `index.md`, `master_run_summary.json`, CSV histories,
and plots without claiming structured review evidence exists.

The Output Review Workbench is an experimental preview and is not currently
recommended for routine agent review. Prefer `.venv/bin/python -m sim.review`. Use ORW
only when the user explicitly asks for the interactive local workbench:

```bash
.venv/bin/python run_orw.py --output outputs/my_run
```

## Scenario Generation Rules

- Start with the simplest deterministic scenario that can answer the user's
  question. Do not add unrequested physics, sensors, estimators, controllers,
  plots, animations, or campaign machinery.
- Start from a nearby public example whenever possible. When the user asks for
  a new scenario from scratch, use examples for structure but write a new YAML
  file with a distinct scenario name and output directory.
- Use `objects:` entries for spacecraft instead of legacy top-level `chaser`
  or `target` sections.
- Keep units explicit in field names: `_km`, `_s`, `_deg`, `_rad_s`,
  `_km_s`, `_kg`, `_n`, and similar suffixes.
- Keep first drafts short, headless, and deterministic: plots off, animations
  off, modest duration, and public controllers.
- Use simple dynamics first. Do not enable J2, J3, J4, drag, SRP, third bodies,
  high-fidelity propagation, sensing, estimation, Monte Carlo, sensitivity, or
  reports unless the user asks for them or they are necessary for the stated
  study.
- For JSON outputs, use `outputs.stats.save_json: true` for the compact
  `master_run_summary.json`. Set `outputs.stats.save_full_log: true` only when
  the user needs detailed time-history review in `master_run_log.json`.
- Validate every generated config with `--validate-only`.
- Run only trusted YAML. Scenario plugin pointers can import Python code.

Ask a clarifying question when a missing detail changes the study: duration,
initial orbit/TLE/relative state, passive vs controlled behavior, success
metric, termination condition, or fidelity level. Do not ask about incidental
details that are not needed for the requested workflow. For example, do not ask
about sensing/estimation unless observation uncertainty, tracking, access, or
closed-loop knowledge is part of the request.

For deorbit, decay, access, or "realistic" requests, clarify the intended
fidelity instead of silently enabling every force model. A simple deorbit
maneuver geometry demo and a drag-including orbital decay study are different
configs.

## When The User Asks Something New

Agents should generalize from OEL's documented interfaces, not from memorized
example outcomes. For a request that is not exactly one of the checked-in
examples:

1. Translate the request into a small OEL study: objects, initial state,
   dynamics, controller posture, duration, outputs, and success evidence.
2. Decide whether an existing public example is close enough to copy, or
   whether a new scenario YAML file is clearer.
3. Keep only the complexity required by the user's goal.
4. Validate the generated config before execution.
5. Run the deterministic simulator and inspect saved artifacts.
6. Use `.venv/bin/python -m sim.review` when a review store exists.
7. State assumptions, missing evidence, and model limits in the answer.

If no public-core workflow can answer the request, say so and point to the
nearest public alternative instead of pretending an example proves more than it
does.

For common routing decisions, use `docs/agent-capability-routing.md` before
choosing an example. It lists workflow evidence and clarifying-question
triggers for propagation, TLEs, rendezvous, access, attitude, plotting, game
training, comparison, validation, sealed mode, and public-core boundaries.

## Rendezvous Evaluation Notes

For a closed-loop rendezvous scenario, compact summary JSON is enough for a
first smoke check when it includes initial range, closest approach, final
range, delta-v, burn samples, detection rate, and termination status. Treat
"validated and ran" as different from "rendezvous succeeded."

Before claiming success, define the user's range, time, delta-v, and safety
thresholds. A run that closes distance but ends kilometers away demonstrates
controlled approach, not terminal rendezvous.

Enable richer artifacts when the user needs trajectory-quality review:

```yaml
outputs:
  stats:
    save_json: true
    save_csv: true
    save_full_log: true
  review:
    enabled: true
    detail: standard
  plots:
    enabled: true
    figure_ids: ["relative_range", "trajectory_ric_curv_2d_multi", "control_effort"]
```

Use those richer artifacts to inspect range history, range-rate behavior,
controller transients, burn timing, trajectory shape, and whether closure was
monotonic or merely improved at the final sample.

## Example Rails

Request:

> Propagate one passive satellite in a 7000 km circular, 45 degree orbit for
> one minute using two-body dynamics, no plots, and saved JSON stats.

Use:

```bash
.venv/bin/python run_simulation.py --config agents/examples/public_agent_single_satellite.yaml --validate-only
.venv/bin/python run_simulation.py --config agents/examples/public_agent_single_satellite.yaml
```

Request:

> Give me a quick closed-loop rendezvous example with public controllers.

Use:

```bash
.venv/bin/python run_simulation.py --config examples/configs/public_closed_loop_rendezvous_lqr.yaml --validate-only
.venv/bin/python run_simulation.py --config examples/configs/public_closed_loop_rendezvous_lqr.yaml
```

Request:

> Help me practice manual RPO.

Use:

```bash
.venv/bin/python run_game.py examples/configs/public_manual_rpo_training.yaml
```

## Natural User Requests

Users should be able to ask for OEL work in ordinary language. Do not require
them to provide a structured template. Interpret the request, choose the
nearest public workflow, create or edit YAML, validate, run, inspect outputs,
and evaluate the result.

Examples of acceptable user requests:

- "Make me a simple satellite propagation scenario and tell me what happened."
- "Create a short rendezvous case where the chaser starts 3 km behind the
  target."
- "Can you check when this TLE is visible from Colorado Springs?"

For TLE requests, say explicitly that OEL uses TLE lines to initialize an ECI
state and then runs configured OEL numerical propagation. Do not describe the
result as SGP4/general-perturbations propagation unless an SGP4 workflow is
actually added.
- "Build an attitude-hold scenario with an initial pointing error."
- "Evaluate the run in this output folder and tell me whether it supports my
  goal."

If the request leaves out necessary details, make conservative public defaults
visible in the generated YAML and explain them in the result summary. Ask a
clarifying question only when a reasonable public default would be misleading
or risky.

## Agent-Generated Example Configs

These examples are intentionally short and headless so users can validate,
run, and inspect them quickly. They are useful starting points and regression
fixtures, not a complete list of supported agent workflows:

```bash
.venv/bin/python run_simulation.py --config agents/examples/public_agent_single_satellite.yaml --validate-only
.venv/bin/python run_simulation.py --config agents/examples/public_agent_rendezvous_lqr.yaml --validate-only
.venv/bin/python run_simulation.py --config agents/examples/public_agent_mission_recovery_plus_c_burn.yaml --validate-only
.venv/bin/python run_simulation.py --config agents/examples/public_agent_mission_reconstitution_trade_space.yaml --validate-only
.venv/bin/python run_simulation.py --config agents/examples/public_agent_ground_access.yaml --validate-only
.venv/bin/python run_simulation.py --config agents/examples/public_agent_attitude_hold.yaml --validate-only
```

These examples enable standard review output. After running one, inspect
the configured output directory's `review/run.sqlite` with
`.venv/bin/python -m sim.review` and cite the query used for any important
metric.

For a repeatable evaluation set around these examples, use the task cards in
`docs/agent-task-cards.md`.

## Explanation Rules

- Explain public equations, frames, controls, outputs, and limitations from
  checked-in docs and source.
- Distinguish model behavior from real mission behavior.
- Do not claim validation beyond the artifacts and tests that actually exist.
- When a result matters, point the user to the generated output file and the
  command that produced it.

## Agent Feedback Loop

Agents may help users submit public-safe feedback about OEL itself when a
workflow problem appears during normal use. Feedback-worthy issues include
missing examples, confusing validation messages, docs gaps, review-store
limitations, insufficient output artifacts, or conflicts in agent guidance.

Follow `docs/agent-feedback-loop.md`:

- prepare a short public-safe feedback draft,
- show the user what would be sent,
- ask for explicit approval,
- submit only after approval using the GitHub Agent Feedback issue template,
- link the issue back to the user.

Do not use public feedback for vulnerabilities, secrets, customer data, CUI,
export-controlled data, classified information, private configs, or private
generated report packets. Use the private `SECURITY.md` process instead.

## Safety And IP Guidance

- Do not commit API keys, local customer data, generated AI report packets, or
  non-public outputs.
- Do not copy tuned parameters, optimizer traces, customer material, or
  generated validation evidence into public examples.
- Keep public examples educational, inspectable, and reproducible.
- If a user asks for capability that is not present in the public core,
  redirect to documented public CLI/API behavior and examples.
