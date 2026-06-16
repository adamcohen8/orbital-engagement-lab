# OEL Agents

OEL Agents are instructions, examples, and checks that help AI coding agents
use Orbital Engagement Lab safely and productively. They are an orchestration
layer around documented OEL workflows. They do not replace the deterministic
physics engine, controllers, validators, or output writers.

The central promise is general help with OEL workflows, not completion of a
fixed example catalog. Examples, task cards, and answer keys exist to onboard
and evaluate agents; the durable interface is documented scenario YAML,
validation, deterministic execution, review evidence, generated artifacts, and
honest interpretation.

The intended user loop is:

```text
ask an agent -> generate or edit YAML -> validate -> run -> inspect artifacts -> evaluate
```

The product wedge is:

```text
natural-language request -> scenario YAML -> validation -> deterministic run
-> review-store query -> artifact-supported explanation
```

Agents orchestrate the workflow. OEL's deterministic simulator, validators,
output writers, and review store remain the authority for physics, controller
behavior, and reported results.

## Agent Roles

Public OEL Agents can:

- help users understand and modify public code;
- generate valid public scenario YAML from natural language;
- validate configs before execution;
- run public simulations through approved commands;
- explain public orbital mechanics models, equations, controllers, and outputs;
- support educational and game workflows;
- encourage transparent, contributor-friendly changes.

## How Coding Agents Should Use OEL

Codex, Cursor, Claude Code, Gemini CLI, and similar tools should begin with:

1. Read `AGENTS.md`.
2. Read `agents/public/AGENTS.md`.
3. Read the task-relevant docs, usually `docs/scenario-yaml.md`,
   `docs/quickstart.md`, `docs/python-api.md`, or `docs/game-mode-roadmap.md`.
4. Start from the nearest public config or example when it fits, or create a
   scoped new scenario when the user's request is different.
5. Validate any generated or edited scenario before running it.
6. Prefer `.venv/bin/python -m sim.review` when the run includes `review/run.sqlite`.
7. Summarize results only from generated artifacts.
8. Use `agents/public/evaluation-rubric.md` to judge whether the scenario
   supports the user's goal.

For evaluator-facing trials, use
[`agent-evaluation-packet.md`](agent-evaluation-packet.md). For evidence-backed
output inspection, use [`agent-review-queries.md`](agent-review-queries.md).
For the shortest reproducible adoption workflows, use
[`agent-golden-paths.md`](agent-golden-paths.md).
For a machine-readable validate/run/inspect fast lane that writes reusable
evidence packets, use [`agent-task-runner.md`](agent-task-runner.md).
For mapping broad user intents to workflows, starting docs, evidence, and
public-core limits, use
[`agent-capability-routing.md`](agent-capability-routing.md).
For public-safe upstream feedback discovered by agents, use
[`agent-feedback-loop.md`](agent-feedback-loop.md).
For repeatable public agent task checks, use
[`agent-task-cards.md`](agent-task-cards.md). Treat those cards as evaluation
fixtures, not as the limits of OEL Agent support.

Agents should keep changes scoped, cite files they changed or inspected, and
add smoke coverage when they create reusable examples.

## Canonical Commands

Validate environment:

```bash
.venv/bin/python run_simulation.py --doctor
```

Validate config:

```bash
.venv/bin/python run_simulation.py --config <scenario.yaml> --validate-only
```

Run scenario:

```bash
.venv/bin/python run_simulation.py --config <scenario.yaml>
```

Run public quickstart:

```bash
.venv/bin/python run_simulation.py --quickstart --validate-only
.venv/bin/python run_simulation.py --quickstart
```

Run smoke checks:

```bash
.venv/bin/python -m pytest sim/tests/test_oel_agents.py
.venv/bin/python -m pytest sim/tests/test_quickstart_5min.py
```

Run all agent-generated example checks:

```bash
.venv/bin/python -m pytest sim/tests/test_oel_agents.py
```

Generate public report artifacts:

```bash
.venv/bin/python run_simulation.py --quickstart
```

The public core writes deterministic Markdown, JSON, CSV, and plot artifacts.

Inspect outputs:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path

summary_path = Path("outputs/quickstart_5min/master_run_summary.json")
print(json.dumps(json.loads(summary_path.read_text()), indent=2))
PY
```

For a normal completed run, inspect `index.md` first, then
`master_run_summary.json`, CSV histories, plots, and any custom analysis files.
When the run has `outputs.review.enabled: true`, prefer the review query API
over ad hoc parsing of large logs:

```bash
.venv/bin/python -m sim.review outputs/my_run --query "SELECT scenario_name, duration_s FROM run_metadata"
.venv/bin/python -m sim.review outputs/my_run --query "SELECT time_s, range_km FROM relative_state LIMIT 20" --json
```

Common review queries are maintained in
[`agent-review-queries.md`](agent-review-queries.md). Agents should state the
query used when a conclusion depends on review-store evidence.

For table review, prefer `.venv/bin/python -m sim.review`. For custom brief or
report figures, use the OEL-styled review plotting API described in
[`agent-custom-plots.md`](agent-custom-plots.md):

```bash
.venv/bin/python -m sim.review plot outputs/my_run --recipe relative_velocity_components --style light
.venv/bin/python -m sim.review.plot outputs/my_run --sql "SELECT time_s, range_km FROM relative_state ORDER BY time_s" --x time_s --y range_km
.venv/bin/python run_evidence_studio.py --output outputs/my_run
```

Evidence Studio is an experimental local viewer/workbench for completed outputs;
do not treat it as the primary agent interface.

For repeatable agent handoffs, `sim.agent_task` can prepare review-enabled
scenario copies, validate/run bundled recipes, inspect completed outputs,
compare two configs, create standard review plots, and write
`agent_evidence_packet.json`:

```bash
.venv/bin/python -m sim.agent_task list
.venv/bin/python -m sim.agent_task run quickstart_review --output-root outputs/agent_tasks --json
.venv/bin/python -m sim.agent_task inspect outputs/quickstart_5min --query run_metadata --json
```

Treat this as an orchestration helper around documented workflows, not as a
replacement for scenario YAML, deterministic execution, or review-store
evidence.

## Output Freshness

Generated `outputs/` folders are derived evidence, not source of truth. Before
citing a completed run, prefer evidence from the run you just validated and
executed, or from a current task-runner `agent_evidence_packet.json`.

For checked-in public agent examples, use the canonical output roots under
`outputs/agents/<scenario_name>/`. Older local folders such as
`outputs/public_agent_*`, previous validation harness runs, or ad hoc
experiment folders may contain stale indexes, strict-JSON issues, or old
next-command hints. If a folder's freshness is unclear, rerun the source config
or inspect a freshly generated review store before drawing conclusions.

## Natural Requests

Users should not need special wording to use OEL Agents. They can ask in plain language,
and the agent should translate the request into the documented OEL workflow.

Example user requests:

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

## When The User Asks Something New

Agents should generalize from the documented OEL workflow:

```text
intent -> minimal scenario design -> YAML -> validate -> run
-> review query or artifact inspection -> evidence-backed answer
```

For new requests:

1. Restate the user goal as a small simulation study.
2. Identify the minimum objects, initial state, dynamics, controller posture,
   duration, outputs, and success evidence needed.
3. Use a public example as a scaffold only when it is genuinely close.
4. Create a distinct scenario name and output directory for new or modified
   scenarios.
5. Validate before execution, run only trusted configs, then inspect artifacts.
6. Explain what the artifacts support and what they do not support.

Do not force a new request into a task card just because a task card exists. If
the public core cannot answer the request, say so and point to the nearest
public alternative.

Use [`agent-capability-routing.md`](agent-capability-routing.md) when the
request is broader than an example. It maps common intents to public workflows,
evidence tables, clarifying-question triggers, and claims agents should avoid.

## Evaluation Fixtures

These tasks are recommended first checks for a new OEL-capable agent. They are
evaluation fixtures for the evidence loop, not the conceptual boundary of OEL
Agents. Each one should validate before running and summarize only from saved
artifacts.

1. Python API minimal propagation: use `ScenarioBuilder` to generate a small
   YAML artifact, validate and run it through the CLI, then query
   `run_metadata`.
2. Passive propagation: use
   `agents/examples/public_agent_single_satellite.yaml` to propagate one
   satellite and inspect `object_state` evidence.
3. Closed-loop rendezvous: use
   `agents/examples/public_agent_rendezvous_lqr.yaml` to inspect relative
   range, closest approach, burn events, and whether the run supports a
   terminal-rendezvous claim.
4. Mission recovery: use
   `agents/examples/public_agent_mission_recovery_plus_c_burn.yaml` to inspect
   a +C burn, final-vs-initial orbital elements, and recovery delta-v evidence.
5. Mission reconstitution trade space: use
   `agents/examples/public_agent_mission_reconstitution_trade_space.yaml` to
   compare min-time, min-delta-v, and constrained planner candidates from
   saved review evidence.
6. Ground access: use
   `agents/examples/public_agent_ground_access.yaml` to inspect access samples
   and state that TLE input initializes an OEL numerical propagation, not SGP4.
7. Attitude hold: use
   `agents/examples/public_agent_attitude_hold.yaml` to inspect body-rate and
   attitude-control evidence.
8. One-variable comparison: copy a nearby example, change one parameter, run
   both cases, and compare only metrics or histories present in artifacts.

The maintained card set for these tasks lives in
[`agent-task-cards.md`](agent-task-cards.md).
The first-run golden paths for minimal propagation, closed-loop rendezvous, and
mission recovery/reconstitution live in
[`agent-golden-paths.md`](agent-golden-paths.md).

The agent should choose simple public defaults when the request is
underspecified, write those defaults into the generated YAML, validate before
running, and state any assumptions in the result summary. The agent should ask a
clarifying question when a missing detail changes the study, and should not ask
about incidental details that are not needed for the requested workflow.

## Scenario Simplicity And Clarifying Questions

Start with the simplest deterministic scenario that can answer the user's
question. Add complexity only when the user asks for it or the requested study
requires it.

Ask a clarifying question for mission-shaping gaps:

- duration or time horizon,
- initial orbit, TLE, altitude, or relative state,
- passive vs controlled behavior,
- success metric or termination condition,
- fidelity level for requests such as "realistic", "high fidelity",
  "operational", "deorbit", "decay", or "access",
- whether the user wants plots, review-store inspection, or only summary
  outputs.

Default quietly for incidental choices:

- headless run,
- plots and animations off unless requested,
- attitude disabled unless attitude matters,
- no sensing or estimation unless observation uncertainty, tracking, access, or
  closed-loop knowledge is part of the request,
- no Monte Carlo, sensitivity, campaign, optimizer, or report workflow unless
  requested,
- simple dynamics first.

Do not silently enable J2, J3, J4, drag, SRP, third bodies, or high-fidelity
propagation. For example, if the user asks for a simple deorbit study, ask
whether they want a simple maneuver geometry case or a drag-including decay
study. If the user asks to propagate a satellite for two hours, use simple
propagation unless they ask for perturbations or realistic force modeling.

## Agent Feedback Loop

When an agent discovers OEL workflow friction while helping a user, it may ask
whether the user wants to submit public feedback. Feedback is useful for
missing examples, confusing validation errors, insufficient artifacts for a
reasonable question, review-store query gaps, docs ambiguity, or conflicts in
agent guidance.

Agents must not submit feedback silently. They should prepare a public-safe
draft, show the user what would be sent, ask for explicit approval, and then
open a GitHub Issue with the Agent Feedback template. See
[`agent-feedback-loop.md`](agent-feedback-loop.md).

Never include secrets, API keys, customer data, CUI, export-controlled data,
classified information, private configs, or private generated report packets in
public feedback. Use `SECURITY.md` for vulnerabilities or sensitive-data
exposure.

## Agent Example Cookbook

The `agents/examples/` configs are short, headless smoke scenarios designed for
agent-assisted creation, testing, and evaluation. Use them as scaffolds and
regression fixtures; create a scoped new YAML scenario when the user's request
does not match an example:

| Goal | Example | Validate |
| --- | --- | --- |
| Passive orbit propagation | `agents/examples/public_agent_single_satellite.yaml` | `.venv/bin/python run_simulation.py --config agents/examples/public_agent_single_satellite.yaml --validate-only` |
| Closed-loop rendezvous | `agents/examples/public_agent_rendezvous_lqr.yaml` | `.venv/bin/python run_simulation.py --config agents/examples/public_agent_rendezvous_lqr.yaml --validate-only` |
| Mission recovery +C burn | `agents/examples/public_agent_mission_recovery_plus_c_burn.yaml` | `.venv/bin/python run_simulation.py --config agents/examples/public_agent_mission_recovery_plus_c_burn.yaml --validate-only` |
| Mission reconstitution trade space | `agents/examples/public_agent_mission_reconstitution_trade_space.yaml` | `.venv/bin/python run_simulation.py --config agents/examples/public_agent_mission_reconstitution_trade_space.yaml --validate-only` |
| TLE ground-station access | `agents/examples/public_agent_ground_access.yaml` | `.venv/bin/python run_simulation.py --config agents/examples/public_agent_ground_access.yaml --validate-only` |
| Attitude hold | `agents/examples/public_agent_attitude_hold.yaml` | `.venv/bin/python run_simulation.py --config agents/examples/public_agent_attitude_hold.yaml --validate-only` |

These examples enable standard review output so agents can practice querying
`review/run.sqlite` after a successful run.

Run any validated example by removing `--validate-only`.

## Evaluating Rendezvous Runs

For a short headless rendezvous smoke test, `master_run_summary.json` can show
whether the run completed, which objects were active, initial/final range,
closest approach, delta-v, burn samples, detection rate, and termination status
when those fields are available.

Do not treat partial closure as terminal rendezvous. Define success thresholds
before the run, such as final range, closest approach, time allowed, delta-v
budget, keepout/safety constraints, and whether full-duration completion is
required.

Use richer outputs when the user needs to evaluate rendezvous quality rather
than just run validity:

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

Those artifacts and review tables help the agent inspect range history,
range-rate behavior, controller transients, burn timing, trajectory shape, and
whether closure was monotonic.

## Scenario Generation Pattern

When turning natural language into YAML:

1. Identify the scenario type: propagation, rendezvous, attitude hold, ground
   access, re-entry, game/training, or another documented public workflow.
2. Choose the nearest public config. If the user asks for a new scenario from
   scratch, use that config for structure but write a new YAML file with a
   distinct scenario name and output directory.
3. Keep units explicit and preserve OEL field names.
4. Prefer public controllers, presets, and examples.
5. Keep first drafts headless unless the user asks for plots or animation.
6. Use simple dynamics, no sensing/estimation, and no perturbation stack unless
   the request requires them.
7. Ask for mission-shaping missing details before writing YAML.
8. Use `outputs.stats.save_json: true` for summary JSON. Add
   `outputs.stats.save_full_log: true` only when the user needs detailed
   time-history review.
9. Validate the config.
10. Run the config only after validation passes.
11. Evaluate the run with `agents/public/evaluation-rubric.md`.

Example request:

> Propagate one passive satellite in a circular 7000 km orbit for one minute.

Example commands:

```bash
.venv/bin/python run_simulation.py --config agents/examples/public_agent_single_satellite.yaml --validate-only
.venv/bin/python run_simulation.py --config agents/examples/public_agent_single_satellite.yaml
```

## Safety And IP Guidance

Only run scenario YAML from trusted sources. OEL configs can reference
importable Python modules/classes for controllers, guidance, mission strategies,
mission execution, and other extension points.

Public agent instructions and examples should stay educational, inspectable,
and reproducible. Keep tuned parameters, optimizer traces, customer data,
generated validation evidence, API keys, and AI report packets out of public
commits.

## Public Placement

- Root `AGENTS.md`: public-safe defaults for common coding agents.
- `agents/public/AGENTS.md`: detailed public playbook.
- `agents/examples/`: public agent-generated YAML examples and smoke fixtures.
