# OEL Agents

OEL Agents are instructions, examples, and checks that help AI coding agents
use Orbital Engagement Lab safely and productively. They are an orchestration
layer around documented OEL workflows. They do not replace the deterministic
physics engine, controllers, validators, or output writers.

The intended user loop is:

```text
ask an agent -> generate or edit YAML -> validate -> run -> inspect artifacts -> evaluate
```

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
4. Start from the nearest public config or example.
5. Validate any generated or edited scenario before running it.
6. Prefer `python -m sim.review` when the run includes `review/run.sqlite`.
7. Summarize results only from generated artifacts.
8. Use `agents/public/evaluation-rubric.md` to judge whether the scenario
   supports the user's goal.

Agents should keep changes scoped, cite files they changed or inspected, and
add smoke coverage when they create reusable examples.

## Canonical Commands

Validate environment:

```bash
python run_simulation.py --doctor
```

Validate config:

```bash
python run_simulation.py --config <scenario.yaml> --validate-only
```

Run scenario:

```bash
python run_simulation.py --config <scenario.yaml>
```

Run public quickstart:

```bash
python run_simulation.py --quickstart --validate-only
python run_simulation.py --quickstart
```

Run smoke checks:

```bash
python -m pytest sim/tests/test_oel_agents.py
python -m pytest sim/tests/test_quickstart_5min.py
```

Run all agent-generated example checks:

```bash
python -m pytest sim/tests/test_oel_agents.py
```

Generate public report artifacts:

```bash
python run_simulation.py --quickstart
```

The public core writes deterministic Markdown, JSON, CSV, and plot artifacts.

Inspect outputs:

```bash
python - <<'PY'
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
python -m sim.review outputs/my_run --query "SELECT scenario_name, duration_s FROM run_metadata"
python -m sim.review outputs/my_run --query "SELECT time_s, range_km FROM relative_state LIMIT 20" --json
```

For agent review, prefer `python -m sim.review`. The Output Review Workbench is
an experimental preview and is not currently recommended for routine review.
Use it only when the user explicitly asks for an interactive local workbench:

```bash
python run_orw.py --output outputs/my_run
```

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

## Agent Example Cookbook

The `agents/examples/` configs are short, headless smoke scenarios designed for
agent-assisted creation, testing, and evaluation:

| Goal | Example | Validate |
| --- | --- | --- |
| Passive orbit propagation | `agents/examples/public_agent_single_satellite.yaml` | `python run_simulation.py --config agents/examples/public_agent_single_satellite.yaml --validate-only` |
| Closed-loop rendezvous | `agents/examples/public_agent_rendezvous_lqr.yaml` | `python run_simulation.py --config agents/examples/public_agent_rendezvous_lqr.yaml --validate-only` |
| TLE ground-station access | `agents/examples/public_agent_ground_access.yaml` | `python run_simulation.py --config agents/examples/public_agent_ground_access.yaml --validate-only` |
| Attitude hold | `agents/examples/public_agent_attitude_hold.yaml` | `python run_simulation.py --config agents/examples/public_agent_attitude_hold.yaml --validate-only` |

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
python run_simulation.py --config agents/examples/public_agent_single_satellite.yaml --validate-only
python run_simulation.py --config agents/examples/public_agent_single_satellite.yaml
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
