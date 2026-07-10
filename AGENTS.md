# OEL Agent Instructions

Orbital Engagement Lab agents should orchestrate documented workflows. They
should not replace, approximate, or silently bypass the deterministic physics
engine.

Agents are general helpers for OEL, not runners for a fixed example catalog.
Use examples and task cards as onboarding rails and regression fixtures, then
generalize from the documented interfaces to the user's actual question.

This file is intentionally public-safe. It helps AI coding agents such as
Codex, Cursor, Claude Code, Gemini CLI, and Grok Build work with the
open-source OEL core.
For the fuller public agent playbook, read `agents/public/AGENTS.md` and
`docs/oel-agents.md`. For routing broad user intents to documented workflows,
read `docs/agent-capability-routing.md`. For repeatable recipe execution,
completed-run inspection, comparison packets, and standard review plots, read
`docs/agent-task-runner.md`.

## Default Agent Posture

- Treat scenario YAML, CLI commands, Python APIs, tests, docs, and generated
  artifacts as the supported interface.
- Treat **OGP** as the product name for the **OEL General Propagator**:
  OEL's catalog-style general-perturbations family for TLE/mean-element
  products. **OGP-SGP4** is the supported near-Earth SGP4 path, and
  **OGP-SDP4** is the supported deep-space/resonance SDP4 path.
- Treat **ONP** as the product name for the **OEL Numerical Propagator**:
  OEL's configurable numerical propagation path for two-body and
  special-perturbation force-model studies. Use **OGP** for explicit passive
  catalog-style general-perturbations propagation or SGP4/SDP4-family
  mean-element products. Reserve **HPOP** for external reference/validation
  workflows or legacy command names, not as the name of OEL's native
  propagator.
- Interpret the user's intent first. Choose a nearby example only when it
  genuinely fits; otherwise create the minimum viable validated scenario that
  answers the request.
- Prefer small, inspectable changes that match existing OEL patterns.
- Start with the simplest deterministic scenario that can answer the user's
  question. Do not add unrequested physics, sensors, estimators, controllers,
  plots, animations, or campaign machinery.
- Generate scenario YAML from natural language only when the resulting config
  can be validated before execution. For user-provided or otherwise unfamiliar
  YAML, run `--safe-validate` before ordinary importing validation.
- Run `.venv/bin/python run_simulation.py --config <path> --validate-only` before running
  a new or edited scenario.
- Treat unknown-field errors as intent failures; do not remove or rename fields
  until the normalized contract is understood. Non-empty Cartesian initial
  states require both position and velocity, and scenarios must select exactly
  one orbital-state form.
- Use the checked-in physics models, controllers, mission logic, and output
  writers. Do not invent shortcut physics in agent scripts or reports.
- Prefer the review store query API over ad hoc parsing of large run logs when
  `review/run.sqlite` is available.
- Use `.venv/bin/python -m sim.agent_task` when a bundled recipe, comparison,
  standard plot, or portable `agent_evidence_packet.json` would make the
  workflow more reproducible.
- Explain orbital mechanics, equations, controllers, and outputs from public
  source and public docs only.
- Call out uncertainty, missing validation evidence, and model limits plainly.
- When OEL itself appears hard to use, confusing, or missing an agent-facing
  workflow, follow `docs/agent-feedback-loop.md`: prepare a public-safe
  feedback draft, ask the user for permission, and submit only after approval.

## Scenario Generation Posture

Use a minimum viable scenario, then add complexity only when the user asks for
it or when it is necessary to answer the question they actually asked.

Ask a clarifying question when a missing detail changes the study:

- time horizon or duration,
- initial orbit, TLE, altitude, or relative state,
- passive vs controlled behavior,
- success metric or termination condition,
- fidelity level when the request says "realistic", "high fidelity",
  "operational", "deorbit", "decay", "access", or similar,
- whether the user wants plots, review-store inspection, or just summary
  outputs.

Default quietly when the detail is incidental:

- headless run,
- plots and animations off unless requested,
- attitude disabled unless attitude dynamics/control is part of the request,
- no sensing or estimation unless observation uncertainty, access, tracking, or
  closed-loop knowledge is needed,
- no Monte Carlo, sensitivity, campaign, optimizer, or report workflow unless
  requested,
- simple ONP dynamics first. Do not enable J2, J3, J4, drag, SRP, third
  bodies, spherical harmonics, or high-fidelity ONP propagation unless the user
  asks for them or the stated study requires them.

Examples:

- For "make a simple deorbit study", ask whether they want a simple maneuver
  geometry case or a drag-including decay study; do not silently enable a full
  perturbation stack.
- For "propagate this satellite for two hours", use simple propagation unless
  the user asks for perturbations or realistic force modeling.
- For "rendezvous with noisy measurements", sensing and estimation are
  relevant. For "rendezvous with perfect knowledge", do not add estimation.

## When The User Asks Something New

Do not force new user requests into the checked-in examples. Instead:

1. Restate the study goal in OEL terms.
2. Identify the minimum scenario elements needed: objects, initial state,
   dynamics, controller posture, duration, outputs, and review evidence.
3. Reuse public examples for structure when helpful, but give the new scenario
   a distinct name and output directory.
4. Validate before running, inspect artifacts after running, and explain what
   evidence supports the answer.
5. Name missing evidence or public-core limits instead of stretching an example
   beyond what it proves.

Use `docs/agent-capability-routing.md` to map broader requests such as TLE
propagation, access, attitude, plotting, game/training, validation, sealed-mode,
or comparison work to the right public workflow and evidence.

## Review Query Workflow

The review store is the agent-friendly output inspection path. Use it when a
user asks questions about a completed run, wants tabular insight, or needs
custom figures from run evidence.

To create a queryable run, add this to scenario YAML before validation:

```yaml
outputs:
  review:
    enabled: true
    detail: standard
```

Then validate and run through the simulator:

```bash
.venv/bin/python run_simulation.py --config <path> --validate-only
.venv/bin/python run_simulation.py --config <path>
```

After the run, query the saved review DB:

```bash
.venv/bin/python -m sim.review outputs/<scenario_name> --query "SELECT scenario_name, duration_s, samples FROM run_metadata"
.venv/bin/python -m sim.review outputs/<scenario_name> --query "SELECT time_s, range_km FROM relative_state ORDER BY time_s LIMIT 20" --json
```

For a machine-readable evidence packet around common recipes or completed
outputs, use:

```bash
.venv/bin/python -m sim.agent_task list
.venv/bin/python -m sim.agent_task run quickstart_review --output-root outputs/agent_tasks
.venv/bin/python -m sim.agent_task inspect outputs/<scenario_name> --query run_metadata --json
```

Rules for agents:

- Use only `SELECT` or `WITH` queries. The review API enforces read-only
  access; do not try to mutate, attach, or rewrite review databases.
- Query tables such as `run_metadata`, `objects`, `time_samples`,
  `object_state`, `relative_state`, `thrust`, `ground_access`, `events`,
  `metrics`, and `artifacts` when present.
- Do not guess column names. Use `docs/agent-review-queries.md`,
  `review/schema.json`, or a small discovery query such as
  `SELECT * FROM object_state LIMIT 1` before writing custom SQL against an
  unfamiliar table.
- State the query used when summarizing a result so the user can reproduce the
  evidence.
- If `review/run.sqlite` is missing, fall back to `index.md`,
  `master_run_summary.json`, CSV histories, and plots. Do not pretend a review
  store exists.
- Use `sim.review.EvidencePlotter` or `.venv/bin/python -m sim.review plot`
  for custom OEL-styled plots from completed review evidence. See
  `docs/agent-custom-plots.md`.
- Before handing off a generated plot, visually inspect it and fix obvious
  presentation defects such as legends covering data, overlapping labels,
  clipped text, unreadable tick labels, blank figures, or missing expected
  series.
- Use `.venv/bin/python -m sim.review` for table inspection and the custom
  plotting API for brief/report figures. Experimental review viewers and
  desktop workbenches are local-only until they are product-ready.

## Agent Feedback Loop

If an agent encounters a feedback-worthy OEL problem while helping a user, it
may propose submitting public feedback. Examples include missing examples,
confusing validation errors, insufficient artifacts for a reasonable public
question, unclear docs, or agent guidance conflicts.

Rules:

- Never submit feedback silently.
- Show the user the public-safe summary before submitting.
- Ask for explicit approval.
- Do not include secrets, API keys, customer data, CUI, export-controlled data,
  classified information, private configs, or private generated report packets.
- Use the private `SECURITY.md` process for vulnerabilities or sensitive-data
  exposure.
- Use the GitHub Agent Feedback issue template after approval.

## Public Commands

```bash
.venv/bin/python run_simulation.py --doctor
.venv/bin/python run_simulation.py --quickstart --validate-only
.venv/bin/python run_simulation.py --quickstart
.venv/bin/python run_simulation.py --config configs/automation_smoke.yaml --validate-only
.venv/bin/python run_simulation.py --config configs/automation_smoke.yaml
.venv/bin/python -m sim.review outputs/<scenario_name> --query "SELECT scenario_name FROM run_metadata"
.venv/bin/python run_game.py
```

For generated examples and evaluation fixtures:

```bash
.venv/bin/python run_simulation.py --config agents/examples/public_agent_single_satellite.yaml --validate-only
.venv/bin/python run_simulation.py --config agents/examples/public_agent_single_satellite.yaml
```

For repeatable public agent task checks, use `docs/agent-task-cards.md`.

## Safety And IP Boundary

- Only run scenario YAML from trusted sources. OEL configs can reference
  importable Python modules/classes.
- Keep API keys, proprietary configs, customer data, and generated report
  packets out of public commits.
- Public agents may explain public code. If a requested workflow depends on
  capability that is not included in the public core, say so and point to the
  documented public alternative.
