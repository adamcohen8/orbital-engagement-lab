# OEL Agent Instructions

Orbital Engagement Lab agents should orchestrate documented workflows. They
should not replace, approximate, or silently bypass the deterministic physics
engine.

This file is intentionally public-safe. It helps AI coding agents such as
Codex, Cursor, Claude Code, and Gemini CLI work with the open-source OEL core.
For the fuller public agent playbook, read `agents/public/AGENTS.md` and
`docs/oel-agents.md`.

## Default Agent Posture

- Treat scenario YAML, CLI commands, Python APIs, tests, docs, and generated
  artifacts as the supported interface.
- Prefer small, inspectable changes that match existing OEL patterns.
- Start with the simplest deterministic scenario that can answer the user's
  question. Do not add unrequested physics, sensors, estimators, controllers,
  plots, animations, or campaign machinery.
- Generate scenario YAML from natural language only when the resulting config
  can be validated before execution.
- Run `python run_simulation.py --config <path> --validate-only` before running
  a new or edited scenario.
- Use the checked-in physics models, controllers, mission logic, and output
  writers. Do not invent shortcut physics in agent scripts or reports.
- Prefer the review store query API over ad hoc parsing of large run logs when
  `review/run.sqlite` is available.
- Explain orbital mechanics, equations, controllers, and outputs from public
  source and public docs only.
- Call out uncertainty, missing validation evidence, and model limits plainly.

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
- simple dynamics first. Do not enable J2, J3, J4, drag, SRP, third bodies, or
  high-fidelity propagation unless the user asks for them or the stated study
  requires them.

Examples:

- For "make a simple deorbit study", ask whether they want a simple maneuver
  geometry case or a drag-including decay study; do not silently enable a full
  perturbation stack.
- For "propagate this satellite for two hours", use simple propagation unless
  the user asks for perturbations or realistic force modeling.
- For "rendezvous with noisy measurements", sensing and estimation are
  relevant. For "rendezvous with perfect knowledge", do not add estimation.

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
python run_simulation.py --config <path> --validate-only
python run_simulation.py --config <path>
```

After the run, query the saved review DB:

```bash
python -m sim.review outputs/<scenario_name> --query "SELECT scenario_name, duration_s, samples FROM run_metadata"
python -m sim.review outputs/<scenario_name> --query "SELECT time_s, range_km FROM relative_state ORDER BY time_s LIMIT 20" --json
```

Rules for agents:

- Use only `SELECT` or `WITH` queries. The review API enforces read-only
  access; do not try to mutate, attach, or rewrite review databases.
- Query tables such as `run_metadata`, `objects`, `time_samples`,
  `object_state`, `relative_state`, `thrust`, `ground_access`, `events`,
  `metrics`, and `artifacts` when present.
- State the query used when summarizing a result so the user can reproduce the
  evidence.
- If `review/run.sqlite` is missing, fall back to `index.md`,
  `master_run_summary.json`, CSV histories, and plots. Do not pretend a review
  store exists.
- Do not recommend ORW for routine agent analysis. `run_orw.py` is an
  experimental preview and should only be used when the user explicitly asks
  for the interactive local workbench. Prefer `python -m sim.review`.

## Public Commands

```bash
python run_simulation.py --doctor
python run_simulation.py --quickstart --validate-only
python run_simulation.py --quickstart
python run_simulation.py --config configs/automation_smoke.yaml --validate-only
python run_simulation.py --config configs/automation_smoke.yaml
python -m sim.review outputs/<scenario_name> --query "SELECT scenario_name FROM run_metadata"
python run_game.py
```

For generated examples:

```bash
python run_simulation.py --config agents/examples/public_agent_single_satellite.yaml --validate-only
python run_simulation.py --config agents/examples/public_agent_single_satellite.yaml
```

## Safety And IP Boundary

- Only run scenario YAML from trusted sources. OEL configs can reference
  importable Python modules/classes.
- Keep API keys, proprietary configs, customer data, and generated report
  packets out of public commits.
- Public agents may explain public code. If a requested workflow depends on
  capability that is not included in the public core, say so and point to the
  documented public alternative.
