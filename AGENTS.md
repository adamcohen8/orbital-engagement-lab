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
- Generate scenario YAML from natural language only when the resulting config
  can be validated before execution.
- Run `python run_simulation.py --config <path> --validate-only` before running
  a new or edited scenario.
- Use the checked-in physics models, controllers, mission logic, and output
  writers. Do not invent shortcut physics in agent scripts or reports.
- Explain orbital mechanics, equations, controllers, and outputs from public
  source and public docs only.
- Call out uncertainty, missing validation evidence, and model limits plainly.

## Public Commands

```bash
python run_simulation.py --doctor
python run_simulation.py --quickstart --validate-only
python run_simulation.py --quickstart
python run_simulation.py --config configs/automation_smoke.yaml --validate-only
python run_simulation.py --config configs/automation_smoke.yaml
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
