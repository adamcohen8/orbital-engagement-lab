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

## Supported Workflows

1. Read the relevant docs and examples.
2. Draft or edit scenario YAML.
3. Validate the config before execution.
4. Run the scenario through `run_simulation.py`.
5. Inspect generated `index.md`, `master_run_summary.json`, CSV, and plot
   artifacts.
6. Evaluate the run with `agents/public/evaluation-rubric.md`.
7. Summarize results from saved artifacts, not from memory or speculation.
8. Add tests or smoke checks for new agent-facing examples.

## Canonical Commands

Validate environment:

```bash
python run_simulation.py --doctor
```

Validate config:

```bash
python run_simulation.py --config configs/automation_smoke.yaml --validate-only
```

Run scenario:

```bash
python run_simulation.py --config configs/automation_smoke.yaml
```

Run smoke test:

```bash
python -m pytest sim/tests/test_oel_agents.py
```

Generate a report-like public artifact set:

```bash
python run_simulation.py --quickstart
```

Inspect outputs:

```bash
python - <<'PY'
import json
from pathlib import Path

summary = Path("outputs/quickstart_5min/master_run_summary.json")
print(json.dumps(json.loads(summary.read_text()), indent=2)[:4000])
PY
```

Public AI report generation is not a default open-source workflow. In the
public core, treat `index.md`, JSON summaries, CSV files, plots, and game
debriefs as report artifacts.

## Scenario Generation Rules

- Start from a nearby public example whenever possible. When the user asks for
  a new scenario from scratch, use examples for structure but write a new YAML
  file with a distinct scenario name and output directory.
- Use `objects:` entries for spacecraft instead of legacy top-level `chaser`
  or `target` sections.
- Keep units explicit in field names: `_km`, `_s`, `_deg`, `_rad_s`,
  `_km_s`, `_kg`, `_n`, and similar suffixes.
- Keep first drafts short, headless, and deterministic: plots off, animations
  off, modest duration, and public controllers.
- For JSON outputs, use `outputs.stats.save_json: true` for the compact
  `master_run_summary.json`. Set `outputs.stats.save_full_log: true` only when
  the user needs detailed time-history review in `master_run_log.json`.
- Validate every generated config with `--validate-only`.
- Run only trusted YAML. Scenario plugin pointers can import Python code.

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
  plots:
    enabled: true
    figure_ids: ["relative_range", "trajectory_ric_curv_2d_multi", "control_effort"]
```

Use those richer artifacts to inspect range history, range-rate behavior,
controller transients, burn timing, trajectory shape, and whether closure was
monotonic or merely improved at the final sample.

## Natural-Language To YAML Examples

Request:

> Propagate one passive satellite in a 7000 km circular, 45 degree orbit for
> one minute using two-body dynamics, no plots, and saved JSON stats.

Use:

```bash
python run_simulation.py --config agents/examples/public_agent_single_satellite.yaml --validate-only
python run_simulation.py --config agents/examples/public_agent_single_satellite.yaml
```

Request:

> Give me a quick closed-loop rendezvous example with public controllers.

Use:

```bash
python run_simulation.py --config examples/configs/public_closed_loop_rendezvous_lqr.yaml --validate-only
python run_simulation.py --config examples/configs/public_closed_loop_rendezvous_lqr.yaml
```

Request:

> Help me practice manual RPO.

Use:

```bash
python run_game.py examples/configs/public_manual_rpo_training.yaml
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
- "Build an attitude-hold scenario with an initial pointing error."
- "Evaluate the run in this output folder and tell me whether it supports my
  goal."

If the request leaves out necessary details, make conservative public defaults
visible in the generated YAML and explain them in the result summary. Ask a
clarifying question only when a reasonable public default would be misleading
or risky.

## Agent-Generated Example Configs

These examples are intentionally short and headless so users can validate,
run, and inspect them quickly:

```bash
python run_simulation.py --config agents/examples/public_agent_single_satellite.yaml --validate-only
python run_simulation.py --config agents/examples/public_agent_rendezvous_lqr.yaml --validate-only
python run_simulation.py --config agents/examples/public_agent_ground_access.yaml --validate-only
python run_simulation.py --config agents/examples/public_agent_attitude_hold.yaml --validate-only
```

## Explanation Rules

- Explain public equations, frames, controls, outputs, and limitations from
  checked-in docs and source.
- Distinguish model behavior from real mission behavior.
- Do not claim validation beyond the artifacts and tests that actually exist.
- When a result matters, point the user to the generated output file and the
  command that produced it.

## Safety And IP Guidance

- Do not commit API keys, local customer data, generated AI report packets, or
  non-public outputs.
- Do not copy tuned parameters, optimizer traces, customer material, or
  generated validation evidence into public examples.
- Keep public examples educational, inspectable, and reproducible.
- If a user asks for capability that is not present in the public core,
  redirect to documented public CLI/API behavior and examples.
