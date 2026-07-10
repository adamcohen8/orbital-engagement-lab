# Agent Golden Paths

Use these golden paths when an agent or first-time user needs the shortest
credible route from a plain-language request to OEL evidence. Each path keeps
the run deterministic, validates before execution, writes a review store, and
ends with an answer that cites saved evidence.

These paths are adoption rails, not the boundary of what OEL Agents can do. If
the user asks a different question, use
[`agent-capability-routing.md`](agent-capability-routing.md) to choose the
smallest documented workflow that can answer it.

## Golden Path Index

| Path | Use when the user asks | Configs | Required saved queries |
| --- | --- | --- | --- |
| Minimal propagation | "Propagate a simple satellite" or "show me the smallest public run" | `agents/examples/public_agent_single_satellite.yaml`; Python API artifact fixture `agents/examples/public_agent_python_api_minimal_propagation.yaml` | `run_metadata`, `objects`, `passive_final_state`, `artifacts` |
| Closed-loop rendezvous | "Run a short public rendezvous example" or "did the chaser reach the target?" | `agents/examples/public_agent_rendezvous_lqr.yaml` | `run_metadata`, `rendezvous_metrics`, `rendezvous_closest_approach`, `relative_final_state`, `burn_activity`, `burn_events` |
| Mission recovery and reconstitution | "What recovery burn is needed after a simple disturbance?" or "compare recovery candidates" | `agents/examples/public_agent_mission_recovery_plus_c_burn.yaml`; `agents/examples/public_agent_mission_reconstitution_trade_space.yaml` | `run_metadata`, `burn_activity`, `mission_recovery_summary`, `mission_recovery_elements`, `mission_recovery_candidates`, `mission_recovery_burns` |

## Shared Loop

For every golden path:

1. State the study goal in OEL terms.
2. Validate the config before running it.
3. Run the deterministic simulator.
4. Query `review/run.sqlite` with saved queries or explicit `SELECT` SQL.
5. Inspect `index.md` and `master_run_summary.json` for artifact context.
6. Explain what the evidence supports and what it does not support.

Use the canonical output directories shown in this guide, usually
`outputs/agents/<scenario_name>/`. Do not treat older sibling folders or stale
local validation outputs as current evidence unless you regenerated them during
the same workflow.

## Minimal Propagation

Study goal: propagate one passive satellite for five minutes with simple
two-body dynamics and zero-control baselines.

Validate and run:

```bash
.venv/bin/python run_simulation.py --config agents/examples/public_agent_single_satellite.yaml --validate-only
.venv/bin/python run_simulation.py --config agents/examples/public_agent_single_satellite.yaml
```

For the Python API authoring path, use the task card to regenerate the artifact
fixture, then validate and run the produced YAML:

```bash
.venv/bin/python run_simulation.py --config agents/examples/public_agent_python_api_minimal_propagation.yaml --validate-only
.venv/bin/python run_simulation.py --config agents/examples/public_agent_python_api_minimal_propagation.yaml
```

Required evidence:

```bash
.venv/bin/python -m sim.review outputs/agents/public_agent_single_satellite --saved-query run_metadata
.venv/bin/python -m sim.review outputs/agents/public_agent_single_satellite --saved-query objects
.venv/bin/python -m sim.review outputs/agents/public_agent_single_satellite --saved-query passive_final_state
.venv/bin/python -m sim.review outputs/agents/public_agent_single_satellite --saved-query artifacts
```

Answer shape:

- Status: validation and run result.
- Scenario name, duration, timestep, and active object.
- Dynamics and control posture.
- Final state evidence from `passive_final_state` or equivalent SQL.
- Output directory and artifacts inspected.
- Limit: educational deterministic smoke scenario, not operational ephemeris
  accuracy or mission validation.

## Closed-Loop Rendezvous

Study goal: run a short two-satellite rendezvous scenario with the public HCW
LQR controller, then decide whether the evidence supports a terminal
rendezvous claim.

Validate and run:

```bash
.venv/bin/python run_simulation.py --config agents/examples/public_agent_rendezvous_lqr.yaml --validate-only
.venv/bin/python run_simulation.py --config agents/examples/public_agent_rendezvous_lqr.yaml
```

Required evidence:

```bash
.venv/bin/python -m sim.review outputs/agents/public_agent_rendezvous_lqr --saved-query run_metadata
.venv/bin/python -m sim.review outputs/agents/public_agent_rendezvous_lqr --saved-query rendezvous_metrics
.venv/bin/python -m sim.review outputs/agents/public_agent_rendezvous_lqr --saved-query rendezvous_closest_approach
.venv/bin/python -m sim.review outputs/agents/public_agent_rendezvous_lqr --saved-query relative_final_state
.venv/bin/python -m sim.review outputs/agents/public_agent_rendezvous_lqr --saved-query burn_activity
.venv/bin/python -m sim.review outputs/agents/public_agent_rendezvous_lqr --saved-query burn_events
```

Answer shape:

- Status: validation and run result.
- Initial range, closest approach, final range, and final range rate.
- Burn activity and any burn events recorded.
- Explicit success threshold used before saying "terminal rendezvous."
- Limit: range closure alone is not rendezvous success; do not infer safety,
  robustness, or controller superiority from one deterministic run.

## Mission Recovery And Reconstitution

Study goal: apply a simple configured disturbance burn, compare final-vs-initial
orbital elements, and inspect the recovery estimate or planner candidate trade
space.

For a compact recovery estimate after a +C disturbance:

```bash
.venv/bin/python run_simulation.py --config agents/examples/public_agent_mission_recovery_plus_c_burn.yaml --validate-only
.venv/bin/python run_simulation.py --config agents/examples/public_agent_mission_recovery_plus_c_burn.yaml
```

For the planner trade-space path:

```bash
.venv/bin/python run_simulation.py --config agents/examples/public_agent_mission_reconstitution_trade_space.yaml --validate-only
.venv/bin/python run_simulation.py --config agents/examples/public_agent_mission_reconstitution_trade_space.yaml
```

Required evidence:

```bash
.venv/bin/python -m sim.review outputs/agents/public_agent_mission_reconstitution_trade_space --saved-query run_metadata
.venv/bin/python -m sim.review outputs/agents/public_agent_mission_reconstitution_trade_space --saved-query burn_activity
.venv/bin/python -m sim.review outputs/agents/public_agent_mission_reconstitution_trade_space --saved-query mission_recovery_summary
.venv/bin/python -m sim.review outputs/agents/public_agent_mission_reconstitution_trade_space --saved-query mission_recovery_elements
.venv/bin/python -m sim.review outputs/agents/public_agent_mission_reconstitution_trade_space --saved-query mission_recovery_candidates
.venv/bin/python -m sim.review outputs/agents/public_agent_mission_reconstitution_trade_space --saved-query mission_recovery_burns
```

Answer shape:

- Status: validation and run result.
- Disturbance burn evidence from `burn_activity` and/or burn rows.
- Initial-vs-final orbital-element comparison.
- Recovery delta-v, time, propellant, and planner candidates when recorded.
- Recommended candidate only when `mission_recovery_candidates` and
  `mission_recovery_burns` support it.
- Limit: this is deterministic public review evidence for simple recovery
  reasoning, not an operational recovery plan, finite-burn optimization, or
  uncertainty analysis.

## Maintenance Checklist

When changing one of these paths:

- Keep the config path, output directory, and saved queries aligned.
- Update the matching task card and answer example.
- Keep `outputs.review.enabled: true`.
- Run `sim/tests/test_oel_agents.py`.
- If this is part of a public release, follow the public export workflow and
  update the changelog before opening a release pull request.
