# Agent Task Cards

Agent task cards are the public OEL Agents evaluation set. They are
fixtures for testing whether an agent follows the evidence loop; they are not
the boundary of what OEL Agents can help users do.

The general workflow remains:

```text
natural-language request -> scenario YAML -> validation -> deterministic run
-> review-store query -> artifact-supported explanation
```

For the shortest first-run adoption paths across propagation, rendezvous, and
mission recovery/reconstitution, use
[`agent-golden-paths.md`](agent-golden-paths.md) before reaching for the full
card set.

Each card includes a user prompt, assumptions, an example config, validation and
run commands, required review queries, expected answer shape, pass criteria, and
red flags.

## Cards

| Task | Card | Answer example | Example config |
| --- | --- | --- | --- |
| Python API minimal propagation | [Python API Minimal Propagation](../agents/tasks/python_api_minimal_propagation.md) | [Answer](../agents/tasks/examples/python_api_minimal_propagation_answer.md) | `agents/examples/public_agent_python_api_minimal_propagation.yaml` |
| Passive propagation | [Passive Propagation](../agents/tasks/passive_propagation.md) | [Answer](../agents/tasks/examples/passive_propagation_answer.md) | `agents/examples/public_agent_single_satellite.yaml` |
| Closed-loop rendezvous | [Closed-Loop Rendezvous](../agents/tasks/closed_loop_rendezvous.md) | [Answer](../agents/tasks/examples/closed_loop_rendezvous_answer.md) | `agents/examples/public_agent_rendezvous_lqr.yaml` |
| Mission recovery +C burn | [Mission Recovery +C Burn](../agents/tasks/mission_recovery_plus_c_burn.md) | [Answer](../agents/tasks/examples/mission_recovery_plus_c_burn_answer.md) | `agents/examples/public_agent_mission_recovery_plus_c_burn.yaml` |
| Mission reconstitution trade space | [Mission Reconstitution Trade Space](../agents/tasks/mission_reconstitution_trade_space.md) | [Answer](../agents/tasks/examples/mission_reconstitution_trade_space_answer.md) | `agents/examples/public_agent_mission_reconstitution_trade_space.yaml` |
| Ground access from TLE | [Ground Access From TLE](../agents/tasks/ground_access_from_tle.md) | [Answer](../agents/tasks/examples/ground_access_from_tle_answer.md) | `agents/examples/public_agent_ground_access.yaml` |
| Attitude hold | [Attitude Hold](../agents/tasks/attitude_hold.md) | [Answer](../agents/tasks/examples/attitude_hold_answer.md) | `agents/examples/public_agent_attitude_hold.yaml` |
| Compare one change | [Compare One Change](../agents/tasks/compare_one_change.md) | [Answer](../agents/tasks/examples/compare_one_change_answer.md) | `agents/examples/public_agent_rendezvous_lqr.yaml` |

## How To Use Them

For agent evaluation:

1. Give the card's user prompt to the agent.
2. Watch whether the agent chooses the nearest public example or creates a
   scoped new scenario.
3. Require validation before execution.
4. Require artifact inspection after execution.
5. Require review-store queries when `review/run.sqlite` exists.
6. Judge the response against the card's pass criteria and red flags.

For maintainers:

- Keep `AGENTS.md` and `docs/oel-agents.md` focused on general agent doctrine.
- Keep cards tied to runnable public examples.
- Keep required review queries executable against the example output.
- Update `sim/tests/test_oel_agents.py` when adding a card.
- Prefer adding a new card when a user-facing agent workflow becomes common.
