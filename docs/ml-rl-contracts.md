# ML/RL Policy Contracts

OEL treats ML/RL policies as research artifacts unless their observation,
reward, safety, and provenance contracts are explicit.

## Observation Contract

Default Gym observations are observer-owned belief state:

- `belief.<agent>.state[...]` for the controlled object's own estimate
- `knowledge.<observer>.<target>.state[...]` for observer-owned target tracks

Truth-state observations are allowed only for explicit oracle baselines:

- `truth.*` fields expose simulator internal truth.
- Policies using `truth.*` must be labeled `oracle_baseline`.
- Simulator metrics such as `metrics.range_km` are also privileged unless a
  scenario demonstrates that the same value is produced by an onboard observer.

## Policy Cards

Every trained or evaluated policy should have a policy card covering:

- training envelope: scenario, duration, timestep, force-model flags, and
  episode variations
- observation source: belief, knowledge, simulator metric, or truth
- privileged data: truth-state or simulator-derived fields visible to the
  policy
- reward: reward implementation, terms, and weights
- safety gates: termination rules, keep-out/capture gates, actuator limits, and
  human/operator restrictions
- OOD checks: out-of-distribution sweeps that were run or are still missing
- provenance: code version, git status, Python environment, config, seeds, and
  artifact paths

Generate a card from the generic Gym environment config:

```python
from machine_learning import GymEnvConfig, build_policy_card, write_policy_card

env_cfg = GymEnvConfig(scenario=scenario, controlled_agent_id="chaser")
card = build_policy_card(
    env_cfg,
    policy_name="rpo_chaser_belief_policy",
    ood_checks={
        "planned": [
            "initial-condition sweep",
            "sensor-noise/dropout sweep",
            "actuator-limit sweep",
            "force-model/fidelity sweep",
        ]
    },
)
write_policy_card(card, "outputs/policy_cards/rpo_chaser_belief_policy.json")
write_policy_card(card, "outputs/policy_cards/rpo_chaser_belief_policy.md")
```

If the card reports `oracle_baseline: true`, use the result for debugging or
upper-bound comparison only.
