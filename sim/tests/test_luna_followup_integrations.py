"""Bounded regressions for rollout, self-play, and Gym seed handoff."""

from __future__ import annotations

import numpy as np

from machine_learning import rendezvous_env as rendezvous_module
from machine_learning.gym_env import GymEnvConfig, GymSimulationEnv, MultiAgentEnvConfig, MultiAgentSimulationEnv
from machine_learning.self_play import LinearPolicy, SelfPlayTrainerConfig, run_self_play_training
from machine_learning.training_adapter import collect_multi_agent_rollout, collect_vector_rollout
from sim.dynamics.orbit.two_body import propagate_two_body_rk4
from sim.tests.test_rl_gym_env import _base_scenario


class _ShortVectorEnv:
    auto_reset = False

    def __init__(self):
        self.steps = 0

    def reset(self):
        self.steps = 0
        return np.zeros((1, 1), dtype=np.float32), {}

    def step(self, actions):
        self.steps += 1
        return (np.zeros((1, 1), dtype=np.float32), np.zeros(1), np.zeros(1, dtype=bool),
                np.array([self.steps == 2]), {"time_s": [self.steps]})


class _ShortMultiAgentEnv:
    def __init__(self):
        self.steps = 0

    def reset(self, *, seed=None):
        self.steps = 0
        return {"agent": np.ones(1, dtype=np.float32)}, {"agent": {}}

    def step(self, actions):
        self.steps += 1
        return ({"agent": np.ones(1, dtype=np.float32)}, {"agent": 0.0},
                {"agent": False}, {"agent": self.steps == 2},
                {"agent": {"time_s": self.steps}})


def test_rollout_collectors_stop_before_post_terminal_steps():
    vector = _ShortVectorEnv()
    batch = collect_vector_rollout(vector, policy_fn=lambda obs: np.zeros_like(obs), horizon=4)
    assert vector.steps == 2
    assert batch.rewards.shape == (2, 1)
    assert batch.truncated[-1, 0]

    multi = _ShortMultiAgentEnv()
    batch_multi = collect_multi_agent_rollout(
        multi, policy_fns_by_agent={"agent": lambda obs: np.zeros(1)}, horizon=4
    )
    assert multi.steps == 2
    assert batch_multi.rewards_by_agent["agent"].shape == (2,)
    assert batch_multi.truncated_by_agent["agent"][-1]


class _FixedMutationPolicy(LinearPolicy):
    direction: float = 1.0

    def mutate(self, rng, sigma):
        return np.array([[self.direction]])


class _ActionRewardEnv:
    def reset(self, *, seed=None):
        return {"agent": np.ones(1, dtype=np.float32)}, {"agent": {}}

    def step(self, actions):
        reward = float(actions["agent"][0])
        return ({"agent": np.ones(1, dtype=np.float32)}, {"agent": reward},
                {"agent": True}, {"agent": False}, {"agent": {}})


def test_self_play_applies_only_evaluated_improving_direction():
    for direction, expected in ((-1.0, 0.0), (1.0, 0.5)):
        policy = _FixedMutationPolicy(weights=np.zeros((1, 1)), bias=np.zeros(1))
        policy.direction = direction
        run_self_play_training(
            _ActionRewardEnv(), policies_by_agent={"agent": policy},
            trainer_cfg=SelfPlayTrainerConfig(iterations=1, rollout_horizon=1,
                                              learning_rate=0.5, snapshot_interval=0),
        )
        assert policy.weights[0, 0] == expected


def test_gym_episode_seed_reaches_runtime_metadata_in_both_paths():
    scenario = _base_scenario()
    scenario["objects"]["chaser"]["knowledge"] = {"targets": ["target"]}
    single = GymSimulationEnv(GymEnvConfig(scenario=scenario, controlled_agent_id="chaser"))
    single.reset(seed=11)
    assert single.scenario_cfg.metadata["seed"] == 11
    first_sensor_draw = single.agents["chaser"].knowledge_base._rng.random()
    single.reset(seed=11)
    assert single.agents["chaser"].knowledge_base._rng.random() == first_sensor_draw
    single.reset(seed=12)
    assert single.scenario_cfg.metadata["seed"] == 12
    assert single.agents["chaser"].knowledge_base._rng.random() != first_sensor_draw

    multi = MultiAgentSimulationEnv(MultiAgentEnvConfig(
        scenario=scenario, controlled_agent_ids=("chaser", "target")
    ))
    multi.reset(seed=13)
    assert multi.scenario_cfg.metadata["seed"] == 13
    first_multi_sensor_draw = multi.agents["chaser"].knowledge_base._rng.random()
    multi.reset(seed=13)
    assert multi.agents["chaser"].knowledge_base._rng.random() == first_multi_sensor_draw
    multi.reset(seed=14)
    assert multi.scenario_cfg.metadata["seed"] == 14
    assert multi.agents["chaser"].knowledge_base._rng.random() != first_multi_sensor_draw


def test_rendezvous_lookahead_kernel_preserves_episode_outputs(monkeypatch):
    config = rendezvous_module.RLRendezvousConfig()
    actions = [np.array([0.0, 0.0, 0.0, float(i % 3 == 0)]) for i in range(30)]

    def run_episode():
        env = rendezvous_module.RLRendezvousEnv(config)
        rows = [env.reset()]
        for action in actions:
            observation, reward, done, _ = env.step(action)
            rows.append(np.concatenate((observation, [reward, float(done)])))
        return rows

    optimized = run_episode()

    def python_step(state, dt_s, mu):
        return propagate_two_body_rk4(state, dt_s, mu, np.zeros(3))

    monkeypatch.setattr(rendezvous_module, "propagate_two_body_rk4_kernel", python_step)
    original = run_episode()
    for before, after in zip(original, optimized):
        np.testing.assert_array_equal(before, after)
