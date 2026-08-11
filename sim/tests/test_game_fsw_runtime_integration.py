from __future__ import annotations

import numpy as np
import pytest

from sim.api import SimulationConfig
from sim.dynamics.orbit.cr3bp import cr3bp_moon_state_km_s
from sim.game.attempt_lifecycle import (
    _request_pilot_input_poll_for_transition,
    _start_game_attempt,
)
from sim.game.manual import KeyboardCommandState
from sim.game.training import RPOTrainingConfig


def test_game_session_uses_stack_commands_and_physical_realization() -> None:
    config = SimulationConfig.from_yaml("sim/game/configs/game_mode_basic.yaml")
    session, adapter, _ = _start_game_attempt(
        config,
        command_state=KeyboardCommandState(firing=True),
        training_cfg=RPOTrainingConfig(enabled=False),
        controlled_object_id="chaser",
        attitude_rate_deg_s=8.0,
        control_mode="attitude_thrust",
        ric_reference_object_id="target",
    )
    snapshot = session.step(dt_s=0.1)
    runtime = session._engine.agents["chaser"].flight_software_runtime
    assert runtime is not None
    assert runtime.evidence.outputs[-1].commands
    assert runtime.evidence.receipts
    assert runtime.evidence.realizations
    assert np.linalg.norm(snapshot.applied_thrust["chaser"]) > 0.0
    assert adapter.timeline


@pytest.mark.parametrize("released_before_step", [False, True])
def test_game_session_realizes_timed_translation_tap_as_fractional_thrust(
    released_before_step: bool,
) -> None:
    config = SimulationConfig.from_yaml("sim/game/configs/game_training_rpo_04_rendezvous.yaml")
    state = KeyboardCommandState(pitch=1.0, use_timing_accumulator=True)
    state.accumulate_timed_input(0.02, speed_multiple=10.0, control_mode="ric_translation")
    if released_before_step:
        state.pitch = 0.0
    session, _, _ = _start_game_attempt(
        config,
        command_state=state,
        training_cfg=RPOTrainingConfig.from_metadata(dict(config.scenario.metadata)),
        controlled_object_id="chaser",
        attitude_rate_deg_s=8.0,
        control_mode="ric_translation",
        ric_reference_object_id="target",
    )

    snapshot = session.step(dt_s=1.0)

    expected_acceleration = 2.0e-6 if released_before_step else 1.0e-5
    assert np.linalg.norm(snapshot.applied_thrust["chaser"]) == pytest.approx(expected_acceleration)
    assert state.pitch_sim_s == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("config_path", "coast_dt_s"),
    (
        ("sim/game/configs/game_training_rpo_04_rendezvous.yaml", 0.25),
        ("sim/game/configs/game_training_rpo_bonus_cislunar_rendezvous.yaml", 1.0),
    ),
)
def test_game_pilot_input_transitions_release_fsw_at_next_physics_interval(
    config_path: str,
    coast_dt_s: float,
) -> None:
    config = SimulationConfig.from_yaml(config_path)
    state = KeyboardCommandState(use_timing_accumulator=True)
    session, adapter, _ = _start_game_attempt(
        config,
        command_state=state,
        training_cfg=RPOTrainingConfig.from_metadata(dict(config.scenario.metadata)),
        controlled_object_id="chaser",
        attitude_rate_deg_s=8.0,
        control_mode="ric_translation",
        ric_reference_object_id="target",
    )
    session.step(dt_s=coast_dt_s)

    state.pitch = 1.0
    assert _request_pilot_input_poll_for_transition(
        session,
        adapter,
        state,
        controlled_object_id="chaser",
    )
    assert not _request_pilot_input_poll_for_transition(
        session,
        adapter,
        state,
        controlled_object_id="chaser",
    )
    firing = session.step(dt_s=coast_dt_s)

    state.pitch = 0.0
    assert _request_pilot_input_poll_for_transition(
        session,
        adapter,
        state,
        controlled_object_id="chaser",
    )
    released = session.step(dt_s=coast_dt_s)

    runtime = session._engine.agents["chaser"].flight_software_runtime
    assert np.linalg.norm(firing.applied_thrust["chaser"]) > 0.0
    assert np.linalg.norm(released.applied_thrust["chaser"]) == pytest.approx(0.0)
    assert runtime.evidence.invocations[-1]["task_releases"][0]["release_reasons"] == [
        "publisher_poll"
    ]


def test_game_session_realizes_timed_attitude_thrust_tap_for_accumulated_duration() -> None:
    config = SimulationConfig.from_yaml("sim/game/configs/game_mode_basic.yaml")

    def run(*, burn: bool) -> tuple[KeyboardCommandState, object, object]:
        state = KeyboardCommandState(use_timing_accumulator=True)
        if burn:
            state.firing = True
            state.accumulate_timed_input(0.02, speed_multiple=10.0, control_mode="attitude_thrust")
            state.firing = False
        session, adapter, _ = _start_game_attempt(
            config,
            command_state=state,
            training_cfg=RPOTrainingConfig(enabled=False),
            controlled_object_id="chaser",
            attitude_rate_deg_s=8.0,
            control_mode="attitude_thrust",
            ric_reference_object_id="target",
        )
        snapshot = None
        for _ in range(6):
            snapshot = session.step(dt_s=0.05)
        assert snapshot is not None
        return state, adapter, snapshot

    burn_state, burn_adapter, burn_snapshot = run(burn=True)
    _, _, coast_snapshot = run(burn=False)
    realized_delta_v_km_s = np.linalg.norm(
        burn_snapshot.truth["chaser"][3:6] - coast_snapshot.truth["chaser"][3:6]
    )

    assert realized_delta_v_km_s == pytest.approx(0.00018 * 0.2, rel=1.0e-4)
    assert burn_state.firing_sim_s == pytest.approx(0.0)
    assert burn_adapter.timeline[-1].event.payload.released_actions == ("fire",)


def test_game_delta_v_budget_limits_physical_realization() -> None:
    config = SimulationConfig.from_yaml("sim/game/configs/game_mode_basic.yaml")
    session, _, _ = _start_game_attempt(
        config,
        command_state=KeyboardCommandState(firing=True),
        training_cfg=RPOTrainingConfig(enabled=False),
        controlled_object_id="chaser",
        attitude_rate_deg_s=8.0,
        control_mode="attitude_thrust",
        ric_reference_object_id="target",
    )
    runtime = session._engine.agents["chaser"].flight_software_runtime
    runtime.max_delta_v_m_s = 0.001
    snapshot = session.step(dt_s=0.25)

    realized_delta_v_m_s = float(np.linalg.norm(snapshot.applied_thrust["chaser"])) * 0.25 * 1000.0
    assert realized_delta_v_m_s <= 0.001 + 1.0e-12
    assert runtime.used_delta_v_m_s <= 0.001 + 1.0e-12
    assert any(item.saturated for item in runtime.evidence.realizations)


def test_cislunar_game_translation_realizes_moon_centered_radial_thrust() -> None:
    config = SimulationConfig.from_yaml("sim/game/configs/game_training_rpo_bonus_cislunar_rendezvous.yaml")
    training = RPOTrainingConfig.from_metadata(dict(config.scenario.metadata))
    session, _, initial = _start_game_attempt(
        config,
        command_state=KeyboardCommandState(pitch=1.0),
        training_cfg=training,
        controlled_object_id="chaser",
        attitude_rate_deg_s=45.0,
        control_mode="moon_ric_translation",
        ric_reference_object_id="target",
    )

    snapshot = session.step(dt_s=1.0)
    target_state = np.asarray(initial.truth["target"][:6], dtype=float)
    target_moon = target_state - cr3bp_moon_state_km_s()
    expected_radial = target_moon[:3] / np.linalg.norm(target_moon[:3])
    applied = np.asarray(snapshot.applied_thrust["chaser"], dtype=float)

    assert np.linalg.norm(applied) == pytest.approx(1.25e-6)
    assert applied / np.linalg.norm(applied) == pytest.approx(expected_radial, abs=1.0e-12)
