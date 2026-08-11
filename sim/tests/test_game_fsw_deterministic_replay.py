from __future__ import annotations

from sim.flight_software import (
    FlightSoftwareInputBatch,
    GamePilotMode,
    GamePilotReferenceFlightSoftwareStack,
    canonical_json_bytes,
)
from sim.game.fsw_inputs import GamePilotInputAdapter
from sim.game.manual import KeyboardCommandState
from sim.tests.fsw_v2_helpers import SATELLITE_ID, boot_event, clock, ideal_event
from sim.tests.game_fsw_v2_helpers import game_stack_config


def _command_stream(render_steps: int) -> bytes:
    config = game_stack_config(GamePilotMode.TRANSLATION)
    adapter = GamePilotInputAdapter(config.profile, source_id="keyboard", boot_id="boot-wp4")
    stack = GamePilotReferenceFlightSoftwareStack(config)
    stack.boot(boot_event())
    initial = stack.snapshot()
    state = KeyboardCommandState(pitch=1.0, throttle=1.0)
    outputs = []
    for invocation in range(1, 4):
        for _ in range(render_steps):
            pass
        event = adapter.sample(state, at=clock(invocation))
        outputs.append(
            stack.step(FlightSoftwareInputBatch(SATELLITE_ID, invocation, clock(invocation), (ideal_event(invocation, invocation), event)))
        )
    stack.restore(initial)
    return canonical_json_bytes(tuple(output.commands for output in outputs))


def test_identical_input_timeline_reproduces_commands_across_render_rates() -> None:
    assert _command_stream(1) == _command_stream(17)


def test_live_adapter_publishes_only_initial_state_and_transitions() -> None:
    config = game_stack_config(GamePilotMode.TRANSLATION)
    adapter = GamePilotInputAdapter(config.profile, source_id="keyboard", boot_id="boot-wp4")
    state = KeyboardCommandState(pitch=1.0, throttle=1.0)

    assert adapter.sample_if_changed(state, at=clock(1)) is not None
    assert adapter.sample_if_changed(state, at=clock(2)) is None
    state.pitch = -1.0
    assert adapter.sample_if_changed(state, at=clock(3)) is not None
    assert len(adapter.timeline) == 2


def test_live_adapter_consumes_fractional_translation_input_after_key_release() -> None:
    config = game_stack_config(GamePilotMode.TRANSLATION)
    adapter = GamePilotInputAdapter(config.profile, source_id="keyboard", boot_id="boot-wp4")
    state = KeyboardCommandState(pitch=1.0, throttle=1.0, use_timing_accumulator=True)
    state.accumulate_timed_input(0.02, speed_multiple=10.0, control_mode="ric_translation")
    state.pitch = 0.0

    event = adapter.sample_control_interval_if_changed(state, at=clock(1), control_interval_s=1.0)

    assert event is not None
    axes = {axis.control_id: axis.value for axis in event.payload.axes}
    assert axes[config.profile.radial_axis] == 0.2
    assert state.pitch_sim_s == 0.0


def test_live_adapter_scales_timed_firing_throttle_and_publishes_release() -> None:
    config = game_stack_config(GamePilotMode.ATTITUDE_THRUST)
    adapter = GamePilotInputAdapter(config.profile, source_id="keyboard", boot_id="boot-wp4")
    state = KeyboardCommandState(firing=True, throttle=0.5, use_timing_accumulator=True)
    state.accumulate_timed_input(0.02, speed_multiple=10.0, control_mode="attitude_thrust")
    state.firing = False

    fired = adapter.sample_control_interval_if_changed(state, at=clock(1), control_interval_s=1.0)
    released = adapter.sample_control_interval_if_changed(state, at=clock(2), control_interval_s=1.0)

    assert fired is not None
    assert fired.payload.pressed_actions == (config.profile.firing_action,)
    fired_axes = {axis.control_id: axis.value for axis in fired.payload.axes}
    assert fired_axes[config.profile.throttle_axis] == -0.8
    assert released is not None
    assert released.payload.released_actions == (config.profile.firing_action,)
