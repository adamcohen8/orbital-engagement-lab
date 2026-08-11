from __future__ import annotations

from sim.game.fsw_inputs import GameSimulationClock


def test_pause_does_not_advance_simulation_or_onboard_time() -> None:
    clock = GameSimulationClock(clock_id="game", tick_period_ns=1_000_000)
    clock.advance_wall_time(0.1)
    before = clock.tag
    clock.set_paused(True)
    assert clock.advance_wall_time(10.0) == 0
    assert clock.tag == before


def test_speed_changes_wall_to_sim_mapping_without_render_frame_dependence() -> None:
    one_frame = GameSimulationClock(clock_id="game", tick_period_ns=1_000_000)
    many_frames = GameSimulationClock(clock_id="game", tick_period_ns=1_000_000)
    for clock in (one_frame, many_frames):
        clock.set_speed_multiplier(4.0)
    one_frame.advance_wall_time(0.25)
    for _ in range(25):
        many_frames.advance_wall_time(0.01)
    assert one_frame.tag == many_frames.tag
