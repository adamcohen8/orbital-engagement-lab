from __future__ import annotations

import numpy as np
import pytest

from sim.control.attitude.reference_pointing import NadirPointingController, RICAxisPointingController
from sim.control.orbit.reference_rpo import (
    KeepOutStandoffController,
    PassiveSafeRetreatController,
    ProportionalNavigationController,
    RICFlyaroundController,
    RICRelativeHoldController,
    RICWaypointController,
    TerminalBrakingController,
    VBarApproachController,
)
from sim.core.models import StateBelief, StateTruth
from sim.mission.execution.reference_commands import (
    AbortSafeHoldRetreatExecution,
    BurnUntilConditionExecution,
    CommandReplayExecution,
    KeepOutGateExecution,
    OneShotImpulseExecution,
    PulseTrainExecution,
    TimedFiniteBurnExecution,
    WaypointSequencerExecution,
)
from sim.utils.frames import ric_rect_to_curv


def _relative_belief(relative_rect: list[float]) -> StateBelief:
    chief = np.array([7000.0, 0.0, 0.0, 0.0, 7.546, 0.0])
    relative_curv = ric_rect_to_curv(np.asarray(relative_rect, dtype=float), r0_km=7000.0)
    return StateBelief(state=np.hstack((relative_curv, chief)), covariance=np.eye(12), last_update_t_s=0.0)


def _attitude_belief() -> StateBelief:
    return StateBelief(
        state=np.array([7000.0, 0.0, 0.0, 0.0, 7.546, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        covariance=np.eye(13),
        last_update_t_s=0.0,
    )


def _truth(t_s: float = 0.0) -> StateTruth:
    return StateTruth(
        position_eci_km=np.array([7000.0, 0.0, 0.0]),
        velocity_eci_km_s=np.array([0.0, 7.546, 0.0]),
        attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_rate_body_rad_s=np.zeros(3),
        mass_kg=100.0,
        t_s=t_s,
    )


def test_relative_hold_and_vbar_approach_emit_bounded_reference_commands() -> None:
    belief = _relative_belief([0.0, -1.0, 0.0, 0.0, 0.0, 0.0])
    hold = RICRelativeHoldController(max_accel_km_s2=1.0e-6, desired_state_ric=np.zeros(6))
    approach = VBarApproachController(
        max_accel_km_s2=1.0e-6,
        terminal_state_ric=np.zeros(6),
        approach_speed_m_s=0.1,
    )

    hold_command = hold.act(belief, 0.0, 2.0)
    approach_command = approach.act(belief, 0.0, 2.0)

    assert hold_command.mode_flags["mode"] == "ric_relative_hold"
    assert approach_command.mode_flags["mode"] == "i_bar_approach"
    assert approach_command.mode_flags["desired_velocity_ric_km_s"][1] > 0.0
    assert np.linalg.norm(approach_command.thrust_eci_km_s2) <= 1.0e-6 + 1.0e-15


def test_waypoint_advances_only_inside_position_and_velocity_gate() -> None:
    controller = RICWaypointController(
        waypoints_ric=[[0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]],
        max_accel_km_s2=1.0e-6,
    )
    command = controller.act(_relative_belief([0, 0, 0, 0, 0, 0]), 0.0, 2.0)
    assert command.mode_flags["waypoint_advanced"] is True
    assert command.mode_flags["waypoint_index"] == 1


def test_keep_out_retreat_and_terminal_braking_have_auditable_phases() -> None:
    inside = _relative_belief([0.05, 0, 0, -0.0002, 0, 0])
    keepout = KeepOutStandoffController(keep_out_radius_km=0.1, max_accel_km_s2=1.0e-6)
    retreat = PassiveSafeRetreatController(max_accel_km_s2=1.0e-6)
    braking = TerminalBrakingController(max_accel_km_s2=1.0e-6)

    assert keepout.act(inside, 0.0, 2.0).mode_flags["phase"] == "protect"
    assert retreat.act(inside, 0.0, 2.0).mode_flags["phase"] == "retreat_burn"
    assert braking.act(inside, 0.0, 2.0).mode_flags["closing_speed_m_s"] > 0.0


def test_flyaround_and_proportional_navigation_emit_bounded_guidance() -> None:
    belief = _relative_belief([0.1, -1.0, 0.0, 0.0, 0.0001, 0.0])
    flyaround = RICFlyaroundController(radius_km=1.0, max_accel_km_s2=1.0e-6)
    proportional = ProportionalNavigationController(max_accel_km_s2=1.0e-6)
    fly_command = flyaround.act(belief, 0.0, 2.0)
    pn_command = proportional.act(belief, 0.0, 2.0)
    assert fly_command.mode_flags["mode"] == "ric_flyaround"
    assert pn_command.mode_flags["mode"] == "proportional_navigation_pursuit"
    assert np.linalg.norm(pn_command.thrust_eci_km_s2) <= 1.0e-6 + 1.0e-15


def test_reference_pointing_generates_finite_quaternion_targets() -> None:
    nadir = NadirPointingController(kp=0.1, kd=0.1, max_torque_nm=0.05)
    ric = RICAxisPointingController(ric_direction=np.array([0.0, 1.0, 0.0]))
    for controller in (nadir, ric):
        command = controller.act(_attitude_belief(), 0.0, 2.0)
        desired = np.asarray(command.mode_flags["desired_attitude_quat_bn"])
        assert np.linalg.norm(desired) == pytest.approx(1.0)
        assert np.all(np.isfinite(command.torque_body_nm))


def test_timed_impulse_pulse_and_sequence_boundaries() -> None:
    timed = TimedFiniteBurnExecution(start_time_s=1.0, duration_s=2.0, acceleration=[1e-6, 0, 0])
    assert timed.update(intent={}, truth=_truth(), t_s=0.0)["mission_mode"]["phase"] == "wait"
    assert timed.update(intent={}, truth=_truth(1.0), t_s=1.0)["mission_mode"]["phase"] == "burn"
    assert timed.update(intent={}, truth=_truth(3.0), t_s=3.0)["mission_mode"]["phase"] == "complete"

    impulse = OneShotImpulseExecution(impulse_time_s=0.5, delta_v_m_s=[1.0, 0, 0], equivalent_duration_s=2.0)
    fired = impulse.update(intent={}, truth=_truth(), t_s=0.0, dt_s=1.0)
    assert fired["thrust_eci_km_s2"][0] == pytest.approx(0.0005)
    assert impulse.update(intent={}, truth=_truth(1.0), t_s=1.0, dt_s=1.0)["mission_mode"]["phase"] == "complete"

    pulse = PulseTrainExecution(acceleration=[1e-6, 0, 0], period_s=10.0, pulse_width_s=2.0)
    assert pulse.update(intent={}, truth=_truth(), t_s=1.0)["mission_mode"]["phase"] == "pulse"
    assert pulse.update(intent={}, truth=_truth(5.0), t_s=5.0)["mission_mode"]["phase"] == "coast"

    sequence = WaypointSequencerExecution(
        phases=[
            {"name": "burn", "duration_s": 2.0, "acceleration": [1e-6, 0, 0]},
            {"name": "coast", "duration_s": 2.0, "acceleration": [0, 0, 0]},
        ]
    )
    assert sequence.update(intent={}, truth=_truth(), t_s=0.0)["mission_mode"]["phase"] == "burn"
    row = sequence.update(intent={}, truth=_truth(2.0), t_s=2.0)
    assert row["mission_mode"]["phase"] == "coast"
    assert row["mission_mode"]["phase_advanced"] is True


def test_abort_burn_until_and_replay_are_deterministic() -> None:
    abort = AbortSafeHoldRetreatExecution(retreat_acceleration_ric_km_s2=[0, -1e-6, 0])
    safe = abort.update(intent={"abort_requested": True}, truth=_truth())
    retreat = abort.update(intent={"abort_requested": True, "retreat_requested": True}, truth=_truth())
    assert safe["mission_mode"]["phase"] == "safe_hold"
    assert np.linalg.norm(retreat["thrust_eci_km_s2"]) > 0.0

    burn = BurnUntilConditionExecution(acceleration=[1e-6, 0, 0], max_duration_s=2.0)
    assert burn.update(intent={}, truth=_truth(), t_s=0.0)["mission_mode"]["phase"] == "burn"
    assert burn.update(intent={}, truth=_truth(2.0), t_s=2.0)["mission_mode"]["phase"] == "complete"

    replay = CommandReplayExecution(rows=[{"time_s": 1.0, "thrust_eci_km_s2": [1e-6, 0, 0]}])
    assert replay.update(intent={}, t_s=0.0)["mission_mode"]["phase"] == "wait"
    assert replay.update(intent={}, t_s=1.0)["thrust_eci_km_s2"][0] == pytest.approx(1e-6)

    target = StateBelief(state=np.array([6999.95, 0, 0, 0, 0, 0]), covariance=np.eye(6), last_update_t_s=0)
    keepout = KeepOutGateExecution(target_id="target", keep_out_radius_km=0.1, retreat_accel_km_s2=1e-6)
    gated = keepout.update(intent={}, truth=_truth(), own_knowledge={"target": target})
    assert gated["mission_mode"]["phase"] == "override"
    assert gated["command_mode_flags"]["gate_reason"] == "keep_out_override"


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: VBarApproachController(max_accel_km_s2=-1.0), "non-negative"),
        (lambda: RICWaypointController(waypoints_ric=[], max_accel_km_s2=1e-6), "at least one"),
        (lambda: PulseTrainExecution(acceleration=[1, 0, 0], period_s=1.0, pulse_width_s=2.0), "pulse_width"),
    ],
)
def test_reference_primitives_reject_invalid_authority_and_sequences(factory, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()
