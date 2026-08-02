from __future__ import annotations

import numpy as np

from sim.control.attitude.replay import AttitudeReplayController
from sim.control.attitude.snap import SnapAttitudeController
from sim.control.attitude.snap_hold import SnapAndHoldRICAttitudeController
from sim.core.models import StateBelief


def _belief() -> StateBelief:
    return StateBelief(
        state=np.hstack(
            (
                np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0]),
                np.array([1.0, 0.0, 0.0, 0.0]),
                np.zeros(3),
            )
        ),
        covariance=np.eye(13),
        last_update_t_s=0.0,
    )


def test_snap_attitude_emits_consumed_state_override() -> None:
    controller = SnapAttitudeController(
        desired_state6=np.array([np.pi, 0.0, 0.0, 0.1, -0.2, 0.3]),
    )

    command = controller.act(_belief(), t_s=0.0, budget_ms=1.0)
    override = command.mode_flags["attitude_state_override"]

    np.testing.assert_allclose(np.abs(override["q_next_bn"]), [0.0, 1.0, 0.0, 0.0], atol=1.0e-12)
    np.testing.assert_allclose(override["w_next_body_rad_s"], [0.1, -0.2, 0.3], atol=0.0)


def test_snap_hold_ric_emits_consumed_state_override() -> None:
    controller = SnapAndHoldRICAttitudeController(
        desired_state6_ric=np.array([0.0, 0.0, 0.0, 0.1, -0.2, 0.3]),
    )

    command = controller.act(_belief(), t_s=0.0, budget_ms=1.0)
    override = command.mode_flags["attitude_state_override"]

    np.testing.assert_allclose(np.abs(override["q_next_bn"]), [1.0, 0.0, 0.0, 0.0], atol=1.0e-12)
    np.testing.assert_allclose(override["w_next_body_rad_s"], [0.1, -0.2, 0.3], atol=0.0)


def test_replay_samples_the_actuation_interval_endpoint() -> None:
    controller = AttitudeReplayController(
        times_s=[0.0, 10.0],
        attitude_quat_bn=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        angular_rate_body_rad_s=[[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]],
    )
    controller.set_actuation_interval(0.0, 10.0)

    command = controller.act(_belief(), t_s=0.0, budget_ms=1.0)
    override = command.mode_flags["attitude_state_override"]

    np.testing.assert_allclose(np.abs(override["q_next_bn"]), [0.0, 1.0, 0.0, 0.0], atol=1.0e-12)
    np.testing.assert_allclose(override["w_next_body_rad_s"], [0.1, 0.2, 0.3], atol=0.0)
