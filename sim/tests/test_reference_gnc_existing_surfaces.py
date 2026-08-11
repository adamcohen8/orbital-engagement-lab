from __future__ import annotations

import numpy as np

from sim.control.attitude.baseline import QuaternionPDController, ReactionWheelPDController
from sim.control.attitude.detumble_pd import ECIDetumblePDController, RICDetumblePDController
from sim.control.attitude.surrogate_snap import SurrogateSnapRICController
from sim.core.models import StateBelief, StateTruth
from sim.mission.modules import EvadeMissionStrategy, InspectMissionStrategy


def _attitude_belief(*, rate: tuple[float, float, float] = (0.01, -0.02, 0.03)) -> StateBelief:
    return StateBelief(
        state=np.array([7000.0, 0.0, 0.0, 0.0, 7.546, 0.0, 1.0, 0.0, 0.0, 0.0, *rate]),
        covariance=np.eye(13),
        last_update_t_s=0.0,
    )


def _truth() -> StateTruth:
    return StateTruth(
        position_eci_km=np.array([7000.0, 0.0, 0.0]),
        velocity_eci_km_s=np.array([0.0, 7.546, 0.0]),
        attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_rate_body_rad_s=np.zeros(3),
        mass_kg=100.0,
        t_s=0.0,
    )


def _wheel_pd() -> ReactionWheelPDController:
    return ReactionWheelPDController(
        wheel_axes_body=np.eye(3),
        wheel_torque_limits_nm=np.array([0.1, 0.1, 0.1]),
        kp=np.array([0.2, 0.2, 0.2]),
        kd=np.array([1.0, 1.0, 1.0]),
    )


def test_quaternion_pd_damps_body_rate_and_respects_torque_limit() -> None:
    controller = QuaternionPDController(kp=0.2, kd=1.0, max_torque_nm=0.01)
    command = controller.act(_attitude_belief(), 0.0, 2.0)
    assert np.dot(command.torque_body_nm, np.array([0.01, -0.02, 0.03])) < 0.0
    assert np.linalg.norm(command.torque_body_nm) <= 0.01 + 1.0e-12


def test_eci_and_ric_detumble_profiles_emit_named_rate_damping_modes() -> None:
    belief = _attitude_belief()
    eci = ECIDetumblePDController(pd=_wheel_pd(), rate_only=True)
    ric = RICDetumblePDController(pd=_wheel_pd(), rate_only=True)
    assert eci.act(belief, 0.0, 2.0).mode_flags["mode"] == "pd_detumble_eci"
    assert ric.act(belief, 0.0, 2.0).mode_flags["mode"] == "pd_detumble_ric"


def test_surrogate_snap_ric_cancels_rate_before_slew() -> None:
    controller = SurrogateSnapRICController(
        desired_attitude_quat_br=np.array([1.0, 0.0, 0.0, 0.0]),
        cancel_rate_mag_rad_s2=0.001,
        default_dt_s=1.0,
    )
    command = controller.act(_attitude_belief(rate=(0.1, 0.0, 0.0)), 0.0, 2.0)
    override = command.mode_flags["attitude_state_override"]
    assert override["phase"] == "rate_cancel"
    assert abs(override["w_next_body_rad_s"][0]) < 0.1


def test_evade_and_inspect_use_target_knowledge_with_opposite_intent() -> None:
    truth = _truth()
    target = StateBelief(
        state=np.array([7001.0, 0.0, 0.0, 0.0, 7.546, 0.0]),
        covariance=np.eye(6),
        last_update_t_s=0.0,
    )
    knowledge = {"target": target}
    evade = EvadeMissionStrategy(target_id="target", max_accel_km_s2=1.0e-6)
    inspect = InspectMissionStrategy(target_id="target", max_accel_km_s2=1.0e-6)

    evade_out = evade.update(truth=truth, own_knowledge=knowledge)
    inspect_out = inspect.update(truth=truth, own_knowledge=knowledge)

    assert evade_out["fallback_thrust_eci_km_s2"][0] < 0.0
    assert inspect_out["desired_state_eci_6"] is not None
    assert np.linalg.norm(inspect_out["desired_attitude_quat_bn"]) == 1.0
