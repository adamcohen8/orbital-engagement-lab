from __future__ import annotations

import numpy as np

from sim.control.attitude.bdot_magnetorquer import MagnetorquerBdotController
from sim.control.attitude.cmg_steering import CMGSteeringController
from sim.control.attitude.wheel_desaturation import WheelDesaturationController
from sim.control.orbit.electric_propulsion import ElectricPropulsionController
from sim.control.orbit.gimbaled_thruster import GimbaledThrusterController
from sim.control.orbit.rcs_allocator import RCSAllocationAwareController
from sim.core.models import Command, StateBelief


class _ConstantController:
    def __init__(self, thrust=None, torque=None):
        self.thrust = np.zeros(3, dtype=float) if thrust is None else np.array(thrust, dtype=float)
        self.torque = np.zeros(3, dtype=float) if torque is None else np.array(torque, dtype=float)

    def act(self, belief, t_s, budget_ms):
        return Command(thrust_eci_km_s2=self.thrust.copy(), torque_body_nm=self.torque.copy(), mode_flags={"mode": "base"})


def _joint_belief(*, omega=(0.0, 0.0, 0.0), extra=()) -> StateBelief:
    state = np.hstack(
        (
            np.array([7000.0, 0.0, 0.0], dtype=float),
            np.array([0.0, 7.5, 0.0], dtype=float),
            np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            np.array(omega, dtype=float),
            np.array(extra, dtype=float),
        )
    )
    return StateBelief(state=state, covariance=np.eye(state.size), last_update_t_s=0.0)


def test_magnetorquer_bdot_outputs_b_field_aware_torque() -> None:
    ctrl = MagnetorquerBdotController(magnetic_field_body_t=np.array([0.0, 0.0, 5.0e-5]), gain=1.0e5)

    cmd = ctrl.act(_joint_belief(omega=(0.01, 0.0, 0.02)), t_s=0.0, budget_ms=1.0)

    assert cmd.mode_flags["mode"] == "magnetorquer_bdot"
    assert "magnetic_field_body_t" in cmd.mode_flags
    assert abs(float(cmd.torque_body_nm[2])) < 1e-15
    assert np.linalg.norm(cmd.torque_body_nm) > 0.0


def test_wheel_desaturation_reads_extended_state_momentum() -> None:
    ctrl = WheelDesaturationController(momentum_state_slice=(13, 16), momentum_threshold_nms=0.1, max_unload_torque_nm=0.02)

    cmd = ctrl.act(_joint_belief(extra=(1.0, 0.0, 0.0)), t_s=0.0, budget_ms=1.0)

    assert cmd.mode_flags["wheel_desaturation_requested"] is True
    assert np.allclose(cmd.torque_body_nm, np.array([-0.02, 0.0, 0.0]))


def test_cmg_steering_wraps_and_caps_base_torque() -> None:
    ctrl = CMGSteeringController(
        base_controller=_ConstantController(torque=[1.0, -1.0, 0.1]),
        max_torque_nm=[0.5, 0.5, 0.5],
        momentum_nms=[2.0, 2.0, 2.0],
        gimbal_rate_limit_rad_s=[0.1, 0.2, 0.3],
    )

    cmd = ctrl.act(_joint_belief(), t_s=0.0, budget_ms=1.0)

    assert cmd.mode_flags["mode"] == "cmg_steering"
    assert np.allclose(cmd.torque_body_nm, np.array([0.2, -0.4, 0.1]))


def test_rcs_allocation_controller_outputs_achievable_acceleration() -> None:
    ctrl = RCSAllocationAwareController(
        base_controller=_ConstantController(thrust=[1.0e-3, 0.0, 0.0]),
        mass_kg=1000.0,
        thrusters=[
            {"name": "x", "position_body_m": [0.0, 0.0, 0.0], "force_direction_body": [1.0, 0.0, 0.0], "max_thrust_n": 0.5}
        ],
    )

    cmd = ctrl.act(_joint_belief(), t_s=0.0, budget_ms=1.0)

    assert cmd.mode_flags["mode"] == "rcs_allocation_aware"
    assert np.allclose(cmd.thrust_eci_km_s2, np.array([5.0e-7, 0.0, 0.0]))
    assert cmd.mode_flags["rcs_thruster_forces_n"] == [0.5]


def test_electric_propulsion_controller_caps_power_limited_thrust() -> None:
    ctrl = ElectricPropulsionController(
        base_controller=_ConstantController(thrust=[1.0e-3, 0.0, 0.0]),
        mass_kg=500.0,
        max_thrust_n=2.0,
        max_power_w=100.0,
        power_per_newton_w=200.0,
    )

    cmd = ctrl.act(_joint_belief(), t_s=0.0, budget_ms=1.0)

    assert cmd.mode_flags["mode"] == "electric_propulsion_guidance"
    assert np.allclose(cmd.thrust_eci_km_s2, np.array([1.0e-6, 0.0, 0.0]))


def test_gimbaled_thruster_controller_blanks_unreachable_direction() -> None:
    ctrl = GimbaledThrusterController(
        base_controller=_ConstantController(thrust=[0.0, 1.0e-3, 0.0]),
        neutral_direction_body=[-1.0, 0.0, 0.0],
        max_gimbal_angle_rad=np.deg2rad(5.0),
    )

    cmd = ctrl.act(_joint_belief(), t_s=0.0, budget_ms=1.0)

    assert cmd.mode_flags["mode"] == "gimbaled_thruster_guidance"
    assert np.allclose(cmd.thrust_eci_km_s2, np.zeros(3))
    assert cmd.mode_flags["gimbal_angle_request_rad"] > np.deg2rad(5.0)
