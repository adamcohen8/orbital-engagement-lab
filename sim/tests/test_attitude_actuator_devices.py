from __future__ import annotations

import unittest

import numpy as np

from sim.actuators.attitude import (
    AttitudeActuator,
    ControlMomentGyroLimits,
    MagnetorquerLimits,
    ReactionWheelLimits,
    WheelDesaturationLimits,
)
from sim.core.models import Command


class TestAttitudeActuatorDevices(unittest.TestCase):
    def test_magnetorquer_uses_b_cross_m_physical_torque(self):
        actuator = AttitudeActuator(magnetorquers=MagnetorquerLimits(max_dipole_a_m2=np.array([10.0, 10.0, 10.0])))
        command = Command(
            torque_body_nm=np.array([1.0e-5, 0.0, 0.0], dtype=float),
            mode_flags={"magnetic_field_body_t": np.array([0.0, 0.0, 5.0e-5], dtype=float)},
        )

        out = actuator.apply(command, limits={}, dt_s=1.0)

        self.assertTrue(np.allclose(out.torque_body_nm, command.torque_body_nm, atol=1e-12))
        self.assertEqual(out.mode_flags["magnetorquer_mode"], "physical_b_cross_m")
        self.assertIn("magnetorquer_dipole_cmd_a_m2", out.mode_flags)

    def test_control_moment_gyro_limits_torque_by_momentum_and_gimbal_rate(self):
        actuator = AttitudeActuator(
            control_moment_gyros=ControlMomentGyroLimits(
                max_torque_nm=np.array([1.0, 1.0, 1.0], dtype=float),
                momentum_nms=np.array([2.0, 2.0, 2.0], dtype=float),
                gimbal_rate_limit_rad_s=np.array([0.1, 0.1, 0.1], dtype=float),
            )
        )
        command = Command(torque_body_nm=np.array([0.5, 0.0, 0.0], dtype=float))

        out = actuator.apply(command, limits={}, dt_s=1.0)

        self.assertTrue(np.allclose(out.torque_body_nm, np.array([0.2, 0.0, 0.0], dtype=float)))
        self.assertTrue(np.allclose(out.mode_flags["cmg_torque_cap_nm"], [0.2, 0.2, 0.2]))

    def test_wheel_desaturation_adds_external_unload_torque(self):
        actuator = AttitudeActuator(
            reaction_wheels=ReactionWheelLimits(
                max_torque_nm=np.array([0.1, 0.1, 0.1], dtype=float),
                max_momentum_nms=np.array([1.0, 1.0, 1.0], dtype=float),
                wheel_axes_body=np.eye(3),
            ),
            wheel_desaturation=WheelDesaturationLimits(
                momentum_fraction_threshold=0.1,
                unload_gain_s_inv=1.0,
                max_unload_torque_nm=0.05,
            ),
        )
        actuator.wheel_momentum_nms = np.array([1.0, 0.0, 0.0], dtype=float)

        out = actuator.apply(Command(torque_body_nm=np.zeros(3)), limits={}, dt_s=1.0)

        self.assertTrue(bool(out.mode_flags["wheel_desaturation_active"]))
        self.assertLess(float(out.torque_body_nm[0]), 0.0)
        self.assertAlmostEqual(float(np.linalg.norm(out.torque_body_nm)), 0.05, places=8)


if __name__ == "__main__":
    unittest.main()
