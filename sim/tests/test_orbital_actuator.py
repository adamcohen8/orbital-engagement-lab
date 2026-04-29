from __future__ import annotations

import unittest

import numpy as np

from sim.actuators.attitude import AttitudeActuator, ReactionWheelLimits
from sim.actuators.combined import CombinedActuator
from sim.actuators.orbital import OrbitalActuator, OrbitalActuatorLimits
from sim.core.models import Command


class TestOrbitalActuator(unittest.TestCase):
    def test_delta_mass_scales_with_current_vehicle_mass(self):
        actuator = OrbitalActuator()
        limits = {
            "orbital": OrbitalActuatorLimits(
                max_accel_km_s2=1.0,
                max_throttle_rate_km_s2_s=1.0,
                isp_s=250.0,
            )
        }
        command = Command(
            thrust_eci_km_s2=np.array([0.002, 0.0, 0.0], dtype=float),
            mode_flags={"current_mass_kg": 500.0},
        )

        out = actuator.apply(command, limits, dt_s=4.0)

        expected = 500.0 * 2.0 / (250.0 * 9.80665) * 4.0
        self.assertAlmostEqual(float(out.mode_flags["delta_mass_kg"]), expected, places=10)

    def test_max_thrust_n_limits_applied_accel_by_current_mass(self):
        actuator = OrbitalActuator()
        limits = {
            "orbital": OrbitalActuatorLimits(
                max_accel_km_s2=10.0,
                max_thrust_n=500.0,
                max_throttle_rate_km_s2_s=10.0,
                isp_s=250.0,
            )
        }
        command = Command(
            thrust_eci_km_s2=np.array([0.002, 0.0, 0.0], dtype=float),
            mode_flags={"current_mass_kg": 500.0},
        )

        out = actuator.apply(command, limits, dt_s=1.0)

        self.assertTrue(np.allclose(np.array(out.thrust_eci_km_s2, dtype=float), np.array([0.001, 0.0, 0.0], dtype=float)))
        self.assertAlmostEqual(float(out.mode_flags["effective_max_accel_km_s2"]), 0.001, places=12)

    def test_orbital_actuator_couples_applied_thrust_to_current_attitude(self):
        actuator = OrbitalActuator()
        limits = {
            "orbital": OrbitalActuatorLimits(
                max_accel_km_s2=1.0,
                max_throttle_rate_km_s2_s=1.0,
                isp_s=250.0,
                thruster_direction_body=np.array([0.0, 0.0, 1.0], dtype=float),
            )
        }
        command = Command(
            thrust_eci_km_s2=np.array([0.002, 0.0, 0.0], dtype=float),
            mode_flags={
                "current_mass_kg": 500.0,
                "current_attitude_quat_bn": np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            },
        )

        out = actuator.apply(command, limits, dt_s=1.0)

        self.assertTrue(np.allclose(np.array(out.thrust_eci_km_s2, dtype=float), np.array([0.0, 0.0, -0.002], dtype=float)))

    def test_orbital_actuator_adds_thruster_mount_torque(self):
        actuator = OrbitalActuator()
        limits = {
            "orbital": OrbitalActuatorLimits(
                max_accel_km_s2=1.0,
                max_throttle_rate_km_s2_s=1.0,
                isp_s=250.0,
                thruster_direction_body=np.array([0.0, 0.0, 1.0], dtype=float),
                thruster_position_body_m=np.array([0.2, 0.0, 0.0], dtype=float),
            )
        }
        command = Command(
            thrust_eci_km_s2=np.array([2.0e-6, 0.0, 0.0], dtype=float),
            torque_body_nm=np.array([0.1, 0.0, 0.0], dtype=float),
            mode_flags={
                "current_mass_kg": 500.0,
                "current_attitude_quat_bn": np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            },
        )

        out = actuator.apply(command, limits, dt_s=1.0)

        self.assertTrue(np.allclose(np.array(out.torque_body_nm, dtype=float), np.array([0.1, 0.2, 0.0], dtype=float)))

    def test_combined_actuator_preserves_thruster_torque_after_attitude_limits(self):
        actuator = CombinedActuator(
            orbital=OrbitalActuator(),
            attitude=AttitudeActuator(
                reaction_wheels=ReactionWheelLimits(
                    max_torque_nm=np.array([0.1, 0.1, 0.1], dtype=float),
                    max_momentum_nms=np.array([1.0, 1.0, 1.0], dtype=float),
                )
            ),
        )
        limits = {
            "orbital": OrbitalActuatorLimits(
                max_accel_km_s2=1.0,
                max_throttle_rate_km_s2_s=1.0,
                isp_s=250.0,
                thruster_direction_body=np.array([0.0, 0.0, 1.0], dtype=float),
                thruster_position_body_m=np.array([0.2, 0.0, 0.0], dtype=float),
            )
        }
        command = Command(
            thrust_eci_km_s2=np.array([2.0e-6, 0.0, 0.0], dtype=float),
            torque_body_nm=np.array([0.2, 0.0, 0.0], dtype=float),
            mode_flags={
                "current_mass_kg": 500.0,
                "current_attitude_quat_bn": np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            },
        )

        out = actuator.apply(command, limits, dt_s=1.0)

        self.assertTrue(np.allclose(np.array(out.torque_body_nm, dtype=float), np.array([0.1, 0.2, 0.0], dtype=float)))


if __name__ == "__main__":
    unittest.main()
