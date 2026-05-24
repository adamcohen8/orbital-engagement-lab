from __future__ import annotations

import unittest

import numpy as np

from sim.actuators.attitude import AttitudeActuator, ReactionWheelLimits
from sim.actuators.combined import CombinedActuator
from sim.actuators.faults import ActuatorFaultConfig, apply_actuator_faults
from sim.actuators.orbital import (
    ElectricPropulsionLimits,
    GimbaledThrusterLimits,
    OrbitalActuator,
    OrbitalActuatorLimits,
    RcsClusterLimits,
    RcsThruster,
)
from sim.actuators.presets import BASIC_RCS_6DOF
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

        self.assertTrue(
            np.allclose(np.array(out.thrust_eci_km_s2, dtype=float), np.array([0.001, 0.0, 0.0], dtype=float))
        )
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

        self.assertTrue(
            np.allclose(np.array(out.thrust_eci_km_s2, dtype=float), np.array([0.0, 0.0, -0.002], dtype=float))
        )

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

    def test_rcs_cluster_allocates_body_force_and_torque(self):
        actuator = OrbitalActuator()
        cluster = RcsClusterLimits(
            thrusters=(
                RcsThruster(
                    name="plus-x",
                    position_body_m=np.array([0.0, 1.0, 0.0], dtype=float),
                    force_direction_body=np.array([1.0, 0.0, 0.0], dtype=float),
                    max_thrust_n=2.0,
                    isp_s=230.0,
                ),
                RcsThruster(
                    name="plus-y",
                    position_body_m=np.array([1.0, 0.0, 0.0], dtype=float),
                    force_direction_body=np.array([0.0, 1.0, 0.0], dtype=float),
                    max_thrust_n=2.0,
                    isp_s=230.0,
                ),
            ),
            allocation_mode="torque_only",
        )
        limits = {"orbital": OrbitalActuatorLimits(max_accel_km_s2=1.0, rcs_cluster=cluster)}
        command = Command(
            thrust_eci_km_s2=np.zeros(3, dtype=float),
            torque_body_nm=np.array([0.0, 0.0, -1.0], dtype=float),
            mode_flags={
                "current_mass_kg": 1000.0,
                "current_attitude_quat_bn": np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            },
        )

        out = actuator.apply(command, limits, dt_s=1.0)

        forces = np.array(out.mode_flags["rcs_thruster_forces_n"], dtype=float)
        self.assertEqual(forces.size, 2)
        self.assertTrue(np.all(forces >= 0.0))
        self.assertLessEqual(float(np.max(forces)), 2.0)
        self.assertTrue(np.linalg.norm(out.thrust_eci_km_s2) > 0.0)
        self.assertTrue(np.linalg.norm(out.torque_body_nm) > 0.0)

    def test_basic_rcs_6dof_preset_has_full_force_and_torque_authority(self):
        thrusters = BASIC_RCS_6DOF["orbital"]["rcs_cluster"]["thrusters"]
        columns = []
        for thruster in thrusters:
            force_dir = np.array(thruster["force_direction_body"], dtype=float).reshape(3)
            force_dir = force_dir / np.linalg.norm(force_dir)
            position = np.array(thruster["position_body_m"], dtype=float).reshape(3)
            columns.append(np.hstack((force_dir, np.cross(position, force_dir))))
        allocation = np.column_stack(columns)

        self.assertEqual(np.linalg.matrix_rank(allocation), 6)
        self.assertEqual(np.linalg.matrix_rank(allocation[3:, :]), 3)

        cluster = RcsClusterLimits(
            thrusters=tuple(
                RcsThruster(
                    name=str(thruster["name"]),
                    position_body_m=np.array(thruster["position_body_m"], dtype=float),
                    force_direction_body=np.array(thruster["force_direction_body"], dtype=float),
                    max_thrust_n=float(thruster["max_thrust_n"]),
                    min_impulse_bit_n_s=float(thruster.get("min_impulse_bit_n_s", 0.0)),
                    isp_s=float(thruster.get("isp_s", 230.0)),
                )
                for thruster in thrusters
            ),
            allocation_mode="force_torque",
        )
        actuator = OrbitalActuator()
        limits = {"orbital": OrbitalActuatorLimits(max_accel_km_s2=1.0, rcs_cluster=cluster)}
        mass_kg = 1000.0
        for axis in range(6):
            target = np.zeros(6, dtype=float)
            target[axis] = 0.1 if axis < 3 else 0.05
            out = actuator.apply(
                Command(
                    thrust_eci_km_s2=target[:3] / mass_kg / 1e3,
                    torque_body_nm=target[3:],
                    mode_flags={
                        "current_mass_kg": mass_kg,
                        "current_attitude_quat_bn": np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
                    },
                ),
                limits,
                dt_s=1.0,
            )
            achieved = np.hstack((out.mode_flags["rcs_force_body_n"], out.mode_flags["rcs_torque_body_nm"]))
            self.assertTrue(np.allclose(achieved, target, atol=1e-12))

    def test_electric_propulsion_power_limits_thrust_and_tracks_propellant(self):
        actuator = OrbitalActuator()
        limits = {
            "orbital": OrbitalActuatorLimits(
                max_accel_km_s2=1.0,
                max_throttle_rate_km_s2_s=1.0,
                electric_propulsion=ElectricPropulsionLimits(
                    max_thrust_n=1.0,
                    isp_s=1600.0,
                    max_power_w=100.0,
                    power_per_newton_w=200.0,
                ),
            )
        }
        command = Command(
            thrust_eci_km_s2=np.array([0.01, 0.0, 0.0], dtype=float),
            mode_flags={"current_mass_kg": 500.0},
        )

        out = actuator.apply(command, limits, dt_s=2.0)

        self.assertAlmostEqual(float(np.linalg.norm(out.thrust_eci_km_s2)), 1.0e-6, places=12)
        self.assertAlmostEqual(float(out.mode_flags["electric_propulsion_max_thrust_n"]), 0.5, places=12)
        self.assertGreater(float(out.mode_flags["electric_propulsion_delta_mass_kg"]), 0.0)
        self.assertAlmostEqual(
            float(out.mode_flags["delta_mass_kg"]),
            float(out.mode_flags["electric_propulsion_delta_mass_kg"]),
            places=12,
        )

    def test_gimbaled_thruster_slews_toward_desired_force_axis(self):
        actuator = OrbitalActuator()
        limits = {
            "orbital": OrbitalActuatorLimits(
                max_accel_km_s2=1.0,
                max_throttle_rate_km_s2_s=1.0,
                isp_s=250.0,
                gimbaled_thruster=GimbaledThrusterLimits(
                    neutral_direction_body=np.array([-1.0, 0.0, 0.0], dtype=float),
                    position_body_m=np.array([0.0, 0.2, 0.0], dtype=float),
                    max_gimbal_angle_rad=np.deg2rad(10.0),
                    max_gimbal_rate_rad_s=np.deg2rad(5.0),
                ),
            )
        }
        command = Command(
            thrust_eci_km_s2=np.array([0.0, 0.001, 0.0], dtype=float),
            mode_flags={
                "current_mass_kg": 500.0,
                "current_attitude_quat_bn": np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            },
        )

        out = actuator.apply(command, limits, dt_s=1.0)

        self.assertIn("gimbal_direction_body", out.mode_flags)
        self.assertLessEqual(float(out.mode_flags["gimbal_angle_rad"]), np.deg2rad(5.0) + 1e-12)
        self.assertTrue(bool(out.mode_flags["gimbal_rate_limited"]))

    def test_actuator_fault_layer_scales_and_biases_applied_command(self):
        command = Command(
            thrust_eci_km_s2=np.array([1.0, 0.0, 0.0], dtype=float),
            torque_body_nm=np.array([0.0, 1.0, 0.0], dtype=float),
        )
        out = apply_actuator_faults(
            command,
            ActuatorFaultConfig(
                thrust_scale=0.5,
                torque_scale=0.25,
                thrust_bias_eci_km_s2=np.array([0.1, 0.0, 0.0], dtype=float),
                torque_bias_body_nm=np.array([0.0, 0.1, 0.0], dtype=float),
            ),
        )

        self.assertTrue(np.allclose(out.thrust_eci_km_s2, np.array([0.6, 0.0, 0.0], dtype=float)))
        self.assertTrue(np.allclose(out.torque_body_nm, np.array([0.0, 0.35, 0.0], dtype=float)))


if __name__ == "__main__":
    unittest.main()
