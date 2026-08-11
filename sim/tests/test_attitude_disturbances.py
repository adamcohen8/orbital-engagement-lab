from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from sim.core.models import Command, StateTruth
from sim.dynamics import model as dynamics_model_module
from sim.dynamics.attitude.disturbances import DisturbanceTorqueConfig, DisturbanceTorqueModel
from sim.dynamics.model import OrbitalAttitudeDynamics
from sim.dynamics.orbit.propagator import OrbitPropagator


class _AttitudeCoupledDisturbance:
    def total_torque_body_nm(self, state: StateTruth, env: dict | None = None) -> np.ndarray:
        q = np.array(state.attitude_quat_bn, dtype=float).reshape(4)
        return np.array([0.0, 0.0, 50.0 * q[2]], dtype=float)


class _MidpointRecordingDisturbance:
    def __init__(self) -> None:
        self.samples: list[tuple[np.ndarray, np.ndarray, float]] = []

    def total_torque_body_nm(self, state: StateTruth, env: dict | None = None) -> np.ndarray:
        self.samples.append(
            (
                np.array(state.position_eci_km, dtype=float),
                np.array(state.velocity_eci_km_s, dtype=float),
                float(state.t_s),
            )
        )
        return np.zeros(3, dtype=float)


class _EnvironmentRecordingPropagator:
    def __init__(self) -> None:
        self.environment: dict | None = None

    def propagate(self, *, x_eci, env, **kwargs):
        self.environment = dict(env)
        return np.array(x_eci, dtype=float)


class TestAttitudeDisturbances(unittest.TestCase):
    def test_owned_default_orbit_propagator_inherits_acceleration_mode(self):
        dynamics = OrbitalAttitudeDynamics(
            mu_km3_s2=398600.4418,
            inertia_kg_m2=np.eye(3),
            acceleration_mode="auto",
        )

        self.assertEqual(dynamics.orbit_propagator.acceleration_mode, "auto")

    def test_explicit_orbit_propagator_keeps_its_acceleration_mode(self):
        propagator = OrbitPropagator(integrator="rk4", acceleration_mode="off")
        dynamics = OrbitalAttitudeDynamics(
            mu_km3_s2=398600.4418,
            inertia_kg_m2=np.eye(3),
            orbit_propagator=propagator,
            acceleration_mode="auto",
        )

        self.assertIs(dynamics.orbit_propagator, propagator)
        self.assertEqual(propagator.acceleration_mode, "off")

    def test_reused_default_orbit_propagator_is_treated_as_explicit(self):
        first = OrbitalAttitudeDynamics(
            mu_km3_s2=398600.4418,
            inertia_kg_m2=np.eye(3),
            acceleration_mode="auto",
        )
        propagator = first.orbit_propagator
        propagator.acceleration_mode = "off"

        second = OrbitalAttitudeDynamics(
            mu_km3_s2=398600.4418,
            inertia_kg_m2=np.eye(3),
            orbit_propagator=propagator,
            acceleration_mode="auto",
        )

        self.assertIs(second.orbit_propagator, propagator)
        self.assertEqual(propagator.acceleration_mode, "off")
        self.assertFalse(hasattr(propagator, "_pending_orbital_attitude_default_configuration"))

    def test_legacy_aerodynamic_mode_flags_cannot_override_physics(self):
        propagator = _EnvironmentRecordingPropagator()
        dynamics = OrbitalAttitudeDynamics(
            mu_km3_s2=398600.4418,
            inertia_kg_m2=np.eye(3),
            propagate_attitude=False,
            orbit_propagator=propagator,
        )
        state = StateTruth(
            position_eci_km=np.array([6598.137, 0.0, 0.0]),
            velocity_eci_km_s=np.array([0.0, 7.77, 0.0]),
            attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_rate_body_rad_s=np.zeros(3),
            mass_kg=5000.0,
            t_s=0.0,
        )
        command = Command(
            mode_flags={
                "aero_controlled": True,
                "aero_drag_area_m2": 20.0,
                "aero_lift_area_m2": 32.0,
                "aero_lift_coefficient": 0.65,
                "aero_lift_direction_eci": [0.0, 0.0, 2.0],
            }
        )

        dynamics.step(state, command, env={}, dt_s=1.0)

        assert propagator.environment is not None
        self.assertNotIn("drag_area_m2", propagator.environment)
        self.assertNotIn("lift_area_m2", propagator.environment)
        self.assertNotIn("lift_coefficient", propagator.environment)
        self.assertNotIn("lift_direction_eci", propagator.environment)

    def test_disturbance_torque_nonzero_for_representative_state(self):
        inertia = np.diag([120.0, 100.0, 80.0])
        state = StateTruth(
            position_eci_km=np.array([6778.0, 0.0, 0.0]),
            velocity_eci_km_s=np.array([0.0, 7.67, 0.1]),
            attitude_quat_bn=np.array([0.9, 0.2, -0.1, 0.35]),
            angular_rate_body_rad_s=np.array([0.01, -0.015, 0.02]),
            mass_kg=300.0,
            t_s=0.0,
        )
        model = DisturbanceTorqueModel(
            mu_km3_s2=398600.4418,
            inertia_kg_m2=inertia,
            config=DisturbanceTorqueConfig(),
        )

        tau = model.total_torque_body_nm(state)
        self.assertGreater(np.linalg.norm(tau), 0.0)

    def test_magnetic_dipole_field_has_leo_order_of_magnitude(self):
        inertia = np.diag([120.0, 100.0, 80.0])
        state = StateTruth(
            position_eci_km=np.array([6378.137, 0.0, 0.0]),
            velocity_eci_km_s=np.array([0.0, 0.0, 0.0]),
            attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_rate_body_rad_s=np.zeros(3),
            mass_kg=300.0,
            t_s=0.0,
        )
        model = DisturbanceTorqueModel(
            mu_km3_s2=398600.4418,
            inertia_kg_m2=inertia,
            config=DisturbanceTorqueConfig(
                use_gravity_gradient=False,
                use_magnetic=True,
                use_drag=False,
                use_srp=False,
                magnetic_dipole_body_a_m2=np.array([1.0, 0.0, 0.0]),
            ),
        )

        tau = model.total_torque_body_nm(state)

        self.assertGreater(float(np.linalg.norm(tau)), 1e-5)
        self.assertLess(float(np.linalg.norm(tau)), 1e-4)

    def test_dynamics_with_disturbances_changes_angular_rate(self):
        inertia = np.diag([120.0, 100.0, 80.0])
        state = StateTruth(
            position_eci_km=np.array([6778.0, 0.0, 0.0]),
            velocity_eci_km_s=np.array([0.0, 7.67, 0.1]),
            attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_rate_body_rad_s=np.array([0.01, -0.015, 0.02]),
            mass_kg=300.0,
            t_s=0.0,
        )

        no_dist = OrbitalAttitudeDynamics(mu_km3_s2=398600.4418, inertia_kg_m2=inertia)
        with_dist = OrbitalAttitudeDynamics(
            mu_km3_s2=398600.4418,
            inertia_kg_m2=inertia,
            disturbance_model=DisturbanceTorqueModel(
                mu_km3_s2=398600.4418,
                inertia_kg_m2=inertia,
                config=DisturbanceTorqueConfig(),
            ),
        )

        command = Command.zero()
        x_no = no_dist.step(state.copy(), command, env={}, dt_s=2.0)
        x_yes = with_dist.step(state.copy(), command, env={}, dt_s=2.0)
        diff_norm = np.linalg.norm(x_yes.angular_rate_body_rad_s - x_no.angular_rate_body_rad_s)
        self.assertGreater(diff_norm, 0.0)

    def test_compiled_builtin_disturbance_plan_matches_python_fallback(self):
        inertia = np.diag([120.0, 100.0, 80.0])
        state = StateTruth(
            position_eci_km=np.array([6778.0, 10.0, -20.0]),
            velocity_eci_km_s=np.array([0.0, 7.67, 0.1]),
            attitude_quat_bn=np.array([0.96, 0.1, -0.2, 0.15]),
            angular_rate_body_rad_s=np.array([0.01, -0.015, 0.02]),
            mass_kg=300.0,
            t_s=0.0,
        )
        config = DisturbanceTorqueConfig(
            use_gravity_gradient=True,
            use_magnetic=True,
            use_drag=True,
            use_srp=True,
            magnetic_dipole_body_a_m2=np.array([0.05, -0.02, 0.01]),
        )
        env = {
            "density_kg_m3": 1.0e-12,
            "drag_v_rel_eci_m_s": np.array([10.0, 7200.0, -20.0]),
            "drag_v_rel_norm_m_s": float(np.linalg.norm([10.0, 7200.0, -20.0])),
            "magnetic_field_eci_t": np.array([2.0e-5, -1.0e-5, 3.0e-5]),
            "sun_dir_eci_unit": np.array([1.0, 0.0, 0.0]),
            "srp_shadow_factor": 0.7,
            "srp_pressure_n_m2": 4.56e-6,
            "srp_distance_scale": 0.98,
        }
        python_dynamics = OrbitalAttitudeDynamics(
            mu_km3_s2=398600.4418,
            inertia_kg_m2=inertia,
            disturbance_model=DisturbanceTorqueModel(398600.4418, inertia, config),
            orbit_substep_s=1.0,
            attitude_substep_s=0.2,
            acceleration_mode="off",
        )
        compiled_dynamics = OrbitalAttitudeDynamics(
            mu_km3_s2=398600.4418,
            inertia_kg_m2=inertia,
            disturbance_model=DisturbanceTorqueModel(398600.4418, inertia, config),
            orbit_substep_s=1.0,
            attitude_substep_s=0.2,
            acceleration_mode="auto",
        )

        expected = python_dynamics.step(state.copy(), Command.zero(), env=env, dt_s=1.0)
        actual = compiled_dynamics.step(state.copy(), Command.zero(), env=env, dt_s=1.0)

        np.testing.assert_allclose(actual.attitude_quat_bn, expected.attitude_quat_bn, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            actual.angular_rate_body_rad_s,
            expected.angular_rate_body_rad_s,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_compiled_drag_facets_refresh_after_nested_mapping_mutation(self):
        facet = {
            "area_m2": 1.0,
            "drag_cd": 2.0,
            "normal_body": np.array([1.0, 0.0, 0.0]),
            "cp_offset_body_m": np.array([0.0, 1.0, 0.0]),
        }
        config = DisturbanceTorqueConfig(
            use_gravity_gradient=False,
            use_magnetic=False,
            use_drag=True,
            use_srp=False,
            drag_facets=(facet,),
        )
        model = DisturbanceTorqueModel(398600.4418, np.eye(3), config)
        facet["area_m2"] = 10.0
        expected_model = DisturbanceTorqueModel(398600.4418, np.eye(3), config)
        kwargs = {
            "quat_bn": np.array([1.0, 0.0, 0.0, 0.0]),
            "omega_body_rad_s": np.zeros(3),
            "command_torque_body_nm": np.zeros(3),
            "position_eci_km": np.array([6778.0, 0.0, 0.0]),
            "t_s": 0.0,
            "env": {
                "density_kg_m3": 1.0e-12,
                "drag_v_rel_eci_m_s": np.array([1000.0, 0.0, 0.0]),
                "drag_v_rel_norm_m_s": 1000.0,
            },
            "substeps_s": np.array([1.0]),
            "acceleration_mode": "auto",
            "acceleration_enabled": True,
        }

        actual = model.try_propagate_compiled(**kwargs)
        expected = expected_model.try_propagate_compiled(**kwargs)

        assert actual is not None and expected is not None
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        self.assertEqual(model._compiled_drag_facet_areas.tolist(), [10.0])

    def test_compiled_srp_facets_refresh_after_nested_array_mutation(self):
        area = np.array(1.0)
        facet = {
            "area_m2": area,
            "normal_body": np.array([1.0, 0.0, 0.0]),
            "cp_offset_body_m": np.array([0.0, 1.0, 0.0]),
        }
        config = DisturbanceTorqueConfig(
            use_gravity_gradient=False,
            use_magnetic=False,
            use_drag=False,
            use_srp=True,
            srp_facets=(facet,),
        )
        model = DisturbanceTorqueModel(398600.4418, np.eye(3), config)
        area[...] = 10.0
        expected_model = DisturbanceTorqueModel(398600.4418, np.eye(3), config)
        kwargs = {
            "quat_bn": np.array([1.0, 0.0, 0.0, 0.0]),
            "omega_body_rad_s": np.zeros(3),
            "command_torque_body_nm": np.zeros(3),
            "position_eci_km": np.array([6778.0, 0.0, 0.0]),
            "t_s": 0.0,
            "env": {
                "sun_dir_eci_unit": np.array([1.0, 0.0, 0.0]),
                "srp_shadow_factor": 1.0,
                "srp_pressure_n_m2": 4.56e-6,
            },
            "substeps_s": np.array([1.0]),
            "acceleration_mode": "auto",
            "acceleration_enabled": True,
        }

        actual = model.try_propagate_compiled(**kwargs)
        expected = expected_model.try_propagate_compiled(**kwargs)

        assert actual is not None and expected is not None
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        self.assertEqual(model._compiled_srp_facet_areas.tolist(), [10.0])

    def test_disturbance_torque_recomputed_each_attitude_substep(self):
        inertia = np.diag([120.0, 100.0, 80.0])
        state = StateTruth(
            position_eci_km=np.array([6778.0, 0.0, 0.0]),
            velocity_eci_km_s=np.array([0.0, 7.67, 0.0]),
            attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_rate_body_rad_s=np.array([0.0, 0.4, 0.0]),
            mass_kg=300.0,
            t_s=0.0,
        )
        dyn = OrbitalAttitudeDynamics(
            mu_km3_s2=398600.4418,
            inertia_kg_m2=inertia,
            disturbance_model=_AttitudeCoupledDisturbance(),
            orbit_substep_s=1.0,
            attitude_substep_s=0.1,
            acceleration_mode="auto",
        )

        out = dyn.step(state.copy(), Command.zero(), env={}, dt_s=1.0)

        self.assertGreater(abs(float(out.angular_rate_body_rad_s[2])), 1e-8)

    def test_srp_shadow_factor_cached_across_attitude_substeps(self):
        inertia = np.diag([120.0, 100.0, 80.0])
        state = StateTruth(
            position_eci_km=np.array([6778.0, 0.0, 0.0]),
            velocity_eci_km_s=np.array([0.0, 7.67, 0.0]),
            attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_rate_body_rad_s=np.array([0.0, 0.0, 0.0]),
            mass_kg=300.0,
            t_s=0.0,
        )
        dyn = OrbitalAttitudeDynamics(
            mu_km3_s2=398600.4418,
            inertia_kg_m2=inertia,
            disturbance_model=DisturbanceTorqueModel(
                mu_km3_s2=398600.4418,
                inertia_kg_m2=inertia,
                config=DisturbanceTorqueConfig(
                    use_gravity_gradient=False,
                    use_magnetic=False,
                    use_drag=False,
                    use_srp=True,
                ),
            ),
            orbit_substep_s=1.0,
            attitude_substep_s=0.1,
        )

        with patch("sim.dynamics.model.srp_shadow_factor", return_value=1.0) as shadow_mock:
            dyn.step(
                state.copy(),
                Command.zero(),
                env={"sun_dir_eci": np.array([1.0, 0.0, 0.0]), "srp_shadow_model": "none"},
                dt_s=1.0,
            )

        self.assertEqual(shadow_mock.call_count, 1)

    def test_explicit_srp_shadow_factor_is_preserved(self):
        inertia = np.diag([120.0, 100.0, 80.0])
        state = StateTruth(
            position_eci_km=np.array([6778.0, 0.0, 0.0]),
            velocity_eci_km_s=np.array([0.0, 7.67, 0.0]),
            attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_rate_body_rad_s=np.array([0.0, 0.0, 0.0]),
            mass_kg=300.0,
            t_s=0.0,
        )
        dyn = OrbitalAttitudeDynamics(
            mu_km3_s2=398600.4418,
            inertia_kg_m2=inertia,
            disturbance_model=DisturbanceTorqueModel(
                mu_km3_s2=398600.4418,
                inertia_kg_m2=inertia,
                config=DisturbanceTorqueConfig(
                    use_gravity_gradient=False,
                    use_magnetic=False,
                    use_drag=False,
                    use_srp=True,
                ),
            ),
            orbit_substep_s=1.0,
            attitude_substep_s=0.1,
        )

        with patch("sim.dynamics.model.srp_shadow_factor", return_value=1.0) as shadow_mock:
            out = dyn.step(
                state.copy(),
                Command.zero(),
                env={
                    "sun_dir_eci": np.array([1.0, 0.0, 0.0]),
                    "srp_shadow_model": "none",
                    "srp_shadow_factor": 0.0,
                },
                dt_s=1.0,
            )

        self.assertEqual(shadow_mock.call_count, 0)
        self.assertTrue(np.allclose(out.angular_rate_body_rad_s, np.zeros(3)))

    def test_drag_density_cached_across_attitude_substeps(self):
        inertia = np.diag([120.0, 100.0, 80.0])
        state = StateTruth(
            position_eci_km=np.array([6778.0, 0.0, 0.0]),
            velocity_eci_km_s=np.array([0.0, 7.67, 0.0]),
            attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_rate_body_rad_s=np.array([0.0, 0.0, 0.0]),
            mass_kg=300.0,
            t_s=0.0,
        )
        dyn = OrbitalAttitudeDynamics(
            mu_km3_s2=398600.4418,
            inertia_kg_m2=inertia,
            disturbance_model=DisturbanceTorqueModel(
                mu_km3_s2=398600.4418,
                inertia_kg_m2=inertia,
                config=DisturbanceTorqueConfig(
                    use_gravity_gradient=False,
                    use_magnetic=False,
                    use_drag=True,
                    use_srp=False,
                ),
            ),
            orbit_substep_s=1.0,
            attitude_substep_s=0.1,
        )

        with patch.object(dynamics_model_module, "density_from_model", return_value=1.0e-12) as density_mock:
            dyn.step(
                state.copy(),
                Command.zero(),
                env={"atmosphere_model": "exponential"},
                dt_s=1.0,
            )

        self.assertEqual(density_mock.call_count, 1)

    def test_drag_relative_velocity_cached_with_configured_frame(self):
        inertia = np.diag([120.0, 100.0, 80.0])
        state = StateTruth(
            position_eci_km=np.array([6778.0, 0.0, 0.0]),
            velocity_eci_km_s=np.array([0.0, 7.67, 0.0]),
            attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_rate_body_rad_s=np.array([0.0, 0.0, 0.0]),
            mass_kg=300.0,
            t_s=0.0,
        )
        dyn = OrbitalAttitudeDynamics(
            mu_km3_s2=398600.4418,
            inertia_kg_m2=inertia,
            disturbance_model=DisturbanceTorqueModel(
                mu_km3_s2=398600.4418,
                inertia_kg_m2=inertia,
                config=DisturbanceTorqueConfig(
                    use_gravity_gradient=False,
                    use_magnetic=False,
                    use_drag=True,
                    use_srp=False,
                ),
            ),
            orbit_substep_s=1.0,
            attitude_substep_s=0.1,
        )

        with patch(
            "sim.aero.core.atmosphere_relative_velocity_eci_km_s",
            return_value=np.array([0.0, 7.0, 0.0], dtype=float),
        ) as rel_vel_mock:
            dyn.step(
                state.copy(),
                Command.zero(),
                env={
                    "density_kg_m3": 1.0e-12,
                    "drag_frame_model": "hpop_like",
                    "jd_utc_start": 2460310.5,
                    "drag_eop_path": "validation/EOP-All.txt",
                },
                dt_s=1.0,
            )

        self.assertEqual(rel_vel_mock.call_count, 1)
        kwargs = rel_vel_mock.call_args.kwargs
        self.assertEqual(kwargs["frame_model"], "hpop_like")
        self.assertEqual(kwargs["jd_utc_start"], 2460310.5)
        self.assertEqual(kwargs["eop_path"], "validation/EOP-All.txt")

    def test_attitude_substeps_use_midpoint_translational_state(self):
        inertia = np.diag([120.0, 100.0, 80.0])
        state = StateTruth(
            position_eci_km=np.array([6778.0, 0.0, 0.0]),
            velocity_eci_km_s=np.array([0.0, 7.67, 0.0]),
            attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_rate_body_rad_s=np.array([0.0, 0.1, 0.0]),
            mass_kg=300.0,
            t_s=5.0,
        )
        disturbance = _MidpointRecordingDisturbance()
        dyn = OrbitalAttitudeDynamics(
            mu_km3_s2=398600.4418,
            inertia_kg_m2=inertia,
            disturbance_model=disturbance,
            orbit_substep_s=10.0,
            attitude_substep_s=2.0,
        )

        out = dyn.step(state.copy(), Command.zero(), env={}, dt_s=10.0)

        expected_pos = 0.5 * (np.array(state.position_eci_km, dtype=float) + np.array(out.position_eci_km, dtype=float))
        expected_vel = 0.5 * (
            np.array(state.velocity_eci_km_s, dtype=float) + np.array(out.velocity_eci_km_s, dtype=float)
        )
        expected_t = float(state.t_s + 5.0)

        self.assertGreater(len(disturbance.samples), 1)
        for pos, vel, t_s in disturbance.samples:
            self.assertTrue(np.allclose(pos, expected_pos))
            self.assertTrue(np.allclose(vel, expected_vel))
            self.assertAlmostEqual(t_s, expected_t)


if __name__ == "__main__":
    unittest.main()
