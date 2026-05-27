from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import numpy as np

from sim.acceleration.benchmarks import benchmark_attitude_kernel, benchmark_estimation_kernel, benchmark_orbit_kernel
from sim.acceleration.kernels.frames import ric_curv_to_rect_kernel, ric_dcm_ir_from_rv_kernel, ric_rect_to_curv_kernel
from sim.acceleration.kernels.orbit import j2_accel_eci, rk4_zonal_step_state, two_body_accel_eci
from sim.acceleration.kernels.reentry import (
    atmosphere_relative_velocity_eci_km_s_kernel,
    radial_altitude_km_kernel,
    reentry_scalar_metrics_kernel,
)
from sim.acceleration.settings import (
    ACCELERATION_ENV,
    acceleration_context_from_config,
    acceleration_settings_from_config,
)
from sim.acceleration.warmup import warmup_acceleration
from sim.config import scenario_config_from_dict
from sim.core.models import Measurement, StateBelief
from sim.dynamics.attitude.rigid_body import (
    get_attitude_guardrail_stats,
    propagate_attitude_exponential_map,
    reset_attitude_guardrail_stats,
)
from sim.dynamics.orbit.accelerations import OrbitContext, accel_j2, accel_two_body
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2
from sim.dynamics.orbit.propagator import OrbitPropagator, j2_plugin
from sim.dynamics.reentry import (
    ReentryConfig,
    ReentryObjectProperties,
    atmosphere_relative_velocity_eci_km_s,
    radial_altitude_km,
    reentry_metrics_for_state,
)
from sim.estimation.attitude_ekf import AttitudeEKFEstimator
from sim.estimation.orbit_ekf import OrbitEKFEstimator
from sim.utils.frames import (
    eci_relative_to_ric_rect,
    ric_curv_to_rect,
    ric_dcm_ir_from_rv,
    ric_rect_state_to_eci,
    ric_rect_to_curv,
)
from sim.utils.quaternion import normalize_quaternion


class TestAcceleration(unittest.TestCase):
    def setUp(self) -> None:
        self._env_patch = patch.dict(os.environ, {ACCELERATION_ENV: ""}, clear=False)
        self._env_patch.start()

    def tearDown(self) -> None:
        self._env_patch.stop()

    def test_acceleration_settings_from_config(self):
        cfg = scenario_config_from_dict({"simulator": {"acceleration": {"mode": "auto", "warmup": True}}})

        settings = acceleration_settings_from_config(cfg)

        self.assertEqual(settings.requested_mode, "auto")
        self.assertIn(settings.effective_backend, {"python", "numba"})

    def test_acceleration_config_can_lock_out_env_override(self):
        cfg = scenario_config_from_dict(
            {"simulator": {"acceleration": {"mode": "off", "env_override": False}}}
        )

        with patch.dict(os.environ, {ACCELERATION_ENV: "auto"}, clear=False):
            settings = acceleration_settings_from_config(cfg)

        self.assertEqual(settings.requested_mode, "off")
        self.assertFalse(settings.enabled)

    def test_orbit_kernels_match_baseline_accelerations(self):
        r = np.array([7000.0, -20.0, 30.0], dtype=float)

        np.testing.assert_allclose(two_body_accel_eci(r, EARTH_MU_KM3_S2), accel_two_body(r, EARTH_MU_KM3_S2))
        np.testing.assert_allclose(j2_accel_eci(r, EARTH_MU_KM3_S2), accel_j2(r, EARTH_MU_KM3_S2))

    def test_zonal_rk4_kernel_matches_python_propagator(self):
        x = np.array([7000.0, -20.0, 30.0, 0.0, 7.5, 0.01], dtype=float)
        command = np.array([0.0, 1.0e-9, 0.0], dtype=float)
        ctx = OrbitContext(mu_km3_s2=EARTH_MU_KM3_S2, mass_kg=100.0)
        baseline = OrbitPropagator(integrator="rk4", plugins=[j2_plugin], acceleration_mode="off")
        accelerated = OrbitPropagator(integrator="rk4", plugins=[j2_plugin], acceleration_mode="auto")

        expected = baseline.propagate(x, 1.0, 0.0, command, {}, ctx)
        actual = accelerated.propagate(x, 1.0, 0.0, command, {}, ctx)
        direct = rk4_zonal_step_state(x, 1.0, command, EARTH_MU_KM3_S2, True, False, False)

        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(direct, expected, rtol=1e-12, atol=1e-12)

    def test_frame_kernels_match_baseline(self):
        r = np.array([7000.0, -20.0, 30.0], dtype=float)
        v = np.array([0.0, 7.5, 0.01], dtype=float)
        rel = np.array([0.1, -1.0, 0.05, 0.0, 0.0001, -0.00002], dtype=float)
        r0 = float(np.linalg.norm(r))

        np.testing.assert_allclose(ric_dcm_ir_from_rv_kernel(r, v), ric_dcm_ir_from_rv(r, v))
        np.testing.assert_allclose(ric_curv_to_rect_kernel(rel, r0), ric_curv_to_rect(rel, r0))
        np.testing.assert_allclose(ric_rect_to_curv_kernel(rel, r0), ric_rect_to_curv(rel, r0))

    def test_frame_wrappers_use_acceleration_without_changing_results(self):
        r = np.array([7000.0, -20.0, 30.0], dtype=float)
        v = np.array([0.0, 7.5, 0.01], dtype=float)
        chief = np.hstack((r, v))
        rel = np.array([0.1, -1.0, 0.05, 0.0, 0.0001, -0.00002], dtype=float)
        dep = ric_rect_state_to_eci(rel, r, v)
        r0 = float(np.linalg.norm(r))

        baseline = {
            "dcm": ric_dcm_ir_from_rv(r, v),
            "curv_to_rect": ric_curv_to_rect(rel, r0),
            "rect_to_curv": ric_rect_to_curv(rel, r0),
            "rect_state_to_eci": dep,
            "eci_relative": eci_relative_to_ric_rect(dep, chief),
        }

        with patch.dict(os.environ, {ACCELERATION_ENV: "auto"}, clear=False):
            np.testing.assert_allclose(ric_dcm_ir_from_rv(r, v), baseline["dcm"], atol=1.0e-12)
            np.testing.assert_allclose(ric_curv_to_rect(rel, r0), baseline["curv_to_rect"], atol=1.0e-12)
            np.testing.assert_allclose(ric_rect_to_curv(rel, r0), baseline["rect_to_curv"], atol=1.0e-12)
            np.testing.assert_allclose(ric_rect_state_to_eci(rel, r, v), baseline["rect_state_to_eci"], atol=1.0e-12)
            np.testing.assert_allclose(eci_relative_to_ric_rect(dep, chief), baseline["eci_relative"], atol=1.0e-12)

    def test_frame_wrappers_follow_config_context(self):
        cfg = scenario_config_from_dict({"simulator": {"acceleration": {"mode": "auto"}}})
        settings = acceleration_settings_from_config(cfg)
        r = np.array([7000.0, -20.0, 30.0], dtype=float)
        v = np.array([0.0, 7.5, 0.01], dtype=float)
        sentinel = np.eye(3) * 2.0

        with acceleration_context_from_config(cfg):
            with patch("sim.utils.frames.ric_dcm_ir_from_rv_kernel", return_value=sentinel) as kernel:
                actual = ric_dcm_ir_from_rv(r, v)

        if settings.enabled:
            kernel.assert_called_once()
            np.testing.assert_allclose(actual, sentinel)
        else:
            kernel.assert_not_called()

    def test_reentry_kernels_match_baseline(self):
        r = np.array([6478.137, 0.0, 0.0], dtype=float)
        v = np.array([0.0, 7.7, 0.0], dtype=float)
        cfg = ReentryConfig(enabled=True, atmosphere_model="exponential")
        props = ReentryObjectProperties(mass_kg=100.0, drag_area_m2=1.0, cd=2.2, nose_radius_m=0.5)
        rho = 1.0e-7

        np.testing.assert_allclose(radial_altitude_km_kernel(r), radial_altitude_km(r))
        np.testing.assert_allclose(
            atmosphere_relative_velocity_eci_km_s_kernel(r, v),
            atmosphere_relative_velocity_eci_km_s(r, v),
        )
        metrics = reentry_metrics_for_state(
            r_eci_km=r,
            v_eci_km_s=v,
            t_s=0.0,
            dt_s=1.0,
            cfg=cfg,
            props=props,
            env={"density_kg_m3": rho, "atmosphere_model": "exponential"},
            active=True,
            previous_heat_load_j_m2=0.0,
        )
        kernel_values = reentry_scalar_metrics_kernel(
            r,
            v,
            float(metrics["density_kg_m3"]),
            props.mass_kg,
            props.drag_area_m2,
            props.cd,
            props.nose_radius_m,
            cfg.heat_rate_coefficient,
            1.0,
            0.0,
        )

        np.testing.assert_allclose(kernel_values[0], metrics["relative_speed_m_s"])
        np.testing.assert_allclose(kernel_values[1], metrics["dynamic_pressure_pa"])
        np.testing.assert_allclose(kernel_values[2], metrics["drag_decel_m_s2"])
        np.testing.assert_allclose(kernel_values[3], metrics["g_load"])
        np.testing.assert_allclose(kernel_values[4], metrics["heat_rate_w_m2"])
        np.testing.assert_allclose(kernel_values[5], metrics["heat_load_j_m2"])

    def test_warmup_runs_without_numba_requirement(self):
        result = warmup_acceleration(profile="validation")

        self.assertEqual(result["profile"], "validation")
        self.assertGreaterEqual(int(result["kernel_count"]), 1)

    def test_orbit_benchmark_returns_valid_result(self):
        result = benchmark_orbit_kernel(iterations=2, warmup=False)

        self.assertEqual(result.iterations, 2)
        self.assertGreater(result.python_propagator_s, 0.0)
        self.assertGreater(result.accelerated_propagator_s, 0.0)
        self.assertGreater(result.speedup, 0.0)
        self.assertLess(result.state_delta_norm, 1e-6)

    def test_attitude_accelerated_path_matches_baseline(self):
        q0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        w0 = np.array([0.01, -0.02, 0.03], dtype=float)
        inertia = np.diag([100.0, 90.0, 80.0])
        torque = np.array([0.001, -0.002, 0.003], dtype=float)

        reset_attitude_guardrail_stats()
        q_expected, w_expected = propagate_attitude_exponential_map(q0, w0, inertia, torque, 0.1, acceleration_mode="off")
        stats_expected = get_attitude_guardrail_stats()
        reset_attitude_guardrail_stats()
        q_actual, w_actual = propagate_attitude_exponential_map(q0, w0, inertia, torque, 0.1, acceleration_mode="auto")
        stats_actual = get_attitude_guardrail_stats()

        np.testing.assert_allclose(q_actual, q_expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(w_actual, w_expected, rtol=1e-12, atol=1e-12)
        self.assertEqual(stats_actual, stats_expected)

    def test_attitude_accelerated_guardrails_match_baseline(self):
        q0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        w0 = np.array([np.nan, 2.0e6, -np.inf], dtype=float)
        inertia = np.zeros((3, 3), dtype=float)
        torque = np.array([np.inf, np.nan, 2.0e12], dtype=float)

        reset_attitude_guardrail_stats()
        q_expected, w_expected = propagate_attitude_exponential_map(q0, w0, inertia, torque, 0.1, acceleration_mode="off")
        stats_expected = get_attitude_guardrail_stats()
        reset_attitude_guardrail_stats()
        q_actual, w_actual = propagate_attitude_exponential_map(q0, w0, inertia, torque, 0.1, acceleration_mode="auto")
        stats_actual = get_attitude_guardrail_stats()

        np.testing.assert_allclose(q_actual, q_expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(w_actual, w_expected, rtol=1e-12, atol=1e-12)
        self.assertEqual(stats_actual, stats_expected)

    def test_attitude_benchmark_returns_valid_result(self):
        result = benchmark_attitude_kernel(iterations=2, warmup=False)

        self.assertEqual(result.iterations, 2)
        self.assertGreater(result.python_propagator_s, 0.0)
        self.assertGreater(result.accelerated_propagator_s, 0.0)
        self.assertGreater(result.speedup, 0.0)
        self.assertLess(result.quaternion_delta_norm, 1e-6)
        self.assertLess(result.rate_delta_norm, 1e-6)

    def test_estimator_accelerated_paths_match_baseline(self):
        orbit_state = np.array([7000.0, 1.0, -0.5, -0.001, 7.5460, 0.002], dtype=float)
        orbit_belief = StateBelief(state=orbit_state, covariance=np.eye(6) * 1e-3, last_update_t_s=0.0)
        orbit_measurement = Measurement(
            vector=orbit_state + np.array([1e-3, -2e-3, 1e-3, 1e-6, -1e-6, 2e-6]),
            t_s=1.0,
        )
        orbit_kwargs = {
            "mu_km3_s2": EARTH_MU_KM3_S2,
            "dt_s": 1.0,
            "process_noise_diag": np.ones(6) * 1e-10,
            "meas_noise_diag": np.ones(6) * 1e-6,
        }
        orbit_base = OrbitEKFEstimator(**orbit_kwargs, acceleration_mode="off")
        orbit_acc = OrbitEKFEstimator(**orbit_kwargs, acceleration_mode="auto")

        np.testing.assert_allclose(
            orbit_acc._numerical_jacobian(orbit_state, dt_s=1.0),
            orbit_base._numerical_jacobian(orbit_state, dt_s=1.0),
            rtol=1e-11,
            atol=1e-11,
        )
        orbit_expected = orbit_base.update(orbit_belief, orbit_measurement, 1.0)
        orbit_actual = orbit_acc.update(orbit_belief, orbit_measurement, 1.0)
        np.testing.assert_allclose(orbit_actual.state, orbit_expected.state, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(orbit_actual.covariance, orbit_expected.covariance, rtol=1e-11, atol=1e-11)

        attitude_state = np.hstack(
            (normalize_quaternion(np.array([1.0, 0.01, -0.02, 0.0])), np.array([0.01, -0.02, 0.03]))
        )
        attitude_belief = StateBelief(state=attitude_state, covariance=np.eye(7) * 1e-3, last_update_t_s=0.0)
        attitude_measurement = Measurement(
            vector=np.hstack(
                (
                    normalize_quaternion(np.array([1.0, 0.012, -0.018, 0.002])),
                    np.array([0.011, -0.019, 0.029]),
                )
            ),
            t_s=1.0,
        )
        attitude_kwargs = {
            "dt_s": 1.0,
            "inertia_kg_m2": np.diag([10.0, 12.0, 8.0]),
            "process_noise_diag": np.ones(7) * 1e-8,
            "meas_noise_diag": np.ones(7) * 1e-6,
        }
        attitude_base = AttitudeEKFEstimator(**attitude_kwargs, acceleration_mode="off")
        attitude_acc = AttitudeEKFEstimator(**attitude_kwargs, acceleration_mode="auto")

        np.testing.assert_allclose(
            attitude_acc._numerical_jacobian(attitude_state, dt_s=1.0),
            attitude_base._numerical_jacobian(attitude_state, dt_s=1.0),
            rtol=1e-9,
            atol=1e-9,
        )
        attitude_expected = attitude_base.update(attitude_belief, attitude_measurement, 1.0)
        attitude_actual = attitude_acc.update(attitude_belief, attitude_measurement, 1.0)
        np.testing.assert_allclose(attitude_actual.state, attitude_expected.state, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(attitude_actual.covariance, attitude_expected.covariance, rtol=1e-11, atol=1e-11)

    def test_estimation_benchmark_returns_valid_result(self):
        result = benchmark_estimation_kernel(iterations=2, warmup=False)

        self.assertEqual(result.iterations, 2)
        self.assertGreater(result.orbit_jacobian_python_s, 0.0)
        self.assertGreater(result.orbit_jacobian_accelerated_s, 0.0)
        self.assertGreater(result.attitude_jacobian_python_s, 0.0)
        self.assertGreater(result.attitude_jacobian_accelerated_s, 0.0)
        self.assertGreater(result.joint_update_python_s, 0.0)
        self.assertGreater(result.joint_update_accelerated_s, 0.0)
        self.assertLess(result.orbit_jacobian_delta_norm, 1e-6)
        self.assertLess(result.attitude_jacobian_delta_norm, 1e-6)
        self.assertLess(result.joint_state_delta_norm, 1e-6)
