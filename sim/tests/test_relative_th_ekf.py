from __future__ import annotations

import numpy as np

from sim.core.models import Measurement, StateBelief
from sim.dynamics.orbit.elements import coe_to_rv_eci
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2
from sim.dynamics.orbit.two_body import propagate_two_body_rk4
from sim.estimation.relative_hcw_ekf import hcw_state_transition_matrix
from sim.estimation.relative_th_ekf import (
    THRelativeEKFEstimator,
    YARelativeEKFEstimator,
    _th_combined_derivative,
    th_propagate_relative_state,
    th_relative_transition_matrix,
    th_variational_propagate_relative_state_and_stm,
    th_variational_transition_matrix,
    ya_closed_form_propagate_relative_state_and_stm,
    ya_closed_form_transition_matrix,
)
from sim.utils.frames import eci_relative_to_ric_rect, ric_rect_state_to_eci


def _circular_chief(radius_km: float = 7000.0) -> np.ndarray:
    speed = np.sqrt(EARTH_MU_KM3_S2 / radius_km)
    return np.array([radius_km, 0.0, 0.0, 0.0, speed, 0.0], dtype=float)


def _eccentric_chief() -> np.ndarray:
    r_vec, v_vec = coe_to_rv_eci(
        a_km=7600.0,
        ecc=0.12,
        inc_deg=34.0,
        raan_deg=18.0,
        argp_deg=42.0,
        true_anomaly_deg=25.0,
    )
    return np.hstack((r_vec, v_vec))


def _propagate_two_body(state: np.ndarray, duration_s: float, *, step_s: float = 1.0) -> np.ndarray:
    out = np.array(state, dtype=float).reshape(6)
    elapsed = 0.0
    while elapsed < float(duration_s) - 1.0e-12:
        h = min(float(step_s), float(duration_s) - elapsed)
        out = propagate_two_body_rk4(
            x_eci=out,
            dt_s=h,
            mu_km3_s2=EARTH_MU_KM3_S2,
            accel_cmd_eci_km_s2=np.zeros(3),
        )
        elapsed += h
    return out


def _reference_th_combined_derivative(state: np.ndarray) -> np.ndarray:
    chief_r = np.array(state[:3], dtype=float)
    chief_v = np.array(state[3:6], dtype=float)
    rho = np.array(state[6:9], dtype=float)
    rho_dot = np.array(state[9:12], dtype=float)
    r_norm = float(np.linalg.norm(chief_r))
    h_vec = np.cross(chief_r, chief_v)
    h_norm = float(np.linalg.norm(h_vec))
    theta_dot = h_norm / max(r_norm * r_norm, 1.0e-12)
    radial_rate = float(np.dot(chief_r, chief_v)) / r_norm
    theta_ddot = -2.0 * theta_dot * radial_rate / r_norm
    omega = np.array([0.0, 0.0, theta_dot], dtype=float)
    omega_dot = np.array([0.0, 0.0, theta_ddot], dtype=float)
    gravity_gradient = (EARTH_MU_KM3_S2 / (r_norm**3)) * np.array([2.0 * rho[0], -rho[1], -rho[2]])
    rho_ddot = (
        gravity_gradient
        - 2.0 * np.cross(omega, rho_dot)
        - np.cross(omega_dot, rho)
        - np.cross(omega, np.cross(omega, rho))
    )
    chief_acc = -EARTH_MU_KM3_S2 * chief_r / (r_norm**3)
    return np.hstack((chief_v, chief_acc, rho_dot, rho_ddot))


def test_th_scalar_cross_product_fast_path_is_bit_exact() -> None:
    rng = np.random.default_rng(20260824)
    for _ in range(2000):
        state = np.hstack(
            (
                rng.normal(size=3) * 500.0 + np.array([7000.0, 0.0, 0.0]),
                rng.normal(size=3) + np.array([0.0, 7.5, 0.0]),
                rng.normal(size=6) * 0.1,
            )
        )
        actual = _th_combined_derivative(state, EARTH_MU_KM3_S2)
        expected = _reference_th_combined_derivative(state)
        assert np.array_equal(actual, expected)


def test_th_relative_propagation_reduces_to_hcw_for_circular_chief() -> None:
    chief = _circular_chief()
    n = np.sqrt(EARTH_MU_KM3_S2 / np.linalg.norm(chief[:3]) ** 3)
    rel0 = np.array([0.25, -0.4, 0.12, 0.00008, -0.00004, 0.00002], dtype=float)
    dt_s = 180.0

    propagated = th_propagate_relative_state(rel0, dt_s, chief, max_step_s=0.5)
    expected = hcw_state_transition_matrix(n, dt_s) @ rel0

    assert np.allclose(propagated, expected, atol=1e-9)


def test_th_transition_matrix_matches_propagator_finite_difference() -> None:
    chief = _eccentric_chief()
    rel0 = np.array([0.08, -0.15, 0.04, 0.00002, -0.00001, 0.00001], dtype=float)
    dt_s = 75.0
    phi = th_relative_transition_matrix(rel0, dt_s, chief, max_step_s=1.0)
    delta = np.array([1e-4, -2e-4, 5e-5, 2e-7, -1e-7, 1e-7], dtype=float)

    base = th_propagate_relative_state(rel0, dt_s, chief, max_step_s=1.0)
    shifted = th_propagate_relative_state(rel0 + delta, dt_s, chief, max_step_s=1.0)

    assert np.allclose(shifted - base, phi @ delta, atol=5e-10)


def test_variational_stm_matches_finite_difference_stm_for_eccentric_chief() -> None:
    chief = _eccentric_chief()
    rel0 = np.array([0.08, -0.15, 0.04, 0.00002, -0.00001, 0.00001], dtype=float)
    dt_s = 75.0

    finite_difference = th_relative_transition_matrix(rel0, dt_s, chief, max_step_s=0.5)
    variational = th_variational_transition_matrix(rel0, dt_s, chief, max_step_s=0.5)

    assert np.allclose(variational, finite_difference, atol=1e-6)


def test_closed_form_ya_stm_matches_variational_stm_for_eccentric_chief() -> None:
    chief = _eccentric_chief()
    rel0 = np.array([0.08, -0.15, 0.04, 0.00002, -0.00001, 0.00001], dtype=float)
    dt_s = 75.0

    closed_form_state, closed_form_phi = ya_closed_form_propagate_relative_state_and_stm(
        rel0,
        dt_s,
        chief,
        max_step_s=0.5,
    )
    variational_state = th_propagate_relative_state(rel0, dt_s, chief, max_step_s=0.5)
    variational_phi = th_variational_transition_matrix(rel0, dt_s, chief, max_step_s=0.5)

    assert np.allclose(closed_form_state, variational_state, atol=2e-10)
    assert np.allclose(closed_form_phi, variational_phi, atol=2e-8)


def test_closed_form_ya_transition_reduces_to_hcw_for_circular_chief() -> None:
    chief = _circular_chief()
    n = np.sqrt(EARTH_MU_KM3_S2 / np.linalg.norm(chief[:3]) ** 3)
    dt_s = 180.0

    closed_form_phi = ya_closed_form_transition_matrix(dt_s, chief)
    expected = hcw_state_transition_matrix(n, dt_s)

    assert np.allclose(closed_form_phi, expected, atol=1e-8)


def test_th_relative_propagation_tracks_small_eccentric_two_body_truth() -> None:
    chief0 = _eccentric_chief()
    rel0 = np.array([0.04, -0.06, 0.025, 0.000005, -0.000003, 0.000002], dtype=float)
    deputy0 = ric_rect_state_to_eci(rel0, chief0[:3], chief0[3:])
    dt_s = 240.0

    chief_truth = _propagate_two_body(chief0, dt_s)
    deputy_truth = _propagate_two_body(deputy0, dt_s)
    truth_rel = eci_relative_to_ric_rect(deputy_truth, chief_truth)
    predicted = th_propagate_relative_state(rel0, dt_s, chief0, max_step_s=1.0)

    assert np.linalg.norm(predicted[:3] - truth_rel[:3]) < 2.0e-4
    assert np.linalg.norm(predicted[3:] - truth_rel[3:]) < 2.0e-6


def test_th_stm_only_path_is_bitwise_identical_to_full_variational_path() -> None:
    chief = _eccentric_chief()
    for dt_s in (0.0, 5.0, 120.0, 600.0, -125.0):
        optimized = th_variational_transition_matrix(np.zeros(6), dt_s, chief, max_step_s=5.0)
        _state, reference = th_variational_propagate_relative_state_and_stm(
            np.zeros(6), dt_s, chief, max_step_s=5.0
        )
        assert np.array_equal(optimized, reference)


def test_th_relative_ekf_direct_state_update_moves_toward_truth() -> None:
    chief = _eccentric_chief()
    estimator = THRelativeEKFEstimator(
        chief_state_eci_km_s=chief,
        chief_epoch_t_s=0.0,
        dt_s=10.0,
        process_noise_diag=np.ones(6) * 1e-12,
        meas_noise_diag=np.ones(6) * 1e-8,
        measurement_model="relative_state",
        integration_substep_s=1.0,
    )
    truth0 = np.array([0.12, -0.2, 0.04, 0.00003, -0.00002, 0.00001], dtype=float)
    truth = th_propagate_relative_state(truth0, 10.0, chief, max_step_s=1.0)
    belief = StateBelief(
        state=truth0 + np.array([0.05, -0.06, 0.02, 1e-5, -1e-5, 5e-6], dtype=float),
        covariance=np.eye(6) * 1e-2,
        last_update_t_s=0.0,
    )
    measurement = Measurement(vector=truth, t_s=10.0)

    updated = estimator.update(belief, measurement, 10.0)
    assert estimator.last_update_diagnostics is not None
    assert estimator.last_update_diagnostics.update_applied is True
    predicted_only = estimator.update(belief, None, 10.0)

    assert np.linalg.norm(updated.state - truth) < np.linalg.norm(predicted_only.state - truth)
    assert np.allclose(updated.covariance, updated.covariance.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(updated.covariance) >= -1e-12)


def test_ya_relative_ekf_uses_closed_form_ya_stm_prediction() -> None:
    chief = _eccentric_chief()
    estimator = YARelativeEKFEstimator(
        chief_state_eci_km_s=chief,
        chief_epoch_t_s=0.0,
        dt_s=10.0,
        process_noise_diag=np.ones(6) * 1e-12,
        meas_noise_diag=np.ones(6) * 1e-8,
        measurement_model="relative_state",
        integration_substep_s=1.0,
    )
    state = np.array([0.12, -0.2, 0.04, 0.00003, -0.00002, 0.00001], dtype=float)
    covariance = np.diag([1e-4, 2e-4, 3e-4, 1e-8, 2e-8, 3e-8])
    belief = StateBelief(state=state, covariance=covariance, last_update_t_s=0.0)

    predicted = estimator.update(belief, None, 10.0)
    expected_state, expected_phi = ya_closed_form_propagate_relative_state_and_stm(
        state,
        10.0,
        chief,
        max_step_s=1.0,
    )
    expected_cov = expected_phi @ covariance @ expected_phi.T + np.diag(np.ones(6) * 1e-12)

    assert estimator.transition_model == "closed_form_ya"
    assert np.allclose(predicted.state, expected_state)
    assert np.allclose(predicted.covariance, 0.5 * (expected_cov + expected_cov.T))
