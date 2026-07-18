from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from sim.core.models import Measurement, StateBelief
from sim.estimation.relative_hcw_ekf import (
    HCWRelativeEKFEstimator,
    hcw_measurement_dimension,
    hcw_measurement_jacobian,
    hcw_measurement_vector,
    hcw_state_transition_matrix,
)


def _estimator(
    *,
    measurement_model: str = "relative_state",
    measurement_origin: str = "chief",
) -> HCWRelativeEKFEstimator:
    return HCWRelativeEKFEstimator(
        mean_motion_rad_s=0.0011,
        dt_s=10.0,
        process_noise_diag=np.ones(6) * 1e-12,
        meas_noise_diag=np.ones(hcw_measurement_dimension(measurement_model)) * 1e-8,
        measurement_model=measurement_model,
        measurement_origin=measurement_origin,
    )


def test_hcw_state_transition_matches_closed_form_components() -> None:
    n = 0.0011
    dt_s = 120.0
    x0 = np.array([0.2, -0.4, 0.1, 0.0001, -0.0002, 0.00005], dtype=float)

    propagated = hcw_state_transition_matrix(n, dt_s) @ x0

    nt = n * dt_s
    c = np.cos(nt)
    s = np.sin(nt)
    expected = np.array(
        [
            (4.0 - 3.0 * c) * x0[0] + s / n * x0[3] + 2.0 * (1.0 - c) / n * x0[4],
            6.0 * (s - nt) * x0[0] + x0[1] - 2.0 * (1.0 - c) / n * x0[3] + (4.0 * s - 3.0 * nt) / n * x0[4],
            c * x0[2] + s / n * x0[5],
            3.0 * n * s * x0[0] + c * x0[3] + 2.0 * s * x0[4],
            6.0 * n * (c - 1.0) * x0[0] - 2.0 * s * x0[3] + (4.0 * c - 3.0) * x0[4],
            -n * s * x0[2] + c * x0[5],
        ],
        dtype=float,
    )

    assert np.allclose(propagated, expected)


def test_hcw_prediction_reuses_only_the_latest_transition_matrix() -> None:
    estimator = _estimator()
    belief = StateBelief(
        state=np.array([0.2, -0.4, 0.1, 0.0001, -0.0002, 0.00005], dtype=float),
        covariance=np.eye(6) * 1.0e-3,
        last_update_t_s=0.0,
    )

    with patch(
        "sim.estimation.relative_hcw_ekf.hcw_state_transition_matrix",
        wraps=hcw_state_transition_matrix,
    ) as transition:
        first = estimator.update(belief, None, 10.0)
        second = estimator.update(belief, None, 10.0)
        estimator.mean_motion_rad_s += 1.0e-6
        estimator.update(belief, None, 10.0)

    assert transition.call_count == 2
    assert np.array_equal(first.state, second.state)
    assert np.array_equal(first.covariance, second.covariance)


@pytest.mark.parametrize("origin", ["chief", "deputy"])
@pytest.mark.parametrize(
    "model",
    ["relative_range", "relative_range_rate", "relative_angles", "relative_angles_range_rate"],
)
def test_relative_measurement_analytic_jacobian_matches_central_difference(model: str, origin: str) -> None:
    state = np.array([0.8, -0.6, 0.3, 2.0e-4, -3.0e-4, 1.0e-4])
    analytic = hcw_measurement_jacobian(model, state, measurement_origin=origin)
    numerical = np.zeros_like(analytic)
    steps = np.array([1.0e-6] * 3 + [1.0e-9] * 3)
    for axis, step in enumerate(steps):
        plus = state.copy()
        minus = state.copy()
        plus[axis] += step
        minus[axis] -= step
        delta = hcw_measurement_vector(model, plus, measurement_origin=origin) - hcw_measurement_vector(
            model, minus, measurement_origin=origin
        )
        if model.startswith("relative_angles"):
            delta[:2] = (delta[:2] + np.pi) % (2.0 * np.pi) - np.pi
        numerical[:, axis] = delta / (2.0 * step)

    assert analytic == pytest.approx(numerical, rel=2.0e-6, abs=2.0e-8)


def test_angular_filter_rejects_dimensionally_invalid_cartesian_noise_fallback() -> None:
    with pytest.raises(ValueError, match=r"rad\^2"):
        HCWRelativeEKFEstimator(
            mean_motion_rad_s=0.0011,
            dt_s=10.0,
            process_noise_diag=np.ones(6) * 1e-12,
            meas_noise_diag=np.ones(6) * 1e-8,
            measurement_model="relative_angles_range_rate",
        )


def test_hcw_relative_ekf_direct_state_update_moves_toward_truth() -> None:
    estimator = _estimator()
    truth0 = np.array([0.5, -1.2, 0.3, 0.0002, -0.0001, 0.00005], dtype=float)
    truth = hcw_state_transition_matrix(estimator.mean_motion_rad_s, 10.0) @ truth0
    belief = StateBelief(
        state=truth0 + np.array([0.1, -0.2, 0.05, 5e-5, -3e-5, 2e-5], dtype=float),
        covariance=np.eye(6) * 1e-2,
        last_update_t_s=0.0,
    )
    measurement = Measurement(vector=truth, t_s=10.0)

    with patch("numpy.linalg.inv", side_effect=AssertionError("np.linalg.inv should not be used")):
        updated = estimator.update(belief, measurement, 10.0)
    assert estimator.last_update_diagnostics is not None
    assert estimator.last_update_diagnostics.update_applied is True

    predicted_only = estimator.update(belief, None, 10.0)
    assert np.linalg.norm(updated.state - truth) < np.linalg.norm(predicted_only.state - truth)
    assert np.allclose(updated.covariance, updated.covariance.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(updated.covariance) >= -1e-12)


def test_hcw_relative_ekf_range_rate_update_reduces_measurement_residual() -> None:
    estimator = _estimator(measurement_model="relative_range_rate")
    truth = np.array([0.8, 0.6, -0.1, -0.0003, 0.0002, 0.00005], dtype=float)
    belief = StateBelief(
        state=truth + np.array([0.25, -0.15, 0.0, 0.0002, -0.0001, 0.0], dtype=float),
        covariance=np.diag([0.5, 0.5, 0.5, 1e-5, 1e-5, 1e-5]),
        last_update_t_s=0.0,
    )
    measurement = Measurement(vector=hcw_measurement_vector("relative_range_rate", truth), t_s=0.0)

    before = hcw_measurement_vector("relative_range_rate", belief.state) - measurement.vector
    updated = estimator.update(belief, measurement, 0.0)
    after = hcw_measurement_vector("relative_range_rate", updated.state) - measurement.vector

    assert np.linalg.norm(after) < np.linalg.norm(before)


def test_hcw_relative_ekf_deputy_origin_angles_wrap_across_pi() -> None:
    estimator = _estimator(measurement_model="relative_angles", measurement_origin="deputy")
    truth = np.array([1.0, 1e-6, 0.0, 0.0, 0.0, 0.0], dtype=float)
    belief = StateBelief(
        state=np.array([1.0, -1e-6, 0.0, 0.0, 0.0, 0.0], dtype=float),
        covariance=np.eye(6) * 1e-3,
        last_update_t_s=0.0,
    )
    measurement = Measurement(vector=hcw_measurement_vector("relative_angles", truth, measurement_origin="deputy"), t_s=0.0)

    updated = estimator.update(belief, measurement, 0.0)

    assert np.all(np.isfinite(updated.state))
    assert estimator.last_update_diagnostics is not None
    assert abs(float(estimator.last_update_diagnostics.innovation[0])) < 1e-3


def test_hcw_relative_ekf_partial_measurement_is_ignored() -> None:
    estimator = _estimator(measurement_model="relative_angles_range_rate")
    belief = StateBelief(state=np.array([1.0, 0.2, 0.1, 0.0, 0.0, 0.0]), covariance=np.eye(6), last_update_t_s=0.0)

    predicted = estimator.update(belief, None, 1.0)
    partial = estimator.update(belief, Measurement(vector=np.array([0.1, 0.2]), t_s=1.0), 1.0)

    assert np.allclose(partial.state, predicted.state)
    assert np.allclose(partial.covariance, predicted.covariance)
    assert estimator.last_update_diagnostics is not None
    assert estimator.last_update_diagnostics.measurement_available is True
    assert estimator.last_update_diagnostics.update_applied is False


def test_hcw_relative_ekf_updates_at_measurement_epoch_then_propagates_to_output_time() -> None:
    estimator = _estimator()
    belief = StateBelief(
        state=np.array([0.1, -0.2, 0.05, 0.0, 0.0, 0.0], dtype=float),
        covariance=np.eye(6) * 1e-3,
        last_update_t_s=0.0,
    )
    measurement_state = np.array([0.2, -0.1, 0.03, 0.0001, -0.0002, 0.00005], dtype=float)
    measurement = Measurement(vector=measurement_state, t_s=1.0)

    updated = estimator.update(belief, measurement, 3.0)
    at_measurement = estimator.update(belief, measurement, 1.0)
    expected = hcw_state_transition_matrix(estimator.mean_motion_rad_s, 2.0) @ at_measurement.state

    assert updated.last_update_t_s == 3.0
    assert np.allclose(updated.state, expected)


def test_hcw_relative_ekf_uses_full_epoch_measurement_covariance() -> None:
    estimator = _estimator()
    covariance = np.diag([1.0e-8] * 3 + [1.0e-12] * 3)
    covariance[0, 1] = covariance[1, 0] = 0.7e-8
    estimator.set_measurement_covariance(covariance)
    state = np.array([0.2, -0.1, 0.05, 1.0e-5, -2.0e-5, 3.0e-6])
    belief = StateBelief(state=state, covariance=np.zeros((6, 6)), last_update_t_s=0.0)

    estimator.update(belief, Measurement(vector=state, t_s=0.0), 0.0)

    assert estimator.last_update_diagnostics is not None
    np.testing.assert_allclose(
        estimator.last_update_diagnostics.innovation_covariance,
        covariance,
    )
