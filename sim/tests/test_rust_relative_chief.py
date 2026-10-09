"""Native TH/YA chief routing preserves the established signed RK4 reference."""

from types import SimpleNamespace

import numpy as np
import pytest

from sim import rust_relative_backend
from sim.dynamics.orbit.elements import coe_to_rv_eci
from sim.estimation.relative_th_ekf import (
    THRelativeEKFEstimator,
    YARelativeEKFEstimator,
    _propagate_chief_state,
)


def _estimator(kind, state, backend="rust"):
    return kind(
        chief_state_eci_km_s=state,
        chief_epoch_t_s=23.13,
        dt_s=30.0,
        process_noise_diag=np.full(6, 1e-12),
        meas_noise_diag=np.full(6, 1e-6),
        integration_substep_s=10.0,
        numeric_backend=backend,
    )


@pytest.mark.parametrize("kind", [THRelativeEKFEstimator, YARelativeEKFEstimator])
@pytest.mark.parametrize("eccentricity", [0.0, 0.1, 0.7])
@pytest.mark.parametrize("duration", [0.0, -0.0, 0.13, -17.13, 300.13, -300.13, 3600.0])
def test_native_chief_matches_signed_partial_step_reference(kind, eccentricity, duration, monkeypatch):
    native = pytest.importorskip("oel_rust_orbit")
    assert hasattr(native, "relative_chief_propagate_bytes")
    r, v = coe_to_rv_eci(
        a_km=26000.0 if eccentricity == 0.7 else 7600.0,
        ecc=eccentricity,
        inc_deg=51.6,
        raan_deg=20.0,
        argp_deg=30.0,
        true_anomaly_deg=40.0,
    )
    state = np.hstack((r, v))
    estimator = _estimator(kind, state)
    expected = _propagate_chief_state(state, duration, mu_km3_s2=estimator.mu_km3_s2, max_step_s=10.0)
    calls = []
    original = rust_relative_backend.try_chief_state_propagation

    def record(*args, **kwargs):
        calls.append(args[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(rust_relative_backend, "try_chief_state_propagation", record)
    actual = estimator._chief_state_at(23.13 + duration)
    assert len(calls) == 1
    np.testing.assert_allclose(actual[:3], expected[:3], rtol=0, atol=2e-10)
    np.testing.assert_allclose(actual[3:], expected[3:], rtol=0, atol=2e-13)
    assert actual.flags.writeable


@pytest.mark.parametrize("kind", [THRelativeEKFEstimator, YARelativeEKFEstimator])
def test_old_wheel_and_explicit_python_keep_reference(kind, monkeypatch):
    state = np.array([7000.0, 100.0, -80.0, -0.1, 7.4, 0.3])
    estimator = _estimator(kind, state)
    expected = _propagate_chief_state(
        state, 40.26 - estimator.chief_epoch_t_s, mu_km3_s2=estimator.mu_km3_s2, max_step_s=10.0
    )
    monkeypatch.setattr(rust_relative_backend, "_extension", lambda: SimpleNamespace())
    actual = estimator._chief_state_at(40.26)
    np.testing.assert_array_equal(actual, expected)

    def forbidden(*args, **kwargs):
        raise AssertionError("Python selector called the native adapter")

    monkeypatch.setattr(rust_relative_backend, "try_chief_state_propagation", forbidden)
    actual = _estimator(kind, state, "python")._chief_state_at(40.26)
    np.testing.assert_array_equal(actual, expected)


def test_native_chief_zero_signed_state_and_invalid_budget():
    native = pytest.importorskip("oel_rust_orbit")
    state = np.array([7000.0, -0.0, 0.0, 0.0, 7.5, -0.0])
    value = rust_relative_backend.try_chief_state_propagation(state, -0.0, mu_km3_s2=398600.4418, max_step_s=10.0)
    np.testing.assert_array_equal(value.view(np.uint64), state.view(np.uint64))
    with pytest.raises(ValueError, match="bounded RK4 step budget"):
        native.relative_chief_propagate_bytes(state.tolist(), 100000001.0, 398600.4418, 10.0)
    with pytest.raises(ValueError, match="finite dt"):
        native.relative_chief_propagate_bytes(state.tolist(), float("nan"), 398600.4418, 10.0)
