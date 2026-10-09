"""Prepared Schwarzschild binding parity and legacy-wheel compatibility."""

import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from sim import rust_environment_backend as adapter
from sim.dynamics.orbit.accelerations import OrbitContext
from sim.pro_perturbations.schwarzschild import SchwarzschildAcceleration, schwarzschild_acceleration

MU = 398600.4418
STATE = np.array([7000.0, 200.0, 100.0, 0.1, 7.4, 0.8])


@pytest.fixture
def native():
    extension = pytest.importorskip("oel_rust_orbit")
    if not callable(getattr(extension, "perturbation_schwarzschild_prepared", None)):
        pytest.skip("installed wheel lacks prepared Schwarzschild binding")
    return extension


def test_prepared_and_legacy_binding_bitwise_parity(native):
    rng = np.random.default_rng(333)
    states = rng.normal(size=(512, 6))
    states[:, :3] *= np.geomspace(1e-100, 1e150, len(states))[:, None]
    states[:, 3:] *= np.geomspace(1e-100, 1e150, len(states))[:, None]
    states = np.vstack((STATE, [7000, -0.0, 0.0, -0.0, 0.0, -0.0], states))
    for state in states:
        for mu in (MU, np.finfo(float).tiny, np.finfo(float).max):
            expected = np.asarray(native.perturbation_schwarzschild(state.tolist(), mu))
            actual = np.asarray(native.perturbation_schwarzschild_prepared(state, mu))
            np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


@pytest.mark.parametrize("state", [STATE, STATE.tolist(), tuple(STATE), STATE[::-1], STATE.astype(np.float32)])
def test_prepared_adapter_matches_legacy_for_containers(native, monkeypatch, state):
    monkeypatch.setattr(adapter, "_extension", lambda: native)
    actual = adapter.try_schwarzschild_acceleration(state, mu_km3_s2=MU)
    monkeypatch.setattr(adapter, "_extension", lambda: SimpleNamespace(
        perturbation_schwarzschild=native.perturbation_schwarzschild,
    ))
    expected = adapter.try_schwarzschild_acceleration(state, mu_km3_s2=MU)
    assert actual.shape == (3,) and actual.dtype == np.float64
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


@pytest.mark.parametrize("state,mu", [
    (STATE[:5], MU), (STATE.reshape(2, 3), MU),
    (np.full(6, np.nan), MU), (np.full(6, np.inf), MU),
    (np.full(6, np.nan), "invalid"), (STATE[:5], "invalid"),
    (STATE, "invalid"), (STATE, None), (STATE, np.nan),
    (STATE, np.inf), (STATE, 0.0), (STATE, -1.0),
    (np.zeros(6), MU), (np.full(6, 1e-300), MU),
])
def test_prepared_adapter_preserves_invalid_input_errors(native, monkeypatch, state, mu):
    monkeypatch.setattr(adapter, "_extension", lambda: SimpleNamespace(
        perturbation_schwarzschild=native.perturbation_schwarzschild,
    ))
    with pytest.raises((ValueError, TypeError, OverflowError)) as expected:
        adapter.try_schwarzschild_acceleration(state, mu_km3_s2=mu)
    monkeypatch.setattr(adapter, "_extension", lambda: native)
    with pytest.raises(type(expected.value)) as actual:
        adapter.try_schwarzschild_acceleration(state, mu_km3_s2=mu)
    assert str(actual.value) == str(expected.value)


def test_legacy_binding_still_returns_list(native):
    assert isinstance(native.perturbation_schwarzschild(STATE.tolist(), MU), list)


def test_force_pickle_rebuilds_native_route(native, monkeypatch):
    monkeypatch.setattr(adapter, "_extension", lambda: native)
    model = SchwarzschildAcceleration()
    context = OrbitContext(MU, 300.0)
    env = {"_rust_numeric_backend": "rust"}
    expected = np.asarray(native.perturbation_schwarzschild(STATE.tolist(), MU))
    np.testing.assert_array_equal(model(0.0, STATE, env, context).view(np.uint64), expected.view(np.uint64))
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored(0.0, STATE, env, context).view(np.uint64), expected.view(np.uint64))
    assert restored == model


def test_python_force_route_does_not_load_native(monkeypatch):
    def unavailable():
        raise AssertionError("Python force route attempted native discovery")

    monkeypatch.setattr(adapter, "_extension", unavailable)
    np.testing.assert_array_equal(
        SchwarzschildAcceleration()(0.0, STATE, {}, OrbitContext(MU, 300.0)),
        schwarzschild_acceleration(STATE, MU),
    )


def test_missing_symbols_keep_python_force_fallback(monkeypatch):
    monkeypatch.setattr(adapter, "_extension", lambda: SimpleNamespace())
    np.testing.assert_array_equal(
        SchwarzschildAcceleration()(0.0, STATE, {"_rust_numeric_backend": "rust"}, OrbitContext(MU, 300.0)),
        schwarzschild_acceleration(STATE, MU),
    )


def test_invalid_state_precedes_custom_mu_conversion_error(native, monkeypatch):
    class InvalidMu:
        def __float__(self):
            raise RuntimeError("custom conversion failure")

    monkeypatch.setattr(adapter, "_extension", lambda: native)
    with pytest.raises(ValueError, match="state must have shape"):
        adapter.try_schwarzschild_acceleration(np.full(6, np.nan), mu_km3_s2=InvalidMu())
    with pytest.raises(RuntimeError, match="custom conversion failure"):
        adapter.try_schwarzschild_acceleration(STATE, mu_km3_s2=InvalidMu())


def test_missing_symbols_remain_an_optional_fallback(monkeypatch):
    monkeypatch.setattr(adapter, "_extension", lambda: SimpleNamespace())
    assert adapter.try_schwarzschild_acceleration(STATE, mu_km3_s2=MU) is None


def test_legacy_wheel_receives_validated_list(monkeypatch):
    calls = []

    def legacy(state, mu):
        calls.append((state, mu))
        return [1.0, 2.0, 3.0]

    monkeypatch.setattr(adapter, "_extension", lambda: SimpleNamespace(perturbation_schwarzschild=legacy))
    np.testing.assert_array_equal(adapter.try_schwarzschild_acceleration(STATE, mu_km3_s2=MU), [1, 2, 3])
    assert calls == [(STATE.tolist(), MU)]
    with pytest.raises(ValueError, match="state must have shape"):
        adapter.try_schwarzschild_acceleration(np.full(6, np.nan), mu_km3_s2=MU)
    assert len(calls) == 1
