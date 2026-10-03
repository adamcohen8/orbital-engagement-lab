"""Compatibility and validation at the opt-in OD numeric byte boundaries."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from sim import rust_estimation_backend as estimation
from sim import rust_ogp_backend as ogp
from sim import rust_relative_backend as relative
from sim import rust_tracking_backend as tracking
from sim.dynamics.orbit.tle import parse_tle_lines
from sim.estimation.orbit_ekf import OrbitEKFEstimator


@pytest.fixture
def native():
    module = pytest.importorskip("oel_rust_orbit")
    if not hasattr(module, "estimation_ekf_update_bytes"):
        pytest.skip("installed wheel predates OD numeric byte bindings")
    return module


def _legacy(module):
    """Expose the actual legacy native methods while hiding added byte APIs."""

    class Legacy:
        def __getattr__(self, name):
            if name.endswith("_bytes"):
                raise AttributeError(name)
            return getattr(module, name)

    return Legacy()


@pytest.mark.parametrize("dt_s", [0.0, 0.1, 1.0, 10.0, 60.0])
def test_fused_two_body_prediction_preserves_forward_differences_and_covariance(native, dt_s):
    if not hasattr(native, "estimation_two_body_predict_and_jacobian_bytes"):
        pytest.skip("installed wheel predates fused prediction")
    parameters = dict(
        mu_km3_s2=398600.4415,
        dt_s=1.0,
        process_noise_diag=np.full(6, 1.0e-12),
        meas_noise_diag=np.full(6, 1.0e-8),
    )
    python = OrbitEKFEstimator(**parameters, numeric_backend="python")
    rust = OrbitEKFEstimator(**parameters, numeric_backend="rust")
    generator = np.random.default_rng(260927)
    states = [np.array([7001.0, 2.0, -0.5, 0.01, 7.49, -0.015])]
    for _ in range(8):
        states.append(np.concatenate((generator.normal(size=3) * 7000.0, generator.normal(size=3) * 4.0)))
    for state in states:
        matrix = generator.normal(size=(6, 6))
        covariance = matrix @ matrix.T * 1.0e-4
        expected_state = python._propagate_state(state, dt_s=dt_s)
        expected_phi = python._numerical_jacobian(state, base=expected_state, dt_s=dt_s)
        actual_state, actual_phi = estimation.two_body_predict_and_jacobian(state, dt_s, parameters["mu_km3_s2"])
        np.testing.assert_array_equal(actual_state, expected_state)
        np.testing.assert_array_equal(actual_phi, expected_phi)
        expected = python._predict(state, covariance, from_t_s=0.0, to_t_s=dt_s)
        actual = rust._predict(state, covariance, from_t_s=0.0, to_t_s=dt_s)
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_allclose(actual[1], expected[1], rtol=0.0, atol=1.0e-18)
        assert actual_state.flags.writeable and actual_phi.flags.writeable


def test_fused_prediction_older_wheel_and_zero_position_preserve_scalar_owner(native, monkeypatch):
    parameters = dict(
        mu_km3_s2=398600.4415,
        dt_s=1.0,
        process_noise_diag=np.full(6, 1.0e-12),
        meas_noise_diag=np.full(6, 1.0e-8),
    )
    python = OrbitEKFEstimator(**parameters, numeric_backend="python")
    rust = OrbitEKFEstimator(**parameters, numeric_backend="rust")
    monkeypatch.setattr(estimation, "_extension", lambda: _legacy(native))
    for state in (np.array([7001.0, 2.0, -0.5, 0.01, 7.49, -0.015]), np.zeros(6)):
        expected = python._predict(state, np.eye(6), from_t_s=0.0, to_t_s=0.0)
        actual = rust._predict(state, np.eye(6), from_t_s=0.0, to_t_s=0.0)
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])


def test_fused_prediction_rejects_malformed_and_nonfinite_inputs(native):
    if not hasattr(native, "estimation_two_body_predict_and_jacobian_bytes"):
        pytest.skip("installed wheel predates fused prediction")
    call = native.estimation_two_body_predict_and_jacobian_bytes
    state = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0], dtype="<f8").tobytes()
    for raw, dt_s, mu, epsilon in (
        (state[:-1], 1.0, 398600.4415, 1.0e-6),
        (state[:-8], 1.0, 398600.4415, 1.0e-6),
        (state, float("nan"), 398600.4415, 1.0e-6),
        (state, -1.0, 398600.4415, 1.0e-6),
        (state, 1.0, -1.0, 1.0e-6),
        (state, 1.0, 398600.4415, 0.0),
    ):
        with pytest.raises(ValueError):
            call(raw, dt_s, mu, epsilon)


@pytest.mark.parametrize("function", ["ekf", "sigma", "recombine", "covariance", "ric", "tracking"])
def test_byte_adapters_match_legacy_and_return_writable_arrays(native, monkeypatch, function):
    x = np.array([7000.0, -20.0, 80.0, 0.01, 7.5, -0.02])
    p = np.diag([4.0, 3.0, 2.0, 0.2, 0.3, 0.25])
    owner = estimation if function in {"ekf", "sigma", "recombine"} else relative if function in {"covariance", "ric"} else tracking
    sigma, wm, wc = estimation.ukf_sigma_points(x, p, alpha=0.35, beta=2.0, kappa=0.0)
    chief = np.repeat(x[None], 3, axis=0)

    def call():
        if function == "ekf":
            return estimation.ekf_update(x, p, x + 0.01, np.eye(6), p / 2)
        if function == "sigma":
            return estimation.ukf_sigma_points(x, p, alpha=0.35, beta=2.0, kappa=0.0)
        if function == "recombine":
            return estimation.ukf_recombine(sigma, wm, wc, p / 100)
        if function == "covariance":
            return (relative.propagate_covariance_history(p, np.repeat(np.eye(6)[None], 4, axis=0), p / 100),)
        if function == "ric":
            return (relative.eci_relative_to_ric_batch(chief + 0.1, chief),)
        return tracking.relative_measurement_and_jacobian(x + 0.1, x, "relative_angles_range_rate")

    actual = call()
    monkeypatch.setattr(owner, "_extension", lambda: _legacy(native))
    expected = call()
    for left, right in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(left, right)
        if isinstance(left, np.ndarray):
            assert left.flags.writeable


def test_noncontiguous_big_endian_inputs_and_empty_covariance_history(native):
    p = np.eye(6, dtype=">f8")
    transitions = np.repeat(np.eye(6, dtype=">f8")[None], 8, axis=0)[::2, ::-1, ::-1]
    actual = relative.propagate_covariance_history(p, transitions, p / 100)
    np.testing.assert_allclose(actual[-1], np.eye(6) * 1.04, rtol=0.0, atol=1e-14)
    empty = relative.propagate_covariance_history(p, np.empty((0, 6, 6)), p)
    np.testing.assert_array_equal(empty, np.eye(6)[None])


@pytest.mark.parametrize("function,args", [
    ("relative_eci_to_ric_batch_bytes", (b"bad", b"")),
    ("covariance_propagate_history_bytes", (b"bad", b"", b"")),
    ("tracking_measurement_batch_bytes", (b"bad", b"", 0)),
    ("estimation_ukf_sigma_points_bytes", (b"bad", b"", 0.35, 2.0, 0.0)),
])
def test_incomplete_numeric_bytes_fail_closed(native, function, args):
    with pytest.raises(ValueError, match="complete float64"):
        getattr(native, function)(*args)


def test_byte_shape_and_nonfinite_validation(native):
    values = np.zeros(6, dtype="<f8").tobytes()
    with pytest.raises(ValueError, match="equal lengths"):
        native.relative_eci_to_ric_batch_bytes(values, b"")
    with pytest.raises(ValueError, match="finite"):
        native.tracking_measurement_batch_bytes(np.full(6, np.nan, dtype="<f8").tobytes(), values, 0)
    with pytest.raises(ValueError, match="36 values"):
        native.covariance_propagate_history_bytes(values, b"", b"")


def test_uniform_and_ragged_whitening_preserve_validation_and_values(native):
    for residuals, factors in [
        ([np.arange(3.0)] * 4, [np.eye(3)] * 4),
        ([np.arange(2.0), np.arange(3.0)], [np.eye(2), np.eye(3)]),
        ([], []),
    ]:
        actual = estimation.measurement_whiten_variable_rows(residuals, factors)
        expected = np.concatenate(residuals) if residuals else np.empty(0)
        np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match="residual block must contain only finite"):
        estimation.measurement_whiten_variable_rows([np.array([np.nan])], [np.eye(1)])
    with pytest.raises(ValueError, match="each factor must be square"):
        estimation.measurement_whiten_variable_rows([np.ones(2)], [np.eye(3)])


def test_ogp_context_sparse_errors_and_legacy_fallback(native, monkeypatch):
    elements = parse_tle_lines(
        "1 25544U 98067A   24001.00000000  .00016717  00000+0  10270-3 0  9003",
        "2 25544  51.6416  43.6012 0005423  52.3066  50.1234 15.50000000  1004",
    )
    times = [0.0, np.nan, 60.0, -60.0, 0.0]
    current = ogp.RustOGPContext(elements).propagate_many(times)

    class LegacyContext:
        def __init__(self, numeric):
            self.context = native.OGPContext(numeric)
        def propagate(self, time):
            return self.context.propagate(time)
        def propagate_many(self, times):
            return self.context.propagate_many(times)

    monkeypatch.setattr(ogp, "_extension", lambda: SimpleNamespace(OGPContext=LegacyContext))
    legacy = ogp.RustOGPContext(elements).propagate_many(times)
    for actual, expected in zip(current, legacy, strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert current[2][1] == "OGP time offset must be finite"
    assert current[0].flags.writeable and current[1].flags.writeable
    with pytest.raises(ValueError, match="complete float64"):
        native.OGPContext(ogp._numeric_elements(elements)).propagate_many_bytes(b"bad")
    with pytest.raises(ValueError, match="inconsistent shapes"):
        native.OGPObservationBatch.from_bytes(b"", b"", b"")


@pytest.mark.parametrize("regime", ["near", "deep"])
@pytest.mark.parametrize("start_offset_days", [-0.01, 0.0])
def test_ogp_canonical_batch_preserves_scalar_frames_and_query_order(native, regime, start_offset_days):
    from sim.dynamics.orbit.sgp4 import SGP4EphemerisProvider
    from sim.tests.test_tle_initialization import DEEP_SPACE_LINE1, DEEP_SPACE_LINE2, ISS_LINE1, ISS_LINE2

    line1, line2 = (ISS_LINE1, ISS_LINE2) if regime == "near" else (DEEP_SPACE_LINE1, DEEP_SPACE_LINE2)
    elements = parse_tle_lines(line1, line2)
    options = dict(elements=elements, mass_kg=100.0, start_jd_utc=elements.epoch_jd_utc + start_offset_days,
                   duration_s=86400.0, numeric_backend="rust")
    scalar, batched = SGP4EphemerisProvider(**options), SGP4EphemerisProvider(**options)
    times = np.array([0.0, 120.0, 43200.0, 60.0, 0.0, 86400.0, 864.0])
    expected = np.array([np.r_[state.position_eci_km, state.velocity_eci_km_s]
                         for state in map(scalar.canonical_state_at, times)])
    actual = batched.canonical_states_at(times)
    np.testing.assert_array_equal(actual, expected)
    assert actual.flags.writeable
    assert batched.canonical_states_at([]).shape == (0, 6)
    with pytest.raises(ValueError, match="finite"):
        batched.canonical_states_at([np.nan])
    with pytest.raises(ValueError, match="within"):
        batched.canonical_states_at([86401.0])


def test_large_geometry_batch_preserves_scalar_angles_and_exact_ranges(native):
    rng = np.random.default_rng(20260927)
    positions = rng.normal(size=(128, 3)) * 1000.0
    optical = estimation.optical_radec_predictions(positions)
    scalar_optical = np.vstack([estimation.optical_radec_predictions(row[None]) for row in positions])
    np.testing.assert_allclose(optical, scalar_optical, rtol=0.0, atol=1.0e-12)
    states = np.column_stack((positions + 7000.0, rng.normal(size=(128, 3)) * 0.01))
    zeros = np.zeros((128, 3))
    targets = positions.copy()
    targets[0] = 0.0
    rotations = np.repeat(np.eye(3)[None], 128, axis=0)
    arguments = (states, targets, zeros, zeros, zeros, rotations)
    ground = estimation.ground_station_predictions(*arguments)
    scalar_ground = np.vstack([estimation.ground_station_predictions(*(value[index:index + 1] for value in arguments))
                               for index in range(128)])
    np.testing.assert_array_equal(ground[:, 2:], scalar_ground[:, 2:])
    np.testing.assert_allclose(ground[:, :2], scalar_ground[:, :2], rtol=0.0, atol=1.0e-12)
    np.testing.assert_array_equal(ground[0], scalar_ground[0])
    positions[17] = 0.0
    with pytest.raises(ValueError, match="nonzero"):
        estimation.optical_radec_predictions(positions)


def test_th_sparse_derivative_preserves_degenerate_chief(native):
    state = np.arange(6.0)
    result, phi = relative.th_variational_propagate_relative_state_and_stm(
        state, 4.0, np.zeros(6), mu_km3_s2=398600.4415, max_step_s=1.0,
    )
    np.testing.assert_array_equal(result, state)
    np.testing.assert_array_equal(phi, np.eye(6))
