"""Prepared ocean tide parity, invalid inputs, and persistent-model lifecycle."""

import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from sim.dynamics.orbit.accelerations import OrbitContext
from sim.dynamics.orbit.frames import FrameContext
from sim.pro_perturbations.ocean_tides import OceanTides, ocean_coefficients
from sim.pro_perturbations.solid_earth_tides import tidal_acceleration_fixed


def _tables(degree):
    factors = np.array([[2, 0, 0, -2, 0, -2], [0, 0, -1, 1, 1, 1], [1, 0, 0, 0, -1, 0]])
    coefficients = np.zeros((3, degree + 1, degree + 1, 4))
    for wave in range(3):
        for n in range(2, degree + 1):
            for m in range(n + 1):
                coefficients[wave, n, m] = np.array([1, -2, 3, -4]) * (wave + 1) * (m + 1) / (n + 1) * 1e-12
    return factors, coefficients


def _native():
    native = pytest.importorskip("oel_rust_orbit")
    if not callable(getattr(native, "OceanTidesContext", None)):
        pytest.skip("installed wheel predates prepared ocean tide contexts")
    return native


@pytest.mark.parametrize("degree", [2, 4, 6, 20])
@pytest.mark.parametrize("pole", [None, (0.043546, 0.424847)])
def test_prepared_ocean_force_matches_reference(degree, pole):
    native = _native()
    factors, coefficients = _tables(degree)
    context = native.OceanTidesContext(factors.ravel().tolist(), coefficients.ravel().tolist(), degree)
    for tt in [2451545.0, 2455197.5, 2455197.5001, 2459669.5, 2461314.25]:
        ut1 = tt - 69.2832395 / 86400
        c, s = ocean_coefficients(factors, coefficients, tt, ut1, pole)
        nc, ns = context.coefficients(tt, ut1, pole)
        for position in [[6500.0, 2000.0, 1000.0], [0.0, 0.0, 7000.0], [42000.0, -2000.0, 5000.0]]:
            expected = tidal_acceleration_fixed(position, c, s, mu_km3_s2=398600.4418, radius_km=6378.137)
            actual = context.acceleration(tuple(position), tt, ut1, pole, 398600.4418, 6378.137)
            # Keep the existing native tidal force tolerance unchanged.
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-19)
            # Fused and exposed coefficients must feed exactly the same kernel.
            separate = native.perturbation_tidal_acceleration(position, nc, ns, degree, 398600.4418, 6378.137)
            np.testing.assert_array_equal(actual, separate)


def test_prepared_ocean_context_rejects_malformed_numeric_inputs():
    native = _native()
    for factors, coefficients, degree in [([], [], 2), ([0.0] * 5, [0.0] * 36, 2), ([0.0] * 6, [0.0] * 35, 2), ([np.nan] * 6, [0.0] * 36, 2), ([0.0] * 6, [np.inf] * 36, 2), ([0.0] * 6, [0.0] * 36, 21)]:
        with pytest.raises(ValueError):
            native.OceanTidesContext(factors, coefficients, degree)
    context = native.OceanTidesContext([0.0] * 6, [0.0] * 36, 2)
    for tt, ut1, pole in [(np.nan, 2459669.5, None), (2459669.5, np.inf, None), (2459669.5, 2459669.5, (0.04, np.nan))]:
        with pytest.raises(ValueError):
            context.coefficients(tt, ut1, pole)
    for position, mu, radius in [((0.0, 0.0, 0.0), 398600.4418, 6378.137), ((np.nan, 0.0, 0.0), 398600.4418, 6378.137), ((7000.0, 0.0, 0.0), 0.0, 6378.137), ((7000.0, 0.0, 0.0), 398600.4418, np.inf)]:
        with pytest.raises(ValueError):
            context.acceleration(position, 2459669.5, 2459669.5, None, mu, radius)


def _model(tmp_path):
    fes = tmp_path / "synthetic-fes.dat"
    fes.write_text("normalized FES Cnm-Snm 10^-11 DelC+\n055.565 synthetic 2 0 1 2 3 4\n165.555 synthetic 2 1 -2 1 4 3\n")
    return OceanTides(FrameContext(model="iau76_80_eop", jd_utc_start=2459669.5, dut1_s=-0.0992395, dat_s=37, xp_arcsec=0.043546, yp_arcsec=0.424847), str(fes), degree=2, order=1)


def test_ocean_native_tables_and_pickle_are_source_bound(tmp_path):
    model = _model(tmp_path)
    for table in [model._factors, model._coefficients]:
        with pytest.raises(ValueError):
            table.setflags(write=True)
    # A non-pickleable cache must be omitted; source values/digest are retained.
    object.__setattr__(model, "_native_context", lambda: None)
    object.__setattr__(model, "_native_context_ready", True)
    restored = pickle.loads(pickle.dumps(model))
    assert restored._native_context is None
    assert not restored._native_context_ready
    assert restored.coefficient_sha256 == model.coefficient_sha256
    for before, after in [(model._factors, restored._factors), (model._coefficients, restored._coefficients)]:
        np.testing.assert_array_equal(before, after)
        with pytest.raises(ValueError):
            after.setflags(write=True)


def test_ocean_old_wheel_falls_back_without_repeated_context_loading(monkeypatch, tmp_path):
    import sim.rust_environment_backend as adapter

    calls = []
    monkeypatch.setattr(adapter, "_extension", lambda: SimpleNamespace())
    original = adapter.try_create_ocean_tides_context
    monkeypatch.setattr(adapter, "try_create_ocean_tides_context", lambda *args: calls.append(True) or original(*args))
    model = _model(tmp_path)
    state = np.array([6500.0, 2000.0, 1000.0, 0.0, 7.5, 1.0])
    ctx = OrbitContext(398600.4418, 300.0)
    for t in [0.0, 60.0, 120.0]:
        expected = model(t, state, {}, ctx)
        actual = model(t, state, {"_rust_numeric_backend": "rust"}, ctx)
        np.testing.assert_array_equal(actual, expected)
    assert len(calls) == 1
