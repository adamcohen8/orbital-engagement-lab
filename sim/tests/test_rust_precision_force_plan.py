"""Ordered precision force fusion, callback parity, and compatibility policy."""
import pickle
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from sim.dynamics.orbit.accelerations import OrbitContext
from sim.dynamics.orbit.frames import FrameContext
from sim.dynamics.orbit.propagator import OrbitPropagator, j2_plugin, srp_plugin
from sim.dynamics.orbit.rust_force_plan import force_codes, immutable_builtin_plan_signature, make_plan
from sim.pro_perturbations.earth_radiation import EarthRadiationPressure
from sim.pro_perturbations.ocean_tides import OceanTides
from sim.pro_perturbations.schwarzschild import SchwarzschildAcceleration
from sim.pro_perturbations.solid_earth_tides import SolidEarthTides

ROOT = Path(__file__).resolve().parents[2]
X = np.array([7000., 100., 400., -0.2, 7.4, 0.8])
CTX = OrbitContext(398600.4418, 300., area_m2=4., cr=1.2)
ENV = dict(jd_utc_start=2459669.5, sun_pos_eci_km=np.array([1.49e8, 2.e6, -1.e6]), moon_pos_eci_km=np.array([3.7e5, -8.e4, 3.e4]))


@pytest.fixture(autouse=True)
def native():
    module = pytest.importorskip("oel_rust_orbit")
    if getattr(module, "ONP_PRECISION_FORCE_PLAN_VERSION", 0) != 1:
        pytest.skip("native extension predates precision force plan")
    return module


def models():
    frame = FrameContext(model="iau76_80_eop", jd_utc_start=2459669.5, dut1_s=-0.0992395, dat_s=37., xp_arcsec=.043546, yp_arcsec=.424847)
    return [EarthRadiationPressure(frame, quadrature_order=8), SchwarzschildAcceleration(),
            SolidEarthTides(frame, "tide_free"), OceanTides(frame, str(ROOT / "sim/dynamics/orbit/data/ocean_tide_synthetic.txt"), degree=2, order=2)]


@pytest.mark.parametrize("mask", range(1, 16))
def test_every_precision_subset_and_builtin_mix_preserves_callback_trajectory(mask):
    forces = [f for i, f in enumerate(models()) if mask & (1 << i)]
    forces = [j2_plugin, *forces, srp_plugin]
    fused = OrbitPropagator(numeric_backend="rust", plugins=forces)
    callback = OrbitPropagator(numeric_backend="rust", plugins=forces)
    x, y = X.copy(), X.copy()
    for time in [0., 10., 20.]:
        x = fused.propagate(x, 10., time, np.zeros(3), dict(ENV), CTX)
        y = callback.propagate(y, 10., time, np.zeros(3), dict(ENV, _rust_precision_force_plan_disabled=True), CTX)
        np.testing.assert_allclose(x, y, rtol=0., atol=2e-11)
        assert fused.last_numeric_path == "rust_native_force_plan"
        assert callback.last_numeric_path == "rust_python_force_callback"


@pytest.mark.parametrize("integrator", ["rk4", "rkf78", "dopri5"])
def test_reordered_duplicate_models_and_integrators(integrator):
    f = models()
    forces = [f[3], f[0], j2_plugin, f[2], f[1], f[0]]
    prop = OrbitPropagator(numeric_backend="rust", integrator=integrator, plugins=forces)
    reference = OrbitPropagator(numeric_backend="rust", integrator=integrator, plugins=forces)
    actual = prop.propagate(X, 60., 0., np.zeros(3), dict(ENV), CTX)
    expected = reference.propagate(X, 60., 0., np.zeros(3), dict(ENV, _rust_precision_force_plan_disabled=True), CTX)
    np.testing.assert_allclose(actual, expected, rtol=0., atol=2e-11)
    assert prop.last_numeric_path == "rust_native_force_plan"


def test_fused_history_calls_no_python_force_kernels():
    forces = models()
    prop = OrbitPropagator(numeric_backend="rust", plugins=forces)
    reference = OrbitPropagator(numeric_backend="rust", plugins=forces)
    history = [X.copy()]
    for i in range(6):
        history.append(reference.propagate(history[-1], 10., i * 10., np.zeros(3), dict(ENV, _rust_precision_force_plan_disabled=True), CTX))
    from contextlib import ExitStack
    with ExitStack() as stack:
        for cls in (EarthRadiationPressure, SchwarzschildAcceleration, SolidEarthTides, OceanTides):
            stack.enter_context(patch.object(cls, "__call__", side_effect=AssertionError("Python force was dispatched")))
        actual = prop.try_propagate_fixed_steps(X, 10., 6, 0., np.zeros(3), dict(ENV), CTX, sample_stride=1)
    assert prop.last_numeric_path == "rust_native_force_plan_history"
    np.testing.assert_allclose(actual, history, rtol=0., atol=2e-11)
    restored = pickle.loads(pickle.dumps(prop))
    assert restored._rust_force_context_cache is None
    np.testing.assert_array_equal(restored.try_propagate_fixed_steps(X, 10., 6, 0., np.zeros(3), dict(ENV), CTX, sample_stride=1), actual)


def test_older_extensions_custom_classes_and_python_backend_keep_callback():
    import sim.rust_environment_backend as adapter
    prop = OrbitPropagator(numeric_backend="rust", plugins=models())
    with patch.object(adapter, "supports_precision_force_plan", return_value=False):
        assert force_codes(prop, ENV) is None
        assert prop.try_propagate_fixed_steps(X, 10., 1, 0., np.zeros(3), dict(ENV), CTX) is None
        prop.propagate(X, 10., 0., np.zeros(3), dict(ENV), CTX)
        assert prop.last_numeric_path == "rust_python_force_callback"
    with patch.object(adapter, "supports_precision_force_plan", create=False):
        del adapter.supports_precision_force_plan
        assert force_codes(prop, ENV) is None
    class Custom(SchwarzschildAcceleration):
        def __call__(self, *args):
            return np.array([1e-8, 0., 0.])
    prop.plugins = [Custom()]
    assert force_codes(prop, ENV) is None
    python = OrbitPropagator(numeric_backend="python", plugins=models())
    assert python.try_propagate_fixed_steps(X, 10., 1, 0., np.zeros(3), dict(ENV), CTX) is None


def test_native_metadata_and_specification_errors_fail_closed(native):
    prop = OrbitPropagator(numeric_backend="rust", plugins=models())
    *_, stage, context = make_plan(prop, X, 0., dict(ENV), CTX)
    row = stage(0., X)
    for bad in [row[:19], row[:-1], [np.nan] + row[1:]]:
        with pytest.raises(ValueError):
            context.acceleration(X.tolist(), bad, [0., 0., 0.])
    with pytest.raises(ValueError, match="precision specifications"):
        native.ONPForceContext([10], [398600.4418] + [1.] * 14, 0, (0, 0), [])
    with pytest.raises(ValueError):
        native.ONPForceContext([10], [398600.4418] + [1.] * 14, 0, (0, 0), [], [(10, [1.], [])])


def test_zero_radiation_skips_missing_frame_and_ephemeris_resources():
    frame = FrameContext(model="iau76_80_eop", jd_utc_start=2459669.5, eop_path="missing-eop-file")
    prop = OrbitPropagator(numeric_backend="rust", plugins=[EarthRadiationPressure(frame, area_m2=0., quadrature_order=8)])
    result = prop.propagate(X, 10., 0., np.zeros(3), {}, CTX)
    reference = OrbitPropagator(numeric_backend="rust")
    np.testing.assert_array_equal(result, reference.propagate(X, 10., 0., np.zeros(3), {}, CTX))


def test_context_refresh_and_speculative_signature_bind_model_settings():
    forces = models()
    prop = OrbitPropagator(numeric_backend="rust", plugins=forces)
    a = make_plan(prop, X, 0., dict(ENV), CTX)[-1]
    assert make_plan(prop, X, 0., dict(ENV), CTX)[-1] is a
    changed = OrbitContext(398600.4418, 300., area_m2=8., cr=1.2)
    assert make_plan(prop, X, 0., dict(ENV), changed)[-1] is not a
    first = immutable_builtin_plan_signature(prop, ENV, CTX)
    assert first is not None
    prop.plugins = list(reversed(forces))
    assert immutable_builtin_plan_signature(prop, ENV, CTX) != first


@pytest.mark.parametrize("integrator", ["rk4", "rkf78", "dopri5"])
def test_nonuniform_sampled_histories_match_callback_states(integrator):
    prop = OrbitPropagator(numeric_backend="rust", integrator=integrator, plugins=models())
    reference = OrbitPropagator(numeric_backend="rust", integrator=integrator, plugins=models())
    widths = [.3, .7, 1.1]
    times = [0., .3, 1.]
    rows = [X.copy()]
    current = X.copy()
    for i, (time, width) in enumerate(zip(times, widths, strict=True)):
        current = reference.propagate(current, width, time, np.zeros(3), dict(ENV, _rust_precision_force_plan_disabled=True), CTX)
        if i in (0, 2):
            rows.append(current)
    if integrator == "rk4":
        actual = prop.try_propagate_sampled_steps(X, widths, [1, 3], 0., np.zeros(3), dict(ENV), CTX, step_times=times)
    else:
        actual, infos = prop.try_propagate_adaptive_sampled_steps(X, widths, [1, 3], 0., np.zeros(3), dict(ENV), CTX, step_times=times)
        assert len(infos) == 3
    np.testing.assert_allclose(actual, rows, rtol=0., atol=2e-11)


def test_model_specific_frames_and_tide_options_match_callback():
    from dataclasses import replace
    forces = models()
    older = replace(forces[0].frames, jd_utc_start=2455197.5, dut1_s=.1, xp_arcsec=.09, yp_arcsec=-.02)
    forces[0] = replace(forces[0], frames=older, albedo=False, area_m2=12.)
    forces[2] = replace(forces[2], frames=older, tide_system="zero_tide", pole_tide=False)
    forces[3] = replace(forces[3], order=0, pole_tide=False)
    prop = OrbitPropagator(numeric_backend="rust", plugins=forces)
    reference = OrbitPropagator(numeric_backend="rust", plugins=forces)
    actual = prop.propagate(X, 60., 20., np.zeros(3), dict(ENV), CTX)
    expected = reference.propagate(X, 60., 20., np.zeros(3), dict(ENV, _rust_precision_force_plan_disabled=True), CTX)
    np.testing.assert_allclose(actual, expected, rtol=0., atol=2e-11)


def test_exact_frame_cache_refreshes_on_atomic_replacement_and_deletion(tmp_path):
    import os
    from dataclasses import replace
    eop = tmp_path / "eop.txt"
    text = "VERSION test\nNUM_OBSERVED_POINTS 2\n2024 01 01 60310.0 0.10 0.20 0.30 0 0 0 0 0 37\n2024 01 02 60311.0 0.11 0.21 0.31 0 0 0 0 0 37\n"
    eop.write_text(text)
    frame = FrameContext(model="iau76_80_eop", jd_utc_start=2460310.5, eop_path=str(eop))
    forces = [replace(f, frames=frame) if hasattr(f, "frames") else f for f in models()]
    prop = OrbitPropagator(numeric_backend="rust", plugins=forces)
    *_, stage, context = make_plan(prop, X, 0., dict(ENV), CTX)
    original = stage(0., X)
    metadata = eop.stat()
    replacement = tmp_path / "new.txt"
    replacement.write_text(text.replace("0.30", "0.80"))
    os.utime(replacement, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
    os.replace(replacement, eop)
    changed = stage(0., X)
    assert original != changed
    fresh = make_plan(OrbitPropagator(numeric_backend="rust", plugins=forces), X, 0., dict(ENV), CTX)
    np.testing.assert_array_equal(changed, fresh[5](0., X))
    eop.unlink()
    with pytest.raises(FileNotFoundError):
        stage(0., X)


@pytest.mark.parametrize("integrator", ["rk4", "rkf78", "dopri5"])
def test_stage_callback_errors_are_preserved(integrator):
    prop = OrbitPropagator(numeric_backend="rust", integrator=integrator, plugins=[models()[0]])
    def broken(*args):
        raise LookupError("ephemeris unavailable")
    with pytest.raises(LookupError, match="ephemeris unavailable"):
        prop.propagate(X, 10., 0., np.zeros(3), {"ephemeris_callable": broken}, CTX)
