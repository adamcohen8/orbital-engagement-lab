"""Frame-unit and resource-snapshot regressions for Rust ONP stages."""

import numpy as np
import pytest

from sim.aero.core import atmosphere_relative_velocity_eci_km_s
from sim.dynamics.orbit.accelerations import OrbitContext
from sim.dynamics.orbit.frames import (
    FrameContext,
    PreparedFrameEvaluator,
    _load_nut80_table,
    eci_to_ecef_rotation_context,
    eci_to_ecef_rotation_derivative_context,
)
from sim.dynamics.orbit.propagator import OrbitPropagator, drag_plugin


@pytest.fixture
def native_context():
    native = pytest.importorskip("oel_rust_orbit")
    coefficients, terms = _load_nut80_table()
    return native.ONPEnvironmentContext(coefficients.ravel().tolist(), terms.ravel().tolist())


@pytest.fixture
def eop_path(tmp_path):
    path = tmp_path / "eop.txt"
    path.write_text(
        "NUM_OBSERVED_POINTS 2\n"
        "2022 03 31 59669 0.100000 0.200000 -0.100000 0 0 0 0 0 37\n"
        "2022 04 01 59670 0.200000 0.300000 -0.120000 0 0 0 0 0 37\n"
    )
    return path


@pytest.mark.parametrize("epoch", [None, 2459669.5])
def test_rust_simple_atmosphere_velocity_is_eci(epoch):
    """The native state batch's ECEF velocity must not enter the ECI force sum."""
    native = pytest.importorskip("oel_rust_orbit")
    r = np.array([7000., 300., 900.])
    v = np.array([0.1, 7.4, 0.8])
    options = dict(t_s=120., frame_model="simple", jd_utc_start=epoch)
    expected = atmosphere_relative_velocity_eci_km_s(r, v, **options)
    ecef = np.asarray(native.environment_state_batch([120.], r.tolist(), v.tolist(), epoch, 7.2921159e-5, True)[1])
    assert np.linalg.norm(ecef - expected) > 0.01
    actual = atmosphere_relative_velocity_eci_km_s(r, v, _numeric_backend="rust", **options)
    np.testing.assert_allclose(actual, expected, rtol=0., atol=4e-15)


@pytest.mark.parametrize("elapsed", [0., 17.5, 120., 86400.])
def test_prepared_native_rotation_and_same_derivative_stencil(native_context, eop_path, elapsed):
    ctx = FrameContext(model="iau76_80_eop", jd_utc_start=2459669.5, eop_path=str(eop_path))
    prepared = PreparedFrameEvaluator(ctx, native_context)
    prepared.refresh()
    np.testing.assert_allclose(prepared.rotation(elapsed), eci_to_ecef_rotation_context(elapsed, ctx), rtol=0., atol=3e-14)
    # Both centered and boundary one-sided stencils use interpolated EOP at
    # every stencil time; the native calculation retains the same samples.
    np.testing.assert_allclose(prepared.derivative(elapsed), eci_to_ecef_rotation_derivative_context(elapsed, ctx), rtol=0., atol=3e-16)


def test_prepared_frame_refresh_observes_rewrite_and_missing_resource(native_context, eop_path):
    ctx = FrameContext(model="iau76_80_eop", jd_utc_start=2459669.5, eop_path=str(eop_path))
    prepared = PreparedFrameEvaluator(ctx, native_context)
    initial = prepared.rotation(120.).copy()
    eop_path.write_text(eop_path.read_text().replace("0.100000", "1.100000"))
    prepared.refresh()
    changed = prepared.rotation(120.)
    assert not np.array_equal(changed, initial)
    np.testing.assert_allclose(changed, eci_to_ecef_rotation_context(120., ctx), rtol=0., atol=3e-14)
    eop_path.unlink()
    with pytest.raises(FileNotFoundError):
        prepared.refresh()


@pytest.mark.parametrize("use_native", [False, True])
def test_prepared_relative_eop_path_rebinds_after_cwd_change(
    native_context, eop_path, tmp_path, monkeypatch, use_native,
):
    other = tmp_path / "other"
    other.mkdir()
    (other / "eop.txt").write_text(eop_path.read_text().replace("0.100000", "1.100000"))
    missing = tmp_path / "missing"
    missing.mkdir()
    monkeypatch.chdir(tmp_path)
    ctx = FrameContext(model="iau76_80_eop", jd_utc_start=2459669.5, eop_path="eop.txt")
    prepared = PreparedFrameEvaluator(ctx, native_context if use_native else None)
    initial = prepared.rotation(120.).copy()
    prepared.derivative(120.)

    monkeypatch.chdir(other)
    prepared.refresh()
    changed = prepared.rotation(120.)
    assert not np.array_equal(changed, initial)
    np.testing.assert_allclose(changed, eci_to_ecef_rotation_context(120., ctx), rtol=0., atol=3e-14)
    np.testing.assert_allclose(
        prepared.derivative(120.), eci_to_ecef_rotation_derivative_context(120., ctx),
        rtol=0., atol=3e-16,
    )

    monkeypatch.chdir(missing)
    with pytest.raises(FileNotFoundError):
        prepared.refresh()
    monkeypatch.chdir(tmp_path)
    prepared.refresh()
    np.testing.assert_array_equal(prepared.rotation(120.), initial)


@pytest.mark.parametrize("use_eop", [False, True])
def test_prepared_absolute_or_no_eop_refresh_does_not_resolve_cwd(
    native_context, eop_path, tmp_path, monkeypatch, use_eop,
):
    from sim.dynamics.orbit import frames

    ctx = FrameContext(
        model="iau76_80_eop" if use_eop else "simple_gmst",
        jd_utc_start=2459669.5, eop_path=str(eop_path) if use_eop else None,
    )
    prepared = PreparedFrameEvaluator(ctx, native_context)
    rotation = prepared.rotation(120.)
    derivative = prepared.derivative(120.)
    other = tmp_path / "unrelated"
    other.mkdir()
    monkeypatch.chdir(other)

    def unexpected_resolution(path):
        raise AssertionError("absolute/no EOP inputs must not re-resolve their path")

    monkeypatch.setattr(frames, "_eop_cache_path", unexpected_resolution)
    prepared.refresh()
    assert prepared.rotation(120.) is rotation
    assert prepared.derivative(120.) is derivative


def test_prepared_frame_rejects_dat_crossing(native_context, eop_path):
    eop_path.write_text(eop_path.read_text().replace("0 0 37\n", "0 0 38\n", 1))
    ctx = FrameContext(model="iau76_80_eop", jd_utc_start=2459669.5, eop_path=str(eop_path))
    prepared = PreparedFrameEvaluator(ctx, native_context)
    with pytest.raises(ValueError, match="DAT changes"):
        prepared.rotation(120.)
    with pytest.raises(ValueError, match="leap-second boundary"):
        prepared.rotation(86400.)


def test_same_stage_time_different_states_recompute_density_and_drag(native_context):
    from sim.dynamics.orbit.rust_force_plan import make_plan

    calls = []
    def density(alt, lat, lon, epoch, env):
        calls.append((alt, lat, lon))
        return 1e-12 * (1. + alt / 1000.)
    env = dict(atmosphere_model="nrlmsise00", nrlmsise00_density_callable=density,
               geodetic_model="wgs84", drag_frame_model="simple", jd_utc_start=2459669.5)
    context = OrbitContext(398600.4418, 300., 1., 2.2, 1.2)
    prop = OrbitPropagator(numeric_backend="rust", plugins=[drag_plugin])
    state = np.array([7000., 300., 900., 0.1, 7.4, 0.8])
    callback = make_plan(prop, state, 0., env, context)[5]
    a = np.asarray(callback(120., state))
    b = np.asarray(callback(120., state * 1.01))
    assert len(calls) == 2 and calls[0] != calls[1]
    assert a[9] != b[9]
    assert not np.array_equal(a[10:13], b[10:13])
    np.testing.assert_allclose(a[10:13], atmosphere_relative_velocity_eci_km_s(state[:3], state[3:], t_s=120., frame_model="simple", jd_utc_start=2459669.5), rtol=0., atol=4e-15)


def test_sampled_schedule_preserves_explicit_decimal_stage_epochs_and_history():
    pytest.importorskip('oel_rust_orbit')
    # Construct through the supported force-plan route; older wheels fall back.
    env = {'density_kg_m3': 1e-12, 'drag_frame_model': 'simple', 'jd_utc_start': 2459669.5}
    ctx = OrbitContext(398600.4418, 300., 1., 2.2, 1.2)
    state = np.array([7000., 300., 900., .1, 7.4, .8])
    widths = [0.1, 0.1, 0.09999999999999998, .1, .050000000000000044]
    times = [0.2, 0.30000000000000004, 0.4, 0.5, 0.6]
    ends = [3, 5]
    scalar = OrbitPropagator(numeric_backend='rust', plugins=[drag_plugin])
    expected = [state.copy()]
    for index, (time, width) in enumerate(zip(times, widths)):
        state = scalar.propagate(state, width, time, np.zeros(3), dict(env), ctx)
        if index + 1 in ends:
            expected.append(state.copy())
    prepared = OrbitPropagator(numeric_backend='rust', plugins=[drag_plugin])
    actual = prepared.try_propagate_sampled_steps(expected[0], widths, ends, .2, np.zeros(3), env, ctx, step_times=times)
    if actual is None:
        pytest.skip('installed wheel predates force-plan sampled history')
    np.testing.assert_array_equal(actual, expected)
    assert prepared.last_numeric_path == 'rust_native_force_plan_sampled_history'


def test_speculative_signature_binds_state_independent_inputs_and_excludes_callbacks(tmp_path):
    from sim.dynamics.orbit.rust_force_plan import immutable_builtin_plan_signature
    prop = OrbitPropagator(numeric_backend='rust', plugins=[drag_plugin])
    ctx = OrbitContext(398600.4418, 300., 1., 2.2, 1.2)
    resource = tmp_path / 'eop.txt'
    resource.write_text('first')
    env = {'density_kg_m3': 1e-12, 'drag_eop_path': str(resource), 'world_truth': object()}
    initial = immutable_builtin_plan_signature(prop, env, ctx)
    assert initial is not None
    env['world_truth'] = object()
    assert immutable_builtin_plan_signature(prop, env, ctx) == initial
    env['density_kg_m3'] = 2e-12
    changed = immutable_builtin_plan_signature(prop, env, ctx)
    assert changed != initial
    resource.write_text('other')
    assert immutable_builtin_plan_signature(prop, env, ctx) != changed
    env['frames'] = FrameContext(model='simple', eop_path=str(resource))
    with_frame = immutable_builtin_plan_signature(prop, env, ctx)
    assert with_frame is not None
    env['frames'] = FrameContext(model='simple', eop_path=str(resource), tt_minus_utc_s=70.)
    assert immutable_builtin_plan_signature(prop, env, ctx) != with_frame
    resource.write_text('next version')
    with_changed_resource = immutable_builtin_plan_signature(prop, env, ctx)
    assert with_changed_resource is not None
    resource.write_text('third version')
    assert immutable_builtin_plan_signature(prop, env, ctx) != with_changed_resource
    class CustomFrameContext(FrameContext):
        pass
    env['frames'] = CustomFrameContext()
    assert immutable_builtin_plan_signature(prop, env, ctx) is None
    env.pop('frames')
    env['nrlmsise00_density_callable'] = lambda *_: 0.
    assert immutable_builtin_plan_signature(prop, env, ctx) is None
    env.pop('nrlmsise00_density_callable')
    prop.plugins.append(lambda *_: np.zeros(3))
    assert immutable_builtin_plan_signature(prop, env, ctx) is None
