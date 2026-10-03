"""Simple-frame native stages retain fresh authoritative density evaluations."""

import numpy as np
import pytest

from sim.aero.core import atmosphere_relative_velocity_eci_km_s
from sim.dynamics.orbit.accelerations import OrbitContext
from sim.dynamics.orbit.propagator import OrbitPropagator, drag_plugin, j2_plugin, j3_plugin
from sim.dynamics.orbit.rust_force_plan import immutable_builtin_plan_signature, make_plan
from sim.rust_environment_backend import try_simple_relative_velocity


def _native():
    native = pytest.importorskip("oel_rust_orbit")
    if not callable(getattr(native, "environment_simple_relative_velocity", None)):
        pytest.skip("installed wheel lacks simple-frame stage fusion")
    return native


@pytest.mark.parametrize("epoch", [None, 2459669.5])
@pytest.mark.parametrize("omega", [7.2921159e-5, 0., -3e-5, 1e-4])
def test_simple_native_relative_velocity_matches_authoritative_frame(epoch, omega):
    _native()
    r = np.array([7000., 300., 900.])
    v = np.array([0.1, 7.4, 0.8])
    for time in (-120., 0., 120., 3000.):
        expected = atmosphere_relative_velocity_eci_km_s(r, v, t_s=time,
            frame_model="simple", jd_utc_start=epoch, earth_rotation_rad_s=omega)
        actual = try_simple_relative_velocity(r, v, time, jd_utc_start=epoch,
            atmosphere_rotation_rad_s=omega)
        np.testing.assert_allclose(actual, expected, rtol=0., atol=4e-15)


@pytest.mark.parametrize("epoch", [None, 2459669.5])
def test_native_simple_drag_sampled_stage_density_is_fresh_and_matches_full_stages(epoch):
    _native()
    prop = OrbitPropagator(numeric_backend="rust", plugins=[j2_plugin, drag_plugin, j3_plugin])
    x = np.array([6800., 300., 900., 0.1, 7.4, 0.8])
    ctx = OrbitContext(398600.4418, 300., 4., 2.2, 1.2)
    env = dict(atmosphere_model="exponential", drag_frame_model="simple", jd_utc_start=epoch,
        exponential_reference_density_kg_m3=1e-8, exponential_reference_altitude_km=400.,
        exponential_scale_height_km=57., exponential_ceiling_altitude_km=1000.)
    *_, callback, context = make_plan(prop, x, 0., env, ctx)
    widths = np.array([0.03, 0.03, 0.03, 0.01, 0.03, 0.03, 0.03, 0.01], dtype="<f8")
    times = np.array([0., 0.03, 0.06, 0.09, 0.1, 0.13, 0.16, 0.19], dtype="<f8")
    ends = [4, 8]
    command = [1e-5, -2e-5, 3e-5]
    expected = np.frombuffer(context.sampled_history(x, times.tobytes(), widths.tobytes(),
        ends, command, callback), dtype="<f8").reshape(-1, 6)
    seen = []

    def density(time, state):
        seen.append((time, np.array(state)))
        return callback._simple_drag_density_callback(time, state)

    actual = np.frombuffer(context.sampled_history(x, times.tobytes(), widths.tobytes(),
        ends, command, density, True, epoch), dtype="<f8").reshape(-1, 6)
    assert len(seen) == 4 * len(widths)
    for index, (time, width) in enumerate(zip(times, widths)):
        np.testing.assert_array_equal([row[0] for row in seen[index * 4:index * 4 + 4]],
            [time, time + width / 2., time + width / 2., time + width])
        assert not np.array_equal(seen[index * 4 + 1][1], seen[index * 4 + 2][1])
    np.testing.assert_allclose(actual, expected, rtol=0., atol=2e-12)


def test_eop_drag_excludes_simple_native_stage_mode():
    prop = OrbitPropagator(numeric_backend="rust", plugins=[drag_plugin])
    ctx = OrbitContext(398600.4418, 300., 1., 2.2, 1.2)
    *_, callback, _ = make_plan(prop, np.array([7000., 0., 0., 0., 7.5, 0.]), 0.,
        dict(density_kg_m3=1e-12, drag_frame_model="iau76_80_eop", jd_utc_start=2459669.5,
             dut1_s=-0.12, xp_arcsec=0.08, yp_arcsec=0.31, dat_s=37.), ctx)
    assert not hasattr(callback, "_simple_drag_density_callback")


def test_force_context_fast_container_extraction_retains_custom_sequence_semantics():
    pytest.importorskip("oel_rust_orbit")
    prop = OrbitPropagator(numeric_backend="rust", plugins=[j2_plugin])
    ctx = OrbitContext(398600.4418, 300., 1., 2.2, 1.2)
    state = [7000., 300., 900., 0.1, 7.4, 0.8]
    *_, callback, context = make_plan(prop, np.array(state), 120., {}, ctx)
    stage = callback(120., state)

    class CustomSequence(list):
        def __init__(self, values):
            super().__init__(values)
            self.iterations = 0

        def __getitem__(self, index):
            raise AssertionError("the established Vec path consumes __iter__, not __getitem__")

        def __iter__(self):
            self.iterations += 1
            return iter([value * 1.01 for value in super().__iter__()])

    command = [1e-5, 2e-5, -3e-5]
    custom_state, custom_stage, custom_command = map(CustomSequence, (state, stage, command))
    expected = context.acceleration([value * 1.01 for value in state],
        [value * 1.01 for value in stage], [value * 1.01 for value in command])
    actual = context.acceleration(custom_state, custom_stage, custom_command)
    np.testing.assert_array_equal(actual, expected)
    assert custom_state.iterations == custom_stage.iterations == custom_command.iterations == 1
    callback_rows = []

    def custom_callback(_time, _state):
        row = CustomSequence(stage)
        callback_rows.append(row)
        return row

    next_expected = context.rk4(state, 0., 0.1, command,
        lambda _time, _state: [value * 1.01 for value in stage])
    next_actual = context.rk4(state, 0., 0.1, command, custom_callback)
    np.testing.assert_array_equal(next_actual, next_expected)
    assert len(callback_rows) == 4
    assert all(row.iterations == 1 for row in callback_rows)


def test_force_input_malformed_float_precedes_wrong_length_error():
    native = pytest.importorskip("oel_rust_orbit")
    with pytest.raises(TypeError):
        native.perturbation_third_body([1., object()], [384400., 0., 0.], 4902.8)
    with pytest.raises(ValueError, match="position must have three values"):
        native.perturbation_third_body([1., 2.], [384400., 0., 0.], 4902.8)
    prop = OrbitPropagator(numeric_backend="rust", plugins=[j2_plugin])
    ctx = OrbitContext(398600.4418, 300., 1., 2.2, 1.2)
    state = [7000., 300., 900., 0.1, 7.4, 0.8]
    *_, callback, context = make_plan(prop, np.array(state), 0., {}, ctx)
    stage = callback(0., state)
    with pytest.raises(TypeError):
        context.acceleration([1., object()], stage, [0., 0., 0.])
    with pytest.raises(ValueError, match="state must have 6 values"):
        context.acceleration([1., 2.], stage, [0., 0., 0.])


def test_legacy_vector_binding_conversion_order_precedes_scalar_and_state_validation():
    native = pytest.importorskip("oel_rust_orbit")
    seen = []

    class Coercion:
        def __init__(self, name, error=False):
            self.name, self.error = name, error

        def __float__(self):
            seen.append(self.name)
            if self.error:
                raise TypeError(self.name)
            return 0.

    with pytest.raises(TypeError, match="vector first"):
        native.perturbation_third_body([1., Coercion("vector first", True)],
            [384400., 0., 0.], Coercion("scalar later", True))
    assert seen == ["vector first"]
    prop = OrbitPropagator(numeric_backend="rust", plugins=[j2_plugin])
    ctx = OrbitContext(398600.4418, 300., 1., 2.2, 1.2)
    state = [7000., 300., 900., 0.1, 7.4, 0.8]
    *_, callback, context = make_plan(prop, np.array(state), 0., {}, ctx)
    stage = callback(0., state)
    seen.clear()
    with pytest.raises(TypeError, match="stage conversion"):
        context.acceleration([1., 2.], [Coercion("stage conversion", True)], [0., 0., 0.])
    assert seen == ["stage conversion"]
    seen.clear()
    stage[9] = Coercion("stage hook")
    with pytest.raises(ValueError, match="finite"):
        context.acceleration([np.inf] * 6, stage, [Coercion("command hook"), 0., 0.])
    assert seen == ["stage hook", "command hook"]


def test_unhashable_custom_plugin_cannot_impersonate_builtin_and_keeps_scalar_fallback():
    pytest.importorskip("oel_rust_orbit")

    class CustomPlugin:
        __hash__ = None

        def __init__(self):
            self.calls = 0

        def __eq__(self, _other):
            return True

        def __call__(self, _time, _state, _env, _ctx):
            self.calls += 1
            return np.zeros(3)

    plugin = CustomPlugin()
    prop = OrbitPropagator(numeric_backend="rust", plugins=[plugin])
    ctx = OrbitContext(398600.4418, 300., 1., 2.2, 1.2)
    state = np.array([7000., 300., 900., 0.1, 7.4, 0.8])
    assert immutable_builtin_plan_signature(prop, {}, ctx) is None
    assert prop.try_propagate_sampled_steps(state, [0.1], [1], 0., np.zeros(3), {}, ctx) is None
    assert prop.try_propagate_fixed_steps(state, 0.1, 1, 0., np.zeros(3), {}, ctx) is None
    with pytest.raises(ValueError, match="does not support"):
        make_plan(prop, state, 0., {}, ctx)
    actual = prop.propagate(state, 0.1, 0., np.zeros(3), {}, ctx)
    reference = OrbitPropagator(numeric_backend="rust").propagate(state, 0.1, 0., np.zeros(3), {}, ctx)
    assert plugin.calls == 4
    np.testing.assert_allclose(actual, reference, rtol=0., atol=2e-12)
