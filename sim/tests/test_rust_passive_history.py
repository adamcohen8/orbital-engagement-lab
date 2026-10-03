"""Native scenario histories retain scalar steps and event boundaries."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from sim.config import scenario_config_from_dict
from sim.dynamics.orbit.propagator import OrbitContext, OrbitPropagator, j2_plugin, j3_plugin, j4_plugin
from sim.single_run import _SingleRunEngine

native = pytest.importorskip("oel_rust_orbit")
pytestmark = pytest.mark.skipif(
    not hasattr(native, "propagate_passive_sampled_eci_bytes"), reason="requires Rust wheel 0.8",
)


def _raw(tmp_path, *, j2=False, duration=None, dt=0.7, substep=0.1):
    return {
        "scenario_name": "rust_passive_history",
        "objects": {"satellite": {
            "kind": "satellite", "runtime_profile": "trajectory_only",
            "initial_state": {
                "position_eci_km": [7000.0, 0.0, 0.0],
                "velocity_eci_km_s": [0.0, 7.5, 1.0],
            },
        }},
        "simulator": {
            "duration_s": 12 * dt if duration is None else duration, "dt_s": dt,
            "dynamics": {
                "orbit": {"numeric_backend": "rust", "j2": j2, "orbit_substep_s": substep},
                "attitude": {"enabled": False},
            },
        },
        "outputs": {
            "output_dir": str(tmp_path), "plots": {"enabled": False},
            "stats": {"print_summary": False, "save_full_log": False},
        },
    }


def _engine(raw, *, batch=True, callback=None):
    engine = _SingleRunEngine(scenario_config_from_dict(deepcopy(raw)), step_callback=callback)
    if not batch:
        engine.satellite_stepper.passive_history.step = lambda **_kwargs: None
    return engine


def _finish(engine):
    while not engine.done:
        engine.step()
    return engine


@pytest.mark.parametrize("j2", [False, True])
@pytest.mark.parametrize("dt,substep", [(0.7, 0.1), (2.0, 0.25)])
def test_native_batch_matches_scalar_rust_history_and_progress(tmp_path, j2, dt, substep):
    raw = _raw(tmp_path, j2=j2, dt=dt, substep=substep)
    callback_steps = []
    batched = _finish(_engine(raw, callback=lambda step, _total: callback_steps.append(step)))
    scalar = _finish(_engine(raw, batch=False))
    np.testing.assert_array_equal(batched.t_s[:batched.current_index + 1], scalar.t_s[:scalar.current_index + 1])
    np.testing.assert_array_equal(
        batched.truth_hist["satellite"][:batched.current_index + 1],
        scalar.truth_hist["satellite"][:scalar.current_index + 1],
    )
    cache = batched.satellite_stepper.passive_history
    assert cache.preparation_count == 1
    assert cache.consumed_samples == batched.current_index
    assert callback_steps == list(range(batched.current_index + 1))


@pytest.mark.parametrize("zonal", [False, True])
@pytest.mark.parametrize("integrator", ["rkf78", "adaptive", "dopri5"])
def test_adaptive_native_history_matches_scalar_truth_and_step_evidence(tmp_path, zonal, integrator):
    if not hasattr(native, "propagate_adaptive_sampled_eci_bytes"):
        pytest.skip("requires adaptive sampled history kernel")
    raw = _raw(tmp_path, j2=zonal, dt=0.6, substep=0.03, duration=8.4)
    orbit = raw["simulator"]["dynamics"]["orbit"]
    orbit.update({"integrator": integrator, "j3": zonal, "j4": zonal})
    batched, scalar = _finish(_engine(raw)), _finish(_engine(raw, batch=False))
    count = batched.current_index + 1
    np.testing.assert_array_equal(batched.t_s[:count], scalar.t_s[:count])
    np.testing.assert_array_equal(
        batched.truth_hist["satellite"][:count], scalar.truth_hist["satellite"][:count],
    )
    batched_propagator = batched.agents["satellite"].dynamics.orbit_propagator
    scalar_propagator = scalar.agents["satellite"].dynamics.orbit_propagator
    assert batched_propagator.adaptive_step_info == scalar_propagator.adaptive_step_info
    assert batched_propagator.last_adaptive_step_info == scalar_propagator.last_adaptive_step_info
    assert batched.satellite_stepper.passive_history.preparation_count == 1


def test_adaptive_history_invalidates_on_impulse_and_cadence_change(tmp_path):
    if not hasattr(native, "propagate_adaptive_sampled_eci_bytes"):
        pytest.skip("requires adaptive sampled history kernel")
    raw = _raw(tmp_path, j2=True, dt=0.7, substep=0.1, duration=8.4)
    raw["simulator"]["dynamics"]["orbit"]["integrator"] = "dopri5"
    batched, scalar = _engine(raw), _engine(raw, batch=False)
    for engine in (batched, scalar):
        engine.step()
        engine.apply_impulse("satellite", [0.2, -0.1, 0.05], maneuver_id="adaptive_impulse")
        engine.step(dt_s=0.13)
        _finish(engine)
    count = batched.current_index + 1
    np.testing.assert_array_equal(
        batched.truth_hist["satellite"][:count], scalar.truth_hist["satellite"][:count],
    )
    assert batched.impulsive_maneuvers == scalar.impulsive_maneuvers
    assert batched.satellite_stepper.passive_history.preparation_count >= 2


@pytest.mark.parametrize("integrator", ["rkf78", "dopri5"])
@pytest.mark.parametrize("perturb", [False, True])
def test_adaptive_full_force_history_preserves_truth_diagnostics_and_invalidation(tmp_path, integrator, perturb):
    raw = _raw(tmp_path, dt=0.6, substep=0.03, duration=8.4)
    raw["simulator"]["initial_jd_utc"] = 2459669.5
    raw["simulator"]["environment"] = {"atmosphere_model": "exponential", "srp_shadow_model": "conical"}
    raw["simulator"]["dynamics"]["orbit"].update({
        "integrator": integrator, "drag": True, "drag_frame_model": "simple",
        "srp": True, "third_body_sun": True, "third_body_moon": True,
    })
    batched, scalar = _engine(raw), _engine(raw, batch=False)
    for engine in (batched, scalar):
        for _ in range(3):
            engine.step()
        if perturb:
            engine.apply_impulse("satellite", [.2, -.1, .05], maneuver_id="adaptive_full_force")
            engine.agents["satellite"].dynamics = replace(engine.agents["satellite"].dynamics, cd=2.3)
            engine.step(dt_s=0.13)
        _finish(engine)
    np.testing.assert_array_equal(batched.truth_hist["satellite"], scalar.truth_hist["satellite"])
    np.testing.assert_array_equal(batched.t_s, scalar.t_s)
    assert batched.impulsive_maneuvers == scalar.impulsive_maneuvers
    actual = batched.agents["satellite"].dynamics.orbit_propagator
    expected = scalar.agents["satellite"].dynamics.orbit_propagator
    assert actual.adaptive_step_info == expected.adaptive_step_info
    assert actual.last_adaptive_step_info == expected.last_adaptive_step_info
    assert actual._rkf78_h_next == expected._rkf78_h_next
    assert batched.satellite_stepper.passive_history.preparation_count >= (2 if perturb else 1)


@pytest.mark.parametrize("j2", [False, True])
@pytest.mark.parametrize("requested_dt", [0.1, 0.5, 2.0, 10.0])
def test_overridden_cadence_reuses_batches_and_matches_scalar_history(tmp_path, j2, requested_dt):
    steps = 32
    raw = _raw(tmp_path, j2=j2, duration=float(np.ceil(steps * requested_dt)), dt=1.0, substep=0.25)
    batched, scalar = _engine(raw), _engine(raw, batch=False)
    for engine in (batched, scalar):
        for _ in range(steps):
            engine.step(dt_s=requested_dt)
    np.testing.assert_array_equal(
        batched.t_s[:batched.current_index + 1], scalar.t_s[:scalar.current_index + 1],
    )
    np.testing.assert_array_equal(
        batched.truth_hist["satellite"][:batched.current_index + 1],
        scalar.truth_hist["satellite"][:scalar.current_index + 1],
    )
    cache = batched.satellite_stepper.passive_history
    assert cache.consumed_samples == steps
    assert cache.preparation_count <= 2
    assert batched.dt == scalar.dt == 1.0


def test_changing_override_cadence_reuses_each_segment_and_preserves_impulses(tmp_path):
    cadences = (0.1, 0.5, 2.0, 0.5)
    steps_per_segment = 20
    raw = _raw(tmp_path, j2=True, duration=sum(cadences) * steps_per_segment + 1.0, dt=1.0, substep=0.25)
    batched, scalar = _engine(raw), _engine(raw, batch=False)
    for engine in (batched, scalar):
        for segment, dt in enumerate(cadences):
            if segment == 2:
                engine.apply_impulse("satellite", [2.0, -1.0, 0.5], maneuver_id="cadence_change")
            for _ in range(steps_per_segment):
                engine.step(dt_s=dt)
    np.testing.assert_array_equal(
        batched.t_s[:batched.current_index + 1], scalar.t_s[:scalar.current_index + 1],
    )
    np.testing.assert_array_equal(
        batched.truth_hist["satellite"][:batched.current_index + 1],
        scalar.truth_hist["satellite"][:scalar.current_index + 1],
    )
    assert batched.impulsive_maneuvers == scalar.impulsive_maneuvers
    cache = batched.satellite_stepper.passive_history
    assert cache.consumed_samples == len(cadences) * steps_per_segment
    assert cache.preparation_count <= len(cadences) + 1


def test_impulse_and_custom_step_invalidate_future_rows(tmp_path):
    raw = _raw(tmp_path)
    batched, scalar = _engine(raw), _engine(raw, batch=False)
    for engine in (batched, scalar):
        engine.step()
        engine.apply_impulse("satellite", [2.0, -1.0, 0.5], maneuver_id="impulse")
        engine.step(dt_s=0.13)
        _finish(engine)
    np.testing.assert_array_equal(
        batched.truth_hist["satellite"][:batched.current_index + 1],
        scalar.truth_hist["satellite"][:scalar.current_index + 1],
    )
    assert batched.impulsive_maneuvers == scalar.impulsive_maneuvers
    assert batched.satellite_stepper.passive_history.preparation_count >= 2


def test_bounded_chunks_and_changed_gravity_preserve_scalar_history(tmp_path):
    raw = _raw(tmp_path, duration=51.3, dt=0.1, substep=0.1)
    batched, scalar = _engine(raw), _engine(raw, batch=False)
    for engine in (batched, scalar):
        for _ in range(257):
            engine.step()
        agent = engine.agents["satellite"]
        agent.dynamics = replace(agent.dynamics, mu_km3_s2=agent.dynamics.mu_km3_s2 * 1.00001)
        _finish(engine)
    np.testing.assert_array_equal(
        batched.truth_hist["satellite"][:batched.current_index + 1],
        scalar.truth_hist["satellite"][:scalar.current_index + 1],
    )
    assert batched.satellite_stepper.passive_history.preparation_count >= 3


def test_native_batch_preserves_first_impact(tmp_path):
    raw = _raw(tmp_path, duration=10.5, dt=0.7)
    raw["objects"]["satellite"]["initial_state"] = {
        "position_eci_km": [6380.0, 0.0, 0.0], "velocity_eci_km_s": [-20.0, 0.0, 0.0],
    }
    batched, scalar = _finish(_engine(raw)), _finish(_engine(raw, batch=False))
    assert batched.terminated_early and scalar.terminated_early
    assert batched.termination_reason == scalar.termination_reason
    assert batched.termination_time_s == scalar.termination_time_s
    assert batched.current_index == scalar.current_index == 1
    np.testing.assert_array_equal(batched.truth_hist["satellite"][:2], scalar.truth_hist["satellite"][:2])


def test_custom_force_callbacks_are_not_precomputed(tmp_path):
    raw = _raw(tmp_path, duration=2.0, dt=1.0)
    engine = _engine(raw)
    calls = []

    def force(time_s, _state, _env, _context):
        calls.append(time_s)
        return np.zeros(3)

    engine.agents["satellite"].dynamics.orbit_propagator.plugins = [force]
    engine.step()
    assert calls
    assert max(calls) <= 1.0 + 1e-12
    assert engine.satellite_stepper.passive_history.preparation_count == 0


@pytest.mark.parametrize("force", ["zonal", "exponential_drag"])
def test_builtin_force_sampled_history_preserves_nested_decimal_clocks_and_invalidation(tmp_path, force):
    if not hasattr(getattr(native, "ONPForceContext", object), "sampled_history"):
        pytest.skip("requires sampled built-in force wheel")
    raw = _raw(tmp_path, dt=0.6, substep=0.03, duration=8.4)
    orbit = raw["simulator"]["dynamics"]["orbit"]
    if force == "zonal":
        orbit.update({"j2": True, "j3": True, "j4": True})
    else:
        orbit.update({"drag": True, "drag_frame_model": "simple"})
        raw["simulator"]["initial_jd_utc"] = 2459669.5
        raw["simulator"]["environment"] = {"atmosphere_model": "exponential"}
    batched, scalar = _engine(raw), _engine(raw, batch=False)
    for engine in (batched, scalar):
        engine.sim_substep_s = 0.1
        for _ in range(3):
            engine.step()
        engine.apply_impulse("satellite", [0.2, -0.1, 0.05], maneuver_id="sampled_force")
        agent = engine.agents["satellite"]
        agent.dynamics = replace(agent.dynamics, cd=agent.dynamics.cd * 1.01)
        _finish(engine)
    np.testing.assert_array_equal(batched.truth_hist["satellite"], scalar.truth_hist["satellite"])
    np.testing.assert_array_equal(batched.t_s, scalar.t_s)
    assert batched.impulsive_maneuvers == scalar.impulsive_maneuvers
    assert 2 <= batched.satellite_stepper.passive_history.preparation_count <= 3
    assert batched.satellite_stepper.passive_history.consumed_samples == batched.current_index


def test_callable_environment_excludes_speculative_builtin_forces(tmp_path):
    raw = _raw(tmp_path)
    raw["simulator"]["dynamics"]["orbit"].update({"j3": True, "drag": True, "drag_frame_model": "simple"})
    raw["simulator"]["initial_jd_utc"] = 2459669.5
    raw["simulator"]["environment"] = {"density_kg_m3": 1e-12}
    engine = _engine(raw)
    engine.base_environment["custom_model"] = lambda: None
    engine.step()
    assert engine.satellite_stepper.passive_history.preparation_count == 0


def test_zonal_signature_binds_order_gravity_and_cadence_without_irrelevant_env(tmp_path):
    raw = _raw(tmp_path, dt=1.0, substep=0.25, duration=20.0)
    raw["simulator"]["dynamics"]["orbit"].update({"j2": True, "j3": True, "j4": True})
    batched, scalar = _engine(raw), _engine(raw, batch=False)
    for engine in (batched, scalar):
        engine.step()
        # Zonal forces never evaluate weather/ephemeris fields or callbacks.
        engine.base_environment["unused_density_callback"] = lambda: None
        engine.step()
        agent = engine.agents["satellite"]
        agent.dynamics = replace(agent.dynamics, mu_km3_s2=agent.dynamics.mu_km3_s2 * 1.00001)
        engine.step()
        agent.dynamics.orbit_propagator.plugins = [j4_plugin, j2_plugin, j3_plugin]
        engine.step()
        agent.dynamics.orbit_propagator.plugins = [j3_plugin, j2_plugin]
        for _ in range(3):
            engine.step(dt_s=0.5)
        _finish(engine)
    np.testing.assert_array_equal(batched.truth_hist["satellite"], scalar.truth_hist["satellite"])
    np.testing.assert_array_equal(batched.t_s, scalar.t_s)
    assert 5 <= batched.satellite_stepper.passive_history.preparation_count <= 6
    assert batched.satellite_stepper.passive_history.consumed_samples == batched.current_index


def test_callable_equality_cannot_impersonate_builtin_j2(tmp_path):
    engine = _engine(_raw(tmp_path, duration=2.0, dt=1.0))
    calls = []

    class Force:
        def __eq__(self, other):
            return True

        def __call__(self, time_s, state, env, context):
            calls.append(time_s)
            return np.zeros(3)

    engine.agents["satellite"].dynamics.orbit_propagator.plugins = [Force()]
    # Definition uses identity; do not run the unrelated scalar propagator's
    # own force dispatch with this deliberately unusual comparison behavior.
    cache = engine.satellite_stepper.passive_history
    truth = engine.agents["satellite"].truth
    assert cache._definition(engine.agents["satellite"], truth) is None
    assert calls == []


def test_nonuniform_native_kernel_matches_direct_configured_rk4():
    widths = np.array([0.3, 0.3, 0.1, 0.3, 0.05], dtype="<f8")
    initial = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 1.0])
    raw = native.propagate_passive_sampled_eci_bytes(initial.tolist(), widths.tobytes(), [3, 5], 398600.4418, False)
    rows = np.frombuffer(raw, dtype="<f8").reshape(3, 6)
    scalar = OrbitPropagator(numeric_backend="rust")
    state, time_s = initial, 0.0
    for index, h in enumerate(widths):
        state = scalar.propagate(
            x_eci=state, t_s=time_s, dt_s=float(h), env={},
            ctx=OrbitContext(398600.4418, 100.0, 1.0, 2.2, 1.2),
            command_accel_eci_km_s2=np.zeros(3),
        )
        time_s += float(h)
        if index == 2:
            np.testing.assert_array_equal(rows[1], state)
    np.testing.assert_array_equal(rows[2], state)


@pytest.mark.skipif(not hasattr(native, "cr3bp_sampled_history_bytes"), reason="requires sampled CR3BP wheel")
def test_cr3bp_batch_retains_scalar_history_cadence_and_callbacks(tmp_path):
    raw = _raw(tmp_path, duration=1200.0, dt=60.0, substep=12.0)
    raw["objects"]["satellite"]["initial_state"] = {"cr3bp_halo": {"family": "l1_northern"}}
    raw["simulator"]["termination"] = {"earth_impact_enabled": False}
    raw["simulator"]["dynamics"]["orbit"].update({"model": "cr3bp", "cr3bp_system": "earth_moon"})
    callbacks = []
    batched = _finish(_engine(raw, callback=lambda step, _: callbacks.append(step)))
    scalar = _finish(_engine(raw, batch=False))
    np.testing.assert_array_equal(batched.truth_hist["satellite"], scalar.truth_hist["satellite"])
    np.testing.assert_array_equal(batched.t_s, scalar.t_s)
    assert callbacks == list(range(batched.current_index + 1))
    assert batched.satellite_stepper.passive_history.preparation_count == 1
    assert batched.satellite_stepper.passive_history.consumed_samples == batched.current_index


@pytest.mark.parametrize("deep_space", [False, True])
def test_ogp_batch_retains_exact_history_and_sample_callbacks(tmp_path, deep_space):
    from sim.tests.test_tle_initialization import (
        DEEP_SPACE_LINE1,
        DEEP_SPACE_LINE2,
        _sgp4_config,
    )

    raw = _sgp4_config(output_frame="teme")
    raw["objects"] = {key: raw.pop(key) for key in ("rocket", "chaser", "target")}
    raw["outputs"]["output_dir"] = str(tmp_path)
    raw["objects"]["target"]["general"]["numeric_backend"] = "rust"
    raw["simulator"].update({"duration_s": 1800.0, "dt_s": 60.0})
    if deep_space:
        raw["objects"]["target"]["initial_state"]["tle"] = {"line1": DEEP_SPACE_LINE1, "line2": DEEP_SPACE_LINE2}
    callbacks = []
    batched = _finish(_engine(raw, callback=lambda step, _: callbacks.append(step)))
    scalar = _finish(_engine(raw, batch=False))
    np.testing.assert_array_equal(batched.truth_hist["target"], scalar.truth_hist["target"])
    np.testing.assert_array_equal(batched.t_s, scalar.t_s)
    assert callbacks == list(range(batched.current_index + 1))
    assert batched.satellite_stepper.passive_history.preparation_count == 1
    assert batched.satellite_stepper.passive_history.consumed_samples == batched.current_index
