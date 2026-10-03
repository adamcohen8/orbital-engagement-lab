"""Exact stage-frame reuse, invalidation, and adaptive evidence contracts."""

from dataclasses import asdict
from unittest.mock import patch

import numpy as np
import pytest

from sim.aero import core
from sim.dynamics.orbit.accelerations import OrbitContext
from sim.dynamics.orbit.propagator import OrbitPropagator, drag_plugin

R = np.array([7000.0, 300.0, 900.0])
V = np.array([0.1, 7.4, 0.8])


@pytest.fixture
def eop_path(tmp_path):
    # Synthetic constant EOP records keep this test independent of validation assets.
    path = tmp_path / "eop.txt"
    path.write_text(
        "NUM_OBSERVED_POINTS 2\n"
        "2022 03 31 59669 0.100000 0.200000 -0.100000 0 0 0 0 0 37\n"
        "2022 04 01 59670 0.100000 0.200000 -0.100000 0 0 0 0 0 37\n"
    )
    return path


@pytest.mark.parametrize("frame_model", ["simple", "inertial_z", "iau76_80_eop"])
def test_frame_reuse_recomputes_each_state_exactly(frame_model, eop_path):
    options = dict(frame_model=frame_model, jd_utc_start=2459669.5)
    if frame_model == "iau76_80_eop":
        options["eop_path"] = str(eop_path)
    states = [(R, V), (R * 1.02, V * 1.01), (R * 0.99, -V)]
    expected = [core.atmosphere_relative_velocity_eci_km_s(r, v, t_s=120.0, **options) for r, v in states]
    with patch.object(
        core, "eci_to_ecef_rotation_derivative_context", wraps=core.eci_to_ecef_rotation_derivative_context
    ) as derivative:
        cache = {}
        actual = [
            core.atmosphere_relative_velocity_eci_km_s(r, v, t_s=120.0, _frame_cache=cache, **options)
            for r, v in states
        ]
        assert derivative.call_count == (1 if frame_model == "iau76_80_eop" else 0)
    for a, b in zip(expected, actual):
        np.testing.assert_array_equal(a, b)
    assert len(cache) == (1 if frame_model == "iau76_80_eop" else 0)


@pytest.mark.parametrize(
    "parameter,value",
    [
        ("jd_utc_start", 2459669.5001),
        ("dut1_s", 0.3),
        ("xp_arcsec", 0.1),
        ("yp_arcsec", 0.2),
        ("dat_s", 37.0),
        ("tt_minus_utc_s", 70.184),
        ("ddpsi_rad", 1e-6),
        ("ddeps_rad", 2e-6),
    ],
)
def test_manual_frame_parameter_changes_invalidate_reuse(parameter, value):
    options = dict(frame_model="iau76_80_eop", jd_utc_start=2459669.5)
    cache = {}
    core.atmosphere_relative_velocity_eci_km_s(R, V, t_s=120.0, _frame_cache=cache, **options)
    options[parameter] = value
    expected = core.atmosphere_relative_velocity_eci_km_s(R, V, t_s=120.0, **options)
    actual = core.atmosphere_relative_velocity_eci_km_s(R, V, t_s=120.0, _frame_cache=cache, **options)
    np.testing.assert_array_equal(actual, expected)
    assert len(cache) == 2


def test_eop_rewrite_invalidates_reuse_and_missing_file_fails(eop_path):
    path = eop_path
    options = dict(frame_model="iau76_80_eop", jd_utc_start=2459669.5, eop_path=str(path))
    cache = {}
    original = core.atmosphere_relative_velocity_eci_km_s(R, V, t_s=120.0, _frame_cache=cache, **options)
    path.write_text(path.read_text().replace("0.100000", "1.100000"))
    expected = core.atmosphere_relative_velocity_eci_km_s(R, V, t_s=120.0, **options)
    actual = core.atmosphere_relative_velocity_eci_km_s(R, V, t_s=120.0, _frame_cache=cache, **options)
    np.testing.assert_array_equal(actual, expected)
    assert not np.array_equal(original, actual)
    assert len(cache) == 2
    path.unlink()
    with pytest.raises(FileNotFoundError):
        core.atmosphere_relative_velocity_eci_km_s(R, V, t_s=120.0, _frame_cache=cache, **options)


def test_frame_cache_bound_and_backward_queries(eop_path):
    options = dict(frame_model="iau76_80_eop", jd_utc_start=2459669.5, eop_path=str(eop_path))
    cache = {}
    for t in [*range(120, 190), 121, 120, 140]:
        expected = core.atmosphere_relative_velocity_eci_km_s(R, V, t_s=t, **options)
        actual = core.atmosphere_relative_velocity_eci_km_s(R, V, t_s=t, _frame_cache=cache, **options)
        np.testing.assert_array_equal(actual, expected)
        assert len(cache) <= 64


@pytest.mark.parametrize("integrator", ["rk4", "rkf78", "dopri5"])
def test_stage_reuse_preserves_adaptive_evidence(integrator, eop_path):
    pytest.importorskip("oel_rust_orbit")
    import sim.dynamics.orbit.rust_force_plan as adapter

    x = np.r_[R, V]
    context = OrbitContext(398600.4415, 300.0, 1.0, 2.2, 1.2)
    env = dict(
        density_kg_m3=1e-12, drag_frame_model="iau76_80_eop", drag_eop_path=str(eop_path), jd_utc_start=2459669.5
    )
    # Use a fresh cache for every derivative to construct the reference route.
    authoritative = core.atmosphere_relative_velocity_eci_km_s

    def no_cache(*args, **kwargs):
        kwargs.pop("_frame_cache", None)
        return authoritative(*args, **kwargs)

    results = []
    for enabled in (False, True):
        prop = OrbitPropagator(
            integrator=integrator,
            plugins=[drag_plugin],
            numeric_backend="rust",
            acceleration_mode="off",
            adaptive_atol=1e-12,
            adaptive_rtol=1e-10,
        )
        state = x.copy()
        history = []
        with patch.object(adapter, "atmosphere_relative_velocity_eci_km_s", authoritative if enabled else no_cache):
            for index in range(12):
                state = prop.propagate(state, 20.0, index * 20.0, np.zeros(3), dict(env), context)
                history.append(state.copy())
        results.append(
            (np.asarray(history), None if prop.adaptive_step_info is None else asdict(prop.adaptive_step_info))
        )
    np.testing.assert_array_equal(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]
