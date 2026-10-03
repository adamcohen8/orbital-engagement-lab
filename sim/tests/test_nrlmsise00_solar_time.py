"""Solar geometry must follow the density frame and selected ephemeris."""
from datetime import datetime, timezone

import numpy as np
import pytest

from sim.acceleration.settings import acceleration_context
from sim.dynamics.orbit import atmosphere
from sim.dynamics.orbit.epoch import datetime_to_julian_date
from sim.dynamics.orbit.frames import FrameContext, eci_to_ecef_rotation_context
from sim.dynamics.orbit.nrlmsise00_backend import nrlmsise00_density


@pytest.fixture(autouse=True)
def reference_acceleration():
    with acceleration_context("off", allow_env_override=False):
        yield


def _geometry(year=2022, t_s=45.0, hour=12.0):
    epoch = datetime(year, 3, 31, tzinfo=timezone.utc)
    env = dict(jd_utc_start=datetime_to_julian_date(epoch), density_frame_model="iau76_80_eop",
               geodetic_model="wgs84", dut1_s=-0.1, xp_arcsec=0.15, yp_arcsec=-0.2, dat_s=37.0,
               f107=150.0, f107a=150.0, ap=4.0, nrlmsise00_ap_a=[4.0]*7)
    context = FrameContext(model="iau76_80_eop", jd_utc_start=env["jd_utc_start"],
                           dut1_s=-0.1, xp_arcsec=0.15, yp_arcsec=-0.2, dat_s=37.0)
    rotation = eci_to_ecef_rotation_context(t_s, context)
    angle = (hour-12.0)*np.pi/12.0
    position = rotation.T @ (6778.137*np.array([np.cos(angle), np.sin(angle), 0.0]))
    env["sun_pos_eci_km"] = rotation.T @ np.array([1.5e8, 0.0, 0.0])
    return position, atmosphere._datetime_from_env_t_s(env, t_s), env, rotation


@pytest.mark.parametrize("year", [2000, 2022, 2060])
@pytest.mark.parametrize("hour", [0.0, 6.0, 12.0, 18.0])
def test_density_tracks_solar_meridian_in_full_frame(year, hour):
    position, when, env, _rotation = _geometry(year=year, hour=hour)
    # These states are constructed at known solar hours in the terrestrial
    # frame. A J2000 right ascension mixed with sidereal time fails this check.
    expected = nrlmsise00_density(400.0, 0.0, (hour-12.0)*15.0, when, env, lst_hr=hour)
    with acceleration_context("off"):
        actual = atmosphere.density_nrlmsise00(position, 45.0, env)
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=0.0)


def test_sampled_sun_takes_precedence_and_coverage_fails_closed():
    position, when, env, _rotation = _geometry()
    sun = env["sun_pos_eci_km"].copy()
    env.update(sun_ephemeris_time_s=[0.0, 90.0], sun_ephemeris_eci_km=[sun, sun])
    env["sun_pos_eci_km"] *= -1.0
    expected = nrlmsise00_density(400.0, 0.0, 0.0, when, env, lst_hr=12.0)
    actual = atmosphere.density_nrlmsise00(position, 45.0, env)
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=0.0)
    with pytest.raises(ValueError, match="outside supplied coverage"):
        atmosphere.density_nrlmsise00(position, 91.0, env)


def test_ephemeris_callable_supplies_solar_geometry():
    position, when, env, _rotation = _geometry()
    sun = env.pop("sun_pos_eci_km")
    calls = []
    def ephemeris(jd, _env):
        calls.append(jd)
        return dict(sun_pos_eci_km=sun, moon_pos_eci_km=np.array([384400., 0., 0.]))
    env.update(ephemeris_mode="callable", ephemeris_callable=ephemeris)
    expected = nrlmsise00_density(400.0, 0.0, 0.0, when, env, lst_hr=12.0)
    actual = atmosphere.density_nrlmsise00(position, 45.0, env)
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=0.0)
    assert calls == [datetime_to_julian_date(when)]


def test_override_and_custom_density_do_not_resolve_unused_sun():
    position, when, env, _rotation = _geometry()
    def broken_ephemeris(*_args):
        raise AssertionError("unused ephemeris resolved")
    env.update(ephemeris_callable=broken_ephemeris)
    env.pop("sun_pos_eci_km")
    env["nrlmsise00_lst_hr"] = 7.0
    expected = nrlmsise00_density(400.0, 0.0, 0.0, when, env, lst_hr=7.0)
    np.testing.assert_allclose(atmosphere.density_nrlmsise00(position, 45.0, env),
                               expected, rtol=2e-13, atol=0.0)
    env.pop("nrlmsise00_lst_hr")
    env["nrlmsise00_density_callable"] = lambda *_args: 8e-12
    assert atmosphere.density_nrlmsise00(position, 45.0, env) == 8e-12


def test_density_keeps_utc_weather_day_at_negative_dut1_midnight(monkeypatch):
    position, when, env, _rotation = _geometry(t_s=0.0)
    seen = []
    def backend(_alt, _lat, _lon, dt, _env, *, lst_hr):
        seen.append((dt, lst_hr))
        return 1e-12
    monkeypatch.setattr(atmosphere, "_nrlmsise00_backend", lambda: backend)
    atmosphere.density_nrlmsise00(position, 0.0, env)
    assert seen[0][0] == when
    assert seen[0][0].timetuple().tm_yday == 90
    assert seen[0][1] == pytest.approx(12.0, abs=1e-12)


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_direct_geodetic_backend_default_uses_solar_time(backend):
    if backend == "rust":
        pytest.importorskip("oel_rust_orbit")
    _position, when, env, _rotation = _geometry(t_s=0.0)
    env["_rust_numeric_backend"] = backend
    expected = nrlmsise00_density(400.0, 0.0, 0.0, when, env, lst_hr=12.0)
    actual = nrlmsise00_density(400.0, 0.0, 0.0, when, env)
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=0.0)


@pytest.mark.parametrize("z_sign", [-1, 1])
@pytest.mark.parametrize("x_km", [0.0, 1e-7])
def test_exact_and_near_density_axis_use_finite_terrestrial_longitude(z_sign, x_km):
    _position, when, env, rotation = _geometry()
    position = rotation.T @ np.array([x_km, 0.0, z_sign*6900.0])
    actual = atmosphere.density_nrlmsise00(position, 45.0, env)
    assert np.isfinite(actual) and actual > 0.0
    # Once a terrestrial longitude is chosen at the degenerate axis, the
    # automatic and explicit solar hours must refer to that same longitude.
    alt, lat, lon = atmosphere._altitude_lat_lon_deg_from_eci(position, 45.0, env)
    expected = nrlmsise00_density(alt, lat, lon, when, env, lst_hr=(12.0+lon/15.0)%24.0)
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=0.0)


@pytest.mark.parametrize("integrator", ["rk4", "rkf78", "dopri5"])
@pytest.mark.parametrize("prepared", [False, True])
def test_rust_nrlmsise_history_matches_python(integrator, prepared, tmp_path):
    native = pytest.importorskip("oel_rust_orbit")
    if not hasattr(native, "ONPStageContext"):
        pytest.skip("requires prepared native stages")
    from sim.dynamics.orbit.accelerations import OrbitContext
    from sim.dynamics.orbit.propagator import OrbitPropagator, drag_plugin
    from sim.dynamics.orbit.rust_force_plan import make_plan
    eop = tmp_path / "eop.txt"
    eop.write_text("NUM_OBSERVED_POINTS 2\n"
                   "2022 03 31 59669 .1 .2 -.1 0 0 0 0 0 37\n"
                   "2022 04 01 59670 .2 .3 -.12 0 0 0 0 0 37\n")
    env = dict(jd_utc_start=2459669.5, atmosphere_model="nrlmsise00", geodetic_model="wgs84",
               density_frame_model="iau76_80_eop", drag_frame_model="iau76_80_eop",
               density_eop_path=str(eop), drag_eop_path=str(eop),
               f107=175., f107a=165., ap=12., nrlmsise00_ap_a=[12.]*7,
               sun_ephemeris_time_s=[0., 300.],
               sun_ephemeris_eci_km=[[1.5e8, 2e7, 1e6], [1.5e8, 2.1e7, 1e6]],
               _rust_stage_preparation_disabled=not prepared)
    context = OrbitContext(398600.4415, 300., 1., 2.2, 1.2)
    state0 = np.array([6778., 200., 50., -.2, 7.6, .8])
    histories = []
    for backend in ("python", "rust"):
        prop = OrbitPropagator(numeric_backend=backend, integrator=integrator,
                               plugins=[drag_plugin], acceleration_mode="off",
                               adaptive_atol=1e-12, adaptive_rtol=1e-12)
        if backend == "rust":
            stage = make_plan(prop, state0, 0., env, context)[5]
            assert isinstance(stage, native.ONPStageContext) == prepared
        state = state0.copy()
        history = [state.copy()]
        for index in range(12):
            state = prop.propagate(state, 10., index*10., np.zeros(3), env, context)
            history.append(state.copy())
        histories.append(np.array(history))
        if backend == "rust":
            assert prop.last_numeric_path == "rust_native_force_plan"
    np.testing.assert_allclose(histories[1], histories[0], rtol=0., atol=2e-11)
