from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from sim.acceleration.settings import acceleration_context
from sim.dynamics.orbit.accelerations import OrbitContext
from sim.dynamics.orbit.propagator import (
    OrbitPropagator,
    drag_plugin,
    j2_plugin,
    j3_plugin,
    j4_plugin,
    lift_plugin,
    spherical_harmonics_plugin,
    srp_plugin,
    third_body_moon_plugin,
    third_body_planets_plugin,
    third_body_sun_plugin,
)
from sim.dynamics.orbit.spherical_harmonics import (
    SphericalHarmonicTerm,
    compile_spherical_harmonic_terms,
)


def _case() -> tuple[np.ndarray, dict, OrbitContext]:
    terms = [
        SphericalHarmonicTerm(
            n=2,
            m=0,
            c_nm=-4.841693259705e-4,
            normalized=True,
        )
    ]
    env = {
        "_compiled_spherical_harmonics_terms": compile_spherical_harmonic_terms(terms),
        "_parsed_spherical_harmonics_terms": terms,
        "spherical_harmonics_terms": terms,
        "spherical_harmonics_reference_radius_km": 6378.1363,
        "spherical_harmonics_frame_model": "iau76_80_eop",
        "density_frame_model": "iau76_80_eop",
        "drag_frame_model": "iau76_80_eop",
        "geodetic_model": "wgs84",
        "atmosphere_model": "nrlmsise00",
        "jd_utc_start": 2459669.5,
        "dut1_s": 0.1,
        "xp_arcsec": 0.05,
        "yp_arcsec": -0.03,
        "dat_s": 37.0,
        "tt_minus_utc_s": 69.184,
        "f107": 150.0,
        "f107a": 150.0,
        "ap": 4.0,
        "nrlmsise00_ap_a": [4.0] * 7,
        "srp_shadow_model": "conical",
        "sun_pos_eci_km": np.array([1.49e8, 2.0e6, -1.0e6], dtype=float),
        "moon_pos_eci_km": np.array([3.7e5, -8.0e4, 3.0e4], dtype=float),
    }
    state = np.array([700.0, 4400.0, 5100.0, -7.3, -1.1, 1.9], dtype=float)
    context = OrbitContext(mu_km3_s2=398600.4415, mass_kg=300.0, area_m2=1.0, cd=2.2, cr=1.2)
    return state, env, context


def _assert_compiled_steps_are_exact(plugins: list, step_count: int = 12) -> None:
    state, env, context = _case()
    command = np.zeros(3, dtype=float)
    with acceleration_context("auto", allow_env_override=False):
        reference = OrbitPropagator(integrator="rk4", plugins=plugins, acceleration_mode="auto")
        assert reference._builtin_rk4_fast_path_enabled()
        accelerated = OrbitPropagator(integrator="rk4", plugins=plugins, acceleration_mode="auto")
        for step in range(step_count):
            t_s = float(step * 10)
            expected = reference._propagate_builtin_rk4(
                x_eci=state,
                dt_s=10.0,
                t_s=t_s,
                command_accel_eci_km_s2=command,
                env=env,
                ctx=context,
            )
            actual = accelerated.propagate(state, 10.0, t_s, command, env, context)
            assert np.array_equal(actual, expected)
            state = expected


def test_compiled_force_plan_is_exact_for_drag_and_full_force_subsets() -> None:
    _assert_compiled_steps_are_exact([spherical_harmonics_plugin, drag_plugin])
    _assert_compiled_steps_are_exact(
        [
            spherical_harmonics_plugin,
            drag_plugin,
            srp_plugin,
            third_body_sun_plugin,
            third_body_moon_plugin,
        ]
    )


def test_compiled_force_plan_falls_back_for_disturbed_ap_and_lower_atmosphere() -> None:
    state, env, context = _case()
    command = np.zeros(3, dtype=float)
    with acceleration_context("auto", allow_env_override=False):
        disturbed_env = dict(env)
        disturbed_env["nrlmsise00_ap_a"] = [4.0, 12.0, 4.0, 4.0, 4.0, 4.0, 4.0]
        propagator = OrbitPropagator(
            integrator="rk4",
            plugins=[spherical_harmonics_plugin, drag_plugin],
            acceleration_mode="auto",
        )
        assert (
            propagator._try_propagate_compiled_builtin_rk4(
                x_eci=state,
                dt_s=10.0,
                t_s=0.0,
                command_accel_eci_km_s2=command,
                env=disturbed_env,
                ctx=context,
            )
            is None
        )

        low_state = state.copy()
        low_state[:3] *= 6550.0 / np.linalg.norm(low_state[:3])
        assert (
            propagator._try_propagate_compiled_builtin_rk4(
                x_eci=low_state,
                dt_s=10.0,
                t_s=0.0,
                command_accel_eci_km_s2=command,
                env=env,
                ctx=context,
            )
            is None
        )


def _custom_acceleration(t_s: float, x_eci: np.ndarray, env: dict, context: OrbitContext) -> np.ndarray:
    del t_s, context
    return np.array([float(env["custom_acceleration_scale"]) * x_eci[0], -2.0e-12, 3.0e-12])


def _assert_staged_propagation_is_exact(
    *,
    integrator: str,
    plugins: list,
    env_updates: dict,
    dt_s: float = 10.0,
) -> None:
    state, env, context = _case()
    env = {**env, **env_updates}
    command = np.array([1.0e-9, -2.0e-9, 3.0e-9], dtype=float)
    kwargs = {
        "integrator": integrator,
        "plugins": plugins,
        "adaptive_atol": 1.0e-12,
        "adaptive_rtol": 1.0e-12,
    }
    with acceleration_context("off", allow_env_override=False):
        reference = OrbitPropagator(acceleration_mode="off", **kwargs)
        expected = reference.propagate(state, dt_s, 12.3, command, env, context)
    with acceleration_context("auto", allow_env_override=False):
        accelerated = OrbitPropagator(acceleration_mode="auto", **kwargs)
        actual = accelerated.propagate(state, dt_s, 12.3, command, env, context)

    assert np.array_equal(actual, expected)
    if integrator == "rkf78":
        assert accelerated.last_adaptive_step_info == reference.last_adaptive_step_info


def test_staged_compiled_path_is_exact_for_remaining_forces_and_custom_plugin() -> None:
    _assert_staged_propagation_is_exact(
        integrator="rk4",
        plugins=[
            j2_plugin,
            j3_plugin,
            j4_plugin,
            drag_plugin,
            lift_plugin,
            srp_plugin,
            third_body_sun_plugin,
            third_body_moon_plugin,
            third_body_planets_plugin,
            _custom_acceleration,
        ],
        env_updates={
            "density_kg_m3": 1.2e-12,
            "lift_coefficient": 0.3,
            "lift_direction_eci": [0.2, 0.8, -0.1],
            "third_body_planets": ["venus"] * 10 + ["mars"],
            "venus_pos_eci_km": np.array([5.0e7, 1.0e8, 2.0e7]),
            "mars_pos_eci_km": np.array([-7.0e7, 1.5e8, -3.0e7]),
            "custom_acceleration_scale": 1.0e-15,
        },
    )


def test_staged_compiled_path_preserves_rkf78_state_and_step_control_exactly() -> None:
    _assert_staged_propagation_is_exact(
        integrator="rkf78",
        plugins=[j2_plugin, drag_plugin, lift_plugin, third_body_planets_plugin, _custom_acceleration],
        env_updates={
            "density_kg_m3": 1.2e-12,
            "lift_coefficient": 0.3,
            "lift_direction_eci": [0.2, 0.8, -0.1],
            "third_body_planets": ["venus"],
            "venus_pos_eci_km": np.array([5.0e7, 1.0e8, 2.0e7]),
            "custom_acceleration_scale": 1.0e-15,
        },
    )


def test_staged_compiled_drag_is_exact_for_every_atmosphere_family() -> None:
    def density_callable(alt_km, lat_deg, lon_deg, dt_utc, env):
        del alt_km, lat_deg, lon_deg, dt_utc, env
        return 7.5e-12

    common = {
        "atmo_epoch_utc": datetime(2024, 3, 20, 12, tzinfo=timezone.utc),
        "geodetic_model": "wgs84",
        "drag_frame_model": "simple",
        "f107": 150.0,
        "f107a": 150.0,
        "ap": 4.0,
        "jacchia70_f10": 150.0,
        "jacchia70_f10b": 150.0,
        "jacchia70_ap": 4.0,
    }
    cases = [
        ("exponential", {}),
        ("ussa1976", {}),
        ("nrlmsise00", {"nrlmsise00_density_callable": density_callable}),
        ("msis86", {"msis86_density_callable": density_callable}),
        ("jacchia70", {"jacchia70_density_callable": density_callable}),
        ("jb2006", {"jb2006_density_callable": density_callable}),
        ("jb2008", {"jb2008_density_callable": density_callable}),
        ("harris_priester", {}),
    ]
    for model, model_env in cases:
        _assert_staged_propagation_is_exact(
            integrator="rk4",
            plugins=[j2_plugin, j3_plugin, j4_plugin, drag_plugin],
            env_updates={**common, **model_env, "atmosphere_model": model},
        )


def test_staged_compiled_density_callable_receives_original_environment() -> None:
    def density_callable(alt_km, lat_deg, lon_deg, dt_utc, env):
        del alt_km, lat_deg, lon_deg, dt_utc
        return 2.0e-12 if "jd_utc" in env else 1.0e-12

    _assert_staged_propagation_is_exact(
        integrator="rk4",
        plugins=[j2_plugin, j3_plugin, drag_plugin, srp_plugin, _custom_acceleration],
        env_updates={
            "nrlmsise00_density_callable": density_callable,
            "custom_acceleration_scale": 1.0e-15,
        },
        dt_s=10.0,
    )
