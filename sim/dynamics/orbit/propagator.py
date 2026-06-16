from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from sim.acceleration.kernels.orbit import rk4_zonal_step_state
from sim.acceleration.settings import acceleration_settings_from_mode
from sim.dynamics.orbit.accelerations import (
    OrbitContext,
    accel_drag,
    accel_j2,
    accel_j3,
    accel_j4,
    accel_lift,
    accel_srp,
    accel_third_body,
    accel_two_body,
)
from sim.dynamics.orbit.atmosphere import density_from_model
from sim.dynamics.orbit.cr3bp import cr3bp_system, propagate_cr3bp_state
from sim.dynamics.orbit.eclipse import resolve_srp_geometry, srp_shadow_factor
from sim.dynamics.orbit.environment import (
    EARTH_RADIUS_KM,
    JUPITER_MU_KM3_S2,
    MARS_MU_KM3_S2,
    MERCURY_MU_KM3_S2,
    MOON_MU_KM3_S2,
    NEPTUNE_MU_KM3_S2,
    PLUTO_MU_KM3_S2,
    SATURN_MU_KM3_S2,
    SUN_MU_KM3_S2,
    URANUS_MU_KM3_S2,
    VENUS_MU_KM3_S2,
)
from sim.dynamics.orbit.epoch import resolve_body_position_eci_km, resolve_sun_moon_positions
from sim.dynamics.orbit.integrators import (
    AdaptiveStepInfo,
    combine_adaptive_step_info,
    integrate_adaptive,
    rk4_step_state,
)
from sim.dynamics.orbit.spherical_harmonics import (
    accel_spherical_harmonics_terms,
    load_real_earth_gravity_terms,
    parse_spherical_harmonic_terms,
)

AccelerationPlugin = Callable[[float, np.ndarray, dict, OrbitContext], np.ndarray]
PLANETARY_MU_KM3_S2 = {
    "mercury": MERCURY_MU_KM3_S2,
    "venus": VENUS_MU_KM3_S2,
    "mars": MARS_MU_KM3_S2,
    "jupiter": JUPITER_MU_KM3_S2,
    "saturn": SATURN_MU_KM3_S2,
    "uranus": URANUS_MU_KM3_S2,
    "neptune": NEPTUNE_MU_KM3_S2,
    "pluto": PLUTO_MU_KM3_S2,
}
_ZERO3 = np.zeros(3, dtype=float)


def j2_plugin(t_s: float, x_eci: np.ndarray, env: dict, ctx: OrbitContext) -> np.ndarray:
    return accel_j2(x_eci[:3], ctx.mu_km3_s2)


def j3_plugin(t_s: float, x_eci: np.ndarray, env: dict, ctx: OrbitContext) -> np.ndarray:
    return accel_j3(x_eci[:3], ctx.mu_km3_s2)


def j4_plugin(t_s: float, x_eci: np.ndarray, env: dict, ctx: OrbitContext) -> np.ndarray:
    return accel_j4(x_eci[:3], ctx.mu_km3_s2)


def spherical_harmonics_plugin(t_s: float, x_eci: np.ndarray, env: dict, ctx: OrbitContext) -> np.ndarray:
    """
    Generic spherical-harmonics perturbation plugin.

    Expects one of:
    1) `env["spherical_harmonics_terms"]` as list[dict], each with:
    - n: degree
    - m: order
    - c_nm (or c): cosine coefficient
    - s_nm (or s): sine coefficient (optional)
    - normalized: whether coefficients are fully normalized (optional; default False)

    2) Real-coefficient mode:
    - spherical_harmonics_use_real_coefficients: bool (True)
    - spherical_harmonics_model: e.g., "EGM96" (optional; default EGM96)
    - spherical_harmonics_coeff_path: local .gfc path (optional)
    - spherical_harmonics_max_degree: int (optional; default 8)
    - spherical_harmonics_max_order: int (optional; default max_degree)
    - spherical_harmonics_allow_download: bool (optional; default True)

    Optional env fields:
    - spherical_harmonics_fd_step_km
    """
    terms = env.get("_parsed_spherical_harmonics_terms")
    if terms is None:
        terms = parse_spherical_harmonic_terms(env.get("spherical_harmonics_terms"))
        if terms:
            env["_parsed_spherical_harmonics_terms"] = terms
    if not terms and bool(env.get("spherical_harmonics_use_real_coefficients", False)):
        n_max = int(env.get("spherical_harmonics_max_degree", 8))
        m_max = int(env.get("spherical_harmonics_max_order", n_max))
        model = str(env.get("spherical_harmonics_model", "EGM96"))
        coeff_path = env.get("spherical_harmonics_coeff_path")
        allow_download = bool(env.get("spherical_harmonics_allow_download", True))
        cache_key = (
            n_max,
            m_max,
            model,
            None if coeff_path is None else str(coeff_path),
            allow_download,
        )
        cached_terms = env.get("_real_spherical_harmonics_cache")
        if cached_terms is None or cached_terms[0] != cache_key:
            terms = load_real_earth_gravity_terms(
                max_degree=n_max,
                max_order=m_max,
                model=model,
                coeff_path=None if coeff_path is None else str(coeff_path),
                allow_download=allow_download,
            )
            env["_real_spherical_harmonics_cache"] = (cache_key, terms)
        else:
            terms = cached_terms[1]
    if not terms:
        return np.zeros(3)
    fd_step_km = float(env.get("spherical_harmonics_fd_step_km", 1e-3))
    jd_utc_start = env.get("jd_utc_start")
    re_km = float(env.get("spherical_harmonics_reference_radius_km", EARTH_RADIUS_KM))
    frame_model = str(env.get("spherical_harmonics_frame_model", "simple"))
    eop_path = env.get("spherical_harmonics_eop_path")
    if jd_utc_start is None and "jd_utc" in env:
        jd_utc_start = float(env["jd_utc"]) - float(t_s) / 86400.0
    return accel_spherical_harmonics_terms(
        r_eci_km=x_eci[:3],
        t_s=t_s,
        terms=terms,
        mu_km3_s2=ctx.mu_km3_s2,
        re_km=re_km,
        fd_step_km=fd_step_km,
        jd_utc_start=None if jd_utc_start is None else float(jd_utc_start),
        frame_model=frame_model,
        eop_path=None if eop_path is None else str(eop_path),
    )


def drag_plugin(t_s: float, x_eci: np.ndarray, env: dict, ctx: OrbitContext) -> np.ndarray:
    density = env.get("density_kg_m3")
    if density is None:
        atmo_model = str(env.get("atmosphere_model", "exponential")).lower()
        density = density_from_model(
            atmo_model,
            x_eci[:3],
            t_s,
            env=env,
        )
    return accel_drag(
        x_eci[:3],
        x_eci[3:],
        t_s,
        ctx.mass_kg,
        ctx.area_m2,
        ctx.cd,
        {
            "density_kg_m3": density,
            "drag_area_m2": env.get("drag_area_m2", ctx.area_m2),
            "jd_utc_start": env.get("jd_utc_start"),
            "drag_frame_model": env.get("drag_frame_model", "simple"),
            "drag_eop_path": env.get("drag_eop_path"),
            "drag_earth_rotation_rad_s": env.get("drag_earth_rotation_rad_s"),
        },
    )


def lift_plugin(t_s: float, x_eci: np.ndarray, env: dict, ctx: OrbitContext) -> np.ndarray:
    cl = float(env.get("lift_coefficient", env.get("cl", 0.0)) or 0.0)
    lift_direction = env.get("lift_direction_eci")
    if lift_direction is None or cl == 0.0:
        return np.zeros(3)
    density = env.get("density_kg_m3")
    if density is None:
        atmo_model = str(env.get("atmosphere_model", "exponential")).lower()
        density = density_from_model(
            atmo_model,
            x_eci[:3],
            t_s,
            env=env,
        )
    return accel_lift(
        x_eci[:3],
        x_eci[3:],
        t_s,
        ctx.mass_kg,
        ctx.area_m2,
        cl,
        np.array(lift_direction, dtype=float).reshape(3),
        {
            "density_kg_m3": density,
            "lift_area_m2": env.get("lift_area_m2", env.get("drag_area_m2", ctx.area_m2)),
            "drag_earth_rotation_rad_s": env.get("drag_earth_rotation_rad_s"),
        },
    )


def srp_plugin(t_s: float, x_eci: np.ndarray, env: dict, ctx: OrbitContext) -> np.ndarray:
    srp_geometry = resolve_srp_geometry(x_eci[:3], t_s, env)
    return accel_srp(
        x_eci[:3],
        ctx.mass_kg,
        ctx.area_m2,
        ctx.cr,
        t_s,
        {
            "srp_geometry": srp_geometry,
            "srp_sun_dir_eci": srp_geometry["sun_dir_sc_eci"],
            "srp_distance_scale": srp_geometry["distance_scale"],
            "srp_shadow_factor": srp_shadow_factor(
                r_sc_eci_km=x_eci[:3],
                t_s=t_s,
                env=env,
                srp_geometry=srp_geometry,
            ),
            "srp_area_m2": env.get("srp_area_m2", ctx.area_m2),
            "srp_shadow_model": env.get("srp_shadow_model", "conical"),
        },
    )


def third_body_moon_plugin(t_s: float, x_eci: np.ndarray, env: dict, ctx: OrbitContext) -> np.ndarray:
    moon = env.get("moon_pos_eci_km")
    if moon is None:
        _, moon = resolve_sun_moon_positions(env, t_s)
    return accel_third_body(x_eci[:3], moon, MOON_MU_KM3_S2)


def third_body_sun_plugin(t_s: float, x_eci: np.ndarray, env: dict, ctx: OrbitContext) -> np.ndarray:
    sun = env.get("sun_pos_eci_km")
    if sun is None:
        sun, _ = resolve_sun_moon_positions(env, t_s)
    return accel_third_body(x_eci[:3], sun, SUN_MU_KM3_S2)


def third_body_planets_plugin(t_s: float, x_eci: np.ndarray, env: dict, ctx: OrbitContext) -> np.ndarray:
    selected = env.get("third_body_planets", [])
    if isinstance(selected, str):
        selected_names = [selected.strip().lower()]
    else:
        selected_names = [str(v).strip().lower() for v in selected]
    if any(v in ("all", "*") for v in selected_names):
        selected_names = list(PLANETARY_MU_KM3_S2.keys())

    acc = np.zeros(3)
    for name in selected_names:
        if name not in PLANETARY_MU_KM3_S2:
            continue
        pos = resolve_body_position_eci_km(name, env=env, t_s=t_s)
        mu = float(env.get(f"{name}_mu_km3_s2", PLANETARY_MU_KM3_S2[name]))
        acc += accel_third_body(x_eci[:3], pos, mu)
    return acc


@dataclass
class OrbitPropagator:
    model: str = "two_body"
    cr3bp_system_name: str = "earth_moon"
    integrator: str = "rk4"
    plugins: list[AccelerationPlugin] = field(default_factory=list)
    adaptive_atol: float = 1e-9
    adaptive_rtol: float = 1e-7
    acceleration_mode: str = "off"
    _rkf78_h_next: float | None = field(default=None, init=False, repr=False)
    _rkf78_last_t_s: float | None = field(default=None, init=False, repr=False)
    _acceleration_enabled_cache: bool | None = field(default=None, init=False, repr=False)
    _zonal_rk4_fast_path_checked: bool = field(default=False, init=False, repr=False)
    _zonal_rk4_fast_path_flags_cache: tuple[bool, bool, bool] | None = field(default=None, init=False, repr=False)
    last_adaptive_step_info: AdaptiveStepInfo | None = field(default=None, init=False, repr=False)
    adaptive_step_info: AdaptiveStepInfo | None = field(default=None, init=False, repr=False)

    def propagate(
        self,
        x_eci: np.ndarray,
        dt_s: float,
        t_s: float,
        command_accel_eci_km_s2: np.ndarray,
        env: dict,
        ctx: OrbitContext,
    ) -> np.ndarray:
        if str(self.model or "two_body").strip().lower() == "cr3bp":
            return propagate_cr3bp_state(
                x_eci,
                dt_s,
                t_s,
                command_accel_eci_km_s2,
                system=cr3bp_system(self.cr3bp_system_name),
            )

        fast_flags = self._zonal_rk4_fast_path_flags()
        if self._acceleration_enabled() and fast_flags is not None:
            include_j2, include_j3, include_j4 = fast_flags
            return rk4_zonal_step_state(
                np.asarray(x_eci, dtype=float).reshape(6),
                float(dt_s),
                np.asarray(command_accel_eci_km_s2, dtype=float).reshape(3),
                float(ctx.mu_km3_s2),
                include_j2,
                include_j3,
                include_j4,
            )

        def deriv(t_local: float, x_local: np.ndarray) -> np.ndarray:
            dx = np.empty(6, dtype=float)
            dx[:3] = x_local[3:]
            a = accel_two_body(x_local[:3], ctx.mu_km3_s2) + command_accel_eci_km_s2
            for plugin in self.plugins:
                a += plugin(t_local, x_local, env, ctx)
            dx[3:] = a
            return dx

        if self.integrator in ("rkf78", "dopri5", "adaptive"):
            adaptive_method = "rkf78" if self.integrator in ("rkf78", "adaptive") else "dopri5"
            if self._rkf78_last_t_s is None or float(t_s) < float(self._rkf78_last_t_s) - 1e-12:
                self._rkf78_h_next = None
            x_next, step_info = integrate_adaptive(
                deriv_fn=deriv,
                t_s=t_s,
                x=x_eci,
                dt_s=dt_s,
                atol=self.adaptive_atol,
                rtol=self.adaptive_rtol,
                method=adaptive_method,
                h_init=self._rkf78_h_next,
                return_info=True,
            )
            self._rkf78_h_next = step_info.suggested_next_step_s
            self._rkf78_last_t_s = float(t_s + dt_s)
            self.last_adaptive_step_info = step_info
            previous = [] if self.adaptive_step_info is None else [self.adaptive_step_info]
            self.adaptive_step_info = combine_adaptive_step_info(adaptive_method, [*previous, step_info])
            return x_next
        return rk4_step_state(deriv_fn=deriv, t_s=t_s, x=x_eci, dt_s=dt_s)

    def _acceleration_enabled(self) -> bool:
        if self._acceleration_enabled_cache is None:
            self._acceleration_enabled_cache = bool(acceleration_settings_from_mode(self.acceleration_mode).enabled)
        return bool(self._acceleration_enabled_cache)

    def _zonal_rk4_fast_path_flags(self) -> tuple[bool, bool, bool] | None:
        if self._zonal_rk4_fast_path_checked:
            return self._zonal_rk4_fast_path_flags_cache
        self._zonal_rk4_fast_path_checked = True
        if str(self.integrator).strip().lower() != "rk4":
            return None
        supported = {j2_plugin, j3_plugin, j4_plugin}
        if any(plugin not in supported for plugin in self.plugins):
            return None
        self._zonal_rk4_fast_path_flags_cache = (
            j2_plugin in self.plugins,
            j3_plugin in self.plugins,
            j4_plugin in self.plugins,
        )
        return self._zonal_rk4_fast_path_flags_cache
