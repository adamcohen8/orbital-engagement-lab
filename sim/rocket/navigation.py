from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.dynamics.orbit.atmosphere import atmosphere_state_from_model
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2, EARTH_RADIUS_KM
from sim.dynamics.orbit.frames import frame_context_from_environment, rotation_between, transform_position
from sim.rocket.models import RocketSimConfig, RocketState, RocketVehicleConfig
from sim.utils.geodesy import ecef_to_geodetic_deg_km, enu_to_ecef_rotation
from sim.utils.quaternion import quaternion_to_dcm_bn


def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= eps:
        return np.zeros_like(v)
    return v / n


def _orbital_elements_basic(
    r_km: np.ndarray, v_km_s: np.ndarray, mu_km3_s2: float = EARTH_MU_KM3_S2
) -> tuple[float, float]:
    r = float(np.linalg.norm(r_km))
    v2 = float(np.dot(v_km_s, v_km_s))
    if r <= 0.0:
        return np.inf, np.inf
    eps = 0.5 * v2 - mu_km3_s2 / r
    a_km = np.inf if abs(eps) < 1e-14 else float(-mu_km3_s2 / (2.0 * eps))
    h = np.cross(r_km, v_km_s)
    e_vec = np.cross(v_km_s, h) / mu_km3_s2 - r_km / r
    return a_km, float(np.linalg.norm(e_vec))


def _apo_peri_alt_km(
    r_km: np.ndarray, v_km_s: np.ndarray, mu_km3_s2: float = EARTH_MU_KM3_S2
) -> tuple[float, float, float, float]:
    a_km, e = _orbital_elements_basic(r_km, v_km_s, mu_km3_s2)
    if not np.isfinite(a_km) or a_km <= 0.0:
        return np.inf, -np.inf, a_km, e
    ra_km = float(a_km * (1.0 + e))
    rp_km = float(a_km * (1.0 - e))
    return ra_km - EARTH_RADIUS_KM, rp_km - EARTH_RADIUS_KM, a_km, e


def _geodetic_state_from_eci(
    r_eci_km: np.ndarray,
    t_s: float,
    jd_utc_start: float | None = None,
    frame_context=None,
) -> tuple[float, float, float]:
    frame_context = frame_context or frame_context_from_environment({"jd_utc_start": jd_utc_start})
    r_ecef = transform_position(
        np.array(r_eci_km, dtype=float).reshape(3),
        "eci",
        "ecef",
        t_s=float(t_s),
        context=frame_context,
    )
    return ecef_to_geodetic_deg_km(r_ecef)


def _resolve_wind_eci_m_s(
    *,
    position_eci_km: np.ndarray,
    t_s: float,
    sim_cfg: RocketSimConfig,
    state: RocketState | None = None,
) -> np.ndarray:
    env = dict(getattr(sim_cfg, "atmosphere_env", {}) or {})
    jd_utc_start = env.get("jd_utc_start")
    frame_context = frame_context_from_environment(env)
    lat_deg, lon_deg, alt_km = _geodetic_state_from_eci(
        position_eci_km,
        t_s,
        jd_utc_start=jd_utc_start,
        frame_context=frame_context,
    )
    wind_enu = np.array(sim_cfg.wind_enu_m_s, dtype=float).reshape(3)
    wind_cb = sim_cfg.wind_enu_callable
    if callable(wind_cb):
        wind_enu = wind_enu + np.array(wind_cb(alt_km, lat_deg, lon_deg, t_s, state, sim_cfg), dtype=float).reshape(3)
    wind_ecef = enu_to_ecef_rotation(lat_deg, lon_deg) @ wind_enu
    wind_eci = rotation_between("ecef", "eci", t_s=float(t_s), context=frame_context) @ (wind_ecef / 1e3)
    return wind_eci * 1e3


@dataclass(frozen=True)
class RocketNavState:
    t_s: float
    latitude_deg: float
    longitude_deg: float
    altitude_km: float
    radius_km: float
    speed_km_s: float
    vertical_speed_km_s: float
    horizontal_speed_km_s: float
    flight_path_angle_deg: float
    apoapsis_alt_km: float
    periapsis_alt_km: float
    sma_km: float
    eccentricity: float
    dynamic_pressure_pa: float
    mach: float
    alpha_deg: float
    beta_deg: float
    qbar_times_alpha_pa_deg: float
    thrust_to_weight: float
    propellant_remaining_kg: float
    propellant_remaining_fraction: float
    active_stage_index: int
    stages_complete: bool
    thrust_axis_eci: np.ndarray
    thrust_axis_body: np.ndarray
    relative_wind_body_m_s: np.ndarray


def build_rocket_nav_state(
    state: RocketState,
    sim_cfg: RocketSimConfig,
    vehicle_cfg: RocketVehicleConfig,
    *,
    throttle_cmd: float = 0.0,
    thrust_n: float | None = None,
) -> RocketNavState:
    r = np.array(state.position_eci_km, dtype=float).reshape(3)
    v = np.array(state.velocity_eci_km_s, dtype=float).reshape(3)
    r_norm = float(np.linalg.norm(r))
    speed = float(np.linalg.norm(v))
    r_hat = _unit(r)
    vertical_speed = float(np.dot(v, r_hat)) if r_norm > 0.0 else 0.0
    horizontal_speed = float(max(speed * speed - vertical_speed * vertical_speed, 0.0) ** 0.5)
    fpa = float(np.rad2deg(np.arctan2(vertical_speed, max(horizontal_speed, 1e-12))))
    jd_utc_start = dict(getattr(sim_cfg, "atmosphere_env", {}) or {}).get("jd_utc_start")
    lat_deg, lon_deg, alt_geo_km = _geodetic_state_from_eci(r, state.t_s, jd_utc_start=jd_utc_start)
    alt_km = float(alt_geo_km if sim_cfg.use_wgs84_geodesy else r_norm - EARTH_RADIUS_KM)
    apo_alt, peri_alt, sma_km, ecc = _apo_peri_alt_km(r, v, EARTH_MU_KM3_S2)

    env = {"atmosphere_model": sim_cfg.atmosphere_model, **dict(sim_cfg.atmosphere_env)}
    if sim_cfg.use_wgs84_geodesy:
        env["geodetic_model"] = "wgs84"
    atmos = atmosphere_state_from_model(
        model=str(sim_cfg.atmosphere_model).lower(),
        r_eci_km=r,
        t_s=state.t_s,
        env=env,
    )
    c_bn = quaternion_to_dcm_bn(state.attitude_quat_bn)
    omega_earth = np.array([0.0, 0.0, 7.2921159e-5], dtype=float)
    v_atm_eci_km_s = np.cross(omega_earth, r)
    wind_eci_m_s = _resolve_wind_eci_m_s(position_eci_km=r, t_s=state.t_s, sim_cfg=sim_cfg, state=state)
    v_rel_eci_m_s = (v - v_atm_eci_km_s) * 1e3 - wind_eci_m_s
    v_rel_body_m_s = c_bn @ v_rel_eci_m_s
    rel_speed_m_s = float(np.linalg.norm(v_rel_body_m_s))
    q_dyn = 0.5 * float(max(atmos["density_kg_m3"], 0.0)) * rel_speed_m_s * rel_speed_m_s
    sound_speed = float(max(atmos["sound_speed_m_s"], 1e-6))
    mach = rel_speed_m_s / sound_speed
    u, v_lat, w = float(v_rel_body_m_s[0]), float(v_rel_body_m_s[1]), float(v_rel_body_m_s[2])
    if rel_speed_m_s <= 1e-12:
        alpha_deg = 0.0
        beta_deg = 0.0
    else:
        alpha_deg = float(np.rad2deg(np.arctan2(w, max(u, 1e-12))))
        beta_deg = float(np.rad2deg(np.arcsin(np.clip(v_lat / rel_speed_m_s, -1.0, 1.0))))

    stage_prop = np.array(state.stage_prop_remaining_kg, dtype=float).reshape(-1)
    prop_remaining = float(np.sum(stage_prop))
    prop0 = sum(float(stage.propellant_mass_kg) for stage in vehicle_cfg.stack.stages)
    active_stage_index = int(state.active_stage_index)
    stages_complete = active_stage_index >= len(vehicle_cfg.stack.stages)
    thrust_axis_body = _unit(np.array(state.thrust_vector_body, dtype=float).reshape(3))
    thrust_axis_eci = c_bn.T @ thrust_axis_body
    if thrust_n is None:
        thrust_n = 0.0
        if active_stage_index < len(vehicle_cfg.stack.stages):
            stage = vehicle_cfg.stack.stages[active_stage_index]
            thrust_n = float(np.clip(throttle_cmd, 0.0, 1.0)) * float(stage.max_thrust_n)
    weight_n = float(max(state.mass_kg, 0.0)) * 9.80665

    return RocketNavState(
        t_s=float(state.t_s),
        latitude_deg=float(lat_deg),
        longitude_deg=float(lon_deg),
        altitude_km=alt_km,
        radius_km=r_norm,
        speed_km_s=speed,
        vertical_speed_km_s=vertical_speed,
        horizontal_speed_km_s=horizontal_speed,
        flight_path_angle_deg=fpa,
        apoapsis_alt_km=float(apo_alt),
        periapsis_alt_km=float(peri_alt),
        sma_km=float(sma_km),
        eccentricity=float(ecc),
        dynamic_pressure_pa=float(max(q_dyn, 0.0)),
        mach=float(max(mach, 0.0)),
        alpha_deg=alpha_deg,
        beta_deg=beta_deg,
        qbar_times_alpha_pa_deg=float(max(q_dyn, 0.0) * abs(alpha_deg)),
        thrust_to_weight=float(0.0 if weight_n <= 0.0 else max(float(thrust_n), 0.0) / weight_n),
        propellant_remaining_kg=prop_remaining,
        propellant_remaining_fraction=float(0.0 if prop0 <= 0.0 else np.clip(prop_remaining / prop0, 0.0, 1.0)),
        active_stage_index=active_stage_index,
        stages_complete=stages_complete,
        thrust_axis_eci=thrust_axis_eci,
        thrust_axis_body=thrust_axis_body,
        relative_wind_body_m_s=v_rel_body_m_s,
    )
