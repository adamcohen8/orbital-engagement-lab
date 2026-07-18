from __future__ import annotations

from typing import Any, Callable

import numpy as np

from sim.aero import aero_spec_get
from sim.config import SimulationScenarioConfig, iter_object_sections
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2, EARTH_RADIUS_KM
from sim.presets.thrusters import resolve_thruster_mount_from_specs
from sim.utils.quaternion import quaternion_to_dcm_bn


def _quat_error_angle_deg(q_des: np.ndarray, q_cur: np.ndarray) -> float:
    qd = np.array(q_des, dtype=float).reshape(-1)
    qc = np.array(q_cur, dtype=float).reshape(-1)
    if qd.size != 4 or qc.size != 4:
        return float("nan")
    nd = float(np.linalg.norm(qd))
    nc = float(np.linalg.norm(qc))
    if nd <= 0.0 or nc <= 0.0:
        return float("nan")
    qd /= nd
    qc /= nc
    dot = float(np.clip(np.dot(qd, qc), -1.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(abs(dot))))


def _orbital_elements_basic(
    r_km: np.ndarray,
    v_km_s: np.ndarray,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
) -> tuple[float, float]:
    r = float(np.linalg.norm(r_km))
    v2 = float(np.dot(v_km_s, v_km_s))
    if r <= 0.0:
        return np.inf, np.inf
    eps = 0.5 * v2 - mu_km3_s2 / r
    a = np.inf if abs(eps) < 1e-14 else float(-mu_km3_s2 / (2.0 * eps))
    h = np.cross(r_km, v_km_s)
    e_vec = np.cross(v_km_s, h) / mu_km3_s2 - r_km / r
    e = float(np.linalg.norm(e_vec))
    return a, e


def _rocket_metric_array(
    rocket_metrics: dict[str, np.ndarray] | None,
    name: str,
    size: int,
    default: float = np.nan,
) -> np.ndarray:
    out = np.full(size, default, dtype=float)
    if rocket_metrics is None or name not in rocket_metrics:
        return out
    arr = np.array(rocket_metrics[name], dtype=float).reshape(-1)
    n = min(size, arr.size)
    if n > 0:
        out[:n] = arr[:n]
    return out


def _last_finite_value(series: np.ndarray) -> float:
    arr = np.array(series, dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    return float(finite[-1]) if finite.size else float("nan")


def _max_abs_finite_value(series: np.ndarray) -> float:
    arr = np.array(series, dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    return float(np.max(np.abs(finite))) if finite.size else float("nan")


def _max_finite_value(series: np.ndarray) -> float:
    arr = np.array(series, dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    return float(np.max(finite)) if finite.size else float("nan")


def _first_true_time(t_s: np.ndarray, mask: np.ndarray) -> float | None:
    idx = np.flatnonzero(np.array(mask, dtype=bool))
    if idx.size == 0:
        return None
    i = int(idx[0])
    if i < 0 or i >= t_s.size:
        return None
    return float(t_s[i])


def _rocket_launch_site(cfg: SimulationScenarioConfig) -> tuple[float, float] | None:
    initial_state = dict(getattr(cfg.rocket, "initial_state", {}) or {})
    try:
        return float(initial_state["launch_lat_deg"]), float(initial_state["launch_lon_deg"])
    except (KeyError, TypeError, ValueError):
        return None


def _haversine_distance_km(lat0_deg: float, lon0_deg: float, lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    lat0 = np.deg2rad(float(lat0_deg))
    lon0 = np.deg2rad(float(lon0_deg))
    lat = np.deg2rad(np.array(lat_deg, dtype=float))
    lon = np.deg2rad(np.array(lon_deg, dtype=float))
    dlat = lat - lat0
    dlon = lon - lon0
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat0) * np.cos(lat) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(np.clip(a, 0.0, 1.0)), np.sqrt(np.clip(1.0 - a, 0.0, 1.0)))
    return EARTH_RADIUS_KM * c


def _rocket_target_altitude_cfg(cfg: SimulationScenarioConfig) -> tuple[float, float, float]:
    dyn = dict(getattr(cfg.simulator.dynamics, "rocket", {}) or {})
    target = float(dyn.get("target_altitude_km", np.nan))
    tol = float(dyn.get("target_altitude_tolerance_km", np.nan))
    ecc_max = float(dyn.get("target_eccentricity_max", np.nan))
    return target, tol, ecc_max


def _compute_satellite_delta_v_remaining(
    *,
    cfg: SimulationScenarioConfig,
    truth_hist: dict[str, np.ndarray],
    resolve_satellite_isp_s: Callable[[dict[str, Any]], float],
) -> dict[str, dict[str, Any]]:
    g0_m_s2 = 9.80665
    out: dict[str, dict[str, Any]] = {}
    for oid, sec in iter_object_sections(cfg, enabled_only=True, kind="satellite"):
        hist = truth_hist.get(oid)
        if hist is None or sec is None or hist.shape[0] == 0:
            continue
        specs = dict(getattr(sec, "specs", {}) or {})
        dry_mass_kg = float(specs.get("dry_mass_kg", np.nan))
        fuel_mass_kg = float(specs.get("fuel_mass_kg", np.nan))
        if not (np.isfinite(dry_mass_kg) and np.isfinite(fuel_mass_kg)):
            continue
        if dry_mass_kg <= 0.0 or fuel_mass_kg < 0.0:
            continue
        m0 = dry_mass_kg + fuel_mass_kg
        if m0 <= dry_mass_kg:
            continue
        isp_s = resolve_satellite_isp_s(specs)
        if isp_s <= 0.0:
            continue
        dv0_m_s = float(isp_s * g0_m_s2 * np.log(m0 / dry_mass_kg))
        if dv0_m_s <= 0.0:
            continue
        m_hist = np.clip(np.array(hist[:, 13], dtype=float), dry_mass_kg, m0)
        dv_rem_m_s = isp_s * g0_m_s2 * np.log(m_hist / dry_mass_kg)
        out[oid] = {
            "initial_m_s": dv0_m_s,
            "remaining_m_s": dv_rem_m_s,
        }
    return out


def _thruster_mounts_by_object(cfg: SimulationScenarioConfig) -> dict[str, dict[str, np.ndarray] | None]:
    out: dict[str, dict[str, np.ndarray] | None] = {}
    for oid, sec in iter_object_sections(cfg, enabled_only=True, kind="satellite"):
        mount = resolve_thruster_mount_from_specs(getattr(sec, "specs", None) if sec is not None else None)
        if mount is None:
            out[oid] = None
            continue
        out[oid] = {
            "position_body_m": np.array(mount.position_body_m, dtype=float),
            "direction_body": np.array(mount.thrust_direction_body, dtype=float),
        }
    return out


def _unit_vector_or_none(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.array(value, dtype=float).reshape(3)
    except (TypeError, ValueError):
        return None
    n = float(np.linalg.norm(arr))
    if not np.isfinite(n) or n <= 0.0:
        return None
    return arr / n


def _thruster_direction_body_by_object(cfg: SimulationScenarioConfig) -> dict[str, np.ndarray]:
    plot_default = _unit_vector_or_none(cfg.outputs.plots.get("thrust_direction_body"))
    out: dict[str, np.ndarray] = {}
    for oid, sec in iter_object_sections(cfg, enabled_only=True, kind="satellite"):
        direction = None
        mission_execution = getattr(sec, "mission_execution", None)
        params = dict(getattr(mission_execution, "params", {}) or {})
        if "thruster_direction_body" in params:
            direction = _unit_vector_or_none(params.get("thruster_direction_body"))
        if direction is None:
            mount = resolve_thruster_mount_from_specs(getattr(sec, "specs", None) if sec is not None else None)
            direction = None if mount is None else _unit_vector_or_none(mount.thrust_direction_body)
        if direction is None:
            direction = plot_default
        out[oid] = np.array(direction if direction is not None else [1.0, 0.0, 0.0], dtype=float)
    return out


def _lift_axis_body_by_object(cfg: SimulationScenarioConfig) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for oid, sec in iter_object_sections(cfg, enabled_only=True, kind="satellite"):
        specs = dict(getattr(sec, "specs", {}) or {})
        axis = _unit_vector_or_none(aero_spec_get(specs, ("lift_axis_body", "lift_vector_body")))
        if axis is not None:
            out[oid] = axis
    return out


def _thrust_alignment_error_deg_series(
    *,
    t_s: np.ndarray,
    truth_hist: np.ndarray,
    thrust_hist: np.ndarray,
    thruster_direction_body: np.ndarray,
) -> np.ndarray:
    err_deg = np.full(t_s.shape, np.nan, dtype=float)
    thrust_dir_body = _unit_vector_or_none(thruster_direction_body)
    if thrust_dir_body is None:
        thrust_dir_body = np.array([1.0, 0.0, 0.0], dtype=float)
    for k in range(min(truth_hist.shape[0], thrust_hist.shape[0], t_s.size)):
        a_cmd = np.array(thrust_hist[k, :], dtype=float)
        if not np.all(np.isfinite(a_cmd)):
            continue
        a_norm = float(np.linalg.norm(a_cmd))
        if a_norm <= 1e-15:
            continue
        q_bn = np.array(truth_hist[k, 6:10], dtype=float)
        if not np.all(np.isfinite(q_bn)):
            continue
        c_bn = quaternion_to_dcm_bn(q_bn)
        thrust_axis_eci = c_bn.T @ thrust_dir_body
        burn_dir_eci = -a_cmd / a_norm
        cosang = float(np.clip(np.dot(thrust_axis_eci, burn_dir_eci), -1.0, 1.0))
        if not np.isfinite(cosang):
            continue
        err_deg[k] = float(np.degrees(np.arccos(cosang)))
    return err_deg
