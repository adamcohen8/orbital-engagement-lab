from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Callable

import numpy as np

from sim.aero import aero_spec_get
from sim.config import (
    SimulationScenarioConfig,
    default_pair_object_ids,
    default_reference_object_id,
    iter_object_sections,
)
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2, EARTH_RADIUS_KM
from sim.dynamics.orbit.frames import frame_context_from_mapping
from sim.ground_stations import evaluate_ground_station_access
from sim.plotting.style import artifact_metadata, oel_plot_context, save_oel_figure, style_name_from_config
from sim.presets.rockets import RocketStackPreset
from sim.presets.thrusters import resolve_thruster_mount_from_specs
from sim.utils.figure_size import cap_figsize
from sim.utils.frames import ric_dcm_ir_from_rv
from sim.utils.ground_track import ground_track_from_eci_history
from sim.utils.quaternion import quaternion_to_dcm_bn


def _private_bridge_figure_ids() -> list[str]:
    module_name = ".".join(("integrations", "c" + "f" + "s" + "_sil", "plots"))
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        return []
    return [str(item) for item in tuple(getattr(module, "FIGURE_IDS", ()) or ())]


AVAILABLE_FIGURE_IDS = [
    "run_dashboard",
    "rendezvous_summary",
    "rendezvous_summary_curvilinear",
    "orbit_eci",
    "orbital_element_a",
    "orbital_element_ecc",
    "orbital_element_inc",
    "orbital_element_raan",
    "orbital_element_argp",
    "orbital_element_true_anomaly",
    "orbital_elements_summary",
    "orbital_elements_angles",
    "ground_track",
    "ground_track_multi",
    "trajectory_ecef",
    "trajectory_ric_rect",
    "trajectory_ric_curv",
    "trajectory_ric_rect_2d",
    "trajectory_ric_curv_2d",
    "trajectory_eci_multi",
    "trajectory_ecef_multi",
    "trajectory_ric_rect_multi",
    "trajectory_ric_curv_multi",
    "trajectory_ric_rect_2d_multi",
    "trajectory_ric_rect_2d_multi_target_burns",
    "trajectory_ric_curv_2d_multi",
    "trajectory_ric_curv_2d_multi_target_burns",
    "attitude",
    "quaternion_eci",
    "quaternion_ric",
    "rates_eci",
    "rates_ric",
    "relative_range",
    "knowledge_timeline",
    "control_thrust",
    "control_thrust_multi",
    "control_thrust_ric",
    "control_thrust_ric_multi",
    "control_effort",
    "estimation_error",
    "estimation_error_components",
    "knowledge_filtering",
    "sensor_access",
    "ground_station_access",
    "quaternion_error",
    "attitude_control_summary",
    "rocket_ascent_diagnostics",
    "rocket_gnc_diagnostics",
    "rocket_orbital_elements",
    "rocket_fuel_remaining",
    "rocket_mission_timeline",
    "rocket_downrange_altitude",
    "rocket_maxq_throttle",
    "rocket_tvc_aero_authority",
    "rocket_insertion_scorecard",
    "reentry_summary",
    "reentry_aero",
    "reentry_thermal",
    "atmospheric_pass",
    "satellite_delta_v_remaining",
    "thrust_alignment_error",
    "mission_recovery_trade_space",
] + _private_bridge_figure_ids()

PLOT_PRESETS = {
    "minimal": ["run_dashboard"],
    "orbit": ["run_dashboard", "trajectory_eci_multi", "ground_track_multi", "orbital_elements_summary"],
    "rendezvous": [
        "run_dashboard",
        "rendezvous_summary",
        "trajectory_ric_curv_2d_multi",
        "relative_range",
        "control_effort",
    ],
    "attitude": ["run_dashboard", "quaternion_eci", "rates_eci", "quaternion_error", "attitude_control_summary"],
    "estimation": [
        "estimation_error",
        "estimation_error_components",
        "knowledge_filtering",
        "knowledge_timeline",
        "sensor_access",
    ],
    "access": ["ground_station_access", "ground_track_multi"],
    "rocket": [
        "run_dashboard",
        "rocket_ascent_diagnostics",
        "rocket_gnc_diagnostics",
        "rocket_orbital_elements",
        "rocket_fuel_remaining",
        "rocket_mission_timeline",
        "rocket_downrange_altitude",
        "rocket_maxq_throttle",
        "rocket_tvc_aero_authority",
        "rocket_insertion_scorecard",
    ],
    "reentry": ["reentry_summary", "reentry_aero", "reentry_thermal"],
    "aero_assist": ["atmospheric_pass", "reentry_aero", "reentry_thermal", "trajectory_eci_multi"],
    "mission_recovery": ["mission_recovery_trade_space"],
    "debug": list(AVAILABLE_FIGURE_IDS),
}


def _expanded_figure_ids(plots_cfg: dict[str, Any]) -> list[str]:
    raw_presets = plots_cfg.get("preset", plots_cfg.get("presets", []))
    if isinstance(raw_presets, str):
        presets = [raw_presets]
    elif isinstance(raw_presets, list):
        presets = [str(x) for x in raw_presets]
    else:
        presets = []

    expanded: list[str] = []
    for preset in presets:
        key = preset.strip().lower()
        if not key:
            continue
        if key not in PLOT_PRESETS:
            valid = ", ".join(sorted(PLOT_PRESETS.keys()))
            raise ValueError(f"Unknown plot preset '{preset}'. Valid presets: {valid}")
        expanded.extend(PLOT_PRESETS[key])
    expanded.extend(str(x) for x in list(plots_cfg.get("figure_ids", []) or []))

    out: list[str] = []
    seen: set[str] = set()
    for figure_id in expanded:
        fid = str(figure_id).strip()
        if not fid or fid in seen:
            continue
        out.append(fid)
        seen.add(fid)
    return out


AVAILABLE_ANIMATION_TYPES = [
    "ground_track",
    "ground_track_multi",
    "attitude_ric_thruster",
    "battlespace_dashboard",
    "ric_curv_prism_multi",
    "ric_prism_side_by_side",
    "target_reference_ric_curv_3d",
    "target_reference_ric_curv_2d",
    "target_reference_ric_curv_2d_ri",
    "target_reference_ric_curv_2d_ic",
    "target_reference_ric_curv_2d_rc",
]


def _plot_private_bridge_outputs(
    *,
    figure_ids: list[str],
    bridge_hist: dict[str, list[dict[str, Any]]] | None,
    outdir: Path,
    mode: str,
    dpi: int,
) -> dict[str, str]:
    if not bridge_hist:
        return {}
    module_name = ".".join(("integrations", "c" + "f" + "s" + "_sil", "plots"))
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        return {}
    plotter = getattr(module, "plot_bridge_outputs", None)
    if not callable(plotter):
        return {}
    return plotter(figure_ids=figure_ids, bridge_hist=bridge_hist, outdir=outdir, mode=mode, dpi=dpi)


def _load_plotting_functions() -> dict[str, Any]:
    from sim.utils.plotting import plot_attitude_tumble, plot_orbit_eci
    from sim.utils.plotting_capabilities import (
        animate_battlespace_dashboard,
        animate_ground_track,
        animate_multi_ground_track,
        animate_multi_rectangular_prism_ric_curv,
        animate_multi_ric_2d_projections,
        animate_multi_trajectory_frame,
        animate_rectangular_prism_attitude,
        animate_side_by_side_rectangular_prism_ric_attitude,
        plot_body_rates,
        plot_control_commands,
        plot_multi_control_commands,
        plot_multi_ric_2d_projections,
        plot_multi_trajectory_frame,
        plot_quaternion_components,
        plot_ric_2d_projections,
        plot_trajectory_frame,
    )

    return {
        "plot_orbit_eci": plot_orbit_eci,
        "plot_attitude_tumble": plot_attitude_tumble,
        "animate_battlespace_dashboard": animate_battlespace_dashboard,
        "animate_rectangular_prism_attitude": animate_rectangular_prism_attitude,
        "animate_multi_ric_2d_projections": animate_multi_ric_2d_projections,
        "plot_body_rates": plot_body_rates,
        "plot_control_commands": plot_control_commands,
        "plot_multi_control_commands": plot_multi_control_commands,
        "animate_multi_trajectory_frame": animate_multi_trajectory_frame,
        "plot_multi_ric_2d_projections": plot_multi_ric_2d_projections,
        "plot_multi_trajectory_frame": plot_multi_trajectory_frame,
        "plot_quaternion_components": plot_quaternion_components,
        "plot_ric_2d_projections": plot_ric_2d_projections,
        "plot_trajectory_frame": plot_trajectory_frame,
        "animate_ground_track": animate_ground_track,
        "animate_multi_ground_track": animate_multi_ground_track,
        "animate_multi_rectangular_prism_ric_curv": animate_multi_rectangular_prism_ric_curv,
        "animate_side_by_side_rectangular_prism_ric_attitude": animate_side_by_side_rectangular_prism_ric_attitude,
    }


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


def plot_outputs(
    *,
    cfg: SimulationScenarioConfig,
    t_s: np.ndarray,
    truth_hist: dict[str, np.ndarray],
    target_reference_orbit_truth: np.ndarray | None,
    thrust_hist: dict[str, np.ndarray],
    desired_attitude_hist: dict[str, np.ndarray] | None,
    knowledge_hist: dict[str, dict[str, np.ndarray]],
    rocket_metrics: dict[str, np.ndarray] | None,
    outdir: Path,
    resolve_rocket_stack: Callable[[dict[str, Any]], RocketStackPreset],
    resolve_satellite_isp_s: Callable[[dict[str, Any]], float],
    belief_hist: dict[str, np.ndarray] | None = None,
    knowledge_measurement_hist: dict[str, dict[str, np.ndarray]] | None = None,
    bridge_hist: dict[str, list[dict[str, Any]]] | None = None,
    reentry_metrics: dict[str, dict[str, np.ndarray]] | None = None,
) -> dict[str, str]:
    if not bool(cfg.outputs.plots.get("enabled", True)):
        return {}
    metadata = artifact_metadata(scenario_name=str(cfg.scenario_name or ""))
    style_name = style_name_from_config(dict(cfg.outputs.plots or {}))
    with oel_plot_context(style_name=style_name, metadata=metadata):
        return _plot_outputs_impl(
            cfg=cfg,
            t_s=t_s,
            truth_hist=truth_hist,
            target_reference_orbit_truth=target_reference_orbit_truth,
            thrust_hist=thrust_hist,
            desired_attitude_hist=desired_attitude_hist,
            knowledge_hist=knowledge_hist,
            rocket_metrics=rocket_metrics,
            outdir=outdir,
            resolve_rocket_stack=resolve_rocket_stack,
            resolve_satellite_isp_s=resolve_satellite_isp_s,
            belief_hist=belief_hist,
            knowledge_measurement_hist=knowledge_measurement_hist,
            bridge_hist=bridge_hist,
            reentry_metrics=reentry_metrics,
        )


def _plot_outputs_impl(
    *,
    cfg: SimulationScenarioConfig,
    t_s: np.ndarray,
    truth_hist: dict[str, np.ndarray],
    target_reference_orbit_truth: np.ndarray | None,
    thrust_hist: dict[str, np.ndarray],
    desired_attitude_hist: dict[str, np.ndarray] | None,
    knowledge_hist: dict[str, dict[str, np.ndarray]],
    rocket_metrics: dict[str, np.ndarray] | None,
    outdir: Path,
    resolve_rocket_stack: Callable[[dict[str, Any]], RocketStackPreset],
    resolve_satellite_isp_s: Callable[[dict[str, Any]], float],
    belief_hist: dict[str, np.ndarray] | None = None,
    knowledge_measurement_hist: dict[str, dict[str, np.ndarray]] | None = None,
    bridge_hist: dict[str, list[dict[str, Any]]] | None = None,
    reentry_metrics: dict[str, dict[str, np.ndarray]] | None = None,
) -> dict[str, str]:
    out: dict[str, str] = {}
    if not bool(cfg.outputs.plots.get("enabled", True)):
        return out
    mode = cfg.outputs.mode
    figure_ids = _expanded_figure_ids(dict(cfg.outputs.plots or {}))
    ric_2d_planes = list(cfg.outputs.plots.get("ric_2d_planes", ["ri", "ic", "rc"]) or ["ri", "ic", "rc"])
    frame_context = frame_context_from_mapping(
        dict(getattr(cfg.simulator, "frames", {}) or {}),
        jd_utc_start=cfg.simulator.initial_jd_utc,
        source="scenario",
    )
    reference_object_id = str(cfg.outputs.plots.get("reference_object_id", "")).strip()
    reference_object_label = str(cfg.outputs.plots.get("reference_object_label", "")).strip() or None
    reference_truth_override = None
    if target_reference_orbit_truth is not None:
        ref_arr = np.array(target_reference_orbit_truth, dtype=float)
        if ref_arr.ndim == 2 and ref_arr.shape[1] >= 6 and np.any(np.isfinite(ref_arr[:, 0])):
            reference_truth_override = ref_arr
    if reference_truth_override is not None:
        reference_truth = reference_truth_override
        ric_truth_hist = dict(truth_hist)
        reference_object_id = ""
    else:
        if reference_object_id and reference_object_id not in truth_hist:
            reference_object_id = ""
        if not reference_object_id:
            reference_object_id = default_reference_object_id(cfg, available_ids=truth_hist.keys()) or ""
        reference_truth = truth_hist.get(reference_object_id) if reference_object_id else None
        ric_truth_hist = (
            {oid: hist for oid, hist in truth_hist.items() if oid != reference_object_id}
            if reference_object_id
            else dict(truth_hist)
        )
    if not figure_ids:
        return out
    plot_fns = _load_plotting_functions()
    plot_orbit_eci = plot_fns["plot_orbit_eci"]
    plot_attitude_tumble = plot_fns["plot_attitude_tumble"]
    plot_body_rates = plot_fns["plot_body_rates"]
    plot_control_commands = plot_fns["plot_control_commands"]
    plot_multi_control_commands = plot_fns["plot_multi_control_commands"]
    plot_multi_ric_2d_projections = plot_fns["plot_multi_ric_2d_projections"]
    plot_multi_trajectory_frame = plot_fns["plot_multi_trajectory_frame"]
    plot_quaternion_components = plot_fns["plot_quaternion_components"]
    plot_ric_2d_projections = plot_fns["plot_ric_2d_projections"]
    plot_trajectory_frame = plot_fns["plot_trajectory_frame"]
    if any(
        fid in figure_ids
        for fid in (
            "run_dashboard",
            "rendezvous_summary",
            "rendezvous_summary_curvilinear",
            "control_effort",
            "estimation_error",
            "estimation_error_components",
            "knowledge_filtering",
            "sensor_access",
            "ground_station_access",
            "attitude_control_summary",
            "orbital_element_a",
            "orbital_element_ecc",
            "orbital_element_inc",
            "orbital_element_raan",
            "orbital_element_argp",
            "orbital_element_true_anomaly",
            "orbital_elements_summary",
            "orbital_elements_angles",
            "ground_track",
            "ground_track_multi",
            "reentry_summary",
            "reentry_aero",
            "reentry_thermal",
            "atmospheric_pass",
        )
    ):
        from sim.plotting import (
            plot_atmospheric_pass,
            plot_attitude_control_summary,
            plot_control_effort,
            plot_estimation_error,
            plot_estimation_error_components,
            plot_ground_station_access,
            plot_ground_track_from_payload,
            plot_knowledge_filtering,
            plot_orbital_element,
            plot_orbital_elements_angles,
            plot_orbital_elements_summary,
            plot_reentry_aero,
            plot_reentry_summary,
            plot_reentry_thermal,
            plot_rendezvous_summary,
            plot_rendezvous_summary_curvilinear,
            plot_run_dashboard,
            plot_sensor_access,
        )
    dpi = int(cfg.outputs.plots.get("dpi", 150))
    show = mode in ("interactive", "both")
    close = mode == "save"
    save_enabled = mode in ("save", "both")
    draw_ground_track_map = bool(cfg.outputs.plots.get("draw_earth_map", False))

    if "run_dashboard" in figure_ids:
        p = outdir / "run_dashboard.png"
        plot_run_dashboard(
            t_s=t_s,
            truth_by_object=truth_hist,
            thrust_by_object=thrust_hist,
            belief_by_object=belief_hist or {},
            target_reference_orbit_truth=target_reference_orbit_truth,
            reference_object_id=reference_object_id or None,
            out_path=p if save_enabled else None,
            show=show,
            close=close,
            dpi=dpi,
        )
        if save_enabled:
            out["run_dashboard"] = str(p)

    if "rendezvous_summary" in figure_ids:
        p = outdir / "rendezvous_summary.png"
        keepout_radius = cfg.outputs.plots.get("keepout_radius_km")
        plot_rendezvous_summary(
            t_s=t_s,
            truth_by_object=truth_hist,
            target_reference_orbit_truth=target_reference_orbit_truth,
            reference_object_id=reference_object_id or None,
            keepout_radius_km=None if keepout_radius is None else float(keepout_radius),
            out_path=p if save_enabled else None,
            show=show,
            close=close,
            dpi=dpi,
        )
        if save_enabled:
            out["rendezvous_summary"] = str(p)

    if "rendezvous_summary_curvilinear" in figure_ids:
        p = outdir / "rendezvous_summary_curvilinear.png"
        keepout_radius = cfg.outputs.plots.get("keepout_radius_km")
        plot_rendezvous_summary_curvilinear(
            t_s=t_s,
            truth_by_object=truth_hist,
            thrust_by_object=thrust_hist,
            target_reference_orbit_truth=target_reference_orbit_truth,
            reference_object_id=reference_object_id or None,
            keepout_radius_km=None if keepout_radius is None else float(keepout_radius),
            out_path=p if save_enabled else None,
            show=show,
            close=close,
            dpi=dpi,
        )
        if save_enabled:
            out["rendezvous_summary_curvilinear"] = str(p)

    if "control_effort" in figure_ids:
        p = outdir / "control_effort.png"
        plot_control_effort(
            t_s=t_s,
            thrust_by_object=thrust_hist,
            out_path=p if save_enabled else None,
            show=show,
            close=close,
            dpi=dpi,
        )
        if save_enabled:
            out["control_effort"] = str(p)

    if "estimation_error" in figure_ids:
        p = outdir / "estimation_error.png"
        plot_estimation_error(
            t_s=t_s,
            truth_by_object=truth_hist,
            belief_by_object=belief_hist or {},
            out_path=p if save_enabled else None,
            show=show,
            close=close,
            dpi=dpi,
        )
        if save_enabled:
            out["estimation_error"] = str(p)

    if "estimation_error_components" in figure_ids:
        p = outdir / "estimation_error_components.png"
        plot_estimation_error_components(
            t_s=t_s,
            truth_by_object=truth_hist,
            belief_by_object=belief_hist or {},
            out_path=p if save_enabled else None,
            show=show,
            close=close,
            dpi=dpi,
        )
        if save_enabled:
            out["estimation_error_components"] = str(p)

    if "knowledge_filtering" in figure_ids:
        p = outdir / "knowledge_filtering.png"
        plot_knowledge_filtering(
            t_s=t_s,
            truth_by_object=truth_hist,
            knowledge_by_observer=knowledge_hist,
            knowledge_measurements_by_observer=knowledge_measurement_hist or {},
            out_path=p if save_enabled else None,
            show=show,
            close=close,
            dpi=dpi,
        )
        if save_enabled:
            out["knowledge_filtering"] = str(p)

    if "sensor_access" in figure_ids:
        p = outdir / "sensor_access.png"
        plot_sensor_access(
            t_s=t_s,
            truth_by_object=truth_hist,
            knowledge_by_observer=knowledge_hist,
            out_path=p if save_enabled else None,
            show=show,
            close=close,
            dpi=dpi,
        )
        if save_enabled:
            out["sensor_access"] = str(p)

    if "ground_station_access" in figure_ids:
        ground_access, _ = evaluate_ground_station_access(
            ground_stations=list(cfg.ground_stations),
            t_s=t_s,
            truth_hist=truth_hist,
            jd_utc_start=cfg.simulator.initial_jd_utc,
            frame_context=frame_context,
        )
        p = outdir / "ground_station_access.png"
        plot_ground_station_access(
            t_s=t_s,
            ground_station_access=ground_access,
            out_path=p if save_enabled else None,
            show=show,
            close=close,
            dpi=dpi,
        )
        if save_enabled:
            out["ground_station_access"] = str(p)

    if "attitude_control_summary" in figure_ids:
        p = outdir / "attitude_control_summary.png"
        plot_attitude_control_summary(
            t_s=t_s,
            truth_by_object=truth_hist,
            thrust_by_object=thrust_hist,
            desired_attitude_by_object=desired_attitude_hist or {},
            thrust_axis_body_by_object=_thruster_direction_body_by_object(cfg),
            out_path=p if save_enabled else None,
            show=show,
            close=close,
            dpi=dpi,
        )
        if save_enabled:
            out["attitude_control_summary"] = str(p)

    reentry_cfg = dict(dict(cfg.simulator.dynamics or {}).get("reentry", {}) or {})
    if "reentry_summary" in figure_ids:
        p = outdir / "reentry_summary.png"
        plot_reentry_summary(
            t_s=t_s,
            reentry_metrics_by_object=reentry_metrics or {},
            begin_altitude_km=(
                None if reentry_cfg.get("begin_altitude_km") is None else float(reentry_cfg.get("begin_altitude_km"))
            ),
            out_path=p if save_enabled else None,
            show=show,
            close=close,
            dpi=dpi,
        )
        if save_enabled:
            out["reentry_summary"] = str(p)

    if "reentry_aero" in figure_ids:
        p = outdir / "reentry_aero.png"
        plot_reentry_aero(
            t_s=t_s,
            reentry_metrics_by_object=reentry_metrics or {},
            out_path=p if save_enabled else None,
            show=show,
            close=close,
            dpi=dpi,
        )
        if save_enabled:
            out["reentry_aero"] = str(p)

    if "reentry_thermal" in figure_ids:
        p = outdir / "reentry_thermal.png"
        plot_reentry_thermal(
            t_s=t_s,
            reentry_metrics_by_object=reentry_metrics or {},
            out_path=p if save_enabled else None,
            show=show,
            close=close,
            dpi=dpi,
        )
        if save_enabled:
            out["reentry_thermal"] = str(p)

    if "atmospheric_pass" in figure_ids:
        p = outdir / "atmospheric_pass.png"
        plot_atmospheric_pass(
            t_s=t_s,
            truth_by_object=truth_hist,
            reentry_metrics_by_object=reentry_metrics or {},
            lift_axis_body_by_object=_lift_axis_body_by_object(cfg),
            begin_altitude_km=(
                None if reentry_cfg.get("begin_altitude_km") is None else float(reentry_cfg.get("begin_altitude_km"))
            ),
            out_path=p if save_enabled else None,
            show=show,
            close=close,
            dpi=dpi,
        )
        if save_enabled:
            out["atmospheric_pass"] = str(p)

    orbital_element_ids = {
        "orbital_element_a": "a",
        "orbital_element_ecc": "ecc",
        "orbital_element_inc": "inc",
        "orbital_element_raan": "raan",
        "orbital_element_argp": "argp",
        "orbital_element_true_anomaly": "true_anomaly",
    }
    orbital_object_id = str(cfg.outputs.plots.get("orbital_elements_object_id", "") or "").strip() or None
    for figure_id, element_id in orbital_element_ids.items():
        if figure_id not in figure_ids:
            continue
        p = outdir / f"{figure_id}.png"
        plot_orbital_element(
            t_s=t_s,
            truth_by_object=truth_hist,
            element_id=element_id,
            object_id=orbital_object_id,
            out_path=p if save_enabled else None,
            show=show,
            close=close,
            dpi=dpi,
        )
        if save_enabled:
            out[figure_id] = str(p)

    if "orbital_elements_summary" in figure_ids:
        p = outdir / "orbital_elements_summary.png"
        plot_orbital_elements_summary(
            t_s=t_s,
            truth_by_object=truth_hist,
            object_id=orbital_object_id,
            out_path=p if save_enabled else None,
            show=show,
            close=close,
            dpi=dpi,
        )
        if save_enabled:
            out["orbital_elements_summary"] = str(p)

    if "orbital_elements_angles" in figure_ids:
        p = outdir / "orbital_elements_angles.png"
        plot_orbital_elements_angles(
            t_s=t_s,
            truth_by_object=truth_hist,
            object_id=orbital_object_id,
            out_path=p if save_enabled else None,
            show=show,
            close=close,
            dpi=dpi,
        )
        if save_enabled:
            out["orbital_elements_angles"] = str(p)

    out.update(
        _plot_private_bridge_outputs(
            figure_ids=figure_ids,
            bridge_hist=bridge_hist,
            outdir=outdir,
            mode=mode,
            dpi=dpi,
        )
    )

    if "ground_track_multi" in figure_ids:
        p = outdir / "ground_track_multi.png"
        plot_ground_track_from_payload(
            t_s=t_s,
            truth_by_object=truth_hist,
            jd_utc_start=cfg.simulator.initial_jd_utc,
            draw_earth_map=draw_ground_track_map,
            out_path=p if save_enabled else None,
            show=show,
            close=close,
            dpi=dpi,
        )
        if save_enabled:
            out["ground_track_multi"] = str(p)

    if "ground_track" in figure_ids:
        for oid, hist in truth_hist.items():
            if hist.size == 0 or not np.any(np.isfinite(hist[:, 0])):
                continue
            p = outdir / f"{oid}_ground_track.png"
            plot_ground_track_from_payload(
                t_s=t_s,
                truth_by_object={oid: hist},
                jd_utc_start=cfg.simulator.initial_jd_utc,
                object_id=oid,
                draw_earth_map=draw_ground_track_map,
                out_path=p if save_enabled else None,
                show=show,
                close=close,
                dpi=dpi,
            )
            if save_enabled:
                out[f"{oid}_ground_track"] = str(p)

    for oid, hist in truth_hist.items():
        if not np.any(np.isfinite(hist[:, 0])):
            continue
        if "orbit_eci" in figure_ids:
            p = outdir / f"{oid}_orbit_eci.png"
            plot_orbit_eci(hist, mode=mode, out_path=str(p))
            if mode in ("save", "both"):
                out[f"{oid}_orbit_eci"] = str(p)
        if "attitude" in figure_ids:
            p = outdir / f"{oid}_attitude.png"
            plot_attitude_tumble(t_s=t_s, truth_hist=hist, mode=mode, out_path=str(p))
            if mode in ("save", "both"):
                out[f"{oid}_attitude"] = str(p)

    if "relative_range" in figure_ids:
        import matplotlib.pyplot as plt

        ids = list(truth_hist.keys())
        fig, ax = plt.subplots(figsize=cap_figsize(10, 5))
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a = truth_hist[ids[i]][:, :3]
                b = truth_hist[ids[j]][:, :3]
                mask = np.isfinite(a[:, 0]) & np.isfinite(b[:, 0])
                if not np.any(mask):
                    continue
                rr = np.linalg.norm(a - b, axis=1)
                ax.plot(t_s[mask], rr[mask], label=f"{ids[i]}-{ids[j]}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Range (km)")
        ax.set_title("Relative Range")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        p = outdir / "relative_ranges.png"
        if mode in ("save", "both"):
            save_oel_figure(fig, p, dpi=int(cfg.outputs.plots.get("dpi", 150)), artifact_id=p.stem)
            out["relative_ranges"] = str(p)
        if mode in ("interactive", "both"):
            plt.show(block=False)
        else:
            plt.close(fig)

    if "quaternion_error" in figure_ids and desired_attitude_hist is not None:
        import matplotlib.pyplot as plt

        for oid, hist in truth_hist.items():
            q_des_hist = desired_attitude_hist.get(oid) if isinstance(desired_attitude_hist, dict) else None
            if q_des_hist is None or q_des_hist.shape[0] == 0:
                continue
            n_s = min(hist.shape[0], q_des_hist.shape[0], t_s.size)
            if n_s <= 0:
                continue
            qd = np.array(q_des_hist[:n_s, :], dtype=float)
            qc = np.array(hist[:n_s, 6:10], dtype=float)
            for k in range(1, n_s):
                if not np.all(np.isfinite(qd[k, :])) and np.all(np.isfinite(qd[k - 1, :])):
                    qd[k, :] = qd[k - 1, :]
            err_deg = np.full(n_s, np.nan, dtype=float)
            for k in range(n_s):
                if not (np.all(np.isfinite(qd[k, :])) and np.all(np.isfinite(qc[k, :]))):
                    continue
                err_deg[k] = _quat_error_angle_deg(qd[k, :], qc[k, :])
            finite = np.isfinite(err_deg)
            if not np.any(finite):
                continue
            fig, ax = plt.subplots(figsize=cap_figsize(10, 5))
            ax.plot(t_s[:n_s][finite], err_deg[finite], linewidth=1.4)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Error Angle (deg)")
            ax.set_title(f"Quaternion Tracking Error ({oid})")
            ax.grid(True, alpha=0.3)
            p = outdir / f"{oid}_quaternion_error.png"
            if mode in ("save", "both"):
                save_oel_figure(fig, p, dpi=int(cfg.outputs.plots.get("dpi", 150)), artifact_id=p.stem)
                out[f"{oid}_quaternion_error"] = str(p)
            if mode in ("interactive", "both"):
                plt.show(block=False)
            else:
                plt.close(fig)

    if "trajectory_eci_multi" in figure_ids:
        p = outdir / "trajectory_eci_multi.png"
        plot_multi_trajectory_frame(t_s, truth_hist, frame="eci", mode=mode, out_path=str(p))
        if mode in ("save", "both"):
            out["trajectory_eci_multi"] = str(p)
    if "trajectory_ecef_multi" in figure_ids:
        p = outdir / "trajectory_ecef_multi.png"
        plot_multi_trajectory_frame(
            t_s,
            truth_hist,
            frame="ecef",
            mode=mode,
            out_path=str(p),
            frame_context=frame_context,
        )
        if mode in ("save", "both"):
            out["trajectory_ecef_multi"] = str(p)
    if "trajectory_ric_rect_multi" in figure_ids and reference_truth is not None:
        p = outdir / "trajectory_ric_rect_multi.png"
        plot_multi_trajectory_frame(
            t_s,
            ric_truth_hist,
            frame="ric_rect",
            reference_truth_hist=reference_truth,
            reference_label=reference_object_label,
            mode=mode,
            out_path=str(p),
        )
        if mode in ("save", "both"):
            out["trajectory_ric_rect_multi"] = str(p)
    if "trajectory_ric_curv_multi" in figure_ids and reference_truth is not None:
        p = outdir / "trajectory_ric_curv_multi.png"
        plot_multi_trajectory_frame(
            t_s,
            ric_truth_hist,
            frame="ric_curv",
            reference_truth_hist=reference_truth,
            reference_label=reference_object_label,
            mode=mode,
            out_path=str(p),
        )
        if mode in ("save", "both"):
            out["trajectory_ric_curv_multi"] = str(p)
    if "trajectory_ric_rect_2d_multi" in figure_ids and reference_truth is not None:
        p = outdir / "trajectory_ric_rect_2d_multi.png"
        plot_multi_ric_2d_projections(
            t_s,
            ric_truth_hist,
            frame="ric_rect",
            reference_truth_hist=reference_truth,
            reference_label=reference_object_label,
            planes=ric_2d_planes,
            mode=mode,
            out_path=str(p),
        )
        if mode in ("save", "both"):
            out["trajectory_ric_rect_2d_multi"] = str(p)
    if "trajectory_ric_rect_2d_multi_target_burns" in figure_ids and reference_truth is not None:
        p = outdir / "trajectory_ric_rect_2d_multi_target_burns.png"
        burn_marker_object_ids = [
            str(oid)
            for oid in list(cfg.outputs.plots.get("burn_marker_object_ids", ["target"]) or ["target"])
            if str(oid).strip()
        ]
        plot_multi_ric_2d_projections(
            t_s,
            ric_truth_hist,
            frame="ric_rect",
            reference_truth_hist=reference_truth,
            reference_label=reference_object_label,
            burn_marker_by_object=thrust_hist,
            burn_marker_object_ids=burn_marker_object_ids,
            planes=ric_2d_planes,
            mode=mode,
            out_path=str(p),
        )
        if mode in ("save", "both"):
            out["trajectory_ric_rect_2d_multi_target_burns"] = str(p)
    if "trajectory_ric_curv_2d_multi" in figure_ids and reference_truth is not None:
        p = outdir / "trajectory_ric_curv_2d_multi.png"
        plot_multi_ric_2d_projections(
            t_s,
            ric_truth_hist,
            frame="ric_curv",
            reference_truth_hist=reference_truth,
            reference_label=reference_object_label,
            planes=ric_2d_planes,
            mode=mode,
            out_path=str(p),
        )
        if mode in ("save", "both"):
            out["trajectory_ric_curv_2d_multi"] = str(p)
    if "trajectory_ric_curv_2d_multi_target_burns" in figure_ids and reference_truth is not None:
        p = outdir / "trajectory_ric_curv_2d_multi_target_burns.png"
        burn_marker_object_ids = [
            str(oid)
            for oid in list(cfg.outputs.plots.get("burn_marker_object_ids", ["target"]) or ["target"])
            if str(oid).strip()
        ]
        plot_multi_ric_2d_projections(
            t_s,
            ric_truth_hist,
            frame="ric_curv",
            reference_truth_hist=reference_truth,
            reference_label=reference_object_label,
            burn_marker_by_object=thrust_hist,
            burn_marker_object_ids=burn_marker_object_ids,
            planes=ric_2d_planes,
            mode=mode,
            out_path=str(p),
        )
        if mode in ("save", "both"):
            out["trajectory_ric_curv_2d_multi_target_burns"] = str(p)

    for oid, hist in truth_hist.items():
        if not np.any(np.isfinite(hist[:, 0])):
            continue
        if "quaternion_eci" in figure_ids:
            p = outdir / f"{oid}_quat_eci.png"
            plot_quaternion_components(t_s, hist, frame="eci", layout="single", mode=mode, out_path=str(p))
            if mode in ("save", "both"):
                out[f"{oid}_quat_eci"] = str(p)
        if "quaternion_ric" in figure_ids:
            p = outdir / f"{oid}_quat_ric.png"
            plot_quaternion_components(t_s, hist, frame="ric", layout="single", mode=mode, out_path=str(p))
            if mode in ("save", "both"):
                out[f"{oid}_quat_ric"] = str(p)
        if "rates_eci" in figure_ids:
            p = outdir / f"{oid}_rates_eci.png"
            plot_body_rates(t_s, hist, frame="eci", layout="subplots", mode=mode, out_path=str(p))
            if mode in ("save", "both"):
                out[f"{oid}_rates_eci"] = str(p)
        if "rates_ric" in figure_ids:
            p = outdir / f"{oid}_rates_ric.png"
            plot_body_rates(t_s, hist, frame="ric", layout="subplots", mode=mode, out_path=str(p))
            if mode in ("save", "both"):
                out[f"{oid}_rates_ric"] = str(p)
        if "trajectory_ecef" in figure_ids:
            p = outdir / f"{oid}_traj_ecef.png"
            plot_trajectory_frame(t_s, hist, frame="ecef", mode=mode, out_path=str(p), frame_context=frame_context)
            if mode in ("save", "both"):
                out[f"{oid}_traj_ecef"] = str(p)
        if "trajectory_ric_rect" in figure_ids and reference_truth is not None and oid != reference_object_id:
            p = outdir / f"{oid}_traj_ric_rect.png"
            plot_trajectory_frame(
                t_s,
                hist,
                frame="ric_rect",
                reference_truth_hist=reference_truth,
                mode=mode,
                out_path=str(p),
            )
            if mode in ("save", "both"):
                out[f"{oid}_traj_ric_rect"] = str(p)
        if "trajectory_ric_curv" in figure_ids and reference_truth is not None and oid != reference_object_id:
            p = outdir / f"{oid}_traj_ric_curv.png"
            plot_trajectory_frame(
                t_s,
                hist,
                frame="ric_curv",
                reference_truth_hist=reference_truth,
                mode=mode,
                out_path=str(p),
            )
            if mode in ("save", "both"):
                out[f"{oid}_traj_ric_curv"] = str(p)
        if "trajectory_ric_rect_2d" in figure_ids and reference_truth is not None and oid != reference_object_id:
            p = outdir / f"{oid}_traj_ric_rect_2d.png"
            plot_ric_2d_projections(
                t_s,
                hist,
                frame="ric_rect",
                reference_truth_hist=reference_truth,
                planes=ric_2d_planes,
                mode=mode,
                out_path=str(p),
            )
            if mode in ("save", "both"):
                out[f"{oid}_traj_ric_rect_2d"] = str(p)
        if "trajectory_ric_curv_2d" in figure_ids and reference_truth is not None and oid != reference_object_id:
            p = outdir / f"{oid}_traj_ric_curv_2d.png"
            plot_ric_2d_projections(
                t_s,
                hist,
                frame="ric_curv",
                reference_truth_hist=reference_truth,
                planes=ric_2d_planes,
                mode=mode,
                out_path=str(p),
            )
            if mode in ("save", "both"):
                out[f"{oid}_traj_ric_curv_2d"] = str(p)

    if "rocket_ascent_diagnostics" in figure_ids and "rocket" in truth_hist:
        import matplotlib.pyplot as plt

        x = truth_hist["rocket"]
        r = x[:, 0:3]
        v = x[:, 3:6]
        m = x[:, 13]
        alt_km = np.linalg.norm(r, axis=1) - EARTH_RADIUS_KM
        speed_km_s = np.linalg.norm(v, axis=1)
        q_dyn = np.zeros_like(t_s)
        mach = np.zeros_like(t_s)
        stage = np.zeros_like(t_s)
        throttle = np.zeros_like(t_s)
        if rocket_metrics is not None:
            if "q_dyn_pa" in rocket_metrics:
                q_dyn = np.array(rocket_metrics["q_dyn_pa"], dtype=float).reshape(-1)[: t_s.size]
            if "mach" in rocket_metrics:
                mach = np.array(rocket_metrics["mach"], dtype=float).reshape(-1)[: t_s.size]
            if "stage_index" in rocket_metrics:
                stage = np.array(rocket_metrics["stage_index"], dtype=float).reshape(-1)[: t_s.size]
            if "throttle_cmd" in rocket_metrics:
                throttle = np.array(rocket_metrics["throttle_cmd"], dtype=float).reshape(-1)[: t_s.size]
        a_cmd = np.linalg.norm(np.nan_to_num(thrust_hist.get("rocket", np.zeros((t_s.size, 3))), nan=0.0), axis=1)

        fig, ax = plt.subplots(4, 1, figsize=cap_figsize(11, 11), sharex=True)

        ax0r = ax[0].twinx()
        l00 = ax[0].plot(t_s, alt_km, label="altitude (km)", color="tab:blue")
        l01 = ax0r.plot(t_s, speed_km_s, label="speed (km/s)", color="tab:orange")
        ax[0].set_ylabel("altitude (km)")
        ax0r.set_ylabel("speed (km/s)")
        ax[0].set_title("Rocket Ascent: Altitude and Speed")
        ax[0].grid(True, alpha=0.3)
        ax[0].legend(l00 + l01, [ln.get_label() for ln in (l00 + l01)], loc="best")

        ax1r = ax[1].twinx()
        l10 = ax[1].plot(t_s, q_dyn, label="q_dyn (Pa)", color="tab:green")
        l11 = ax1r.plot(t_s, mach, label="Mach", color="tab:red")
        ax[1].set_ylabel("dynamic pressure (Pa)")
        ax1r.set_ylabel("Mach")
        ax[1].set_title("Dynamic Pressure and Mach")
        ax[1].grid(True, alpha=0.3)
        ax[1].legend(l10 + l11, [ln.get_label() for ln in (l10 + l11)], loc="best")

        ax2r = ax[2].twinx()
        l20 = ax[2].plot(t_s, m, label="mass (kg)", color="tab:purple")
        l21 = ax2r.step(t_s, stage, where="post", label="stage index", color="tab:brown")
        ax[2].set_ylabel("mass (kg)")
        ax2r.set_ylabel("stage index")
        ax[2].set_title("Mass and Stage")
        ax[2].grid(True, alpha=0.3)
        ax[2].legend(l20 + l21, [ln.get_label() for ln in (l20 + l21)], loc="best")

        ax3r = ax[3].twinx()
        l30 = ax[3].plot(t_s, throttle, label="throttle", color="tab:cyan")
        l31 = ax3r.plot(t_s, a_cmd, label="|a_cmd| (km/s^2)", color="tab:gray")
        ax[3].set_ylabel("throttle")
        ax3r.set_ylabel("|a_cmd| (km/s^2)")
        ax[3].set_xlabel("time (s)")
        ax[3].set_title("Throttle and Commanded Acceleration")
        ax[3].grid(True, alpha=0.3)
        ax[3].legend(l30 + l31, [ln.get_label() for ln in (l30 + l31)], loc="best")
        fig.tight_layout()
        p = outdir / "rocket_ascent_diagnostics.png"
        if mode in ("save", "both"):
            save_oel_figure(fig, p, dpi=int(cfg.outputs.plots.get("dpi", 150)), artifact_id=p.stem)
            out["rocket_ascent_diagnostics"] = str(p)
        if mode == "save":
            plt.close(fig)

    if "rocket_gnc_diagnostics" in figure_ids and "rocket" in truth_hist:
        import matplotlib.pyplot as plt

        def _metric(name: str, default: float = 0.0) -> np.ndarray:
            if rocket_metrics is None or name not in rocket_metrics:
                return np.full(t_s.size, default, dtype=float)
            arr = np.array(rocket_metrics[name], dtype=float).reshape(-1)
            out_arr = np.full(t_s.size, np.nan, dtype=float)
            n = min(t_s.size, arr.size)
            out_arr[:n] = arr[:n]
            return out_arr

        fpa = _metric("flight_path_angle_deg")
        vertical_speed = _metric("vertical_speed_km_s")
        alpha = _metric("alpha_deg")
        beta = _metric("beta_deg")
        tvc = _metric("tvc_gimbal_deg")
        twr = _metric("thrust_to_weight")
        apo = _metric("apoapsis_alt_km", np.nan)
        peri = _metric("periapsis_alt_km", np.nan)

        fig, ax = plt.subplots(4, 1, figsize=cap_figsize(11, 11), sharex=True)

        ax0r = ax[0].twinx()
        l00 = ax[0].plot(t_s, fpa, label="flight path angle (deg)", color="tab:blue")
        l01 = ax0r.plot(t_s, vertical_speed, label="vertical speed (km/s)", color="tab:orange")
        ax[0].set_ylabel("FPA (deg)")
        ax0r.set_ylabel("vertical speed (km/s)")
        ax[0].set_title("Rocket GNC: Flight-Path State")
        ax[0].grid(True, alpha=0.3)
        ax[0].legend(l00 + l01, [ln.get_label() for ln in (l00 + l01)], loc="best")

        l10 = ax[1].plot(t_s, alpha, label="alpha (deg)", color="tab:red")
        l11 = ax[1].plot(t_s, beta, label="beta (deg)", color="tab:purple")
        ax[1].set_ylabel("angle (deg)")
        ax[1].set_title("Aero Angles")
        ax[1].grid(True, alpha=0.3)
        ax[1].legend(l10 + l11, [ln.get_label() for ln in (l10 + l11)], loc="best")

        ax2r = ax[2].twinx()
        l20 = ax[2].plot(t_s, tvc, label="TVC gimbal (deg)", color="tab:green")
        l21 = ax2r.plot(t_s, twr, label="thrust-to-weight", color="tab:brown")
        ax[2].set_ylabel("gimbal (deg)")
        ax2r.set_ylabel("T/W")
        ax[2].set_title("Control Authority")
        ax[2].grid(True, alpha=0.3)
        ax[2].legend(l20 + l21, [ln.get_label() for ln in (l20 + l21)], loc="best")

        l30 = ax[3].plot(t_s, apo, label="apogee alt (km)", color="tab:cyan")
        l31 = ax[3].plot(t_s, peri, label="perigee alt (km)", color="tab:gray")
        ax[3].set_ylabel("altitude (km)")
        ax[3].set_xlabel("time (s)")
        ax[3].set_title("Targeting Energy")
        ax[3].grid(True, alpha=0.3)
        ax[3].legend(l30 + l31, [ln.get_label() for ln in (l30 + l31)], loc="best")

        fig.tight_layout()
        p = outdir / "rocket_gnc_diagnostics.png"
        if mode in ("save", "both"):
            save_oel_figure(fig, p, dpi=int(cfg.outputs.plots.get("dpi", 150)), artifact_id=p.stem)
            out["rocket_gnc_diagnostics"] = str(p)
        if mode == "save":
            plt.close(fig)

    if "rocket_orbital_elements" in figure_ids and "rocket" in truth_hist:
        import matplotlib.pyplot as plt

        x = truth_hist["rocket"]
        a_km = np.full(t_s.size, np.nan, dtype=float)
        e = np.full(t_s.size, np.nan, dtype=float)
        for k in range(min(t_s.size, x.shape[0])):
            a_km[k], e[k] = _orbital_elements_basic(x[k, 0:3], x[k, 3:6], EARTH_MU_KM3_S2)

        fig, ax = plt.subplots(2, 1, figsize=cap_figsize(10, 7), sharex=True)
        ax[0].plot(t_s, a_km)
        ax[0].set_ylabel("a (km)")
        ax[0].set_title("Rocket Orbital Elements")
        ax[0].grid(True, alpha=0.3)

        ax[1].plot(t_s, e)
        ax[1].set_ylabel("e")
        ax[1].set_xlabel("time (s)")
        ax[1].grid(True, alpha=0.3)
        fig.tight_layout()
        p = outdir / "rocket_orbital_elements.png"
        if mode in ("save", "both"):
            save_oel_figure(fig, p, dpi=int(cfg.outputs.plots.get("dpi", 150)), artifact_id=p.stem)
            out["rocket_orbital_elements"] = str(p)
        if mode == "save":
            plt.close(fig)

    if "rocket_fuel_remaining" in figure_ids and "rocket" in truth_hist:
        import matplotlib.pyplot as plt

        x = truth_hist["rocket"]
        m = np.array(x[:, 13], dtype=float).reshape(-1)
        stack = resolve_rocket_stack(dict(cfg.rocket.specs or {}))
        payload_kg = float((cfg.rocket.specs or {}).get("payload_mass_kg", 150.0))
        dry_total_kg = float(sum(float(s.dry_mass_kg) for s in stack.stages) + payload_kg)
        prop0_kg = float(sum(float(s.propellant_mass_kg) for s in stack.stages))
        if prop0_kg > 0.0:
            fuel_rem_kg = np.clip(m - dry_total_kg, 0.0, prop0_kg)
            fuel_pct = 100.0 * fuel_rem_kg / prop0_kg
        else:
            fuel_pct = np.zeros_like(m)

        fig, ax = plt.subplots(figsize=cap_figsize(10, 4.5))
        ax.plot(t_s, fuel_pct, linewidth=1.6)
        ax.set_ylim(-1.0, 101.0)
        ax.set_ylabel("Fuel Remaining (%)")
        ax.set_xlabel("time (s)")
        ax.set_title("Rocket Fuel Remaining")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = outdir / "rocket_fuel_remaining.png"
        if mode in ("save", "both"):
            save_oel_figure(fig, p, dpi=int(cfg.outputs.plots.get("dpi", 150)), artifact_id=p.stem)
            out["rocket_fuel_remaining"] = str(p)
        if mode == "save":
            plt.close(fig)

    if "rocket_mission_timeline" in figure_ids and "rocket" in truth_hist:
        import matplotlib.pyplot as plt

        x = truth_hist["rocket"]
        target_alt_km, alt_tol_km, ecc_max = _rocket_target_altitude_cfg(cfg)
        alt_km = _rocket_metric_array(rocket_metrics, "altitude_km", t_s.size)
        if not np.any(np.isfinite(alt_km)):
            alt_km = np.linalg.norm(x[:, 0:3], axis=1) - EARTH_RADIUS_KM
        apo = _rocket_metric_array(rocket_metrics, "apoapsis_alt_km", t_s.size)
        peri = _rocket_metric_array(rocket_metrics, "periapsis_alt_km", t_s.size)
        ecc = _rocket_metric_array(rocket_metrics, "eccentricity", t_s.size)
        q_dyn = _rocket_metric_array(rocket_metrics, "q_dyn_pa", t_s.size, 0.0)
        stage = _rocket_metric_array(rocket_metrics, "stage_index", t_s.size, 0.0)

        events: list[tuple[float, str, str]] = [(float(t_s[0]) if t_s.size else 0.0, "Liftoff", "tab:green")]
        guidance = getattr(cfg.rocket, "base_guidance", None)
        guidance_params = dict(getattr(guidance, "params", {}) or {})
        for key, label in (("pitch_start_s", "Pitch start"), ("pitch_end_s", "Pitch complete")):
            value = guidance_params.get(key)
            if value is not None:
                events.append((float(value), label, "tab:blue"))
        finite_q = np.isfinite(q_dyn)
        if np.any(finite_q):
            i_q = int(np.nanargmax(np.where(finite_q, q_dyn, np.nan)))
            events.append((float(t_s[i_q]), "Max Q", "tab:red"))
        finite_stage = np.isfinite(stage)
        if np.any(finite_stage):
            for idx in np.flatnonzero(np.diff(stage[finite_stage]) > 0.5):
                event_t = float(t_s[np.flatnonzero(finite_stage)[idx + 1]])
                events.append((event_t, "Stage event", "tab:purple"))
        insertion_mask = np.zeros(t_s.size, dtype=bool)
        if np.isfinite(target_alt_km) and np.isfinite(alt_tol_km) and np.isfinite(ecc_max):
            altitude_ok = np.abs(alt_km - target_alt_km) <= alt_tol_km
            orbit_ok = np.isfinite(apo) & np.isfinite(peri) & (ecc <= ecc_max)
            insertion_mask = altitude_ok & orbit_ok
        insertion_t = _first_true_time(t_s, insertion_mask)
        if insertion_t is not None:
            events.append((insertion_t, "Insertion band", "tab:orange"))
        if t_s.size:
            events.append((float(t_s[-1]), "Final sample", "tab:gray"))

        fig, ax = plt.subplots(figsize=cap_figsize(11, 3.8))
        ax.axhline(0.0, color="0.3", linewidth=1.4)
        for i, (event_t, label, color) in enumerate(sorted(events, key=lambda row: row[0])):
            offset = 0.34 if i % 2 == 0 else -0.34
            ax.vlines(event_t, 0.0, offset, color=color, linewidth=1.6)
            ax.scatter([event_t], [0.0], color=color, s=42, zorder=3)
            ax.text(event_t, offset, label, ha="center", va="bottom" if offset > 0 else "top", fontsize=8)
        ax.set_xlabel("Time (s)")
        ax.set_yticks([])
        ax.set_ylim(-0.9, 0.9)
        ax.set_title("Rocket Mission Timeline")
        ax.grid(True, axis="x", alpha=0.25)
        fig.tight_layout()
        p = outdir / "rocket_mission_timeline.png"
        if mode in ("save", "both"):
            save_oel_figure(fig, p, dpi=int(cfg.outputs.plots.get("dpi", 150)), artifact_id=p.stem)
            out["rocket_mission_timeline"] = str(p)
        if mode in ("interactive", "both"):
            plt.show(block=False)
        else:
            plt.close(fig)

    if "rocket_downrange_altitude" in figure_ids and "rocket" in truth_hist:
        import matplotlib.pyplot as plt

        x = truth_hist["rocket"]
        launch_site = _rocket_launch_site(cfg)
        lat, lon, _ = ground_track_from_eci_history(
            x[:, 0:3],
            t_s=t_s[: x.shape[0]],
            jd_utc_start=cfg.simulator.initial_jd_utc,
            frame_context=frame_context,
        )
        if launch_site is None:
            lat0, lon0 = float(lat[0]), float(lon[0])
        else:
            lat0, lon0 = launch_site
        downrange_km = _haversine_distance_km(lat0, lon0, lat, lon)
        alt_km = _rocket_metric_array(rocket_metrics, "altitude_km", t_s.size)
        if not np.any(np.isfinite(alt_km)):
            alt_km = np.linalg.norm(x[:, 0:3], axis=1) - EARTH_RADIUS_KM
        speed = _rocket_metric_array(rocket_metrics, "speed_km_s", t_s.size)

        fig, ax = plt.subplots(figsize=cap_figsize(10, 5.5))
        n = min(downrange_km.size, alt_km.size, t_s.size)
        if n > 0 and np.any(np.isfinite(downrange_km[:n]) & np.isfinite(alt_km[:n])):
            if np.any(np.isfinite(speed[:n])):
                sc = ax.scatter(downrange_km[:n], alt_km[:n], c=speed[:n], s=9, cmap="viridis")
                fig.colorbar(sc, ax=ax, label="speed (km/s)")
            else:
                ax.plot(downrange_km[:n], alt_km[:n], linewidth=1.5)
            ax.scatter([downrange_km[0]], [alt_km[0]], color="tab:green", s=35, label="start")
            ax.scatter([downrange_km[n - 1]], [alt_km[n - 1]], color="tab:red", s=35, label="final")
            ax.legend(loc="best")
        else:
            ax.text(0.5, 0.5, "No valid downrange/altitude samples", transform=ax.transAxes, ha="center")
        ax.set_xlabel("Downrange distance (km)")
        ax.set_ylabel("Altitude (km)")
        ax.set_title("Rocket Altitude vs Downrange Distance")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = outdir / "rocket_downrange_altitude.png"
        if mode in ("save", "both"):
            save_oel_figure(fig, p, dpi=int(cfg.outputs.plots.get("dpi", 150)), artifact_id=p.stem)
            out["rocket_downrange_altitude"] = str(p)
        if mode in ("interactive", "both"):
            plt.show(block=False)
        else:
            plt.close(fig)

    if "rocket_maxq_throttle" in figure_ids and "rocket" in truth_hist:
        import matplotlib.pyplot as plt

        q_dyn = _rocket_metric_array(rocket_metrics, "q_dyn_pa", t_s.size, 0.0)
        throttle = _rocket_metric_array(rocket_metrics, "throttle_cmd", t_s.size, 0.0)
        mach = _rocket_metric_array(rocket_metrics, "mach", t_s.size, 0.0)
        alt_km = _rocket_metric_array(rocket_metrics, "altitude_km", t_s.size)
        fig, axes = plt.subplots(3, 1, figsize=cap_figsize(11, 8), sharex=True)
        axes[0].plot(t_s, q_dyn, color="tab:red", label="dynamic pressure")
        max_q_cfg = None
        for modifier in list(getattr(cfg.rocket, "guidance_modifiers", []) or []):
            params = dict(getattr(modifier, "params", {}) or {})
            if params.get("max_q_pa") is not None:
                max_q_cfg = float(params.get("max_q_pa"))
                break
        if max_q_cfg is not None:
            axes[0].axhline(max_q_cfg, color="black", linestyle="--", label=f"limit {max_q_cfg:.0f} Pa")
        if np.any(np.isfinite(q_dyn)):
            i_q = int(np.nanargmax(np.where(np.isfinite(q_dyn), q_dyn, np.nan)))
            axes[0].axvline(t_s[i_q], color="tab:red", linestyle=":", alpha=0.8)
            axes[0].text(t_s[i_q], q_dyn[i_q], " max Q", fontsize=8, va="bottom")
        axes[0].set_ylabel("q (Pa)")
        axes[0].set_title("Max-Q Throttle Limiting")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="best")

        axes[1].plot(t_s, throttle, color="tab:blue", label="throttle")
        axes[1].set_ylabel("throttle")
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="best")

        axes[2].plot(t_s, mach, color="tab:purple", label="Mach")
        if np.any(np.isfinite(alt_km)):
            ax_alt = axes[2].twinx()
            ax_alt.plot(t_s, alt_km, color="tab:gray", alpha=0.7, label="altitude")
            ax_alt.set_ylabel("altitude (km)")
        axes[2].set_xlabel("Time (s)")
        axes[2].set_ylabel("Mach")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc="best")
        fig.tight_layout()
        p = outdir / "rocket_maxq_throttle.png"
        if mode in ("save", "both"):
            save_oel_figure(fig, p, dpi=int(cfg.outputs.plots.get("dpi", 150)), artifact_id=p.stem)
            out["rocket_maxq_throttle"] = str(p)
        if mode in ("interactive", "both"):
            plt.show(block=False)
        else:
            plt.close(fig)

    if "rocket_tvc_aero_authority" in figure_ids and "rocket" in truth_hist:
        import matplotlib.pyplot as plt

        tvc = _rocket_metric_array(rocket_metrics, "tvc_gimbal_deg", t_s.size, 0.0)
        alpha = _rocket_metric_array(rocket_metrics, "alpha_deg", t_s.size, 0.0)
        beta = _rocket_metric_array(rocket_metrics, "beta_deg", t_s.size, 0.0)
        aero_force = _rocket_metric_array(rocket_metrics, "aero_force_n", t_s.size, 0.0)
        aero_moment = _rocket_metric_array(rocket_metrics, "aero_moment_nm", t_s.size, 0.0)
        q_dyn = _rocket_metric_array(rocket_metrics, "q_dyn_pa", t_s.size, 0.0)
        twr = _rocket_metric_array(rocket_metrics, "thrust_to_weight", t_s.size)

        fig, axes = plt.subplots(4, 1, figsize=cap_figsize(11, 10), sharex=True)
        axes[0].plot(t_s, tvc, color="tab:green", label="TVC gimbal")
        tvc_limit = float(dict(cfg.simulator.dynamics.rocket).get("tvc_max_gimbal_deg", np.nan))
        if np.isfinite(tvc_limit):
            axes[0].axhline(tvc_limit, color="black", linestyle="--", linewidth=0.9)
            axes[0].axhline(-tvc_limit, color="black", linestyle="--", linewidth=0.9)
        axes[0].set_ylabel("deg")
        axes[0].set_title("TVC and Aero Authority")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="best")

        axes[1].plot(t_s, alpha, label="alpha", color="tab:red")
        axes[1].plot(t_s, beta, label="beta", color="tab:purple")
        axes[1].set_ylabel("deg")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="best")

        ax_force = axes[2].twinx()
        l0 = axes[2].plot(t_s, aero_force, label="aero force", color="tab:orange")
        l1 = ax_force.plot(t_s, aero_moment, label="aero moment", color="tab:brown")
        axes[2].set_ylabel("force (N)")
        ax_force.set_ylabel("moment (N-m)")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(l0 + l1, [ln.get_label() for ln in l0 + l1], loc="best")

        ax_twr = axes[3].twinx()
        l2 = axes[3].plot(t_s, q_dyn, label="q", color="tab:blue")
        l3 = ax_twr.plot(t_s, twr, label="T/W", color="tab:gray")
        axes[3].set_xlabel("Time (s)")
        axes[3].set_ylabel("q (Pa)")
        ax_twr.set_ylabel("T/W")
        axes[3].grid(True, alpha=0.3)
        axes[3].legend(l2 + l3, [ln.get_label() for ln in l2 + l3], loc="best")
        fig.tight_layout()
        p = outdir / "rocket_tvc_aero_authority.png"
        if mode in ("save", "both"):
            save_oel_figure(fig, p, dpi=int(cfg.outputs.plots.get("dpi", 150)), artifact_id=p.stem)
            out["rocket_tvc_aero_authority"] = str(p)
        if mode in ("interactive", "both"):
            plt.show(block=False)
        else:
            plt.close(fig)

    if "rocket_insertion_scorecard" in figure_ids and "rocket" in truth_hist:
        import matplotlib.pyplot as plt

        target_alt_km, alt_tol_km, ecc_max = _rocket_target_altitude_cfg(cfg)
        final_alt = _last_finite_value(_rocket_metric_array(rocket_metrics, "altitude_km", t_s.size))
        final_apo = _last_finite_value(_rocket_metric_array(rocket_metrics, "apoapsis_alt_km", t_s.size))
        final_peri = _last_finite_value(_rocket_metric_array(rocket_metrics, "periapsis_alt_km", t_s.size))
        final_ecc = _last_finite_value(_rocket_metric_array(rocket_metrics, "eccentricity", t_s.size))
        prop_frac = _last_finite_value(
            _rocket_metric_array(rocket_metrics, "propellant_remaining_fraction", t_s.size)
        )
        max_q = _max_finite_value(_rocket_metric_array(rocket_metrics, "q_dyn_pa", t_s.size))
        max_alpha = _max_abs_finite_value(_rocket_metric_array(rocket_metrics, "alpha_deg", t_s.size))
        max_tvc = _max_finite_value(_rocket_metric_array(rocket_metrics, "tvc_gimbal_deg", t_s.size))
        max_force = _max_finite_value(_rocket_metric_array(rocket_metrics, "aero_force_n", t_s.size))
        metrics_rows = [
            ("Final altitude", final_alt, "km", target_alt_km, alt_tol_km),
            ("Final apogee", final_apo, "km", target_alt_km, alt_tol_km),
            ("Final perigee", final_peri, "km", target_alt_km, alt_tol_km),
            ("Final eccentricity", final_ecc, "", ecc_max, None),
            ("Propellant remaining", prop_frac, "fraction", None, None),
            ("Max dynamic pressure", max_q, "Pa", None, None),
            ("Max |alpha|", max_alpha, "deg", None, None),
            ("Max TVC gimbal", max_tvc, "deg", float(dict(cfg.simulator.dynamics.rocket).get("tvc_max_gimbal_deg", np.nan)), None),
            ("Max aero force", max_force, "N", None, None),
        ]
        fig, ax = plt.subplots(figsize=cap_figsize(10, 5.8))
        ax.axis("off")
        title = "Rocket Insertion Scorecard"
        if np.isfinite(target_alt_km):
            title += f" (target {target_alt_km:.0f} km)"
        ax.set_title(title, fontsize=14, pad=16)
        table_data = []
        row_colors = []
        for name, value, unit, target, tol in metrics_rows:
            value_txt = "n/a" if not np.isfinite(value) else f"{value:.3g}"
            if unit:
                value_txt = f"{value_txt} {unit}"
            target_txt = ""
            passed = None
            if target is not None and np.isfinite(float(target)):
                if tol is not None and np.isfinite(float(tol)):
                    target_txt = f"{float(target):.3g} +/- {float(tol):.3g}"
                    passed = bool(np.isfinite(value) and abs(value - float(target)) <= float(tol))
                elif name == "Final eccentricity":
                    target_txt = f"<= {float(target):.3g}"
                    passed = bool(np.isfinite(value) and value <= float(target))
                elif "TVC" in name:
                    target_txt = f"<= {float(target):.3g}"
                    passed = bool(np.isfinite(value) and value <= float(target))
            status = "OK" if passed is True else ("Check" if passed is False else "")
            table_data.append([name, value_txt, target_txt, status])
            row_colors.append("#eaf6ea" if passed is True else ("#fdeaea" if passed is False else "#f7f7f7"))
        table = ax.table(
            cellText=table_data,
            colLabels=["Metric", "Value", "Target / Limit", "Status"],
            loc="center",
            cellLoc="left",
            colLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.35)
        for row_idx, color in enumerate(row_colors, start=1):
            for col_idx in range(4):
                table[(row_idx, col_idx)].set_facecolor(color)
        for col_idx in range(4):
            table[(0, col_idx)].set_facecolor("#d9e8f5")
            table[(0, col_idx)].set_text_props(weight="bold")
        fig.tight_layout()
        p = outdir / "rocket_insertion_scorecard.png"
        if mode in ("save", "both"):
            save_oel_figure(fig, p, dpi=int(cfg.outputs.plots.get("dpi", 150)), artifact_id=p.stem)
            out["rocket_insertion_scorecard"] = str(p)
        if mode in ("interactive", "both"):
            plt.show(block=False)
        else:
            plt.close(fig)

    satellite_dv_by_object = _compute_satellite_delta_v_remaining(
        cfg=cfg,
        truth_hist=truth_hist,
        resolve_satellite_isp_s=resolve_satellite_isp_s,
    )

    if "satellite_delta_v_remaining" in figure_ids:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=cap_figsize(10, 5))
        plotted = False
        for oid in sorted(satellite_dv_by_object.keys()):
            dv_entry = satellite_dv_by_object.get(oid)
            if dv_entry is None:
                continue
            dv0_m_s = float(dv_entry["initial_m_s"])
            dv_rem_m_s = np.array(dv_entry["remaining_m_s"], dtype=float)
            pct = np.clip(100.0 * dv_rem_m_s / dv0_m_s, 0.0, 100.0)
            ax.plot(t_s[: pct.size], pct, label=f"{oid}")
            plotted = True

        if plotted:
            ax.set_ylim(-1.0, 101.0)
            ax.set_xlabel("time (s)")
            ax.set_ylabel("Delta-V Remaining (%)")
            ax.set_title("Satellite Delta-V Remaining")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
            fig.tight_layout()
            p = outdir / "satellite_delta_v_remaining.png"
            if mode in ("save", "both"):
                save_oel_figure(fig, p, dpi=int(cfg.outputs.plots.get("dpi", 150)), artifact_id=p.stem)
                out["satellite_delta_v_remaining"] = str(p)
            if mode == "save":
                plt.close(fig)
        else:
            plt.close(fig)

    thrust_hist_ric: dict[str, np.ndarray] = {}
    if ("control_thrust_ric" in figure_ids) or ("control_thrust_ric_multi" in figure_ids):
        for oid, u in thrust_hist.items():
            hist = truth_hist.get(oid)
            if hist is None or hist.size == 0:
                continue
            n_s = min(u.shape[0], hist.shape[0], t_s.size)
            ur = np.full((u.shape[0], 3), np.nan, dtype=float)
            for k in range(n_s):
                a_eci = np.array(u[k, :], dtype=float)
                rv = np.array(hist[k, 0:6], dtype=float)
                if not (np.all(np.isfinite(a_eci)) and np.all(np.isfinite(rv))):
                    continue
                c_ir = ric_dcm_ir_from_rv(rv[:3], rv[3:6])
                ur[k, :] = c_ir.T @ a_eci
            thrust_hist_ric[oid] = ur

    if "control_thrust" in figure_ids:
        for oid, u in thrust_hist.items():
            if not np.any(np.isfinite(u[:, 0])):
                continue
            p = outdir / f"{oid}_control_thrust.png"
            plot_control_commands(
                t_s,
                u,
                layout="subplots",
                input_labels=["ax", "ay", "az"],
                title=f"Thrust Commands ({oid})",
                y_label="km/s^2",
                mode=mode,
                out_path=str(p),
            )
            if mode in ("save", "both"):
                out[f"{oid}_control_thrust"] = str(p)

    if "control_thrust_ric" in figure_ids:
        for oid, u in thrust_hist_ric.items():
            if not np.any(np.isfinite(u[:, 0])):
                continue
            p = outdir / f"{oid}_control_thrust_ric.png"
            plot_control_commands(
                t_s,
                u,
                layout="subplots",
                input_labels=["aR", "aI", "aC"],
                title=f"Thrust Commands RIC ({oid})",
                y_label="km/s^2",
                mode=mode,
                out_path=str(p),
            )
            if mode in ("save", "both"):
                out[f"{oid}_control_thrust_ric"] = str(p)

    if "control_thrust_multi" in figure_ids:
        for i_comp, lbl in enumerate(("ax", "ay", "az")):
            p = outdir / f"control_thrust_multi_{lbl}.png"
            plot_multi_control_commands(
                t_s,
                thrust_hist,
                component_index=i_comp,
                title=f"Thrust Command Overlay ({lbl})",
                y_label="km/s^2",
                mode=mode,
                out_path=str(p),
            )
            if mode in ("save", "both"):
                out[f"control_thrust_multi_{lbl}"] = str(p)

    if "control_thrust_ric_multi" in figure_ids:
        for i_comp, lbl in enumerate(("aR", "aI", "aC")):
            p = outdir / f"control_thrust_ric_multi_{lbl}.png"
            plot_multi_control_commands(
                t_s,
                thrust_hist_ric,
                component_index=i_comp,
                title=f"Thrust Command Overlay RIC ({lbl})",
                y_label="km/s^2",
                mode=mode,
                out_path=str(p),
            )
            if mode in ("save", "both"):
                out[f"control_thrust_ric_multi_{lbl}"] = str(p)

    if "thrust_alignment_error" in figure_ids:
        import matplotlib.pyplot as plt

        thrust_dir_by_object = _thruster_direction_body_by_object(cfg)

        for oid, hist in truth_hist.items():
            u = thrust_hist.get(oid)
            if u is None or hist.size == 0:
                continue
            thrust_norm = np.linalg.norm(np.nan_to_num(u, nan=0.0), axis=1)
            if not np.any(thrust_norm > 1e-15):
                continue
            err_deg = _thrust_alignment_error_deg_series(
                t_s=t_s,
                truth_hist=hist,
                thrust_hist=u,
                thruster_direction_body=thrust_dir_by_object.get(oid, np.array([1.0, 0.0, 0.0], dtype=float)),
            )

            fig, ax = plt.subplots(figsize=cap_figsize(10, 5))
            finite = np.isfinite(err_deg)
            if np.any(finite):
                t_f = np.array(t_s[finite], dtype=float)
                e_f = np.array(err_deg[finite], dtype=float)
                ax.plot(t_f, e_f, linewidth=1.2, marker="o", markersize=2.5)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No valid burn/alignment samples in this run",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                )
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Angle Error (deg)")
            ax.set_title(f"Attitude vs Thrust Vector Error ({oid})")
            ax.grid(True, alpha=0.3)
            p = outdir / f"{oid}_thrust_alignment_error.png"
            if mode in ("save", "both"):
                save_oel_figure(fig, p, dpi=int(cfg.outputs.plots.get("dpi", 150)), artifact_id=p.stem)
                out[f"{oid}_thrust_alignment_error"] = str(p)
            if mode in ("interactive", "both"):
                plt.show(block=False)
            else:
                plt.close(fig)

    if "knowledge_timeline" in figure_ids:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=cap_figsize(10, 5))
        i = 0
        for obs, by_tgt in knowledge_hist.items():
            for tgt, hist in by_tgt.items():
                known = np.any(np.isfinite(hist), axis=1).astype(float)
                ax.plot(t_s, known + i * 1.2, label=f"{obs}->{tgt}")
                i += 1
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Known (offset)")
        ax.set_title("Knowledge Timeline")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        p = outdir / "knowledge_timeline.png"
        if mode in ("save", "both"):
            save_oel_figure(fig, p, dpi=int(cfg.outputs.plots.get("dpi", 150)), artifact_id=p.stem)
            out["knowledge_timeline"] = str(p)
        if mode in ("interactive", "both"):
            plt.show(block=False)
        else:
            plt.close(fig)

    return out


def animate_outputs(
    *,
    cfg: SimulationScenarioConfig,
    t_s: np.ndarray,
    truth_hist: dict[str, np.ndarray],
    thrust_hist: dict[str, np.ndarray],
    target_reference_orbit_truth: np.ndarray | None,
    outdir: Path,
    resolve_satellite_isp_s: Callable[[dict[str, Any]], float],
) -> dict[str, str]:
    anim_cfg = dict(cfg.outputs.animations or {})
    if not bool(anim_cfg.get("enabled", False)):
        return {}
    plot_style_name = style_name_from_config(dict(cfg.outputs.plots or {}))
    style_name = str(anim_cfg.get("style", plot_style_name) or plot_style_name).strip().lower()
    metadata = artifact_metadata(scenario_name=str(getattr(cfg, "scenario_name", "") or ""))
    with oel_plot_context(style_name=style_name, metadata=metadata):
        return _animate_outputs_impl(
            cfg=cfg,
            t_s=t_s,
            truth_hist=truth_hist,
            thrust_hist=thrust_hist,
            target_reference_orbit_truth=target_reference_orbit_truth,
            outdir=outdir,
            resolve_satellite_isp_s=resolve_satellite_isp_s,
        )


def _animate_outputs_impl(
    *,
    cfg: SimulationScenarioConfig,
    t_s: np.ndarray,
    truth_hist: dict[str, np.ndarray],
    thrust_hist: dict[str, np.ndarray],
    target_reference_orbit_truth: np.ndarray | None,
    outdir: Path,
    resolve_satellite_isp_s: Callable[[dict[str, Any]], float],
) -> dict[str, str]:
    out: dict[str, str] = {}
    anim_cfg = dict(cfg.outputs.animations or {})
    if not bool(anim_cfg.get("enabled", False)):
        return out

    mode = cfg.outputs.mode
    fps = float(anim_cfg.get("fps", 30.0))
    speed_multiple = float(anim_cfg.get("speed_multiple", 10.0))
    frame_stride = int(anim_cfg.get("frame_stride", 1))
    draw_earth_map = bool(anim_cfg.get("draw_earth_map", True))
    types = list(anim_cfg.get("types", []) or [])
    if not types:
        return out
    frame_context = frame_context_from_mapping(
        dict(getattr(cfg.simulator, "frames", {}) or {}),
        jd_utc_start=cfg.simulator.initial_jd_utc,
        source="scenario",
    )
    plot_fns = _load_plotting_functions()
    animate_battlespace_dashboard = plot_fns["animate_battlespace_dashboard"]
    animate_rectangular_prism_attitude = plot_fns["animate_rectangular_prism_attitude"]
    animate_ground_track = plot_fns["animate_ground_track"]
    animate_multi_ric_2d_projections = plot_fns["animate_multi_ric_2d_projections"]
    animate_multi_ground_track = plot_fns["animate_multi_ground_track"]
    animate_multi_trajectory_frame = plot_fns["animate_multi_trajectory_frame"]
    animate_multi_rectangular_prism_ric_curv = plot_fns["animate_multi_rectangular_prism_ric_curv"]
    animate_side_by_side_rectangular_prism_ric_attitude = plot_fns[
        "animate_side_by_side_rectangular_prism_ric_attitude"
    ]
    satellite_dv_by_object = _compute_satellite_delta_v_remaining(
        cfg=cfg,
        truth_hist=truth_hist,
        resolve_satellite_isp_s=resolve_satellite_isp_s,
    )

    if "attitude_ric_thruster" in types:
        dims_map_raw = anim_cfg.get("attitude_ric_thruster_dims_m", {})
        dims_map = dict(dims_map_raw) if isinstance(dims_map_raw, dict) else {}
        thruster_mounts = _thruster_mounts_by_object(cfg)
        object_ids = anim_cfg.get("attitude_ric_thruster_object_ids")
        if isinstance(object_ids, list):
            attitude_object_ids = [str(oid) for oid in object_ids if str(oid) in truth_hist]
        else:
            attitude_object_ids = sorted(truth_hist.keys())
        active_threshold = float(anim_cfg.get("attitude_ric_thruster_active_threshold_km_s2", 1e-15))
        default_dims_m = np.array([4.0, 2.0, 2.0], dtype=float)
        for oid in attitude_object_ids:
            hist = np.array(truth_hist.get(oid, np.array([])), dtype=float)
            if hist.ndim != 2 or hist.shape[0] == 0 or not np.any(np.isfinite(hist[:, 0])):
                continue
            dims = np.array(dims_map.get(oid, default_dims_m), dtype=float).reshape(-1)
            if dims.size != 3:
                dims = default_dims_m.copy()
            thrust = np.array(thrust_hist.get(oid, np.zeros((hist.shape[0], 3))), dtype=float)
            thrust_norm = (
                np.linalg.norm(np.nan_to_num(thrust, nan=0.0), axis=1)
                if thrust.ndim == 2
                else np.zeros(hist.shape[0], dtype=float)
            )
            active_mask = thrust_norm > active_threshold
            p = outdir / f"{oid}_attitude_ric_thruster.mp4"
            color_cycle = ["#1F77B4", "#D62728", "#2CA02C", "#9467BD", "#8C564B", "#17BECF"]
            body_facecolor = color_cycle[sum(ord(ch) for ch in oid) % len(color_cycle)]
            animate_rectangular_prism_attitude(
                t_s=t_s[: hist.shape[0]],
                truth_hist=hist,
                lx_m=float(dims[0]),
                ly_m=float(dims[1]),
                lz_m=float(dims[2]),
                frame="ric",
                thruster_active_mask=active_mask,
                thruster_position_body_m=None
                if thruster_mounts.get(oid) is None
                else thruster_mounts[oid]["position_body_m"],
                thruster_direction_body=None
                if thruster_mounts.get(oid) is None
                else thruster_mounts[oid]["direction_body"],
                body_facecolor=body_facecolor,
                thruster_inactive_facecolor="#808080",
                thruster_active_facecolor="#D95F02",
                mode=mode,
                out_path=str(p),
                fps=fps,
                speed_multiple=speed_multiple,
            )
            if mode in ("save", "both"):
                out[f"{oid}_attitude_ric_thruster"] = str(p)

    if "ground_track_multi" in types:
        p = outdir / "ground_track_multi.mp4"
        animate_multi_ground_track(
            t_s=t_s,
            truth_hist_by_object=truth_hist,
            jd_utc_start=cfg.simulator.initial_jd_utc,
            mode=mode,
            out_path=str(p),
            fps=fps,
            speed_multiple=speed_multiple,
            draw_earth_map=draw_earth_map,
            frame_stride=frame_stride,
            frame_context=frame_context,
        )
        if mode in ("save", "both"):
            out["ground_track_multi"] = str(p)

    if "ground_track" in types:
        for oid, hist in truth_hist.items():
            if hist.size == 0 or not np.any(np.isfinite(hist[:, 0])):
                continue
            lat_deg, lon_deg, _ = ground_track_from_eci_history(
                hist[:, :3],
                t_s=t_s,
                jd_utc_start=cfg.simulator.initial_jd_utc,
                frame_context=frame_context,
            )
            p = outdir / f"{oid}_ground_track.mp4"
            animate_ground_track(
                lon_deg=lon_deg,
                lat_deg=lat_deg,
                t_s=t_s,
                jd_utc_start=cfg.simulator.initial_jd_utc,
                mode=mode,
                out_path=str(p),
                fps=fps,
                speed_multiple=speed_multiple,
                draw_earth_map=draw_earth_map,
                frame_stride=frame_stride,
            )
            if mode in ("save", "both"):
                out[f"{oid}_ground_track"] = str(p)

    if "ric_curv_prism_multi" in types:
        p = outdir / "ric_curv_prism_multi.mp4"
        target_object_id = str(
            anim_cfg.get("target_object_id", default_reference_object_id(cfg, available_ids=truth_hist.keys()) or "")
        )
        prism_obj_ids = anim_cfg.get("ric_curv_prism_object_ids")
        if not isinstance(prism_obj_ids, list):
            prism_obj_ids = None
        dims_map_raw = anim_cfg.get("ric_curv_prism_dims_m", {})
        dims_map = dict(dims_map_raw) if isinstance(dims_map_raw, dict) else {}
        animate_multi_rectangular_prism_ric_curv(
            t_s=t_s,
            truth_hist_by_object=truth_hist,
            target_object_id=target_object_id,
            object_ids=prism_obj_ids,
            prism_dims_m_by_object=dims_map,
            mode=mode,
            out_path=str(p),
            fps=fps,
            speed_multiple=speed_multiple,
            frame_stride=frame_stride,
        )
        if mode in ("save", "both"):
            out["ric_curv_prism_multi"] = str(p)

    if "ric_prism_side_by_side" in types:
        p = outdir / "ric_prism_side_by_side.mp4"
        default_pair = default_pair_object_ids(cfg, available_ids=truth_hist.keys()) or ("", "")
        left_object_id = str(anim_cfg.get("ric_side_by_side_left_object_id", default_pair[1] or default_pair[0]))
        right_object_id = str(anim_cfg.get("ric_side_by_side_right_object_id", default_pair[0]))
        dims_map_raw = anim_cfg.get("ric_side_by_side_dims_m", {})
        dims_map = dict(dims_map_raw) if isinstance(dims_map_raw, dict) else {}
        animate_side_by_side_rectangular_prism_ric_attitude(
            t_s=t_s,
            truth_hist_by_object=truth_hist,
            left_object_id=left_object_id,
            right_object_id=right_object_id,
            prism_dims_m_by_object=dims_map,
            mode=mode,
            out_path=str(p),
            fps=fps,
            speed_multiple=speed_multiple,
            frame_stride=frame_stride,
        )
        if mode in ("save", "both"):
            out["ric_prism_side_by_side"] = str(p)

    reference_truth = None
    if target_reference_orbit_truth is not None:
        ref_arr = np.array(target_reference_orbit_truth, dtype=float)
        if ref_arr.ndim == 2 and ref_arr.shape[1] >= 6 and np.any(np.isfinite(ref_arr[:, 0])):
            reference_truth = ref_arr
    if reference_truth is not None:
        object_ids = anim_cfg.get("target_reference_ric_curv_object_ids")
        if isinstance(object_ids, list):
            ref_object_ids = [str(oid) for oid in object_ids if str(oid) in truth_hist]
        else:
            preferred_pair = default_pair_object_ids(cfg, available_ids=truth_hist.keys())
            ref_object_ids = [oid for oid in (preferred_pair or ()) if oid in truth_hist]
            if not ref_object_ids:
                ref_object_ids = sorted(truth_hist.keys())
        ref_truth_hist = {oid: truth_hist[oid] for oid in ref_object_ids}

        if "target_reference_ric_curv_3d" in types and ref_truth_hist:
            p = outdir / "target_reference_ric_curv_3d.mp4"
            animate_multi_trajectory_frame(
                t_s=t_s,
                truth_hist_by_object=ref_truth_hist,
                frame="ric_curv",
                reference_truth_hist=reference_truth,
                mode=mode,
                out_path=str(p),
                fps=fps,
                speed_multiple=speed_multiple,
                frame_stride=frame_stride,
                show_trajectory=bool(anim_cfg.get("target_reference_ric_curv_3d_show_trajectory", True)),
            )
            if mode in ("save", "both"):
                out["target_reference_ric_curv_3d"] = str(p)

        if "battlespace_dashboard" in types and ref_truth_hist:
            preferred_pair = default_pair_object_ids(cfg, available_ids=truth_hist.keys()) or ("", "")
            target_object_id = str(
                anim_cfg.get("battlespace_dashboard_target_object_id", preferred_pair[1] or preferred_pair[0])
            )
            chaser_object_id = str(anim_cfg.get("battlespace_dashboard_chaser_object_id", preferred_pair[0]))
            if target_object_id in truth_hist and chaser_object_id in truth_hist:
                p = outdir / "battlespace_dashboard.mp4"
                dims_map_raw = anim_cfg.get("battlespace_dashboard_attitude_dims_m", {})
                dims_map = dict(dims_map_raw) if isinstance(dims_map_raw, dict) else {}
                thruster_mounts = _thruster_mounts_by_object(cfg)
                animate_battlespace_dashboard(
                    t_s=t_s,
                    truth_hist_by_object=truth_hist,
                    reference_truth_hist=reference_truth,
                    target_object_id=target_object_id,
                    chaser_object_id=chaser_object_id,
                    thrust_hist_by_object=thrust_hist,
                    delta_v_remaining_m_s_by_object={
                        oid: np.array(entry["remaining_m_s"], dtype=float)
                        for oid, entry in satellite_dv_by_object.items()
                    },
                    prism_dims_m_by_object=dims_map,
                    thruster_mounts_by_object=thruster_mounts,
                    thruster_active_threshold_km_s2=float(
                        anim_cfg.get("battlespace_dashboard_thruster_active_threshold_km_s2", 1e-15)
                    ),
                    show_trajectory=bool(anim_cfg.get("battlespace_dashboard_show_trajectory", True)),
                    mode=mode,
                    out_path=str(p),
                    fps=fps,
                    speed_multiple=speed_multiple,
                    frame_stride=frame_stride,
                )
                if mode in ("save", "both"):
                    out["battlespace_dashboard"] = str(p)

        if "target_reference_ric_curv_2d" in types and ref_truth_hist:
            p = outdir / "target_reference_ric_curv_2d.mp4"
            animate_multi_ric_2d_projections(
                t_s=t_s,
                truth_hist_by_object=ref_truth_hist,
                frame="ric_curv",
                reference_truth_hist=reference_truth,
                planes=list(
                    anim_cfg.get("target_reference_ric_curv_2d_planes", ["ri", "ic", "rc"]) or ["ri", "ic", "rc"]
                ),
                mode=mode,
                out_path=str(p),
                fps=fps,
                speed_multiple=speed_multiple,
                frame_stride=frame_stride,
                show_trajectory=bool(anim_cfg.get("target_reference_ric_curv_2d_show_trajectory", True)),
            )
            if mode in ("save", "both"):
                out["target_reference_ric_curv_2d"] = str(p)

        per_plane_types = {
            "target_reference_ric_curv_2d_ri": "ri",
            "target_reference_ric_curv_2d_ic": "ic",
            "target_reference_ric_curv_2d_rc": "rc",
        }
        for anim_type, plane in per_plane_types.items():
            if anim_type not in types or not ref_truth_hist:
                continue
            p = outdir / f"{anim_type}.mp4"
            animate_multi_ric_2d_projections(
                t_s=t_s,
                truth_hist_by_object=ref_truth_hist,
                frame="ric_curv",
                reference_truth_hist=reference_truth,
                planes=[plane],
                mode=mode,
                out_path=str(p),
                fps=fps,
                speed_multiple=speed_multiple,
                frame_stride=frame_stride,
                show_trajectory=bool(anim_cfg.get(f"{anim_type}_show_trajectory", True)),
            )
            if mode in ("save", "both"):
                out[anim_type] = str(p)

    return out
