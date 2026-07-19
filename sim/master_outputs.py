from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Callable

import numpy as np

from sim.config import SimulationScenarioConfig
from sim.plotting.output_animations import render_animations
from sim.plotting.output_context import build_plot_output_context
from sim.plotting.output_helpers import (
    _compute_satellite_delta_v_remaining as _compute_satellite_delta_v_remaining,
)
from sim.plotting.output_helpers import (
    _first_true_time as _first_true_time,
)
from sim.plotting.output_helpers import (
    _haversine_distance_km as _haversine_distance_km,
)
from sim.plotting.output_helpers import (
    _last_finite_value as _last_finite_value,
)
from sim.plotting.output_helpers import (
    _lift_axis_body_by_object as _lift_axis_body_by_object,
)
from sim.plotting.output_helpers import (
    _max_abs_finite_value as _max_abs_finite_value,
)
from sim.plotting.output_helpers import (
    _max_finite_value as _max_finite_value,
)
from sim.plotting.output_helpers import (
    _orbital_elements_basic as _orbital_elements_basic,
)
from sim.plotting.output_helpers import (
    _quat_error_angle_deg as _quat_error_angle_deg,
)
from sim.plotting.output_helpers import (
    _rocket_launch_site as _rocket_launch_site,
)
from sim.plotting.output_helpers import (
    _rocket_metric_array as _rocket_metric_array,
)
from sim.plotting.output_helpers import (
    _rocket_target_altitude_cfg as _rocket_target_altitude_cfg,
)
from sim.plotting.output_helpers import (
    _thrust_alignment_error_deg_series as _thrust_alignment_error_deg_series,
)
from sim.plotting.output_helpers import (
    _thruster_direction_body_by_object as _thruster_direction_body_by_object,
)
from sim.plotting.output_helpers import (
    _thruster_mounts_by_object as _thruster_mounts_by_object,
)
from sim.plotting.output_helpers import (
    _unit_vector_or_none as _unit_vector_or_none,
)
from sim.plotting.output_registry import render_plot_outputs
from sim.plotting.style import artifact_metadata, oel_plot_context, style_name_from_config
from sim.plotting.summary_outputs import _plot_private_bridge_outputs as _plot_private_bridge_outputs
from sim.presets.rockets import RocketStackPreset


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
    if not bool(cfg.outputs.plots.get("enabled", True)):
        return {}
    figure_ids = _expanded_figure_ids(dict(cfg.outputs.plots or {}))
    if not figure_ids:
        return {}
    context = build_plot_output_context(
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
        figure_ids=figure_ids,
        plot_fns=_load_plotting_functions(),
        belief_hist=belief_hist,
        knowledge_measurement_hist=knowledge_measurement_hist,
        bridge_hist=bridge_hist,
        reentry_metrics=reentry_metrics,
    )
    return render_plot_outputs(context)


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
    return render_animations(
        cfg=cfg,
        t_s=t_s,
        truth_hist=truth_hist,
        thrust_hist=thrust_hist,
        target_reference_orbit_truth=target_reference_orbit_truth,
        outdir=outdir,
        resolve_satellite_isp_s=resolve_satellite_isp_s,
        plot_fns=_load_plotting_functions(),
    )
