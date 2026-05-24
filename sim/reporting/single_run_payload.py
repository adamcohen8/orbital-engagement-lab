from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sim.config import SimulationScenarioConfig, default_pair_object_ids, default_reference_object_id
from sim.ground_stations import evaluate_ground_station_access


@dataclass(frozen=True)
class SingleRunPayloadContext:
    cfg: SimulationScenarioConfig
    object_ids: list[str]
    dt_s: float
    t_s: np.ndarray
    truth_hist: dict[str, np.ndarray]
    target_reference_orbit_truth: np.ndarray | None
    belief_hist: dict[str, np.ndarray]
    thrust_hist: dict[str, np.ndarray]
    torque_hist: dict[str, np.ndarray]
    desired_attitude_hist: dict[str, np.ndarray]
    knowledge_hist: dict[str, dict[str, np.ndarray]]
    bridge_hist: dict[str, list[dict[str, Any]]]
    controller_debug_hist: dict[str, list[dict[str, Any]]]
    rocket_throttle_cmd: np.ndarray
    rocket_metrics: dict[str, np.ndarray]
    thrust_stats: dict[str, dict[str, Any]]
    attitude_guardrail_stats: dict[str, int]
    knowledge_detection_by_observer: dict[str, Any]
    knowledge_consistency_by_observer: dict[str, Any]
    terminated_early: bool
    termination_reason: str | None
    termination_time_s: float | None
    termination_object_id: str | None
    rocket_inserted: bool
    rocket_insertion_time_s: float | None


def _summarize_actuator_diagnostics(controller_debug_hist: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for oid, rows in dict(controller_debug_hist or {}).items():
        object_summary: dict[str, Any] = {
            "actuator_stack_samples": 0,
            "fault_stuck_off_samples": 0,
            "rcs_active_samples": 0,
            "magnetorquer_active_samples": 0,
            "cmg_active_samples": 0,
            "wheel_desaturation_samples": 0,
            "max_gimbal_angle_rad": 0.0,
            "max_commanded_rcs_thruster_force_n": 0.0,
            "max_electric_propulsion_thrust_n": 0.0,
        }
        for row in list(rows or []):
            flags = dict(row.get("mode_flags", {}) or {})
            if bool(flags.get("actuator_stack_enabled", False)):
                object_summary["actuator_stack_samples"] += 1
            if bool(flags.get("actuator_fault_stuck_off", False)):
                object_summary["fault_stuck_off_samples"] += 1
            if flags.get("rcs_thruster_forces_n") is not None:
                forces = np.array(flags.get("rcs_thruster_forces_n", []), dtype=float).reshape(-1)
                finite = forces[np.isfinite(forces)]
                if finite.size and float(np.max(np.abs(finite))) > 1e-15:
                    object_summary["rcs_active_samples"] += 1
                    object_summary["max_commanded_rcs_thruster_force_n"] = max(
                        float(object_summary["max_commanded_rcs_thruster_force_n"]),
                        float(np.max(np.abs(finite))),
                    )
            if str(flags.get("magnetorquer_mode", "") or ""):
                object_summary["magnetorquer_active_samples"] += 1
            if flags.get("cmg_torque_body_nm") is not None:
                cmg = np.array(flags.get("cmg_torque_body_nm", []), dtype=float).reshape(-1)
                if cmg.size and float(np.linalg.norm(cmg)) > 1e-15:
                    object_summary["cmg_active_samples"] += 1
            if bool(flags.get("wheel_desaturation_active", False)):
                object_summary["wheel_desaturation_samples"] += 1
            if flags.get("gimbal_angle_rad") is not None:
                object_summary["max_gimbal_angle_rad"] = max(
                    float(object_summary["max_gimbal_angle_rad"]),
                    abs(float(flags.get("gimbal_angle_rad", 0.0))),
                )
            if flags.get("electric_propulsion_thrust_n") is not None:
                object_summary["max_electric_propulsion_thrust_n"] = max(
                    float(object_summary["max_electric_propulsion_thrust_n"]),
                    abs(float(flags.get("electric_propulsion_thrust_n", 0.0))),
                )
        if any(float(v) > 0.0 for v in object_summary.values()):
            summary[str(oid)] = object_summary
    return summary


def build_single_run_payload(context: SingleRunPayloadContext) -> dict[str, Any]:
    ground_station_access, ground_station_access_summary = evaluate_ground_station_access(
        ground_stations=list(context.cfg.ground_stations),
        t_s=context.t_s,
        truth_hist=context.truth_hist,
        jd_utc_start=context.cfg.simulator.initial_jd_utc,
    )
    reference_object_id = default_reference_object_id(context.cfg, available_ids=context.object_ids)
    primary_pair = default_pair_object_ids(context.cfg, available_ids=context.object_ids)
    summary = {
        "scenario_name": context.cfg.scenario_name,
        "scenario_description": context.cfg.scenario_description,
        "objects": sorted(str(item) for item in context.object_ids),
        "samples": int(context.t_s.size),
        "dt_s": float(context.dt_s),
        "duration_s": float(context.t_s[-1]) if context.t_s.size else 0.0,
        "terminated_early": bool(context.terminated_early),
        "termination_reason": context.termination_reason,
        "termination_time_s": context.termination_time_s,
        "termination_object_id": context.termination_object_id,
        "rocket_insertion_achieved": bool(context.rocket_inserted),
        "rocket_insertion_time_s": context.rocket_insertion_time_s,
        "target_reference_orbit_enabled": bool(context.target_reference_orbit_truth is not None),
        "reference_object_id": reference_object_id,
        "primary_object_pair": list(primary_pair) if primary_pair is not None else [],
        "thrust_stats": context.thrust_stats,
        "attitude_guardrail_stats": context.attitude_guardrail_stats,
        "knowledge_detection_by_observer": context.knowledge_detection_by_observer,
        "knowledge_consistency_by_observer": context.knowledge_consistency_by_observer,
        "ground_station_access_summary": ground_station_access_summary,
        "actuator_diagnostics_summary": _summarize_actuator_diagnostics(context.controller_debug_hist),
        "plot_outputs": {},
        "animation_outputs": {},
    }
    rocket_summary: dict[str, Any] = {}
    if context.rocket_metrics:
        def _last_finite(name: str) -> float | None:
            arr = np.array(context.rocket_metrics.get(name, []), dtype=float).reshape(-1)
            finite = arr[np.isfinite(arr)]
            return None if finite.size == 0 else float(finite[-1])

        def _max_finite(name: str) -> float | None:
            arr = np.array(context.rocket_metrics.get(name, []), dtype=float).reshape(-1)
            finite = arr[np.isfinite(arr)]
            return None if finite.size == 0 else float(np.max(finite))

        def _max_abs_finite(name: str) -> float | None:
            arr = np.array(context.rocket_metrics.get(name, []), dtype=float).reshape(-1)
            finite = arr[np.isfinite(arr)]
            return None if finite.size == 0 else float(np.max(np.abs(finite)))

        rocket_summary = {
            "final_altitude_km": _last_finite("altitude_km"),
            "final_speed_km_s": _last_finite("speed_km_s"),
            "final_apoapsis_alt_km": _last_finite("apoapsis_alt_km"),
            "final_periapsis_alt_km": _last_finite("periapsis_alt_km"),
            "final_eccentricity": _last_finite("eccentricity"),
            "final_propellant_remaining_fraction": _last_finite("propellant_remaining_fraction"),
            "max_dynamic_pressure_pa": _max_finite("q_dyn_pa"),
            "max_mach": _max_finite("mach"),
            "max_abs_alpha_deg": _max_abs_finite("alpha_deg"),
            "max_tvc_gimbal_deg": _max_finite("tvc_gimbal_deg"),
            "max_aero_force_n": _max_finite("aero_force_n"),
            "max_aero_moment_nm": _max_finite("aero_moment_nm"),
        }
        summary["rocket_metrics_summary"] = rocket_summary
    return {
        "summary": summary,
        "time_s": context.t_s.tolist(),
        "truth_by_object": {k: v.tolist() for k, v in context.truth_hist.items()},
        "target_reference_orbit_truth": (
            [] if context.target_reference_orbit_truth is None else context.target_reference_orbit_truth.tolist()
        ),
        "belief_by_object": {k: v.tolist() for k, v in context.belief_hist.items()},
        "applied_thrust_by_object": {k: v.tolist() for k, v in context.thrust_hist.items()},
        "applied_torque_by_object": {k: v.tolist() for k, v in context.torque_hist.items()},
        "desired_attitude_by_object": {k: v.tolist() for k, v in context.desired_attitude_hist.items()},
        "knowledge_by_observer": {
            o: {t: a.tolist() for t, a in bt.items()} for o, bt in context.knowledge_hist.items()
        },
        "knowledge_detection_by_observer": dict(context.knowledge_detection_by_observer),
        "knowledge_consistency_by_observer": dict(context.knowledge_consistency_by_observer),
        "ground_station_access": ground_station_access,
        "ground_station_access_summary": ground_station_access_summary,
        "bridge_events_by_object": context.bridge_hist,
        "controller_debug_by_object": context.controller_debug_hist,
        "rocket_throttle_cmd": context.rocket_throttle_cmd.tolist(),
        "rocket_metrics": {k: v.tolist() for k, v in context.rocket_metrics.items()},
    }
