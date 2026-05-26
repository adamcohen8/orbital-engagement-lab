from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sim.config import SimulationScenarioConfig
from sim.master_outputs import animate_outputs as _animate_outputs_impl
from sim.master_outputs import plot_outputs as _plot_outputs_impl
from sim.reporting.ground_station_access_reports import write_ground_station_access_reports
from sim.reporting.output_index import write_output_index
from sim.reporting.review_store import write_single_run_review_store
from sim.runtime_support import _resolve_rocket_stack, _resolve_satellite_isp_s
from sim.utils.io import write_json


@dataclass(frozen=True)
class SingleRunArtifactContext:
    cfg: SimulationScenarioConfig
    outdir: Path
    t_s: np.ndarray
    truth_hist: dict[str, np.ndarray]
    target_reference_orbit_truth: np.ndarray | None
    belief_hist: dict[str, np.ndarray]
    thrust_hist: dict[str, np.ndarray]
    desired_attitude_hist: dict[str, np.ndarray]
    knowledge_hist: dict[str, dict[str, np.ndarray]]
    knowledge_measurement_hist: dict[str, dict[str, np.ndarray]]
    rocket_metrics: dict[str, np.ndarray]
    reentry_metrics: dict[str, dict[str, np.ndarray]]
    bridge_hist: dict[str, list[dict[str, Any]]]


def format_single_run_summary(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    scenario_description = str(summary.get("scenario_description", "") or "").strip()
    lines.append("")
    lines.append("=" * 72)
    lines.append("MASTER SIMULATION SUMMARY")
    lines.append("=" * 72)
    lines.append(f"Scenario   : {summary.get('scenario_name', 'unknown')}")
    if scenario_description:
        lines.append(f"Desc       : {scenario_description}")
    objects = [str(item) for item in list(summary.get("objects", []) or [])]
    lines.append(f"Objects    : {', '.join(objects)}")
    lines.append(f"Samples    : {summary.get('samples', 0)}")
    lines.append(
        f"Timing     : dt={_fmt_float(float(summary.get('dt_s', 0.0)), 3)} s, "
        f"duration={_fmt_float(float(summary.get('duration_s', 0.0)), 1)} s"
    )
    lines.append("-" * 72)
    if bool(summary.get("terminated_early", False)):
        lines.append(
            "Termination: EARLY "
            f"(reason={summary.get('termination_reason')}, "
            f"t={summary.get('termination_time_s')}, "
            f"object={summary.get('termination_object_id')})"
        )
    else:
        lines.append("Termination: nominal (full duration reached)")
    if "rocket" in objects and "rocket_insertion_achieved" in summary:
        ins_ok = bool(summary.get("rocket_insertion_achieved", False))
        ins_t = summary.get("rocket_insertion_time_s")
        lines.append(f"Insertion  : achieved at t={ins_t}" if ins_ok else "Insertion  : not achieved")
    reentry_summary = dict(summary.get("reentry_summary_by_object", {}) or {})
    if reentry_summary:
        entered = [oid for oid, item in reentry_summary.items() if bool(dict(item or {}).get("entered_reentry"))]
        lines.append(
            "Re-entry   : "
            + (", ".join(sorted(entered)) if entered else "configured, threshold not crossed")
        )
    thrust_stats = dict(summary.get("thrust_stats", {}) or {})
    if thrust_stats:
        lines.append("-" * 72)
        lines.append("Thrust Stats")
        lines.append(f"{'Object':<14}{'Burn Samples':>14}{'Max Accel (km/s^2)':>24}{'Total dV (m/s)':>18}")
        for oid in sorted(thrust_stats.keys()):
            stats = dict(thrust_stats.get(oid, {}) or {})
            lines.append(
                f"{oid:<14}"
                f"{int(stats.get('burn_samples', 0)):>14d}"
                f"{float(stats.get('max_accel_km_s2', 0.0)):>24.3e}"
                f"{float(stats.get('total_dv_m_s', 0.0)):>18.3f}"
            )
    plot_outputs = dict(summary.get("plot_outputs", {}) or {})
    anim_outputs = dict(summary.get("animation_outputs", {}) or {})
    guardrails = dict(summary.get("attitude_guardrail_stats", {}) or {})
    lines.append("-" * 72)
    lines.append(f"Artifacts  : plots={len(plot_outputs)}  animations={len(anim_outputs)}")
    lines.append(f"Guardrails : attitude_events={int(sum(int(v) for v in guardrails.values())) if guardrails else 0}")
    lines.append("=" * 72)
    return "\n".join(lines)


def write_single_run_artifacts(
    payload: dict[str, Any],
    context: SingleRunArtifactContext,
) -> dict[str, Any]:
    summary = payload.setdefault("summary", {})
    _add_relative_range_summary(summary=summary, context=context)
    plot_outputs = _plot_outputs(
        cfg=context.cfg,
        t_s=context.t_s,
        truth_hist=context.truth_hist,
        target_reference_orbit_truth=context.target_reference_orbit_truth,
        belief_hist=context.belief_hist,
        thrust_hist=context.thrust_hist,
        desired_attitude_hist=context.desired_attitude_hist,
        knowledge_hist=context.knowledge_hist,
        knowledge_measurement_hist=context.knowledge_measurement_hist,
        rocket_metrics=context.rocket_metrics if context.rocket_metrics else None,
        reentry_metrics=context.reentry_metrics,
        bridge_hist=context.bridge_hist,
        outdir=context.outdir,
    )
    animation_outputs = _animate_outputs(
        cfg=context.cfg,
        t_s=context.t_s,
        truth_hist=context.truth_hist,
        thrust_hist=context.thrust_hist,
        target_reference_orbit_truth=context.target_reference_orbit_truth,
        outdir=context.outdir,
    )
    summary["plot_outputs"] = plot_outputs
    summary["animation_outputs"] = animation_outputs
    access_report_paths, access_report_views = write_ground_station_access_reports(
        outdir=context.outdir,
        ground_station_access=dict(payload.get("ground_station_access", {}) or {}),
        ground_station_access_summary=dict(payload.get("ground_station_access_summary", {}) or {}),
        t_s=context.t_s,
        initial_jd_utc=context.cfg.simulator.initial_jd_utc,
    )
    if access_report_paths:
        summary["ground_station_access_report_epoch_utc"] = access_report_views.get("epoch_utc")
        summary["ground_station_access_report_epoch_jd_utc"] = access_report_views.get("epoch_jd_utc")
        summary["ground_station_access_report_outputs"] = access_report_paths
        payload["ground_station_access_report_views"] = access_report_views
    bridge_outputs, bridge_summary = _write_private_bridge_artifacts(outdir=context.outdir, bridge_hist=context.bridge_hist)
    if bridge_outputs:
        summary["bridge_extension_outputs"] = bridge_outputs
        summary["bridge_extension_summary"] = bridge_summary

    artifacts: dict[str, Any] = {}
    if bool(context.cfg.outputs.stats.get("save_json", True)):
        artifacts["summary_json"] = str(context.outdir / "master_run_summary.json")
    if bool(context.cfg.outputs.stats.get("save_full_log", True)):
        artifacts["run_log_json"] = str(context.outdir / "master_run_log.json")
    if plot_outputs:
        artifacts["plots"] = plot_outputs
    if animation_outputs:
        artifacts["animations"] = animation_outputs
    if access_report_paths:
        artifacts["ground_station_access_reports"] = access_report_paths
    if bridge_outputs:
        artifacts["bridge_extensions"] = bridge_outputs
    review_outputs = _write_review_store(payload=payload, context=context, artifacts=artifacts)
    if review_outputs:
        summary["review_outputs"] = review_outputs
        payload["review_outputs"] = review_outputs
        artifacts["review_store"] = review_outputs
    index_path = write_output_index(
        outdir=context.outdir,
        workflow="single_run",
        title=str(context.cfg.scenario_name or "single_run"),
        summary=summary,
        artifacts=artifacts,
    )
    summary["output_index_md"] = str(index_path)
    payload["output_index_md"] = str(index_path)
    if bool(context.cfg.outputs.stats.get("save_json", True)):
        write_json(str(context.outdir / "master_run_summary.json"), summary)
    if bool(context.cfg.outputs.stats.get("save_full_log", True)):
        write_json(str(context.outdir / "master_run_log.json"), payload)
    if bool(context.cfg.outputs.stats.get("print_summary", True)):
        print(format_single_run_summary(summary))
    return payload


def _write_review_store(
    *,
    payload: dict[str, Any],
    context: SingleRunArtifactContext,
    artifacts: dict[str, Any],
) -> dict[str, str]:
    review_cfg = context.cfg.outputs.review
    if not bool(review_cfg.enabled):
        return {}
    try:
        return write_single_run_review_store(payload=payload, context=context, artifacts=artifacts)
    except Exception as exc:
        if bool(review_cfg.strict):
            raise
        status = f"failed:{type(exc).__name__}: {exc}"
        payload.setdefault("summary", {})["review_store_status"] = status
        return {}


def _write_private_bridge_artifacts(
    *,
    outdir: Path,
    bridge_hist: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, str], dict[str, Any]]:
    if not bridge_hist:
        return {}, {}
    module_name = ".".join(("integrations", "c" + "f" + "s" + "_sil", "artifacts"))
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        return {}, {}
    writer = getattr(module, "write_bridge_artifacts", None)
    if not callable(writer):
        return {}, {}
    return writer(outdir=outdir, bridge_hist=bridge_hist)


def _add_relative_range_summary(*, summary: dict[str, Any], context: SingleRunArtifactContext) -> None:
    pair = [str(item) for item in list(summary.get("primary_object_pair", []) or [])]
    if len(pair) != 2:
        return
    deputy_id, chief_id = pair
    deputy = context.truth_hist.get(deputy_id)
    chief = context.truth_hist.get(chief_id)
    if deputy is None or chief is None or deputy.ndim != 2 or chief.ndim != 2:
        return
    n = int(min(deputy.shape[0], chief.shape[0], context.t_s.size))
    if n <= 0 or deputy.shape[1] < 3 or chief.shape[1] < 3:
        return
    rel = np.asarray(deputy[:n, :3], dtype=float) - np.asarray(chief[:n, :3], dtype=float)
    ranges = np.linalg.norm(rel, axis=1)
    finite = np.isfinite(ranges)
    if not bool(np.any(finite)):
        return
    finite_indices = np.flatnonzero(finite)
    closest_local = int(np.argmin(ranges[finite]))
    closest_idx = int(finite_indices[closest_local])
    final_idx = int(finite_indices[-1])
    initial_idx = int(finite_indices[0])
    summary["relative_range_summary"] = {
        "object_pair": [deputy_id, chief_id],
        "initial_range_km": float(ranges[initial_idx]),
        "final_range_km": float(ranges[final_idx]),
        "closest_approach_km": float(ranges[closest_idx]),
        "closest_approach_time_s": float(context.t_s[closest_idx]),
    }


def _plot_outputs(
    *,
    cfg: SimulationScenarioConfig,
    t_s: np.ndarray,
    truth_hist: dict[str, np.ndarray],
    target_reference_orbit_truth: np.ndarray | None,
    belief_hist: dict[str, np.ndarray] | None,
    thrust_hist: dict[str, np.ndarray],
    desired_attitude_hist: dict[str, np.ndarray] | None,
    knowledge_hist: dict[str, dict[str, np.ndarray]],
    knowledge_measurement_hist: dict[str, dict[str, np.ndarray]],
    rocket_metrics: dict[str, np.ndarray] | None,
    reentry_metrics: dict[str, dict[str, np.ndarray]] | None,
    bridge_hist: dict[str, list[dict[str, Any]]] | None,
    outdir: Path,
) -> dict[str, str]:
    return _plot_outputs_impl(
        cfg=cfg,
        t_s=t_s,
        truth_hist=truth_hist,
        target_reference_orbit_truth=target_reference_orbit_truth,
        thrust_hist=thrust_hist,
        belief_hist=belief_hist,
        desired_attitude_hist=desired_attitude_hist,
        knowledge_hist=knowledge_hist,
        knowledge_measurement_hist=knowledge_measurement_hist,
        rocket_metrics=rocket_metrics,
        reentry_metrics=reentry_metrics or {},
        bridge_hist=bridge_hist,
        outdir=outdir,
        resolve_rocket_stack=_resolve_rocket_stack,
        resolve_satellite_isp_s=_resolve_satellite_isp_s,
    )


def _animate_outputs(
    *,
    cfg: SimulationScenarioConfig,
    t_s: np.ndarray,
    truth_hist: dict[str, np.ndarray],
    thrust_hist: dict[str, np.ndarray],
    target_reference_orbit_truth: np.ndarray | None,
    outdir: Path,
) -> dict[str, str]:
    return _animate_outputs_impl(
        cfg=cfg,
        t_s=t_s,
        truth_hist=truth_hist,
        thrust_hist=thrust_hist,
        target_reference_orbit_truth=target_reference_orbit_truth,
        outdir=outdir,
        resolve_satellite_isp_s=_resolve_satellite_isp_s,
    )


def _fmt_float(x: float, digits: int = 3) -> str:
    return f"{float(x):.{digits}f}"
