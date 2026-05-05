from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sim.config import SimulationScenarioConfig
from sim.master_outputs import animate_outputs as _animate_outputs_impl
from sim.master_outputs import plot_outputs as _plot_outputs_impl
from sim.reporting.output_index import write_output_index
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
    rocket_metrics: dict[str, np.ndarray]
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
    plot_outputs = _plot_outputs(
        cfg=context.cfg,
        t_s=context.t_s,
        truth_hist=context.truth_hist,
        target_reference_orbit_truth=context.target_reference_orbit_truth,
        belief_hist=context.belief_hist,
        thrust_hist=context.thrust_hist,
        desired_attitude_hist=context.desired_attitude_hist,
        knowledge_hist=context.knowledge_hist,
        rocket_metrics=context.rocket_metrics if context.rocket_metrics else None,
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

    artifacts: dict[str, Any] = {}
    if bool(context.cfg.outputs.stats.get("save_json", True)):
        artifacts["summary_json"] = str(context.outdir / "master_run_summary.json")
    if bool(context.cfg.outputs.stats.get("save_full_log", True)):
        artifacts["run_log_json"] = str(context.outdir / "master_run_log.json")
    if plot_outputs:
        artifacts["plots"] = plot_outputs
    if animation_outputs:
        artifacts["animations"] = animation_outputs
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
    rocket_metrics: dict[str, np.ndarray] | None,
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
        rocket_metrics=rocket_metrics,
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
