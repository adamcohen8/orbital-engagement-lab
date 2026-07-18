from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from sim.config import SimulationScenarioConfig, default_reference_object_id
from sim.dynamics.orbit.frames import FrameContext, frame_context_from_mapping
from sim.presets.rockets import RocketStackPreset


@dataclass(frozen=True)
class PlotOutputContext:
    """Inputs and run-scoped decisions shared by output renderer families."""

    cfg: SimulationScenarioConfig
    t_s: np.ndarray
    truth_hist: dict[str, np.ndarray]
    target_reference_orbit_truth: np.ndarray | None
    thrust_hist: dict[str, np.ndarray]
    desired_attitude_hist: dict[str, np.ndarray] | None
    knowledge_hist: dict[str, dict[str, np.ndarray]]
    rocket_metrics: dict[str, np.ndarray] | None
    outdir: Path
    resolve_rocket_stack: Callable[[dict[str, Any]], RocketStackPreset]
    resolve_satellite_isp_s: Callable[[dict[str, Any]], float]
    belief_hist: dict[str, np.ndarray] | None
    knowledge_measurement_hist: dict[str, dict[str, np.ndarray]] | None
    bridge_hist: dict[str, list[dict[str, Any]]] | None
    reentry_metrics: dict[str, dict[str, np.ndarray]] | None
    figure_ids: tuple[str, ...]
    plot_fns: dict[str, Any]
    mode: str
    ric_2d_planes: tuple[str, ...]
    frame_context: FrameContext
    reference_object_id: str
    reference_object_label: str | None
    reference_truth: np.ndarray | None
    ric_truth_hist: dict[str, np.ndarray]
    dpi: int
    show: bool
    close: bool
    save_enabled: bool
    draw_ground_track_map: bool


def build_plot_output_context(
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
    figure_ids: list[str],
    plot_fns: dict[str, Any],
    belief_hist: dict[str, np.ndarray] | None = None,
    knowledge_measurement_hist: dict[str, dict[str, np.ndarray]] | None = None,
    bridge_hist: dict[str, list[dict[str, Any]]] | None = None,
    reentry_metrics: dict[str, dict[str, np.ndarray]] | None = None,
) -> PlotOutputContext:
    plots_cfg = dict(cfg.outputs.plots or {})
    mode = cfg.outputs.mode
    ric_2d_planes = tuple(plots_cfg.get("ric_2d_planes", ["ri", "ic", "rc"]) or ["ri", "ic", "rc"])
    frame_context = frame_context_from_mapping(
        dict(getattr(cfg.simulator, "frames", {}) or {}),
        jd_utc_start=cfg.simulator.initial_jd_utc,
        source="scenario",
    )
    reference_object_id = str(plots_cfg.get("reference_object_id", "")).strip()
    reference_object_label = str(plots_cfg.get("reference_object_label", "")).strip() or None
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
    return PlotOutputContext(
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
        figure_ids=tuple(figure_ids),
        plot_fns=plot_fns,
        mode=mode,
        ric_2d_planes=ric_2d_planes,
        frame_context=frame_context,
        reference_object_id=reference_object_id,
        reference_object_label=reference_object_label,
        reference_truth=reference_truth,
        ric_truth_hist=ric_truth_hist,
        dpi=int(plots_cfg.get("dpi", 150)),
        show=mode in ("interactive", "both"),
        close=mode == "save",
        save_enabled=mode in ("save", "both"),
        draw_ground_track_map=bool(plots_cfg.get("draw_earth_map", False)),
    )
