from __future__ import annotations

import numpy as np

from sim.plotting.output_context import PlotOutputContext

FIGURE_IDS = (
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
    "quaternion_eci",
    "quaternion_ric",
    "rates_eci",
    "rates_ric",
)


def render_trajectory_outputs(context: PlotOutputContext) -> dict[str, str]:
    cfg = context.cfg
    t_s = context.t_s
    truth_hist = context.truth_hist
    thrust_hist = context.thrust_hist
    outdir = context.outdir
    figure_ids = context.figure_ids
    frame_context = context.frame_context
    reference_object_id = context.reference_object_id
    reference_object_label = context.reference_object_label
    reference_truth = context.reference_truth
    ric_truth_hist = context.ric_truth_hist
    ric_2d_planes = list(context.ric_2d_planes)
    mode = context.mode
    plot_body_rates = context.plot_fns["plot_body_rates"]
    plot_multi_ric_2d_projections = context.plot_fns["plot_multi_ric_2d_projections"]
    plot_multi_trajectory_frame = context.plot_fns["plot_multi_trajectory_frame"]
    plot_quaternion_components = context.plot_fns["plot_quaternion_components"]
    plot_ric_2d_projections = context.plot_fns["plot_ric_2d_projections"]
    plot_trajectory_frame = context.plot_fns["plot_trajectory_frame"]
    object_state_frames = context.object_state_frames
    eci_truth_hist = {
        object_id: hist
        for object_id, hist in truth_hist.items()
        if object_state_frames.get(object_id, "eci") == "eci"
    }
    out: dict[str, str] = {}
    if "trajectory_eci_multi" in figure_ids and eci_truth_hist:
        p = outdir / "trajectory_eci_multi.png"
        plot_multi_trajectory_frame(t_s, eci_truth_hist, frame="eci", mode=mode, out_path=str(p))
        if mode in ("save", "both"):
            out["trajectory_eci_multi"] = str(p)
    if "trajectory_ecef_multi" in figure_ids and eci_truth_hist:
        p = outdir / "trajectory_ecef_multi.png"
        plot_multi_trajectory_frame(
            t_s,
            eci_truth_hist,
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
        if object_state_frames.get(oid, "eci") != "eci":
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

    return out
