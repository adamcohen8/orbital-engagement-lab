from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from sim.config import SimulationScenarioConfig, default_pair_object_ids, default_reference_object_id
from sim.dynamics.orbit.frames import frame_context_from_mapping
from sim.plotting.output_helpers import _compute_satellite_delta_v_remaining, _thruster_mounts_by_object
from sim.utils.ground_track import ground_track_from_eci_history


def render_animations(
    *,
    cfg: SimulationScenarioConfig,
    t_s: np.ndarray,
    truth_hist: dict[str, np.ndarray],
    thrust_hist: dict[str, np.ndarray],
    target_reference_orbit_truth: np.ndarray | None,
    outdir: Path,
    resolve_satellite_isp_s: Callable[[dict[str, Any]], float],
    plot_fns: dict[str, Any],
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
