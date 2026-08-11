from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np

from sim.ground_stations import evaluate_ground_station_access
from sim.plotting.output_context import PlotOutputContext
from sim.plotting.output_helpers import (
    _lift_axis_body_by_object,
    _quat_error_angle_deg,
    _thruster_direction_body_by_object,
)
from sim.plotting.style import save_oel_figure
from sim.utils.figure_size import cap_figsize

FIGURE_IDS = (
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
    "attitude",
    "relative_range",
    "control_effort",
    "estimation_error",
    "estimation_error_components",
    "knowledge_filtering",
    "sensor_access",
    "ground_station_access",
    "quaternion_error",
    "attitude_control_summary",
    "reentry_summary",
    "reentry_aero",
    "reentry_thermal",
    "atmospheric_pass",
    "mission_recovery_trade_space",
)


def _plot_private_bridge_outputs(
    *,
    figure_ids: tuple[str, ...],
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


def render_summary_outputs(context: PlotOutputContext) -> dict[str, str]:
    if str(context.mode or "save").strip().lower() == "save":
        import matplotlib

        matplotlib.use("Agg", force=True)
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

    cfg = context.cfg
    t_s = context.t_s
    truth_hist = {
        object_id: hist
        for object_id, hist in context.truth_hist.items()
        if context.object_state_frames.get(object_id, "eci") == "eci"
    }
    target_reference_orbit_truth = context.target_reference_orbit_truth
    thrust_hist = context.thrust_hist
    desired_attitude_hist = context.desired_attitude_hist
    knowledge_hist = context.knowledge_hist
    outdir = context.outdir
    belief_hist = context.belief_hist
    knowledge_measurement_hist = context.knowledge_measurement_hist
    bridge_hist = context.bridge_hist
    reentry_metrics = context.reentry_metrics
    figure_ids = context.figure_ids
    frame_context = context.frame_context
    reference_object_id = context.reference_object_id
    mode = context.mode
    dpi = context.dpi
    show = context.show
    close = context.close
    save_enabled = context.save_enabled
    draw_ground_track_map = context.draw_ground_track_map
    plot_orbit_eci = context.plot_fns["plot_orbit_eci"]
    plot_attitude_tumble = context.plot_fns["plot_attitude_tumble"]
    out: dict[str, str] = {}
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
            object_state_frames=context.object_state_frames,
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
    orbital_elements_cache: dict[
        str,
        tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]],
    ] = {}
    for figure_id, element_id in orbital_element_ids.items():
        if figure_id not in figure_ids:
            continue
        p = outdir / f"{figure_id}.png"
        plot_orbital_element(
            t_s=t_s,
            truth_by_object=truth_hist,
            element_id=element_id,
            object_id=orbital_object_id,
            orbital_elements_cache=orbital_elements_cache,
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
            orbital_elements_cache=orbital_elements_cache,
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
            orbital_elements_cache=orbital_elements_cache,
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

    return out
