from __future__ import annotations

import numpy as np

from sim.plotting.output_context import PlotOutputContext
from sim.plotting.output_helpers import (
    _compute_satellite_delta_v_remaining,
    _thrust_alignment_error_deg_series,
    _thruster_direction_body_by_object,
)
from sim.plotting.style import save_oel_figure
from sim.utils.figure_size import cap_figsize
from sim.utils.frames import ric_dcm_ir_from_rv

FIGURE_IDS = (
    "control_thrust",
    "control_thrust_multi",
    "control_thrust_ric",
    "control_thrust_ric_multi",
    "satellite_delta_v_remaining",
    "thrust_alignment_error",
)


def render_control_outputs(context: PlotOutputContext) -> dict[str, str]:
    cfg = context.cfg
    t_s = context.t_s
    truth_hist = context.truth_hist
    thrust_hist = context.thrust_hist
    outdir = context.outdir
    resolve_satellite_isp_s = context.resolve_satellite_isp_s
    figure_ids = context.figure_ids
    mode = context.mode
    plot_control_commands = context.plot_fns["plot_control_commands"]
    plot_multi_control_commands = context.plot_fns["plot_multi_control_commands"]
    out: dict[str, str] = {}
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

    return out
