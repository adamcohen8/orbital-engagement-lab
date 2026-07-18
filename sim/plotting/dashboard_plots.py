from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np

from sim.plotting.single_run_context import (
    _choose_reference,
    _choose_subject,
    _payload_arrays,
    _plot_eci_trajectories,
    _ric_position,
    _ric_position_for_summary,
    _ric_projection_axis_limits,
    _save_show_close,
)
from sim.plotting.single_run_math import _cumulative_delta_v_m_s
from sim.plotting.style import role_color
from sim.utils.figure_size import cap_figsize

ArrayMap = dict[str, np.ndarray]
NestedArrayMap = dict[str, dict[str, np.ndarray]]
OrbitalElementSeriesCache = dict[
    str,
    tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]],
]
RICSummaryFrame = Literal["rectangular", "curvilinear"]

def plot_run_dashboard(
    payload: dict[str, Any] | None = None,
    *,
    t_s: np.ndarray | None = None,
    truth_by_object: ArrayMap | None = None,
    thrust_by_object: ArrayMap | None = None,
    belief_by_object: ArrayMap | None = None,
    target_reference_orbit_truth: np.ndarray | None = None,
    reference_object_id: str | None = None,
    object_id: str | None = None,
    out_path: str | Path | None = None,
    show: bool = False,
    close: bool = False,
    dpi: int = 150,
) -> plt.Figure:
    if payload is not None:
        t_s, truth_by_object, thrust_by_object, belief_by_object, _, target_reference_orbit_truth = _payload_arrays(
            payload
        )
    t = np.array([] if t_s is None else t_s, dtype=float).reshape(-1)
    truth = dict(truth_by_object or {})
    thrust = dict(thrust_by_object or {})
    ref_id, ref = _choose_reference(truth, target_reference_orbit_truth, reference_object_id)
    subj_id, subj = _choose_subject(truth, ref_id, object_id)

    fig = plt.figure(figsize=cap_figsize(14, 9))
    ax_traj = fig.add_subplot(2, 3, 1, projection="3d")
    _plot_eci_trajectories(ax_traj, truth)

    ax_range = fig.add_subplot(2, 3, 2)
    ax_ric = fig.add_subplot(2, 3, 3)
    ax_thrust = fig.add_subplot(2, 3, 4)
    ax_dv = fig.add_subplot(2, 3, 5)
    ax_rate = fig.add_subplot(2, 3, 6)

    if subj is not None and ref is not None:
        n = min(subj.shape[0], ref.shape[0], t.size)
        rel = subj[:n, :3] - ref[:n, :3]
        rel_v = subj[:n, 3:6] - ref[:n, 3:6]
        rng = np.linalg.norm(rel, axis=1)
        spd = np.linalg.norm(rel_v, axis=1)
        ax_range.plot(t[:n], rng, label="range")
        ax_range_t = ax_range.twinx()
        ax_range_t.plot(t[:n], spd, color="tab:orange", label="speed")
        ax_range.set_ylabel("range (km)")
        ax_range_t.set_ylabel("relative speed (km/s)")
        ax_range.set_title(f"Relative Motion ({subj_id} vs {ref_id})")
        ax_range.grid(True, alpha=0.3)

        ric = _ric_position(subj[:n, :], ref[:n, :])
        labels = ("R", "I", "C")
        for i, label in enumerate(labels):
            ax_ric.plot(t[: ric.shape[0]], ric[:, i], label=label)
        ax_ric.set_title("RIC Position Components")
        ax_ric.set_xlabel("time (s)")
        ax_ric.set_ylabel("km")
        ax_ric.grid(True, alpha=0.3)
        ax_ric.legend(loc="best")
    else:
        ax_range.text(0.5, 0.5, "No relative pair available", ha="center", va="center", transform=ax_range.transAxes)
        ax_ric.text(0.5, 0.5, "No RIC reference available", ha="center", va="center", transform=ax_ric.transAxes)

    for oid, u in sorted(thrust.items()):
        if u.ndim != 2 or u.shape[1] < 3:
            continue
        n = min(u.shape[0], t.size)
        mag = np.linalg.norm(np.nan_to_num(u[:n, :3], nan=0.0), axis=1)
        ax_thrust.plot(t[:n], mag, label=oid)
        ax_dv.plot(t[:n], _cumulative_delta_v_m_s(t[:n], u[:n, :3]), label=oid)
    ax_thrust.set_title("Applied Thrust Magnitude")
    ax_thrust.set_xlabel("time (s)")
    ax_thrust.set_ylabel("km/s^2")
    ax_thrust.grid(True, alpha=0.3)
    ax_dv.set_title("Cumulative Delta-V")
    ax_dv.set_xlabel("time (s)")
    ax_dv.set_ylabel("m/s")
    ax_dv.grid(True, alpha=0.3)
    if thrust:
        ax_thrust.legend(loc="best")
        ax_dv.legend(loc="best")

    for oid, hist in sorted(truth.items()):
        if hist.ndim != 2 or hist.shape[1] < 13:
            continue
        n = min(hist.shape[0], t.size)
        rate = np.linalg.norm(np.nan_to_num(hist[:n, 10:13], nan=0.0), axis=1)
        ax_rate.plot(t[:n], rate, label=oid)
    ax_rate.set_title("Body Rate Norm")
    ax_rate.set_xlabel("time (s)")
    ax_rate.set_ylabel("rad/s")
    ax_rate.grid(True, alpha=0.3)
    if truth:
        ax_rate.legend(loc="best")

    fig.suptitle("Run Dashboard")
    fig.tight_layout()
    _save_show_close(fig, out_path=out_path, show=show, close=close, dpi=dpi)
    return fig


def plot_rendezvous_summary(
    payload: dict[str, Any] | None = None,
    *,
    t_s: np.ndarray | None = None,
    truth_by_object: ArrayMap | None = None,
    thrust_by_object: ArrayMap | None = None,
    target_reference_orbit_truth: np.ndarray | None = None,
    reference_object_id: str | None = None,
    object_id: str | None = None,
    keepout_radius_km: float | None = None,
    ric_frame: RICSummaryFrame = "rectangular",
    combine_range_speed: bool = False,
    include_delta_v: bool = False,
    out_path: str | Path | None = None,
    show: bool = False,
    close: bool = False,
    dpi: int = 150,
) -> plt.Figure:
    frame_key = str(ric_frame or "rectangular").strip().lower()
    if frame_key in {"rect", "rectangular", "ric_rect"}:
        summary_frame: RICSummaryFrame = "rectangular"
    elif frame_key in {"curv", "curvilinear", "ric_curv"}:
        summary_frame = "curvilinear"
    else:
        raise ValueError("ric_frame must be one of: rectangular, curvilinear.")
    if include_delta_v and not combine_range_speed:
        combine_range_speed = True
    if payload is not None:
        t_s, truth_by_object, thrust_by_object, _, _, target_reference_orbit_truth = _payload_arrays(payload)
    t = np.array([] if t_s is None else t_s, dtype=float).reshape(-1)
    truth = dict(truth_by_object or {})
    thrust = dict(thrust_by_object or {})
    ref_id, ref = _choose_reference(truth, target_reference_orbit_truth, reference_object_id)
    subj_id, subj = _choose_subject(truth, ref_id, object_id)

    fig, axes = plt.subplots(2, 3, figsize=cap_figsize(14, 8))
    if subj is None or ref is None:
        for ax in axes.ravel():
            ax.text(0.5, 0.5, "No rendezvous pair available", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        _save_show_close(fig, out_path=out_path, show=show, close=close, dpi=dpi)
        return fig

    n = min(subj.shape[0], ref.shape[0], t.size)
    rel = subj[:n, :3] - ref[:n, :3]
    rel_v = subj[:n, 3:6] - ref[:n, 3:6]
    rng = np.linalg.norm(rel, axis=1)
    spd = np.linalg.norm(rel_v, axis=1)
    ric = _ric_position_for_summary(subj[:n, :], ref[:n, :], frame=summary_frame)

    planes = ((1, 0, "I", "R"), (1, 2, "I", "C"), (2, 0, "C", "R"))
    for ax, (ix, iy, xlab, ylab) in zip(axes[0], planes):
        ax.plot(ric[:, ix], ric[:, iy], linewidth=2.0, color=role_color("actual"), label="actual")
        if ric.shape[0]:
            ax.scatter([ric[0, ix]], [ric[0, iy]], color=role_color("coast"), s=24, label="start")
            ax.scatter(
                [ric[-1, ix]],
                [ric[-1, iy]],
                facecolors="none",
                edgecolors=role_color("chaser"),
                s=46,
                linewidths=1.5,
                label="final",
            )
        if keepout_radius_km is not None and np.isfinite(float(keepout_radius_km)) and float(keepout_radius_km) > 0.0:
            circ = plt.Circle(
                (0.0, 0.0),
                float(keepout_radius_km),
                color=role_color("safety_zone"),
                fill=False,
                linestyle="--",
                alpha=0.7,
            )
            ax.add_patch(circ)
        ax.set_xlabel(f"{xlab} (km)")
        ax.set_ylabel(f"{ylab} (km)")
        ax.set_title(f"{xlab}-{ylab} Projection")
        ax.grid(True, alpha=0.3)
        xlim, ylim = _ric_projection_axis_limits(
            ric,
            axis_indices=(ix, iy),
            keepout_radius_km=keepout_radius_km,
        )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    if combine_range_speed:
        ax_range = axes[1, 0]
        ax_speed = ax_range.twinx()
        ax_range.plot(t[:n], rng, color=role_color("actual"), label="range")
        ax_speed.plot(t[:n], spd, color=role_color("chaser"), label="speed")
        ax_range.set_title("Relative Range and Speed")
        ax_range.set_ylabel("range (km)")
        ax_speed.set_ylabel("speed (km/s)")
        ax_speed.grid(False)
        handles_1, labels_1 = ax_range.get_legend_handles_labels()
        handles_2, labels_2 = ax_speed.get_legend_handles_labels()
        ax_range.legend(handles_1 + handles_2, labels_1 + labels_2, loc="best")
    else:
        axes[1, 0].plot(t[:n], rng, color=role_color("actual"))
        axes[1, 0].set_title("Relative Range")
        axes[1, 0].set_ylabel("km")
    axes[1, 0].set_xlabel("time (s)")
    axes[1, 0].grid(True, alpha=0.3)

    component_ax = axes[1, 1] if combine_range_speed else axes[1, 2]
    if not combine_range_speed:
        axes[1, 1].plot(t[:n], spd, color=role_color("chaser"))
        axes[1, 1].set_title("Relative Speed")
        axes[1, 1].set_ylabel("km/s")
        axes[1, 1].set_xlabel("time (s)")
        axes[1, 1].grid(True, alpha=0.3)

    for i, label in enumerate(("R", "I", "C")):
        component_ax.plot(t[:n], ric[:, i], label=label)
    frame_label = "Curvilinear RIC" if summary_frame == "curvilinear" else "RIC"
    component_ax.set_title(f"{frame_label} Components")
    component_ax.set_ylabel("km")
    component_ax.set_xlabel("time (s)")
    component_ax.grid(True, alpha=0.3)
    component_ax.legend(loc="best")

    if include_delta_v:
        ax_dv = axes[1, 2]
        plotted_dv = False
        for oid in [str(subj_id or ""), str(ref_id or "")]:
            if not oid or oid not in thrust:
                continue
            u = thrust.get(oid)
            if u is None or u.ndim != 2 or u.shape[1] < 3:
                continue
            n_u = min(u.shape[0], t.size)
            if n_u <= 0:
                continue
            ax_dv.plot(t[:n_u], _cumulative_delta_v_m_s(t[:n_u], u[:n_u, :3]), label=oid)
            plotted_dv = True
        if not plotted_dv:
            ax_dv.text(0.5, 0.5, "No thrust history available", ha="center", va="center", transform=ax_dv.transAxes)
        ax_dv.set_title("Cumulative Delta-V")
        ax_dv.set_ylabel("m/s")
        ax_dv.set_xlabel("time (s)")
        ax_dv.grid(True, alpha=0.3)
        if plotted_dv:
            ax_dv.legend(loc="best")

    title = "Curvilinear Rendezvous Summary" if summary_frame == "curvilinear" else "Rendezvous Summary"
    fig.suptitle(f"{title} ({subj_id} vs {ref_id})")
    fig.tight_layout()
    _save_show_close(fig, out_path=out_path, show=show, close=close, dpi=dpi)
    return fig


def plot_rendezvous_summary_curvilinear(
    payload: dict[str, Any] | None = None,
    **kwargs: Any,
) -> plt.Figure:
    kwargs.setdefault("ric_frame", "curvilinear")
    kwargs.setdefault("combine_range_speed", True)
    kwargs.setdefault("include_delta_v", True)
    return plot_rendezvous_summary(payload, **kwargs)


def plot_control_effort(
    payload: dict[str, Any] | None = None,
    *,
    t_s: np.ndarray | None = None,
    thrust_by_object: ArrayMap | None = None,
    object_id: str | None = None,
    max_accel_km_s2: float | None = None,
    out_path: str | Path | None = None,
    show: bool = False,
    close: bool = False,
    dpi: int = 150,
) -> plt.Figure:
    if payload is not None:
        t_s, _, thrust_by_object, _, _, _ = _payload_arrays(payload)
    t = np.array([] if t_s is None else t_s, dtype=float).reshape(-1)
    thrust = dict(thrust_by_object or {})
    ids = [object_id] if object_id and object_id in thrust else sorted(thrust.keys())

    fig, axes = plt.subplots(2, 1, figsize=cap_figsize(11, 7), sharex=True)
    for oid in ids:
        u = thrust.get(oid)
        if u is None or u.ndim != 2 or u.shape[1] < 3:
            continue
        n = min(u.shape[0], t.size)
        labels = ("x", "y", "z")
        for i, label in enumerate(labels):
            axes[0].plot(t[:n], u[:n, i], linewidth=1.0, label=f"{oid} {label}")
        mag = np.linalg.norm(np.nan_to_num(u[:n, :3], nan=0.0), axis=1)
        axes[1].plot(t[:n], mag, label=f"{oid} |a|")
        axes[1].plot(t[:n], _cumulative_delta_v_m_s(t[:n], u[:n, :3]), linestyle="--", label=f"{oid} dv")
    if max_accel_km_s2 is not None and np.isfinite(float(max_accel_km_s2)) and float(max_accel_km_s2) > 0.0:
        axes[1].axhline(float(max_accel_km_s2), color="tab:red", linestyle=":", label="max accel")
    axes[0].set_title("Applied Thrust Components")
    axes[0].set_ylabel("km/s^2")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    axes[1].set_title("Magnitude and Cumulative Delta-V")
    axes[1].set_ylabel("km/s^2 / m/s")
    axes[1].set_xlabel("time (s)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")
    fig.tight_layout()
    _save_show_close(fig, out_path=out_path, show=show, close=close, dpi=dpi)
    return fig
