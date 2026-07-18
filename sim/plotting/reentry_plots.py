from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np

from sim.plotting.single_run_context import _payload_arrays, _payload_reentry_metrics, _save_show_close
from sim.utils.figure_size import cap_figsize
from sim.utils.quaternion import quaternion_to_dcm_bn

ArrayMap = dict[str, np.ndarray]
NestedArrayMap = dict[str, dict[str, np.ndarray]]
OrbitalElementSeriesCache = dict[
    str,
    tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]],
]
RICSummaryFrame = Literal["rectangular", "curvilinear"]

def _plot_reentry_series(
    ax: plt.Axes,
    *,
    t_s: np.ndarray,
    reentry_metrics_by_object: NestedArrayMap,
    metric_key: str,
    ylabel: str,
    scale: float = 1.0,
    active_only: bool = True,
) -> bool:
    plotted = False
    for oid, metrics in sorted(reentry_metrics_by_object.items()):
        series = np.array(metrics.get(metric_key, []), dtype=float).reshape(-1)
        if series.size == 0:
            continue
        n = min(series.size, t_s.size)
        if n <= 0:
            continue
        y = series[:n] * float(scale)
        finite = np.isfinite(y)
        if active_only:
            active = np.array(metrics.get("active", []), dtype=float).reshape(-1)
            if active.size:
                active_mask = np.zeros(n, dtype=bool)
                n_active = min(active.size, n)
                active_mask[:n_active] = active[:n_active] > 0.5
                finite &= active_mask
        if not bool(np.any(finite)):
            continue
        ax.plot(t_s[:n][finite], y[finite], linewidth=1.25, label=oid)
        plotted = True
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend(loc="best")
    return plotted


def _mark_reentry_threshold(
    ax: plt.Axes,
    *,
    begin_altitude_km: float | None,
) -> None:
    if begin_altitude_km is not None:
        ax.axhline(float(begin_altitude_km), color="tab:red", linestyle="--", linewidth=1.0, label="entry threshold")


def plot_reentry_summary(
    payload: dict[str, Any] | None = None,
    *,
    t_s: np.ndarray | None = None,
    reentry_metrics_by_object: NestedArrayMap | None = None,
    begin_altitude_km: float | None = None,
    out_path: str | Path | None = None,
    show: bool = False,
    close: bool = False,
    dpi: int = 150,
) -> plt.Figure:
    t = np.array([] if t_s is None else t_s, dtype=float).reshape(-1)
    metrics = dict(reentry_metrics_by_object or {})
    if payload is not None:
        t = np.array(payload.get("time_s", []), dtype=float).reshape(-1)
        metrics = _payload_reentry_metrics(payload)
    fig, axes = plt.subplots(2, 2, figsize=cap_figsize(13, 8), constrained_layout=True)
    ax_alt, ax_q, ax_g, ax_heat = axes.reshape(-1)
    altitude_plotted = _plot_reentry_series(
        ax_alt,
        t_s=t,
        reentry_metrics_by_object=metrics,
        metric_key="altitude_km",
        ylabel="altitude (km)",
        active_only=False,
    )
    _mark_reentry_threshold(ax_alt, begin_altitude_km=begin_altitude_km)
    if altitude_plotted or begin_altitude_km is not None:
        ax_alt.legend(loc="best")
    _plot_reentry_series(
        ax_q,
        t_s=t,
        reentry_metrics_by_object=metrics,
        metric_key="dynamic_pressure_pa",
        ylabel="dynamic pressure (kPa)",
        scale=1.0e-3,
    )
    _plot_reentry_series(
        ax_g,
        t_s=t,
        reentry_metrics_by_object=metrics,
        metric_key="g_load",
        ylabel="drag decel (g)",
    )
    _plot_reentry_series(
        ax_heat,
        t_s=t,
        reentry_metrics_by_object=metrics,
        metric_key="heat_rate_w_m2",
        ylabel="heat rate (MW/m^2)",
        scale=1.0e-6,
    )
    for ax in axes[-1, :]:
        ax.set_xlabel("time (s)")
    fig.suptitle("Atmospheric Re-Entry Summary")
    _save_show_close(fig, out_path=out_path, show=show, close=close, dpi=dpi)
    return fig


def plot_reentry_aero(
    payload: dict[str, Any] | None = None,
    *,
    t_s: np.ndarray | None = None,
    reentry_metrics_by_object: NestedArrayMap | None = None,
    out_path: str | Path | None = None,
    show: bool = False,
    close: bool = False,
    dpi: int = 150,
) -> plt.Figure:
    t = np.array([] if t_s is None else t_s, dtype=float).reshape(-1)
    metrics = dict(reentry_metrics_by_object or {})
    if payload is not None:
        t = np.array(payload.get("time_s", []), dtype=float).reshape(-1)
        metrics = _payload_reentry_metrics(payload)
    fig, axes = plt.subplots(2, 2, figsize=cap_figsize(13, 8), constrained_layout=True)
    ax_rho, ax_v, ax_q, ax_decel = axes.reshape(-1)
    _plot_reentry_series(
        ax_rho,
        t_s=t,
        reentry_metrics_by_object=metrics,
        metric_key="density_kg_m3",
        ylabel="density (kg/m^3)",
    )
    ax_rho.set_yscale("log")
    _plot_reentry_series(
        ax_v,
        t_s=t,
        reentry_metrics_by_object=metrics,
        metric_key="relative_speed_m_s",
        ylabel="relative speed (km/s)",
        scale=1.0e-3,
    )
    _plot_reentry_series(
        ax_q,
        t_s=t,
        reentry_metrics_by_object=metrics,
        metric_key="dynamic_pressure_pa",
        ylabel="dynamic pressure (kPa)",
        scale=1.0e-3,
    )
    _plot_reentry_series(
        ax_decel,
        t_s=t,
        reentry_metrics_by_object=metrics,
        metric_key="drag_decel_m_s2",
        ylabel="drag decel (m/s^2)",
    )
    for ax in axes[-1, :]:
        ax.set_xlabel("time (s)")
    fig.suptitle("Atmospheric Re-Entry Aerodynamics")
    _save_show_close(fig, out_path=out_path, show=show, close=close, dpi=dpi)
    return fig


def plot_reentry_thermal(
    payload: dict[str, Any] | None = None,
    *,
    t_s: np.ndarray | None = None,
    reentry_metrics_by_object: NestedArrayMap | None = None,
    out_path: str | Path | None = None,
    show: bool = False,
    close: bool = False,
    dpi: int = 150,
) -> plt.Figure:
    t = np.array([] if t_s is None else t_s, dtype=float).reshape(-1)
    metrics = dict(reentry_metrics_by_object or {})
    if payload is not None:
        t = np.array(payload.get("time_s", []), dtype=float).reshape(-1)
        metrics = _payload_reentry_metrics(payload)
    fig, axes = plt.subplots(2, 1, figsize=cap_figsize(12, 7), constrained_layout=True)
    _plot_reentry_series(
        axes[0],
        t_s=t,
        reentry_metrics_by_object=metrics,
        metric_key="heat_rate_w_m2",
        ylabel="heat rate (MW/m^2)",
        scale=1.0e-6,
    )
    _plot_reentry_series(
        axes[1],
        t_s=t,
        reentry_metrics_by_object=metrics,
        metric_key="heat_load_j_m2",
        ylabel="heat load (MJ/m^2)",
        scale=1.0e-6,
    )
    axes[1].set_xlabel("time (s)")
    fig.suptitle("Atmospheric Re-Entry Thermal Loads")
    _save_show_close(fig, out_path=out_path, show=show, close=close, dpi=dpi)
    return fig


def _cross_track_axis_from_truth(hist: np.ndarray) -> np.ndarray | None:
    arr = np.array(hist, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 6 or arr.shape[0] == 0:
        return None
    for row in arr:
        r = row[:3]
        v = row[3:6]
        h = np.cross(r, v)
        norm = float(np.linalg.norm(h))
        if norm > 0.0 and np.all(np.isfinite(h)):
            return h / norm
    return None


def _plot_cross_track_kinematics(ax: plt.Axes, *, t_s: np.ndarray, truth_by_object: ArrayMap) -> bool:
    plotted = False
    for oid, hist in sorted(truth_by_object.items()):
        axis = _cross_track_axis_from_truth(hist)
        if axis is None:
            continue
        n = min(hist.shape[0], t_s.size)
        if n <= 0:
            continue
        cross_track_km = hist[:n, :3] @ axis
        cross_track_m_s = (hist[:n, 3:6] @ axis) * 1.0e3
        finite = np.isfinite(cross_track_km) & np.isfinite(cross_track_m_s)
        if not bool(np.any(finite)):
            continue
        ax.plot(t_s[:n][finite], cross_track_km[finite], linewidth=1.25, label=f"{oid} C pos (km)")
        ax.plot(t_s[:n][finite], cross_track_m_s[finite], linestyle="--", linewidth=1.0, label=f"{oid} C vel (m/s)")
        plotted = True
    ax.set_ylabel("cross-track")
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend(loc="best")
    return plotted


def _plot_lift_axis_alignment(
    ax: plt.Axes,
    *,
    t_s: np.ndarray,
    truth_by_object: ArrayMap,
    lift_axis_body_by_object: dict[str, np.ndarray] | None,
) -> bool:
    axes_by_object = dict(lift_axis_body_by_object or {})
    plotted = False
    for oid, hist in sorted(truth_by_object.items()):
        if hist.shape[1] < 10:
            continue
        cross_track_axis = _cross_track_axis_from_truth(hist)
        if cross_track_axis is None:
            continue
        lift_axis_body = np.array(axes_by_object.get(oid, np.array([0.0, 0.0, 1.0])), dtype=float).reshape(3)
        norm = float(np.linalg.norm(lift_axis_body))
        if norm <= 0.0:
            continue
        lift_axis_body = lift_axis_body / norm
        n = min(hist.shape[0], t_s.size)
        alignment = np.full(n, np.nan)
        for k in range(n):
            q = hist[k, 6:10]
            if np.all(np.isfinite(q)):
                lift_axis_eci = quaternion_to_dcm_bn(q).T @ lift_axis_body
                alignment[k] = float(np.dot(lift_axis_eci, cross_track_axis))
        finite = np.isfinite(alignment)
        if not bool(np.any(finite)):
            continue
        ax.plot(t_s[:n][finite], alignment[finite], linewidth=1.25, label=oid)
        plotted = True
    ax.set_ylabel("lift-axis C alignment")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend(loc="best")
    return plotted


def plot_atmospheric_pass(
    payload: dict[str, Any] | None = None,
    *,
    t_s: np.ndarray | None = None,
    truth_by_object: ArrayMap | None = None,
    reentry_metrics_by_object: NestedArrayMap | None = None,
    lift_axis_body_by_object: dict[str, np.ndarray] | None = None,
    begin_altitude_km: float | None = None,
    out_path: str | Path | None = None,
    show: bool = False,
    close: bool = False,
    dpi: int = 150,
) -> plt.Figure:
    t = np.array([] if t_s is None else t_s, dtype=float).reshape(-1)
    truth = dict(truth_by_object or {})
    metrics = dict(reentry_metrics_by_object or {})
    if payload is not None:
        t, truth, _, _, _, _ = _payload_arrays(payload)
        metrics = _payload_reentry_metrics(payload)

    fig, axes = plt.subplots(3, 2, figsize=cap_figsize(14, 10), constrained_layout=True)
    ax_alt, ax_aero, ax_q, ax_ct, ax_heat, ax_axis = axes.reshape(-1)
    altitude_plotted = _plot_reentry_series(
        ax_alt,
        t_s=t,
        reentry_metrics_by_object=metrics,
        metric_key="altitude_km",
        ylabel="altitude (km)",
        active_only=False,
    )
    _mark_reentry_threshold(ax_alt, begin_altitude_km=begin_altitude_km)
    if altitude_plotted or begin_altitude_km is not None:
        ax_alt.legend(loc="best")
    _plot_reentry_series(
        ax_aero,
        t_s=t,
        reentry_metrics_by_object=metrics,
        metric_key="drag_decel_m_s2",
        ylabel="aero accel (m/s^2)",
    )
    for oid, metrics_by_key in sorted(metrics.items()):
        lift = np.array(metrics_by_key.get("lift_accel_m_s2", []), dtype=float).reshape(-1)
        active = np.array(metrics_by_key.get("active", []), dtype=float).reshape(-1)
        n = min(lift.size, active.size, t.size)
        finite = np.isfinite(lift[:n]) & (active[:n] > 0.5) if n > 0 else np.zeros(0, dtype=bool)
        if bool(np.any(finite)):
            ax_aero.plot(t[:n][finite], lift[:n][finite], linestyle="--", linewidth=1.25, label=f"{oid} lift")
    handles, _ = ax_aero.get_legend_handles_labels()
    if handles:
        ax_aero.legend(loc="best")
    _plot_reentry_series(
        ax_q,
        t_s=t,
        reentry_metrics_by_object=metrics,
        metric_key="dynamic_pressure_pa",
        ylabel="dynamic pressure (kPa)",
        scale=1.0e-3,
    )
    _plot_cross_track_kinematics(ax_ct, t_s=t, truth_by_object=truth)
    _plot_reentry_series(
        ax_heat,
        t_s=t,
        reentry_metrics_by_object=metrics,
        metric_key="heat_load_j_m2",
        ylabel="heat load (MJ/m^2)",
        scale=1.0e-6,
        active_only=False,
    )
    _plot_lift_axis_alignment(
        ax_axis,
        t_s=t,
        truth_by_object=truth,
        lift_axis_body_by_object=lift_axis_body_by_object,
    )
    for ax in axes[-1, :]:
        ax.set_xlabel("time (s)")
    fig.suptitle("Atmospheric Pass and Aero-Assist Summary")
    _save_show_close(fig, out_path=out_path, show=show, close=close, dpi=dpi)
    return fig
