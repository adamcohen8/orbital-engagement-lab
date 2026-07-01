from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np

from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2, EARTH_RADIUS_KM
from sim.dynamics.orbit.frames import frame_context_from_mapping
from sim.plotting.style import role_color, show_save_close_oel
from sim.utils.figure_size import cap_figsize
from sim.utils.frames import ric_dcm_ir_from_rv, ric_rect_to_curv
from sim.utils.ground_track import ground_track_from_eci_history, split_ground_track_dateline
from sim.utils.plot_windows import RIC_FOLLOW_MARGIN, windows_from_points
from sim.utils.plotting import _draw_earth_sphere_3d
from sim.utils.plotting_capabilities import _setup_ground_track_axes
from sim.utils.quaternion import quaternion_to_dcm_bn

ArrayMap = dict[str, np.ndarray]
NestedArrayMap = dict[str, dict[str, np.ndarray]]
RICSummaryFrame = Literal["rectangular", "curvilinear"]

ORBITAL_ELEMENT_SPECS: dict[str, tuple[str, str]] = {
    "a": ("Semi-Major Axis", "km"),
    "ecc": ("Eccentricity", ""),
    "inc": ("Inclination", "deg"),
    "raan": ("RAAN", "deg"),
    "argp": ("Argument of Perigee", "deg"),
    "true_anomaly": ("True Anomaly", "deg"),
}


def _object_color(object_id: str) -> str | None:
    oid = str(object_id or "").lower()
    if "target" in oid:
        return role_color("target")
    if "chaser" in oid or "deputy" in oid or "red" in oid:
        return role_color("chaser")
    return None


def _as_array(value: Any, *, cols: int | None = None) -> np.ndarray:
    arr = np.array(value if value is not None else [], dtype=float)
    if arr.ndim == 1 and arr.size == 0:
        return np.zeros((0, 0 if cols is None else cols), dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _array_map(value: Any) -> ArrayMap:
    if not isinstance(value, dict):
        return {}
    out: ArrayMap = {}
    for key, arr in value.items():
        parsed = _as_array(arr)
        if parsed.ndim == 2 and parsed.shape[0] > 0:
            out[str(key)] = parsed
    return out


def _nested_array_map(value: Any) -> NestedArrayMap:
    if not isinstance(value, dict):
        return {}
    out: NestedArrayMap = {}
    for outer, inner in value.items():
        if not isinstance(inner, dict):
            continue
        parsed_inner = _array_map(inner)
        if parsed_inner:
            out[str(outer)] = parsed_inner
    return out


def _reentry_metric_map(value: Any) -> NestedArrayMap:
    return _nested_array_map(value)


def _payload_arrays(
    payload: dict[str, Any],
) -> tuple[np.ndarray, ArrayMap, ArrayMap, ArrayMap, NestedArrayMap, np.ndarray | None]:
    t_s = np.array(payload.get("time_s", []), dtype=float).reshape(-1)
    truth = _array_map(payload.get("truth_by_object", {}))
    thrust = _array_map(payload.get("applied_thrust_by_object", {}))
    belief = _array_map(payload.get("belief_by_object", {}))
    knowledge = _nested_array_map(payload.get("knowledge_by_observer", {}))
    ref = _as_array(payload.get("target_reference_orbit_truth", []), cols=14)
    ref_out = ref if ref.ndim == 2 and ref.shape[0] > 0 and ref.shape[1] >= 6 else None
    return t_s, truth, thrust, belief, knowledge, ref_out


def _payload_reentry_metrics(payload: dict[str, Any] | None) -> NestedArrayMap:
    if payload is None:
        return {}
    return _reentry_metric_map(payload.get("reentry_metrics_by_object", {}))


def _save_show_close(fig: plt.Figure, *, out_path: str | Path | None, show: bool, close: bool, dpi: int) -> None:
    mode = "both" if out_path is not None and show else "save" if out_path is not None else "interactive" if show else "none"
    if mode == "none":
        if close:
            plt.close(fig)
        return
    show_save_close_oel(
        fig,
        mode=mode,
        out_path=out_path,
        dpi=int(dpi),
        plt_module=plt,
        close=close,
        show_block=False,
    )


def _choose_reference(
    truth_by_object: ArrayMap,
    target_reference_orbit_truth: np.ndarray | None,
    reference_object_id: str | None,
) -> tuple[str, np.ndarray] | tuple[None, None]:
    if target_reference_orbit_truth is not None and target_reference_orbit_truth.shape[1] >= 6:
        return "reference", target_reference_orbit_truth
    if reference_object_id and reference_object_id in truth_by_object:
        return reference_object_id, truth_by_object[reference_object_id]
    if "target" in truth_by_object:
        return "target", truth_by_object["target"]
    if truth_by_object:
        key = sorted(truth_by_object.keys())[0]
        return key, truth_by_object[key]
    return None, None


def _choose_subject(
    truth_by_object: ArrayMap, reference_id: str | None, object_id: str | None = None
) -> tuple[str, np.ndarray] | tuple[None, None]:
    if object_id and object_id in truth_by_object:
        return object_id, truth_by_object[object_id]
    preferred = [k for k in ("chaser", "target", "rocket") if k in truth_by_object and k != reference_id]
    if preferred:
        key = preferred[0]
        return key, truth_by_object[key]
    for key in sorted(truth_by_object.keys()):
        if key != reference_id:
            return key, truth_by_object[key]
    if truth_by_object:
        key = sorted(truth_by_object.keys())[0]
        return key, truth_by_object[key]
    return None, None


def _ric_position(subject: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return _ric_relative_state(subject, reference)[:, :3]


def _ric_relative_state(subject: np.ndarray, reference: np.ndarray) -> np.ndarray:
    n = min(subject.shape[0], reference.shape[0])
    out = np.full((n, 6), np.nan, dtype=float)
    for k in range(n):
        rv_ref = reference[k, :6]
        rv_sub = subject[k, :6]
        if not (np.all(np.isfinite(rv_ref)) and np.all(np.isfinite(rv_sub))):
            continue
        c_ir = ric_dcm_ir_from_rv(rv_ref[:3], rv_ref[3:6])
        out[k, :3] = c_ir.T @ (rv_sub[:3] - rv_ref[:3])
        out[k, 3:6] = c_ir.T @ (rv_sub[3:6] - rv_ref[3:6])
    return out


def _ric_position_for_summary(subject: np.ndarray, reference: np.ndarray, *, frame: RICSummaryFrame) -> np.ndarray:
    rect_state = _ric_relative_state(subject, reference)
    if frame == "rectangular":
        return rect_state[:, :3]
    out = np.full((rect_state.shape[0], 3), np.nan, dtype=float)
    for k in range(rect_state.shape[0]):
        rv_ref = reference[k, :6]
        x_rect = rect_state[k, :]
        if not (np.all(np.isfinite(rv_ref)) and np.all(np.isfinite(x_rect))):
            continue
        out[k, :] = ric_rect_to_curv(x_rect, r0_km=float(np.linalg.norm(rv_ref[:3])))[:3]
    return out


def _ric_projection_axis_limits(
    ric: np.ndarray,
    *,
    axis_indices: tuple[int, int],
    keepout_radius_km: float | None = None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    points: list[np.ndarray] = []
    if ric.ndim == 2 and ric.shape[1] >= 3:
        finite = ric[np.all(np.isfinite(ric[:, :3]), axis=1), :3]
        points.extend(np.array(row, dtype=float) for row in finite)
    if keepout_radius_km is not None and np.isfinite(float(keepout_radius_km)) and float(keepout_radius_km) > 0.0:
        radius = float(keepout_radius_km)
        for sign in (-1.0, 1.0):
            point = np.zeros(3, dtype=float)
            point[axis_indices[0]] = sign * radius
            points.append(point)
            point = np.zeros(3, dtype=float)
            point[axis_indices[1]] = sign * radius
            points.append(point)
    min_span = 1.0
    if keepout_radius_km is not None and np.isfinite(float(keepout_radius_km)) and float(keepout_radius_km) > 0.0:
        min_span = max(min_span, 2.0 * float(keepout_radius_km))
    xlim, ylim = windows_from_points(
        points,
        axis_indices=axis_indices,
        min_span=min_span,
        margin=RIC_FOLLOW_MARGIN,
    )
    return xlim, ylim


def _finite_rows(arr: np.ndarray, cols: int = 3) -> np.ndarray:
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < cols:
        return np.zeros(0, dtype=bool)
    return np.all(np.isfinite(arr[:, :cols]), axis=1)


def _set_equal_3d(ax: Any, points: list[np.ndarray], *, center_at_origin: bool = False) -> None:
    finite_parts = []
    for arr in points:
        a = np.array(arr, dtype=float)
        if a.ndim == 2 and a.shape[1] >= 3:
            finite = a[np.all(np.isfinite(a[:, :3]), axis=1), :3]
            if finite.size:
                finite_parts.append(finite)
    if not finite_parts:
        lim = EARTH_RADIUS_KM * 1.25
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        return
    pts = np.vstack(finite_parts)
    if center_at_origin:
        center = np.zeros(3, dtype=float)
        half = max(float(np.max(np.abs(pts[:, :3]))) * 1.08, EARTH_RADIUS_KM * 1.15, 1.0)
    else:
        center = np.mean(pts, axis=0)
        span = max(float(np.max(np.ptp(pts, axis=0))), 1.0)
        half = 0.6 * span
        if np.linalg.norm(center) < EARTH_RADIUS_KM * 1.1:
            half = max(half, EARTH_RADIUS_KM * 1.15)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1, 1, 1))


def _plot_eci_trajectories(ax: Any, truth_by_object: ArrayMap) -> None:
    _draw_earth_sphere_3d(ax)
    plotted: list[np.ndarray] = []
    for oid, hist in sorted(truth_by_object.items()):
        if hist.shape[1] < 3:
            continue
        r = hist[:, :3]
        mask = _finite_rows(r)
        if not np.any(mask):
            continue
        color = _object_color(oid)
        ax.plot(r[mask, 0], r[mask, 1], r[mask, 2], linewidth=1.8, label=oid, color=color)
        idx = np.where(mask)[0]
        ax.scatter([r[idx[0], 0]], [r[idx[0], 1]], [r[idx[0], 2]], color=role_color("coast"), s=18)
        ax.scatter(
            [r[idx[-1], 0]],
            [r[idx[-1], 1]],
            [r[idx[-1], 2]],
            facecolors="none",
            edgecolors=color or role_color("actual"),
            s=38,
            linewidths=1.4,
        )
        plotted.append(r)
    earth_extent = np.array(
        [
            [-EARTH_RADIUS_KM, -EARTH_RADIUS_KM, -EARTH_RADIUS_KM],
            [EARTH_RADIUS_KM, EARTH_RADIUS_KM, EARTH_RADIUS_KM],
        ],
        dtype=float,
    )
    _set_equal_3d(ax, plotted + [earth_extent], center_at_origin=True)
    ax.set_xlabel("ECI x (km)")
    ax.set_ylabel("ECI y (km)")
    ax.set_zlabel("ECI z (km)")
    ax.set_title("Trajectory")
    if plotted:
        ax.legend(loc="best")


def _time_for(arr: np.ndarray, t_s: np.ndarray) -> np.ndarray:
    n = min(arr.shape[0], t_s.size)
    return t_s[:n]


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


def _cumulative_delta_v_m_s(t_s: np.ndarray, accel_km_s2: np.ndarray) -> np.ndarray:
    mag = np.linalg.norm(np.nan_to_num(accel_km_s2, nan=0.0), axis=1)
    if mag.size == 0:
        return mag
    dt = np.diff(t_s[: mag.size], prepend=t_s[0] if t_s.size else 0.0)
    dt = np.clip(dt, 0.0, None)
    return np.cumsum(mag * dt) * 1000.0


def _quat_error_angle_deg(q_des: np.ndarray, q_cur: np.ndarray) -> float:
    qd = np.array(q_des, dtype=float).reshape(-1)
    qc = np.array(q_cur, dtype=float).reshape(-1)
    if qd.size != 4 or qc.size != 4:
        return float("nan")
    nd = float(np.linalg.norm(qd))
    nc = float(np.linalg.norm(qc))
    if nd <= 0.0 or nc <= 0.0:
        return float("nan")
    qd = qd / nd
    qc = qc / nc
    dot = abs(float(np.dot(qd, qc)))
    dot = float(np.clip(dot, -1.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def _quat_error_series_deg(
    *,
    truth_hist: np.ndarray,
    desired_attitude_hist: np.ndarray | None,
    n_s: int,
) -> np.ndarray:
    err_deg = np.full(n_s, np.nan, dtype=float)
    if desired_attitude_hist is None or desired_attitude_hist.size == 0:
        return err_deg
    qd = np.array(desired_attitude_hist[:n_s, :], dtype=float)
    qc = np.array(truth_hist[:n_s, 6:10], dtype=float)
    for k in range(1, n_s):
        if not np.all(np.isfinite(qd[k, :])) and np.all(np.isfinite(qd[k - 1, :])):
            qd[k, :] = qd[k - 1, :]
    for k in range(n_s):
        if not (np.all(np.isfinite(qd[k, :])) and np.all(np.isfinite(qc[k, :]))):
            continue
        err_deg[k] = _quat_error_angle_deg(qd[k, :], qc[k, :])
    return err_deg


def _thrust_alignment_error_deg_series(
    *,
    truth_hist: np.ndarray,
    thrust_hist: np.ndarray,
    thrust_axis_body: np.ndarray,
    n_s: int,
) -> np.ndarray:
    axis_body = np.array(thrust_axis_body, dtype=float).reshape(-1)
    if axis_body.size != 3:
        axis_body = np.array([1.0, 0.0, 0.0], dtype=float)
    norm_axis = float(np.linalg.norm(axis_body))
    if norm_axis <= 0.0:
        axis_body = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        axis_body = axis_body / norm_axis
    err_deg = np.full(n_s, np.nan, dtype=float)
    for k in range(n_s):
        a_cmd = np.array(thrust_hist[k, :3], dtype=float)
        a_norm = float(np.linalg.norm(a_cmd))
        if a_norm <= 1e-15 or not np.all(np.isfinite(a_cmd)):
            continue
        q_bn = np.array(truth_hist[k, 6:10], dtype=float)
        if not np.all(np.isfinite(q_bn)):
            continue
        c_bn = quaternion_to_dcm_bn(q_bn)
        thrust_axis_eci = c_bn.T @ axis_body
        burn_dir_eci = -a_cmd / a_norm
        cosang = float(np.clip(np.dot(thrust_axis_eci, burn_dir_eci), -1.0, 1.0))
        err_deg[k] = float(np.degrees(np.arccos(cosang)))
    return err_deg


def _safe_angle_deg(cos_value: float, *, flip: bool = False) -> float:
    angle = float(np.degrees(np.arccos(float(np.clip(cos_value, -1.0, 1.0)))))
    return 360.0 - angle if flip and angle > 0.0 else angle


def _classical_orbital_elements_series(
    truth_hist: np.ndarray,
    *,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
) -> dict[str, np.ndarray]:
    arr = np.array(truth_hist, dtype=float)
    n = arr.shape[0] if arr.ndim == 2 else 0
    out = {key: np.full(n, np.nan, dtype=float) for key in ORBITAL_ELEMENT_SPECS}
    if n == 0 or arr.shape[1] < 6 or not np.isfinite(mu_km3_s2) or mu_km3_s2 <= 0.0:
        return out

    h_tol = 1e-10
    n_tol = 1e-10
    e_tol = 1e-8
    k_hat = np.array([0.0, 0.0, 1.0], dtype=float)
    for idx in range(n):
        r_vec = np.array(arr[idx, 0:3], dtype=float)
        v_vec = np.array(arr[idx, 3:6], dtype=float)
        if not (np.all(np.isfinite(r_vec)) and np.all(np.isfinite(v_vec))):
            continue
        r = float(np.linalg.norm(r_vec))
        if r <= 0.0:
            continue
        h_vec = np.cross(r_vec, v_vec)
        h = float(np.linalg.norm(h_vec))
        if h <= h_tol:
            continue
        v2 = float(np.dot(v_vec, v_vec))
        eps = 0.5 * v2 - mu_km3_s2 / r
        if abs(eps) > 1e-14:
            out["a"][idx] = float(-mu_km3_s2 / (2.0 * eps))
        e_vec = np.cross(v_vec, h_vec) / mu_km3_s2 - r_vec / r
        ecc = float(np.linalg.norm(e_vec))
        out["ecc"][idx] = ecc
        out["inc"][idx] = _safe_angle_deg(h_vec[2] / h)

        n_vec = np.cross(k_hat, h_vec)
        n_norm = float(np.linalg.norm(n_vec))
        if n_norm > n_tol:
            out["raan"][idx] = _safe_angle_deg(n_vec[0] / n_norm, flip=n_vec[1] < 0.0)
        if n_norm > n_tol and ecc > e_tol:
            out["argp"][idx] = _safe_angle_deg(
                float(np.dot(n_vec, e_vec)) / (n_norm * ecc),
                flip=e_vec[2] < 0.0,
            )
        if ecc > e_tol:
            out["true_anomaly"][idx] = _safe_angle_deg(
                float(np.dot(e_vec, r_vec)) / (ecc * r),
                flip=float(np.dot(r_vec, v_vec)) < 0.0,
            )
    return out


def _orbital_element_object_ids(truth_by_object: ArrayMap, object_id: str | None) -> list[str]:
    if object_id:
        return [object_id] if object_id in truth_by_object else []
    return sorted(truth_by_object.keys())


def _plot_element_on_axis(
    ax: plt.Axes,
    *,
    t_s: np.ndarray,
    truth_by_object: ArrayMap,
    element_id: str,
    object_id: str | None,
    label_prefix: bool = False,
) -> bool:
    plotted = False
    for oid in _orbital_element_object_ids(truth_by_object, object_id):
        hist = truth_by_object.get(oid)
        if hist is None:
            continue
        series = _classical_orbital_elements_series(hist).get(element_id)
        if series is None:
            continue
        n = min(t_s.size, series.size)
        if n <= 0:
            continue
        y = np.array(series[:n], dtype=float)
        finite = np.isfinite(y)
        if not np.any(finite):
            continue
        label = f"{oid} {element_id}" if label_prefix else oid
        ax.plot(t_s[:n], y, linewidth=1.2, label=label)
        plotted = True
    return plotted


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


def plot_estimation_error(
    payload: dict[str, Any] | None = None,
    *,
    t_s: np.ndarray | None = None,
    truth_by_object: ArrayMap | None = None,
    belief_by_object: ArrayMap | None = None,
    out_path: str | Path | None = None,
    show: bool = False,
    close: bool = False,
    dpi: int = 150,
) -> plt.Figure:
    if payload is not None:
        t_s, truth_by_object, _, belief_by_object, _, _ = _payload_arrays(payload)
    t = np.array([] if t_s is None else t_s, dtype=float).reshape(-1)
    truth = dict(truth_by_object or {})
    belief = dict(belief_by_object or {})
    fig, axes = plt.subplots(2, 1, figsize=cap_figsize(11, 7), sharex=True)
    plotted = False
    for oid, b in sorted(belief.items()):
        x = truth.get(oid)
        if x is None or x.shape[1] < 6 or b.shape[1] < 6:
            continue
        n = min(x.shape[0], b.shape[0], t.size)
        pos_err = np.linalg.norm(b[:n, :3] - x[:n, :3], axis=1)
        vel_err = np.linalg.norm(b[:n, 3:6] - x[:n, 3:6], axis=1)
        axes[0].plot(t[:n], pos_err, label=oid)
        axes[1].plot(t[:n], vel_err, label=oid)
        plotted = True
    if not plotted:
        for ax in axes:
            ax.text(0.5, 0.5, "No belief/truth pair available", ha="center", va="center", transform=ax.transAxes)
    axes[0].set_title("Position Estimation Error")
    axes[0].set_ylabel("km")
    axes[0].grid(True, alpha=0.3)
    axes[1].set_title("Velocity Estimation Error")
    axes[1].set_ylabel("km/s")
    axes[1].set_xlabel("time (s)")
    axes[1].grid(True, alpha=0.3)
    if plotted:
        axes[0].legend(loc="best")
        axes[1].legend(loc="best")
    fig.tight_layout()
    _save_show_close(fig, out_path=out_path, show=show, close=close, dpi=dpi)
    return fig


def plot_estimation_error_components(
    payload: dict[str, Any] | None = None,
    *,
    t_s: np.ndarray | None = None,
    truth_by_object: ArrayMap | None = None,
    belief_by_object: ArrayMap | None = None,
    out_path: str | Path | None = None,
    show: bool = False,
    close: bool = False,
    dpi: int = 150,
) -> plt.Figure:
    if payload is not None:
        t_s, truth_by_object, _, belief_by_object, _, _ = _payload_arrays(payload)
    t = np.array([] if t_s is None else t_s, dtype=float).reshape(-1)
    truth = dict(truth_by_object or {})
    belief = dict(belief_by_object or {})
    fig, axes = plt.subplots(2, 1, figsize=cap_figsize(12, 8), sharex=True)
    plotted = False
    pos_labels = ("x", "y", "z")
    vel_labels = ("vx", "vy", "vz")
    for oid, b in sorted(belief.items()):
        x = truth.get(oid)
        if x is None or x.shape[1] < 6 or b.shape[1] < 6:
            continue
        n = min(x.shape[0], b.shape[0], t.size)
        err = b[:n, :6] - x[:n, :6]
        for i, label in enumerate(pos_labels):
            axes[0].plot(t[:n], err[:, i], linewidth=1.0, label=f"{oid} {label}")
        for i, label in enumerate(vel_labels):
            axes[1].plot(t[:n], err[:, i + 3], linewidth=1.0, label=f"{oid} {label}")
        plotted = True
    if not plotted:
        for ax in axes:
            ax.text(0.5, 0.5, "No belief/truth pair available", ha="center", va="center", transform=ax.transAxes)
    axes[0].set_title("Position Estimation Error Components")
    axes[0].set_ylabel("km")
    axes[0].grid(True, alpha=0.3)
    axes[1].set_title("Velocity Estimation Error Components")
    axes[1].set_ylabel("km/s")
    axes[1].set_xlabel("time (s)")
    axes[1].grid(True, alpha=0.3)
    if plotted:
        axes[0].legend(loc="best", ncol=2)
        axes[1].legend(loc="best", ncol=2)
    fig.tight_layout()
    _save_show_close(fig, out_path=out_path, show=show, close=close, dpi=dpi)
    return fig


def plot_knowledge_filtering(
    payload: dict[str, Any] | None = None,
    *,
    t_s: np.ndarray | None = None,
    truth_by_object: ArrayMap | None = None,
    knowledge_by_observer: NestedArrayMap | None = None,
    knowledge_measurements_by_observer: NestedArrayMap | None = None,
    knowledge_noise_by_observer: dict[str, dict[str, Any]] | None = None,
    out_path: str | Path | None = None,
    show: bool = False,
    close: bool = False,
    dpi: int = 150,
) -> plt.Figure:
    if payload is not None:
        t_s, truth_by_object, _, _, knowledge_by_observer, _ = _payload_arrays(payload)
        knowledge_measurements_by_observer = _nested_array_map(payload.get("knowledge_measurements_by_observer", {}))
    t = np.array([] if t_s is None else t_s, dtype=float).reshape(-1)
    truth = dict(truth_by_object or {})
    estimates = dict(knowledge_by_observer or {})
    measurements = dict(knowledge_measurements_by_observer or {})
    noise_by_observer = dict(knowledge_noise_by_observer or {})
    fig, axes = plt.subplots(2, 3, figsize=cap_figsize(15, 8), constrained_layout=True)
    ax_range, ax_pos, ax_vel, ax_pos_hist, ax_vel_hist, ax_norm_hist = axes.reshape(-1)
    plotted = False
    pos_hist_values: list[np.ndarray] = []
    vel_hist_values: list[np.ndarray] = []
    norm_hist_values: list[np.ndarray] = []
    pos_sigma_markers_m: list[tuple[float, float]] = []
    vel_sigma_markers_mm_s: list[tuple[float, float]] = []
    for obs, by_target in sorted(estimates.items()):
        for target, estimate in sorted(by_target.items()):
            target_truth = truth.get(target)
            observer_truth = truth.get(obs)
            measurement = measurements.get(obs, {}).get(target)
            if target_truth is None or target_truth.shape[1] < 6 or estimate.shape[1] < 6:
                continue
            n = min(target_truth.shape[0], estimate.shape[0], t.size)
            if n <= 0:
                continue
            label = f"{obs}->{target}"
            if observer_truth is not None and observer_truth.shape[1] >= 3:
                nr = min(n, observer_truth.shape[0])
                truth_range = np.linalg.norm(target_truth[:nr, :3] - observer_truth[:nr, :3], axis=1)
                estimate_range = np.linalg.norm(estimate[:nr, :3] - observer_truth[:nr, :3], axis=1)
                ax_range.plot(t[:nr], truth_range, color="black", linewidth=1.4, label=f"{label} truth")
                ax_range.plot(t[:nr], estimate_range, linewidth=1.1, label=f"{label} estimate")
                if measurement is not None and measurement.shape[1] >= 3:
                    nm = min(nr, measurement.shape[0])
                    meas_range = np.linalg.norm(measurement[:nm, :3] - observer_truth[:nm, :3], axis=1)
                    valid_meas = np.all(np.isfinite(measurement[:nm, :3]), axis=1)
                    ax_range.scatter(t[:nm][valid_meas], meas_range[valid_meas], s=8, alpha=0.35, label=f"{label} meas")
            pos_err_est = np.linalg.norm(estimate[:n, :3] - target_truth[:n, :3], axis=1)
            vel_err_est = np.linalg.norm(estimate[:n, 3:6] - target_truth[:n, 3:6], axis=1)
            ax_pos.plot(t[:n], pos_err_est, linewidth=1.2, label=f"{label} estimate")
            ax_vel.plot(t[:n], vel_err_est * 1000.0, linewidth=1.2, label=f"{label} estimate")
            if measurement is not None and measurement.shape[1] >= 6:
                nm = min(n, measurement.shape[0])
                meas_pos_err = measurement[:nm, :3] - target_truth[:nm, :3]
                meas_vel_err = measurement[:nm, 3:6] - target_truth[:nm, 3:6]
                valid_pos = np.all(np.isfinite(meas_pos_err), axis=1)
                valid_vel = np.all(np.isfinite(meas_vel_err), axis=1)
                ax_pos.scatter(
                    t[:nm][valid_pos],
                    np.linalg.norm(meas_pos_err[valid_pos], axis=1),
                    s=8,
                    alpha=0.35,
                    label=f"{label} measurement",
                )
                ax_vel.scatter(
                    t[:nm][valid_vel],
                    np.linalg.norm(meas_vel_err[valid_vel], axis=1) * 1000.0,
                    s=8,
                    alpha=0.35,
                    label=f"{label} measurement",
                )
                if np.any(valid_pos):
                    pos_hist_values.append(meas_pos_err[valid_pos].reshape(-1))
                if np.any(valid_vel):
                    vel_hist_values.append(meas_vel_err[valid_vel].reshape(-1))
                noise = noise_by_observer.get(obs, {})
                pos_sigma = np.array(noise.get("pos_sigma_km", []), dtype=float).reshape(-1)
                vel_sigma = np.array(noise.get("vel_sigma_km_s", []), dtype=float).reshape(-1)
                pos_bias = np.array(noise.get("pos_bias_km", np.zeros(3)), dtype=float).reshape(-1)
                vel_bias = np.array(noise.get("vel_bias_km_s", np.zeros(3)), dtype=float).reshape(-1)
                if pos_sigma.size in (1, 3) and np.any(pos_sigma > 0.0) and np.any(valid_pos):
                    ps = np.full(3, float(pos_sigma[0])) if pos_sigma.size == 1 else pos_sigma[:3]
                    pb = (
                        np.zeros(3, dtype=float)
                        if pos_bias.size == 0
                        else np.full(3, float(pos_bias[0]))
                        if pos_bias.size == 1
                        else pos_bias[:3]
                    )
                    usable = ps > 0.0
                    norm_hist_values.append(((meas_pos_err[valid_pos][:, usable] - pb[usable]) / ps[usable]).reshape(-1))
                    pos_sigma_markers_m.append((float(np.mean(pb)) * 1000.0, float(np.sqrt(np.mean(ps**2))) * 1000.0))
                if vel_sigma.size in (1, 3) and np.any(vel_sigma > 0.0) and np.any(valid_vel):
                    vs = np.full(3, float(vel_sigma[0])) if vel_sigma.size == 1 else vel_sigma[:3]
                    vb = (
                        np.zeros(3, dtype=float)
                        if vel_bias.size == 0
                        else np.full(3, float(vel_bias[0]))
                        if vel_bias.size == 1
                        else vel_bias[:3]
                    )
                    usable = vs > 0.0
                    norm_hist_values.append(((meas_vel_err[valid_vel][:, usable] - vb[usable]) / vs[usable]).reshape(-1))
                    vel_sigma_markers_mm_s.append(
                        (float(np.mean(vb)) * 1.0e6, float(np.sqrt(np.mean(vs**2))) * 1.0e6)
                    )
            plotted = True

    def _unique_sigma_markers(markers: list[tuple[float, float]]) -> list[tuple[float, float]]:
        unique: list[tuple[float, float]] = []
        for bias, sigma in markers:
            if not any(np.isclose(bias, b, rtol=1e-9, atol=1e-12) and np.isclose(sigma, s, rtol=1e-9, atol=1e-12) for b, s in unique):
                unique.append((bias, sigma))
        return unique

    if pos_hist_values:
        values_m = np.concatenate(pos_hist_values) * 1000.0
        finite = values_m[np.isfinite(values_m)]
        if finite.size:
            ax_pos_hist.hist(finite, bins=40, density=True, alpha=0.75, color="tab:blue", label="residuals")
            ax_pos_hist.axvline(float(np.mean(finite)), color="black", linestyle="--", linewidth=1.0, label="mean")
            for marker_idx, (bias_m, sigma_m) in enumerate(_unique_sigma_markers(pos_sigma_markers_m)):
                label = "+/- cfg sigma" if marker_idx == 0 else None
                ax_pos_hist.axvline(bias_m - sigma_m, color="tab:red", linestyle=":", linewidth=1.0, label=label)
                ax_pos_hist.axvline(bias_m + sigma_m, color="tab:red", linestyle=":", linewidth=1.0)
            ax_pos_hist.legend(loc="best")
    if vel_hist_values:
        values_mm_s = np.concatenate(vel_hist_values) * 1.0e6
        finite = values_mm_s[np.isfinite(values_mm_s)]
        if finite.size:
            ax_vel_hist.hist(finite, bins=40, density=True, alpha=0.75, color="tab:orange", label="residuals")
            ax_vel_hist.axvline(float(np.mean(finite)), color="black", linestyle="--", linewidth=1.0, label="mean")
            for marker_idx, (bias_mm_s, sigma_mm_s) in enumerate(_unique_sigma_markers(vel_sigma_markers_mm_s)):
                label = "+/- cfg sigma" if marker_idx == 0 else None
                ax_vel_hist.axvline(
                    bias_mm_s - sigma_mm_s, color="tab:red", linestyle=":", linewidth=1.0, label=label
                )
                ax_vel_hist.axvline(bias_mm_s + sigma_mm_s, color="tab:red", linestyle=":", linewidth=1.0)
            ax_vel_hist.legend(loc="best")
    if norm_hist_values:
        finite = np.concatenate(norm_hist_values)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            ax_norm_hist.hist(finite, bins=40, density=True, alpha=0.7, color="tab:green", label="normalized residuals")
            x = np.linspace(-4.0, 4.0, 241)
            pdf = np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)
            ax_norm_hist.plot(x, pdf, color="black", linewidth=1.1, label="N(0,1)")
            ax_norm_hist.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
            ax_norm_hist.legend(loc="best")
    if not plotted:
        for ax in axes.reshape(-1):
            ax.text(
                0.5,
                0.5,
                "No truth/measurement/estimate chain available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
    ax_range.set_title("Truth / Measurement / Estimate Range")
    ax_range.set_ylabel("km")
    ax_pos.set_title("Position Error Norm")
    ax_pos.set_ylabel("km")
    ax_vel.set_title("Velocity Error Norm")
    ax_vel.set_ylabel("m/s")
    ax_vel.set_xlabel("time (s)")
    ax_pos_hist.set_title("Position Measurement Residuals")
    ax_pos_hist.set_xlabel("measurement - truth (m)")
    ax_pos_hist.set_ylabel("density")
    ax_vel_hist.set_title("Velocity Measurement Residuals")
    ax_vel_hist.set_xlabel("measurement - truth (mm/s)")
    ax_vel_hist.set_ylabel("density")
    ax_norm_hist.set_title("Normalized Measurement Residuals")
    ax_norm_hist.set_xlabel("(measurement - truth - bias) / sigma")
    ax_norm_hist.set_ylabel("density")
    for ax in axes.reshape(-1):
        ax.grid(True, alpha=0.3)
    for ax in (ax_range, ax_pos, ax_vel):
        if ax.lines or ax.collections:
            ax.legend(loc="best")
    _save_show_close(fig, out_path=out_path, show=show, close=close, dpi=dpi)
    return fig


def plot_sensor_access(
    payload: dict[str, Any] | None = None,
    *,
    t_s: np.ndarray | None = None,
    truth_by_object: ArrayMap | None = None,
    knowledge_by_observer: NestedArrayMap | None = None,
    out_path: str | Path | None = None,
    show: bool = False,
    close: bool = False,
    dpi: int = 150,
) -> plt.Figure:
    if payload is not None:
        t_s, truth_by_object, _, _, knowledge_by_observer, _ = _payload_arrays(payload)
    t = np.array([] if t_s is None else t_s, dtype=float).reshape(-1)
    truth = dict(truth_by_object or {})
    knowledge = dict(knowledge_by_observer or {})
    pairs: list[tuple[str, str, np.ndarray]] = []
    for obs, by_target in sorted(knowledge.items()):
        for target, hist in sorted(by_target.items()):
            if hist.ndim == 2 and hist.shape[0] > 0:
                pairs.append((obs, target, hist))

    fig, axes = plt.subplots(3, 1, figsize=cap_figsize(12, 9), sharex=True)
    if not pairs:
        for ax in axes:
            ax.text(0.5, 0.5, "No knowledge history available", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        _save_show_close(fig, out_path=out_path, show=show, close=close, dpi=dpi)
        return fig

    y_ticks = []
    y_labels = []
    for row, (obs, target, hist) in enumerate(pairs):
        n = min(hist.shape[0], t.size)
        known = np.any(np.isfinite(hist[:n, :]), axis=1).astype(float)
        axes[0].step(t[:n], known + row * 1.25, where="post", linewidth=1.4)
        y_ticks.append(row * 1.25 + 0.5)
        y_labels.append(f"{obs}->{target}")

        obs_truth = truth.get(obs)
        target_truth = truth.get(target)
        if (
            obs_truth is not None
            and target_truth is not None
            and obs_truth.shape[1] >= 3
            and target_truth.shape[1] >= 3
        ):
            nr = min(obs_truth.shape[0], target_truth.shape[0], t.size)
            rel = target_truth[:nr, :3] - obs_truth[:nr, :3]
            axes[1].plot(t[:nr], np.linalg.norm(rel, axis=1), label=f"{obs}->{target}")

        if hist.shape[1] >= 6 and target_truth is not None and target_truth.shape[1] >= 6:
            ne = min(hist.shape[0], target_truth.shape[0], t.size)
            err = hist[:ne, :6] - target_truth[:ne, :6]
            finite = np.all(np.isfinite(err[:, :3]), axis=1)
            pos_err = np.full(ne, np.nan, dtype=float)
            pos_err[finite] = np.linalg.norm(err[finite, :3], axis=1)
            axes[2].plot(t[:ne], pos_err, label=f"{obs}->{target}")

    axes[0].set_title("Sensor / Knowledge Access Timeline")
    axes[0].set_ylabel("access")
    axes[0].set_yticks(y_ticks)
    axes[0].set_yticklabels(y_labels)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-0.2, max(y_ticks) + 0.95)

    axes[1].set_title("Observer-Target Range")
    axes[1].set_ylabel("km")
    axes[1].grid(True, alpha=0.3)
    if axes[1].lines:
        axes[1].legend(loc="best")

    axes[2].set_title("Knowledge Position Error vs Target Truth")
    axes[2].set_ylabel("km")
    axes[2].set_xlabel("time (s)")
    axes[2].grid(True, alpha=0.3)
    if axes[2].lines:
        axes[2].legend(loc="best")

    fig.tight_layout()
    _save_show_close(fig, out_path=out_path, show=show, close=close, dpi=dpi)
    return fig


def plot_ground_track_from_payload(
    payload: dict[str, Any] | None = None,
    *,
    t_s: np.ndarray | None = None,
    truth_by_object: ArrayMap | None = None,
    jd_utc_start: float | None = None,
    object_id: str | None = None,
    draw_earth_map: bool = False,
    out_path: str | Path | None = None,
    show: bool = False,
    close: bool = False,
    dpi: int = 150,
) -> plt.Figure:
    if payload is not None:
        t_s, truth_by_object, _, _, _, _ = _payload_arrays(payload)
        frame_context = frame_context_from_mapping(
            dict(payload.get("frame_provenance", {}) or {}),
            jd_utc_start=jd_utc_start,
            source="payload",
        )
    else:
        frame_context = frame_context_from_mapping({}, jd_utc_start=jd_utc_start, source="plot")
    t = np.array([] if t_s is None else t_s, dtype=float).reshape(-1)
    truth = dict(truth_by_object or {})
    ids = [object_id] if object_id and object_id in truth else sorted(truth.keys())
    fig, ax, is_cartopy = _setup_ground_track_axes(title="Ground Track", draw_earth_map=bool(draw_earth_map))
    for oid in ids:
        hist = truth.get(oid)
        if hist is None or hist.shape[1] < 3:
            continue
        n = min(hist.shape[0], t.size)
        lat, lon, _ = ground_track_from_eci_history(
            hist[:n, :3],
            t_s=t[:n],
            jd_utc_start=jd_utc_start,
            frame_context=frame_context,
        )
        lon_p, lat_p = split_ground_track_dateline(lon_deg=lon, lat_deg=lat, jump_threshold_deg=180.0)
        if is_cartopy:
            import cartopy.crs as ccrs  # type: ignore

            ax.plot(lon_p, lat_p, linewidth=1.4, label=oid, transform=ccrs.PlateCarree(), zorder=3)
        else:
            ax.plot(lon_p, lat_p, linewidth=1.4, label=oid)
        finite = np.isfinite(lon) & np.isfinite(lat)
        idx = np.where(finite)[0]
        if idx.size:
            if is_cartopy:
                ax.scatter(
                    [lon[idx[0]]],
                    [lat[idx[0]]],
                    color="green",
                    s=18,
                    transform=ccrs.PlateCarree(),
                    zorder=4,
                )
                ax.scatter(
                    [lon[idx[-1]]],
                    [lat[idx[-1]]],
                    color="red",
                    s=18,
                    transform=ccrs.PlateCarree(),
                    zorder=4,
                )
            else:
                ax.scatter([lon[idx[0]]], [lat[idx[0]]], color="green", s=18)
                ax.scatter([lon[idx[-1]]], [lat[idx[-1]]], color="red", s=18)
    if ids:
        ax.legend(loc="best")
    fig.tight_layout()
    _save_show_close(fig, out_path=out_path, show=show, close=close, dpi=dpi)
    return fig


def plot_orbital_element(
    payload: dict[str, Any] | None = None,
    *,
    element_id: str,
    t_s: np.ndarray | None = None,
    truth_by_object: ArrayMap | None = None,
    object_id: str | None = None,
    out_path: str | Path | None = None,
    show: bool = False,
    close: bool = False,
    dpi: int = 150,
) -> plt.Figure:
    if payload is not None:
        t_s, truth_by_object, _, _, _, _ = _payload_arrays(payload)
    element_key = str(element_id or "").strip()
    if element_key not in ORBITAL_ELEMENT_SPECS:
        valid = ", ".join(sorted(ORBITAL_ELEMENT_SPECS))
        raise ValueError(f"Unknown orbital element '{element_id}'. Valid elements: {valid}")
    t = np.array([] if t_s is None else t_s, dtype=float).reshape(-1)
    truth = dict(truth_by_object or {})
    title, unit = ORBITAL_ELEMENT_SPECS[element_key]

    fig, ax = plt.subplots(figsize=cap_figsize(10, 5))
    plotted = _plot_element_on_axis(
        ax,
        t_s=t,
        truth_by_object=truth,
        element_id=element_key,
        object_id=object_id,
    )
    if not plotted:
        ax.text(0.5, 0.5, "No valid COE samples available", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(f"{title} Over Time")
    ax.set_xlabel("time (s)")
    ax.set_ylabel(title if not unit else f"{title} ({unit})")
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend(loc="best")
    fig.tight_layout()
    _save_show_close(fig, out_path=out_path, show=show, close=close, dpi=dpi)
    return fig


def plot_orbital_elements_summary(
    payload: dict[str, Any] | None = None,
    *,
    t_s: np.ndarray | None = None,
    truth_by_object: ArrayMap | None = None,
    object_id: str | None = None,
    out_path: str | Path | None = None,
    show: bool = False,
    close: bool = False,
    dpi: int = 150,
) -> plt.Figure:
    if payload is not None:
        t_s, truth_by_object, _, _, _, _ = _payload_arrays(payload)
    t = np.array([] if t_s is None else t_s, dtype=float).reshape(-1)
    truth = dict(truth_by_object or {})

    fig, axes = plt.subplots(3, 2, figsize=cap_figsize(13, 10), sharex=True)
    for ax, element_key in zip(axes.ravel(), ORBITAL_ELEMENT_SPECS.keys()):
        title, unit = ORBITAL_ELEMENT_SPECS[element_key]
        plotted = _plot_element_on_axis(
            ax,
            t_s=t,
            truth_by_object=truth,
            element_id=element_key,
            object_id=object_id,
        )
        if not plotted:
            ax.text(0.5, 0.5, "No valid samples", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_ylabel(unit or title)
        ax.grid(True, alpha=0.3)
        if plotted:
            ax.legend(loc="best")
    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 1].set_xlabel("time (s)")
    fig.suptitle("Classical Orbital Elements")
    fig.tight_layout()
    _save_show_close(fig, out_path=out_path, show=show, close=close, dpi=dpi)
    return fig


def plot_orbital_elements_angles(
    payload: dict[str, Any] | None = None,
    *,
    t_s: np.ndarray | None = None,
    truth_by_object: ArrayMap | None = None,
    object_id: str | None = None,
    out_path: str | Path | None = None,
    show: bool = False,
    close: bool = False,
    dpi: int = 150,
) -> plt.Figure:
    if payload is not None:
        t_s, truth_by_object, _, _, _, _ = _payload_arrays(payload)
    t = np.array([] if t_s is None else t_s, dtype=float).reshape(-1)
    truth = dict(truth_by_object or {})
    angle_ids = ("inc", "raan", "argp", "true_anomaly")

    fig, ax = plt.subplots(figsize=cap_figsize(11, 5.5))
    plotted = False
    for element_key in angle_ids:
        plotted = (
            _plot_element_on_axis(
                ax,
                t_s=t,
                truth_by_object=truth,
                element_id=element_key,
                object_id=object_id,
                label_prefix=True,
            )
            or plotted
        )
    if not plotted:
        ax.text(0.5, 0.5, "No valid angular COE samples available", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Orbital Element Angles")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("deg")
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    _save_show_close(fig, out_path=out_path, show=show, close=close, dpi=dpi)
    return fig


def plot_ground_station_access(
    payload: dict[str, Any] | None = None,
    *,
    t_s: np.ndarray | None = None,
    ground_station_access: dict[str, Any] | None = None,
    out_path: str | Path | None = None,
    show: bool = False,
    close: bool = False,
    dpi: int = 150,
) -> plt.Figure:
    if payload is not None:
        t_s = np.array(payload.get("time_s", []), dtype=float).reshape(-1)
        ground_station_access = dict(payload.get("ground_station_access", {}) or {})
    t = np.array([] if t_s is None else t_s, dtype=float).reshape(-1)
    access_root = dict(ground_station_access or {})

    pairs: list[tuple[str, str, dict[str, Any]]] = []
    for station_id, station_payload in sorted(access_root.items()):
        targets = dict(dict(station_payload or {}).get("targets", {}) or {})
        for target_id, target_payload in sorted(targets.items()):
            pairs.append((str(station_id), str(target_id), dict(target_payload or {})))

    fig, axes = plt.subplots(3, 1, figsize=cap_figsize(12, 9), sharex=True)
    if not pairs or t.size == 0:
        for ax in axes:
            ax.text(
                0.5,
                0.5,
                "No ground-station access history available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        fig.tight_layout()
        _save_show_close(fig, out_path=out_path, show=show, close=close, dpi=dpi)
        return fig

    y_ticks = []
    y_labels = []
    for row, (station_id, target_id, target_payload) in enumerate(pairs):
        label = f"{station_id}->{target_id}"
        access = np.array(target_payload.get("access", []), dtype=float).reshape(-1)
        los = np.array(target_payload.get("line_of_sight", []), dtype=float).reshape(-1)
        elev_values = list(target_payload.get("elevation_deg", []) or [])
        range_values = list(target_payload.get("range_km", []) or [])
        elev = np.array([float("nan") if value is None else float(value) for value in elev_values], dtype=float)
        rng = np.array([float("nan") if value is None else float(value) for value in range_values], dtype=float)
        n_access = min(t.size, access.size)
        if n_access:
            axes[0].step(t[:n_access], access[:n_access] + row * 1.3, where="post", linewidth=1.5)
        n_los = min(t.size, los.size)
        if n_los:
            axes[0].step(
                t[:n_los],
                0.35 * los[:n_los] + row * 1.3,
                where="post",
                linewidth=0.9,
                linestyle=":",
                alpha=0.65,
            )
        n_elev = min(t.size, elev.size)
        if n_elev:
            axes[1].plot(t[:n_elev], elev[:n_elev], linewidth=1.2, label=label)
        n_rng = min(t.size, rng.size)
        if n_rng:
            axes[2].plot(t[:n_rng], rng[:n_rng], linewidth=1.2, label=label)
        y_ticks.append(row * 1.3 + 0.5)
        y_labels.append(label)

    axes[0].set_title("Ground-Station Access Timeline")
    axes[0].set_ylabel("access")
    axes[0].set_yticks(y_ticks)
    axes[0].set_yticklabels(y_labels)
    axes[0].set_ylim(-0.2, max(y_ticks) + 0.95)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Elevation")
    axes[1].set_ylabel("deg")
    axes[1].grid(True, alpha=0.3)
    if axes[1].lines:
        axes[1].legend(loc="best")

    axes[2].set_title("Slant Range")
    axes[2].set_ylabel("km")
    axes[2].set_xlabel("time (s)")
    axes[2].grid(True, alpha=0.3)
    if axes[2].lines:
        axes[2].legend(loc="best")

    fig.tight_layout()
    _save_show_close(fig, out_path=out_path, show=show, close=close, dpi=dpi)
    return fig


def plot_attitude_control_summary(
    payload: dict[str, Any] | None = None,
    *,
    t_s: np.ndarray | None = None,
    truth_by_object: ArrayMap | None = None,
    thrust_by_object: ArrayMap | None = None,
    desired_attitude_by_object: ArrayMap | None = None,
    thrust_axis_body_by_object: dict[str, np.ndarray] | None = None,
    object_id: str | None = None,
    out_path: str | Path | None = None,
    show: bool = False,
    close: bool = False,
    dpi: int = 150,
) -> plt.Figure:
    if payload is not None:
        t_s, truth_by_object, thrust_by_object, _, _, _ = _payload_arrays(payload)
        desired_attitude_by_object = _array_map(payload.get("desired_attitude_by_object", {}))
    t = np.array([] if t_s is None else t_s, dtype=float).reshape(-1)
    truth = dict(truth_by_object or {})
    thrust = dict(thrust_by_object or {})
    desired = dict(desired_attitude_by_object or {})
    axes_by_object = dict(thrust_axis_body_by_object or {})
    ids = [object_id] if object_id and object_id in truth else sorted(truth.keys())

    fig, axes = plt.subplots(4, 1, figsize=cap_figsize(12, 10), sharex=True)
    plotted = False
    for oid in ids:
        hist = truth.get(oid)
        if hist is None or hist.ndim != 2 or hist.shape[1] < 13:
            continue
        n = min(hist.shape[0], t.size)
        if n <= 0:
            continue
        q_err = _quat_error_series_deg(
            truth_hist=hist,
            desired_attitude_hist=desired.get(oid),
            n_s=n,
        )
        rate_norm = np.linalg.norm(np.nan_to_num(hist[:n, 10:13], nan=0.0), axis=1)
        u = thrust.get(oid, np.zeros((n, 3), dtype=float))
        n_u = min(u.shape[0], n)
        thrust_mag = np.full(n, np.nan, dtype=float)
        if n_u > 0 and u.ndim == 2 and u.shape[1] >= 3:
            thrust_mag[:n_u] = np.linalg.norm(np.nan_to_num(u[:n_u, :3], nan=0.0), axis=1)
            align = _thrust_alignment_error_deg_series(
                truth_hist=hist,
                thrust_hist=u,
                thrust_axis_body=axes_by_object.get(oid, np.array([1.0, 0.0, 0.0], dtype=float)),
                n_s=n_u,
            )
        else:
            align = np.full(n, np.nan, dtype=float)
        axes[0].plot(t[:n], q_err, linewidth=1.2, label=oid)
        axes[1].plot(t[:n], rate_norm, linewidth=1.2, label=oid)
        axes[2].plot(t[:n], thrust_mag, linewidth=1.2, label=oid)
        axes[3].plot(t[: align.size], align, linewidth=1.2, label=oid)
        plotted = True

    if not plotted:
        for ax in axes:
            ax.text(
                0.5,
                0.5,
                "No attitude-control history available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    axes[0].set_title("Quaternion Tracking Error")
    axes[0].set_ylabel("deg")
    axes[1].set_title("Body Rate Norm")
    axes[1].set_ylabel("rad/s")
    axes[2].set_title("Applied Thrust Magnitude")
    axes[2].set_ylabel("km/s^2")
    axes[3].set_title("Thrust Alignment Error")
    axes[3].set_ylabel("deg")
    axes[3].set_xlabel("time (s)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        if plotted:
            ax.legend(loc="best")
    fig.tight_layout()
    _save_show_close(fig, out_path=out_path, show=show, close=close, dpi=dpi)
    return fig
