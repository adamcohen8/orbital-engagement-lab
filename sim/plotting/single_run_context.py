from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np

from sim.dynamics.orbit.environment import EARTH_RADIUS_KM
from sim.plotting.style import role_color, show_save_close_oel
from sim.utils.frames import ric_dcm_ir_from_rv, ric_rect_to_curv
from sim.utils.plot_windows import RIC_FOLLOW_MARGIN, windows_from_points
from sim.utils.plotting import _draw_earth_sphere_3d

ArrayMap = dict[str, np.ndarray]
NestedArrayMap = dict[str, dict[str, np.ndarray]]
OrbitalElementSeriesCache = dict[
    str,
    tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]],
]
RICSummaryFrame = Literal["rectangular", "curvilinear"]

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
