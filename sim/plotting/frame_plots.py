from __future__ import annotations

from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np

from sim.dynamics.orbit.environment import EARTH_RADIUS_KM
from sim.dynamics.orbit.frames import FrameContext, frame_context_from_mapping, transform_position
from sim.plotting.capability_common import _show_save_close
from sim.utils.figure_size import cap_figsize
from sim.utils.frames import ric_dcm_ir_from_rv, ric_rect_to_curv
from sim.utils.quaternion import dcm_to_quaternion_bn, quaternion_to_dcm_bn

PlotMode = Literal["interactive", "save", "both"]
FrameName = Literal["eci", "ecef", "ric_rect", "ric_curv"]
AttitudeFrame = Literal["eci", "ric"]
Layout = Literal["single", "subplots"]


def _truth_quaternion_in_frame(truth_hist: np.ndarray, frame: AttitudeFrame) -> np.ndarray:
    q_bn = np.array(truth_hist[:, 6:10], dtype=float)
    if frame == "eci":
        return q_bn
    out = np.zeros_like(q_bn)
    for k in range(truth_hist.shape[0]):
        r = truth_hist[k, 0:3]
        v = truth_hist[k, 3:6]
        c_bn = quaternion_to_dcm_bn(q_bn[k, :])
        c_ir = ric_dcm_ir_from_rv(r, v)
        c_br = c_bn @ c_ir
        out[k, :] = dcm_to_quaternion_bn(c_br)
    return out


def _rates_in_frame(truth_hist: np.ndarray, frame: AttitudeFrame) -> np.ndarray:
    w_body = np.array(truth_hist[:, 10:13], dtype=float)
    out = np.zeros_like(w_body)
    q_bn = np.array(truth_hist[:, 6:10], dtype=float)
    for k in range(truth_hist.shape[0]):
        r = truth_hist[k, 0:3]
        v = truth_hist[k, 3:6]
        c_bn = quaternion_to_dcm_bn(q_bn[k, :])
        if frame == "eci":
            out[k, :] = c_bn.T @ w_body[k, :]
            continue
        c_ir = ric_dcm_ir_from_rv(r, v)
        c_br = c_bn @ c_ir
        out[k, :] = c_br.T @ w_body[k, :]
    return out


def plot_quaternion_components(
    t_s: np.ndarray,
    truth_hist: np.ndarray,
    *,
    frame: AttitudeFrame = "eci",
    layout: Layout = "single",
    mode: PlotMode = "interactive",
    out_path: str | None = None,
) -> None:
    q = _truth_quaternion_in_frame(truth_hist, frame)
    labels = ["q0", "q1", "q2", "q3"]
    if layout == "single":
        fig, ax = plt.subplots(figsize=cap_figsize(10, 5))
        for i in range(4):
            ax.plot(t_s, q[:, i], label=labels[i])
        ax.set_title(f"Quaternion Components ({frame.upper()} frame)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Quaternion")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    else:
        fig, axes = plt.subplots(4, 1, figsize=cap_figsize(10, 9), sharex=True)
        for i, ax in enumerate(axes):
            ax.plot(t_s, q[:, i], linewidth=1.3)
            ax.set_ylabel(labels[i])
            ax.grid(True, alpha=0.3)
        axes[0].set_title(f"Quaternion Components ({frame.upper()} frame)")
        axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    _show_save_close(fig, mode=mode, out_path=out_path)


def plot_body_rates(
    t_s: np.ndarray,
    truth_hist: np.ndarray,
    *,
    frame: AttitudeFrame = "eci",
    layout: Layout = "subplots",
    mode: PlotMode = "interactive",
    out_path: str | None = None,
) -> None:
    w = _rates_in_frame(truth_hist, frame)
    labels = ["wx", "wy", "wz"]
    if layout == "single":
        fig, ax = plt.subplots(figsize=cap_figsize(10, 5))
        for i in range(3):
            ax.plot(t_s, w[:, i], label=labels[i])
        ax.set_title(f"Angular Velocity Components ({frame.upper()} frame)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("rad/s")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    else:
        fig, axes = plt.subplots(3, 1, figsize=cap_figsize(10, 8), sharex=True)
        for i, ax in enumerate(axes):
            ax.plot(t_s, w[:, i], linewidth=1.3)
            ax.set_ylabel(f"{labels[i]} (rad/s)")
            ax.grid(True, alpha=0.3)
        axes[0].set_title(f"Angular Velocity Components ({frame.upper()} frame)")
        axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    _show_save_close(fig, mode=mode, out_path=out_path)


def _trajectory_in_frame(
    t_s: np.ndarray,
    truth_hist: np.ndarray,
    *,
    frame: FrameName,
    jd_utc_start: float | None = None,
    frame_context: FrameContext | None = None,
    reference_truth_hist: np.ndarray | None = None,
) -> np.ndarray:
    r_eci = np.array(truth_hist[:, 0:3], dtype=float)
    if frame == "eci":
        return r_eci
    if frame == "ecef":
        frame_ctx = frame_context or frame_context_from_mapping(
            {},
            jd_utc_start=jd_utc_start,
            source="plot",
        )
        out = np.zeros_like(r_eci)
        for k in range(r_eci.shape[0]):
            out[k, :] = transform_position(r_eci[k, :], "eci", "ecef", t_s=float(t_s[k]), context=frame_ctx)
        return out
    if reference_truth_hist is None:
        raise ValueError("reference_truth_hist is required for RIC frame plots.")
    r_ref = np.array(reference_truth_hist[:, 0:3], dtype=float)
    v_ref = np.array(reference_truth_hist[:, 3:6], dtype=float)
    rel_rect = np.zeros_like(r_eci)
    for k in range(r_eci.shape[0]):
        c_ir = ric_dcm_ir_from_rv(r_ref[k, :], v_ref[k, :])
        rel_rect[k, :] = c_ir.T @ (r_eci[k, :] - r_ref[k, :])
    if frame == "ric_rect":
        return rel_rect
    out = np.zeros_like(rel_rect)
    for k in range(rel_rect.shape[0]):
        x_rect = np.hstack((rel_rect[k, :], np.zeros(3)))
        x_curv = ric_rect_to_curv(x_rect, r0_km=float(np.linalg.norm(r_ref[k, :])))
        out[k, :] = x_curv[:3]
    return out


def _first_last_finite_indices(r: np.ndarray) -> tuple[int | None, int | None]:
    arr = np.array(r, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return None, None
    mask = np.all(np.isfinite(arr), axis=1)
    idx = np.where(mask)[0]
    if idx.size == 0:
        return None, None
    return int(idx[0]), int(idx[-1])


def _draw_earth_sphere_3d(ax: Any, radius_km: float = EARTH_RADIUS_KM) -> None:
    u = np.linspace(0.0, 2.0 * np.pi, 48)
    v = np.linspace(0.0, np.pi, 24)
    x = radius_km * np.outer(np.cos(u), np.sin(v))
    y = radius_km * np.outer(np.sin(u), np.sin(v))
    z = radius_km * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, rstride=1, cstride=1, color="#6EA8D9", alpha=0.18, linewidth=0.0, zorder=0)
    ax.plot_wireframe(x, y, z, rstride=6, cstride=6, color="#5D86AA", alpha=0.15, linewidth=0.4, zorder=0)


def _bottom_center_figure_legend(fig: plt.Figure, handles: list[Any], labels: list[str]) -> None:
    if not handles:
        return
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=len(labels),
        borderaxespad=0.0,
        fontsize="small",
        handlelength=1.6,
        handletextpad=0.5,
        columnspacing=0.9,
    )


def _reference_origin_label(reference_label: str | None) -> str | None:
    label = str(reference_label or "").strip()
    return label or None


def _draw_ric_reference_origin_3d(ax: Any, *, label: str | None) -> None:
    legend_label = _reference_origin_label(label)
    if legend_label is None:
        return
    ax.scatter(
        [0.0],
        [0.0],
        [0.0],
        marker="*",
        s=90,
        color="#F8FAFC",
        edgecolors="#111827",
        linewidths=0.7,
        label=legend_label,
        zorder=8,
    )


def _draw_ric_reference_origin_2d(axes: list[Any], *, label: str | None) -> None:
    legend_label = _reference_origin_label(label)
    if legend_label is None:
        return
    for idx, ax in enumerate(axes):
        ax.scatter(
            [0.0],
            [0.0],
            marker="*",
            s=80,
            color="#F8FAFC",
            edgecolors="#111827",
            linewidths=0.7,
            label=legend_label if idx == 0 else "_nolegend_",
            zorder=8,
        )


def plot_trajectory_frame(
    t_s: np.ndarray,
    truth_hist: np.ndarray,
    *,
    frame: FrameName = "eci",
    jd_utc_start: float | None = None,
    frame_context: FrameContext | None = None,
    reference_truth_hist: np.ndarray | None = None,
    reference_label: str | None = None,
    mode: PlotMode = "interactive",
    out_path: str | None = None,
) -> None:
    r = _trajectory_in_frame(
        t_s=t_s,
        truth_hist=truth_hist,
        frame=frame,
        jd_utc_start=jd_utc_start,
        frame_context=frame_context,
        reference_truth_hist=reference_truth_hist,
    )
    fig = plt.figure(figsize=cap_figsize(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    if frame in ("ric_rect", "ric_curv"):
        # Display RIC with radial on y-axis: x=I, y=R, z=C.
        ix, iy, iz = 1, 0, 2
        xlbl, ylbl, zlbl = "I", "R", "C"
    else:
        ix, iy, iz = 0, 1, 2
        xlbl, ylbl, zlbl = "x", "y", "z"
        if frame in ("eci", "ecef"):
            _draw_earth_sphere_3d(ax)
    ax.plot(r[:, ix], r[:, iy], r[:, iz], linewidth=1.4)
    i0, i1 = _first_last_finite_indices(r)
    if i0 is not None:
        ax.scatter([r[i0, ix]], [r[i0, iy]], [r[i0, iz]], color="green", s=28, zorder=5)
    if i1 is not None:
        ax.scatter([r[i1, ix]], [r[i1, iy]], [r[i1, iz]], color="red", s=28, zorder=5)
    ax.set_title(f"Trajectory ({frame.upper()})")
    ax.set_xlabel(xlbl)
    ax.set_ylabel(ylbl)
    ax.set_zlabel(zlbl)
    fig.tight_layout()
    _show_save_close(fig, mode=mode, out_path=out_path)


def plot_multi_trajectory_frame(
    t_s: np.ndarray,
    truth_hist_by_object: dict[str, np.ndarray],
    *,
    frame: FrameName = "eci",
    jd_utc_start: float | None = None,
    frame_context: FrameContext | None = None,
    reference_truth_hist: np.ndarray | None = None,
    reference_label: str | None = None,
    mode: PlotMode = "interactive",
    out_path: str | None = None,
) -> None:
    fig = plt.figure(figsize=cap_figsize(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    if frame in ("ric_rect", "ric_curv"):
        # Display RIC with radial on y-axis: x=I, y=R, z=C.
        ix, iy, iz = 1, 0, 2
        xlbl, ylbl, zlbl = "I", "R", "C"
    else:
        ix, iy, iz = 0, 1, 2
        xlbl, ylbl, zlbl = "x", "y", "z"
        if frame in ("eci", "ecef"):
            _draw_earth_sphere_3d(ax)
    if frame in ("ric_rect", "ric_curv"):
        _draw_ric_reference_origin_3d(ax, label=reference_label)
    for oid, hist in truth_hist_by_object.items():
        if hist.size == 0 or not np.any(np.isfinite(hist[:, 0])):
            continue
        r = _trajectory_in_frame(
            t_s=t_s,
            truth_hist=hist,
            frame=frame,
            jd_utc_start=jd_utc_start,
            frame_context=frame_context,
            reference_truth_hist=reference_truth_hist,
        )
        ax.plot(r[:, ix], r[:, iy], r[:, iz], linewidth=1.4, label=oid)
        i0, i1 = _first_last_finite_indices(r)
        if i0 is not None:
            ax.scatter([r[i0, ix]], [r[i0, iy]], [r[i0, iz]], color="green", s=24, zorder=5)
        if i1 is not None:
            ax.scatter([r[i1, ix]], [r[i1, iy]], [r[i1, iz]], color="red", s=24, zorder=5)
    ax.set_title(f"Trajectories ({frame.upper()})")
    ax.set_xlabel(xlbl)
    ax.set_ylabel(ylbl)
    ax.set_zlabel(zlbl)
    handles, labels = ax.get_legend_handles_labels()
    _bottom_center_figure_legend(fig, handles, labels)
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 1.0))
    _show_save_close(fig, mode=mode, out_path=out_path)


def _ric_2d_plane_axes(plane: str) -> tuple[int, int, str, str]:
    p = str(plane).strip().lower()
    if p == "ri":
        return 1, 0, "I", "R"
    if p == "ic":
        return 1, 2, "I", "C"
    if p == "rc":
        return 2, 0, "C", "R"
    raise ValueError("plane must be one of: 'ri', 'ic', 'rc'.")


def plot_ric_2d_projections(
    t_s: np.ndarray,
    truth_hist: np.ndarray,
    *,
    frame: Literal["ric_rect", "ric_curv"] = "ric_rect",
    reference_truth_hist: np.ndarray,
    reference_label: str | None = None,
    planes: list[str] | None = None,
    mode: PlotMode = "interactive",
    out_path: str | None = None,
) -> None:
    if frame not in ("ric_rect", "ric_curv"):
        raise ValueError("frame must be 'ric_rect' or 'ric_curv'.")
    r = _trajectory_in_frame(
        t_s=t_s,
        truth_hist=truth_hist,
        frame=frame,
        reference_truth_hist=reference_truth_hist,
    )
    p_list = planes if planes is not None and len(planes) > 0 else ["ri", "ic", "rc"]
    fig, axes = plt.subplots(1, len(p_list), figsize=cap_figsize(5.0 * len(p_list), 4.5))
    if len(p_list) == 1:
        axes = [axes]
    for ax, p in zip(axes, p_list):
        ix, iy, xlbl, ylbl = _ric_2d_plane_axes(p)
        ax.plot(r[:, ix], r[:, iy], linewidth=1.4)
        i0, i1 = _first_last_finite_indices(r[:, [ix, iy]])
        if i0 is not None:
            ax.scatter([r[i0, ix]], [r[i0, iy]], color="green", s=24, zorder=5)
        if i1 is not None:
            ax.scatter([r[i1, ix]], [r[i1, iy]], color="red", s=24, zorder=5)
        ax.set_xlabel(xlbl)
        ax.set_ylabel(ylbl)
        ax.set_title(f"{xlbl}-{ylbl}")
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"RIC 2D Projections ({'Rect' if frame == 'ric_rect' else 'Curvilinear'})")
    fig.tight_layout()
    _show_save_close(fig, mode=mode, out_path=out_path)


def plot_multi_ric_2d_projections(
    t_s: np.ndarray,
    truth_hist_by_object: dict[str, np.ndarray],
    *,
    frame: Literal["ric_rect", "ric_curv"] = "ric_rect",
    reference_truth_hist: np.ndarray,
    reference_label: str | None = None,
    burn_marker_by_object: dict[str, np.ndarray] | None = None,
    burn_marker_object_ids: list[str] | None = None,
    burn_marker_threshold_km_s2: float = 1.0e-12,
    planes: list[str] | None = None,
    mode: PlotMode = "interactive",
    out_path: str | None = None,
) -> None:
    if frame not in ("ric_rect", "ric_curv"):
        raise ValueError("frame must be 'ric_rect' or 'ric_curv'.")
    p_list = planes if planes is not None and len(planes) > 0 else ["ri", "ic", "rc"]
    fig, axes = plt.subplots(1, len(p_list), figsize=cap_figsize(5.0 * len(p_list), 5.5))
    if len(p_list) == 1:
        axes = [axes]
    _draw_ric_reference_origin_2d(list(axes), label=reference_label)
    burn_sources = dict(burn_marker_by_object or {})
    burn_object_ids = (
        [str(oid) for oid in burn_marker_object_ids]
        if burn_marker_object_ids is not None
        else sorted(str(oid) for oid in burn_sources.keys())
    )
    burn_marker_labeled: set[str] = set()
    for oid, hist in truth_hist_by_object.items():
        if hist.size == 0 or not np.any(np.isfinite(hist[:, 0])):
            continue
        r = _trajectory_in_frame(
            t_s=t_s,
            truth_hist=hist,
            frame=frame,
            reference_truth_hist=reference_truth_hist,
        )
        for ax, p in zip(axes, p_list):
            ix, iy, _, _ = _ric_2d_plane_axes(p)
            ax.plot(r[:, ix], r[:, iy], linewidth=1.2, label=oid)
            i0, i1 = _first_last_finite_indices(r[:, [ix, iy]])
            if i0 is not None:
                ax.scatter([r[i0, ix]], [r[i0, iy]], color="green", s=18, zorder=5)
            if i1 is not None:
                ax.scatter([r[i1, ix]], [r[i1, iy]], color="red", s=18, zorder=5)
            for burn_oid in burn_object_ids:
                if burn_oid not in burn_sources:
                    continue
                if burn_oid in truth_hist_by_object and burn_oid != oid:
                    continue
                u = np.array(burn_sources.get(burn_oid), dtype=float)
                if u.ndim != 2 or u.shape[1] < 3:
                    continue
                n = min(u.shape[0], r.shape[0])
                if n <= 0:
                    continue
                active = np.linalg.norm(np.nan_to_num(u[:n, :3], nan=0.0), axis=1) > float(
                    burn_marker_threshold_km_s2
                )
                active &= np.all(np.isfinite(r[:n, [ix, iy]]), axis=1)
                if not np.any(active):
                    continue
                label = f"{burn_oid} burn" if burn_oid not in burn_marker_labeled else None
                ax.scatter(
                    r[:n, ix][active],
                    r[:n, iy][active],
                    color="#F97316",
                    edgecolors="#111827",
                    linewidths=0.4,
                    s=24,
                    zorder=6,
                    label=label,
                )
                burn_marker_labeled.add(burn_oid)
    for ax, p in zip(axes, p_list):
        _, _, xlbl, ylbl = _ric_2d_plane_axes(p)
        ax.set_xlabel(xlbl)
        ax.set_ylabel(ylbl)
        ax.set_title(f"{xlbl}-{ylbl}")
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    _bottom_center_figure_legend(fig, handles, labels)
    fig.suptitle(f"RIC 2D Projections Multi ({'Rect' if frame == 'ric_rect' else 'Curvilinear'})")
    fig.tight_layout(rect=(0.0, 0.17, 1.0, 1.0))
    _show_save_close(fig, mode=mode, out_path=out_path)
