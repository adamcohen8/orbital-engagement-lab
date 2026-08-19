from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from sim.dynamics.orbit.frames import FrameContext
from sim.plotting.animation_quality import (
    STRICT_AGENT_ANIMATION_QUALITY,
    animation_time_decimal_places,
    fixed_time_text_width,
    format_animation_time,
    save_animation_with_quality,
)
from sim.plotting.attitude_geometry import _symmetric_limit_from_arrays
from sim.plotting.capability_common import _object_role_color, _play_interactive_animation
from sim.plotting.frame_plots import _draw_earth_sphere_3d, _ric_2d_plane_axes, _trajectory_in_frame
from sim.plotting.style import (
    role_color,
    save_oel_animation,
)
from sim.utils.figure_size import cap_figsize
from sim.utils.plot_windows import RIC_FOLLOW_MARGIN
from sim.utils.plot_windows import windows_from_points as _windows_from_points

PlotMode = Literal["interactive", "save", "both"]
FrameName = Literal["eci", "ecef", "ric_rect", "ric_curv"]
AttitudeFrame = Literal["eci", "ric"]
Layout = Literal["single", "subplots"]


def animate_multi_ric_2d_projections(
    t_s: np.ndarray,
    truth_hist_by_object: dict[str, np.ndarray],
    *,
    frame: Literal["ric_rect", "ric_curv"] = "ric_curv",
    reference_truth_hist: np.ndarray,
    planes: list[str] | None = None,
    mode: PlotMode = "interactive",
    out_path: str | None = None,
    fps: float = 30.0,
    speed_multiple: float = 10.0,
    frame_stride: int = 1,
    show_trajectory: bool = True,
) -> None:
    if frame not in ("ric_rect", "ric_curv"):
        raise ValueError("frame must be 'ric_rect' or 'ric_curv'.")
    trajectories: dict[str, np.ndarray] = {}
    for oid, hist in truth_hist_by_object.items():
        if hist.size == 0 or not np.any(np.isfinite(hist[:, 0])):
            continue
        trajectories[oid] = _trajectory_in_frame(
            t_s=t_s,
            truth_hist=hist,
            frame=frame,
            reference_truth_hist=reference_truth_hist,
        )
    if not trajectories:
        return

    p_list = planes if planes is not None and len(planes) > 0 else ["ri", "ic", "rc"]
    fig, axes = plt.subplots(1, len(p_list), figsize=cap_figsize(5.0 * len(p_list), 4.5))
    if len(p_list) == 1:
        axes = [axes]

    line_by_plane_obj: dict[tuple[str, str], Any] = {}
    dot_by_plane_obj: dict[tuple[str, str], Any] = {}
    ax_by_plane: dict[str, Any] = {}
    for ax, p in zip(axes, p_list):
        _, _, xlbl, ylbl = _ric_2d_plane_axes(p)
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_xlabel(xlbl)
        ax.set_ylabel(ylbl)
        ax.set_title(f"{xlbl}-{ylbl}")
        ax.grid(True, alpha=0.3)
        ax_by_plane[p] = ax
        for oid in sorted(trajectories.keys()):
            color = _object_role_color(oid)
            (line,) = ax.plot([], [], linewidth=1.2, label=oid, color=color)
            (dot,) = ax.plot([], [], marker="o", markersize=4, color=color)
            line_by_plane_obj[(p, oid)] = line
            dot_by_plane_obj[(p, oid)] = dot
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.035),
            ncol=min(len(labels), 5),
            frameon=True,
        )

    requested_stride = int(max(frame_stride, 1))
    max_frames = max(arr.shape[0] for arr in trajectories.values())
    allowed_frames = min(
        STRICT_AGENT_ANIMATION_QUALITY.max_frames,
        max(2, int(np.floor(STRICT_AGENT_ANIMATION_QUALITY.max_duration_s * max(float(fps), 1.0)))),
    )
    stride = max(requested_stride, int(np.ceil(max_frames / allowed_frames)))
    frame_ids = np.arange(0, max_frames, stride, dtype=int)
    if frame_ids.size == 0 or frame_ids[-1] != (max_frames - 1):
        frame_ids = np.append(frame_ids, max_frames - 1)

    camera_windows: dict[str, list[tuple[tuple[float, float], tuple[float, float]]]] = {}
    format_limits: dict[tuple[int, str], tuple[float, float]] = {}
    for plane_index, p in enumerate(p_list):
        ix, iy, _, _ = _ric_2d_plane_axes(p)
        current_by_frame: list[np.ndarray] = []
        for frame_i in frame_ids:
            current_by_frame.append(
                np.asarray(
                    [arr[min(int(frame_i), arr.shape[0] - 1), :] for arr in trajectories.values()],
                    dtype=float,
                )
            )
        x_span = max((float(np.ptp(points[:, ix])) for points in current_by_frame), default=0.0)
        y_span = max((float(np.ptp(points[:, iy])) for points in current_by_frame), default=0.0)
        x_span = max(x_span, 1.0) * (1.0 + 2.0 * RIC_FOLLOW_MARGIN)
        y_span = max(y_span, 1.0) * (1.0 + 2.0 * RIC_FOLLOW_MARGIN)
        camera_windows[p] = [
            (
                (float(np.mean(points[:, ix])) - 0.5 * x_span, float(np.mean(points[:, ix])) + 0.5 * x_span),
                (float(np.mean(points[:, iy])) - 0.5 * y_span, float(np.mean(points[:, iy])) + 0.5 * y_span),
            )
            for points in current_by_frame
        ]
        format_limits[(plane_index, "x")] = (-0.5 * x_span, 0.5 * x_span)
        format_limits[(plane_index, "y")] = (-0.5 * y_span, 0.5 * y_span)

    selected_times = np.asarray(
        [float(t_s[min(int(frame_i), t_s.size - 1)]) if t_s.size else 0.0 for frame_i in frame_ids],
        dtype=float,
    )
    time_decimals = animation_time_decimal_places(selected_times)
    time_width = fixed_time_text_width(selected_times, decimal_places=time_decimals)
    fig.suptitle(f"RIC 2D Projections Animation ({'Curvilinear' if frame == 'ric_curv' else 'Rectangular'})")
    time_text = fig.text(
        0.98,
        0.905,
        "",
        ha="right",
        va="top",
        fontsize=9,
        family="monospace",
        gid="oel_animation_time",
    )
    fig.tight_layout(rect=(0.01, 0.105, 0.99, 0.85))

    def update(i: int):
        artists = []
        frame_i = int(frame_ids[i])
        for p in p_list:
            ix, iy, _, _ = _ric_2d_plane_axes(p)
            for oid, arr in trajectories.items():
                idx = min(frame_i, arr.shape[0] - 1)
                start = 0 if show_trajectory else idx
                seg = arr[start : idx + 1, :]
                line_by_plane_obj[(p, oid)].set_data(seg[:, ix], seg[:, iy])
                dot_by_plane_obj[(p, oid)].set_data([arr[idx, ix]], [arr[idx, iy]])
                artists.extend([line_by_plane_obj[(p, oid)], dot_by_plane_obj[(p, oid)]])
            xlim, ylim = camera_windows[p][i]
            ax_by_plane[p].set_xlim(*xlim)
            ax_by_plane[p].set_ylim(*ylim)
        t_now = float(t_s[min(frame_i, t_s.size - 1)]) if t_s.size else 0.0
        time_text.set_text(
            "Sim time: "
            + format_animation_time(t_now, decimal_places=time_decimals, width=time_width)
            + " s"
        )
        artists.append(time_text)
        return artists

    dt = float(np.median(np.diff(t_s))) if t_s.size > 1 else 1.0
    interval_ms = 1000.0 * dt * float(stride) / max(speed_multiple, 1e-6)
    if mode in ("interactive", "both"):
        _play_interactive_animation(fig, update=update, frame_count=int(frame_ids.size), interval_ms=interval_ms)
    if mode in ("save", "both"):
        ani = animation.FuncAnimation(fig, update, frames=int(frame_ids.size), interval=interval_ms, blit=False)
        if out_path is None:
            raise ValueError("out_path is required when mode is 'save' or 'both'.")
        p = Path(out_path)
        try:
            save_animation_with_quality(
                ani,
                fig,
                p,
                update=update,
                frame_values=tuple(range(int(frame_ids.size))),
                frame_times_s=tuple(float(value) for value in selected_times),
                fps=fps,
                camera_policy="follow",
                artifact_id=p.stem,
                format_limits=format_limits,
                source={
                    "renderer_id": "multi_ric_2d_projections",
                    "frame": frame,
                    "requested_frame_stride": requested_stride,
                    "effective_frame_stride": stride,
                },
            )
        except Exception as exc:
            print(f"Warning: failed to save animation ({exc}).")
        del ani
    plt.close(fig)

def animate_trajectory_frame(
    t_s: np.ndarray,
    truth_hist: np.ndarray,
    *,
    frame: FrameName = "eci",
    jd_utc_start: float | None = None,
    frame_context: FrameContext | None = None,
    reference_truth_hist: np.ndarray | None = None,
    mode: PlotMode = "interactive",
    out_path: str | None = None,
    fps: float = 30.0,
    speed_multiple: float = 10.0,
) -> None:
    r = _trajectory_in_frame(
        t_s=t_s,
        truth_hist=truth_hist,
        frame=frame,
        jd_utc_start=jd_utc_start,
        frame_context=frame_context,
        reference_truth_hist=reference_truth_hist,
    )
    lim = _symmetric_limit_from_arrays([r[:, 0], r[:, 1], r[:, 2]], min_lim=1.0, margin=1.0)
    fig = plt.figure(figsize=cap_figsize(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    init_lim = 1.0 if frame in ("ric_rect", "ric_curv") else lim
    ax.set_xlim(-init_lim, init_lim)
    ax.set_ylim(-init_lim, init_lim)
    ax.set_zlim(-init_lim, init_lim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_title(f"Trajectory Animation ({frame.upper()})")
    (line,) = ax.plot([], [], [], linewidth=1.4, color=role_color("actual"))
    (dot,) = ax.plot([], [], [], marker="o", markersize=4, color=role_color("actual"))

    def update(i: int):
        line.set_data(r[: i + 1, 0], r[: i + 1, 1])
        line.set_3d_properties(r[: i + 1, 2])
        dot.set_data([r[i, 0]], [r[i, 1]])
        dot.set_3d_properties([r[i, 2]])
        if frame in ("ric_rect", "ric_curv"):
            xlim, ylim, zlim = _windows_from_points(
                [r[i, :]],
                axis_indices=(0, 1, 2),
                min_span=1.0,
                margin=RIC_FOLLOW_MARGIN,
            )
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_zlim(*zlim)
            ax.set_box_aspect(
                (
                    max(xlim[1] - xlim[0], 1e-6),
                    max(ylim[1] - ylim[0], 1e-6),
                    max(zlim[1] - zlim[0], 1e-6),
                )
            )
        ax.set_xlabel(f"t={t_s[i]:.1f}s")
        return [line, dot]

    dt = float(np.median(np.diff(t_s))) if t_s.size > 1 else 1.0
    interval_ms = 1000.0 * dt / max(speed_multiple, 1e-6)
    if mode in ("interactive", "both"):
        _play_interactive_animation(fig, update=update, frame_count=int(t_s.size), interval_ms=interval_ms)
    if mode in ("save", "both"):
        ani = animation.FuncAnimation(fig, update, frames=t_s.size, interval=interval_ms, blit=False)
        if out_path is None:
            raise ValueError("out_path is required when mode is 'save' or 'both'.")
        p = Path(out_path)
        try:
            save_oel_animation(ani, fig, p, fps=fps, artifact_id=p.stem)
        except Exception as exc:
            print(f"Warning: failed to save animation ({exc}).")
        del ani
    plt.close(fig)


def animate_multi_trajectory_frame(
    t_s: np.ndarray,
    truth_hist_by_object: dict[str, np.ndarray],
    *,
    frame: FrameName = "eci",
    jd_utc_start: float | None = None,
    frame_context: FrameContext | None = None,
    reference_truth_hist: np.ndarray | None = None,
    mode: PlotMode = "interactive",
    out_path: str | None = None,
    fps: float = 30.0,
    speed_multiple: float = 10.0,
    frame_stride: int = 1,
    show_trajectory: bool = True,
) -> None:
    trajectories: dict[str, np.ndarray] = {}
    for oid, hist in truth_hist_by_object.items():
        arr = np.array(hist, dtype=float)
        if arr.ndim != 2 or arr.shape[0] == 0 or not np.any(np.isfinite(arr[:, 0])):
            continue
        trajectories[oid] = _trajectory_in_frame(
            t_s=t_s,
            truth_hist=arr,
            frame=frame,
            jd_utc_start=jd_utc_start,
            frame_context=frame_context,
            reference_truth_hist=reference_truth_hist,
        )
    if not trajectories:
        return

    lim = 0.0
    for arr in trajectories.values():
        lim = max(lim, _symmetric_limit_from_arrays([arr[:, 0], arr[:, 1], arr[:, 2]], min_lim=1.0, margin=1.0))

    fig = plt.figure(figsize=cap_figsize(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    if frame in ("ric_rect", "ric_curv"):
        ix, iy, iz = 1, 0, 2
        xlbl, ylbl, zlbl = "I", "R", "C"
    else:
        ix, iy, iz = 0, 1, 2
        xlbl, ylbl, zlbl = "x", "y", "z"
        if frame in ("eci", "ecef"):
            _draw_earth_sphere_3d(ax)
    init_lim = 1.0 if frame in ("ric_rect", "ric_curv") else lim
    ax.set_xlim(-init_lim, init_lim)
    ax.set_ylim(-init_lim, init_lim)
    ax.set_zlim(-init_lim, init_lim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_title(f"Trajectories Animation ({frame.upper()})")
    ax.set_xlabel(xlbl)
    ax.set_ylabel(ylbl)
    ax.set_zlabel(zlbl)

    line_by_obj: dict[str, Any] = {}
    dot_by_obj: dict[str, Any] = {}
    for oid in sorted(trajectories.keys()):
        color = _object_role_color(oid)
        (line,) = ax.plot([], [], [], linewidth=1.4, label=oid, color=color)
        (dot,) = ax.plot([], [], [], marker="o", markersize=4, color=color)
        line_by_obj[oid] = line
        dot_by_obj[oid] = dot
    ax.legend(loc="best")

    stride = int(max(frame_stride, 1))
    max_frames = max(arr.shape[0] for arr in trajectories.values())
    frame_ids = np.arange(0, max_frames, stride, dtype=int)
    if frame_ids.size == 0 or frame_ids[-1] != (max_frames - 1):
        frame_ids = np.append(frame_ids, max_frames - 1)

    def update(i: int):
        artists = []
        frame_i = int(frame_ids[i])
        current_points: list[np.ndarray] = []
        for oid, arr in trajectories.items():
            idx = min(frame_i, arr.shape[0] - 1)
            start = 0 if show_trajectory else idx
            seg = arr[start : idx + 1, :]
            line_by_obj[oid].set_data(seg[:, ix], seg[:, iy])
            line_by_obj[oid].set_3d_properties(seg[:, iz])
            dot_by_obj[oid].set_data([arr[idx, ix]], [arr[idx, iy]])
            dot_by_obj[oid].set_3d_properties([arr[idx, iz]])
            if frame in ("ric_rect", "ric_curv"):
                current_points.append(arr[idx, :])
            artists.extend([line_by_obj[oid], dot_by_obj[oid]])
        if frame in ("ric_rect", "ric_curv"):
            xlim, ylim, zlim = _windows_from_points(
                current_points,
                axis_indices=(ix, iy, iz),
                min_span=1.0,
                margin=RIC_FOLLOW_MARGIN,
            )
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_zlim(*zlim)
            ax.set_box_aspect(
                (
                    max(xlim[1] - xlim[0], 1e-6),
                    max(ylim[1] - ylim[0], 1e-6),
                    max(zlim[1] - zlim[0], 1e-6),
                )
            )
        t_now = float(t_s[min(frame_i, t_s.size - 1)]) if t_s.size else 0.0
        ax.set_title(f"Trajectories Animation ({frame.upper()})  t={t_now:.1f}s")
        return artists

    dt = float(np.median(np.diff(t_s))) if t_s.size > 1 else 1.0
    interval_ms = 1000.0 * dt * float(stride) / max(speed_multiple, 1e-6)
    if mode in ("interactive", "both"):
        _play_interactive_animation(fig, update=update, frame_count=int(frame_ids.size), interval_ms=interval_ms)
    if mode in ("save", "both"):
        ani = animation.FuncAnimation(fig, update, frames=int(frame_ids.size), interval=interval_ms, blit=False)
        if out_path is None:
            raise ValueError("out_path is required when mode is 'save' or 'both'.")
        p = Path(out_path)
        try:
            save_oel_animation(ani, fig, p, fps=fps, artifact_id=p.stem)
        except Exception as exc:
            print(f"Warning: failed to save animation ({exc}).")
        del ani
    plt.close(fig)
