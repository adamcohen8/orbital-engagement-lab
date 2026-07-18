from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from sim.plotting.attitude_geometry import (
    _attitude_display_axes,
    _attitude_rotation_history,
    _marker_frame_faces,
    _permute_face_vertices,
    _rectangular_prism_faces,
    _rectangular_prism_frame_vertices,
    _rectangular_prism_vertices_body,
    _thruster_marker_geometry_body,
)
from sim.plotting.capability_common import _play_interactive_animation
from sim.plotting.frame_plots import _ric_2d_plane_axes, _trajectory_in_frame
from sim.plotting.style import (
    role_color,
    save_oel_animation,
)
from sim.utils.figure_size import cap_figsize
from sim.utils.plot_windows import RIC_FOLLOW_MARGIN
from sim.utils.plot_windows import attitude_axis_limits as _attitude_axis_limits
from sim.utils.plot_windows import fuel_fraction_from_remaining_series as _fuel_fraction_from_remaining_series
from sim.utils.plot_windows import windows_from_points as _windows_from_points

PlotMode = Literal["interactive", "save", "both"]
FrameName = Literal["eci", "ecef", "ric_rect", "ric_curv"]
AttitudeFrame = Literal["eci", "ric"]
Layout = Literal["single", "subplots"]


def animate_rectangular_prism_attitude(
    t_s: np.ndarray,
    truth_hist: np.ndarray,
    *,
    lx_m: float,
    ly_m: float,
    lz_m: float,
    frame: AttitudeFrame = "eci",
    thruster_active_mask: np.ndarray | None = None,
    thruster_position_body_m: np.ndarray | None = None,
    thruster_direction_body: np.ndarray | None = None,
    body_facecolor: str = "#1F77B4",
    thruster_inactive_facecolor: str = "#808080",
    thruster_active_facecolor: str = "#D95F02",
    mode: PlotMode = "interactive",
    out_path: str | None = None,
    fps: float = 30.0,
    speed_multiple: float = 10.0,
) -> None:
    thruster_inactive_edgecolor = "#5F5F5F"
    thruster_active_edgecolor = "#D95F02"
    verts_body = _rectangular_prism_vertices_body(lx_m=lx_m, ly_m=ly_m, lz_m=lz_m)
    faces = _rectangular_prism_faces()
    c_anim = _attitude_rotation_history(truth_hist=truth_hist, frame=frame)
    marker_points_body, marker_faces = _thruster_marker_geometry_body(
        lx_m=lx_m,
        ly_m=ly_m,
        lz_m=lz_m,
        thruster_position_body_m=thruster_position_body_m,
        thruster_direction_body=thruster_direction_body,
    )

    active_mask = np.zeros(truth_hist.shape[0], dtype=bool)
    if thruster_active_mask is not None:
        mask_arr = np.array(thruster_active_mask, dtype=bool).reshape(-1)
        n_copy = min(mask_arr.size, active_mask.size)
        active_mask[:n_copy] = mask_arr[:n_copy]

    max_dim = 0.7 * max(lx_m, ly_m, lz_m)
    display_perm, axis_labels = _attitude_display_axes(frame)
    xlim, ylim, zlim = _attitude_axis_limits(frame, max_dim)
    fig = plt.figure(figsize=cap_figsize(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_title(f"Rectangular Prism Attitude Animation ({frame.upper()})")
    ax.set_xlabel(f"{axis_labels[0]} (m)")
    ax.set_ylabel(f"{axis_labels[1]} (m)")
    ax.set_zlabel(f"{axis_labels[2]} (m)")
    poly = Poly3DCollection([], alpha=0.35, facecolor=body_facecolor, edgecolor="k", linewidth=0.7)
    ax.add_collection3d(poly)
    thruster_poly = Poly3DCollection(
        [],
        alpha=1.0,
        facecolor=thruster_inactive_facecolor,
        edgecolor=thruster_inactive_edgecolor,
        linewidth=0.85,
    )
    ax.add_collection3d(thruster_poly)

    def _frame_verts(i: int) -> list[np.ndarray]:
        return _rectangular_prism_frame_vertices(
            body_vertices=verts_body,
            rotation_history=c_anim,
            faces=faces,
            frame_idx=i,
        )

    def update(i: int):
        poly.set_verts(_permute_face_vertices(_frame_verts(i), display_perm))
        poly.set_facecolor(body_facecolor)
        thruster_poly.set_verts(
            _permute_face_vertices(
                _marker_frame_faces(
                    marker_points_body=marker_points_body,
                    rotation_history=c_anim,
                    faces=marker_faces,
                    frame_idx=i,
                ),
                display_perm,
            )
        )
        thruster_poly.set_facecolor(thruster_active_facecolor if bool(active_mask[i]) else thruster_inactive_facecolor)
        thruster_poly.set_edgecolor(thruster_active_edgecolor if bool(active_mask[i]) else thruster_inactive_edgecolor)
        ax.set_title(f"Rectangular Prism Attitude Animation ({frame.upper()})  t={t_s[i]:.1f}s")
        return [poly, thruster_poly]

    dt = float(np.median(np.diff(t_s))) if t_s.size > 1 else 1.0
    interval_ms = 1000.0 * dt / max(speed_multiple, 1e-6)
    ani = animation.FuncAnimation(fig, update, frames=t_s.size, interval=interval_ms, blit=False)

    if mode in ("save", "both"):
        if out_path is None:
            raise ValueError("out_path is required when mode is 'save' or 'both'.")
        p = Path(out_path)
        try:
            save_oel_animation(ani, fig, p, fps=fps, artifact_id=p.stem)
        except Exception as exc:
            print(f"Warning: failed to save animation ({exc}).")
    if mode in ("interactive", "both"):
        plt.show()
    plt.close(fig)


def animate_battlespace_dashboard(
    t_s: np.ndarray,
    truth_hist_by_object: dict[str, np.ndarray],
    *,
    reference_truth_hist: np.ndarray,
    target_object_id: str = "target",
    chaser_object_id: str = "chaser",
    thrust_hist_by_object: dict[str, np.ndarray] | None = None,
    delta_v_remaining_m_s_by_object: dict[str, np.ndarray] | None = None,
    prism_dims_m_by_object: dict[str, list[float] | np.ndarray] | None = None,
    thruster_mounts_by_object: dict[str, dict[str, np.ndarray] | None] | None = None,
    thruster_active_threshold_km_s2: float = 1e-15,
    show_trajectory: bool = True,
    mode: PlotMode = "interactive",
    out_path: str | None = None,
    fps: float = 30.0,
    speed_multiple: float = 10.0,
    frame_stride: int = 1,
) -> None:
    display_perm = np.array([1, 2, 0], dtype=int)
    target_hist_raw = np.array(truth_hist_by_object.get(target_object_id, np.array([])), dtype=float)
    chaser_hist_raw = np.array(truth_hist_by_object.get(chaser_object_id, np.array([])), dtype=float)
    ref_hist_raw = np.array(reference_truth_hist, dtype=float)
    if target_hist_raw.ndim != 2 or chaser_hist_raw.ndim != 2 or ref_hist_raw.ndim != 2:
        return
    if target_hist_raw.shape[0] == 0 or chaser_hist_raw.shape[0] == 0 or ref_hist_raw.shape[0] == 0:
        return

    n_frames = min(t_s.size, target_hist_raw.shape[0], chaser_hist_raw.shape[0], ref_hist_raw.shape[0])
    if n_frames <= 0:
        return

    t_plot = np.array(t_s[:n_frames], dtype=float)
    target_hist = target_hist_raw[:n_frames, :]
    chaser_hist = chaser_hist_raw[:n_frames, :]
    ref_hist = ref_hist_raw[:n_frames, :]

    rel_truth_by_object = {
        target_object_id: target_hist,
        chaser_object_id: chaser_hist,
    }
    curv_traj_by_object = {
        oid: _trajectory_in_frame(
            t_s=t_plot,
            truth_hist=hist,
            frame="ric_curv",
            reference_truth_hist=ref_hist,
        )
        for oid, hist in rel_truth_by_object.items()
    }

    default_dims = np.array([4.0, 2.0, 2.0], dtype=float)
    dims_map = prism_dims_m_by_object or {}
    mount_map = thruster_mounts_by_object or {}
    body_vertices_by_object: dict[str, np.ndarray] = {}
    marker_points_by_object: dict[str, np.ndarray] = {}
    marker_faces_by_object: dict[str, list[list[int]]] = {}
    rotations_by_object: dict[str, np.ndarray] = {}
    active_by_object: dict[str, np.ndarray] = {}
    faces = _rectangular_prism_faces()
    for oid, hist in rel_truth_by_object.items():
        dims = np.array(dims_map.get(oid, default_dims), dtype=float).reshape(-1)
        if dims.size != 3 or not np.all(np.isfinite(dims)) or np.any(dims <= 0.0):
            dims = default_dims.copy()
        body_vertices_by_object[oid] = _rectangular_prism_vertices_body(
            lx_m=float(dims[0]),
            ly_m=float(dims[1]),
            lz_m=float(dims[2]),
        )
        mount = mount_map.get(oid) if isinstance(mount_map.get(oid), dict) else {}
        marker_points_by_object[oid], marker_faces_by_object[oid] = _thruster_marker_geometry_body(
            lx_m=float(dims[0]),
            ly_m=float(dims[1]),
            lz_m=float(dims[2]),
            thruster_position_body_m=None if not isinstance(mount, dict) else mount.get("position_body_m"),
            thruster_direction_body=None if not isinstance(mount, dict) else mount.get("direction_body"),
        )
        rotations_by_object[oid] = _attitude_rotation_history(truth_hist=hist, frame="ric")
        thrust_hist = np.array((thrust_hist_by_object or {}).get(oid, np.zeros((n_frames, 3))), dtype=float)
        thrust_local = thrust_hist[:n_frames, :] if thrust_hist.ndim == 2 else np.zeros((n_frames, 3), dtype=float)
        active_by_object[oid] = np.linalg.norm(np.nan_to_num(thrust_local, nan=0.0), axis=1) > float(
            thruster_active_threshold_km_s2
        )

    dv_remaining_by_object: dict[str, np.ndarray] = {}
    for oid in (target_object_id, chaser_object_id):
        arr = np.array(
            (delta_v_remaining_m_s_by_object or {}).get(oid, np.full(n_frames, np.nan)), dtype=float
        ).reshape(-1)
        dv_remaining_by_object[oid] = (
            arr[:n_frames] if arr.size >= n_frames else np.pad(arr, (0, n_frames - arr.size), constant_values=np.nan)
        )
    fuel_fraction_by_object = {
        oid: _fuel_fraction_from_remaining_series(dv_remaining_by_object[oid])
        for oid in (target_object_id, chaser_object_id)
    }

    rel_r_km = chaser_hist[:, 0:3] - target_hist[:, 0:3]
    rel_v_km_s = chaser_hist[:, 3:6] - target_hist[:, 3:6]
    rel_range_km = np.linalg.norm(rel_r_km, axis=1)
    rel_speed_km_s = np.linalg.norm(rel_v_km_s, axis=1)

    fig = plt.figure(figsize=cap_figsize(12, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 0.85])
    ax_ri = fig.add_subplot(gs[0, 0])
    ax_chaser = fig.add_subplot(gs[0, 1], projection="3d")
    ax_rc = fig.add_subplot(gs[1, 0])
    ax_target = fig.add_subplot(gs[1, 1], projection="3d")

    color_by_object = {target_object_id: role_color("target"), chaser_object_id: role_color("chaser")}
    thruster_inactive_facecolor = "#808080"
    thruster_active_facecolor = "#D95F02"
    thruster_inactive_edgecolor = "#5F5F5F"
    thruster_active_edgecolor = "#D95F02"

    for ax, plane, lim, title in (
        (ax_ri, "ri", 1.0, "RI Relative Motion"),
        (ax_rc, "rc", 1.0, "RC Relative Motion"),
    ):
        _, _, xlbl, ylbl = _ric_2d_plane_axes(plane)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel(f"{xlbl} (km)")
        ax.set_ylabel(f"{ylbl} (km)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    prism_poly_by_object: dict[str, Poly3DCollection] = {}
    thruster_poly_by_object: dict[str, Poly3DCollection] = {}
    for oid, ax, title in (
        (chaser_object_id, ax_chaser, "Chaser Attitude + Thrust (RIC)"),
        (target_object_id, ax_target, "Target Attitude + Thrust (RIC)"),
    ):
        body_vertices = body_vertices_by_object[oid]
        body_span = np.ptp(body_vertices, axis=0)
        lim = 0.7 * float(max(np.max(body_span), 1.0))
        xlim, ylim, zlim = _attitude_axis_limits("ric", lim)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=22.0, azim=35.0)
        ax.set_xlabel("I (m)")
        ax.set_ylabel("C (m)")
        ax.set_zlabel("R (m)")
        ax.set_title(title)
        poly = Poly3DCollection([], alpha=0.35, facecolor=color_by_object[oid], edgecolor="k", linewidth=0.7)
        ax.add_collection3d(poly)
        prism_poly_by_object[oid] = poly
        thruster_poly = Poly3DCollection(
            [],
            alpha=1.0,
            facecolor=thruster_inactive_facecolor,
            edgecolor=thruster_inactive_edgecolor,
            linewidth=0.85,
        )
        ax.add_collection3d(thruster_poly)
        thruster_poly_by_object[oid] = thruster_poly

    ri_line_by_object: dict[str, Any] = {}
    ri_dot_by_object: dict[str, Any] = {}
    rc_line_by_object: dict[str, Any] = {}
    rc_dot_by_object: dict[str, Any] = {}
    ri_ix, ri_iy, _, _ = _ric_2d_plane_axes("ri")
    rc_ix, rc_iy, _, _ = _ric_2d_plane_axes("rc")
    for oid in (target_object_id, chaser_object_id):
        color = color_by_object[oid]
        (ri_line,) = ax_ri.plot([], [], linewidth=1.5, color=color, label=oid)
        (ri_dot,) = ax_ri.plot([], [], marker="o", markersize=5, color=color)
        (rc_line,) = ax_rc.plot([], [], linewidth=1.5, color=color, label=oid)
        (rc_dot,) = ax_rc.plot([], [], marker="o", markersize=5, color=color)
        ri_line_by_object[oid] = ri_line
        ri_dot_by_object[oid] = ri_dot
        rc_line_by_object[oid] = rc_line
        rc_dot_by_object[oid] = rc_dot
    ax_ri.legend(loc="best")
    ax_rc.legend(loc="best")

    fig.suptitle("Battlespace Visualization Dashboard", fontsize=14)
    fig.tight_layout(rect=[0.0, 0.06, 1.0, 0.95])

    fuel_fill_by_object: dict[str, Rectangle] = {}

    def _add_fuel_meter(attitude_ax: Any, oid: str) -> None:
        bbox = attitude_ax.get_position()
        meter_width = 0.022
        meter_height = bbox.height * 0.72
        meter_left = min(bbox.x1 + 0.012, 0.975 - meter_width)
        meter_bottom = bbox.y0 + 0.14 * bbox.height
        meter_ax = fig.add_axes([meter_left, meter_bottom, meter_width, meter_height])
        meter_ax.set_xlim(0.0, 1.0)
        meter_ax.set_ylim(0.0, 1.0)
        meter_ax.set_xticks([])
        meter_ax.set_yticks([0.0, 0.5, 1.0])
        meter_ax.set_yticklabels([])
        meter_ax.set_title("Fuel", fontsize=8, pad=4)
        for spine in meter_ax.spines.values():
            spine.set_edgecolor("#666666")
            spine.set_linewidth(0.8)
        meter_ax.set_facecolor("#f3f3f3")
        meter_ax.add_patch(Rectangle((0.12, 0.0), 0.76, 1.0, facecolor="#ffffff", edgecolor="#999999", linewidth=0.8))
        fill = Rectangle((0.12, 0.0), 0.76, 0.0, facecolor="#7fbf3f", edgecolor="none", alpha=0.95)
        meter_ax.add_patch(fill)
        fuel_fill_by_object[oid] = fill

    _add_fuel_meter(ax_chaser, chaser_object_id)
    _add_fuel_meter(ax_target, target_object_id)

    status_text = fig.text(
        0.5,
        0.015,
        "",
        ha="center",
        va="bottom",
        fontsize=9,
        family="monospace",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
    )
    stride = int(max(frame_stride, 1))
    frame_ids = np.arange(0, n_frames, stride, dtype=int)
    if frame_ids.size == 0 or frame_ids[-1] != (n_frames - 1):
        frame_ids = np.append(frame_ids, n_frames - 1)

    def update(i: int):
        artists: list[Any] = []
        frame_i = int(frame_ids[i])
        for oid, traj in curv_traj_by_object.items():
            start = 0 if show_trajectory else frame_i
            seg = traj[start : frame_i + 1, :]
            ri_line_by_object[oid].set_data(seg[:, ri_ix], seg[:, ri_iy])
            ri_dot_by_object[oid].set_data([traj[frame_i, ri_ix]], [traj[frame_i, ri_iy]])
            rc_line_by_object[oid].set_data(seg[:, rc_ix], seg[:, rc_iy])
            rc_dot_by_object[oid].set_data([traj[frame_i, rc_ix]], [traj[frame_i, rc_iy]])
            artists.extend(
                [
                    ri_line_by_object[oid],
                    ri_dot_by_object[oid],
                    rc_line_by_object[oid],
                    rc_dot_by_object[oid],
                ]
            )

        current_points = [traj[min(frame_i, traj.shape[0] - 1), :] for traj in curv_traj_by_object.values()]
        (ri_xlim, ri_ylim) = _windows_from_points(
            current_points,
            axis_indices=(ri_ix, ri_iy),
            min_span=1.0,
            margin=RIC_FOLLOW_MARGIN,
        )
        (rc_xlim, rc_ylim) = _windows_from_points(
            current_points,
            axis_indices=(rc_ix, rc_iy),
            min_span=1.0,
            margin=RIC_FOLLOW_MARGIN,
        )
        ax_ri.set_xlim(*ri_xlim)
        ax_ri.set_ylim(*ri_ylim)
        ax_rc.set_xlim(*rc_xlim)
        ax_rc.set_ylim(*rc_ylim)

        for oid in (chaser_object_id, target_object_id):
            prism_poly_by_object[oid].set_verts(
                _permute_face_vertices(
                    _rectangular_prism_frame_vertices(
                        body_vertices=body_vertices_by_object[oid],
                        rotation_history=rotations_by_object[oid],
                        faces=faces,
                        frame_idx=frame_i,
                    ),
                    display_perm,
                )
            )
            prism_poly_by_object[oid].set_facecolor(color_by_object[oid])
            thruster_poly_by_object[oid].set_verts(
                _permute_face_vertices(
                    _marker_frame_faces(
                        marker_points_body=marker_points_by_object[oid],
                        rotation_history=rotations_by_object[oid],
                        faces=marker_faces_by_object[oid],
                        frame_idx=frame_i,
                    ),
                    display_perm,
                )
            )
            thruster_poly_by_object[oid].set_facecolor(
                thruster_active_facecolor if bool(active_by_object[oid][frame_i]) else thruster_inactive_facecolor
            )
            thruster_poly_by_object[oid].set_edgecolor(
                thruster_active_edgecolor if bool(active_by_object[oid][frame_i]) else thruster_inactive_edgecolor
            )
            artists.append(prism_poly_by_object[oid])
            artists.append(thruster_poly_by_object[oid])
            frac = float(fuel_fraction_by_object[oid][frame_i])
            if np.isfinite(frac):
                frac_clip = float(np.clip(frac, 0.0, 1.0))
                fuel_fill_by_object[oid].set_height(frac_clip)
                fuel_fill_by_object[oid].set_facecolor(plt.get_cmap("RdYlGn")(frac_clip))
            else:
                fuel_fill_by_object[oid].set_height(0.0)
                fuel_fill_by_object[oid].set_facecolor("#bdbdbd")
            artists.append(fuel_fill_by_object[oid])

        status_text.set_text(
            f"t = {t_plot[frame_i]:7.1f} s   Relative Range = {rel_range_km[frame_i]:8.3f} km   Relative Speed = {rel_speed_km_s[frame_i]:8.5f} km/s"
        )
        artists.append(status_text)
        return artists

    dt = float(np.median(np.diff(t_plot))) if t_plot.size > 1 else 1.0
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
