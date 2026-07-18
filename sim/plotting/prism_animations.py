from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from sim.plotting.attitude_geometry import _permute_face_vertices
from sim.plotting.capability_common import _object_role_color
from sim.plotting.style import (
    role_color,
    save_oel_animation,
)
from sim.utils.figure_size import cap_figsize
from sim.utils.frames import ric_dcm_ir_from_rv, ric_rect_to_curv
from sim.utils.plot_windows import attitude_axis_limits as _attitude_axis_limits
from sim.utils.quaternion import quaternion_to_dcm_bn

PlotMode = Literal["interactive", "save", "both"]
FrameName = Literal["eci", "ecef", "ric_rect", "ric_curv"]
AttitudeFrame = Literal["eci", "ric"]
Layout = Literal["single", "subplots"]


def animate_multi_rectangular_prism_ric_curv(
    t_s: np.ndarray,
    truth_hist_by_object: dict[str, np.ndarray],
    *,
    target_object_id: str = "target",
    object_ids: list[str] | None = None,
    prism_dims_m_by_object: dict[str, list[float] | tuple[float, float, float]] | None = None,
    mode: PlotMode = "interactive",
    out_path: str | None = None,
    fps: float = 30.0,
    speed_multiple: float = 10.0,
    frame_stride: int = 1,
) -> None:
    """Animate multiple spacecraft as rectangular prisms in target-centered curvilinear RIC.

    Display axes are arranged as (I, C, R), so radial is the vertical axis.
    """
    target_hist = truth_hist_by_object.get(target_object_id)
    if target_hist is None:
        return
    tgt = np.array(target_hist, dtype=float)
    if tgt.ndim != 2 or tgt.shape[0] == 0 or tgt.shape[1] < 10:
        return

    all_ids = sorted(list(truth_hist_by_object.keys()))
    if object_ids is None:
        if "target" in all_ids and "chaser" in all_ids:
            obj_ids = ["target", "chaser"]
        else:
            obj_ids = all_ids
    else:
        obj_ids = [oid for oid in object_ids if oid in truth_hist_by_object]
    if not obj_ids:
        return
    if target_object_id not in obj_ids:
        obj_ids = [target_object_id, *obj_ids]

    dims_map = dict(prism_dims_m_by_object or {})
    default_dims_m = np.array([4.0, 2.0, 2.0], dtype=float)
    perm_icr_from_ric = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    n_frames = int(
        min(
            [t_s.size] + [np.array(truth_hist_by_object[oid], dtype=float).shape[0] for oid in obj_ids] + [tgt.shape[0]]
        )
    )
    if n_frames <= 0:
        return
    t_loc = np.array(t_s[:n_frames], dtype=float)

    pos_by_obj: dict[str, np.ndarray] = {}
    c_by_obj: dict[str, np.ndarray] = {}
    verts_body_km_by_obj: dict[str, np.ndarray] = {}

    for oid in obj_ids:
        hist = np.array(truth_hist_by_object[oid], dtype=float)
        arr = hist[:n_frames, :]
        pos_by_obj[oid] = np.full((n_frames, 3), np.nan, dtype=float)
        c_by_obj[oid] = np.full((n_frames, 3, 3), np.nan, dtype=float)
        dims = np.array(dims_map.get(oid, default_dims_m), dtype=float).reshape(-1)
        if dims.size != 3:
            dims = default_dims_m.copy()
        lx_km, ly_km, lz_km = (dims * 1e-3).tolist()
        verts_body_km_by_obj[oid] = np.array(
            [
                [-0.5 * lx_km, -0.5 * ly_km, -0.5 * lz_km],
                [-0.5 * lx_km, -0.5 * ly_km, +0.5 * lz_km],
                [-0.5 * lx_km, +0.5 * ly_km, -0.5 * lz_km],
                [-0.5 * lx_km, +0.5 * ly_km, +0.5 * lz_km],
                [+0.5 * lx_km, -0.5 * ly_km, -0.5 * lz_km],
                [+0.5 * lx_km, -0.5 * ly_km, +0.5 * lz_km],
                [+0.5 * lx_km, +0.5 * ly_km, -0.5 * lz_km],
                [+0.5 * lx_km, +0.5 * ly_km, +0.5 * lz_km],
            ],
            dtype=float,
        )

        for k in range(n_frames):
            r_t = tgt[k, 0:3]
            v_t = tgt[k, 3:6]
            if not (np.all(np.isfinite(r_t)) and np.all(np.isfinite(v_t))):
                continue
            c_ir = ric_dcm_ir_from_rv(r_t, v_t)

            r = arr[k, 0:3]
            q_bn = arr[k, 6:10]
            if not (np.all(np.isfinite(r)) and np.all(np.isfinite(q_bn))):
                continue

            dr_rect = c_ir.T @ (r - r_t)
            x_curv = ric_rect_to_curv(
                np.hstack((dr_rect, np.zeros(3, dtype=float))),
                r0_km=float(np.linalg.norm(r_t)),
            )
            ric_curv_pos = x_curv[:3]  # [R, I, C] in km-equivalent curvilinear coordinates
            pos_by_obj[oid][k, :] = np.array([ric_curv_pos[1], ric_curv_pos[2], ric_curv_pos[0]], dtype=float)

            c_bn = quaternion_to_dcm_bn(q_bn)
            c_rb = c_ir.T @ c_bn.T  # body -> RIC
            c_by_obj[oid][k, :, :] = perm_icr_from_ric @ c_rb  # body -> display (I,C,R)

    # Global limits over all objects and all frames.
    all_pos = np.vstack([v for v in pos_by_obj.values() if v.size > 0])
    finite = np.isfinite(all_pos)
    if not np.any(finite):
        return
    lim = float(max(np.nanmax(np.abs(all_pos)), 1.0))

    faces = [
        [0, 1, 3, 2],
        [4, 5, 7, 6],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [0, 2, 6, 4],
        [1, 3, 7, 5],
    ]

    fig = plt.figure(figsize=cap_figsize(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("I (km)")
    ax.set_ylabel("C (km)")
    ax.set_zlabel("R (km)")
    ax.set_title("Target-Centered Curvilinear RIC Prism Animation")

    poly_by_obj: dict[str, Poly3DCollection] = {}
    trail_by_obj: dict[str, Any] = {}
    dot_by_obj: dict[str, Any] = {}
    for oid in obj_ids:
        color = _object_role_color(oid) or role_color("actual")
        poly = Poly3DCollection([], alpha=0.35, facecolor=color, edgecolor="k", linewidth=0.7)
        ax.add_collection3d(poly)
        poly_by_obj[oid] = poly
        (trail,) = ax.plot([], [], [], linewidth=1.2, color=color, label=oid)
        (dot,) = ax.plot([], [], [], marker="o", markersize=4, color=color)
        trail_by_obj[oid] = trail
        dot_by_obj[oid] = dot
    ax.legend(loc="best")

    frame_ids = np.arange(0, n_frames, max(int(frame_stride), 1), dtype=int)
    if frame_ids.size == 0 or frame_ids[-1] != (n_frames - 1):
        frame_ids = np.append(frame_ids, n_frames - 1)

    def _frame_verts(oid: str, i_frame: int) -> list[np.ndarray] | None:
        p = pos_by_obj[oid][i_frame, :]
        c_bd = c_by_obj[oid][i_frame, :, :]
        if not (np.all(np.isfinite(p)) and np.all(np.isfinite(c_bd))):
            return None
        verts = (c_bd @ verts_body_km_by_obj[oid].T).T + p
        return [verts[idx, :] for idx in faces]

    def update(i: int):
        k = int(frame_ids[i])
        artists: list[Any] = []
        for oid in obj_ids:
            poly = poly_by_obj[oid]
            fv = _frame_verts(oid, k)
            if fv is None:
                poly.set_verts([])
            else:
                poly.set_verts(fv)
            artists.append(poly)

            tr = trail_by_obj[oid]
            dd = dot_by_obj[oid]
            p = pos_by_obj[oid][: k + 1, :]
            mask = np.all(np.isfinite(p), axis=1)
            if np.any(mask):
                p_ok = p[mask, :]
                tr.set_data(p_ok[:, 0], p_ok[:, 1])
                tr.set_3d_properties(p_ok[:, 2])
                dd.set_data([p_ok[-1, 0]], [p_ok[-1, 1]])
                dd.set_3d_properties([p_ok[-1, 2]])
            else:
                tr.set_data([], [])
                tr.set_3d_properties([])
                dd.set_data([], [])
                dd.set_3d_properties([])
            artists.extend([tr, dd])
        ax.set_title(f"Target-Centered Curvilinear RIC Prism Animation (t={t_loc[k]:.1f}s)")
        return artists

    dt = float(np.median(np.diff(t_loc))) if t_loc.size > 1 else 1.0
    interval_ms = 1000.0 * dt / max(speed_multiple, 1e-6)
    ani = animation.FuncAnimation(fig, update, frames=int(frame_ids.size), interval=interval_ms, blit=False)

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


def animate_side_by_side_rectangular_prism_ric_attitude(
    t_s: np.ndarray,
    truth_hist_by_object: dict[str, np.ndarray],
    *,
    left_object_id: str = "target",
    right_object_id: str = "chaser",
    prism_dims_m_by_object: dict[str, list[float] | tuple[float, float, float]] | None = None,
    mode: PlotMode = "interactive",
    out_path: str | None = None,
    fps: float = 30.0,
    speed_multiple: float = 10.0,
    frame_stride: int = 1,
) -> None:
    left_hist = np.array(truth_hist_by_object.get(left_object_id, np.empty((0, 14))), dtype=float)
    right_hist = np.array(truth_hist_by_object.get(right_object_id, np.empty((0, 14))), dtype=float)
    if left_hist.ndim != 2 or right_hist.ndim != 2:
        return
    n_frames = int(min(t_s.size, left_hist.shape[0], right_hist.shape[0]))
    if n_frames <= 0:
        return
    t_loc = np.array(t_s[:n_frames], dtype=float)
    left_hist = left_hist[:n_frames, :]
    right_hist = right_hist[:n_frames, :]

    dims_map = dict(prism_dims_m_by_object or {})
    default_dims_m = np.array([4.0, 2.0, 2.0], dtype=float)
    faces = [
        [0, 1, 3, 2],
        [4, 5, 7, 6],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [0, 2, 6, 4],
        [1, 3, 7, 5],
    ]

    def _dims_km(oid: str) -> np.ndarray:
        dims = np.array(dims_map.get(oid, default_dims_m), dtype=float).reshape(-1)
        if dims.size != 3:
            dims = default_dims_m.copy()
        return dims * 1e-3

    def _verts_body_km(oid: str) -> np.ndarray:
        lx_km, ly_km, lz_km = _dims_km(oid).tolist()
        return np.array(
            [
                [-0.5 * lx_km, -0.5 * ly_km, -0.5 * lz_km],
                [-0.5 * lx_km, -0.5 * ly_km, +0.5 * lz_km],
                [-0.5 * lx_km, +0.5 * ly_km, -0.5 * lz_km],
                [-0.5 * lx_km, +0.5 * ly_km, +0.5 * lz_km],
                [+0.5 * lx_km, -0.5 * ly_km, -0.5 * lz_km],
                [+0.5 * lx_km, -0.5 * ly_km, +0.5 * lz_km],
                [+0.5 * lx_km, +0.5 * ly_km, -0.5 * lz_km],
                [+0.5 * lx_km, +0.5 * ly_km, +0.5 * lz_km],
            ],
            dtype=float,
        )

    left_verts_body = _verts_body_km(left_object_id)
    right_verts_body = _verts_body_km(right_object_id)
    lim_km = float(max(np.max(np.abs(left_verts_body)), np.max(np.abs(right_verts_body)), 1e-3)) * 2.2

    def _body_to_ric_dcm(hist: np.ndarray) -> np.ndarray:
        c_arr = np.full((n_frames, 3, 3), np.nan, dtype=float)
        for k in range(n_frames):
            r = hist[k, 0:3]
            v = hist[k, 3:6]
            q_bn = hist[k, 6:10]
            if not (np.all(np.isfinite(r)) and np.all(np.isfinite(v)) and np.all(np.isfinite(q_bn))):
                continue
            c_bn = quaternion_to_dcm_bn(q_bn)
            c_ir = ric_dcm_ir_from_rv(r, v)
            c_arr[k, :, :] = c_ir.T @ c_bn.T  # body -> local RIC
        return c_arr

    c_left = _body_to_ric_dcm(left_hist)
    c_right = _body_to_ric_dcm(right_hist)

    fig = plt.figure(figsize=cap_figsize(12, 6))
    ax_left = fig.add_subplot(1, 2, 1, projection="3d")
    ax_right = fig.add_subplot(1, 2, 2, projection="3d")
    xlim, ylim, zlim = _attitude_axis_limits("ric", lim_km)
    for ax, title in ((ax_left, left_object_id), (ax_right, right_object_id)):
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("I (km)")
        ax.set_ylabel("C (km)")
        ax.set_zlabel("R (km)")
        ax.set_title(f"{title} Body in Local RIC")

    poly_left = Poly3DCollection(
        [],
        alpha=0.4,
        facecolor=_object_role_color(left_object_id) or role_color("target"),
        edgecolor="k",
        linewidth=0.7,
    )
    poly_right = Poly3DCollection(
        [],
        alpha=0.4,
        facecolor=_object_role_color(right_object_id) or role_color("chaser"),
        edgecolor="k",
        linewidth=0.7,
    )
    ax_left.add_collection3d(poly_left)
    ax_right.add_collection3d(poly_right)

    frame_ids = np.arange(0, n_frames, max(int(frame_stride), 1), dtype=int)
    if frame_ids.size == 0 or frame_ids[-1] != (n_frames - 1):
        frame_ids = np.append(frame_ids, n_frames - 1)

    def _frame_verts(c_arr: np.ndarray, verts_body: np.ndarray, i_frame: int) -> list[np.ndarray] | None:
        c_rb = c_arr[i_frame, :, :]
        if not np.all(np.isfinite(c_rb)):
            return None
        verts = (c_rb @ verts_body.T).T
        return _permute_face_vertices([verts[idx, :] for idx in faces], np.array([1, 2, 0], dtype=int))

    def update(i: int):
        k = int(frame_ids[i])
        lv = _frame_verts(c_left, left_verts_body, k)
        rv = _frame_verts(c_right, right_verts_body, k)
        poly_left.set_verts([] if lv is None else lv)
        poly_right.set_verts([] if rv is None else rv)
        fig.suptitle(f"Side-by-Side Local RIC Attitude Animation (t={t_loc[k]:.1f}s)")
        return [poly_left, poly_right]

    dt = float(np.median(np.diff(t_loc))) if t_loc.size > 1 else 1.0
    interval_ms = 1000.0 * dt / max(speed_multiple, 1e-6)
    ani = animation.FuncAnimation(fig, update, frames=int(frame_ids.size), interval=interval_ms, blit=False)

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
