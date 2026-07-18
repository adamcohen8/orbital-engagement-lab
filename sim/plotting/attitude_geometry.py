from __future__ import annotations

from typing import Literal

import numpy as np

from sim.utils.frames import ric_dcm_ir_from_rv
from sim.utils.quaternion import quaternion_to_dcm_bn
from sim.utils.thruster_plot_geometry import thruster_marker_geometry_body

PlotMode = Literal["interactive", "save", "both"]
FrameName = Literal["eci", "ecef", "ric_rect", "ric_curv"]
AttitudeFrame = Literal["eci", "ric"]
Layout = Literal["single", "subplots"]

def _rectangular_prism_vertices_body(lx_m: float, ly_m: float, lz_m: float) -> np.ndarray:
    return np.array(
        [
            [-0.5 * lx_m, -0.5 * ly_m, -0.5 * lz_m],
            [-0.5 * lx_m, -0.5 * ly_m, +0.5 * lz_m],
            [-0.5 * lx_m, +0.5 * ly_m, -0.5 * lz_m],
            [-0.5 * lx_m, +0.5 * ly_m, +0.5 * lz_m],
            [+0.5 * lx_m, -0.5 * ly_m, -0.5 * lz_m],
            [+0.5 * lx_m, -0.5 * ly_m, +0.5 * lz_m],
            [+0.5 * lx_m, +0.5 * ly_m, -0.5 * lz_m],
            [+0.5 * lx_m, +0.5 * ly_m, +0.5 * lz_m],
        ],
        dtype=float,
    )


def _rectangular_prism_faces() -> list[list[int]]:
    return [
        [0, 1, 3, 2],
        [4, 5, 7, 6],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [0, 2, 6, 4],
        [1, 3, 7, 5],
    ]


def _attitude_rotation_history(truth_hist: np.ndarray, frame: AttitudeFrame) -> np.ndarray:
    q_bn = np.array(truth_hist[:, 6:10], dtype=float)
    c_anim = np.zeros((truth_hist.shape[0], 3, 3), dtype=float)
    for k in range(truth_hist.shape[0]):
        c_bn = quaternion_to_dcm_bn(q_bn[k, :])
        if frame == "eci":
            c_anim[k, :, :] = c_bn.T
        else:
            r = truth_hist[k, 0:3]
            v = truth_hist[k, 3:6]
            c_ir = ric_dcm_ir_from_rv(r, v)
            c_anim[k, :, :] = c_ir.T @ c_bn.T
    return c_anim


def _rectangular_prism_frame_vertices(
    body_vertices: np.ndarray,
    rotation_history: np.ndarray,
    faces: list[list[int]],
    frame_idx: int,
) -> list[np.ndarray]:
    verts = (rotation_history[frame_idx, :, :] @ body_vertices.T).T
    return [verts[idx, :] for idx in faces]


def _thruster_marker_geometry_body(
    *,
    lx_m: float,
    ly_m: float,
    lz_m: float,
    thruster_position_body_m: np.ndarray | None = None,
    thruster_direction_body: np.ndarray | None = None,
) -> tuple[np.ndarray, list[list[int]]]:
    return thruster_marker_geometry_body(
        lx_m=lx_m,
        ly_m=ly_m,
        lz_m=lz_m,
        thruster_position_body_m=thruster_position_body_m,
        thruster_direction_body=thruster_direction_body,
    )


def _marker_frame_faces(
    marker_points_body: np.ndarray,
    rotation_history: np.ndarray,
    faces: list[list[int]],
    frame_idx: int,
) -> list[np.ndarray]:
    pts = (rotation_history[frame_idx, :, :] @ marker_points_body.T).T
    return [pts[idx, :] for idx in faces]


def _attitude_display_axes(frame: AttitudeFrame) -> tuple[np.ndarray, tuple[str, str, str]]:
    if frame == "ric":
        # Display local RIC attitude with radial vertical: x=I, y=C, z=R.
        return np.array([1, 2, 0], dtype=int), ("I", "C", "R")
    return np.array([0, 1, 2], dtype=int), ("x", "y", "z")


def _permute_face_vertices(face_vertices: list[np.ndarray], permutation: np.ndarray) -> list[np.ndarray]:
    perm = np.array(permutation, dtype=int).reshape(3)
    return [np.array(face, dtype=float)[:, perm] for face in face_vertices]


def _symmetric_limit_from_arrays(
    arrays: list[np.ndarray],
    *,
    min_lim: float = 1.0,
    margin: float = 1.15,
) -> float:
    lim = 0.0
    for arr in arrays:
        a = np.array(arr, dtype=float)
        finite = a[np.isfinite(a)]
        if finite.size > 0:
            lim = max(lim, float(np.max(np.abs(finite))))
    return float(max(min_lim, margin * lim))
