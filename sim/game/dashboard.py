from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from sim.api import SimulationSnapshot
from sim.utils.plot_windows import RIC_FOLLOW_MARGIN
from sim.utils.plotting_capabilities import (
    _attitude_axis_limits,
    _attitude_rotation_history,
    _marker_frame_faces,
    _permute_face_vertices,
    _rectangular_prism_faces,
    _rectangular_prism_frame_vertices,
    _rectangular_prism_vertices_body,
    _ric_2d_plane_axes,
    _thruster_marker_geometry_body,
    _trajectory_in_frame,
    _windows_from_points,
)


@dataclass
class LiveBattlespaceDashboard:
    target_object_id: str = "target"
    chaser_object_id: str = "chaser"
    prism_dims_m_by_object: dict[str, list[float] | np.ndarray] = field(default_factory=dict)
    thruster_mounts_by_object: dict[str, dict[str, np.ndarray] | None] = field(default_factory=dict)
    dry_mass_kg_by_object: dict[str, float | None] = field(default_factory=dict)
    fuel_capacity_kg_by_object: dict[str, float | None] = field(default_factory=dict)
    thruster_active_threshold_km_s2: float = 1.0e-15
    max_history: int = 600
    show_trajectory: bool = True

    def __post_init__(self) -> None:
        self.t_s: list[float] = []
        self.truth_hist: dict[str, list[np.ndarray]] = {
            self.target_object_id: [],
            self.chaser_object_id: [],
        }
        self.thrust_hist: dict[str, list[np.ndarray]] = {
            self.target_object_id: [],
            self.chaser_object_id: [],
        }
        self._closed = False
        self._build_figure()

    @property
    def closed(self) -> bool:
        return self._closed or not plt.fignum_exists(self.fig.number)

    def connect_key_handlers(self, on_press: Any, on_release: Any) -> None:
        self.fig.canvas.mpl_connect("key_press_event", on_press)
        self.fig.canvas.mpl_connect("key_release_event", on_release)
        self.fig.canvas.mpl_connect("close_event", lambda _event: setattr(self, "_closed", True))

    def push_snapshot(self, snapshot: SimulationSnapshot) -> None:
        self.t_s.append(float(snapshot.time_s))
        for oid in (self.target_object_id, self.chaser_object_id):
            if oid in snapshot.truth:
                self.truth_hist[oid].append(np.array(snapshot.truth[oid], dtype=float))
            if oid in snapshot.applied_thrust:
                self.thrust_hist[oid].append(np.array(snapshot.applied_thrust[oid], dtype=float).reshape(3))
        while len(self.t_s) > int(max(self.max_history, 2)):
            self.t_s.pop(0)
            for oid in (self.target_object_id, self.chaser_object_id):
                if self.truth_hist[oid]:
                    self.truth_hist[oid].pop(0)
                if self.thrust_hist[oid]:
                    self.thrust_hist[oid].pop(0)

    def render(self, *, command_status: str = "") -> None:
        if self.closed:
            return
        if len(self.t_s) == 0:
            return
        try:
            self._update_artists(command_status=command_status)
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception as exc:  # pragma: no cover - keep live UI from killing sim loop.
            self.status_text.set_text(f"Dashboard update failed: {exc}")

    def _build_figure(self) -> None:
        self.fig = plt.figure(figsize=(12, 10))
        gs = self.fig.add_gridspec(2, 2, width_ratios=[1.2, 0.85])
        self.ax_ri = self.fig.add_subplot(gs[0, 0])
        self.ax_chaser = self.fig.add_subplot(gs[0, 1], projection="3d")
        self.ax_rc = self.fig.add_subplot(gs[1, 0])
        self.ax_target = self.fig.add_subplot(gs[1, 1], projection="3d")

        for ax, plane, title in (
            (self.ax_ri, "ri", "RI Relative Motion"),
            (self.ax_rc, "rc", "RC Relative Motion"),
        ):
            _, _, xlbl, ylbl = _ric_2d_plane_axes(plane)
            ax.set_xlim(-1.0, 1.0)
            ax.set_ylim(-1.0, 1.0)
            ax.set_xlabel(f"{xlbl} (km)")
            ax.set_ylabel(f"{ylbl} (km)")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

        self.color_by_object = {
            self.target_object_id: "#1F77B4",
            self.chaser_object_id: "#D62728",
        }
        self.faces = _rectangular_prism_faces()
        self.display_perm = np.array([1, 2, 0], dtype=int)
        self.body_vertices_by_object: dict[str, np.ndarray] = {}
        self.marker_points_by_object: dict[str, np.ndarray] = {}
        self.marker_faces_by_object: dict[str, list[list[int]]] = {}
        self.prism_poly_by_object: dict[str, Poly3DCollection] = {}
        self.thruster_poly_by_object: dict[str, Poly3DCollection] = {}
        self.fuel_fill_by_object: dict[str, Rectangle] = {}

        for oid, ax, title in (
            (self.chaser_object_id, self.ax_chaser, "Chaser Attitude + Thrust (RIC)"),
            (self.target_object_id, self.ax_target, "Target Attitude + Thrust (RIC)"),
        ):
            dims = np.array(self.prism_dims_m_by_object.get(oid, [4.0, 2.0, 2.0]), dtype=float).reshape(-1)
            if dims.size != 3 or np.any(dims <= 0.0) or not np.all(np.isfinite(dims)):
                dims = np.array([4.0, 2.0, 2.0], dtype=float)
            body_vertices = _rectangular_prism_vertices_body(float(dims[0]), float(dims[1]), float(dims[2]))
            self.body_vertices_by_object[oid] = body_vertices
            mount = self.thruster_mounts_by_object.get(oid) if isinstance(self.thruster_mounts_by_object.get(oid), dict) else {}
            self.marker_points_by_object[oid], self.marker_faces_by_object[oid] = _thruster_marker_geometry_body(
                lx_m=float(dims[0]),
                ly_m=float(dims[1]),
                lz_m=float(dims[2]),
                thruster_position_body_m=None if not isinstance(mount, dict) else mount.get("position_body_m"),
                thruster_direction_body=None if not isinstance(mount, dict) else mount.get("direction_body"),
            )
            lim = 0.7 * float(max(np.max(np.ptp(body_vertices, axis=0)), 1.0))
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
            poly = Poly3DCollection([], alpha=0.35, facecolor=self.color_by_object[oid], edgecolor="k", linewidth=0.7)
            ax.add_collection3d(poly)
            self.prism_poly_by_object[oid] = poly
            thrust_poly = Poly3DCollection([], alpha=1.0, facecolor="#808080", edgecolor="#5F5F5F", linewidth=0.85)
            ax.add_collection3d(thrust_poly)
            self.thruster_poly_by_object[oid] = thrust_poly
            self._add_fuel_meter(attitude_ax=ax, oid=oid)

        self.ri_line_by_object = {}
        self.ri_dot_by_object = {}
        self.rc_line_by_object = {}
        self.rc_dot_by_object = {}
        for oid in (self.target_object_id, self.chaser_object_id):
            color = self.color_by_object[oid]
            (self.ri_line_by_object[oid],) = self.ax_ri.plot([], [], linewidth=1.5, color=color, label=oid)
            (self.ri_dot_by_object[oid],) = self.ax_ri.plot([], [], marker="o", markersize=5, color=color)
            (self.rc_line_by_object[oid],) = self.ax_rc.plot([], [], linewidth=1.5, color=color, label=oid)
            (self.rc_dot_by_object[oid],) = self.ax_rc.plot([], [], marker="o", markersize=5, color=color)
        self.ax_ri.legend(loc="best")
        self.ax_rc.legend(loc="best")

        self.command_text = self.fig.text(0.015, 0.015, "", ha="left", va="bottom", fontsize=9, family="monospace")
        self.status_text = self.fig.text(
            0.5,
            0.015,
            "",
            ha="center",
            va="bottom",
            fontsize=9,
            family="monospace",
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
        )
        self.fig.suptitle("Battlespace Game Mode", fontsize=14)
        self.fig.tight_layout(rect=[0.0, 0.06, 1.0, 0.95])
        plt.ion()
        self.fig.show()

    def _add_fuel_meter(self, *, attitude_ax: Any, oid: str) -> None:
        bbox = attitude_ax.get_position()
        meter_width = 0.022
        meter_height = bbox.height * 0.72
        meter_left = min(bbox.x1 + 0.012, 0.975 - meter_width)
        meter_bottom = bbox.y0 + 0.14 * bbox.height
        meter_ax = self.fig.add_axes([meter_left, meter_bottom, meter_width, meter_height])
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
        self.fuel_fill_by_object[oid] = fill

    def _hist_array(self, oid: str) -> np.ndarray:
        hist = self.truth_hist.get(oid, [])
        if not hist:
            return np.empty((0, 14), dtype=float)
        return np.vstack(hist)

    def _thrust_array(self, oid: str, n: int) -> np.ndarray:
        hist = self.thrust_hist.get(oid, [])
        if not hist:
            return np.zeros((n, 3), dtype=float)
        arr = np.vstack(hist)
        if arr.shape[0] < n:
            pad = np.zeros((n - arr.shape[0], 3), dtype=float)
            arr = np.vstack((pad, arr))
        return arr[-n:, :]

    def _fuel_fraction(self, oid: str, current_mass_kg: float) -> float:
        dry = self.dry_mass_kg_by_object.get(oid)
        capacity = self.fuel_capacity_kg_by_object.get(oid)
        if dry is None or capacity is None:
            return float("nan")
        dry_f = float(dry)
        cap_f = float(capacity)
        if not np.isfinite(current_mass_kg) or not np.isfinite(dry_f) or not np.isfinite(cap_f) or cap_f <= 0.0:
            return float("nan")
        return float(np.clip((float(current_mass_kg) - dry_f) / cap_f, 0.0, 1.0))

    def _update_artists(self, *, command_status: str) -> None:
        target_hist = self._hist_array(self.target_object_id)
        chaser_hist = self._hist_array(self.chaser_object_id)
        n = min(len(self.t_s), target_hist.shape[0], chaser_hist.shape[0])
        if n <= 0:
            return
        t_plot = np.array(self.t_s[-n:], dtype=float)
        target_hist = target_hist[-n:, :]
        chaser_hist = chaser_hist[-n:, :]
        ref_hist = target_hist[:, 0:6]
        curv_traj_by_object = {
            self.target_object_id: _trajectory_in_frame(t_plot, target_hist, frame="ric_curv", reference_truth_hist=ref_hist),
            self.chaser_object_id: _trajectory_in_frame(t_plot, chaser_hist, frame="ric_curv", reference_truth_hist=ref_hist),
        }

        ri_ix, ri_iy, _, _ = _ric_2d_plane_axes("ri")
        rc_ix, rc_iy, _, _ = _ric_2d_plane_axes("rc")
        for oid, traj in curv_traj_by_object.items():
            start = 0 if self.show_trajectory else max(traj.shape[0] - 1, 0)
            seg = traj[start:, :]
            self.ri_line_by_object[oid].set_data(seg[:, ri_ix], seg[:, ri_iy])
            self.ri_dot_by_object[oid].set_data([traj[-1, ri_ix]], [traj[-1, ri_iy]])
            self.rc_line_by_object[oid].set_data(seg[:, rc_ix], seg[:, rc_iy])
            self.rc_dot_by_object[oid].set_data([traj[-1, rc_ix]], [traj[-1, rc_iy]])

        current_points = [traj[-1, :] for traj in curv_traj_by_object.values()]
        ri_xlim, ri_ylim = _windows_from_points(current_points, axis_indices=(ri_ix, ri_iy), min_span=1.0, margin=RIC_FOLLOW_MARGIN)
        rc_xlim, rc_ylim = _windows_from_points(current_points, axis_indices=(rc_ix, rc_iy), min_span=1.0, margin=RIC_FOLLOW_MARGIN)
        self.ax_ri.set_xlim(*ri_xlim)
        self.ax_ri.set_ylim(*ri_ylim)
        self.ax_rc.set_xlim(*rc_xlim)
        self.ax_rc.set_ylim(*rc_ylim)

        hist_by_object = {self.target_object_id: target_hist, self.chaser_object_id: chaser_hist}
        for oid, hist in hist_by_object.items():
            rotations = _attitude_rotation_history(truth_hist=hist, frame="ric")
            frame_i = hist.shape[0] - 1
            self.prism_poly_by_object[oid].set_verts(
                _permute_face_vertices(
                    _rectangular_prism_frame_vertices(
                        body_vertices=self.body_vertices_by_object[oid],
                        rotation_history=rotations,
                        faces=self.faces,
                        frame_idx=frame_i,
                    ),
                    self.display_perm,
                )
            )
            self.thruster_poly_by_object[oid].set_verts(
                _permute_face_vertices(
                    _marker_frame_faces(
                        marker_points_body=self.marker_points_by_object[oid],
                        rotation_history=rotations,
                        faces=self.marker_faces_by_object[oid],
                        frame_idx=frame_i,
                    ),
                    self.display_perm,
                )
            )
            active = float(np.linalg.norm(self._thrust_array(oid, n)[-1, :])) > float(self.thruster_active_threshold_km_s2)
            self.thruster_poly_by_object[oid].set_facecolor("#D95F02" if active else "#808080")
            self.thruster_poly_by_object[oid].set_edgecolor("#D95F02" if active else "#5F5F5F")
            fill = self.fuel_fill_by_object.get(oid)
            if fill is not None:
                frac = self._fuel_fraction(oid, float(hist[-1, 13] if hist.shape[1] > 13 else np.nan))
                if np.isfinite(frac):
                    fill.set_height(frac)
                    fill.set_facecolor(plt.get_cmap("RdYlGn")(frac))
                else:
                    fill.set_height(0.0)
                    fill.set_facecolor("#bdbdbd")

        rel_r_km = chaser_hist[-1, 0:3] - target_hist[-1, 0:3]
        rel_v_km_s = chaser_hist[-1, 3:6] - target_hist[-1, 3:6]
        self.status_text.set_text(
            f"t = {t_plot[-1]:7.1f} s   Relative Range = {np.linalg.norm(rel_r_km):8.3f} km   "
            f"Relative Speed = {np.linalg.norm(rel_v_km_s):8.5f} km/s"
        )
        self.command_text.set_text(command_status)
