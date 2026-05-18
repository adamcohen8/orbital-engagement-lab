from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sim.api import SimulationSnapshot
from sim.game.training import (
    ApproachGateConfig,
    ForbiddenRegionConfig,
    InspectionGateConfig,
    relative_ric_state_from_arrays,
)
from sim.utils.frames import ric_dcm_ir_from_rv

EARTH_MU_KM3_S2 = 398600.4418
PLOT_OVERLAY_MARGIN = 1.18
MAX_TRAIL_DRAW_POINTS = 260
MAX_GHOST_DRAW_POINTS = 120
TEXT_CACHE_LIMIT = 512


@dataclass
class PygameRPODashboard:
    target_object_id: str = "target"
    chaser_object_id: str = "chaser"
    reference_object_id: str | None = None
    keepout_radius_km: float | None = None
    goal_range_km: float | None = None
    goal_range_tolerance_km: float | None = None
    goal_radius_km: float | None = None
    goal_relative_ric_km: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    goal_nmt_radial_amplitude_km: float | None = None
    goal_nmt_cross_track_amplitude_km: float = 0.0
    goal_nmt_cross_track_phase_deg: float = 0.0
    goal_nmt_center_ric_km: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    fullscreen: bool = True
    max_history: int = 900
    title: str = "Orbital Engagement Lab - RPO Trainer"
    coast_prediction_horizon_s: float = 300.0
    coast_prediction_orbit_fraction: float | None = 1.0
    coast_prediction_dt_s: float = 10.0
    show_target_coast_prediction: bool = False
    burn_marker_threshold_km_s2: float = 1.0e-12
    forbidden_regions: tuple[ForbiddenRegionConfig, ...] = ()
    approach_gates: tuple[ApproachGateConfig, ...] = ()
    inspection_gates: tuple[InspectionGateConfig, ...] = ()
    plot_overlays_in_zoom: bool = True
    plot_overlays_in_zoom_by_plane: dict[str, bool] = field(default_factory=dict)
    plot_axis_scale: dict[str, tuple[float, float]] = field(default_factory=dict)
    plot_fixed_axis_half_span_km: dict[str, tuple[float | None, float | None]] = field(default_factory=dict)
    plot_equal_axis_scale_planes: tuple[str, ...] = ()
    proximity_ring_plot_planes: tuple[str, ...] = ("RI", "RC", "IC")
    camera_mode: str = "reference"

    def __post_init__(self) -> None:
        try:
            import pygame
        except ImportError as exc:  # pragma: no cover - exercised only without optional dependency.
            raise RuntimeError("Pygame game backend requires `pygame`. Install with `pip install .[game]`.") from exc
        self.pygame = pygame
        pygame.init()
        pygame.font.init()
        flags = pygame.FULLSCREEN | pygame.SCALED if self.fullscreen else pygame.RESIZABLE
        self.screen = pygame.display.set_mode((1280, 720), flags)
        pygame.display.set_caption(self.title)
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Menlo", 18) or pygame.font.Font(None, 18)
        self.small_font = pygame.font.SysFont("Menlo", 14) or pygame.font.Font(None, 14)
        self.large_font = pygame.font.SysFont("Menlo", 26) or pygame.font.Font(None, 26)
        self.closed = False
        self.t_s: list[float] = []
        self.rel_hist: list[np.ndarray] = []
        self.target_rel_hist: list[np.ndarray] = []
        self.thrust_hist: list[np.ndarray] = []
        self.thrust_ric_hist: list[np.ndarray] = []
        self.mean_motion_rad_s: float | None = None
        self._frame_cache: dict[str, np.ndarray] = {}
        self._text_cache: dict[tuple[int, str, tuple[int, int, int]], Any] = {}

    def close(self) -> None:
        self.closed = True
        try:
            self.pygame.event.set_grab(False)
            self.pygame.mouse.set_visible(True)
            self.pygame.display.quit()
            self.pygame.quit()
        except Exception:
            pass

    def clear(self) -> None:
        self.t_s.clear()
        self.rel_hist.clear()
        self.target_rel_hist.clear()
        self.thrust_hist.clear()
        self.thrust_ric_hist.clear()
        self.mean_motion_rad_s = None
        self._frame_cache = {}

    def push_snapshot(self, snapshot: SimulationSnapshot) -> None:
        target = snapshot.truth.get(self.target_object_id)
        chaser = snapshot.truth.get(self.chaser_object_id)
        reference_id = str(self.reference_object_id or self.target_object_id)
        reference = snapshot.truth.get(reference_id)
        if reference is None:
            reference = target
        if target is None or chaser is None or reference is None:
            return
        rel = relative_ric_state_from_arrays(reference, chaser)
        target_rel = relative_ric_state_from_arrays(reference, target)
        reference_arr = np.array(reference, dtype=float).reshape(-1)
        if reference_arr.size >= 6:
            r_norm = float(np.linalg.norm(reference_arr[:3]))
            if r_norm > 0.0 and np.isfinite(r_norm):
                self.mean_motion_rad_s = float(np.sqrt(EARTH_MU_KM3_S2 / (r_norm**3)))
        self.t_s.append(float(snapshot.time_s))
        self.rel_hist.append(rel)
        thrust = snapshot.applied_thrust.get(self.chaser_object_id, np.zeros(3, dtype=float))
        thrust_eci = np.array(thrust, dtype=float).reshape(3)
        self.thrust_hist.append(thrust_eci)
        self.target_rel_hist.append(target_rel)
        if reference_arr.size >= 6:
            c_ir = ric_dcm_ir_from_rv(reference_arr[:3], reference_arr[3:6])
            self.thrust_ric_hist.append(c_ir.T @ thrust_eci)
        else:
            self.thrust_ric_hist.append(thrust_eci)
        while len(self.t_s) > int(max(self.max_history, 2)):
            self.t_s.pop(0)
            self.rel_hist.pop(0)
            if self.target_rel_hist:
                self.target_rel_hist.pop(0)
            if self.thrust_hist:
                self.thrust_hist.pop(0)
            if self.thrust_ric_hist:
                self.thrust_ric_hist.pop(0)

    def draw(
        self,
        *,
        command_status: str = "",
        coach_hint: str = "",
        mission_state: str = "active",
        mission_metrics: tuple[str, ...] = (),
        objective_checklist: tuple[str, ...] = (),
        speed_multiple: float = 1.0,
        briefing_lines: tuple[str, ...] = (),
        debrief_lines: tuple[str, ...] = (),
    ) -> None:
        pygame = self.pygame
        if self.closed:
            return
        width, height = self.screen.get_size()
        self._prepare_frame_cache()
        self.screen.fill((12, 16, 22))
        top = pygame.Rect(36, 18, width - 72, 84)
        left = pygame.Rect(36, 124, max((width - 108) // 2, 200), max(height - 256, 250))
        right = pygame.Rect(left.right + 36, 78, left.width, left.height)
        right.y = left.y
        hud = pygame.Rect(36, height - 112, width - 72, 86)
        self._draw_top_bar(
            top,
            mission_state=mission_state,
            mission_metrics=mission_metrics,
            objective_checklist=objective_checklist,
        )
        self._draw_panel(left, "RI Plane: in-track vs radial", x_axis=1, y_axis=0)
        self._draw_panel(right, "RC Plane: cross-track vs radial", x_axis=2, y_axis=0)
        self._draw_hud(hud, command_status=command_status, coach_hint=coach_hint, speed_multiple=speed_multiple)
        if briefing_lines:
            self._draw_briefing(briefing_lines)
        if mission_state in {"passed", "failed"}:
            self._draw_mission_banner(mission_state, debrief_lines=debrief_lines)
        pygame.display.flip()
        self._frame_cache = {}

    def tick(self, fps: float = 60.0) -> None:
        self.clock.tick(float(max(fps, 1.0)))

    def _draw_panel(self, rect: Any, title: str, x_axis: int, y_axis: int) -> None:
        pygame = self.pygame
        pygame.draw.rect(self.screen, (20, 27, 36), rect, border_radius=10)
        pygame.draw.rect(self.screen, (80, 92, 110), rect, width=1, border_radius=10)
        self._text(title, (rect.x + 14, rect.y + 10), self.font, (230, 235, 242))
        plot = rect.inflate(-48, -72)
        plot.y += 28
        pygame.draw.rect(self.screen, (8, 11, 16), plot)
        pygame.draw.rect(self.screen, (72, 84, 102), plot, width=1)
        if not self.rel_hist:
            return
        rel = self._frame_cache.get("rel")
        target_rel = self._frame_cache.get("target_rel")
        if rel is None or target_rel is None:
            rel = np.vstack(self.rel_hist)
            target_rel = (
                np.vstack(self.target_rel_hist[-rel.shape[0] :])
                if self.target_rel_hist
                else np.zeros((rel.shape[0], 6), dtype=float)
            )
        target_current = target_rel[-1, :3] if target_rel.size else np.zeros(3, dtype=float)
        ghost = self._frame_cache.get("ghost")
        if ghost is None:
            ghost = self._coast_prediction()
        target_ghost = self._frame_cache.get("target_ghost")
        if target_ghost is None:
            target_ghost = self._target_coast_prediction(target_rel)
        goal = np.array(self.goal_relative_ric_km, dtype=float).reshape(-1)
        nmt = self._frame_cache.get("nmt")
        if nmt is None:
            nmt = self._nmt_points()
        chaser_current = rel[-1, :3]
        camera_center = self._camera_center_ric(
            chaser_current=chaser_current,
            target_current=target_current,
            x_axis=x_axis,
            y_axis=y_axis,
        )
        pts_for_scale = [
            (chaser_current - camera_center)[[x_axis, y_axis]].reshape(1, 2),
            (target_current - camera_center)[[x_axis, y_axis]].reshape(1, 2),
        ]
        min_span_km = self._minimum_plot_span_km(
            x_axis=x_axis,
            y_axis=y_axis,
            offset=target_current - camera_center,
        )
        scale_x, scale_y = self._axis_scales_for_plot(
            plot,
            pts=pts_for_scale,
            min_span_km=min_span_km,
            x_axis=x_axis,
            y_axis=y_axis,
        )
        axis_scale_x, axis_scale_y = self._axis_scale_for_plane(x_axis=x_axis, y_axis=y_axis)

        def to_px(point: np.ndarray) -> tuple[int, int]:
            shifted = np.array(point, dtype=float).reshape(-1)[:3] - camera_center
            x = float(shifted[x_axis])
            y = float(shifted[y_axis])
            px = plot.centerx + int(round(x * scale_x))
            py = plot.centery - int(round(y * scale_y))
            return px, py

        def circle_rect(center: tuple[int, int], radius_km: float) -> Any:
            rx = max(1, int(round(float(radius_km) * scale_x)))
            ry = max(1, int(round(float(radius_km) * scale_y)))
            return self.pygame.Rect(center[0] - rx, center[1] - ry, rx * 2, ry * 2)

        previous_clip = self.screen.get_clip()
        self.screen.set_clip(plot)
        self._draw_grid(plot, scale_x=scale_x, scale_y=scale_y)
        self._draw_forbidden_regions(plot, x_axis=x_axis, y_axis=y_axis, to_px=to_px, offset=target_current)
        self._draw_inspection_gates(plot, x_axis=x_axis, y_axis=y_axis, to_px=to_px, offset=target_current)
        self._draw_approach_gates(plot, x_axis=x_axis, y_axis=y_axis, to_px=to_px, offset=target_current)
        if self._show_proximity_rings_for_plane(x_axis=x_axis, y_axis=y_axis):
            if self.keepout_radius_km is not None and float(self.keepout_radius_km) > 0.0:
                center = to_px(target_current)
                pygame.draw.ellipse(
                    self.screen, (190, 68, 68), circle_rect(center, float(self.keepout_radius_km)), width=2
                )
            if self.goal_range_km is not None and float(self.goal_range_km) > 0.0:
                center = to_px(target_current)
                outer = float(self.goal_range_km)
                inner = float(self.goal_range_km)
                if self.goal_range_tolerance_km is not None:
                    tol = max(float(self.goal_range_tolerance_km), 0.0)
                    inner = max(float(self.goal_range_km) - tol, 0.0)
                    outer = float(self.goal_range_km) + tol
                pygame.draw.ellipse(self.screen, (78, 178, 112), circle_rect(center, outer), width=2)
                if inner > 0.0 and not np.isclose(inner, outer):
                    pygame.draw.ellipse(self.screen, (78, 178, 112), circle_rect(center, inner), width=2)
            if self.goal_radius_km is not None and float(self.goal_radius_km) > 0.0 and goal.size == 3:
                center = to_px(target_current + goal)
                pygame.draw.ellipse(
                    self.screen, (78, 178, 112), circle_rect(center, float(self.goal_radius_km)), width=2
                )
        if nmt.size:
            nmt_pts = [to_px(row) for row in nmt]
            if x_axis == 1 and y_axis == 0:
                pygame.draw.lines(self.screen, (78, 178, 112), True, nmt_pts, width=2)
            else:
                self._draw_polyline_dashed(nmt_pts, color=(78, 178, 112), dash_px=7, gap_px=6, width=2)
        pygame.draw.line(self.screen, (90, 104, 124), (plot.left, plot.centery), (plot.right, plot.centery), width=1)
        pygame.draw.line(self.screen, (90, 104, 124), (plot.centerx, plot.top), (plot.centerx, plot.bottom), width=1)
        pygame.draw.circle(self.screen, (80, 92, 112), (plot.centerx, plot.centery), 4)
        target_px = to_px(target_current)
        if bool(getattr(self, "show_target_coast_prediction", False)) and target_ghost.size:
            target_ghost_pts = [to_px(row) for row in _sample_rows(target_ghost, MAX_GHOST_DRAW_POINTS)]
            self._draw_polyline_dashed(target_ghost_pts, color=(135, 150, 172), dash_px=10, gap_px=7, width=2)
        if target_rel.size and len(target_rel) >= 2:
            target_trail = [to_px(row) for row in _sample_rows(target_rel[-self.max_history :], MAX_TRAIL_DRAW_POINTS)]
            pygame.draw.lines(self.screen, (245, 205, 92), False, target_trail, width=2)
        pygame.draw.circle(self.screen, (245, 205, 92), target_px, 6)

        trail_rows = _sample_rows(rel[-self.max_history :], MAX_TRAIL_DRAW_POINTS)
        trail = [to_px(row) for row in trail_rows]
        if ghost.size:
            ghost_pts = [to_px(row) for row in _sample_rows(ghost, MAX_GHOST_DRAW_POINTS)]
            self._draw_polyline_dashed(ghost_pts, color=(135, 150, 172), dash_px=8, gap_px=8, width=2)
        if len(trail) >= 2:
            pygame.draw.lines(self.screen, (215, 86, 86), False, trail, width=2)
        self._draw_burn_markers(rel=rel, to_px=to_px)
        chaser = trail[-1]
        pygame.draw.circle(self.screen, (245, 92, 92), chaser, 7)

        if rel.shape[1] >= 6:
            v = np.zeros(3, dtype=float)
            v[[x_axis, y_axis]] = rel[-1, [x_axis + 3, y_axis + 3]]
            v_px = np.array([v[x_axis] * 120.0 * scale_x, v[y_axis] * 120.0 * scale_y], dtype=float)
            self._draw_vector(
                to_px(rel[-1]), v_px, color=(245, 205, 92), scale=1.0, label="Vrel"
            )
        if self.thrust_ric_hist:
            thrust_ric = self.thrust_ric_hist[-1]
            if np.linalg.norm(thrust_ric) > 0.0:
                vec = (
                    np.array([thrust_ric[x_axis] * axis_scale_x, thrust_ric[y_axis] * axis_scale_y], dtype=float)
                    * 5.0e5
                )
                self._draw_vector(chaser, vec, color=(92, 220, 160), scale=1.0, label="Thrust")
        self.screen.set_clip(previous_clip)

        xlbl = "I km" if x_axis == 1 else "C km"
        ylbl = "R km"
        self._text(xlbl, (plot.right - 56, plot.centery + 8), self.small_font, (170, 180, 195))
        self._text(ylbl, (plot.centerx + 8, plot.top + 8), self.small_font, (170, 180, 195))

    def _prepare_frame_cache(self) -> None:
        if not self.rel_hist:
            self._frame_cache = {}
            return
        rel = np.vstack(self.rel_hist)
        target_rel = (
            np.vstack(self.target_rel_hist[-rel.shape[0] :])
            if self.target_rel_hist
            else np.zeros((rel.shape[0], 6), dtype=float)
        )
        self._frame_cache = {
            "rel": rel,
            "target_rel": target_rel,
            "ghost": self._coast_prediction_from(rel[-1]),
            "target_ghost": self._target_coast_prediction(target_rel),
            "nmt": self._nmt_points(),
        }

    def _draw_grid(
        self,
        plot: Any,
        *,
        scale: float | None = None,
        scale_x: float | None = None,
        scale_y: float | None = None,
    ) -> None:
        pygame = self.pygame
        if scale_x is None:
            scale_x = scale
        if scale_y is None:
            scale_y = scale
        if scale_x is None or scale_y is None or scale_x <= 0.0 or scale_y <= 0.0:
            return
        visible_span_km = max(plot.width / max(scale_x, 1e-9), plot.height / max(scale_y, 1e-9))
        step_km = self._nice_step(visible_span_km / 6.0)
        if step_km <= 0.0:
            return
        max_x_km = plot.width / max(scale_x, 1e-9)
        max_y_km = plot.height / max(scale_y, 1e-9)
        for k in np.arange(-max_x_km, max_x_km + step_km, step_km):
            x = plot.centerx + int(round(float(k) * scale_x))
            if plot.left <= x <= plot.right:
                pygame.draw.line(self.screen, (30, 38, 50), (x, plot.top), (x, plot.bottom), width=1)
        for k in np.arange(-max_y_km, max_y_km + step_km, step_km):
            y = plot.centery - int(round(float(k) * scale_y))
            if plot.top <= y <= plot.bottom:
                pygame.draw.line(self.screen, (30, 38, 50), (plot.left, y), (plot.right, y), width=1)

    def _range_goal_projection_points(self, *, x_axis: int, y_axis: int, offset: np.ndarray) -> list[np.ndarray]:
        if self.goal_range_km is None:
            return []
        radius = float(self.goal_range_km)
        if self.goal_range_tolerance_km is not None:
            radius += max(float(self.goal_range_tolerance_km), 0.0)
        if radius <= 0.0 or not np.isfinite(radius):
            return []
        center = np.array([offset[x_axis], offset[y_axis]], dtype=float)
        return [
            np.array(
                [
                    center + np.array([radius, 0.0], dtype=float),
                    center + np.array([-radius, 0.0], dtype=float),
                    center + np.array([0.0, radius], dtype=float),
                    center + np.array([0.0, -radius], dtype=float),
                ],
                dtype=float,
            )
        ]

    def _forbidden_region_projection_points(
        self, *, x_axis: int, y_axis: int, offset: np.ndarray
    ) -> list[np.ndarray]:
        pts: list[np.ndarray] = []
        for region in self.forbidden_regions:
            if not _region_visible_on_plane(region, x_axis=x_axis, y_axis=y_axis):
                continue
            if region.kind == "annular_sector":
                polygon = region.sector_polygon_ric()
                if polygon.size:
                    polygon = polygon.copy()
                    polygon[:, :3] += np.array(offset, dtype=float).reshape(1, 3)
                    pts.append(polygon[:, [x_axis, y_axis]])
                continue
            if region.kind == "cylinder":
                polygon = _cylinder_projection_polygon_ric(region, x_axis=x_axis, y_axis=y_axis)
                if polygon.size:
                    polygon = polygon.copy()
                    polygon[:, :3] += np.array(offset, dtype=float).reshape(1, 3)
                    pts.append(polygon[:, [x_axis, y_axis]])
                continue
            bounds = _finite_projected_region_bounds(region, x_axis=x_axis, y_axis=y_axis)
            if bounds is None:
                continue
            lo, hi = bounds
            corners = np.array(
                [
                    [lo[0], lo[1]],
                    [lo[0], hi[1]],
                    [hi[0], lo[1]],
                    [hi[0], hi[1]],
                ],
                dtype=float,
            )
            corners += np.array([offset[x_axis], offset[y_axis]], dtype=float).reshape(1, 2)
            pts.append(corners)
        return pts

    def _draw_forbidden_regions(self, plot: Any, *, x_axis: int, y_axis: int, to_px: Any, offset: np.ndarray) -> None:
        if not self.forbidden_regions:
            return
        pygame = self.pygame
        for region in self.forbidden_regions:
            if not _region_visible_on_plane(region, x_axis=x_axis, y_axis=y_axis):
                continue
            if region.kind == "annular_sector":
                polygon = region.sector_polygon_ric()
                if not polygon.size:
                    continue
                polygon = polygon.copy()
                polygon[:, :3] += np.array(offset, dtype=float).reshape(1, 3)
                points = [to_px(row) for row in polygon]
                if len(points) >= 3:
                    self._draw_translucent_polygon(plot, points, color=(168, 44, 54, 58))
                    pygame.draw.lines(self.screen, (230, 80, 92), True, points, width=1)
                continue
            if region.kind == "cylinder":
                polygon = _cylinder_projection_polygon_ric(region, x_axis=x_axis, y_axis=y_axis)
                if not polygon.size:
                    continue
                polygon = polygon.copy()
                polygon[:, :3] += np.array(offset, dtype=float).reshape(1, 3)
                points = [to_px(row) for row in polygon]
                if len(points) >= 3:
                    self._draw_translucent_polygon(plot, points, color=(168, 44, 54, 58))
                    pygame.draw.lines(self.screen, (230, 80, 92), True, points, width=1)
                continue
            bounds = _finite_projected_region_bounds(region, x_axis=x_axis, y_axis=y_axis)
            if bounds is None:
                continue
            lo, hi = bounds
            p_min = np.zeros(3, dtype=float)
            p_max = np.zeros(3, dtype=float)
            p_min[x_axis] = lo[0] + float(offset[x_axis])
            p_min[y_axis] = lo[1] + float(offset[y_axis])
            p_max[x_axis] = hi[0] + float(offset[x_axis])
            p_max[y_axis] = hi[1] + float(offset[y_axis])
            a = to_px(p_min)
            b = to_px(p_max)
            rect = pygame.Rect(min(a[0], b[0]), min(a[1], b[1]), abs(a[0] - b[0]), abs(a[1] - b[1]))
            clipped = rect.clip(plot)
            if clipped.width <= 0 or clipped.height <= 0:
                continue
            fill = pygame.Surface((clipped.width, clipped.height), pygame.SRCALPHA)
            fill.fill((168, 44, 54, 58))
            self.screen.blit(fill, (clipped.x, clipped.y))
            pygame.draw.rect(self.screen, (230, 80, 92), clipped, width=1)

    def _approach_gate_projection_points(
        self, *, x_axis: int, y_axis: int, offset: np.ndarray
    ) -> list[np.ndarray]:
        pts: list[np.ndarray] = []
        for gate in self.approach_gates:
            bounds = _approach_gate_projected_bounds(gate, x_axis=x_axis, y_axis=y_axis)
            if bounds is None:
                continue
            lo, hi = bounds
            corners = np.array(
                [
                    [lo[0], lo[1]],
                    [lo[0], hi[1]],
                    [hi[0], lo[1]],
                    [hi[0], hi[1]],
                ],
                dtype=float,
            )
            corners += np.array([offset[x_axis], offset[y_axis]], dtype=float).reshape(1, 2)
            pts.append(corners)
        return pts

    def _inspection_gate_projection_points(
        self, *, x_axis: int, y_axis: int, offset: np.ndarray
    ) -> list[np.ndarray]:
        pts: list[np.ndarray] = []
        for gate in self.inspection_gates:
            lo = np.array(gate.center_ric_km, dtype=float).reshape(3) - np.array(
                gate.half_width_ric_km, dtype=float
            ).reshape(3)
            hi = np.array(gate.center_ric_km, dtype=float).reshape(3) + np.array(
                gate.half_width_ric_km, dtype=float
            ).reshape(3)
            corners = np.array(
                [
                    [lo[x_axis], lo[y_axis]],
                    [lo[x_axis], hi[y_axis]],
                    [hi[x_axis], lo[y_axis]],
                    [hi[x_axis], hi[y_axis]],
                ],
                dtype=float,
            )
            corners += np.array([offset[x_axis], offset[y_axis]], dtype=float).reshape(1, 2)
            pts.append(corners)
        return pts

    def _draw_approach_gates(self, plot: Any, *, x_axis: int, y_axis: int, to_px: Any, offset: np.ndarray) -> None:
        if not self.approach_gates:
            return
        pygame = self.pygame
        for gate in self.approach_gates:
            bounds = _approach_gate_projected_bounds(gate, x_axis=x_axis, y_axis=y_axis)
            if bounds is None:
                continue
            lo, hi = bounds
            p_min = np.zeros(3, dtype=float)
            p_max = np.zeros(3, dtype=float)
            p_min[x_axis] = lo[0] + float(offset[x_axis])
            p_min[y_axis] = lo[1] + float(offset[y_axis])
            p_max[x_axis] = hi[0] + float(offset[x_axis])
            p_max[y_axis] = hi[1] + float(offset[y_axis])
            a = to_px(p_min)
            b = to_px(p_max)
            rect = pygame.Rect(min(a[0], b[0]), min(a[1], b[1]), abs(a[0] - b[0]), abs(a[1] - b[1]))
            clipped = rect.clip(plot)
            if clipped.width <= 0 or clipped.height <= 0:
                continue
            fill = pygame.Surface((clipped.width, clipped.height), pygame.SRCALPHA)
            fill.fill((245, 205, 92, 46))
            self.screen.blit(fill, (clipped.x, clipped.y))
            pygame.draw.rect(self.screen, (245, 205, 92), clipped, width=1)

    def _draw_inspection_gates(self, plot: Any, *, x_axis: int, y_axis: int, to_px: Any, offset: np.ndarray) -> None:
        if not self.inspection_gates:
            return
        pygame = self.pygame
        for idx, gate in enumerate(self.inspection_gates, start=1):
            lo = np.array(gate.center_ric_km, dtype=float) - np.array(gate.half_width_ric_km, dtype=float)
            hi = np.array(gate.center_ric_km, dtype=float) + np.array(gate.half_width_ric_km, dtype=float)
            p_min = np.zeros(3, dtype=float)
            p_max = np.zeros(3, dtype=float)
            p_min[x_axis] = lo[x_axis] + float(offset[x_axis])
            p_min[y_axis] = lo[y_axis] + float(offset[y_axis])
            p_max[x_axis] = hi[x_axis] + float(offset[x_axis])
            p_max[y_axis] = hi[y_axis] + float(offset[y_axis])
            a = to_px(p_min)
            b = to_px(p_max)
            rect = pygame.Rect(min(a[0], b[0]), min(a[1], b[1]), abs(a[0] - b[0]), abs(a[1] - b[1]))
            clipped = rect.clip(plot)
            if clipped.width <= 0 or clipped.height <= 0:
                continue
            fill = pygame.Surface((clipped.width, clipped.height), pygame.SRCALPHA)
            fill.fill((78, 178, 112, 42))
            self.screen.blit(fill, (clipped.x, clipped.y))
            pygame.draw.rect(self.screen, (100, 226, 142), clipped, width=1)
            self._text(str(idx), (clipped.x + 5, clipped.y + 4), self.small_font, (170, 250, 190))

    def _draw_hud(self, rect: Any, *, command_status: str, coach_hint: str, speed_multiple: float) -> None:
        pygame = self.pygame
        pygame.draw.rect(self.screen, (18, 24, 32), rect, border_radius=10)
        pygame.draw.rect(self.screen, (82, 96, 118), rect, width=1, border_radius=10)
        if self.rel_hist:
            rel = self.rel_hist[-1]
            rng = float(np.linalg.norm(rel[:3]))
            spd = float(np.linalg.norm(rel[3:]))
            t = self.t_s[-1] if self.t_s else 0.0
            self._text(
                f"t={t:7.1f}s   range={rng:7.3f} km   rel speed={spd:8.5f} km/s",
                (rect.x + 16, rect.y + 12),
                self.font,
                (235, 240, 245),
            )
        self._text(
            coach_hint or "",
            (rect.x + 16, rect.y + 38),
            self.small_font,
            (245, 210, 110),
        )
        command_line = command_status.splitlines()[0] if command_status else ""
        if command_line:
            self._text(command_line, (rect.x + 16, rect.y + 60), self.small_font, (195, 205, 220))
        self._text(
            f"Speed {float(speed_multiple):.0f}x   Up/Down speed   Space pause   . step   R reset   Esc quit",
            (rect.right - 590, rect.y + 58),
            self.small_font,
            (220, 160, 160),
        )

    def _draw_top_bar(
        self,
        rect: Any,
        *,
        mission_state: str,
        mission_metrics: tuple[str, ...],
        objective_checklist: tuple[str, ...] = (),
    ) -> None:
        pygame = self.pygame
        colors = {
            "active": ((18, 24, 32), (82, 96, 118), (230, 235, 242), "LEVEL ACTIVE"),
            "passed": ((18, 54, 36), (88, 190, 122), (190, 255, 205), "LEVEL PASSED"),
            "failed": ((62, 24, 28), (220, 94, 94), (255, 205, 205), "LEVEL FAILED"),
        }
        fill, stroke, text_color, label = colors.get(mission_state, colors["active"])
        pygame.draw.rect(self.screen, fill, rect, border_radius=10)
        pygame.draw.rect(self.screen, stroke, rect, width=2, border_radius=10)
        self._text(label, (rect.x + 16, rect.y + 9), self.large_font, text_color)
        x = rect.x + 16
        y = rect.y + 43
        checklist_width = 300 if objective_checklist else 0
        hidden = 0
        for metric in mission_metrics:
            clean, metric_color = self._metric_text_and_color(metric)
            metric_width = max(118, self.small_font.size(clean)[0] + 26)
            if x + metric_width > rect.right - 16 - checklist_width:
                if y > rect.y + 43:
                    hidden += 1
                    continue
                x = rect.x + 16
                y = rect.y + 63
            self._text(clean, (x, y), self.small_font, metric_color)
            x += metric_width
        if hidden:
            self._text(
                f"+{hidden}",
                (rect.right - 34 - checklist_width, rect.y + 63),
                self.small_font,
                (222, 230, 238),
            )
        if objective_checklist:
            self._draw_objective_checklist(rect, objective_checklist)

    def _draw_objective_checklist(self, rect: Any, objective_checklist: tuple[str, ...]) -> None:
        x = rect.right - 292
        y = rect.y + 12
        self._text("OBJECTIVES", (x, y), self.small_font, (170, 184, 204))
        y += 18
        for item in objective_checklist[:3]:
            clean, color = self._metric_text_and_color(str(item))
            self._text(clean, (x, y), self.small_font, color)
            y += 17

    def _metric_text_and_color(self, metric: str) -> tuple[str, tuple[int, int, int]]:
        text = str(metric)
        if text.startswith("OK "):
            return text, (150, 235, 170)
        if text.startswith("WARN "):
            return text, (245, 210, 110)
        if text.startswith("FAIL "):
            return text, (255, 150, 150)
        if text.startswith("INFO "):
            return text, (150, 205, 245)
        return text, (222, 230, 238)

    def _draw_briefing(self, lines: tuple[str, ...]) -> None:
        pygame = self.pygame
        width, height = self.screen.get_size()
        rect = pygame.Rect(width // 2 - 430, height // 2 - 230, 860, 460)
        pygame.draw.rect(self.screen, (15, 24, 34), rect, border_radius=10)
        pygame.draw.rect(self.screen, (96, 174, 224), rect, width=2, border_radius=10)
        title = self.large_font.render(str(lines[0] if lines else "Mission Brief"), True, (238, 244, 250))
        self.screen.blit(title, (rect.x + 30, rect.y + 24))
        y = rect.y + 70
        for raw in lines[1:9]:
            for line in self._wrap_text(str(raw), 82):
                self._text(line, (rect.x + 32, y), self.font, (206, 218, 232))
                y += 24
            y += 4
        self._text(
            "Press Space to start. Esc returns to level select.",
            (rect.x + 32, rect.bottom - 48),
            self.font,
            (220, 160, 160),
        )

    def _draw_mission_banner(self, mission_state: str, *, debrief_lines: tuple[str, ...] = ()) -> None:
        pygame = self.pygame
        width, height = self.screen.get_size()
        rect = pygame.Rect(width // 2 - 360, height // 2 - 190, 720, 380)
        if mission_state == "passed":
            fill = (24, 86, 48)
            stroke = (108, 232, 142)
            text = "MISSION PASSED"
            sub = "Press R to replay or Esc to quit"
            color = (210, 255, 220)
        else:
            fill = (90, 30, 36)
            stroke = (244, 102, 102)
            text = "MISSION FAILED"
            sub = "Press R to retry or Esc to quit"
            color = (255, 220, 220)
        pygame.draw.rect(self.screen, fill, rect, border_radius=10)
        pygame.draw.rect(self.screen, stroke, rect, width=3, border_radius=10)
        title = self.large_font.render(text, True, color)
        self.screen.blit(title, (rect.centerx - title.get_width() // 2, rect.y + 22))
        y = rect.y + 74
        for line in debrief_lines[:12]:
            surf = self.font.render(str(line), True, color)
            self.screen.blit(surf, (rect.x + 36, y))
            y += 24
        subtitle = self.font.render(sub, True, color)
        self.screen.blit(subtitle, (rect.centerx - subtitle.get_width() // 2, rect.bottom - 42))

    def _draw_vector(
        self,
        origin: tuple[int, int],
        vector: np.ndarray,
        *,
        color: tuple[int, int, int],
        scale: float,
        label: str = "",
    ) -> None:
        pygame = self.pygame
        vec = np.array(vector, dtype=float).reshape(2)
        if not np.all(np.isfinite(vec)) or np.linalg.norm(vec) <= 0.0:
            return
        end = (int(origin[0] + vec[0] * scale), int(origin[1] - vec[1] * scale))
        pygame.draw.line(self.screen, color, origin, end, width=2)
        pygame.draw.circle(self.screen, color, end, 4)
        if label:
            self._text(label, (end[0] + 6, end[1] - 8), self.small_font, color)

    def _draw_burn_markers(self, *, rel: np.ndarray, to_px: Any) -> None:
        if not self.thrust_ric_hist:
            return
        pygame = self.pygame
        thrust = np.vstack(self.thrust_ric_hist[-rel.shape[0] :])
        active = np.linalg.norm(thrust, axis=1) > float(self.burn_marker_threshold_km_s2)
        idxs = np.where(active)[0]
        if idxs.size == 0:
            return
        stride = max(1, int(np.ceil(idxs.size / 80)))
        for idx in idxs[::stride]:
            pygame.draw.circle(self.screen, (255, 145, 60), to_px(rel[int(idx)]), 3)

    def _draw_translucent_polygon(
        self, plot: Any, points: list[tuple[int, int]], *, color: tuple[int, int, int, int]
    ) -> None:
        if len(points) < 3 or plot.width <= 0 or plot.height <= 0:
            return
        pygame = self.pygame
        fill = pygame.Surface((plot.width, plot.height), pygame.SRCALPHA)
        local_points = [(int(x) - plot.x, int(y) - plot.y) for x, y in points]
        pygame.draw.polygon(fill, color, local_points)
        self.screen.blit(fill, (plot.x, plot.y))

    def _draw_polyline_dashed(
        self,
        points: list[tuple[int, int]],
        *,
        color: tuple[int, int, int],
        dash_px: int = 8,
        gap_px: int = 6,
        width: int = 1,
    ) -> None:
        if len(points) < 2:
            return
        pygame = self.pygame
        for start, end in zip(points[:-1], points[1:]):
            p0 = np.array(start, dtype=float)
            p1 = np.array(end, dtype=float)
            seg = p1 - p0
            length = float(np.linalg.norm(seg))
            if length <= 0.0:
                continue
            direction = seg / length
            pos = 0.0
            while pos < length:
                a = p0 + direction * pos
                b = p0 + direction * min(pos + dash_px, length)
                pygame.draw.line(self.screen, color, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), width=width)
                pos += dash_px + gap_px

    def _coast_prediction(self) -> np.ndarray:
        if not self.rel_hist:
            return np.empty((0, 6), dtype=float)
        return self._coast_prediction_from(np.array(self.rel_hist[-1], dtype=float).reshape(6))

    def _target_coast_prediction(self, target_rel: np.ndarray | None = None) -> np.ndarray:
        if not bool(getattr(self, "show_target_coast_prediction", False)):
            return np.empty((0, 6), dtype=float)
        rel = np.array(target_rel, dtype=float) if target_rel is not None else np.empty((0, 6), dtype=float)
        if rel.size == 0:
            if not self.target_rel_hist:
                return np.empty((0, 6), dtype=float)
            rel0 = np.array(self.target_rel_hist[-1], dtype=float).reshape(6)
        else:
            rel0 = rel.reshape(-1, 6)[-1]
        return self._coast_prediction_from(rel0)

    def _coast_prediction_from(self, rel0: np.ndarray) -> np.ndarray:
        rel0 = np.array(rel0, dtype=float).reshape(6)
        n = self.mean_motion_rad_s
        if n is None or not np.isfinite(float(n)) or float(n) <= 0.0:
            return np.empty((0, 6), dtype=float)
        horizon = self._coast_prediction_horizon_s(float(n))
        if horizon <= 0.0:
            return np.empty((0, 6), dtype=float)
        dt = float(max(self.coast_prediction_dt_s, 1.0e-6))
        count = min(int(np.floor(horizon / dt)) + 1, MAX_GHOST_DRAW_POINTS)
        times = np.linspace(0.0, horizon, max(count, 2), dtype=float)
        return np.vstack([_cw_coast_state(rel0, float(t), float(n)) for t in times])

    def _coast_prediction_horizon_s(self, mean_motion_rad_s: float) -> float:
        fraction = self.coast_prediction_orbit_fraction
        n = float(mean_motion_rad_s)
        if fraction is None:
            return float(max(self.coast_prediction_horizon_s, 0.0))
        if not np.isfinite(n) or n <= 0.0:
            return 0.0
        return float(max(fraction, 0.0) * (2.0 * np.pi / n))

    def _nmt_points(self) -> np.ndarray:
        if self.goal_nmt_radial_amplitude_km is None:
            return np.empty((0, 3), dtype=float)
        a_r = float(self.goal_nmt_radial_amplitude_km)
        if not np.isfinite(a_r) or a_r <= 0.0:
            return np.empty((0, 3), dtype=float)
        center = np.array(self.goal_nmt_center_ric_km, dtype=float).reshape(-1)
        if center.size != 3:
            center = np.zeros(3, dtype=float)
        a_c = float(self.goal_nmt_cross_track_amplitude_km)
        if not np.isfinite(a_c):
            a_c = 0.0
        phase = np.deg2rad(float(self.goal_nmt_cross_track_phase_deg))
        theta = np.linspace(0.0, 2.0 * np.pi, 181)
        pts = np.zeros((theta.size, 3), dtype=float)
        pts[:, 0] = center[0] + a_r * np.cos(theta)
        pts[:, 1] = center[1] - 2.0 * a_r * np.sin(theta)
        pts[:, 2] = center[2] + a_c * np.cos(theta + phase)
        return pts

    def _scale_for_plot(self, *, pts: list[np.ndarray], min_span_km: float | None = None) -> float:
        finite: list[np.ndarray] = []
        for arr in pts:
            a = np.array(arr, dtype=float).reshape(-1, 2)
            a = a[np.all(np.isfinite(a), axis=1)]
            if a.size:
                finite.append(a)
        span = self._minimum_plot_span_km() if min_span_km is None else float(max(min_span_km, 0.05))
        if finite:
            all_pts = np.vstack(finite)
            span = max(float(np.max(np.abs(all_pts))) * 1.2, span)
        width, height = self.screen.get_size()
        px_span = max(min(width, height) * 0.28, 80.0)
        return float(px_span / max(span, 1e-9))

    def _axis_scales_for_plot(
        self,
        plot: Any,
        *,
        pts: list[np.ndarray],
        min_span_km: float | None,
        x_axis: int,
        y_axis: int,
    ) -> tuple[float, float]:
        axis_scale_x, axis_scale_y = self._axis_scale_for_plane(x_axis=x_axis, y_axis=y_axis)
        fixed_x, fixed_y = self._fixed_axis_half_span_for_plane(x_axis=x_axis, y_axis=y_axis)
        if fixed_x is None and fixed_y is None:
            scale = self._scale_for_plot(pts=pts, min_span_km=min_span_km)
            scale_x = scale * axis_scale_x
            scale_y = scale * axis_scale_y
        else:
            scale_x = self._scale_for_axis(pts=pts, axis_index=0, min_span_km=min_span_km, plot_px=plot.width)
            scale_y = self._scale_for_axis(pts=pts, axis_index=1, min_span_km=min_span_km, plot_px=plot.height)
            scale_x *= axis_scale_x
            scale_y *= axis_scale_y
            if fixed_x is not None:
                scale_x = self._scale_for_fixed_half_span(plot_px=plot.width, half_span_km=fixed_x)
            if fixed_y is not None:
                scale_y = self._scale_for_fixed_half_span(plot_px=plot.height, half_span_km=fixed_y)
        if self._equal_axis_scale_for_plane(x_axis=x_axis, y_axis=y_axis):
            scale_x = scale_y
        return (scale_x, scale_y)

    def _scale_for_axis(
        self,
        *,
        pts: list[np.ndarray],
        axis_index: int,
        min_span_km: float | None,
        plot_px: float,
    ) -> float:
        finite: list[np.ndarray] = []
        for arr in pts:
            a = np.array(arr, dtype=float).reshape(-1, 2)
            a = a[np.all(np.isfinite(a), axis=1)]
            if a.size:
                finite.append(a[:, int(axis_index)])
        span = float(max(0.05 if min_span_km is None else min_span_km, 0.05))
        if finite:
            values = np.concatenate(finite)
            span = max(float(np.max(np.abs(values))) * 1.2, span)
        px_span = max(float(plot_px) * 0.42, 80.0)
        return float(px_span / max(span, 1e-9))

    @staticmethod
    def _scale_for_fixed_half_span(*, plot_px: float, half_span_km: float) -> float:
        span = float(half_span_km)
        if not np.isfinite(span) or span <= 0.0:
            span = 0.05
        return float(max(float(plot_px) * 0.5, 1.0) / span)

    def _axis_scale_for_plane(self, *, x_axis: int, y_axis: int) -> tuple[float, float]:
        key = _plane_key_for_axes(x_axis=x_axis, y_axis=y_axis)
        raw_scales = getattr(self, "plot_axis_scale", {}) or {}
        raw = raw_scales.get(key, (1.0, 1.0))
        try:
            x_scale, y_scale = raw
        except (TypeError, ValueError):
            return (1.0, 1.0)
        x = float(x_scale)
        y = float(y_scale)
        if not np.isfinite(x) or x <= 0.0:
            x = 1.0
        if not np.isfinite(y) or y <= 0.0:
            y = 1.0
        return (x, y)

    def _fixed_axis_half_span_for_plane(self, *, x_axis: int, y_axis: int) -> tuple[float | None, float | None]:
        key = _plane_key_for_axes(x_axis=x_axis, y_axis=y_axis)
        raw_spans = getattr(self, "plot_fixed_axis_half_span_km", {}) or {}
        raw = raw_spans.get(key, (None, None))
        try:
            x_span, y_span = raw
        except (TypeError, ValueError):
            return (None, None)
        return (_positive_float_or_none(x_span), _positive_float_or_none(y_span))

    def _equal_axis_scale_for_plane(self, *, x_axis: int, y_axis: int) -> bool:
        key = _plane_key_for_axes(x_axis=x_axis, y_axis=y_axis)
        planes = tuple(str(plane or "").strip().upper() for plane in getattr(self, "plot_equal_axis_scale_planes", ()))
        return key in planes

    def _show_proximity_rings_for_plane(self, *, x_axis: int, y_axis: int) -> bool:
        key = _plane_key_for_axes(x_axis=x_axis, y_axis=y_axis)
        planes = tuple(str(plane or "").strip().upper() for plane in getattr(self, "proximity_ring_plot_planes", ()))
        return key in planes

    def _camera_center_ric(
        self,
        *,
        chaser_current: np.ndarray,
        target_current: np.ndarray,
        x_axis: int | None = None,
        y_axis: int | None = None,
    ) -> np.ndarray:
        mode = str(getattr(self, "camera_mode", "reference") or "reference").strip().lower()
        if mode not in {"target_pair", "satellite_pair", "pair"}:
            return np.zeros(3, dtype=float)
        if {x_axis, y_axis} == {0, 2}:
            return np.zeros(3, dtype=float)
        chaser = np.array(chaser_current, dtype=float).reshape(-1)
        target = np.array(target_current, dtype=float).reshape(-1)
        if chaser.size < 3 or target.size < 3:
            return np.zeros(3, dtype=float)
        pair = np.vstack((chaser[:3], target[:3]))
        if not np.all(np.isfinite(pair)):
            return np.zeros(3, dtype=float)
        center = np.zeros(3, dtype=float)
        axes = (0, 1) if x_axis is None or y_axis is None else (int(x_axis), int(y_axis))
        for axis in axes:
            center[axis] = float(np.mean(pair[:, axis]))
        return center

    def _minimum_plot_span_km(
        self,
        *,
        x_axis: int | None = None,
        y_axis: int | None = None,
        offset: np.ndarray | None = None,
    ) -> float:
        span = 0.05
        if x_axis is None or y_axis is None:
            return span
        plane = _plane_key_for_axes(x_axis=int(x_axis), y_axis=int(y_axis))
        overlay_zoom_by_plane = getattr(self, "plot_overlays_in_zoom_by_plane", {}) or {}
        include_overlays = bool(overlay_zoom_by_plane.get(plane, getattr(self, "plot_overlays_in_zoom", True)))
        if not include_overlays:
            return span
        origin_offset = np.zeros(3, dtype=float) if offset is None else np.array(offset, dtype=float).reshape(3)
        overlay_points = [
            *self._forbidden_region_projection_points(x_axis=int(x_axis), y_axis=int(y_axis), offset=origin_offset),
            *self._approach_gate_projection_points(x_axis=int(x_axis), y_axis=int(y_axis), offset=origin_offset),
            *self._inspection_gate_projection_points(x_axis=int(x_axis), y_axis=int(y_axis), offset=origin_offset),
        ]
        finite: list[np.ndarray] = []
        for points in overlay_points:
            projected = np.array(points, dtype=float).reshape(-1, 2)
            projected = projected[np.all(np.isfinite(projected), axis=1)]
            if projected.size:
                finite.append(projected)
        if finite:
            span = max(span, float(np.max(np.abs(np.vstack(finite)))) * PLOT_OVERLAY_MARGIN)
        return span

    @staticmethod
    def _nice_step(value: float) -> float:
        if value <= 0.0 or not np.isfinite(value):
            return 1.0
        exp = np.floor(np.log10(value))
        base = value / (10.0**exp)
        if base <= 1.0:
            nice = 1.0
        elif base <= 2.0:
            nice = 2.0
        elif base <= 5.0:
            nice = 5.0
        else:
            nice = 10.0
        return float(nice * (10.0**exp))

    def _text(self, text: str, pos: tuple[int, int], font: Any, color: tuple[int, int, int]) -> None:
        if not text:
            return
        key = (id(font), str(text), tuple(color))
        surf = self._text_cache.get(key)
        if surf is None:
            if len(self._text_cache) >= TEXT_CACHE_LIMIT:
                self._text_cache.clear()
            surf = font.render(str(text), True, color)
            self._text_cache[key] = surf
        self.screen.blit(surf, pos)

    @staticmethod
    def _wrap_text(value: str, max_chars: int) -> list[str]:
        words = str(value or "").split()
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = word if not current else current + " " + word
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines or [""]


def _finite_projected_region_bounds(
    region: ForbiddenRegionConfig, *, x_axis: int, y_axis: int
) -> tuple[np.ndarray, np.ndarray] | None:
    lower = np.array(region.min_ric_km, dtype=float).reshape(3)
    upper = np.array(region.max_ric_km, dtype=float).reshape(3)
    lo = np.array([lower[x_axis], lower[y_axis]], dtype=float)
    hi = np.array([upper[x_axis], upper[y_axis]], dtype=float)
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        return None
    return lo, hi


def _sample_rows(values: np.ndarray, max_points: int) -> np.ndarray:
    arr = np.array(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    limit = max(int(max_points), 2)
    if arr.shape[0] <= limit:
        return arr
    idx = np.linspace(0, arr.shape[0] - 1, limit, dtype=int)
    idx = np.unique(idx)
    return arr[idx]


def _cylinder_projection_polygon_ric(
    region: ForbiddenRegionConfig, *, x_axis: int, y_axis: int, samples: int = 72
) -> np.ndarray:
    if region.radius_km is None or region.height_km is None:
        return np.zeros((0, 3), dtype=float)
    axis = _region_axis_index(region.axis)
    center = np.array(region.center_ric_km, dtype=float).reshape(3)
    radius = float(region.radius_km)
    half_height = float(region.height_km) / 2.0
    if _cylinder_projection_is_cross_section(region, x_axis=x_axis, y_axis=y_axis):
        theta = np.linspace(0.0, 2.0 * np.pi, max(int(samples), 12), endpoint=True)
        pts = np.tile(center.reshape(1, 3), (theta.size, 1))
        cross_x, cross_y = int(x_axis), int(y_axis)
        pts[:, cross_x] += radius * np.cos(theta)
        pts[:, cross_y] += radius * np.sin(theta)
        return pts
    projected_axes = {int(x_axis), int(y_axis)}
    if axis not in projected_axes:
        return np.zeros((0, 3), dtype=float)
    cross_axis = int(y_axis) if int(x_axis) == axis else int(x_axis)
    corners = np.tile(center.reshape(1, 3), (4, 1))
    corners[:, axis] += np.array([-half_height, half_height, half_height, -half_height], dtype=float)
    corners[:, cross_axis] += np.array([-radius, -radius, radius, radius], dtype=float)
    return corners


def _cylinder_projection_is_cross_section(region: ForbiddenRegionConfig, *, x_axis: int, y_axis: int) -> bool:
    axis = _region_axis_index(region.axis)
    projected_axes = {int(x_axis), int(y_axis)}
    cross_axes = {idx for idx in (0, 1, 2) if idx != axis}
    return projected_axes == cross_axes


def _region_axis_index(axis: str) -> int:
    key = str(axis or "").strip().upper()
    if key == "R":
        return 0
    if key == "I":
        return 1
    if key == "C":
        return 2
    return 1


def _positive_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result) or result <= 0.0:
        return None
    return result


def _plane_key_for_axes(*, x_axis: int, y_axis: int) -> str:
    axis_set = {int(x_axis), int(y_axis)}
    if axis_set == {0, 1}:
        return "RI"
    if axis_set == {0, 2}:
        return "RC"
    if axis_set == {1, 2}:
        return "IC"
    return ""


def _region_visible_on_plane(region: ForbiddenRegionConfig, *, x_axis: int, y_axis: int) -> bool:
    planes = tuple(region.plot_planes or ())
    if not planes:
        return True
    plane = _plane_key_for_axes(x_axis=x_axis, y_axis=y_axis)
    if not plane:
        return False
    return plane in planes


def _approach_gate_projected_bounds(
    gate: ApproachGateConfig, *, x_axis: int, y_axis: int
) -> tuple[np.ndarray, np.ndarray] | None:
    axes = {int(x_axis), int(y_axis)}
    if 0 not in axes:
        return None
    lateral_axis = int(x_axis) if int(y_axis) == 0 else int(y_axis)
    if lateral_axis == 1:
        lateral = gate.max_abs_intrack_km
    elif lateral_axis == 2:
        lateral = gate.max_abs_cross_track_km
    else:
        return None
    if lateral is None:
        return None
    lower = np.zeros(3, dtype=float)
    upper = np.zeros(3, dtype=float)
    lower[0] = float(gate.radial_ric_km) - float(gate.radial_tolerance_km)
    upper[0] = float(gate.radial_ric_km) + float(gate.radial_tolerance_km)
    lower[lateral_axis] = -float(lateral)
    upper[lateral_axis] = float(lateral)
    lo = np.array([lower[x_axis], lower[y_axis]], dtype=float)
    hi = np.array([upper[x_axis], upper[y_axis]], dtype=float)
    return lo, hi


def _cw_coast_state(x0: np.ndarray, t_s: float, mean_motion_rad_s: float) -> np.ndarray:
    x, y, z, xd, yd, zd = np.array(x0, dtype=float).reshape(6)
    n = float(mean_motion_rad_s)
    t = float(t_s)
    nt = n * t
    c = float(np.cos(nt))
    s = float(np.sin(nt))
    if abs(n) <= 1.0e-12:
        return np.array([x + xd * t, y + yd * t, z + zd * t, xd, yd, zd], dtype=float)

    xp = (4.0 - 3.0 * c) * x + (s / n) * xd + (2.0 * (1.0 - c) / n) * yd
    yp = 6.0 * (s - nt) * x + y - (2.0 * (1.0 - c) / n) * xd + ((4.0 * s - 3.0 * nt) / n) * yd
    zp = c * z + (s / n) * zd
    xdp = 3.0 * n * s * x + c * xd + 2.0 * s * yd
    ydp = -6.0 * n * (1.0 - c) * x - 2.0 * s * xd + (4.0 * c - 3.0) * yd
    zdp = -n * s * z + c * zd
    return np.array([xp, yp, zp, xdp, ydp, zdp], dtype=float)
