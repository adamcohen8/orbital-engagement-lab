from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sim.api import SimulationSnapshot
from sim.game.training import relative_ric_state_from_arrays
from sim.utils.frames import ric_dcm_ir_from_rv

EARTH_MU_KM3_S2 = 398600.4418


@dataclass
class PygameRPODashboard:
    target_object_id: str = "target"
    chaser_object_id: str = "chaser"
    keepout_radius_km: float | None = None
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
    coast_prediction_dt_s: float = 10.0
    burn_marker_threshold_km_s2: float = 1.0e-12

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
        self.thrust_hist: list[np.ndarray] = []
        self.thrust_ric_hist: list[np.ndarray] = []
        self.mean_motion_rad_s: float | None = None

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
        self.thrust_hist.clear()
        self.thrust_ric_hist.clear()
        self.mean_motion_rad_s = None

    def push_snapshot(self, snapshot: SimulationSnapshot) -> None:
        target = snapshot.truth.get(self.target_object_id)
        chaser = snapshot.truth.get(self.chaser_object_id)
        if target is None or chaser is None:
            return
        rel = relative_ric_state_from_arrays(target, chaser)
        target_arr = np.array(target, dtype=float).reshape(-1)
        if target_arr.size >= 6:
            r_norm = float(np.linalg.norm(target_arr[:3]))
            if r_norm > 0.0 and np.isfinite(r_norm):
                self.mean_motion_rad_s = float(np.sqrt(EARTH_MU_KM3_S2 / (r_norm**3)))
        self.t_s.append(float(snapshot.time_s))
        self.rel_hist.append(rel)
        thrust = snapshot.applied_thrust.get(self.chaser_object_id, np.zeros(3, dtype=float))
        thrust_eci = np.array(thrust, dtype=float).reshape(3)
        self.thrust_hist.append(thrust_eci)
        if target_arr.size >= 6:
            c_ir = ric_dcm_ir_from_rv(target_arr[:3], target_arr[3:6])
            self.thrust_ric_hist.append(c_ir.T @ thrust_eci)
        else:
            self.thrust_ric_hist.append(thrust_eci)
        while len(self.t_s) > int(max(self.max_history, 2)):
            self.t_s.pop(0)
            self.rel_hist.pop(0)
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
    ) -> None:
        pygame = self.pygame
        if self.closed:
            return
        width, height = self.screen.get_size()
        self.screen.fill((12, 16, 22))
        top = pygame.Rect(36, 18, width - 72, 54)
        left = pygame.Rect(36, 96, max((width - 108) // 2, 200), max(height - 228, 250))
        right = pygame.Rect(left.right + 36, 78, left.width, left.height)
        right.y = left.y
        hud = pygame.Rect(36, height - 112, width - 72, 86)
        self._draw_top_bar(top, mission_state=mission_state, mission_metrics=mission_metrics)
        self._draw_panel(left, "RI Plane: in-track vs radial", x_axis=1, y_axis=0)
        self._draw_panel(right, "RC Plane: cross-track vs radial", x_axis=2, y_axis=0)
        self._draw_hud(hud, command_status=command_status, coach_hint=coach_hint)
        if mission_state in {"passed", "failed"}:
            self._draw_mission_banner(mission_state)
        pygame.display.flip()

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
        rel = np.vstack(self.rel_hist)
        ghost = self._coast_prediction()
        goal = np.array(self.goal_relative_ric_km, dtype=float).reshape(-1)
        pts = [rel[:, [x_axis, y_axis]]]
        if ghost.size:
            pts.append(ghost[:, [x_axis, y_axis]])
        if goal.size == 3:
            pts.append(goal[[x_axis, y_axis]].reshape(1, 2))
        nmt = self._nmt_points()
        if nmt.size:
            pts.append(nmt[:, [x_axis, y_axis]])
        scale = self._scale_for_plot(pts=pts)

        def to_px(point: np.ndarray) -> tuple[int, int]:
            x = float(point[x_axis])
            y = float(point[y_axis])
            px = plot.centerx + int(round(x * scale))
            py = plot.centery - int(round(y * scale))
            return px, py

        self._draw_grid(plot, scale=scale)
        if self.keepout_radius_km is not None and float(self.keepout_radius_km) > 0.0:
            radius_px = max(1, int(round(float(self.keepout_radius_km) * scale)))
            pygame.draw.circle(self.screen, (190, 68, 68), (plot.centerx, plot.centery), radius_px, width=2)
        if self.goal_radius_km is not None and float(self.goal_radius_km) > 0.0 and goal.size == 3:
            center = to_px(goal)
            radius_px = max(1, int(round(float(self.goal_radius_km) * scale)))
            pygame.draw.circle(self.screen, (78, 178, 112), center, radius_px, width=2)
        if nmt.size:
            nmt_pts = [to_px(row) for row in nmt]
            if x_axis == 1 and y_axis == 0:
                pygame.draw.lines(self.screen, (78, 178, 112), True, nmt_pts, width=2)
            else:
                self._draw_polyline_dashed(nmt_pts, color=(78, 178, 112), dash_px=7, gap_px=6, width=2)
        pygame.draw.line(self.screen, (90, 104, 124), (plot.left, plot.centery), (plot.right, plot.centery), width=1)
        pygame.draw.line(self.screen, (90, 104, 124), (plot.centerx, plot.top), (plot.centerx, plot.bottom), width=1)
        pygame.draw.circle(self.screen, (60, 140, 220), (plot.centerx, plot.centery), 5)

        trail = [to_px(row) for row in rel[-self.max_history :]]
        if ghost.size:
            ghost_pts = [to_px(row) for row in ghost]
            self._draw_polyline_dashed(ghost_pts, color=(135, 150, 172), dash_px=8, gap_px=8, width=2)
        if len(trail) >= 2:
            pygame.draw.lines(self.screen, (215, 86, 86), False, trail, width=2)
        self._draw_burn_markers(rel=rel, to_px=to_px)
        chaser = trail[-1]
        pygame.draw.circle(self.screen, (245, 92, 92), chaser, 7)

        if rel.shape[1] >= 6:
            v = np.zeros(3, dtype=float)
            v[[x_axis, y_axis]] = rel[-1, [x_axis + 3, y_axis + 3]]
            self._draw_vector(to_px(rel[-1]), v[[x_axis, y_axis]] * 120.0, color=(245, 205, 92), scale=scale, label="Vrel")
        if self.thrust_ric_hist:
            thrust_ric = self.thrust_ric_hist[-1]
            if np.linalg.norm(thrust_ric) > 0.0:
                vec = np.array([thrust_ric[x_axis], thrust_ric[y_axis]], dtype=float) * 5.0e5
                self._draw_vector(chaser, vec, color=(92, 220, 160), scale=1.0, label="Thrust")

        xlbl = "I km" if x_axis == 1 else "C km"
        ylbl = "R km"
        self._text(xlbl, (plot.right - 56, plot.centery + 8), self.small_font, (170, 180, 195))
        self._text(ylbl, (plot.centerx + 8, plot.top + 8), self.small_font, (170, 180, 195))

    def _draw_grid(self, plot: Any, *, scale: float) -> None:
        pygame = self.pygame
        if scale <= 0.0:
            return
        step_km = self._nice_step(max(plot.width, plot.height) / max(scale, 1e-9) / 6.0)
        if step_km <= 0.0:
            return
        max_km = max(plot.width, plot.height) / max(scale, 1e-9)
        for k in np.arange(-max_km, max_km + step_km, step_km):
            x = plot.centerx + int(round(float(k) * scale))
            y = plot.centery - int(round(float(k) * scale))
            if plot.left <= x <= plot.right:
                pygame.draw.line(self.screen, (30, 38, 50), (x, plot.top), (x, plot.bottom), width=1)
            if plot.top <= y <= plot.bottom:
                pygame.draw.line(self.screen, (30, 38, 50), (plot.left, y), (plot.right, y), width=1)

    def _draw_hud(self, rect: Any, *, command_status: str, coach_hint: str) -> None:
        pygame = self.pygame
        pygame.draw.rect(self.screen, (18, 24, 32), rect, border_radius=10)
        pygame.draw.rect(self.screen, (82, 96, 118), rect, width=1, border_radius=10)
        if self.rel_hist:
            rel = self.rel_hist[-1]
            rng = float(np.linalg.norm(rel[:3]))
            spd = float(np.linalg.norm(rel[3:]))
            t = self.t_s[-1] if self.t_s else 0.0
            self._text(f"t={t:7.1f}s   range={rng:7.3f} km   rel speed={spd:8.5f} km/s", (rect.x + 16, rect.y + 12), self.font, (235, 240, 245))
        self._text(command_status.splitlines()[0] if command_status else "", (rect.x + 16, rect.y + 38), self.small_font, (195, 205, 220))
        line2 = command_status.splitlines()[-1] if command_status else ""
        if line2:
            self._text(line2, (rect.x + 16, rect.y + 58), self.small_font, (195, 205, 220))
        self._text("Space pause   . step   R reset   Esc quit", (rect.right - 340, rect.y + 14), self.small_font, (220, 160, 160))

    def _draw_top_bar(self, rect: Any, *, mission_state: str, mission_metrics: tuple[str, ...]) -> None:
        pygame = self.pygame
        colors = {
            "active": ((18, 24, 32), (82, 96, 118), (230, 235, 242), "LEVEL ACTIVE"),
            "passed": ((18, 54, 36), (88, 190, 122), (190, 255, 205), "LEVEL PASSED"),
            "failed": ((62, 24, 28), (220, 94, 94), (255, 205, 205), "LEVEL FAILED"),
        }
        fill, stroke, text_color, label = colors.get(mission_state, colors["active"])
        pygame.draw.rect(self.screen, fill, rect, border_radius=10)
        pygame.draw.rect(self.screen, stroke, rect, width=2, border_radius=10)
        self._text(label, (rect.x + 16, rect.y + 15), self.large_font, text_color)
        x = rect.x + 210
        for metric in mission_metrics[:5]:
            self._text(metric, (x, rect.y + 18), self.small_font, (222, 230, 238))
            x += max(148, len(metric) * 8 + 18)

    def _draw_mission_banner(self, mission_state: str) -> None:
        pygame = self.pygame
        width, height = self.screen.get_size()
        rect = pygame.Rect(width // 2 - 260, height // 2 - 54, 520, 108)
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
        self.screen.blit(title, (rect.centerx - title.get_width() // 2, rect.y + 24))
        subtitle = self.font.render(sub, True, color)
        self.screen.blit(subtitle, (rect.centerx - subtitle.get_width() // 2, rect.y + 64))

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
        rel0 = np.array(self.rel_hist[-1], dtype=float).reshape(6)
        n = self.mean_motion_rad_s
        if n is None or not np.isfinite(float(n)) or float(n) <= 0.0:
            return np.empty((0, 6), dtype=float)
        horizon = float(max(self.coast_prediction_horizon_s, 0.0))
        dt = float(max(self.coast_prediction_dt_s, 1.0e-6))
        times = np.arange(0.0, horizon + 0.5 * dt, dt, dtype=float)
        return np.vstack([_cw_coast_state(rel0, float(t), float(n)) for t in times])

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

    def _scale_for_plot(self, *, pts: list[np.ndarray]) -> float:
        finite: list[np.ndarray] = []
        for arr in pts:
            a = np.array(arr, dtype=float).reshape(-1, 2)
            a = a[np.all(np.isfinite(a), axis=1)]
            if a.size:
                finite.append(a)
        span = 1.0
        if finite:
            all_pts = np.vstack(finite)
            span = max(float(np.max(np.abs(all_pts))), 0.5)
        if self.keepout_radius_km is not None:
            span = max(span, float(abs(self.keepout_radius_km)) * 1.2)
        if self.goal_radius_km is not None:
            span = max(span, float(abs(self.goal_radius_km)) * 1.2)
        if self.goal_nmt_radial_amplitude_km is not None:
            span = max(span, float(abs(self.goal_nmt_radial_amplitude_km)) * 2.4)
        width, height = self.screen.get_size()
        px_span = max(min(width, height) * 0.28, 80.0)
        return float(px_span / max(span, 1e-9))

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
        surf = font.render(str(text), True, color)
        self.screen.blit(surf, pos)


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
