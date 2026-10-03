"""Desktop trainer orthographic camera; consumes existing dashboard evidence."""

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from sim.game import constraint_mesh
from sim.game.dashboard_common import CHASER_MARKER_COLOR, TARGET_MARKER_COLOR
from sim.game.input import pygame_focus_lost


@dataclass
class SandboxCamera:
    enabled: bool = False
    yaw: float = -0.65
    elevation: float = 0.5
    focus: np.ndarray = field(default_factory=lambda: np.zeros(3))
    span: float = 1.0
    in_track_sign: int = 1
    initialized: bool = False
    drag_button: int = 0
    pan_drag: bool = False
    zoom_until: float = 0.0

    def basis(self):
        right = np.array([0.0, self.in_track_sign * np.cos(self.yaw), np.sin(self.yaw)])
        up = np.array(
            [
                np.cos(self.elevation),
                self.in_track_sign * np.sin(self.yaw) * np.sin(self.elevation),
                -np.cos(self.yaw) * np.sin(self.elevation),
            ]
        )
        return np.stack((right, up))

    def active(self, now=None):
        return self.enabled and (bool(self.drag_button) or (perf_counter() if now is None else now) < self.zoom_until)

    def effective_speed(self, speed, now=None):
        return min(speed, 1.0) if self.active(now) else speed

    def cancel(self):
        self.drag_button = 0
        self.zoom_until = 0.0

    def pan(self, delta_pixels, points, viewport):
        """Keep visible spacecraft inside a margin; allow gradual recovery from outside."""
        size = np.maximum(np.asarray(viewport, dtype=float), 1.0)
        scale = min(size) / self.span
        basis = self.basis()
        projected = np.asarray(points, dtype=float) @ basis.T
        current = self.focus @ basis.T
        half = np.maximum(size / 2 - np.minimum(32.0, size * 0.1), 1.0) / scale
        lower = projected.max(axis=0) - half
        upper = projected.min(axis=0) + half
        # If zoom makes fitting both impossible, allow movement between their
        # visibility boundaries, so the player can recover either spacecraft.
        low, high = np.minimum(lower, upper), np.maximum(lower, upper)
        # Existing off-screen states must not snap the camera or trap recovery.
        low, high = np.minimum(low, current), np.maximum(high, current)
        requested = current + np.asarray(delta_pixels) * [-1, 1] / scale
        bounded = np.clip(requested, low, high)
        self.focus += (bounded - current) @ basis

    def keep_visible(self, points, viewport):
        """Expand uniformly about the current focus when either spacecraft would leave view."""
        size = np.maximum(np.asarray(viewport, dtype=float), 1.0)
        half = np.maximum(size / 2 - np.maximum(32.0, size * 0.1), 1.0)
        projected = (np.asarray(points, dtype=float) - self.focus) @ self.basis().T
        required = float(np.max(np.abs(projected) / half)) * min(size)
        self.span = max(self.span, required, 0.1)

    def fit(self, points, viewport=(1, 1)):
        points = np.asarray(points, dtype=float)
        self.focus = np.mean(points, axis=0)
        basis = self.basis()
        projected = (points - self.focus) @ basis.T
        self.focus += ((projected.min(axis=0) + projected.max(axis=0)) / 2) @ basis
        size = np.maximum(np.asarray(viewport, dtype=float), 1.0)
        # Fit the projected bounds with 20% of the viewport reserved as padding.
        self.span = max(float(np.max(np.ptp(projected, axis=0) / size)) * min(size) / 0.8, 0.1)
        self.initialized = True


class Dashboard3DMixin:
    def _sandbox_camera(self):
        if not hasattr(self, "_camera_3d"):
            self._camera_3d = SandboxCamera()
        return self._camera_3d

    def camera_effective_speed(self, speed):
        return self._sandbox_camera().effective_speed(speed) if self.sandbox_3d_enabled else speed

    def handle_camera_event(self, event, blocked=False):
        if not self.sandbox_3d_enabled:
            return False
        pg = self.pygame
        camera = self._sandbox_camera()
        if blocked or pygame_focus_lost(pg, event):
            camera.cancel()
            return False
        if event.type == pg.MOUSEBUTTONUP and event.button == camera.drag_button:
            camera.drag_button = 0
            return True
        if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
            for label, rect in getattr(self, "_camera_buttons", {}).items():
                if rect.collidepoint(event.pos):
                    if label in ("3D", "2D"):
                        camera.enabled = not camera.enabled
                        camera.cancel()
                    elif label == "Recenter":
                        camera.focus = self._camera_target().copy()
                    elif label == "Fit both":
                        camera.fit(self._camera_objects(), self._camera_plot.size)
                    else:
                        camera.yaw, camera.elevation = {
                            "RI": (0.0, 0.0),
                            "RC": (np.pi / 2, 0.0),
                            "IC": (0.0, -np.pi / 2),
                        }[label]
                    return True
        plot = getattr(self, "_camera_plot", None)
        if not camera.enabled or plot is None:
            return False
        if event.type == pg.MOUSEBUTTONDOWN and event.button in (1, 2) and plot.collidepoint(event.pos):
            camera.drag_button = event.button
            camera.pan_drag = event.button == 2 or bool(pg.key.get_mods() & pg.KMOD_SHIFT)
            return True
        if event.type == pg.MOUSEMOTION and camera.drag_button:
            dx, dy = event.rel
            if camera.pan_drag:
                camera.pan((dx, dy), self._camera_objects(), plot.size)
            else:
                camera.yaw += dx * 0.008
                camera.elevation = float(np.clip(camera.elevation + dy * 0.008, -np.pi / 2, np.pi / 2))
            return True
        if event.type == pg.MOUSEWHEEL and plot.collidepoint(pg.mouse.get_pos()):
            camera.span = float(np.clip(camera.span * np.exp(-np.clip(event.y, -20, 20) * 0.12), 1e-5, 1e8))
            camera.zoom_until = perf_counter() + 0.25
            return True
        return False

    def _camera_target(self):
        rows = self._frame_cache.get("target_rel")
        return np.asarray(rows[-1, :3]) if rows is not None and len(rows) else np.zeros(3)

    def _camera_objects(self):
        rows = self._frame_cache.get("rel")
        chaser = np.asarray(rows[-1, :3]) if rows is not None and len(rows) else np.zeros(3)
        return np.stack((self._camera_target(), chaser))

    def _draw_camera_buttons(self, rect):
        camera = self._sandbox_camera()
        labels = ["2D", "Recenter", "Fit both", "RI", "RC", "IC"] if camera.enabled else ["3D"]
        self._camera_buttons = {}
        x = rect.right
        for label in reversed(labels):
            width = 88 if len(label) > 3 else 44
            button = self.pygame.Rect(x - width, rect.y + 5, width - 5, 27)
            self.pygame.draw.rect(self.screen, (38, 65, 82), button, border_radius=5)
            self._text(label, (button.x + 7, button.y + 4), self.small_font, (220, 239, 248))
            self._camera_buttons[label] = button
            x -= width

    def _draw_sandbox_3d(self, rect):
        pg = self.pygame
        camera = self._sandbox_camera()
        sign = self._axis_display_sign(1)
        if camera.in_track_sign != sign:
            camera.in_track_sign = sign
            camera.initialized = False
        pg.draw.rect(self.screen, (20, 27, 36), rect, border_radius=10)
        self._text("RIC · 3D / orthographic", (rect.x + 14, rect.y + 10), self.font, (230, 235, 242))
        plot = pg.Rect(rect.x + 18, rect.y + 48, rect.width - 36, rect.height - 86)
        self._camera_plot = plot
        if not camera.initialized:
            framing = [self._camera_objects()]
            for region in self.forbidden_regions:
                _, lines = self._region_mesh(region)
                framing.extend(line + self._camera_target() for line in lines)
            target = self._camera_target()
            for gate in self.inspection_gates:
                _, lines = constraint_mesh.box(
                    gate.center_ric_km - gate.half_width_ric_km, gate.center_ric_km + gate.half_width_ric_km
                )
                framing.extend(line + target for line in lines)
            for key in ("nmt", "tutorial_path_sample"):
                rows = self._frame_cache.get(key)
                if rows is not None and len(rows):
                    framing.append(rows[:, :3])
            if self.goal_radius_km is not None and self.goal_radius_km > 0:
                framing.extend(
                    target + self.goal_relative_ric_km + sign * np.eye(3) * self.goal_radius_km for sign in (-1, 1)
                )
            if self.goal_range_km is not None and self.goal_range_km > 0:
                radius = self.goal_range_km + (self.goal_range_tolerance_km or 0.0)
                framing.extend(target + sign * np.eye(3) * radius for sign in (-1, 1))
            for constraint in self.sun_angle_constraints:
                radius = constraint.max_range_km or constraint.beam_radius_km or 4.0
                framing.extend(target + sign * np.eye(3) * radius for sign in (-1, 1))
            camera.fit(np.vstack(framing), plot.size)
        # Enforce on every rendered frame: mouse actions, presets, resize, and
        # spacecraft motion all share the same visibility guarantee.
        camera.keep_visible(self._camera_objects(), plot.size)
        scale = min(plot.size) / camera.span
        basis = camera.basis()

        def project(points):
            xy = (np.asarray(points)[..., :3] - camera.focus) @ basis.T * scale
            return np.clip(xy * [1, -1] + plot.center, -1e8, 1e8).astype(int)

        old_clip = self.screen.get_clip()
        self.screen.set_clip(plot)
        pg.draw.rect(self.screen, (8, 11, 16), plot)
        target = self._camera_target()
        # Cover the visible RI plane, including after panning. Near edge-on,
        # bound the grid rather than inverting an ill-conditioned projection.
        plane = basis[:, :2]
        target_xy = (target - camera.focus) @ basis.T
        half = np.asarray(plot.size) / (2 * scale)
        corners = np.array([[-half[0], -half[1]], [-half[0], half[1]], [half[0], -half[1]], [half[0], half[1]]])
        if abs(np.linalg.det(plane)) > 0.05:
            bounds = (corners - target_xy) @ np.linalg.inv(plane).T
            low, high = bounds.min(axis=0), bounds.max(axis=0)
        else:
            extent = float(np.linalg.norm(half)) * 4
            low = camera.focus[:2] - target[:2] - extent
            high = camera.focus[:2] - target[:2] + extent
        raw_step = max(camera.span / 8, float(np.max(high - low)) / 70)
        magnitude = 10 ** np.floor(np.log10(raw_step))
        step = next(value * magnitude for value in (1, 2, 5, 10) if value * magnitude >= raw_step)
        for axis in (0, 1):
            other = 1 - axis
            for offset in np.arange(np.floor(low[other] / step), np.ceil(high[other] / step) + 1) * step:
                a, b = target.copy(), target.copy()
                a[axis] += low[axis] - step
                b[axis] += high[axis] + step
                a[other] += offset
                b[other] += offset
                pg.draw.line(self.screen, (29, 39, 49), *project([a, b]))
        origin = project(target).astype(float)
        for axis, (label, color) in enumerate(
            zip(("R", "I", "C"), ((245, 125, 115), (120, 220, 150), (120, 175, 255)))
        ):
            direction = basis[:, axis] * [1, -1]
            length = np.linalg.norm(direction)
            if length < 1e-6:
                continue
            direction /= length
            reach = np.linalg.norm(np.asarray(plot.size)) + np.linalg.norm(origin - plot.center)
            a, b = origin - direction * reach, origin + direction * reach
            clipped = plot.clipline(tuple(a), tuple(b))
            if clipped:
                pg.draw.line(self.screen, color, *clipped, 1)
            label_segment = plot.inflate(-60, -48).clipline(tuple(a), tuple(b))
            if label_segment:
                self._text("+" + label, label_segment[1], self.small_font, color)
        self._draw_3d_constraints(plot, project, target)
        for key, color in (
            ("rel", (75, 155, 190)),
            ("ghost", (120, 195, 225)),
            ("target_rel", (125, 145, 100)),
            ("target_ghost", (180, 170, 125)),
        ):
            if key == "target_ghost" and not self.show_target_coast_prediction:
                continue
            rows = self._frame_cache.get(key)
            if rows is not None and len(rows) > 1:
                points = project(rows[:: max(1, len(rows) // 600)])
                pg.draw.lines(self.screen, color, False, points, 1)
        objects = self._camera_objects()
        chaser = objects[1]
        foot = chaser.copy()
        foot[2] = target[2]
        pg.draw.line(self.screen, (75, 95, 115), *project([chaser, foot]), 1)
        for point, label, color in zip(objects, ("Target", "Chaser"), (TARGET_MARKER_COLOR, CHASER_MARKER_COLOR)):
            pos = project(point)
            self._draw_satellite_marker(
                tuple(pos),
                role=label.lower(),
                scale_x=scale,
                scale_y=scale,
                fallback_radius_px=6 if label == "Target" else 7,
            )
            self._text(label, (pos[0] + 10, pos[1] - 16), self.small_font, color)
        bar_px = 80
        bar_start = (plot.left + 24, plot.bottom - 22)
        pg.draw.line(self.screen, (180, 200, 220), bar_start, (bar_start[0] + bar_px, bar_start[1]), 2)
        self._text(f"{bar_px / scale:.3g} km", (bar_start[0], bar_start[1] - 22), self.small_font, (180, 200, 220))
        self.screen.set_clip(old_clip)
        self._text(
            "Drag: orbit   Shift-drag / middle: pan   Wheel: zoom   ·   Equal axis scale",
            (rect.x + 14, rect.bottom - 27),
            self.small_font,
            (165, 184, 201),
        )

    def _draw_3d_constraints(self, plot, project, target):
        pg = self.pygame
        for region in self.forbidden_regions:
            faces, lines = self._region_mesh(region)
            self._draw_constraint_mesh(plot, project, target, faces, lines, (185, 75, 85))
        for index, gate in enumerate(self.inspection_gates, 1):
            faces, lines = constraint_mesh.box(
                gate.center_ric_km - gate.half_width_ric_km, gate.center_ric_km + gate.half_width_ric_km
            )
            self._draw_constraint_mesh(plot, project, target, faces, lines, (100, 215, 155))
            self._text(f"Gate {index}", tuple(project(target + gate.center_ric_km)), self.small_font, (130, 240, 180))
        for constraint in self.sun_angle_constraints:
            center = constraint.allowed_center_at_ric(
                target_state_eci=self._current_target_state_eci_for_sun(), time_s=self._current_time_s()
            )
            center = center / np.linalg.norm(center)
            helper = np.eye(3)[np.argmin(np.abs(center))]
            first = np.cross(center, helper)
            first /= np.linalg.norm(first)
            second = np.cross(center, first)
            phi = np.linspace(0, 2 * np.pi, 49)[:-1]
            angle = np.deg2rad(constraint.allowed_half_angle_deg)
            directions = center * np.cos(angle) + (
                np.cos(phi)[:, None] * first + np.sin(phi)[:, None] * second
            ) * np.sin(angle)
            near = constraint.min_range_km or 0.0
            far = constraint.max_range_km or constraint.beam_radius_km or 4.0
            lower, upper = near * directions, far * directions
            faces, lines = constraint_mesh.extrusion(lower, upper)
            self._draw_constraint_mesh(plot, project, target, faces, lines, (235, 195, 95))
            pg.draw.line(self.screen, (250, 220, 130), *project([target, target + center * far]), 2)
        for key in ("nmt", "tutorial_path_sample"):
            rows = self._frame_cache.get(key)
            if rows is not None and len(rows) > 1:
                pg.draw.lines(self.screen, (100, 220, 150), False, project(rows), 2)
        for rows in self._frame_cache.get("nmt_bounds", ()):
            if len(rows) > 1:
                pg.draw.lines(self.screen, (80, 175, 125), True, project(rows), 1)
        if self.goal_radius_km is not None and self.goal_radius_km > 0:
            _, lines = constraint_mesh.sphere(target + self.goal_relative_ric_km, self.goal_radius_km)
            for line in lines:
                pg.draw.lines(self.screen, (100, 220, 130), False, project(line), 1)
        angles = np.linspace(0, 2 * np.pi, 96)
        for radius, color in ((self.keepout_radius_km, (235, 125, 65)), (self.goal_range_km, (100, 220, 130))):
            if radius is None or radius <= 0:
                continue
            for first, second in ((0, 1), (0, 2), (1, 2)):
                rows = np.tile(target, (len(angles), 1))
                rows[:, first] += radius * np.cos(angles)
                rows[:, second] += radius * np.sin(angles)
                pg.draw.lines(self.screen, color, True, project(rows), 1)
        if self.forbidden_regions:
            self._text(
                "Red: forbidden   Orange: keepout   Green: goals/gates   Gold: allowed Sun angle",
                (plot.left + 12, plot.top + 10),
                self.small_font,
                (220, 185, 180),
            )

    def _region_mesh(self, region):
        cache = getattr(self, "_constraint_mesh_cache", None)
        if cache is None:
            cache = self._constraint_mesh_cache = {}
        key = id(region)
        if key not in cache:
            cache[key] = (region, constraint_mesh.region_mesh(region))
        return cache[key][1]

    def _draw_constraint_mesh(self, plot, project, target, faces, lines, color):
        layer = self.pygame.Surface(plot.size, self.pygame.SRCALPHA)
        for face in faces:
            self.pygame.draw.polygon(layer, (*color, 22), project(face + target) - plot.topleft)
        self.screen.blit(layer, plot.topleft)
        for line in lines:
            self.pygame.draw.lines(self.screen, color, False, project(line + target), 1)
