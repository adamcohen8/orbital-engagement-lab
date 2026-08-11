# ruff: noqa: F401,F403,F405,I001
from .dashboard_common import *
from .geometry import *
from .prediction import *
from .camera import *

class DashboardLayoutMixin:
    def draw(
        self,
        *,
        command_status: str = "",
        coach_hint: str = "",
        mission_state: str = "active",
        level_title: str = "",
        mission_metrics: tuple[str, ...] = (),
        objective_checklist: tuple[str, ...] = (),
        speed_multiple: float = 1.0,
        selected_speed_multiple: float | None = None,
        recording_status: str = "",
        briefing_lines: tuple[str, ...] = (),
        debrief_lines: tuple[str, ...] = (),
        debrief_available: bool = True,
        render_motion: bool = False,
        pause_overlay: bool = False,
    ) -> None:
        pygame = self.pygame
        if self.closed:
            return
        width, height = self.screen.get_size()
        self._render_motion_enabled = bool(render_motion)
        self._render_wall_time_s = perf_counter()
        self._render_speed_multiple = float(max(speed_multiple, 0.0))
        if self._operator_projection_transition_active():
            self._frame_cache_dirty = True
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
            level_title=level_title,
            mission_metrics=mission_metrics,
            objective_checklist=objective_checklist,
        )
        left_panel, right_panel = self._plot_panel_specs()
        self._draw_panel(left, left_panel[0], x_axis=left_panel[1], y_axis=left_panel[2])
        self._draw_panel(right, right_panel[0], x_axis=right_panel[1], y_axis=right_panel[2])
        self._draw_hud(
            hud,
            command_status=command_status,
            coach_hint=coach_hint,
            speed_multiple=speed_multiple,
            selected_speed_multiple=selected_speed_multiple,
            recording_status=recording_status,
        )
        if briefing_lines:
            self._draw_briefing(briefing_lines)
        elif bool(pause_overlay) and mission_state == "active":
            self._draw_pause_overlay()
        if mission_state in {"passed", "failed"}:
            self._draw_mission_banner(
                mission_state,
                debrief_lines=debrief_lines,
                debrief_available=debrief_available,
            )
        pygame.display.flip()

    def draw_ric_primer(self, *, stage_index: int, elapsed_s: float, recording_status: str = "") -> None:
        pygame = self.pygame
        if self.closed:
            return
        width, height = self.screen.get_size()
        stage = _ric_primer_stage(stage_index)
        self.screen.fill((12, 16, 22))
        top = pygame.Rect(36, 18, width - 72, 104)
        panel_y = top.bottom + 24
        left = pygame.Rect(36, panel_y, max((width - 108) // 2, 200), max(height - panel_y - 130, 250))
        right = pygame.Rect(left.right + 36, left.y, left.width, left.height)
        hud = pygame.Rect(36, height - 112, width - 72, 86)
        self._draw_ric_primer_top_bar(
            top,
            stage=stage,
            stage_index=stage_index,
        )
        if stage["eci_plane"] == "RI":
            self._draw_ric_primer_eci_panel(left, stage=stage, elapsed_s=elapsed_s)
            self._draw_ric_primer_local_panel(right, stage=stage, elapsed_s=elapsed_s, plane="RC")
        else:
            self._draw_ric_primer_local_panel(left, stage=stage, elapsed_s=elapsed_s, plane="RI")
            self._draw_ric_primer_eci_panel(right, stage=stage, elapsed_s=elapsed_s)
        self._draw_ric_primer_hud(
            hud,
            stage=stage,
            stage_index=stage_index,
            recording_status=recording_status,
        )
        pygame.display.flip()

    def _draw_ric_primer_top_bar(self, rect: Any, *, stage: dict[str, Any], stage_index: int) -> None:
        pygame = self.pygame
        pygame.draw.rect(self.screen, (18, 24, 32), rect, border_radius=10)
        pygame.draw.rect(self.screen, (82, 96, 118), rect, width=2, border_radius=10)
        right_width = min(420, max(280, rect.width // 3))
        left_width = max(rect.width - right_width - 34, 160)
        title = self._fit_text_px("RIC FRAME PRIMER", self.large_font, left_width)
        self._text(title, (rect.x + 16, rect.y + 9), self.large_font, (230, 235, 242))
        meta = f"INFO Step {stage_index + 1}/3    INFO {stage['title']}"
        self._text(
            self._fit_text_px(meta, self.small_font, left_width),
            (rect.x + 16, rect.y + 54),
            self.small_font,
            (222, 230, 238),
        )
        x = rect.right - right_width - 16
        self._text("OBJECTIVES", (x, rect.y + 12), self.small_font, (170, 184, 204))
        self._text(
            self._fit_text_px(str(stage["title"]), self.small_font, right_width),
            (x, rect.y + 35),
            self.small_font,
            (150, 235, 170),
        )
        self._text(
            self._fit_text_px(str(stage["text"]), self.small_font, right_width),
            (x, rect.y + 58),
            self.small_font,
            (150, 205, 245),
        )

    def _draw_ric_primer_local_panel(self, rect: Any, *, stage: dict[str, Any], elapsed_s: float, plane: str) -> None:
        pygame = self.pygame
        title = "RI Plane" if plane == "RI" else "RC Plane"
        pygame.draw.rect(self.screen, (20, 27, 36), rect, border_radius=10)
        pygame.draw.rect(self.screen, (80, 92, 110), rect, width=1, border_radius=10)
        subtitle_width = min(240, max(rect.width // 2 - 20, 80))
        subtitle = self._fit_text_px(str(stage["local_subtitle"]), self.small_font, subtitle_width)
        title_width = max(rect.width - subtitle_width - 48, 80)
        self._text(
            self._fit_text_px(title, self.font, title_width),
            (rect.x + 14, rect.y + 10),
            self.font,
            (230, 235, 242),
        )
        self._text(subtitle, (rect.right - subtitle_width - 14, rect.y + 13), self.small_font, (170, 184, 204))
        plot = rect.inflate(-48, -72)
        plot.y += 28
        pygame.draw.rect(self.screen, (8, 11, 16), plot)
        pygame.draw.rect(self.screen, (72, 84, 102), plot, width=1)
        scale = min(plot.width, plot.height) / 2.5
        self._draw_grid(plot, scale=scale)
        x_axis = 1 if plane == "RI" else 2
        y_axis = 0
        x_display_sign = self._axis_display_sign(x_axis)
        y_display_sign = self._axis_display_sign(y_axis)
        self._draw_ric_primer_axes(
            plot,
            x_axis=x_axis,
            y_axis=y_axis,
            active_axis=int(stage["axis_index"]),
            x_display_sign=x_display_sign,
            y_display_sign=y_display_sign,
        )
        point = np.zeros(3, dtype=float)
        point[int(stage["axis_index"])] = float(stage["amplitude_km"]) * float(np.sin(float(elapsed_s) * 1.05))

        def to_px(pos: np.ndarray) -> tuple[int, int]:
            x = float(pos[x_axis]) * x_display_sign
            y = float(pos[y_axis]) * y_display_sign
            return plot.centerx + int(round(x * scale)), plot.centery - int(round(y * scale))

        target = to_px(np.zeros(3, dtype=float))
        chaser = to_px(point)
        self._draw_satellite_marker(target, role="target", scale_x=scale, scale_y=scale, fallback_radius_px=6, force_icon=True)
        self._draw_satellite_marker(
            chaser,
            role="chaser",
            scale_x=scale,
            scale_y=scale,
            fallback_radius_px=7,
            force_icon=True,
        )
        pygame.draw.ellipse(
            self.screen,
            (92, 42, 50),
            pygame.Rect(
                target[0] - int(0.25 * scale),
                target[1] - int(0.25 * scale),
                int(0.5 * scale),
                int(0.5 * scale),
            ),
            width=1,
        )
        self._text("Target", (target[0] + 10, target[1] - 10), self.small_font, TARGET_MARKER_COLOR)
        self._text("Chaser", (chaser[0] + 10, chaser[1] + 14), self.small_font, CHASER_MARKER_COLOR)

    def _draw_ric_primer_axes(
        self,
        plot: Any,
        *,
        x_axis: int,
        y_axis: int,
        active_axis: int,
        x_display_sign: float = 1.0,
        y_display_sign: float = 1.0,
    ) -> None:
        pygame = self.pygame
        colors = {
            0: (150, 235, 170),
            1: (245, 205, 92),
            2: (96, 190, 245),
        }
        x_color = colors.get(x_axis, (90, 104, 124)) if active_axis == x_axis else (90, 104, 124)
        y_color = colors.get(y_axis, (90, 104, 124)) if active_axis == y_axis else (90, 104, 124)
        pygame.draw.line(self.screen, x_color, (plot.left + 36, plot.centery), (plot.right - 36, plot.centery), width=2)
        pygame.draw.line(self.screen, y_color, (plot.centerx, plot.bottom - 32), (plot.centerx, plot.top + 32), width=2)
        x_positive_at_right = float(x_display_sign) >= 0.0
        y_positive_at_top = float(y_display_sign) >= 0.0
        x_arrow_origin = (plot.right - 54, plot.centery) if x_positive_at_right else (plot.left + 54, plot.centery)
        y_arrow_origin = (plot.centerx, plot.top + 56) if y_positive_at_top else (plot.centerx, plot.bottom - 56)
        self._draw_vector(x_arrow_origin, np.array([24.0 * float(x_display_sign), 0.0]), color=x_color, scale=1.0)
        self._draw_vector(y_arrow_origin, np.array([0.0, 24.0 * float(y_display_sign)]), color=y_color, scale=1.0)
        x_plus = self._signed_axis_label_for_plot(x_axis, 1)
        x_minus = self._signed_axis_label_for_plot(x_axis, -1)
        y_plus = self._signed_axis_label_for_plot(y_axis, 1)
        y_minus = self._signed_axis_label_for_plot(y_axis, -1)
        x_left_label = x_minus if x_positive_at_right else x_plus
        x_right_label = x_plus if x_positive_at_right else x_minus
        y_top_label = y_plus if y_positive_at_top else y_minus
        y_bottom_label = y_minus if y_positive_at_top else y_plus
        self._text(x_left_label, (plot.left + 42, plot.centery - 18), self.small_font, x_color)
        self._text(x_right_label, (plot.right - 74, plot.centery - 18), self.small_font, x_color)
        self._text(y_top_label, (plot.centerx + 10, plot.top + 34), self.small_font, y_color)
        self._text(y_bottom_label, (plot.centerx + 10, plot.bottom - 58), self.small_font, y_color)

    def _draw_ric_primer_eci_panel(self, rect: Any, *, stage: dict[str, Any], elapsed_s: float) -> None:
        pygame = self.pygame
        pygame.draw.rect(self.screen, (20, 27, 36), rect, border_radius=10)
        pygame.draw.rect(self.screen, (80, 92, 110), rect, width=1, border_radius=10)
        subtitle_width = min(250, max(rect.width // 2 - 20, 80))
        subtitle = self._fit_text_px(str(stage["eci_subtitle"]), self.small_font, subtitle_width)
        title_width = max(rect.width - subtitle_width - 48, 80)
        self._text(
            self._fit_text_px("ECI Orbit", self.font, title_width),
            (rect.x + 14, rect.y + 10),
            self.font,
            (230, 235, 242),
        )
        self._text(subtitle, (rect.right - subtitle_width - 14, rect.y + 13), self.small_font, (170, 184, 204))
        plot = rect.inflate(-48, -72)
        plot.y += 28
        pygame.draw.rect(self.screen, (8, 11, 16), plot)
        pygame.draw.rect(self.screen, (72, 84, 102), plot, width=1)
        if stage["id"] == "cross_track":
            self._draw_ric_primer_cross_track_side_view(plot, elapsed_s=elapsed_s)
        else:
            self._draw_ric_primer_eci_circles(plot, stage=stage, elapsed_s=elapsed_s)

    def _draw_ric_primer_eci_circles(self, plot: Any, *, stage: dict[str, Any], elapsed_s: float) -> None:
        pygame = self.pygame
        center = plot.center
        radius = min(plot.width, plot.height) * 0.34
        phase = float(np.sin(float(elapsed_s) * 1.05))
        target_theta = -0.35
        radius_offset = 0.14 * phase if stage["id"] == "radial" else 0.0
        phase_offset = 0.34 * phase if stage["id"] == "in_track" else 0.0
        pygame.draw.circle(self.screen, (42, 92, 130), center, int(round(radius)), width=2)
        chaser_radius = radius * (1.0 + radius_offset)
        pygame.draw.circle(
            self.screen,
            (170, 68, 74) if stage["id"] == "radial" else (110, 60, 66),
            center,
            int(round(chaser_radius)),
            width=2,
        )
        self._draw_primer_earth(center, int(round(radius * 0.16)))
        target = _point_on_circle(center, radius, target_theta)
        chaser = _point_on_circle(center, chaser_radius, target_theta + phase_offset)
        self._draw_satellite_marker(target, role="target", scale_x=radius, scale_y=radius, fallback_radius_px=7, force_icon=True)
        self._draw_satellite_marker(
            chaser,
            role="chaser",
            scale_x=radius,
            scale_y=radius,
            fallback_radius_px=7,
            force_icon=True,
        )
        self._text("Target", (target[0] - 62, target[1] + 18), self.small_font, (230, 235, 242))
        self._text("Chaser", (chaser[0] + 10, chaser[1] - 16), self.small_font, (230, 235, 242))
        if stage["id"] == "radial":
            self._text(
                f"Chaser radius {1.0 + radius_offset:.2f}x target orbit",
                (plot.x + 18, plot.bottom - 26),
                self.small_font,
                (170, 184, 204),
            )

    def _draw_ric_primer_cross_track_side_view(self, plot: Any, *, elapsed_s: float) -> None:
        pygame = self.pygame
        center = plot.center
        half_span = min(plot.width, plot.height) * 0.36
        inclination_deg = 10.0 * float(np.sin(float(elapsed_s) * 1.05))
        target_line = _line_segment(center, half_span, -0.5)
        chaser_line = _line_segment(center, half_span, -0.5 - inclination_deg / 22.0)
        self._draw_primer_earth(center, max(24, int(round(min(plot.width, plot.height) * 0.07))))
        pygame.draw.line(self.screen, (96, 174, 224), target_line[0], target_line[1], width=3)
        pygame.draw.line(self.screen, CHASER_MARKER_COLOR, chaser_line[0], chaser_line[1], width=3)
        target = _point_along_line(target_line, 0.86)
        chaser = _point_along_line(chaser_line, 0.86)
        self._draw_satellite_marker(target, role="target", scale_x=half_span, scale_y=half_span, fallback_radius_px=7, force_icon=True)
        self._draw_satellite_marker(
            chaser,
            role="chaser",
            scale_x=half_span,
            scale_y=half_span,
            fallback_radius_px=7,
            force_icon=True,
        )
        self._text("Target", (target[0] - 62, target[1] + 18), self.small_font, (230, 235, 242))
        self._text("Chaser", (chaser[0] + 10, chaser[1] - 16), self.small_font, (230, 235, 242))
        self._text(
            f"Side view: chaser inclination {inclination_deg:.1f} deg",
            (plot.x + 18, plot.bottom - 26),
            self.small_font,
            (170, 184, 204),
        )

    def _draw_primer_earth(self, center: tuple[int, int], radius: int) -> None:
        pygame = self.pygame
        radius = max(int(radius), 18)
        pygame.draw.circle(self.screen, (34, 92, 142), center, radius)
        pygame.draw.circle(self.screen, (220, 240, 255), center, radius, width=1)

    def _draw_ric_primer_hud(
        self,
        rect: Any,
        *,
        stage: dict[str, Any],
        stage_index: int,
        recording_status: str = "",
    ) -> None:
        pygame = self.pygame
        pygame.draw.rect(self.screen, (18, 24, 32), rect, border_radius=10)
        pygame.draw.rect(self.screen, (82, 96, 118), rect, width=1, border_radius=10)
        status_text = str(recording_status or "G Clip").strip()
        pill_width = min(max(self._text_width(self.small_font, status_text) + 22, 86), max(rect.width // 2, 86))
        pill = pygame.Rect(rect.right - pill_width - 12, rect.y + 10, pill_width, 24)
        pygame.draw.rect(self.screen, (34, 48, 62), pill, border_radius=4)
        pygame.draw.rect(self.screen, (104, 130, 156), pill, width=1, border_radius=4)
        self._text(
            self._fit_text_px(status_text, self.small_font, pill.width - 16),
            (pill.x + 8, pill.y + 5),
            self.small_font,
            (220, 234, 246),
        )
        text_width = max(pill.x - rect.x - 32, 120)
        self._text(
            self._fit_text_px(f"Step {stage_index + 1}/3  {stage['title']}", self.font, text_width),
            (rect.x + 16, rect.y + 12),
            self.font,
            (235, 240, 245),
        )
        self._text(
            self._fit_text_px(str(stage["hint"]), self.small_font, rect.width - 32),
            (rect.x + 16, rect.y + 38),
            self.small_font,
            (245, 210, 110),
        )
        self._text(
            self._fit_text_px("Space Next   R Replay   Esc Quit", self.small_font, rect.width - 32),
            (rect.x + 16, rect.y + 60),
            self.small_font,
            (195, 205, 220),
        )

    def _draw_panel(self, rect: Any, title: str, x_axis: int, y_axis: int) -> None:
        pygame = self.pygame
        pygame.draw.rect(self.screen, (20, 27, 36), rect, border_radius=10)
        pygame.draw.rect(self.screen, (80, 92, 110), rect, width=1, border_radius=10)
        self._text(title, (rect.x + 14, rect.y + 10), self.font, (230, 235, 242))
        plot = rect.inflate(-48, -72)
        plot.y += int(getattr(self, "_plot_panel_title_gap_px", 28))
        pygame.draw.rect(self.screen, (8, 11, 16), plot)
        pygame.draw.rect(self.screen, (72, 84, 102), plot, width=1)
        if self._plot_view_mode_for_axes(x_axis=x_axis, y_axis=y_axis) == "eci":
            self._draw_eci_orbit_plane_panel(plot)
            return
        if not self.rel_hist:
            return
        rel = self._frame_cache.get("rel")
        target_rel = self._frame_cache.get("target_rel")
        target_reference_rel = self._frame_cache.get("target_reference_rel")
        if rel is None or target_rel is None:
            rel = _dashboard_history_array(self, "_rel_array", self.rel_hist, width=6)
            target_rel = _dashboard_history_array(
                self,
                "_target_rel_array",
                self.target_rel_hist[-rel.shape[0] :],
                width=6,
            )
        if target_reference_rel is None:
            target_reference_rel = _dashboard_history_array(
                self,
                "_target_reference_rel_array",
                getattr(self, "target_reference_rel_hist", [])[-rel.shape[0] :],
                width=6,
            )
        target_current = target_rel[-1, :3] if target_rel.size else np.zeros(3, dtype=float)
        target_state_for_sun = self._current_target_state_eci_for_sun()
        current_time_s = self._current_time_s()
        target_reference_current = (
            target_reference_rel[-1, :3] if target_reference_rel.size else np.zeros(3, dtype=float)
        )
        ghost = self._frame_cache.get("ghost")
        if ghost is None:
            ghost = self._coast_prediction()
        tutorial_path = np.asarray(self._frame_cache.get("tutorial_path_sample", np.empty((0, 6))), dtype=float)
        if tutorial_path.ndim != 2 or tutorial_path.shape[1] < 3:
            tutorial_path = np.empty((0, 6), dtype=float)
        target_ghost = self._frame_cache.get("target_ghost")
        if target_ghost is None:
            target_ghost = self._target_coast_prediction(target_rel)
        prediction_scales_camera = self._prediction_scales_current_camera()
        scale_ghost = ghost if prediction_scales_camera else np.empty((0, 6), dtype=float)
        scale_target_ghost = target_ghost if prediction_scales_camera else np.empty((0, 6), dtype=float)
        goal = np.array(self.goal_relative_ric_km, dtype=float).reshape(-1)
        nmt = self._frame_cache.get("nmt")
        if nmt is None:
            nmt = self._nmt_points()
        nmt_bounds = self._frame_cache.get("nmt_bounds")
        if nmt_bounds is None:
            nmt_bounds = self._nmt_boundary_points()
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
        pts_for_scale.extend(
            self._camera_rule_scale_points(
                rel=rel,
                target_rel=target_rel,
                ghost=scale_ghost,
                target_ghost=scale_target_ghost,
                x_axis=x_axis,
                y_axis=y_axis,
                camera_center=camera_center,
            )
        )
        if tutorial_path.size:
            pts_for_scale.append((tutorial_path[:, :3] - camera_center.reshape(1, 3))[:, [x_axis, y_axis]])
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
        pixel_cache = self._frame_cache.setdefault("pixel_polyline_cache", {})
        x_display_sign = self._axis_display_sign(x_axis)
        y_display_sign = self._axis_display_sign(y_axis)
        transform_key = (
            int(x_axis),
            int(y_axis),
            round(float(x_display_sign), 1),
            round(float(y_display_sign), 1),
            int(plot.centerx),
            int(plot.centery),
            round(float(scale_x), 12),
            round(float(scale_y), 12),
            tuple(round(float(value), 12) for value in camera_center.reshape(3)),
        )

        def to_px(point: np.ndarray) -> tuple[int, int]:
            shifted = np.array(point, dtype=float).reshape(-1)[:3] - camera_center
            x = float(shifted[x_axis]) * x_display_sign
            y = float(shifted[y_axis]) * y_display_sign
            px = plot.centerx + int(round(x * scale_x))
            py = plot.centery - int(round(y * scale_y))
            return px, py

        def rows_to_px(rows: np.ndarray, cache_key: str = "") -> list[tuple[int, int]]:
            arr = np.asarray(rows, dtype=float).reshape(-1, rows.shape[-1] if hasattr(rows, "shape") else 3)
            if arr.shape[1] < 3:
                return []
            key = None
            if cache_key:
                key = (cache_key, transform_key, id(rows), arr.shape)
                cached_points = pixel_cache.get(key)
                if cached_points is not None:
                    return cached_points
            shifted = arr[:, :3] - camera_center.reshape(1, 3)
            px = np.rint(plot.centerx + shifted[:, int(x_axis)] * x_display_sign * scale_x).astype(int)
            py = np.rint(plot.centery - shifted[:, int(y_axis)] * y_display_sign * scale_y).astype(int)
            points = list(zip(px.tolist(), py.tolist()))
            if key is not None:
                pixel_cache[key] = points
            return points

        def circle_rect(center: tuple[int, int], radius_km: float) -> Any:
            rx = max(1, int(round(float(radius_km) * scale_x)))
            ry = max(1, int(round(float(radius_km) * scale_y)))
            return self.pygame.Rect(center[0] - rx, center[1] - ry, rx * 2, ry * 2)

        plot_transforms = self._frame_cache.setdefault("plot_transforms", {})
        plot_transforms[(int(x_axis), int(y_axis))] = {
            "plot": (int(plot.x), int(plot.y), int(plot.width), int(plot.height)),
            "camera_center": tuple(float(value) for value in camera_center.reshape(3)),
            "scale_x": float(scale_x),
            "scale_y": float(scale_y),
            "x_display_sign": float(x_display_sign),
            "y_display_sign": float(y_display_sign),
        }

        previous_clip = self.screen.get_clip()
        self.screen.set_clip(plot)
        self._draw_cislunar_moon_background(
            plot,
            x_axis=x_axis,
            y_axis=y_axis,
            to_px=to_px,
            scale_x=scale_x,
            scale_y=scale_y,
        )
        self._draw_grid(plot, scale_x=scale_x, scale_y=scale_y)
        self._draw_sun_angle_constraints(
            plot,
            x_axis=x_axis,
            y_axis=y_axis,
            to_px=to_px,
            offset=target_current,
            target_state_eci=target_state_for_sun,
            time_s=current_time_s,
        )
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
            if (
                self.max_target_reference_range_km is not None
                and float(self.max_target_reference_range_km) > 0.0
            ):
                self._draw_dashed_ellipse(
                    circle_rect(to_px(target_reference_current), float(self.max_target_reference_range_km)),
                    color=(76, 214, 120),
                    dash_count=72,
                    width=2,
                )
            if self.hard_speed_limit_radius_km is not None and float(self.hard_speed_limit_radius_km) > 0.0:
                center = to_px(target_current)
                pygame.draw.ellipse(
                    self.screen,
                    (232, 194, 74),
                    circle_rect(center, float(self.hard_speed_limit_radius_km)),
                    width=2,
                )
        for boundary_index, boundary in enumerate(nmt_bounds):
            if boundary.size:
                boundary_pts = rows_to_px(boundary, cache_key=f"nmt_boundary:{boundary_index}")
                if len(boundary_pts) >= 2:
                    pygame.draw.lines(self.screen, (78, 178, 112), True, boundary_pts, width=2)
        if _should_draw_nominal_nmt(nmt, nmt_bounds):
            nmt_pts = rows_to_px(nmt, cache_key="nmt")
            self._draw_polyline_dashed(nmt_pts, color=(120, 236, 154), dash_px=18, gap_px=14, width=2)
        if tutorial_path.size:
            tutorial_pts = rows_to_px(tutorial_path, cache_key="tutorial_path")
            self._draw_polyline_dashed(tutorial_pts, color=(92, 240, 132), dash_px=16, gap_px=8, width=3)
        pygame.draw.line(self.screen, (90, 104, 124), (plot.left, plot.centery), (plot.right, plot.centery), width=1)
        pygame.draw.line(self.screen, (90, 104, 124), (plot.centerx, plot.top), (plot.centerx, plot.bottom), width=1)
        pygame.draw.circle(self.screen, (80, 92, 112), (plot.centerx, plot.centery), 4)
        target_px = to_px(target_current)
        target_ghost_sample = self._frame_cache.get("target_ghost_sample", target_ghost)
        if bool(getattr(self, "show_target_coast_prediction", False)) and target_ghost_sample.size:
            target_ghost_pts = rows_to_px(target_ghost_sample, cache_key="target_ghost")
            self._draw_polyline_dashed(target_ghost_pts, color=COAST_PREDICTION_COLOR, dash_px=10, gap_px=7, width=2)
        target_trail_rows = self._frame_cache.get("target_trail", target_rel[-self.max_history :])
        if target_trail_rows.size and len(target_trail_rows) >= 2:
            target_trail = rows_to_px(target_trail_rows, cache_key="target_trail")
            pygame.draw.lines(self.screen, TARGET_TRAIL_COLOR, False, target_trail, width=2)
        self._draw_satellite_marker(target_px, role="target", scale_x=scale_x, scale_y=scale_y, fallback_radius_px=6)

        trail_rows = self._frame_cache.get("rel_trail", rel[-self.max_history :])
        trail = rows_to_px(trail_rows, cache_key="rel_trail")
        ghost_sample = self._frame_cache.get("ghost_sample", ghost)
        if ghost_sample.size:
            ghost_pts = rows_to_px(ghost_sample, cache_key="ghost")
            ghost_color = (
                LIVE_BURN_COLOR
                if bool(self._frame_cache.get("ghost_active_burn", False))
                else COAST_PREDICTION_COLOR
            )
            self._draw_polyline_dashed(ghost_pts, color=ghost_color, dash_px=8, gap_px=8, width=2)
        if len(trail) >= 2:
            pygame.draw.lines(self.screen, CHASER_TRAIL_COLOR, False, trail, width=2)
        self._draw_burn_markers(rel=rel, to_px=to_px, marker_rows=self._frame_cache.get("burn_marker_rel"))
        chaser = trail[-1]
        self._draw_satellite_marker(
            chaser,
            role="chaser",
            scale_x=scale_x,
            scale_y=scale_y,
            fallback_radius_px=7,
            plane=_plane_key_for_axes(x_axis=x_axis, y_axis=y_axis),
            rotation_deg=self._aerodynamic_sprite_rotation_deg(x_axis=x_axis, y_axis=y_axis),
        )
        if bool(getattr(self, "aerodynamic_control_enabled", False)) and (x_axis, y_axis) == (2, 0):
            phi = np.deg2rad(float(getattr(self, "aerodynamic_lift_bank_angle_deg", 0.0)))
            lift_rc = np.array([-np.sin(phi), np.cos(phi)], dtype=float)
            lift_px = 46.0 * lift_rc * np.array([x_display_sign, -y_display_sign], dtype=float)
            self._draw_vector(chaser, lift_px, color=(92, 236, 255), scale=1.0)

        if rel.shape[1] >= 6:
            v_px = self._web_velocity_vector_px(rel[-1], x_axis=x_axis, y_axis=y_axis)
            v_px = v_px * np.array([x_display_sign, y_display_sign], dtype=float)
            self._draw_vector(to_px(rel[-1]), v_px, color=VELOCITY_VECTOR_COLOR, scale=1.0)
        live_accel_ric = np.array(
            getattr(self, "live_prediction_accel_ric_km_s2", np.zeros(3, dtype=float)), dtype=float
        ).reshape(3)
        threshold = float(getattr(self, "burn_marker_threshold_km_s2", 0.0))
        if np.linalg.norm(live_accel_ric) > threshold:
            thrust_vec = self._web_thrust_vector_px(
                live_accel_ric,
                x_axis=x_axis,
                y_axis=y_axis,
                threshold=threshold,
            )
            thrust_vec = thrust_vec * np.array([x_display_sign, y_display_sign], dtype=float)
            controlled_id = str(getattr(self, "controlled_object_id", "") or "")
            target_id = str(getattr(self, "target_object_id", "target") or "target")
            origin = target_px if controlled_id == target_id else chaser
            self._draw_vector(origin, thrust_vec, color=LIVE_BURN_COLOR, scale=1.0)
        self.screen.set_clip(previous_clip)

        self._draw_signed_axis_labels(
            plot,
            x_axis=int(x_axis),
            y_axis=int(y_axis),
            x_display_sign=x_display_sign,
            y_display_sign=y_display_sign,
        )

    def _draw_eci_orbit_plane_panel(self, plot: Any) -> None:
        target_eci = _dashboard_history_array(
            self,
            "_target_eci_array",
            getattr(self, "target_eci_hist", ()),
            width=6,
        )
        chaser_eci = _dashboard_history_array(
            self,
            "_chaser_eci_array",
            getattr(self, "chaser_eci_hist", ()),
            width=6,
        )
        if target_eci.size == 0 or chaser_eci.size == 0:
            return
        frame_key = _relative_frame_key(getattr(self, "relative_frame", "ric"))
        center_state = cr3bp_moon_state_km_s() if frame_key == "moon_ric" else np.zeros(6, dtype=float)
        center_radius_km = MOON_RADIUS_KM if frame_key == "moon_ric" else EARTH_RADIUS_KM
        center_fill = (88, 92, 102) if frame_key == "moon_ric" else (30, 74, 126)
        center_outline = (174, 180, 192) if frame_key == "moon_ric" else (72, 138, 202)
        draw_history_trails = frame_key != "moon_ric"
        panel_label = "Moon-centered target orbit" if frame_key == "moon_ric" else "Earth-centered ECI plane"
        target_centered = target_eci - center_state
        chaser_centered = chaser_eci - center_state
        if frame_key == "moon_ric":
            target_xy = _project_moon_rotating_yz_to_plane(target_centered[:, :3])
            chaser_xy = _project_moon_rotating_yz_to_plane(chaser_centered[:, :3])
        else:
            basis = self._eci_target_plane_basis(target_centered[-1])
            if basis is None:
                self._text("Orbit-plane view unavailable", (plot.x + 16, plot.y + 16), self.small_font, (232, 194, 74))
                return
            i_hat, r_hat, _c_hat = basis
            target_xy = _project_eci_positions_to_plane(target_centered[:, :3], x_hat=i_hat, y_hat=r_hat)
            chaser_xy = _project_eci_positions_to_plane(chaser_centered[:, :3], x_hat=i_hat, y_hat=r_hat)
        if target_xy.size == 0 or chaser_xy.size == 0:
            return
        target_orbit_xy = np.empty((0, 2), dtype=float)
        if frame_key == "moon_ric":
            target_orbit = self._cr3bp_target_orbit_prediction(allow_build=False)
            if target_orbit.size:
                target_orbit_centered = np.array(target_orbit, dtype=float).reshape(-1, 6) - center_state
                target_orbit_xy = _project_moon_rotating_yz_to_plane(target_orbit_centered[:, :3])
        target_radius_km = float(np.linalg.norm(target_centered[-1, :3]))
        half_span_km = max(target_radius_km * 1.18, center_radius_km * 1.25, 1.0)
        target_scale_xy = target_xy if draw_history_trails else target_xy[-1:].copy()
        chaser_scale_xy = chaser_xy if draw_history_trails else chaser_xy[-1:].copy()
        all_xy_rows = [target_scale_xy, chaser_scale_xy, np.array([[0.0, 0.0]], dtype=float)]
        if target_orbit_xy.size and frame_key != "moon_ric":
            all_xy_rows.append(target_orbit_xy)
        all_xy = np.vstack(all_xy_rows)
        max_abs = float(np.nanmax(np.abs(all_xy))) if all_xy.size else 0.0
        if np.isfinite(max_abs):
            half_span_km = max(half_span_km, max_abs * 1.18)
        scale = min(plot.width, plot.height) * 0.5 / half_span_km

        def to_px_xy(xy: np.ndarray) -> tuple[int, int]:
            vec = np.array(xy, dtype=float).reshape(2)
            return (
                plot.centerx + int(round(float(vec[0]) * scale)),
                plot.centery - int(round(float(vec[1]) * scale)),
            )

        def rows_to_px(rows: np.ndarray) -> list[tuple[int, int]]:
            arr = np.array(rows, dtype=float).reshape(-1, 2)
            px = np.rint(plot.centerx + arr[:, 0] * scale).astype(int)
            py = np.rint(plot.centery - arr[:, 1] * scale).astype(int)
            return list(zip(px.tolist(), py.tolist()))

        pygame = self.pygame
        previous_clip = self.screen.get_clip()
        self.screen.set_clip(plot)
        self._draw_grid(plot, scale_x=scale, scale_y=scale)
        if target_orbit_xy.size:
            target_orbit_trail = rows_to_px(_sample_rows(target_orbit_xy, MAX_TARGET_ORBIT_DRAW_POINTS))
            if len(target_orbit_trail) >= 2:
                pygame.draw.lines(self.screen, (66, 76, 92), False, target_orbit_trail, width=1)
        else:
            orbit_radius_px = max(2, int(round(target_radius_km * scale)))
            orbit_rect = pygame.Rect(0, 0, orbit_radius_px * 2, orbit_radius_px * 2)
            orbit_rect.center = plot.center
            pygame.draw.ellipse(self.screen, (66, 76, 92), orbit_rect, width=1)
        pygame.draw.line(self.screen, (90, 104, 124), (plot.left, plot.centery), (plot.right, plot.centery), width=1)
        pygame.draw.line(self.screen, (90, 104, 124), (plot.centerx, plot.top), (plot.centerx, plot.bottom), width=1)
        center_radius_px = max(2, int(round(center_radius_km * scale)))
        pygame.draw.circle(self.screen, center_fill, plot.center, center_radius_px)
        pygame.draw.circle(self.screen, center_outline, plot.center, center_radius_px, width=2)

        if draw_history_trails:
            target_trail = rows_to_px(_sample_rows(target_xy, MAX_TRAIL_DRAW_POINTS))
            chaser_trail = rows_to_px(_sample_rows(chaser_xy, MAX_TRAIL_DRAW_POINTS))
            if len(target_trail) >= 2:
                pygame.draw.lines(self.screen, TARGET_TRAIL_COLOR, False, target_trail, width=2)
            if len(chaser_trail) >= 2:
                pygame.draw.lines(self.screen, CHASER_TRAIL_COLOR, False, chaser_trail, width=2)
        target_px = to_px_xy(target_xy[-1])
        chaser_px = to_px_xy(chaser_xy[-1])
        self._draw_satellite_marker(
            target_px,
            role="target",
            scale_x=scale,
            scale_y=scale,
            fallback_radius_px=6,
        )
        self._draw_satellite_marker(
            chaser_px,
            role="chaser",
            scale_x=scale,
            scale_y=scale,
            fallback_radius_px=7,
        )
        self.screen.set_clip(previous_clip)
        x_label = "T" if frame_key == "moon_ric" else "I"
        y_label = "N" if frame_key == "moon_ric" else "R"
        self._text(x_label, (plot.right - 34, plot.centery + 8), self.small_font, (170, 180, 195))
        self._text(y_label, (plot.centerx + 8, plot.top + 8), self.small_font, (170, 180, 195))
        self._text(panel_label, (plot.x + 14, plot.bottom - 24), self.small_font, (150, 160, 176))

    @staticmethod
    def _eci_target_plane_basis(target_state_eci: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        state = np.array(target_state_eci, dtype=float).reshape(-1)
        if state.size < 6:
            return None
        r_vec = state[:3]
        v_vec = state[3:6]
        r_norm = float(np.linalg.norm(r_vec))
        h_vec = np.cross(r_vec, v_vec)
        h_norm = float(np.linalg.norm(h_vec))
        if r_norm <= 0.0 or h_norm <= 0.0 or not np.isfinite(r_norm) or not np.isfinite(h_norm):
            return None
        r_hat = r_vec / r_norm
        c_hat = h_vec / h_norm
        i_hat = np.cross(c_hat, r_hat)
        i_norm = float(np.linalg.norm(i_hat))
        if i_norm <= 0.0 or not np.isfinite(i_norm):
            return None
        i_hat = i_hat / i_norm
        return i_hat.astype(float), r_hat.astype(float), c_hat.astype(float)

    def _plot_panel_specs(self) -> tuple[tuple[str, int, int], tuple[str, int, int]]:
        if _relative_frame_key(getattr(self, "relative_frame", "ric")) == "cislunar_l1":
            return (
                ("Cislunar XY: Tangential Vs Earth-Moon", 1, 0),
                ("Cislunar YZ: Tangential Vs Out-of-Plane", 1, 2),
            )
        return (
            (self._panel_title_for_axes("RI Plane: In-Track Vs Radial", x_axis=1, y_axis=0), 1, 0),
            (self._panel_title_for_axes("RC Plane: Cross-Track Vs Radial", x_axis=2, y_axis=0), 2, 0),
        )

    def _panel_title_for_axes(self, default_title: str, *, x_axis: int, y_axis: int) -> str:
        if self._plot_view_mode_for_axes(x_axis=x_axis, y_axis=y_axis) != "eci":
            return default_title
        plane = _plane_key_for_axes(x_axis=x_axis, y_axis=y_axis) or "RIC"
        if _relative_frame_key(getattr(self, "relative_frame", "ric")) == "moon_ric":
            return f"Moon View ({plane} Swap): Tangential Vs Normal"
        return f"ECI View ({plane} Swap): In-Track Vs Radial"

    def _plot_view_mode_for_axes(self, *, x_axis: int, y_axis: int) -> str:
        if _relative_frame_key(getattr(self, "relative_frame", "ric")) == "cislunar_l1":
            return "ric"
        plane = _plane_key_for_axes(x_axis=x_axis, y_axis=y_axis)
        modes = getattr(self, "plot_view_modes", {}) or {}
        mode = str(modes.get(plane, "ric") or "ric").strip().lower()
        return "eci" if mode == "eci" else "ric"

    def toggle_eci_plot(self, plane: str) -> str:
        key = str(plane or "").strip().upper()
        if key not in {"RI", "RC"}:
            return "ric"
        modes = dict(getattr(self, "plot_view_modes", {}) or {})
        next_mode = "ric" if str(modes.get(key, "ric")).strip().lower() == "eci" else "eci"
        modes[key] = next_mode
        self.plot_view_modes = modes
        if hasattr(self, "_frame_cache_dirty"):
            self._frame_cache_dirty = True
        return next_mode

    def _axis_label_for_plot(self, axis: int) -> str:
        return f"{self._axis_symbol_for_plot(axis)} km"

    def _axis_symbol_for_plot(self, axis: int) -> str:
        if _relative_frame_key(getattr(self, "relative_frame", "ric")) == "cislunar_l1":
            labels = {
                0: "EM",
                1: "T",
                2: "N",
            }
        else:
            labels = {0: "R", 1: "I", 2: "C"}
        return labels.get(int(axis), "")

    def _signed_axis_label_for_plot(self, axis: int, sign: int) -> str:
        symbol = self._axis_symbol_for_plot(axis)
        if not symbol:
            return ""
        prefix = "+" if int(sign) >= 0 else "-"
        return f"{prefix}{symbol}"

    def _draw_signed_axis_labels(
        self,
        plot: Any,
        *,
        x_axis: int,
        y_axis: int,
        x_display_sign: float,
        y_display_sign: float,
    ) -> None:
        color = (170, 180, 195)
        x_plus = self._signed_axis_label_for_plot(x_axis, 1)
        x_minus = self._signed_axis_label_for_plot(x_axis, -1)
        y_plus = self._signed_axis_label_for_plot(y_axis, 1)
        y_minus = self._signed_axis_label_for_plot(y_axis, -1)

        x_plus_right = float(x_display_sign) >= 0.0
        x_left_label = x_minus if x_plus_right else x_plus
        x_right_label = x_plus if x_plus_right else x_minus
        self._text(x_left_label, (plot.left + 10, plot.centery + 8), self.small_font, color)
        right_x = plot.right - self._text_width(self.small_font, x_right_label) - 10
        self._text(x_right_label, (right_x, plot.centery + 8), self.small_font, color)

        y_plus_top = float(y_display_sign) >= 0.0
        y_top_label = y_plus if y_plus_top else y_minus
        y_bottom_label = y_minus if y_plus_top else y_plus
        self._text(y_top_label, (plot.centerx + 8, plot.top + 8), self.small_font, color)
        self._text(y_bottom_label, (plot.centerx + 8, plot.bottom - 24), self.small_font, color)

    def _axis_display_sign(self, axis: int) -> float:
        if _relative_frame_key(getattr(self, "relative_frame", "ric")) == "cislunar_l1":
            return 1.0
        return frame_convention_display_axis_sign(getattr(self, "frame_convention", FrameConvention()), int(axis))

    def _draw_cislunar_moon_background(
        self,
        plot: Any,
        *,
        x_axis: int,
        y_axis: int,
        to_px: Any,
        scale_x: float,
        scale_y: float,
    ) -> None:
        if not _should_draw_cislunar_moon_background(
            relative_frame=getattr(self, "relative_frame", "ric"),
            x_axis=int(x_axis),
            y_axis=int(y_axis),
        ):
            return
        rect_tuple = _scaled_body_rect_tuple(
            center_px=to_px(np.zeros(3, dtype=float)),
            radius_km=MOON_RADIUS_KM,
            scale_x=scale_x,
            scale_y=scale_y,
        )
        pygame = self.pygame
        rect = pygame.Rect(*rect_tuple)
        clipped = rect.clip(plot)
        if clipped.width <= 0 or clipped.height <= 0:
            return
        fill = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        fill.fill((0, 0, 0, 0))
        local_rect = pygame.Rect(0, 0, rect.width, rect.height)
        pygame.draw.ellipse(fill, (150, 158, 166, 42), local_rect)
        pygame.draw.ellipse(fill, (210, 218, 226, 86), local_rect, width=1)
        # A tiny limb and crater pattern gives the disk lunar character while
        # preserving the exact radius set by local_rect.
        if rect.width >= 18 and rect.height >= 18:
            shade = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
            shade.fill((0, 0, 0, 0))
            pygame.draw.ellipse(shade, (54, 60, 68, 38), pygame.Rect(rect.width // 3, 0, rect.width, rect.height))
            fill.blit(shade, (0, 0))
            for frac_x, frac_y, frac_r, alpha in (
                (0.34, 0.42, 0.035, 30),
                (0.58, 0.32, 0.028, 24),
                (0.64, 0.62, 0.045, 28),
                (0.44, 0.68, 0.024, 22),
            ):
                crater = pygame.Rect(0, 0, max(2, int(rect.width * frac_r)), max(2, int(rect.height * frac_r)))
                crater.center = (int(rect.width * frac_x), int(rect.height * frac_y))
                pygame.draw.ellipse(fill, (72, 78, 86, alpha), crater, width=1)
        source = pygame.Rect(clipped.x - rect.x, clipped.y - rect.y, clipped.width, clipped.height)
        self.screen.blit(fill, (clipped.x, clipped.y), source)

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

    def _sun_angle_projection_points(
        self, *, x_axis: int, y_axis: int, offset: np.ndarray
    ) -> list[np.ndarray]:
        constraints = tuple(getattr(self, "sun_angle_constraints", ()) or ())
        if not constraints:
            return []
        pts: list[np.ndarray] = []
        target_state_eci = self._current_target_state_eci_for_sun()
        time_s = self._current_time_s()
        for constraint in constraints:
            polygon = _sun_angle_sector_polygon_ric(
                constraint,
                x_axis=x_axis,
                y_axis=y_axis,
                target_state_eci=target_state_eci,
                time_s=time_s,
            )
            if polygon.size:
                polygon = polygon.copy()
                polygon[:, :3] += np.array(offset, dtype=float).reshape(1, 3)
                pts.append(polygon[:, [x_axis, y_axis]])
        return pts

    def _draw_sun_angle_constraints(
        self,
        plot: Any,
        *,
        x_axis: int,
        y_axis: int,
        to_px: Any,
        offset: np.ndarray,
        target_state_eci: np.ndarray | None = None,
        time_s: float | None = None,
    ) -> None:
        constraints = tuple(getattr(self, "sun_angle_constraints", ()) or ())
        if not constraints:
            return
        pygame = self.pygame
        for constraint in constraints:
            if not _constraint_visible_on_plane(constraint, x_axis=x_axis, y_axis=y_axis):
                continue
            polygon = _sun_angle_sector_polygon_ric(
                constraint,
                x_axis=x_axis,
                y_axis=y_axis,
                target_state_eci=target_state_eci,
                time_s=time_s,
            )
            if not polygon.size:
                continue
            polygon = polygon.copy()
            polygon[:, :3] += np.array(offset, dtype=float).reshape(1, 3)
            points = [to_px(row) for row in polygon]
            if len(points) >= 3:
                self._draw_translucent_polygon(plot, points, color=(235, 188, 74, 46))
                pygame.draw.lines(self.screen, (247, 207, 101), True, points, width=1)
            centerline = _sun_angle_centerline_points_ric(
                constraint,
                x_axis=x_axis,
                y_axis=y_axis,
                target_state_eci=target_state_eci,
                time_s=time_s,
            )
            if centerline.size:
                centerline = centerline.copy()
                centerline[:, :3] += np.array(offset, dtype=float).reshape(1, 3)
                line = [to_px(row) for row in centerline]
                if len(line) >= 2:
                    pygame.draw.line(self.screen, (255, 231, 153), line[0], line[1], width=2)

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
            if region.kind == "sphere":
                polygon = _sphere_projection_polygon_ric(region, x_axis=x_axis, y_axis=y_axis)
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
            if region.kind == "sphere":
                polygon = _sphere_projection_polygon_ric(region, x_axis=x_axis, y_axis=y_axis)
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
