# ruff: noqa: F401,F403,F405,I001
from .dashboard_common import *
from .geometry import *
from .prediction import *
from .camera import *

class DashboardOverlayMixin:
    def _draw_pause_overlay(self) -> None:
        pygame = self.pygame
        width, height = self.screen.get_size()
        rect_w = min(900, max(width - 128, 560))
        rect_h = min(430, max(height - 152, 340))
        rect = pygame.Rect(width // 2 - rect_w // 2, height // 2 - rect_h // 2, rect_w, rect_h)
        pygame.draw.rect(self.screen, (13, 22, 32), rect, border_radius=10)
        pygame.draw.rect(self.screen, (116, 194, 238), rect, width=2, border_radius=10)

        self._text("PAUSED - RIC MOTION CARD", (rect.x + 28, rect.y + 22), self.large_font, (238, 244, 250))
        self._text(
            "Circular-chief HCW intuition for the local RIC frame.",
            (rect.x + 30, rect.y + 58),
            self.small_font,
            (170, 205, 230),
        )

        column_gap = 28
        left_w = max((rect.width - 76 - column_gap) // 2, 240)
        left_x = rect.x + 30
        right_x = left_x + left_w + column_gap
        section_y = rect.y + 94

        self._draw_pause_text_section(
            "HCW Equations",
            self._pause_overlay_equation_lines(),
            x=left_x,
            y=section_y,
            width_px=left_w,
        )
        takeaway_y = section_y + 138
        self._draw_pause_text_section(
            "Useful Intuition",
            self._pause_overlay_takeaway_lines(),
            x=left_x,
            y=takeaway_y,
            width_px=left_w,
        )

        diagram = pygame.Rect(right_x, section_y, rect.right - right_x - 30, rect.height - 154)
        self._draw_pause_ric_diagram(diagram)
        footer = "Space Resume   Up/Down Speed   R Reset   Esc Level Select"
        self._text(
            self._fit_text_px(footer, self.font, rect.width - 60, preserve_spaces=True),
            (rect.x + 30, rect.bottom - 44),
            self.font,
            (220, 160, 160),
        )

    @staticmethod
    def _pause_overlay_equation_lines() -> tuple[str, ...]:
        return (
            "R'' = 3 n² R + 2 n I' + a_R",
            "I'' = -2 n R' + a_I",
            "C'' = -n² C + a_C",
        )

    @staticmethod
    def _pause_overlay_takeaway_lines() -> tuple[str, ...]:
        return (
            "R is radial: up/down from the target orbit.",
            "I is in-track: ahead/behind along the orbit.",
            "C is cross-track: out of the orbital plane.",
            "Burn, then coast. The curve is the lesson.",
        )

    def _draw_pause_text_section(
        self,
        title: str,
        lines: tuple[str, ...],
        *,
        x: int,
        y: int,
        width_px: int,
    ) -> None:
        self._text(str(title).upper(), (x, y), self.small_font, (150, 205, 245))
        cursor_y = y + 24
        for line in lines:
            for wrapped in self._wrap_text_px(str(line), self.small_font, width_px):
                self._text(wrapped, (x, cursor_y), self.small_font, (222, 230, 238))
                cursor_y += 19

    def _draw_pause_ric_diagram(self, rect: Any) -> None:
        pygame = self.pygame
        pygame.draw.rect(self.screen, (8, 13, 19), rect, border_radius=8)
        pygame.draw.rect(self.screen, (72, 92, 116), rect, width=1, border_radius=8)
        title = self._fit_text_px("Target-Centered RIC Frame", self.small_font, rect.width - 28)
        self._text(title, (rect.x + 14, rect.y + 14), self.small_font, (170, 184, 204))

        origin = (rect.centerx, rect.centery + 28)
        axis_len = max(min(rect.width, rect.height) // 4, 58)
        self._draw_pause_axis(origin, (0, -axis_len), "R", (150, 235, 170), "radial")
        self._draw_pause_axis(origin, (axis_len + 16, 0), "I", (245, 210, 110), "in-track")
        self._draw_pause_axis(origin, (-axis_len // 2, axis_len // 2), "C", (150, 205, 245), "cross-track")

        self._draw_satellite_marker(
            origin,
            role="target",
            scale_x=2000.0,
            scale_y=2000.0,
            fallback_radius_px=7,
            force_icon=True,
        )
        chaser = (origin[0] - axis_len // 3, origin[1] - axis_len // 3)
        self._draw_satellite_marker(
            chaser,
            role="chaser",
            scale_x=2000.0,
            scale_y=2000.0,
            fallback_radius_px=7,
            force_icon=True,
        )
        pygame.draw.arc(
            self.screen,
            (100, 120, 145),
            pygame.Rect(origin[0] - axis_len, origin[1] - axis_len // 2, axis_len * 2, axis_len),
            0.2,
            2.9,
            width=1,
        )
        note = self._fit_text_px(
            "Target-relative, not straight-line flight.",
            self.small_font,
            rect.width - 28,
        )
        self._text(note, (rect.x + 14, rect.bottom - 34), self.small_font, (170, 184, 204))

    def _draw_pause_axis(
        self,
        origin: tuple[int, int],
        delta: tuple[int, int],
        label: str,
        color: tuple[int, int, int],
        description: str,
    ) -> None:
        pygame = self.pygame
        end = (int(origin[0] + delta[0]), int(origin[1] + delta[1]))
        pygame.draw.line(self.screen, color, origin, end, width=3)
        angle = float(np.arctan2(float(end[1] - origin[1]), float(end[0] - origin[0])))
        head = 10.0
        spread = 0.45
        pygame.draw.polygon(
            self.screen,
            color,
            [
                end,
                (
                    int(round(float(end[0]) - head * np.cos(angle - spread))),
                    int(round(float(end[1]) - head * np.sin(angle - spread))),
                ),
                (
                    int(round(float(end[0]) - head * np.cos(angle + spread))),
                    int(round(float(end[1]) - head * np.sin(angle + spread))),
                ),
            ],
        )
        label_pos = (end[0] + 8, end[1] - 8)
        self._text(str(label), label_pos, self.font, color)
        self._text(str(description), (label_pos[0] + 24, label_pos[1] + 2), self.small_font, (205, 216, 230))

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
        start = (int(origin[0]), int(origin[1]))
        end = (int(origin[0] + vec[0] * scale), int(origin[1] - vec[1] * scale))
        angle = float(np.arctan2(float(end[1] - start[1]), float(end[0] - start[0])))
        pygame.draw.line(self.screen, color, origin, end, width=2)
        head = float(WEB_VECTOR_ARROW_HEAD_PX)
        spread = float(WEB_VECTOR_ARROW_HEAD_ANGLE_RAD)
        pygame.draw.polygon(
            self.screen,
            color,
            [
                end,
                (
                    int(round(float(end[0]) - head * np.cos(angle - spread))),
                    int(round(float(end[1]) - head * np.sin(angle - spread))),
                ),
                (
                    int(round(float(end[0]) - head * np.cos(angle + spread))),
                    int(round(float(end[1]) - head * np.sin(angle + spread))),
                ),
            ],
        )
        if label:
            self._text(label, (end[0] + 6, end[1] - 8), self.small_font, color)

    @staticmethod
    def _web_velocity_vector_px(rel_state: np.ndarray, *, x_axis: int, y_axis: int) -> np.ndarray:
        state = np.array(rel_state, dtype=float).reshape(-1)
        if state.size < 6:
            return np.zeros(2, dtype=float)
        return np.array(
            [
                float(state[int(x_axis) + 3]) * WEB_VECTOR_VREL_SCALE_PX_PER_KM_S,
                float(state[int(y_axis) + 3]) * WEB_VECTOR_VREL_SCALE_PX_PER_KM_S,
            ],
            dtype=float,
        )

    @staticmethod
    def _web_thrust_vector_px(
        thrust_ric: np.ndarray,
        *,
        x_axis: int,
        y_axis: int,
        threshold: float = 0.0,
    ) -> np.ndarray:
        thrust = np.array(thrust_ric, dtype=float).reshape(-1)
        if thrust.size < 3:
            return np.zeros(2, dtype=float)
        thrust = thrust[:3]
        norm = float(np.linalg.norm(thrust))
        if not np.isfinite(norm) or norm <= max(float(threshold), 0.0):
            return np.zeros(2, dtype=float)
        unit = thrust / norm
        return (
            np.array([float(unit[int(x_axis)]), float(unit[int(y_axis)])], dtype=float)
            * WEB_VECTOR_THRUST_SCALE_PX
        )

    def _burn_marker_rows(self, *, rel: np.ndarray, thrust: np.ndarray) -> np.ndarray:
        thrust_arr = np.array(thrust, dtype=float).reshape(-1, 3)
        if thrust_arr.size == 0:
            return np.empty((0, 6), dtype=float)
        rel_arr = np.array(rel, dtype=float).reshape(-1, 6)
        count = min(rel_arr.shape[0], thrust_arr.shape[0])
        if count <= 0:
            return np.empty((0, 6), dtype=float)
        rel_arr = rel_arr[-count:]
        thrust_arr = thrust_arr[-count:]
        active = np.linalg.norm(thrust_arr, axis=1) > float(self.burn_marker_threshold_km_s2)
        idxs = np.where(active)[0]
        if idxs.size == 0:
            return np.empty((0, 6), dtype=float)
        stride = max(1, int(np.ceil(idxs.size / 80)))
        return rel_arr[idxs[::stride]]

    def _draw_burn_markers(self, *, rel: np.ndarray, to_px: Any, marker_rows: np.ndarray | None = None) -> None:
        if marker_rows is None and not self.thrust_ric_hist:
            return
        pygame = self.pygame
        markers = (
            self._burn_marker_rows(
                rel=rel,
                thrust=_dashboard_history_array(
                    self,
                    "_thrust_ric_array",
                    self.thrust_ric_hist[-rel.shape[0] :],
                    width=3,
                ),
            )
            if marker_rows is None
            else np.array(marker_rows, dtype=float).reshape(-1, 6)
        )
        if markers.size == 0:
            return
        for row in markers:
            pygame.draw.circle(self.screen, (255, 145, 60), to_px(row), 3)

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
        draw_line = pygame.draw.line
        screen = self.screen
        dash = float(dash_px)
        stride = float(dash_px + gap_px)
        for start, end in zip(points[:-1], points[1:]):
            x0 = float(start[0])
            y0 = float(start[1])
            dx = float(end[0]) - x0
            dy = float(end[1]) - y0
            length = hypot(dx, dy)
            if length <= 0.0:
                continue
            ux = dx / length
            uy = dy / length
            pos = 0.0
            while pos < length:
                stop = min(pos + dash, length)
                draw_line(
                    screen,
                    color,
                    (int(x0 + ux * pos), int(y0 + uy * pos)),
                    (int(x0 + ux * stop), int(y0 + uy * stop)),
                    width=width,
                )
                pos += stride

    def _draw_dashed_ellipse(
        self,
        rect: Any,
        *,
        color: tuple[int, int, int],
        dash_count: int = 72,
        width: int = 1,
    ) -> None:
        count = max(int(dash_count), 12)
        cx = float(rect.centerx)
        cy = float(rect.centery)
        rx = float(rect.width) / 2.0
        ry = float(rect.height) / 2.0
        if rx <= 0.0 or ry <= 0.0:
            return
        points: list[tuple[int, int]] = []
        for idx in range(count + 1):
            theta = 2.0 * np.pi * float(idx) / float(count)
            points.append((int(round(cx + rx * np.cos(theta))), int(round(cy + ry * np.sin(theta)))))
        self._draw_polyline_dashed(points, color=color, dash_px=10, gap_px=7, width=width)
