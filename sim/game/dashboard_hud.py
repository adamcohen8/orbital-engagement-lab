# ruff: noqa: F401,F403,F405,I001
from .dashboard_common import *
from .geometry import *
from .prediction import *
from .camera import *

class DashboardHUDMixin:
    def _draw_hud(
        self,
        rect: Any,
        *,
        command_status: str,
        coach_hint: str,
        speed_multiple: float,
        selected_speed_multiple: float | None = None,
        recording_status: str = "",
    ) -> None:
        pygame = self.pygame
        pygame.draw.rect(self.screen, (18, 24, 32), rect, border_radius=10)
        pygame.draw.rect(self.screen, (82, 96, 118), rect, width=1, border_radius=10)
        if self.rel_hist:
            rel = self.rel_hist[-1]
            rng = float(np.linalg.norm(rel[:3]))
            spd = float(np.linalg.norm(rel[3:]))
            t = self.t_s[-1] if self.t_s else 0.0
            anomaly = self._true_anomaly_indicator_text()
            suffix = f"   {anomaly}" if anomaly else ""
            self._text(
                f"T={t:7.1f}s   Range={format_distance_km(rng)}   Rel Speed={format_speed_km_s(spd)}{suffix}",
                (rect.x + 16, rect.y + 12),
                self.font,
                (235, 240, 245),
            )
        status_text = str(recording_status or "G Clip").strip()
        active = status_text.upper().startswith("REC")
        text_width = self._text_width(self.small_font, status_text)
        pill_width = min(max(text_width + 22, 86), max(rect.width // 2, 86))
        pill = pygame.Rect(rect.right - pill_width - 12, rect.y + 10, pill_width, 24)
        pygame.draw.rect(self.screen, (96, 22, 32) if active else (34, 48, 62), pill, border_radius=4)
        pygame.draw.rect(self.screen, (248, 84, 100) if active else (104, 130, 156), pill, width=1, border_radius=4)
        self._text(
            self._fit_text_px(status_text, self.small_font, pill.width - 16),
            (pill.x + 8, pill.y + 5),
            self.small_font,
            (255, 220, 224) if active else (220, 234, 246),
        )
        hint_text = coach_hint or ""
        hint_is_alert = hint_text.lower().startswith("wrong key")
        if hint_is_alert:
            alert_rect = pygame.Rect(rect.x + 10, rect.y + 34, min(rect.width - 20, 560), 24)
            pygame.draw.rect(self.screen, (96, 28, 34), alert_rect, border_radius=4)
            pygame.draw.rect(self.screen, (242, 92, 104), alert_rect, width=1, border_radius=4)
        self._text(
            hint_text,
            (rect.x + 16, rect.y + 38),
            self.small_font,
            (255, 190, 198) if hint_is_alert else (245, 210, 110),
        )
        selected_speed = None if selected_speed_multiple is None else float(selected_speed_multiple)
        active_speed = float(speed_multiple)
        if selected_speed is not None and not np.isclose(selected_speed, active_speed):
            speed_label = f"Active {active_speed:.0f}x  Coast {selected_speed:.0f}x"
        else:
            speed_label = f"Speed {active_speed:.0f}x"
        footer = f"{speed_label}  Up/Down  Space Pause  R Reset  Esc Select"
        footer_text = self._fit_text_px(
            footer,
            self.small_font,
            min(rect.width - 24, 690),
            preserve_spaces=True,
        )
        footer_width = self._text_width(self.small_font, footer_text)
        footer_x = rect.right - footer_width - 16
        command_line = command_status.splitlines()[0] if command_status else ""
        command_max_width = max(0, footer_x - (rect.x + 16) - 12)
        if command_line and command_max_width > 48:
            command_text = self._fit_text_px(
                command_line,
                self.small_font,
                command_max_width,
                preserve_spaces=True,
            )
            self._text(command_text, (rect.x + 16, rect.y + 60), self.small_font, (195, 205, 220))
        self._text(
            footer_text,
            (footer_x, rect.y + 58),
            self.small_font,
            (220, 160, 160),
        )

    def _draw_top_bar(
        self,
        rect: Any,
        *,
        mission_state: str,
        level_title: str = "",
        mission_metrics: tuple[str, ...],
        objective_checklist: tuple[str, ...] = (),
    ) -> None:
        pygame = self.pygame
        colors = {
            "active": ((18, 24, 32), (82, 96, 118), (230, 235, 242), self._top_bar_label(mission_state, level_title)),
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

    @staticmethod
    def _top_bar_label(mission_state: str, level_title: str = "") -> str:
        if str(mission_state or "").strip().lower() != "active":
            return ""
        title = str(level_title or "").strip()
        return title.upper() if title else "LEVEL ACTIVE"

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
        rect_w = min(860, max(width - 96, 420))
        rect_h = min(500, max(height - 96, 320))
        rect = pygame.Rect(width // 2 - rect_w // 2, height // 2 - rect_h // 2, rect_w, rect_h)
        pygame.draw.rect(self.screen, (15, 24, 34), rect, border_radius=10)
        pygame.draw.rect(self.screen, (96, 174, 224), rect, width=2, border_radius=10)
        title = self.large_font.render(str(lines[0] if lines else "Mission Brief"), True, (238, 244, 250))
        self.screen.blit(title, (rect.x + 30, rect.y + 24))

        content_rect = pygame.Rect(rect.x + 32, rect.y + 70, rect.width - 76, rect.height - 136)
        wrapped_lines = self._briefing_body_lines(lines[1:], content_rect.width)
        content_height = len(wrapped_lines) * BRIEFING_LINE_HEIGHT_PX
        max_scroll = max(content_height - content_rect.height, 0)
        self.briefing_scroll_px = min(max(int(self.briefing_scroll_px), 0), max_scroll)

        previous_clip = self.screen.get_clip()
        self.screen.set_clip(content_rect)
        y = content_rect.y - self.briefing_scroll_px
        for line in wrapped_lines:
            if y + BRIEFING_LINE_HEIGHT_PX >= content_rect.y and y <= content_rect.bottom:
                self._text(line, (content_rect.x, y), self.font, (206, 218, 232))
            y += BRIEFING_LINE_HEIGHT_PX
        self.screen.set_clip(previous_clip)

        if max_scroll > 0:
            self._draw_briefing_scrollbar(rect, content_rect, max_scroll)

        self._text(
            self._briefing_footer_text(max_scroll > 0),
            (rect.x + 32, rect.bottom - 48),
            self.font,
            (220, 160, 160),
        )

    def _briefing_body_lines(self, lines: tuple[str, ...], width_px: int) -> list[str]:
        cache_key = (tuple(str(line) for line in lines), int(width_px), id(self.font))
        briefing_cache = getattr(self, "_briefing_layout_cache", {})
        if briefing_cache.get("key") == cache_key:
            return list(briefing_cache.get("lines", [""]))
        body: list[str] = []
        for raw in lines:
            body.extend(self._wrap_text_px(str(raw), self.font, width_px))
            body.append("")
        while body and body[-1] == "":
            body.pop()
        wrapped = body or [""]
        self._briefing_layout_cache = {"key": cache_key, "lines": tuple(wrapped)}
        return list(wrapped)

    def _draw_briefing_scrollbar(self, rect: Any, content_rect: Any, max_scroll: int) -> None:
        pygame = self.pygame
        track = pygame.Rect(rect.right - 28, content_rect.y, 5, content_rect.height)
        pygame.draw.rect(self.screen, (42, 58, 76), track, border_radius=3)
        thumb_h = max(int(track.height * content_rect.height / (content_rect.height + max_scroll)), 28)
        thumb_y = track.y + int((track.height - thumb_h) * (self.briefing_scroll_px / max_scroll))
        thumb = pygame.Rect(track.x, thumb_y, track.width, thumb_h)
        pygame.draw.rect(self.screen, (116, 194, 238), thumb, border_radius=3)

    @staticmethod
    def _briefing_footer_text(scrollable: bool) -> str:
        if scrollable:
            return "Scroll To Read. Press Space To Start. Esc Returns To Level Select."
        return "Press Space To Start. Esc Returns To Level Select."

    def _draw_mission_banner(
        self,
        mission_state: str,
        *,
        debrief_lines: tuple[str, ...] = (),
        debrief_available: bool = True,
        ) -> None:
        pygame = self.pygame
        width, height = self.screen.get_size()
        rect = pygame.Rect(width // 2 - 360, height // 2 - 190, 720, 380)
        if mission_state == "passed":
            fill = (24, 86, 48)
            stroke = (108, 232, 142)
            text = "MISSION PASSED"
            base_sub = (
                "D Debrief  R Replay  Esc Quit"
                if debrief_available
                else "R Replay  Esc Quit"
            )
            color = (210, 255, 220)
        else:
            fill = (90, 30, 36)
            stroke = (244, 102, 102)
            text = "MISSION FAILED"
            base_sub = (
                "D Debrief  R Retry  Esc Quit"
                if debrief_available
                else "R Retry  Esc Quit"
            )
            color = (255, 220, 220)
        pygame.draw.rect(self.screen, fill, rect, border_radius=10)
        pygame.draw.rect(self.screen, stroke, rect, width=3, border_radius=10)
        title = self.large_font.render(text, True, color)
        self.screen.blit(title, (rect.centerx - title.get_width() // 2, rect.y + 22))
        content_rect = pygame.Rect(rect.x + 36, rect.y + 70, rect.width - 78, rect.height - 136)
        body_lines = self._mission_banner_body_lines(debrief_lines, content_rect.width)
        content_height = len(body_lines) * MISSION_BANNER_LINE_HEIGHT_PX
        max_scroll = max(content_height - content_rect.height, 0)
        self.mission_banner_scroll_px = min(max(int(getattr(self, "mission_banner_scroll_px", 0)), 0), max_scroll)

        previous_clip = self.screen.get_clip()
        self.screen.set_clip(content_rect)
        y = content_rect.y - self.mission_banner_scroll_px
        for line in body_lines:
            if y + MISSION_BANNER_LINE_HEIGHT_PX >= content_rect.y and y <= content_rect.bottom:
                self._text(str(line), (content_rect.x, y), self.font, color)
            y += MISSION_BANNER_LINE_HEIGHT_PX
        self.screen.set_clip(previous_clip)

        if max_scroll > 0:
            self._draw_mission_banner_scrollbar(rect, content_rect, max_scroll)
        sub = self._fit_text_px(
            self._mission_banner_footer_text(base_sub, max_scroll > 0),
            self.font,
            rect.width - 72,
            preserve_spaces=True,
        )
        subtitle = self.font.render(sub, True, color)
        self.screen.blit(subtitle, (rect.centerx - subtitle.get_width() // 2, rect.bottom - 42))

    def _mission_banner_body_lines(self, lines: tuple[str, ...], width_px: int) -> list[str]:
        cache_key = (tuple(str(line) for line in lines), int(width_px), id(self.font))
        layout_cache = getattr(self, "_mission_banner_layout_cache", {})
        if layout_cache.get("key") == cache_key:
            return list(layout_cache.get("lines", [""]))
        wrapped: list[str] = []
        for raw in lines:
            wrapped.extend(self._wrap_mission_banner_line(str(raw), width_px))
        body = wrapped or [""]
        self._mission_banner_layout_cache = {"key": cache_key, "lines": tuple(body)}
        return list(body)

    def _wrap_mission_banner_line(self, value: str, width_px: int) -> list[str]:
        text = str(value or "")
        if self._text_width(self.font, text) <= width_px:
            return [text]
        label_width_chars = 14
        if len(text) > label_width_chars and text[:label_width_chars].strip():
            label = text[:label_width_chars]
            body = text[label_width_chars:].strip()
            body_width = max(width_px - self._text_width(self.font, label), 80)
            wrapped_body = self._wrap_text_px(body, self.font, body_width)
            if wrapped_body:
                return [label + wrapped_body[0], *(" " * label_width_chars + line for line in wrapped_body[1:])]
        return self._wrap_text_px(text, self.font, width_px)

    def _draw_mission_banner_scrollbar(self, rect: Any, content_rect: Any, max_scroll: int) -> None:
        pygame = self.pygame
        track = pygame.Rect(rect.right - 28, content_rect.y, 5, content_rect.height)
        pygame.draw.rect(self.screen, (112, 50, 56), track, border_radius=3)
        thumb_h = max(int(track.height * content_rect.height / (content_rect.height + max_scroll)), 28)
        thumb_y = track.y + int((track.height - thumb_h) * (self.mission_banner_scroll_px / max_scroll))
        thumb = pygame.Rect(track.x, thumb_y, track.width, thumb_h)
        pygame.draw.rect(self.screen, (255, 172, 172), thumb, border_radius=3)

    @staticmethod
    def _mission_banner_footer_text(base_text: str, scrollable: bool) -> str:
        if scrollable:
            return "Scroll/Page  " + str(base_text)
        return str(base_text)
