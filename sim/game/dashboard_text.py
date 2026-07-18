# ruff: noqa: F401,F403,F405,I001
from .dashboard_common import *
from .geometry import *
from .prediction import *
from .camera import *

class DashboardTextMixin:
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

    def _wrap_text_px(self, value: str, font: Any, width_px: int) -> list[str]:
        words = str(value or "").split()
        if not words:
            return [""]
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = word if not current else current + " " + word
            if self._text_width(font, candidate) <= width_px:
                current = candidate
                continue
            if current:
                lines.append(current)
            current = self._fit_text_px(word, font, width_px) if self._text_width(font, word) > width_px else word
        if current:
            lines.append(current)
        return lines or [""]

    def _fit_text_px(self, value: str, font: Any, width_px: int, *, preserve_spaces: bool = False) -> str:
        text = str(value or "") if preserve_spaces else " ".join(str(value or "").split())
        if self._text_width(font, text) <= width_px:
            return text
        ellipsis = "..."
        if self._text_width(font, ellipsis) > width_px:
            return ""
        lo = 0
        hi = len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            candidate = text[:mid].rstrip() + ellipsis
            if self._text_width(font, candidate) <= width_px:
                lo = mid
            else:
                hi = mid - 1
        return text[:lo].rstrip() + ellipsis

    @staticmethod
    def _text_width(font: Any, text: str) -> int:
        if hasattr(font, "size"):
            return int(font.size(str(text))[0])
        surf = font.render(str(text), True, (255, 255, 255))
        if hasattr(surf, "get_width"):
            return int(surf.get_width())
        return len(str(text)) * 8

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
