from __future__ import annotations

from typing import Any

GAME_FONT_FALLBACKS: tuple[str, ...] = ("Menlo", "Consolas", "Monaco", "Courier New")


def game_font(pygame: Any, size_px: int) -> Any:
    size = int(max(size_px, 1))
    return pygame.font.SysFont(GAME_FONT_FALLBACKS, size) or pygame.font.Font(None, size)
