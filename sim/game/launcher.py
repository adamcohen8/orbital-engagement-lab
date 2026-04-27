from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

GAME_CONFIG_DIR = Path(__file__).resolve().parent / "configs"


@dataclass(frozen=True)
class GameScenarioOption:
    path: Path
    scenario_id: str
    title: str
    description: str
    learning_goal: str
    player_brief: str
    pass_criteria: tuple[str, ...]
    instructor_notes: tuple[str, ...]
    difficulty: str
    time_budget_s: float | None
    delta_v_budget_m_s: float | None
    goal_speed_km_s: float | None
    target_delta_v_budget_m_s: float | None
    level_number: int


def discover_game_scenarios(config_dir: Path | None = None) -> tuple[GameScenarioOption, ...]:
    root = Path(config_dir) if config_dir is not None else GAME_CONFIG_DIR
    options: list[GameScenarioOption] = []
    for path in sorted(root.glob("game_training_rpo_*.yaml")):
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        options.append(_scenario_option_from_yaml(path, raw))
    return tuple(sorted(options, key=lambda option: (option.level_number, option.scenario_id)))


def choose_game_scenario(config_dir: Path | None = None) -> Path | None:
    options = discover_game_scenarios(config_dir)
    if not options:
        raise RuntimeError(f"No game training configs found in {Path(config_dir) if config_dir else GAME_CONFIG_DIR}.")
    return _run_launcher(options)


def _scenario_option_from_yaml(path: Path, raw: dict[str, Any]) -> GameScenarioOption:
    metadata = dict(raw.get("metadata", {}) or {})
    game = dict(metadata.get("game", {}) or {})
    training = dict(game.get("training", {}) or {})
    scenario_id = str(training.get("scenario_id", raw.get("scenario_name", path.stem)) or path.stem)
    level_number = _level_number_from_scenario_id(scenario_id)
    return GameScenarioOption(
        path=path,
        scenario_id=scenario_id,
        title=_title_from_scenario_id(scenario_id, level_number=level_number),
        description=str(raw.get("scenario_description", "") or ""),
        learning_goal=str(training.get("learning_goal", "") or ""),
        player_brief=str(training.get("player_brief", "") or ""),
        pass_criteria=_as_str_tuple(training.get("pass_criteria")),
        instructor_notes=_as_str_tuple(training.get("instructor_notes")),
        difficulty=str(game.get("difficulty", "") or ""),
        time_budget_s=_optional_float(training.get("max_time_s")),
        delta_v_budget_m_s=_optional_float(training.get("max_delta_v_m_s")),
        goal_speed_km_s=_optional_float(training.get("max_goal_speed_km_s")),
        target_delta_v_budget_m_s=_optional_float(dict(game.get("defensive_target", {}) or {}).get("max_delta_v_m_s")),
        level_number=level_number,
    )


def _level_number_from_scenario_id(scenario_id: str) -> int:
    parts = str(scenario_id).split("_")
    for part in parts:
        if part.isdigit():
            return int(part)
    return 999


def _title_from_scenario_id(scenario_id: str, *, level_number: int) -> str:
    parts = str(scenario_id).split("_")
    if len(parts) >= 3 and parts[0] == "rpo" and parts[1].isdigit():
        name = " ".join(parts[2:]).title()
        return f"Level {level_number} - {name}"
    return str(scenario_id).replace("_", " ").title()


def _run_launcher(options: tuple[GameScenarioOption, ...]) -> Path | None:
    try:
        import pygame
    except ImportError as exc:  # pragma: no cover - exercised only without optional dependency.
        raise RuntimeError("Game launcher requires `pygame`. Install with `pip install .[game]`.") from exc

    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((1040, 680), pygame.RESIZABLE)
    pygame.display.set_caption("Orbital Engagement Lab - Level Select")
    pygame.event.set_grab(False)
    pygame.mouse.set_visible(True)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Menlo", 18) or pygame.font.Font(None, 18)
    small_font = pygame.font.SysFont("Menlo", 14) or pygame.font.Font(None, 14)
    title_font = pygame.font.SysFont("Menlo", 30) or pygame.font.Font(None, 30)
    selected = 0

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return None
                    if event.key in {pygame.K_DOWN, pygame.K_s}:
                        selected = min(selected + 1, len(options) - 1)
                    elif event.key in {pygame.K_UP, pygame.K_w}:
                        selected = max(selected - 1, 0)
                    elif event.key in {pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE}:
                        return options[selected].path
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    idx = _option_index_at_pos(pygame.mouse.get_pos(), count=len(options))
                    if idx is not None:
                        if idx == selected:
                            return options[selected].path
                        selected = idx
                if event.type == pygame.MOUSEMOTION:
                    idx = _option_index_at_pos(event.pos, count=len(options))
                    if idx is not None:
                        selected = idx

            _draw_launcher(pygame, screen, options=options, selected=selected, font=font, small_font=small_font, title_font=title_font)
            pygame.display.flip()
            clock.tick(60)
    finally:
        pygame.display.quit()
        pygame.quit()


def _option_index_at_pos(pos: tuple[int, int], *, count: int) -> int | None:
    x, y = pos
    if x < 54 or x > 452:
        return None
    idx = int((y - 136) // 78)
    if 0 <= idx < count and 136 + idx * 78 <= y <= 136 + idx * 78 + 64:
        return idx
    return None


def _draw_launcher(
    pygame: Any,
    screen: Any,
    *,
    options: tuple[GameScenarioOption, ...],
    selected: int,
    font: Any,
    small_font: Any,
    title_font: Any,
) -> None:
    width, height = screen.get_size()
    screen.fill((12, 16, 22))
    _text(screen, title_font, "Orbital Engagement Lab", (54, 36), (238, 242, 248))
    _text(screen, font, "Select RPO training level", (56, 78), (172, 186, 206))
    _text(screen, small_font, "Up/Down or W/S select   Enter/Space launch   Esc quit", (56, 106), (220, 160, 160))

    list_rect = pygame.Rect(42, 124, 424, max(height - 164, 480))
    preview_rect = pygame.Rect(490, 124, max(width - 532, 420), max(height - 164, 480))
    pygame.draw.rect(screen, (18, 24, 32), list_rect, border_radius=8)
    pygame.draw.rect(screen, (70, 82, 100), list_rect, width=1, border_radius=8)

    for idx, option in enumerate(options):
        y = 136 + idx * 78
        rect = pygame.Rect(54, y, 398, 64)
        is_selected = idx == selected
        fill = (28, 48, 66) if is_selected else (20, 27, 36)
        stroke = (96, 174, 224) if is_selected else (48, 60, 76)
        pygame.draw.rect(screen, fill, rect, border_radius=8)
        pygame.draw.rect(screen, stroke, rect, width=2 if is_selected else 1, border_radius=8)
        _text(screen, font, option.title, (rect.x + 18, rect.y + 12), (238, 244, 250))
        difficulty = f"Difficulty: {option.difficulty}" if option.difficulty else ""
        _text(screen, small_font, difficulty, (rect.x + 18, rect.y + 38), (162, 178, 198))

    _draw_preview(pygame, screen, preview_rect, option=options[selected], font=font, small_font=small_font)


def _draw_preview(pygame: Any, screen: Any, rect: Any, *, option: GameScenarioOption, font: Any, small_font: Any) -> None:
    pygame.draw.rect(screen, (18, 24, 32), rect, border_radius=8)
    pygame.draw.rect(screen, (70, 82, 100), rect, width=1, border_radius=8)
    y = rect.y + 18
    _text(screen, font, option.title, (rect.x + 20, y), (238, 244, 250))
    y += 34
    _text(screen, small_font, _budget_line(option), (rect.x + 20, y), (162, 178, 198))
    y += 32
    y = _draw_section(screen, small_font, "Objective", option.learning_goal, rect.x + 20, y, rect.width - 40)
    y = _draw_section(screen, small_font, "Brief", option.player_brief or option.description, rect.x + 20, y + 10, rect.width - 40)
    y = _draw_bullets(screen, small_font, "Pass Criteria", option.pass_criteria, rect.x + 20, y + 10, rect.width - 40)
    y = _draw_bullets(screen, small_font, "Instructor Notes", option.instructor_notes, rect.x + 20, y + 10, rect.width - 40)


def _text(screen: Any, font: Any, text: str, pos: tuple[int, int], color: tuple[int, int, int]) -> None:
    if not text:
        return
    surf = font.render(str(text), True, color)
    screen.blit(surf, pos)


def _draw_section(screen: Any, font: Any, title: str, body: str, x: int, y: int, width_px: int) -> int:
    _text(screen, font, title, (x, y), (238, 244, 250))
    y += 22
    for line in _wrap_text(body, max(24, width_px // 8)):
        _text(screen, font, line, (x, y), (182, 194, 210))
        y += 18
    return y


def _draw_bullets(screen: Any, font: Any, title: str, items: tuple[str, ...], x: int, y: int, width_px: int) -> int:
    if not items:
        return y
    _text(screen, font, title, (x, y), (238, 244, 250))
    y += 22
    for item in items:
        wrapped = _wrap_text(item, max(24, (width_px - 18) // 8))
        for idx, line in enumerate(wrapped):
            prefix = "- " if idx == 0 else "  "
            _text(screen, font, prefix + line, (x, y), (182, 194, 210))
            y += 18
    return y


def _budget_line(option: GameScenarioOption) -> str:
    parts = [f"Difficulty: {option.difficulty}" if option.difficulty else "Difficulty: --"]
    if option.time_budget_s is not None:
        parts.append(f"Time: {option.time_budget_s:.0f}s")
    if option.delta_v_budget_m_s is not None:
        parts.append(f"Chaser dV: {option.delta_v_budget_m_s:.1f} m/s")
    if option.goal_speed_km_s is not None:
        parts.append(f"Speed gate: {option.goal_speed_km_s * 1000.0:.2f} m/s")
    if option.target_delta_v_budget_m_s is not None:
        parts.append(f"Target dV: {option.target_delta_v_budget_m_s:.1f} m/s")
    return "   ".join(parts)


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


def _as_str_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value if str(item))
    return (str(value),)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _truncate(value: str, max_chars: int) -> str:
    text = " ".join(str(value).split())
    if len(text) <= max_chars:
        return text
    return text[: max(max_chars - 3, 0)].rstrip() + "..."
