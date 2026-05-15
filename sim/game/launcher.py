from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

GAME_CONFIG_DIR = Path(__file__).resolve().parent / "configs"
GAME_PROGRESS_PATH_ENV = "OEL_GAME_PROGRESS_PATH"
DIFFICULTY_OPTIONS: tuple[str, ...] = ("easy", "medium", "hard", "extreme")
OPTION_X = 54
OPTION_Y = 136
OPTION_WIDTH = 398
OPTION_HEIGHT = 64
OPTION_ROW_HEIGHT = 78
CLEAR_PROGRESS_RECT = (846, 36, 150, 30)
RECORD_VIDEO_RECT = (682, 36, 144, 30)


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
    completed_difficulties: tuple[str, ...]
    level_number: int


@dataclass(frozen=True)
class GameLaunchSelection:
    path: Path
    difficulty: str
    record_video: bool = False


def discover_game_scenarios(config_dir: Path | None = None) -> tuple[GameScenarioOption, ...]:
    root = Path(config_dir) if config_dir is not None else GAME_CONFIG_DIR
    progress = _load_game_progress()
    options: list[GameScenarioOption] = []
    for path in sorted(root.glob("game_training_rpo_*.yaml")):
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        options.append(_scenario_option_from_yaml(path, raw, progress_by_scenario=progress))
    return tuple(sorted(options, key=lambda option: (option.level_number, option.scenario_id)))


def choose_game_scenario(config_dir: Path | None = None) -> Path | None:
    selection = choose_game_launch(config_dir)
    return None if selection is None else selection.path


def choose_game_launch(config_dir: Path | None = None) -> GameLaunchSelection | None:
    options = discover_game_scenarios(config_dir)
    if not options:
        raise RuntimeError(f"No game training configs found in {Path(config_dir) if config_dir else GAME_CONFIG_DIR}.")
    return _run_launcher(options)


def _scenario_option_from_yaml(
    path: Path,
    raw: dict[str, Any],
    *,
    progress_by_scenario: dict[str, tuple[str, ...]] | None = None,
) -> GameScenarioOption:
    metadata = dict(raw.get("metadata", {}) or {})
    game = dict(metadata.get("game", {}) or {})
    training = dict(game.get("training", {}) or {})
    scenario_id = str(training.get("scenario_id", raw.get("scenario_name", path.stem)) or path.stem)
    if progress_by_scenario is not None and scenario_id in progress_by_scenario:
        completed_difficulties = tuple(progress_by_scenario[scenario_id])
    else:
        completed_difficulties = _completed_difficulties_from_game(game)
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
        completed_difficulties=completed_difficulties,
        level_number=level_number,
    )


def record_game_progress(config_path: str | Path, difficulty: str) -> None:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    scenario_id = _scenario_id_from_yaml(path, raw)
    progress = _load_game_progress()
    completed = list(progress.get(scenario_id, ()))
    normalized = _normalize_difficulty(difficulty)
    if normalized not in completed:
        completed.append(normalized)
    progress[scenario_id] = tuple(item for item in DIFFICULTY_OPTIONS if item in completed)
    _save_game_progress(progress)


def clear_game_progress(config_dir: Path | None = None) -> None:
    root = Path(config_dir) if config_dir is not None else GAME_CONFIG_DIR
    progress = _load_game_progress()
    changed = False
    for path in sorted(root.glob("game_training_rpo_*.yaml")):
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        scenario_id = _scenario_id_from_yaml(path, raw)
        if scenario_id not in progress or progress.get(scenario_id, ()) != ():
            progress[scenario_id] = ()
            changed = True
    if changed:
        _save_game_progress(progress)


def _scenario_id_from_yaml(path: Path, raw: dict[str, Any]) -> str:
    metadata = dict(raw.get("metadata", {}) or {})
    game = dict(metadata.get("game", {}) or {})
    training = dict(game.get("training", {}) or {})
    return str(training.get("scenario_id", raw.get("scenario_name", path.stem)) or path.stem)


def _game_progress_path() -> Path:
    override = os.environ.get(GAME_PROGRESS_PATH_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".orbital_engagement_lab" / "game_progress.yaml"


def _load_game_progress() -> dict[str, tuple[str, ...]]:
    path = _game_progress_path()
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    scenarios = dict(raw.get("scenarios", {}) or {}) if isinstance(raw, dict) else {}
    progress: dict[str, tuple[str, ...]] = {}
    for scenario_id, item in scenarios.items():
        if not isinstance(item, dict):
            continue
        completed = item.get("completed_difficulties", ())
        if isinstance(completed, str):
            completed = (completed,)
        values = {_normalize_difficulty(value) for value in completed}
        progress[str(scenario_id)] = tuple(difficulty for difficulty in DIFFICULTY_OPTIONS if difficulty in values)
    return progress


def _save_game_progress(progress: dict[str, tuple[str, ...]]) -> None:
    path = _game_progress_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    scenarios = {}
    for scenario_id, completed in sorted(progress.items()):
        completed_set = set(completed)
        scenarios[scenario_id] = {
            "completed_difficulties": [item for item in DIFFICULTY_OPTIONS if item in completed_set]
        }
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump({"scenarios": scenarios}, f, sort_keys=False)


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


def _run_launcher(options: tuple[GameScenarioOption, ...]) -> GameLaunchSelection | None:
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
    scroll_offset = 0
    difficulty_idx = _difficulty_index(options[selected].difficulty)
    record_video = False

    try:
        while True:
            _, height = screen.get_size()
            for event in pygame.event.get():
                selected_difficulty = DIFFICULTY_OPTIONS[difficulty_idx]
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
                        return GameLaunchSelection(
                            path=options[selected].path,
                            difficulty=selected_difficulty,
                            record_video=record_video,
                        )
                    elif event.key == pygame.K_v:
                        record_video = not record_video
                    elif event.key in {pygame.K_LEFT, pygame.K_a}:
                        difficulty_idx = max(difficulty_idx - 1, 0)
                    elif event.key in {pygame.K_RIGHT, pygame.K_d}:
                        difficulty_idx = min(difficulty_idx + 1, len(DIFFICULTY_OPTIONS) - 1)
                    elif event.key == pygame.K_1:
                        difficulty_idx = 0
                    elif event.key == pygame.K_2:
                        difficulty_idx = 1
                    elif event.key == pygame.K_3:
                        difficulty_idx = 2
                    elif event.key == pygame.K_4:
                        difficulty_idx = 3
                    scroll_offset = _scroll_for_selection(
                        selected, scroll_offset, count=len(options), screen_height=height
                    )
                if event.type == pygame.MOUSEWHEEL:
                    scroll_offset = int(
                        max(0, min(scroll_offset - int(event.y), _max_scroll_offset(len(options), height)))
                    )
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    idx = _option_index_at_pos(pygame.mouse.get_pos(), count=len(options), scroll_offset=scroll_offset)
                    if idx is not None:
                        if idx == selected:
                            return GameLaunchSelection(
                                path=options[selected].path,
                                difficulty=selected_difficulty,
                                record_video=record_video,
                            )
                        selected = idx
                    else:
                        mouse_pos = pygame.mouse.get_pos()
                        if _record_video_at_pos(mouse_pos):
                            record_video = not record_video
                            continue
                        if _clear_progress_at_pos(mouse_pos):
                            clear_game_progress(options[0].path.parent)
                            options = discover_game_scenarios(options[0].path.parent)
                            selected = min(selected, len(options) - 1)
                            difficulty_idx = _difficulty_index(options[selected].difficulty)
                            scroll_offset = _scroll_for_selection(
                                selected, scroll_offset, count=len(options), screen_height=height
                            )
                            continue
                        difficulty = _difficulty_at_pos(mouse_pos)
                        if difficulty is not None:
                            difficulty_idx = _difficulty_index(difficulty)
                if event.type == pygame.MOUSEMOTION:
                    idx = _option_index_at_pos(event.pos, count=len(options), scroll_offset=scroll_offset)
                    if idx is not None:
                        selected = idx
            scroll_offset = _scroll_for_selection(selected, scroll_offset, count=len(options), screen_height=height)
            selected_difficulty = DIFFICULTY_OPTIONS[difficulty_idx]

            _draw_launcher(
                pygame,
                screen,
                options=options,
                selected=selected,
                scroll_offset=scroll_offset,
                selected_difficulty=selected_difficulty,
                record_video=record_video,
                font=font,
                small_font=small_font,
                title_font=title_font,
            )
            pygame.display.flip()
            clock.tick(60)
    finally:
        pygame.display.quit()
        pygame.quit()


def _option_index_at_pos(pos: tuple[int, int], *, count: int, scroll_offset: int = 0) -> int | None:
    x, y = pos
    if x < OPTION_X or x > OPTION_X + OPTION_WIDTH:
        return None
    if y < OPTION_Y:
        return None
    row = int((y - OPTION_Y) // OPTION_ROW_HEIGHT)
    idx = int(scroll_offset) + row
    row_y = OPTION_Y + row * OPTION_ROW_HEIGHT
    if 0 <= idx < count and row_y <= y <= row_y + OPTION_HEIGHT:
        return idx
    return None


def _visible_option_count(screen_height: int) -> int:
    list_bottom = 124 + max(int(screen_height) - 164, 480)
    return max(1, int((list_bottom - OPTION_Y) // OPTION_ROW_HEIGHT))


def _max_scroll_offset(count: int, screen_height: int) -> int:
    return max(int(count) - _visible_option_count(screen_height), 0)


def _scroll_for_selection(selected: int, scroll_offset: int, *, count: int, screen_height: int) -> int:
    max_scroll = _max_scroll_offset(count, screen_height)
    visible = _visible_option_count(screen_height)
    scroll = int(max(0, min(scroll_offset, max_scroll)))
    if selected < scroll:
        return int(max(0, selected))
    if selected >= scroll + visible:
        return int(min(max_scroll, selected - visible + 1))
    return scroll


def _difficulty_index(value: str) -> int:
    key = str(value or "easy").strip().lower()
    if key == "normal":
        key = "medium"
    if key == "expert":
        key = "extreme"
    if key in DIFFICULTY_OPTIONS:
        return DIFFICULTY_OPTIONS.index(key)
    return 0


def _difficulty_at_pos(pos: tuple[int, int]) -> str | None:
    x, y = pos
    if y < 86 or y > 112:
        return None
    for idx, difficulty in enumerate(DIFFICULTY_OPTIONS):
        left = 642 + idx * 86
        right = left + 76
        if left <= x <= right:
            return difficulty
    return None


def _clear_progress_at_pos(pos: tuple[int, int]) -> bool:
    x, y, w, h = CLEAR_PROGRESS_RECT
    px, py = pos
    return x <= px <= x + w and y <= py <= y + h


def _record_video_at_pos(pos: tuple[int, int]) -> bool:
    x, y, w, h = RECORD_VIDEO_RECT
    px, py = pos
    return x <= px <= x + w and y <= py <= y + h


def _draw_launcher(
    pygame: Any,
    screen: Any,
    *,
    options: tuple[GameScenarioOption, ...],
    selected: int,
    scroll_offset: int,
    selected_difficulty: str,
    record_video: bool,
    font: Any,
    small_font: Any,
    title_font: Any,
) -> None:
    width, height = screen.get_size()
    screen.fill((12, 16, 22))
    _text(screen, title_font, "Orbital Engagement Lab", (54, 36), (238, 242, 248))
    _draw_record_video_button(pygame, screen, enabled=record_video, font=small_font)
    _draw_clear_progress_button(pygame, screen, font=small_font)
    _text(screen, font, "Select RPO training level", (56, 78), (172, 186, 206))
    _text(
        screen,
        small_font,
        "Up/Down select   Left/Right difficulty   V video   Enter launch   Esc quit",
        (56, 106),
        (220, 160, 160),
    )
    _draw_difficulty_picker(pygame, screen, selected_difficulty=selected_difficulty, font=small_font)

    list_rect = pygame.Rect(42, 124, 424, max(height - 164, 480))
    preview_rect = pygame.Rect(490, 124, max(width - 532, 420), max(height - 164, 480))
    pygame.draw.rect(screen, (18, 24, 32), list_rect, border_radius=8)
    pygame.draw.rect(screen, (70, 82, 100), list_rect, width=1, border_radius=8)

    visible = _visible_option_count(height)
    for row, option in enumerate(options[scroll_offset : scroll_offset + visible]):
        idx = scroll_offset + row
        y = OPTION_Y + row * OPTION_ROW_HEIGHT
        rect = pygame.Rect(OPTION_X, y, OPTION_WIDTH, OPTION_HEIGHT)
        is_selected = idx == selected
        fill = (28, 48, 66) if is_selected else (20, 27, 36)
        stroke = (96, 174, 224) if is_selected else (48, 60, 76)
        pygame.draw.rect(screen, fill, rect, border_radius=8)
        pygame.draw.rect(screen, stroke, rect, width=2 if is_selected else 1, border_radius=8)
        _text(screen, font, option.title, (rect.x + 18, rect.y + 12), (238, 244, 250))
        _text(
            screen,
            small_font,
            f"Progress: {_progress_stars(option.completed_difficulties)}",
            (rect.x + 18, rect.y + 38),
            (162, 178, 198),
        )

    if len(options) > visible:
        _draw_scrollbar(pygame, screen, list_rect, count=len(options), visible=visible, scroll_offset=scroll_offset)

    _draw_preview(
        pygame,
        screen,
        preview_rect,
        option=options[selected],
        selected_difficulty=selected_difficulty,
        font=font,
        small_font=small_font,
    )


def _draw_scrollbar(pygame: Any, screen: Any, rect: Any, *, count: int, visible: int, scroll_offset: int) -> None:
    track = pygame.Rect(rect.right - 12, rect.y + 12, 4, max(rect.height - 24, 20))
    pygame.draw.rect(screen, (38, 48, 62), track, border_radius=2)
    frac = min(float(visible) / max(float(count), 1.0), 1.0)
    thumb_h = max(int(track.height * frac), 28)
    max_scroll = max(int(count) - int(visible), 1)
    travel = max(track.height - thumb_h, 0)
    thumb_y = track.y + int(round(travel * min(max(float(scroll_offset) / float(max_scroll), 0.0), 1.0)))
    thumb = pygame.Rect(track.x, thumb_y, track.width, thumb_h)
    pygame.draw.rect(screen, (96, 174, 224), thumb, border_radius=2)


def _draw_clear_progress_button(pygame: Any, screen: Any, *, font: Any) -> None:
    rect = pygame.Rect(*CLEAR_PROGRESS_RECT)
    pygame.draw.rect(screen, (32, 38, 48), rect, border_radius=6)
    pygame.draw.rect(screen, (120, 132, 150), rect, width=1, border_radius=6)
    _text(screen, font, "Clear Progress", (rect.x + 15, rect.y + 8), (230, 238, 245))


def _draw_record_video_button(pygame: Any, screen: Any, *, enabled: bool, font: Any) -> None:
    rect = pygame.Rect(*RECORD_VIDEO_RECT)
    fill = (76, 34, 42) if enabled else (32, 38, 48)
    stroke = (245, 94, 108) if enabled else (120, 132, 150)
    pygame.draw.rect(screen, fill, rect, border_radius=6)
    pygame.draw.rect(screen, stroke, rect, width=1, border_radius=6)
    label = "Video: ON" if enabled else "Video: OFF"
    _text(screen, font, label, (rect.x + 18, rect.y + 8), (246, 238, 242))


def _draw_difficulty_picker(pygame: Any, screen: Any, *, selected_difficulty: str, font: Any) -> None:
    _text(screen, font, "Assists", (574, 94), (172, 186, 206))
    for idx, difficulty in enumerate(DIFFICULTY_OPTIONS):
        rect = pygame.Rect(642 + idx * 86, 86, 76, 26)
        is_selected = difficulty == selected_difficulty
        fill = (36, 72, 52) if is_selected else (18, 24, 32)
        stroke = (108, 232, 142) if is_selected else (70, 82, 100)
        pygame.draw.rect(screen, fill, rect, border_radius=6)
        pygame.draw.rect(screen, stroke, rect, width=1, border_radius=6)
        _text(screen, font, difficulty.title(), (rect.x + 9, rect.y + 6), (230, 238, 245))


def _draw_preview(
    pygame: Any,
    screen: Any,
    rect: Any,
    *,
    option: GameScenarioOption,
    selected_difficulty: str,
    font: Any,
    small_font: Any,
) -> None:
    pygame.draw.rect(screen, (18, 24, 32), rect, border_radius=8)
    pygame.draw.rect(screen, (70, 82, 100), rect, width=1, border_radius=8)
    y = rect.y + 18
    _text(screen, font, option.title, (rect.x + 20, y), (238, 244, 250))
    y += 34
    _text(
        screen,
        small_font,
        _budget_line(option, selected_difficulty=selected_difficulty),
        (rect.x + 20, y),
        (162, 178, 198),
    )
    y += 32
    y = _draw_section(screen, small_font, "Objective", option.learning_goal, rect.x + 20, y, rect.width - 40)
    y = _draw_section(
        screen, small_font, "Brief", option.player_brief or option.description, rect.x + 20, y + 10, rect.width - 40
    )
    y = _draw_bullets(screen, small_font, "Pass Criteria", option.pass_criteria, rect.x + 20, y + 10, rect.width - 40)
    y = _draw_bullets(
        screen, small_font, "Instructor Notes", option.instructor_notes, rect.x + 20, y + 10, rect.width - 40
    )


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


def _budget_line(option: GameScenarioOption, *, selected_difficulty: str | None = None) -> str:
    difficulty = selected_difficulty or option.difficulty
    parts = [f"Assists: {difficulty}" if difficulty else "Assists: --"]
    if option.time_budget_s is not None:
        parts.append(f"Time: {option.time_budget_s:.0f}s")
    if option.delta_v_budget_m_s is not None:
        parts.append(f"Chaser dV: {option.delta_v_budget_m_s:.1f} m/s")
    if option.goal_speed_km_s is not None:
        parts.append(f"Speed gate: {option.goal_speed_km_s * 1000.0:.2f} m/s")
    if option.target_delta_v_budget_m_s is not None:
        parts.append(f"Target dV: {option.target_delta_v_budget_m_s:.1f} m/s")
    return "   ".join(parts)


def _completed_difficulties_from_game(game: dict[str, Any]) -> tuple[str, ...]:
    progress = dict(game.get("progress", {}) or {})
    completed = progress.get("completed_difficulties", ())
    if isinstance(completed, str):
        completed = (completed,)
    values = {_normalize_difficulty(item) for item in completed}
    return tuple(item for item in DIFFICULTY_OPTIONS if item in values)


def _normalize_difficulty(value: Any) -> str:
    key = str(value or "easy").strip().lower()
    if key == "normal":
        return "medium"
    if key == "expert":
        return "extreme"
    if key in DIFFICULTY_OPTIONS:
        return key
    return "easy"


def _progress_stars(completed_difficulties: tuple[str, ...]) -> str:
    highest = -1
    for difficulty in completed_difficulties:
        if difficulty in DIFFICULTY_OPTIONS:
            highest = max(highest, DIFFICULTY_OPTIONS.index(difficulty))
    earned = highest + 1
    return "★" * earned + "☆" * (len(DIFFICULTY_OPTIONS) - earned)


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
