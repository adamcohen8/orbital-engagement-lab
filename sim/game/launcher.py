from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from sim.game.fonts import game_font
from sim.game.formatting import format_speed_km_s, format_speed_m_s

GAME_CONFIG_DIR = Path(__file__).resolve().parent / "configs"
LAUNCHER_MUSIC_PATH = Path(__file__).resolve().parent / "music" / "01_insert_coin_to_orbit.wav"
START_SCREEN_LOGO_PATH = Path(__file__).resolve().parent / "assets" / "OEL_RPO_Trainer.png"
GAME_PROGRESS_PATH_ENV = "OEL_GAME_PROGRESS_PATH"
DIFFICULTY_OPTIONS: tuple[str, ...] = ("easy", "medium", "hard", "extreme")
OPTION_X = 54
OPTION_Y = 136
OPTION_WIDTH = 398
OPTION_HEIGHT = 64
OPTION_ROW_HEIGHT = 78
PANEL_TOP = 124
MIN_PANEL_HEIGHT = 480
FOOTER_HEIGHT = 76
FOOTER_BOTTOM_MARGIN = 22
PREVIEW_PADDING = 20
PREVIEW_LINE_HEIGHT = 18
PREVIEW_SECTION_TITLE_GAP = 22
PREVIEW_SECTION_GAP = 10
PREVIEW_SCROLL_STEP_PX = PREVIEW_LINE_HEIGHT * 3
CLEAR_PROGRESS_RECT = (846, 36, 150, 30)
MUSIC_RECT = (518, 36, 144, 30)
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
    high_score: int
    level_number: int


@dataclass(frozen=True)
class GameProgressRecord:
    completed_difficulties: tuple[str, ...] = ()
    high_score: int = 0


@dataclass(frozen=True)
class GameLaunchSelection:
    path: Path
    difficulty: str
    music_enabled: bool = True
    record_video: bool = False


def discover_game_scenarios(config_dir: Path | None = None) -> tuple[GameScenarioOption, ...]:
    root = Path(config_dir) if config_dir is not None else GAME_CONFIG_DIR
    progress = _load_game_progress()
    options: list[GameScenarioOption] = []
    for path in sorted(root.glob("game_training_rpo_*.yaml")):
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        options.append(_scenario_option_from_yaml(path, raw, progress_by_scenario=progress))
    return tuple(sorted(options, key=_scenario_sort_key))


def choose_game_scenario(config_dir: Path | None = None) -> Path | None:
    selection = choose_game_launch(config_dir)
    return None if selection is None else selection.path


def choose_game_launch(config_dir: Path | None = None, *, show_start_screen: bool = True) -> GameLaunchSelection | None:
    options = discover_game_scenarios(config_dir)
    if not options:
        raise RuntimeError(f"No game training configs found in {Path(config_dir) if config_dir else GAME_CONFIG_DIR}.")
    return _run_launcher(options, show_start_screen=show_start_screen)


def _scenario_option_from_yaml(
    path: Path,
    raw: dict[str, Any],
    *,
    progress_by_scenario: dict[str, GameProgressRecord] | None = None,
) -> GameScenarioOption:
    metadata = dict(raw.get("metadata", {}) or {})
    game = dict(metadata.get("game", {}) or {})
    training = dict(game.get("training", {}) or {})
    scenario_id = str(training.get("scenario_id", raw.get("scenario_name", path.stem)) or path.stem)
    if progress_by_scenario is not None and scenario_id in progress_by_scenario:
        record = progress_by_scenario[scenario_id]
        completed_difficulties = tuple(record.completed_difficulties)
        high_score = int(record.high_score)
    else:
        completed_difficulties = _completed_difficulties_from_game(game)
        high_score = _high_score_from_game(game)
    level_number = _level_number_from_scenario_id(scenario_id)
    level_name = str(game.get("level_name", "") or "").strip()
    target_delta_v_budget = _optional_float(training.get("max_target_delta_v_m_s"))
    if target_delta_v_budget is None:
        target_delta_v_budget = _optional_float(dict(game.get("defensive_target", {}) or {}).get("max_delta_v_m_s"))
    return GameScenarioOption(
        path=path,
        scenario_id=scenario_id,
        title=level_name or _title_from_scenario_id(scenario_id, level_number=level_number),
        description=str(raw.get("scenario_description", "") or ""),
        learning_goal=str(training.get("learning_goal", "") or ""),
        player_brief=str(training.get("player_brief", "") or ""),
        pass_criteria=_as_str_tuple(training.get("pass_criteria")),
        instructor_notes=_as_str_tuple(training.get("instructor_notes")),
        difficulty=str(game.get("difficulty", "") or ""),
        time_budget_s=_optional_float(training.get("max_time_s")),
        delta_v_budget_m_s=_optional_float(training.get("max_delta_v_m_s")),
        goal_speed_km_s=_optional_float(training.get("max_goal_speed_km_s")),
        target_delta_v_budget_m_s=target_delta_v_budget,
        completed_difficulties=completed_difficulties,
        high_score=high_score,
        level_number=level_number,
    )


def record_game_progress(
    config_path: str | Path,
    difficulty: str,
    score: int | None = None,
    *,
    completed: bool = True,
) -> None:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    scenario_id = _scenario_id_from_yaml(path, raw)
    progress = _load_game_progress()
    current = progress.get(scenario_id, GameProgressRecord())
    completed_difficulties = list(current.completed_difficulties)
    normalized = _normalize_difficulty(difficulty)
    if bool(completed) and normalized not in completed_difficulties:
        completed_difficulties.append(normalized)
    high_score = int(current.high_score)
    if score is not None:
        high_score = max(high_score, int(max(score, 0)))
    progress[scenario_id] = GameProgressRecord(
        completed_difficulties=tuple(item for item in DIFFICULTY_OPTIONS if item in completed_difficulties),
        high_score=high_score,
    )
    _save_game_progress(progress)


def clear_game_progress(config_dir: Path | None = None) -> None:
    root = Path(config_dir) if config_dir is not None else GAME_CONFIG_DIR
    progress = _load_game_progress()
    changed = False
    for path in sorted(root.glob("game_training_rpo_*.yaml")):
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        scenario_id = _scenario_id_from_yaml(path, raw)
        if scenario_id not in progress or progress.get(scenario_id) != GameProgressRecord():
            progress[scenario_id] = GameProgressRecord()
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


def _load_game_progress() -> dict[str, GameProgressRecord]:
    path = _game_progress_path()
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    scenarios = dict(raw.get("scenarios", {}) or {}) if isinstance(raw, dict) else {}
    progress: dict[str, GameProgressRecord] = {}
    for scenario_id, item in scenarios.items():
        if not isinstance(item, dict):
            continue
        completed = item.get("completed_difficulties", ())
        if isinstance(completed, str):
            completed = (completed,)
        values = {_normalize_difficulty(value) for value in completed}
        progress[str(scenario_id)] = GameProgressRecord(
            completed_difficulties=tuple(difficulty for difficulty in DIFFICULTY_OPTIONS if difficulty in values),
            high_score=max(int(item.get("high_score", 0) or 0), 0),
        )
    return progress


def _save_game_progress(progress: dict[str, GameProgressRecord]) -> None:
    path = _game_progress_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    scenarios = {}
    for scenario_id, record in sorted(progress.items()):
        completed_set = set(record.completed_difficulties)
        scenarios[scenario_id] = {
            "completed_difficulties": [item for item in DIFFICULTY_OPTIONS if item in completed_set],
            "high_score": int(max(record.high_score, 0)),
        }
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump({"scenarios": scenarios}, f, sort_keys=False)


def _level_number_from_scenario_id(scenario_id: str) -> int:
    parts = str(scenario_id).split("_")
    for part in parts:
        if part.isdigit():
            return int(part)
        digits = ""
        for char in part:
            if not char.isdigit():
                break
            digits += char
        if digits:
            return int(digits)
    return 999


def _scenario_sort_key(option: GameScenarioOption) -> tuple[int, str]:
    if option.scenario_id == "rpo_bonus_cislunar_rendezvous":
        return (12, option.scenario_id)
    if option.scenario_id == "rpo_arcade_pursuit":
        return (13, option.scenario_id)
    if option.scenario_id == "rpo_sandbox":
        return (14, option.scenario_id)
    return (option.level_number, option.scenario_id)


def _title_from_scenario_id(scenario_id: str, *, level_number: int) -> str:
    parts = str(scenario_id).split("_")
    if len(parts) >= 3 and parts[0] == "rpo" and parts[1].isdigit():
        name = " ".join(parts[2:]).title()
        return f"Level {level_number} - {name}"
    return str(scenario_id).replace("_", " ").title()


def _run_launcher(options: tuple[GameScenarioOption, ...], *, show_start_screen: bool = True) -> GameLaunchSelection | None:
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
    font = game_font(pygame, 19)
    small_font = game_font(pygame, 15)
    title_font = game_font(pygame, 32)
    hero_font = game_font(pygame, 46)
    selected = 0
    scroll_offset = 0
    preview_scroll_px = 0
    difficulty_idx = _difficulty_index(options[selected].difficulty)
    music_enabled = _start_launcher_music(pygame)
    start_artwork = _load_start_screen_artwork(pygame)
    record_video = False
    start_screen_open = bool(show_start_screen)

    try:
        while True:
            width, height = screen.get_size()
            if start_screen_open:
                for event in pygame.event.get():
                    action = _start_screen_event_action(pygame, event)
                    if action == "quit":
                        return None
                    if action == "begin":
                        start_screen_open = False
                        break
                if start_screen_open:
                    _draw_start_screen(
                        pygame,
                        screen,
                        artwork=start_artwork,
                        hero_font=hero_font,
                        font=font,
                        small_font=small_font,
                    )
                    pygame.display.flip()
                    clock.tick(60)
                    continue

            preview_bounds = _preview_bounds(width, height)
            for event in pygame.event.get():
                selected_difficulty = DIFFICULTY_OPTIONS[difficulty_idx]
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return None
                    if event.key in {pygame.K_DOWN, pygame.K_s}:
                        new_selected = min(selected + 1, len(options) - 1)
                        if new_selected != selected:
                            preview_scroll_px = 0
                        selected = new_selected
                    elif event.key in {pygame.K_UP, pygame.K_w}:
                        new_selected = max(selected - 1, 0)
                        if new_selected != selected:
                            preview_scroll_px = 0
                        selected = new_selected
                    elif event.key in {pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE}:
                        return GameLaunchSelection(
                            path=options[selected].path,
                            difficulty=selected_difficulty,
                            music_enabled=music_enabled,
                            record_video=record_video,
                        )
                    elif event.key == pygame.K_v:
                        record_video = not record_video
                    elif event.key == pygame.K_m:
                        music_enabled = _toggle_launcher_music(pygame, music_enabled=music_enabled)
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
                    if _pos_in_bounds(pygame.mouse.get_pos(), preview_bounds):
                        preview_scroll_px = _clamp_preview_scroll_px(
                            preview_scroll_px - int(event.y) * PREVIEW_SCROLL_STEP_PX,
                            option=options[selected],
                            font=font,
                            small_font=small_font,
                            preview_bounds=preview_bounds,
                        )
                    else:
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
                                music_enabled=music_enabled,
                                record_video=record_video,
                            )
                        preview_scroll_px = 0
                        selected = idx
                    else:
                        mouse_pos = pygame.mouse.get_pos()
                        if _record_video_at_pos(mouse_pos):
                            record_video = not record_video
                            continue
                        if _music_at_pos(mouse_pos):
                            music_enabled = _toggle_launcher_music(pygame, music_enabled=music_enabled)
                            continue
                        if _clear_progress_at_pos(mouse_pos):
                            clear_game_progress(options[0].path.parent)
                            options = discover_game_scenarios(options[0].path.parent)
                            selected = min(selected, len(options) - 1)
                            preview_scroll_px = 0
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
                    if idx is not None and idx != selected:
                        preview_scroll_px = 0
                        selected = idx
            scroll_offset = _scroll_for_selection(selected, scroll_offset, count=len(options), screen_height=height)
            selected_difficulty = DIFFICULTY_OPTIONS[difficulty_idx]
            preview_scroll_px = _clamp_preview_scroll_px(
                preview_scroll_px,
                option=options[selected],
                font=font,
                small_font=small_font,
                preview_bounds=preview_bounds,
            )

            _draw_launcher(
                pygame,
                screen,
                options=options,
                selected=selected,
                scroll_offset=scroll_offset,
                selected_difficulty=selected_difficulty,
                music_enabled=music_enabled,
                preview_scroll_px=preview_scroll_px,
                record_video=record_video,
                font=font,
                small_font=small_font,
                title_font=title_font,
            )
            pygame.display.flip()
            clock.tick(60)
    finally:
        _stop_launcher_music(pygame)
        pygame.display.quit()
        pygame.quit()


def _start_screen_event_action(pygame: Any, event: Any) -> str:
    if event.type == pygame.QUIT:
        return "quit"
    if event.type == pygame.KEYDOWN:
        if getattr(event, "key", None) == pygame.K_ESCAPE:
            return "quit"
        return "begin"
    return "ignore"


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
    list_bottom = PANEL_TOP + _launcher_panel_height(screen_height)
    return max(1, int((list_bottom - OPTION_Y) // OPTION_ROW_HEIGHT))


def _launcher_panel_height(screen_height: int) -> int:
    return max(int(screen_height) - PANEL_TOP - FOOTER_HEIGHT, MIN_PANEL_HEIGHT)


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


def _preview_bounds(screen_width: int, screen_height: int) -> tuple[int, int, int, int]:
    panel_height = _launcher_panel_height(screen_height)
    return (490, PANEL_TOP, max(int(screen_width) - 532, 420), panel_height)


def _pos_in_bounds(pos: tuple[int, int], bounds: tuple[int, int, int, int]) -> bool:
    px, py = pos
    x, y, w, h = bounds
    return x <= px <= x + w and y <= py <= y + h


def _clamp_preview_scroll_px(
    scroll_px: int,
    *,
    option: GameScenarioOption,
    font: Any,
    small_font: Any,
    preview_bounds: tuple[int, int, int, int],
) -> int:
    _, _, width, height = preview_bounds
    content_width = max(int(width) - PREVIEW_PADDING * 2, 1)
    viewport_height = max(int(height) - PREVIEW_PADDING * 2, 1)
    content_height = _preview_content_height(option, font=font, small_font=small_font, width_px=content_width)
    max_scroll = max(int(content_height) - viewport_height, 0)
    return int(max(0, min(int(scroll_px), max_scroll)))


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


def _music_at_pos(pos: tuple[int, int]) -> bool:
    x, y, w, h = MUSIC_RECT
    px, py = pos
    return x <= px <= x + w and y <= py <= y + h


def _start_launcher_music(pygame: Any) -> bool:
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        pygame.mixer.music.load(str(LAUNCHER_MUSIC_PATH))
        pygame.mixer.music.set_volume(0.55)
        pygame.mixer.music.play(-1)
    except (OSError, pygame.error):
        return False
    return True


def _stop_launcher_music(pygame: Any) -> None:
    try:
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
    except pygame.error:
        return


def _toggle_launcher_music(pygame: Any, *, music_enabled: bool) -> bool:
    if music_enabled:
        _stop_launcher_music(pygame)
        return False
    return _start_launcher_music(pygame)


def _load_start_screen_artwork(pygame: Any) -> Any | None:
    try:
        return pygame.image.load(str(START_SCREEN_LOGO_PATH)).convert()
    except (OSError, pygame.error):
        return None


def _start_artwork_rect(
    image_size: tuple[int, int],
    screen_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    image_w, image_h = image_size
    screen_w, screen_h = screen_size
    if image_w <= 0 or image_h <= 0 or screen_w <= 0 or screen_h <= 0:
        return (0, 0, 0, 0)
    scale = min(screen_w / float(image_w), screen_h / float(image_h))
    width = max(int(round(image_w * scale)), 1)
    height = max(int(round(image_h * scale)), 1)
    x = int((screen_w - width) // 2)
    y = int((screen_h - height) // 2)
    return (x, y, width, height)


def _draw_start_screen(
    pygame: Any,
    screen: Any,
    *,
    artwork: Any | None = None,
    hero_font: Any,
    font: Any,
    small_font: Any,
) -> None:
    width, height = screen.get_size()
    if artwork is not None:
        rect_tuple = _start_artwork_rect(artwork.get_size(), (width, height))
        artwork_rect = pygame.Rect(*rect_tuple)
        if artwork_rect.width > 0 and artwork_rect.height > 0:
            scaled_artwork = pygame.transform.smoothscale(
                artwork,
                (artwork_rect.width, artwork_rect.height),
            )
            screen.blit(scaled_artwork, artwork_rect)
            return

    screen.fill((8, 12, 18))
    center = (width // 2, height // 2)
    accent = (96, 174, 224)
    pygame.draw.line(screen, (26, 38, 52), (center[0], 34), (center[0], height - 72), width=1)
    pygame.draw.line(screen, (26, 38, 52), (72, center[1] + 4), (width - 72, center[1] + 4), width=1)
    for radius in (170, 255, 340):
        rect = pygame.Rect(0, 0, radius * 2, int(radius * 0.72))
        rect.center = (center[0], center[1] - 30)
        pygame.draw.ellipse(screen, (22, 38, 56), rect, width=1)
    _text_centered(screen, hero_font, "Orbital Engagement Lab", (center[0], center[1] - 150), (238, 244, 250))
    title_y = center[1] - 82
    _text_centered(screen, hero_font, "RPO TRAINER", (center[0] + 3, title_y + 3), (10, 42, 68))
    _text_centered(screen, hero_font, "RPO TRAINER", (center[0], title_y), (210, 246, 255))
    _text_centered(screen, small_font, "RENDEZVOUS  PROXIMITY  OPERATIONS", (center[0], title_y + 42), (96, 174, 224))
    prompt_rect = pygame.Rect(0, 0, 360, 44)
    prompt_y = min(max(title_y + 104, center[1] + 150), height - 88)
    prompt_rect.center = (center[0], prompt_y)
    pygame.draw.rect(screen, (18, 34, 48), prompt_rect, border_radius=6)
    pygame.draw.rect(screen, accent, prompt_rect, width=1, border_radius=6)
    _text_centered(screen, font, "HIT ANY KEY TO BEGIN", prompt_rect.center, (238, 244, 250))
    _text_centered(screen, small_font, "Esc Quits", (center[0], prompt_rect.bottom + 22), (152, 166, 186))


def _draw_launcher(
    pygame: Any,
    screen: Any,
    *,
    options: tuple[GameScenarioOption, ...],
    selected: int,
    scroll_offset: int,
    selected_difficulty: str,
    music_enabled: bool,
    preview_scroll_px: int,
    record_video: bool,
    font: Any,
    small_font: Any,
    title_font: Any,
) -> None:
    width, height = screen.get_size()
    screen.fill((12, 16, 22))
    _text(screen, title_font, "Orbital Engagement Lab", (54, 36), (238, 242, 248))
    _draw_music_button(pygame, screen, enabled=music_enabled, font=small_font)
    _draw_record_video_button(pygame, screen, enabled=record_video, font=small_font)
    _draw_clear_progress_button(pygame, screen, font=small_font)
    _text(screen, font, "Select RPO Training Level", (56, 78), (172, 186, 206))
    _draw_difficulty_picker(pygame, screen, selected_difficulty=selected_difficulty, font=small_font)

    footer_y = max(height - small_font.get_height() - FOOTER_BOTTOM_MARGIN, PANEL_TOP + 16)
    _text(
        screen,
        small_font,
        "Up/Down Select   Left/Right Difficulty   M Music   V Video   Enter Launch   Esc Quit",
        (56, footer_y),
        (220, 160, 160),
    )

    panel_height = _launcher_panel_height(height)
    list_rect = pygame.Rect(42, PANEL_TOP, 424, panel_height)
    preview_rect = pygame.Rect(*_preview_bounds(width, height))
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
        if _show_progress_text(option):
            _text(
                screen,
                small_font,
                (
                    "Progress: "
                    f"{_progress_stars(option.completed_difficulties)}   "
                    f"High: {_format_high_score(option.high_score)}"
                ),
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
        scroll_px=preview_scroll_px,
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


def _draw_music_button(pygame: Any, screen: Any, *, enabled: bool, font: Any) -> None:
    rect = pygame.Rect(*MUSIC_RECT)
    fill = (34, 60, 76) if enabled else (32, 38, 48)
    stroke = (94, 190, 245) if enabled else (120, 132, 150)
    pygame.draw.rect(screen, fill, rect, border_radius=6)
    pygame.draw.rect(screen, stroke, rect, width=1, border_radius=6)
    label = "Music: ON" if enabled else "Music: OFF"
    _text(screen, font, label, (rect.x + 18, rect.y + 8), (238, 244, 250))


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
    scroll_px: int = 0,
    font: Any,
    small_font: Any,
) -> None:
    pygame.draw.rect(screen, (18, 24, 32), rect, border_radius=8)
    pygame.draw.rect(screen, (70, 82, 100), rect, width=1, border_radius=8)
    content_rect = rect.inflate(-PREVIEW_PADDING * 2, -PREVIEW_PADDING * 2)
    content_height = _preview_content_height(option, font=font, small_font=small_font, width_px=content_rect.width)
    max_scroll = max(int(content_height) - int(content_rect.height), 0)
    scroll = int(max(0, min(int(scroll_px), max_scroll)))
    previous_clip = screen.get_clip()
    screen.set_clip(content_rect.clip(rect))
    y = content_rect.y - scroll
    max_y = content_rect.y + max(content_height, content_rect.height)
    _text(screen, font, _fit_text_px(option.title, font, content_rect.width), (content_rect.x, y), (238, 244, 250))
    y += 34
    budget = _fit_text_px(_budget_line(option), small_font, content_rect.width)
    _text(screen, small_font, budget, (content_rect.x, y), (162, 178, 198))
    y += 32
    high = _format_high_score(option.high_score)
    if _show_progress_text(option) and high != "--":
        _text(screen, small_font, f"High Score: {high}", (content_rect.x, y), (245, 205, 92))
        y += PREVIEW_LINE_HEIGHT + PREVIEW_SECTION_GAP
    y = _draw_section(screen, small_font, "Objective", option.learning_goal, content_rect.x, y, content_rect.width, max_y)
    y = _draw_section(
        screen,
        small_font,
        "Brief",
        option.player_brief or option.description,
        content_rect.x,
        y + PREVIEW_SECTION_GAP,
        content_rect.width,
        max_y,
    )
    y = _draw_bullets(
        screen,
        small_font,
        "Pass Criteria",
        option.pass_criteria,
        content_rect.x,
        y + PREVIEW_SECTION_GAP,
        content_rect.width,
        max_y,
    )
    y = _draw_bullets(
        screen,
        small_font,
        "Instructor Notes",
        option.instructor_notes,
        content_rect.x,
        y + PREVIEW_SECTION_GAP,
        content_rect.width,
        max_y,
    )
    screen.set_clip(previous_clip)
    _draw_preview_scrollbar(pygame, screen, rect, content_height=content_height, scroll_px=scroll)


def _draw_preview_scrollbar(pygame: Any, screen: Any, rect: Any, *, content_height: int, scroll_px: int) -> None:
    viewport_height = max(int(rect.height) - PREVIEW_PADDING * 2, 1)
    if int(content_height) <= viewport_height:
        return
    track = pygame.Rect(rect.right - 12, rect.y + PREVIEW_PADDING, 4, viewport_height)
    pygame.draw.rect(screen, (38, 48, 62), track, border_radius=2)
    frac = min(float(viewport_height) / max(float(content_height), 1.0), 1.0)
    thumb_h = max(int(track.height * frac), 28)
    max_scroll = max(int(content_height) - viewport_height, 1)
    travel = max(track.height - thumb_h, 0)
    thumb_y = track.y + int(round(travel * min(max(float(scroll_px) / float(max_scroll), 0.0), 1.0)))
    thumb = pygame.Rect(track.x, thumb_y, track.width, thumb_h)
    pygame.draw.rect(screen, (96, 174, 224), thumb, border_radius=2)


def _preview_content_height(option: GameScenarioOption, *, font: Any, small_font: Any, width_px: int) -> int:
    y = 0
    y += 34
    y += 32
    if _show_progress_text(option) and option.high_score > 0:
        y += PREVIEW_LINE_HEIGHT + PREVIEW_SECTION_GAP
    y = _section_height(option.learning_goal, small_font, y, width_px)
    y = _section_height(option.player_brief or option.description, small_font, y + PREVIEW_SECTION_GAP, width_px)
    y = _bullets_height(option.pass_criteria, small_font, y + PREVIEW_SECTION_GAP, width_px)
    y = _bullets_height(option.instructor_notes, small_font, y + PREVIEW_SECTION_GAP, width_px)
    return max(y, _text_height(font))


def _section_height(body: str, font: Any, y: int, width_px: int) -> int:
    y += PREVIEW_SECTION_TITLE_GAP
    y += len(_wrap_text_px(body, font, width_px)) * PREVIEW_LINE_HEIGHT
    return y


def _show_progress_text(option: GameScenarioOption) -> bool:
    return str(option.scenario_id) != "rpo_00_tutorial"


def _bullets_height(items: tuple[str, ...], font: Any, y: int, width_px: int) -> int:
    if not items:
        return y
    y += PREVIEW_SECTION_TITLE_GAP
    bullet_width = max(width_px - _text_width(font, "- "), 1)
    for item in items:
        y += len(_wrap_text_px(item, font, bullet_width)) * PREVIEW_LINE_HEIGHT
    return y


def _text(screen: Any, font: Any, text: str, pos: tuple[int, int], color: tuple[int, int, int]) -> None:
    if not text:
        return
    surf = font.render(str(text), True, color)
    screen.blit(surf, pos)


def _text_centered(screen: Any, font: Any, text: str, center: tuple[int, int], color: tuple[int, int, int]) -> None:
    if not text:
        return
    surf = font.render(str(text), True, color)
    screen.blit(surf, (int(center[0] - surf.get_width() // 2), int(center[1] - surf.get_height() // 2)))


def _draw_section(screen: Any, font: Any, title: str, body: str, x: int, y: int, width_px: int, max_y: int) -> int:
    if y + PREVIEW_SECTION_TITLE_GAP > max_y:
        return max_y
    _text(screen, font, title, (x, y), (238, 244, 250))
    y += PREVIEW_SECTION_TITLE_GAP
    lines = _wrap_text_px(body, font, width_px)
    for line in _lines_that_fit(lines, font, width_px, y, max_y):
        _text(screen, font, line, (x, y), (182, 194, 210))
        y += PREVIEW_LINE_HEIGHT
    return y


def _draw_bullets(
    screen: Any,
    font: Any,
    title: str,
    items: tuple[str, ...],
    x: int,
    y: int,
    width_px: int,
    max_y: int,
) -> int:
    if not items:
        return y
    if y + PREVIEW_SECTION_TITLE_GAP > max_y:
        return max_y
    _text(screen, font, title, (x, y), (238, 244, 250))
    y += PREVIEW_SECTION_TITLE_GAP
    for item in items:
        bullet_width = max(width_px - _text_width(font, "- "), 1)
        wrapped = _wrap_text_px(item, font, bullet_width)
        for idx, line in enumerate(wrapped):
            if y + PREVIEW_LINE_HEIGHT > max_y:
                ellipsis_y = y - PREVIEW_LINE_HEIGHT
                if idx > 0 and ellipsis_y >= 0:
                    ellipsis = _fit_text_px("...", font, width_px)
                    _text(screen, font, ellipsis, (x, y - PREVIEW_LINE_HEIGHT), (182, 194, 210))
                return max_y
            prefix = "- " if idx == 0 else "  "
            _text(screen, font, prefix + line, (x, y), (182, 194, 210))
            y += PREVIEW_LINE_HEIGHT
    return y


def _budget_line(option: GameScenarioOption) -> str:
    parts = []
    if option.time_budget_s is not None:
        parts.append(f"Time: {option.time_budget_s:.0f}s")
    if option.delta_v_budget_m_s is not None:
        parts.append(f"Chaser dV: {format_speed_m_s(option.delta_v_budget_m_s)}")
    if option.goal_speed_km_s is not None:
        parts.append(f"Speed Gate: {format_speed_km_s(option.goal_speed_km_s)}")
    if option.target_delta_v_budget_m_s is not None:
        parts.append(f"Target dV: {format_speed_m_s(option.target_delta_v_budget_m_s)}")
    return "   ".join(parts)


def _completed_difficulties_from_game(game: dict[str, Any]) -> tuple[str, ...]:
    progress = dict(game.get("progress", {}) or {})
    completed = progress.get("completed_difficulties", ())
    if isinstance(completed, str):
        completed = (completed,)
    values = {_normalize_difficulty(item) for item in completed}
    return tuple(item for item in DIFFICULTY_OPTIONS if item in values)


def _high_score_from_game(game: dict[str, Any]) -> int:
    progress = dict(game.get("progress", {}) or {})
    return max(int(progress.get("high_score", 0) or 0), 0)


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


def _format_high_score(score: int) -> str:
    value = int(max(score, 0))
    return f"{value:,}" if value > 0 else "--"


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


def _wrap_text_px(value: str, font: Any, width_px: int) -> list[str]:
    words = str(value or "").split()
    if not words:
        return [""]
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else current + " " + word
        if _text_width(font, candidate) <= width_px:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = _fit_text_px(word, font, width_px) if _text_width(font, word) > width_px else word
    if current:
        lines.append(current)
    return lines or [""]


def _lines_that_fit(lines: list[str], font: Any, width_px: int, y: int, max_y: int) -> list[str]:
    if y >= max_y:
        return []
    available = max((max_y - y) // PREVIEW_LINE_HEIGHT, 0)
    if available <= 0:
        return []
    if len(lines) <= available:
        return lines
    kept = lines[:available]
    kept[-1] = _fit_text_px("...", font, width_px)
    return kept


def _fit_text_px(value: str, font: Any, width_px: int) -> str:
    text = " ".join(str(value or "").split())
    if _text_width(font, text) <= width_px:
        return text
    ellipsis = "..."
    if _text_width(font, ellipsis) > width_px:
        return ""
    lo = 0
    hi = len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        candidate = text[:mid].rstrip() + ellipsis
        if _text_width(font, candidate) <= width_px:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo].rstrip() + ellipsis


def _text_width(font: Any, text: str) -> int:
    if hasattr(font, "size"):
        return int(font.size(str(text))[0])
    surf = font.render(str(text), True, (255, 255, 255))
    if hasattr(surf, "get_width"):
        return int(surf.get_width())
    return len(str(text)) * 8


def _text_height(font: Any) -> int:
    if hasattr(font, "get_height"):
        return int(font.get_height())
    if hasattr(font, "size"):
        return int(font.size("Hg")[1])
    return PREVIEW_LINE_HEIGHT


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
