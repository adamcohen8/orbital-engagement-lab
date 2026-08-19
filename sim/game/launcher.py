# ruff: noqa: F401,F821,I001
"""Compatibility façade for scenario discovery and game launch control."""

from . import launcher_common as _common
from . import launcher_models as _models
from . import scenario_catalog as _catalog
from . import launcher_persistence as _persistence
from . import operator_planning as _planning
from . import launcher_widgets as _widgets
from . import launcher_controller as _controller

LAUNCHER_CAPABILITY_FAMILIES = {
    "models": "sim.game.launcher_models",
    "catalog": "sim.game.scenario_catalog",
    "persistence": "sim.game.launcher_persistence",
    "operator_planning": "sim.game.operator_planning",
    "widgets": "sim.game.launcher_widgets",
    "controller": "sim.game.launcher_controller",
}

for _module in (_common, _models, _catalog, _persistence, _planning, _widgets, _controller):
    globals().update({name: value for name, value in vars(_module).items() if not name.startswith("__")})

for _name in (
    "GameScenarioOption", "GameProgressRecord", "GameSettings", "GameLaunchSelection",
    "OperatorPlotContext", "OperatorDisplayState", "OperatorTrajectoryProbe",
):
    globals()[_name].__module__ = __name__
del _module, _name

def discover_game_scenarios(config_dir: Path | None = None) -> tuple[GameScenarioOption, ...]:
    return discover_game_scenarios_for_mode(config_dir, mode="pilot")


def discover_game_scenarios_for_mode(config_dir: Path | None = None, *, mode: str = "pilot") -> tuple[GameScenarioOption, ...]:
    root = Path(config_dir) if config_dir is not None else GAME_CONFIG_DIR
    mode_key = _normalize_game_mode(mode)
    progress = _load_game_progress()
    options: list[GameScenarioOption] = []
    for path in sorted(root.glob("game_training_rpo_*.yaml")):
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        option = _scenario_option_from_yaml(path, raw, progress_by_scenario=progress, mode=mode_key)
        if option.scenario_id in DOWNLOADABLE_GAME_EXCLUDED_SCENARIO_IDS:
            continue
        if mode_key == "operator" and option.scenario_id in OPERATOR_MODE_EXCLUDED_SCENARIO_IDS:
            continue
        options.append(option)
    return tuple(sorted(options, key=_scenario_sort_key))


def choose_game_scenario(config_dir: Path | None = None) -> Path | None:
    selection = choose_game_launch(config_dir)
    return None if selection is None else selection.path


def choose_game_launch(
    config_dir: Path | None = None,
    *,
    show_start_screen: bool = True,
    initial_mode: str = "pilot",
) -> GameLaunchSelection | None:
    settings = _load_game_settings()
    mode_key = _normalize_game_mode(settings.last_game_mode or initial_mode)
    options = discover_game_scenarios_for_mode(config_dir, mode=mode_key)
    if not options:
        raise RuntimeError(f"No game training configs found in {Path(config_dir) if config_dir else GAME_CONFIG_DIR}.")
    return _run_launcher(options, show_start_screen=show_start_screen, initial_mode=mode_key)


def plan_operator_burns_for_config(
    pygame: Any,
    screen: Any,
    clock: Any,
    config_path: str | Path,
    *,
    font: Any,
    small_font: Any,
    title_font: Any,
    initial_plan: OperatorBurnPlan | None = None,
    difficulty: str = "easy",
    frame_convention: FrameConvention | dict[str, Any] | None = None,
    read_only: bool = False,
    demo_title: str = "",
    launch_label: str = "Launch",
    config_override: Any | None = None,
) -> OperatorBurnPlan | None:
    path = Path(config_path)
    if config_override is None:
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    else:
        to_dict = getattr(config_override, "to_dict", None)
        raw = to_dict() if callable(to_dict) else dict(config_override)
    option = _scenario_option_from_yaml(path, raw, mode="operator")
    previous_grab = bool(pygame.event.get_grab())
    previous_visible = bool(pygame.mouse.get_visible())
    pygame.event.set_grab(False)
    pygame.mouse.set_visible(True)
    try:
        return _run_operator_plan_screen(
            pygame,
            screen,
            clock,
            option=option,
            font=font,
            small_font=small_font,
            title_font=title_font,
            initial_plan=initial_plan,
            difficulty=difficulty,
            frame_convention=frame_convention,
            read_only=read_only,
            demo_title=demo_title,
            launch_label=launch_label,
            config_override=config_override,
        )
    finally:
        pygame.event.set_grab(previous_grab)
        pygame.mouse.set_visible(previous_visible)

def _run_launcher(
    options: tuple[GameScenarioOption, ...],
    *,
    show_start_screen: bool = True,
    initial_mode: str = "pilot",
) -> GameLaunchSelection | None:
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
    selected_mode = _normalize_game_mode(initial_mode)
    music_enabled = _start_launcher_music(pygame)
    start_artwork = _load_start_screen_artwork(pygame)
    record_video = False
    settings = _load_game_settings()
    frame_convention = settings.frame_convention
    presentation_mode = settings.presentation_mode
    frame_dialog_open = bool(settings.ask_frame_convention_on_launch)
    frame_dialog_required = frame_dialog_open
    frame_dialog_dont_ask_again = False
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

            if frame_dialog_open:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return None
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            if frame_dialog_required:
                                return None
                            frame_dialog_open = False
                        if event.key in {pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE}:
                            settings = _frame_convention_dialog_settings(
                                settings,
                                frame_convention=frame_convention,
                                presentation_mode=presentation_mode,
                                dont_ask_again=frame_dialog_dont_ask_again,
                                selected_mode=selected_mode,
                            )
                            _save_game_settings(settings)
                            frame_dialog_open = False
                            frame_dialog_required = False
                        elif event.key in {pygame.K_1, pygame.K_o}:
                            frame_convention = frame_convention_from_preset(FRAME_CONVENTION_PRESET_OEL_DEFAULT)
                        elif event.key in {pygame.K_2, pygame.K_s}:
                            frame_convention = frame_convention_from_preset(FRAME_CONVENTION_PRESET_SPACE_FORCE)
                        elif event.key == pygame.K_g:
                            current = PRESENTATION_MODES.index(normalize_presentation_mode(presentation_mode))
                            presentation_mode = PRESENTATION_MODES[(current + 1) % len(PRESENTATION_MODES)]
                        elif event.key == pygame.K_d:
                            frame_dialog_dont_ask_again = not frame_dialog_dont_ask_again
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        action = _frame_convention_dialog_action(
                            event.pos,
                            width=width,
                            height=height,
                        )
                        if action == FRAME_CONVENTION_PRESET_OEL_DEFAULT:
                            frame_convention = frame_convention_from_preset(FRAME_CONVENTION_PRESET_OEL_DEFAULT)
                        elif action == FRAME_CONVENTION_PRESET_SPACE_FORCE:
                            frame_convention = frame_convention_from_preset(FRAME_CONVENTION_PRESET_SPACE_FORCE)
                        elif action in PRESENTATION_MODES:
                            presentation_mode = normalize_presentation_mode(action)
                        elif action == "dont_ask_again":
                            frame_dialog_dont_ask_again = not frame_dialog_dont_ask_again
                        elif action == "continue":
                            settings = _frame_convention_dialog_settings(
                                settings,
                                frame_convention=frame_convention,
                                presentation_mode=presentation_mode,
                                dont_ask_again=frame_dialog_dont_ask_again,
                                selected_mode=selected_mode,
                            )
                            _save_game_settings(settings)
                            frame_dialog_open = False
                            frame_dialog_required = False
                if frame_dialog_open:
                    _draw_launcher(
                        pygame,
                        screen,
                        options=options,
                        selected=selected,
                        scroll_offset=scroll_offset,
                        selected_difficulty=DIFFICULTY_OPTIONS[difficulty_idx],
                        music_enabled=music_enabled,
                        preview_scroll_px=preview_scroll_px,
                        record_video=record_video,
                        selected_mode=selected_mode,
                        font=font,
                        small_font=small_font,
                        title_font=title_font,
                    )
                    _draw_frame_convention_dialog(
                        pygame,
                        screen,
                        convention=frame_convention,
                        presentation_mode=presentation_mode,
                        dont_ask_again=frame_dialog_dont_ask_again,
                        font=font,
                        small_font=small_font,
                        title_font=title_font,
                    )
                    pygame.display.flip()
                    clock.tick(60)
                    continue

            preview_bounds = _preview_bounds(width, height)
            for event in pygame.event.get():
                selected_difficulty = DIFFICULTY_OPTIONS[difficulty_idx]
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.VIDEORESIZE:
                    screen = pygame.display.set_mode((max(event.w, 1040), max(event.h, 680)), pygame.RESIZABLE)
                    continue
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return None
                    if event.key in {pygame.K_DOWN, pygame.K_s}:
                        new_selected = _advance_launcher_selection(selected, 1, count=len(options))
                        if new_selected != selected:
                            preview_scroll_px = 0
                        selected = new_selected
                    elif event.key in {pygame.K_UP, pygame.K_w}:
                        new_selected = _advance_launcher_selection(selected, -1, count=len(options))
                        if new_selected != selected:
                            preview_scroll_px = 0
                        selected = new_selected
                    elif event.key in {pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE}:
                        selection = _selection_for_launch(
                            pygame,
                            screen,
                            clock,
                            option=options[selected],
                            difficulty=selected_difficulty,
                            music_enabled=music_enabled,
                            record_video=record_video,
                            mode=selected_mode,
                            frame_convention=frame_convention,
                            presentation_mode=presentation_mode,
                            font=font,
                            small_font=small_font,
                            title_font=title_font,
                        )
                        if selection is not None:
                            settings = _save_last_game_mode(settings, selected_mode)
                            return selection
                        screen = pygame.display.get_surface() or pygame.display.set_mode((1040, 680), pygame.RESIZABLE)
                    elif event.key == pygame.K_v:
                        record_video = not record_video
                    elif event.key == pygame.K_m:
                        music_enabled = _toggle_launcher_music(pygame, music_enabled=music_enabled)
                    elif event.key == pygame.K_o:
                        selected_mode = _toggle_game_mode(selected_mode)
                        settings = _save_last_game_mode(settings, selected_mode)
                        options = discover_game_scenarios_for_mode(options[0].path.parent, mode=selected_mode)
                        selected = min(selected, len(options) - 1)
                        preview_scroll_px = 0
                        difficulty_idx = _difficulty_index(options[selected].difficulty)
                    elif event.key in {pygame.K_LEFT, pygame.K_a}:
                        difficulty_idx = _advance_launcher_selection(difficulty_idx, -1, count=len(DIFFICULTY_OPTIONS))
                    elif event.key in {pygame.K_RIGHT, pygame.K_d}:
                        difficulty_idx = _advance_launcher_selection(difficulty_idx, 1, count=len(DIFFICULTY_OPTIONS))
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
                            selection = _selection_for_launch(
                                pygame,
                                screen,
                                clock,
                                option=options[selected],
                                difficulty=selected_difficulty,
                                music_enabled=music_enabled,
                                record_video=record_video,
                                mode=selected_mode,
                                frame_convention=frame_convention,
                                presentation_mode=presentation_mode,
                                font=font,
                                small_font=small_font,
                                title_font=title_font,
                            )
                            if selection is not None:
                                settings = _save_last_game_mode(settings, selected_mode)
                                return selection
                            screen = pygame.display.get_surface() or pygame.display.set_mode((1040, 680), pygame.RESIZABLE)
                        preview_scroll_px = 0
                        selected = idx
                    else:
                        mouse_pos = pygame.mouse.get_pos()
                        if _mode_toggle_at_pos(mouse_pos, width=width, height=height):
                            selected_mode = _toggle_game_mode(selected_mode)
                            settings = _save_last_game_mode(settings, selected_mode)
                            options = discover_game_scenarios_for_mode(options[0].path.parent, mode=selected_mode)
                            selected = min(selected, len(options) - 1)
                            preview_scroll_px = 0
                            difficulty_idx = _difficulty_index(options[selected].difficulty)
                            continue
                        if _settings_button_at_pos(mouse_pos, width=width, height=height):
                            frame_dialog_open = True
                            frame_dialog_required = False
                            frame_dialog_dont_ask_again = not bool(settings.ask_frame_convention_on_launch)
                            continue
                        if _record_video_at_pos(mouse_pos):
                            record_video = not record_video
                            continue
                        if _music_at_pos(mouse_pos):
                            music_enabled = _toggle_launcher_music(pygame, music_enabled=music_enabled)
                            continue
                        if _clear_progress_at_pos(mouse_pos):
                            clear_game_progress(options[0].path.parent)
                            options = discover_game_scenarios_for_mode(options[0].path.parent, mode=selected_mode)
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
                selected_mode=selected_mode,
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
