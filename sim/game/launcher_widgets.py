# ruff: noqa: F401,F403,F405,I001
from .launcher_common import *
from .launcher_models import *
from .scenario_catalog import *
from .launcher_persistence import *
from .operator_planning import *

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


def _draw_frame_convention_dialog(
    pygame: Any,
    screen: Any,
    *,
    convention: FrameConvention,
    presentation_mode: str,
    dont_ask_again: bool,
    font: Any,
    small_font: Any,
    title_font: Any,
) -> None:
    width, height = screen.get_size()
    graphics_font = game_font(pygame, 13)
    overlay = pygame.Surface((width, height), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 150))
    screen.blit(overlay, (0, 0))
    rect = pygame.Rect(*_frame_convention_dialog_rect(width, height))
    pygame.draw.rect(screen, (18, 24, 32), rect, border_radius=8)
    pygame.draw.rect(screen, (238, 184, 92), rect, width=1, border_radius=8)
    _text(screen, title_font, "RPO Trainer Settings", (rect.x + 34, rect.y + 28), (238, 242, 248))
    _text(
        screen,
        small_font,
        "Choose the RIC convention and graphics mode for this computer.",
        (rect.x + 36, rect.y + 78),
        (162, 178, 198),
    )

    choices = _frame_convention_dialog_choice_rects(width, height)
    preset = frame_convention_preset(convention)
    _draw_frame_convention_choice(
        pygame,
        screen,
        pygame.Rect(*choices[FRAME_CONVENTION_PRESET_OEL_DEFAULT]),
        "OEL Default",
        "Positive in-track points right; cross-track follows the OEL classroom display.",
        selected=preset == FRAME_CONVENTION_PRESET_OEL_DEFAULT,
        font=font,
        small_font=small_font,
    )

    _draw_frame_convention_choice(
        pygame,
        screen,
        pygame.Rect(*choices[FRAME_CONVENTION_PRESET_SPACE_FORCE]),
        "Space Force",
        "Positive in-track points left; cross-track is provisional until confirmed.",
        selected=preset == FRAME_CONVENTION_PRESET_SPACE_FORCE,
        font=font,
        small_font=small_font,
    )

    _text(screen, font, "Graphics:", (rect.x + 42, rect.y + 322), (172, 186, 206))
    selected_graphics = normalize_presentation_mode(presentation_mode)
    for mode, bounds in _frame_convention_dialog_graphics_rects(width, height).items():
        choice = pygame.Rect(*bounds)
        selected = mode == selected_graphics
        fill = (36, 72, 52) if selected else (12, 16, 22)
        stroke = (108, 232, 142) if selected else (70, 82, 100)
        pygame.draw.rect(screen, fill, choice, border_radius=6)
        pygame.draw.rect(screen, stroke, choice, width=1, border_radius=6)
        _text_centered(screen, graphics_font, GRAPHICS_MODE_LABELS[mode], choice.center, (230, 238, 245))

    checkbox = pygame.Rect(*_frame_convention_dialog_checkbox_rect(width, height))
    pygame.draw.rect(screen, (12, 16, 22), checkbox, border_radius=4)
    pygame.draw.rect(screen, (238, 184, 92), checkbox, width=1, border_radius=4)
    if dont_ask_again:
        pygame.draw.line(screen, (238, 184, 92), (checkbox.x + 5, checkbox.centery), (checkbox.centerx - 1, checkbox.bottom - 6), width=2)
        pygame.draw.line(screen, (238, 184, 92), (checkbox.centerx - 1, checkbox.bottom - 6), (checkbox.right - 5, checkbox.y + 5), width=2)
    _text(screen, font, "Don't ask again", (checkbox.right + 10, checkbox.y + 2), (230, 238, 245))

    _draw_dialog_button(
        pygame,
        screen,
        pygame.Rect(*_frame_convention_dialog_continue_rect(width, height)),
        "Continue",
        font=font,
        enabled=True,
        primary=True,
    )


def _draw_frame_convention_choice(
    pygame: Any,
    screen: Any,
    rect: Any,
    label: str,
    detail: str,
    *,
    selected: bool,
    font: Any,
    small_font: Any,
) -> None:
    fill = (62, 48, 28) if selected else (12, 16, 22)
    stroke = (238, 184, 92) if selected else (70, 82, 100)
    pygame.draw.rect(screen, fill, rect, border_radius=6)
    pygame.draw.rect(screen, stroke, rect, width=1, border_radius=6)
    _text(screen, font, label, (rect.x + 18, rect.y + 8), (246, 240, 226) if selected else (230, 238, 245))
    detail_color = (214, 202, 176) if selected else (156, 170, 190)
    detail_lines = _wrap_text_px(detail, small_font, max(int(rect.width) - 36, 40))[:2]
    for idx, line in enumerate(detail_lines):
        _text(screen, small_font, line, (rect.x + 18, rect.y + 32 + idx * 18), detail_color)


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
    selected_mode: str,
    font: Any,
    small_font: Any,
    title_font: Any,
) -> None:
    width, height = screen.get_size()
    screen.fill((12, 16, 22))
    _text(screen, title_font, "Orbital Engagement Lab", (54, 36), (238, 242, 248))
    _draw_music_button(pygame, screen, enabled=music_enabled, mode=selected_mode, font=small_font)
    _draw_record_video_button(pygame, screen, enabled=record_video, mode=selected_mode, font=small_font)
    _draw_clear_progress_button(pygame, screen, mode=selected_mode, font=small_font)
    _draw_mode_toggle(pygame, screen, mode=selected_mode, font=small_font)
    _draw_settings_button(pygame, screen, mode=selected_mode)
    _text(screen, font, "Select RPO Training Level", (56, 78), (172, 186, 206))
    _draw_difficulty_picker(
        pygame,
        screen,
        selected_difficulty=selected_difficulty,
        selected_mode=selected_mode,
        font=small_font,
    )

    footer_y = max(height - small_font.get_height() - FOOTER_BOTTOM_MARGIN, PANEL_TOP + 16)
    mode_rect = _mode_toggle_rect(width, height)
    footer_text = "Up/Down Select   Left/Right Difficulty   O Mode   M Music   V Video   Enter Launch   Esc Quit"
    footer_max_width = max(mode_rect[0] - 72, 120)
    _text(
        screen,
        small_font,
        _fit_text_px(footer_text, small_font, footer_max_width),
        (56, footer_y),
        (220, 160, 160),
    )

    panel_height = _launcher_panel_height(height)
    list_rect = pygame.Rect(42, PANEL_TOP, 424, panel_height)
    preview_rect = pygame.Rect(*_preview_bounds(width, height))
    pygame.draw.rect(screen, (18, 24, 32), list_rect, border_radius=8)
    pygame.draw.rect(screen, (70, 82, 100), list_rect, width=1, border_radius=8)

    visible = _visible_option_count(height)
    operator_mode = _normalize_game_mode(selected_mode) == "operator"
    for row, option in enumerate(options[scroll_offset : scroll_offset + visible]):
        idx = scroll_offset + row
        y = OPTION_Y + row * OPTION_ROW_HEIGHT
        rect = pygame.Rect(OPTION_X, y, OPTION_WIDTH, OPTION_HEIGHT)
        is_selected = idx == selected
        if operator_mode:
            fill = (62, 48, 28) if is_selected else (20, 27, 36)
            stroke = (238, 184, 92) if is_selected else (62, 50, 32)
        else:
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
        _draw_scrollbar(
            pygame,
            screen,
            list_rect,
            count=len(options),
            visible=visible,
            scroll_offset=scroll_offset,
            mode=selected_mode,
        )

    _draw_preview(
        pygame,
        screen,
        preview_rect,
        option=options[selected],
        scroll_px=preview_scroll_px,
        mode=selected_mode,
        font=font,
        small_font=small_font,
    )


def _draw_scrollbar(
    pygame: Any,
    screen: Any,
    rect: Any,
    *,
    count: int,
    visible: int,
    scroll_offset: int,
    mode: str = "pilot",
) -> None:
    track = pygame.Rect(rect.right - 12, rect.y + 12, 4, max(rect.height - 24, 20))
    pygame.draw.rect(screen, (38, 48, 62), track, border_radius=2)
    frac = min(float(visible) / max(float(count), 1.0), 1.0)
    thumb_h = max(int(track.height * frac), 28)
    max_scroll = max(int(count) - int(visible), 1)
    travel = max(track.height - thumb_h, 0)
    thumb_y = track.y + int(round(travel * min(max(float(scroll_offset) / float(max_scroll), 0.0), 1.0)))
    thumb = pygame.Rect(track.x, thumb_y, track.width, thumb_h)
    thumb_color = (238, 184, 92) if _normalize_game_mode(mode) == "operator" else (96, 174, 224)
    pygame.draw.rect(screen, thumb_color, thumb, border_radius=2)


def _draw_operator_plan_screen(
    pygame: Any,
    screen: Any,
    *,
    option: GameScenarioOption,
    plan: OperatorBurnPlan,
    rows: list[list[str]],
    active_cell: tuple[int, int],
    field_rects: list[list[Any]],
    delete_rects: list[Any],
    table_scroll_row: int,
    plot_context: OperatorPlotContext,
    trajectory_probe: OperatorTrajectoryProbe | None,
    frame_convention: FrameConvention,
    validation_message: str,
    can_launch: bool,
    launch_rect: Any,
    cancel_rect: Any,
    add_burn_rect: Any,
    objectives_rect: Any,
    equation_sheet_rect: Any,
    equation_sheet_visible: bool,
    equation_sheet_scroll_px: int,
    objectives_visible: bool,
    objectives_scroll_px: int,
    read_only: bool,
    demo_title: str,
    launch_label: str,
    font: Any,
    small_font: Any,
    title_font: Any,
) -> None:
    width, height = screen.get_size()
    screen.fill((12, 16, 22))
    mode_title = "Operator Mode"
    _text(screen, title_font, mode_title, (54, 34), (238, 242, 248))
    title_x = 54 + _text_width(title_font, mode_title) + 34
    _text(
        screen,
        title_font,
        _fit_text_px(demo_title or option.title, title_font, max(width - title_x - 54, 120)),
        (title_x, 34),
        (238, 242, 248),
    )

    table = pygame.Rect(*_operator_burn_table_rect(width, height))
    plot_left, plot_right = _operator_game_plot_panel_rects(width, height)
    ri_rect = pygame.Rect(plot_left.x, plot_left.y, plot_left.width, plot_left.height)
    rc_rect = pygame.Rect(plot_right.x, plot_right.y, plot_right.width, plot_right.height)
    _draw_operator_initial_plot(
        pygame,
        screen,
        ri_rect,
        title="Initial RI",
        plot_context=plot_context,
        plan=plan,
        trajectory_probe=trajectory_probe,
        x_axis=1,
        y_axis=0,
        frame_convention=frame_convention,
        font=font,
        small_font=small_font,
    )
    _draw_operator_initial_plot(
        pygame,
        screen,
        rc_rect,
        title="Initial RC",
        plot_context=plot_context,
        plan=plan,
        trajectory_probe=trajectory_probe,
        x_axis=2,
        y_axis=0,
        frame_convention=frame_convention,
        font=font,
        small_font=small_font,
    )
    _draw_dialog_button(
        pygame,
        screen,
        objectives_rect,
        "Hide Mission Brief" if objectives_visible else "Show Mission Brief",
        font=small_font,
        enabled=True,
        primary=False,
    )
    _draw_dialog_button(
        pygame,
        screen,
        equation_sheet_rect,
        "Hide Equation Sheet" if equation_sheet_visible else "Show Equation Sheet",
        font=small_font,
        enabled=True,
        primary=False,
    )
    if objectives_visible:
        _draw_operator_objectives_overlay(
            pygame,
            screen,
            pygame.Rect(*_operator_objectives_overlay_rect(width, height)),
            option=option,
            training_config=plot_context.training_config,
            scroll_px=objectives_scroll_px,
            font=font,
            small_font=small_font,
        )
    if equation_sheet_visible:
        _draw_operator_equation_sheet_overlay(
            pygame,
            screen,
            pygame.Rect(*_operator_objectives_overlay_rect(width, height)),
            scroll_px=equation_sheet_scroll_px,
            font=font,
            small_font=small_font,
        )

    status_color = (162, 232, 174) if can_launch else (245, 126, 126)
    _draw_operator_burn_table(
        pygame,
        screen,
        table,
        rows=rows,
        active_cell=active_cell,
        field_rects=field_rects,
        delete_rects=delete_rects,
        add_burn_rect=add_burn_rect,
        add_enabled=len(rows) < OPERATOR_BURN_MAX_ROWS and not read_only,
        read_only=read_only,
        scroll_row=table_scroll_row,
        validation_message=validation_message,
        validation_color=status_color,
        font=small_font,
    )

    _draw_dialog_button(pygame, screen, cancel_rect, "Cancel", font=small_font, enabled=True, primary=False)
    _draw_dialog_button(pygame, screen, launch_rect, launch_label, font=small_font, enabled=can_launch, primary=True)


def _draw_sandbox_setup_screen(
    pygame: Any,
    screen: Any,
    *,
    level_title: str,
    values: list[str],
    active_index: int,
    panels: tuple[Any, Any],
    field_rects: list[Any],
    launch_rect: Any,
    cancel_rect: Any,
    validation_message: str,
    can_launch: bool,
    font: Any,
    small_font: Any,
    title_font: Any,
) -> None:
    from sim.game.runner_models import _SANDBOX_CHASER_RIC_FIELDS, _SANDBOX_TARGET_COE_FIELDS

    width, _height = screen.get_size()
    screen.fill((12, 16, 22))
    mode_title = "Sandbox Setup"
    _text(screen, title_font, mode_title, (54, 30), (238, 242, 248))
    title_x = 54 + _text_width(title_font, mode_title) + 34
    _text(
        screen,
        title_font,
        _fit_text_px(level_title, title_font, max(width - title_x - 54, 120)),
        (title_x, 30),
        (238, 242, 248),
    )
    _text(
        screen,
        small_font,
        "Define the target orbit and the chaser's target-centered rectangular RIC state.",
        (56, 76),
        (162, 178, 198),
    )
    panel_specs = (
        (panels[0], "Target Classical Orbital Elements", "Two-body target orbit at scenario epoch"),
        (panels[1], "Chaser Relative RIC State", "Position in km; relative velocity in m/s"),
    )
    all_specs = (_SANDBOX_TARGET_COE_FIELDS, _SANDBOX_CHASER_RIC_FIELDS)
    for column, (panel, heading, detail) in enumerate(panel_specs):
        pygame.draw.rect(screen, (18, 24, 32), panel, border_radius=8)
        pygame.draw.rect(screen, (70, 82, 100), panel, width=1, border_radius=8)
        _text(screen, font, heading, (panel.x + 18, panel.y + 14), (238, 244, 250))
        _text(screen, small_font, detail, (panel.x + 18, panel.y + 37), (138, 154, 176))
        for row, (label, unit, _field) in enumerate(all_specs[column]):
            index = column * 6 + row
            rect = field_rects[index]
            label_text = f"{label} ({unit})" if unit else label
            _text(
                screen,
                small_font,
                _fit_text_px(label_text, small_font, max(rect.x - panel.x - 28, 70)),
                (panel.x + 18, rect.y + 8),
                (190, 202, 218),
            )
            active = index == int(active_index)
            fill = (42, 50, 62) if active else (12, 16, 22)
            stroke = (238, 184, 92) if active else (74, 88, 106)
            pygame.draw.rect(screen, fill, rect, border_radius=5)
            pygame.draw.rect(screen, stroke, rect, width=2 if active else 1, border_radius=5)
            value = values[index] if index < len(values) else ""
            _text(
                screen,
                small_font,
                _fit_text_px(value, small_font, max(rect.width - 18, 30)),
                (rect.x + 9, rect.y + 8),
                (246, 240, 226) if active else (230, 238, 245),
            )
    message = validation_message or "Tab changes fields. Arrow keys move within or between columns."
    message_color = (245, 126, 126) if validation_message else (138, 154, 176)
    _text(screen, small_font, message, (56, launch_rect.y + 8), message_color)
    _draw_dialog_button(pygame, screen, cancel_rect, "Cancel", font=small_font, enabled=True, primary=False)
    _draw_dialog_button(pygame, screen, launch_rect, "Continue", font=small_font, enabled=can_launch, primary=True)


def _draw_operator_objectives_overlay(
    pygame: Any,
    screen: Any,
    rect: Any,
    *,
    option: GameScenarioOption,
    training_config: RPOTrainingConfig | None,
    scroll_px: int,
    font: Any,
    small_font: Any,
) -> None:
    pygame.draw.rect(screen, (14, 19, 27), rect, border_radius=8)
    pygame.draw.rect(screen, (238, 184, 92), rect, width=1, border_radius=8)
    inset = 22
    x = int(rect.x) + inset
    _text(screen, font, "Mission Brief", (x, int(rect.y) + 18), (238, 244, 250))
    content_tuple = _operator_objectives_content_rect(rect)
    content = pygame.Rect(*content_tuple)
    width_px = max(content.width, 60)
    content_height = _operator_objectives_content_height(
        option,
        training_config,
        font=small_font,
        width_px=width_px,
    )
    scroll_px = _clamp_operator_objectives_scroll_px(
        scroll_px,
        content_height=content_height,
        viewport_height=content.height,
    )
    previous_clip = screen.get_clip()
    screen.set_clip(content)
    y = int(content.y) - int(scroll_px)
    max_y = y + int(content_height) + PREVIEW_LINE_HEIGHT
    for line in _wrapped_budget_lines(option, small_font, width_px):
        if y + PREVIEW_LINE_HEIGHT > max_y:
            break
        _text(screen, small_font, line, (x, y), (162, 178, 198))
        y += PREVIEW_LINE_HEIGHT
    if y > int(content.y) - int(scroll_px):
        y += PREVIEW_SECTION_GAP
    y = _draw_section(screen, small_font, "Objective", option.learning_goal, x, y, width_px, max_y)
    y = _draw_section(
        screen,
        small_font,
        "Brief",
        option.player_brief or option.description,
        x,
        y + PREVIEW_SECTION_GAP,
        width_px,
        max_y,
    )
    y = _draw_bullets(
        screen,
        small_font,
        "Pass Criteria",
        option.pass_criteria,
        x,
        y + PREVIEW_SECTION_GAP,
        width_px,
        max_y,
    )
    y = _draw_bullets(
        screen,
        small_font,
        "Instructor Notes",
        option.instructor_notes,
        x,
        y + PREVIEW_SECTION_GAP,
        width_px,
        max_y,
    )
    numeric_targets = _operator_objective_numeric_targets(option, training_config)
    if numeric_targets:
        y += PREVIEW_SECTION_GAP
        y = _draw_bullets(
            screen,
            small_font,
            "Numeric Targets",
            numeric_targets,
            x,
            y,
            width_px,
            max_y,
        )
    screen.set_clip(previous_clip)
    _draw_operator_objectives_scrollbar(
        pygame,
        screen,
        content,
        content_height=content_height,
        scroll_px=scroll_px,
    )


def _draw_operator_equation_sheet_overlay(
    pygame: Any,
    screen: Any,
    rect: Any,
    *,
    scroll_px: int,
    font: Any,
    small_font: Any,
) -> None:
    pygame.draw.rect(screen, (14, 19, 27), rect, border_radius=8)
    pygame.draw.rect(screen, (238, 184, 92), rect, width=1, border_radius=8)
    inset = 22
    x = int(rect.x) + inset
    _text(screen, font, "Equation Sheet", (x, int(rect.y) + 18), (238, 244, 250))
    content_tuple = _operator_objectives_content_rect(rect)
    content = pygame.Rect(*content_tuple)
    width_px = max(content.width, 60)
    content_height = _operator_equation_sheet_content_height(small_font, width_px=width_px)
    scroll_px = _clamp_operator_objectives_scroll_px(
        scroll_px,
        content_height=content_height,
        viewport_height=content.height,
    )
    previous_clip = screen.get_clip()
    screen.set_clip(content)
    y = int(content.y) - int(scroll_px)
    max_y = y + int(content_height) + PREVIEW_LINE_HEIGHT
    text_width = _operator_equation_sheet_text_width(width_px)
    y = _draw_section(
        screen,
        small_font,
        "RIC Motion Card",
        "Circular-chief HCW intuition for the local RIC frame.",
        x,
        y,
        text_width,
        max_y,
    )
    y = _draw_operator_lines_section(
        screen,
        small_font,
        "HCW Equations",
        PygameRPODashboard._pause_overlay_equation_lines(),
        x,
        y + PREVIEW_SECTION_GAP,
        text_width,
        max_y,
    )
    y = _draw_operator_lines_section(
        screen,
        small_font,
        "Useful Intuition",
        PygameRPODashboard._pause_overlay_takeaway_lines(),
        x,
        y + PREVIEW_SECTION_GAP,
        text_width,
        max_y,
    )
    screen.set_clip(previous_clip)
    _draw_operator_equation_sheet_ric_diagram(
        pygame,
        screen,
        content,
        font=font,
        small_font=small_font,
    )
    _draw_operator_objectives_scrollbar(
        pygame,
        screen,
        content,
        content_height=content_height,
        scroll_px=scroll_px,
    )


def _draw_operator_equation_sheet_ric_diagram(
    pygame: Any,
    screen: Any,
    content: Any,
    *,
    font: Any,
    small_font: Any,
) -> None:
    width_px = int(content.width)
    if width_px < 760:
        return
    column_gap = 28
    text_width = _operator_equation_sheet_text_width(width_px)
    diagram_x = int(content.x) + text_width + column_gap
    diagram = pygame.Rect(
        diagram_x,
        int(content.y),
        max(int(content.right) - diagram_x, 220),
        int(content.height),
    )
    dashboard = _new_operator_preview_dashboard(pygame, screen, font=font, small_font=small_font)
    dashboard._target_sprite = dashboard._load_marker_sprite(TARGET_SPRITE_PATH)
    dashboard._chaser_sprite = dashboard._load_marker_sprite(CHASER_SPRITE_PATH)
    dashboard._draw_pause_ric_diagram(diagram)


def _draw_operator_lines_section(
    screen: Any,
    font: Any,
    title: str,
    items: tuple[str, ...],
    x: int,
    y: int,
    width_px: int,
    max_y: int,
) -> int:
    if y + PREVIEW_SECTION_TITLE_GAP > max_y:
        return max_y
    _text(screen, font, title, (x, y), (238, 244, 250))
    y += PREVIEW_SECTION_TITLE_GAP
    for item in items:
        for line in _wrap_text_px(item, font, width_px):
            if y + PREVIEW_LINE_HEIGHT > max_y:
                return max_y
            _text(screen, font, line, (x, y), (182, 194, 210))
            y += PREVIEW_LINE_HEIGHT
    return y


def _draw_operator_objectives_scrollbar(
    pygame: Any,
    screen: Any,
    content_rect: Any,
    *,
    content_height: int,
    scroll_px: int,
) -> None:
    viewport_height = max(int(content_rect.height), 1)
    if int(content_height) <= viewport_height:
        return
    track = pygame.Rect(content_rect.right + 8, content_rect.y, 4, viewport_height)
    pygame.draw.rect(screen, (38, 48, 62), track, border_radius=2)
    frac = min(float(viewport_height) / max(float(content_height), 1.0), 1.0)
    thumb_h = max(int(track.height * frac), 28)
    max_scroll = max(int(content_height) - viewport_height, 1)
    travel = max(track.height - thumb_h, 0)
    thumb_y = track.y + int(round(travel * min(max(float(scroll_px) / float(max_scroll), 0.0), 1.0)))
    thumb = pygame.Rect(track.x, thumb_y, track.width, thumb_h)
    pygame.draw.rect(screen, (238, 184, 92), thumb, border_radius=2)


def _draw_operator_prebrief_screen(
    pygame: Any,
    screen: Any,
    *,
    option: GameScenarioOption,
    scroll_px: int,
    continue_rect: Any,
    cancel_rect: Any,
    font: Any,
    small_font: Any,
    title_font: Any,
) -> None:
    width, height = screen.get_size()
    screen.fill((12, 16, 22))
    title = "Operator Prebrief"
    _text(screen, title_font, title, (54, 34), (238, 242, 248))
    title_x = 54 + _text_width(title_font, title) + 34
    _text(
        screen,
        title_font,
        _fit_text_px(option.title, title_font, max(width - title_x - 54, 120)),
        (title_x, 34),
        (238, 242, 248),
    )

    panel = pygame.Rect(42, 104, max(width - 84, 620), max(height - 200, 360))
    pygame.draw.rect(screen, (18, 24, 32), panel, border_radius=8)
    pygame.draw.rect(screen, (70, 82, 100), panel, width=1, border_radius=8)
    content_tuple = _operator_prebrief_content_rect(width, height)
    content = pygame.Rect(*content_tuple)
    previous_clip = screen.get_clip()
    screen.set_clip(content)
    y = content.y - int(scroll_px)
    max_y = content.y + _operator_prebrief_content_height(option, font=font, small_font=small_font, width_px=content.width)
    _text(screen, font, "Mission Brief", (content.x, y), (238, 244, 250))
    y += 34
    for line in _wrapped_budget_lines(option, small_font, content.width):
        _text(screen, small_font, line, (content.x, y), (162, 178, 198))
        y += PREVIEW_LINE_HEIGHT
    y += PREVIEW_SECTION_GAP
    y = _draw_section(screen, small_font, "Objective", option.learning_goal, content.x, y, content.width, max_y)
    y = _draw_section(
        screen,
        small_font,
        "Brief",
        option.player_brief or option.description,
        content.x,
        y + PREVIEW_SECTION_GAP,
        content.width,
        max_y,
    )
    y = _draw_bullets(
        screen,
        small_font,
        "Pass Criteria",
        option.pass_criteria,
        content.x,
        y + PREVIEW_SECTION_GAP,
        content.width,
        max_y,
    )
    _draw_bullets(
        screen,
        small_font,
        "Instructor Notes",
        option.instructor_notes,
        content.x,
        y + PREVIEW_SECTION_GAP,
        content.width,
        max_y,
    )
    screen.set_clip(previous_clip)
    _draw_preview_scrollbar(
        pygame,
        screen,
        content.inflate(PREVIEW_PADDING * 2, PREVIEW_PADDING * 2),
        content_height=_operator_prebrief_content_height(option, font=font, small_font=small_font, width_px=content.width),
        scroll_px=scroll_px,
    )
    _draw_dialog_button(pygame, screen, cancel_rect, "Cancel", font=small_font, enabled=True, primary=False)
    _draw_dialog_button(pygame, screen, continue_rect, "Script Burns", font=small_font, enabled=True, primary=True)


def _draw_dialog_button(
    pygame: Any,
    screen: Any,
    rect: Any,
    label: str,
    *,
    font: Any,
    enabled: bool,
    primary: bool,
) -> None:
    if primary and enabled:
        fill = (36, 72, 52)
        stroke = (108, 232, 142)
    elif primary:
        fill = (38, 42, 48)
        stroke = (76, 84, 96)
    else:
        fill = (32, 38, 48)
        stroke = (120, 132, 150)
    pygame.draw.rect(screen, fill, rect, border_radius=6)
    pygame.draw.rect(screen, stroke, rect, width=1, border_radius=6)
    color = (230, 238, 245) if enabled else (126, 136, 150)
    _text_centered(screen, font, label, rect.center, color)


def _draw_operator_burn_table(
    pygame: Any,
    screen: Any,
    table: Any,
    *,
    rows: list[list[str]],
    active_cell: tuple[int, int],
    field_rects: list[list[Any]],
    delete_rects: list[Any],
    add_burn_rect: Any,
    add_enabled: bool,
    scroll_row: int,
    validation_message: str,
    validation_color: tuple[int, int, int],
    font: Any,
    read_only: bool = False,
) -> None:
    pygame.draw.rect(screen, (18, 24, 32), table, border_radius=6)
    pygame.draw.rect(screen, (70, 82, 100), table, width=1, border_radius=6)
    _text(screen, font, "Script impulsive RIC burns", (table.x + 12, table.y + 8), (238, 244, 250))
    _draw_operator_icon_button(pygame, screen, add_burn_rect, "+", font=font, enabled=add_enabled)
    for col_idx, header in enumerate(OPERATOR_BURN_HEADERS):
        if not field_rects:
            continue
        rect = field_rects[0][col_idx]
        _text(screen, font, header, (rect.x + 4, table.y + 32), (162, 178, 198))
    body_clip = pygame.Rect(table.x + 1, table.y + 52, table.width - 14, max(table.height - 74, 1))
    previous_clip = screen.get_clip()
    screen.set_clip(body_clip)
    for row_idx, row in enumerate(rows):
        if row_idx >= len(field_rects):
            break
        first_rect = field_rects[row_idx][0]
        if first_rect.bottom < body_clip.top or first_rect.top > body_clip.bottom:
            continue
        _text(screen, font, f"{row_idx + 1}", (table.x + 4, field_rects[row_idx][0].y + 7), (126, 138, 154))
        for col_idx, rect in enumerate(field_rects[row_idx]):
            active = (row_idx, col_idx) == active_cell
            fill = (22, 34, 46) if active else (18, 24, 32)
            stroke = (238, 184, 92) if active else (82, 96, 116)
            pygame.draw.rect(screen, fill, rect, border_radius=4)
            pygame.draw.rect(screen, stroke, rect, width=1, border_radius=4)
            value = str(row[col_idx] if col_idx < len(row) else "")
            _text(screen, font, _fit_text_px(value, font, rect.width - 12), (rect.x + 6, rect.y + 7), (230, 238, 245))
        if row_idx < len(delete_rects):
            _draw_operator_icon_button(pygame, screen, delete_rects[row_idx], "X", font=font, enabled=not read_only)
    screen.set_clip(previous_clip)
    _text(
        screen,
        font,
        _fit_text_px(validation_message, font, table.width - 28),
        (table.x + 12, table.bottom - 20),
        validation_color,
    )
    _draw_operator_table_scrollbar(pygame, screen, table, row_count=len(rows), scroll_row=scroll_row)


def _draw_operator_icon_button(
    pygame: Any,
    screen: Any,
    rect: Any,
    label: str,
    *,
    font: Any,
    enabled: bool,
) -> None:
    fill = (32, 38, 48) if enabled else (24, 28, 34)
    stroke = (120, 132, 150) if enabled else (72, 82, 96)
    pygame.draw.rect(screen, fill, rect, border_radius=4)
    pygame.draw.rect(screen, stroke, rect, width=1, border_radius=4)
    color = (230, 238, 245) if enabled else (126, 136, 150)
    _text_centered(screen, font, str(label), rect.center, color)


def _draw_operator_table_scrollbar(
    pygame: Any,
    screen: Any,
    table: Any,
    *,
    row_count: int,
    scroll_row: int,
) -> None:
    visible = _operator_table_visible_rows(table)
    if int(row_count) <= visible:
        return
    track = pygame.Rect(table.right - 9, table.y + 54, 4, max(table.height - 78, 1))
    pygame.draw.rect(screen, (38, 48, 62), track, border_radius=2)
    frac = min(float(visible) / max(float(row_count), 1.0), 1.0)
    thumb_h = max(int(track.height * frac), 18)
    max_scroll = max(int(row_count) - int(visible), 1)
    travel = max(track.height - thumb_h, 0)
    thumb_y = track.y + int(round(travel * min(max(float(scroll_row) / float(max_scroll), 0.0), 1.0)))
    pygame.draw.rect(screen, (96, 174, 224), pygame.Rect(track.x, thumb_y, track.width, thumb_h), border_radius=2)


def _draw_operator_pilot_first_frame_plot(
    pygame: Any,
    screen: Any,
    rect: Any,
    *,
    title: str,
    plot_context: OperatorPlotContext,
    plan: OperatorBurnPlan,
    trajectory_probe: OperatorTrajectoryProbe | None,
    x_axis: int,
    y_axis: int,
    frame_convention: FrameConvention,
    font: Any,
    small_font: Any,
) -> bool:
    dashboard = _operator_preview_dashboard(
        pygame,
        screen,
        plot_context=plot_context,
        frame_convention=frame_convention,
        font=font,
        small_font=small_font,
    )
    if dashboard is None:
        return False
    dashboard._draw_panel(rect, title, x_axis=int(x_axis), y_axis=int(y_axis))
    trajectory, markers = _operator_planned_trajectory(plot_context, plan)
    _draw_operator_planned_trajectory_overlay(
        pygame,
        screen,
        dashboard,
        trajectory=trajectory,
        markers=markers,
        selected_state_ric_km_s=(
            np.asarray(trajectory_probe.state_ric_km_s, dtype=float) if trajectory_probe is not None else None
        ),
        selected_time_s=trajectory_probe.time_s if trajectory_probe is not None else None,
        x_axis=int(x_axis),
        y_axis=int(y_axis),
        font=small_font,
    )
    rel = np.asarray(dashboard._frame_cache.get("rel", np.empty((0, 6))), dtype=float)
    if rel.ndim == 2 and rel.shape[0] > 0 and rel.shape[1] >= 6:
        readout_state = (
            np.asarray(trajectory_probe.state_ric_km_s, dtype=float).reshape(6)
            if trajectory_probe is not None
            else rel[-1]
        )
        _draw_operator_initial_state_readout(
            screen,
            rect,
            rel_state_ric_km_s=readout_state,
            x_axis=x_axis,
            y_axis=y_axis,
            font=font,
            color=(86, 202, 245) if trajectory_probe is not None else (162, 178, 198),
        )
    return True


def _operator_preview_dashboard(
    pygame: Any,
    screen: Any,
    *,
    plot_context: OperatorPlotContext,
    frame_convention: FrameConvention,
    font: Any,
    small_font: Any,
) -> Any | None:
    snapshot = plot_context.pilot_initial_snapshot
    if snapshot is None or not plot_context.pilot_dashboard_kwargs:
        return None
    dashboard = plot_context._preview_dashboard
    if dashboard is None:
        dashboard = _new_operator_preview_dashboard(pygame, screen, font=font, small_font=small_font)
        for key, value in plot_context.pilot_dashboard_kwargs.items():
            setattr(dashboard, key, value)
        dashboard._target_sprite = dashboard._load_marker_sprite(
            _game_asset_path_or_default(getattr(dashboard, "target_sprite_path", None), TARGET_SPRITE_PATH)
        )
        dashboard._chaser_sprite = dashboard._load_marker_sprite(
            _game_asset_path_or_default(getattr(dashboard, "chaser_sprite_path", None), CHASER_SPRITE_PATH)
        )
        dashboard.push_snapshot(snapshot)
        dashboard._prepare_frame_cache()
        object.__setattr__(plot_context, "_preview_dashboard", dashboard)
    dashboard._plot_panel_title_gap_px = 8
    dashboard.pygame = pygame
    dashboard.screen = screen
    dashboard.font = font
    dashboard.small_font = small_font
    dashboard.large_font = font
    dashboard.frame_convention = normalize_frame_convention(frame_convention)
    dashboard._render_motion_enabled = False
    return dashboard


def _draw_operator_planned_trajectory_overlay(
    pygame: Any,
    screen: Any,
    dashboard: Any,
    *,
    trajectory: np.ndarray,
    markers: np.ndarray,
    selected_state_ric_km_s: np.ndarray | None,
    selected_time_s: float | None,
    x_axis: int,
    y_axis: int,
    font: Any,
) -> None:
    if np.asarray(trajectory, dtype=float).size == 0 and np.asarray(markers, dtype=float).size == 0:
        return
    transforms = getattr(dashboard, "_frame_cache", {}).get("plot_transforms", {})
    transform = transforms.get((int(x_axis), int(y_axis)))
    if not isinstance(transform, dict):
        return
    try:
        plot = pygame.Rect(*transform["plot"])
        camera_center = np.asarray(transform["camera_center"], dtype=float).reshape(3)
        scale_x = float(transform["scale_x"])
        scale_y = float(transform["scale_y"])
        x_display_sign = float(transform["x_display_sign"])
        y_display_sign = float(transform["y_display_sign"])
    except (KeyError, TypeError, ValueError):
        return

    def to_px(point: np.ndarray) -> tuple[int, int]:
        shifted = np.asarray(point, dtype=float).reshape(-1)[:3] - camera_center
        return (
            plot.centerx + int(round(float(shifted[int(x_axis)]) * x_display_sign * scale_x)),
            plot.centery - int(round(float(shifted[int(y_axis)]) * y_display_sign * scale_y)),
        )

    previous_clip = screen.get_clip()
    screen.set_clip(plot)
    planned = np.asarray(trajectory, dtype=float).reshape(-1, 6) if np.asarray(trajectory).size else np.empty((0, 6))
    planned = planned[np.all(np.isfinite(planned), axis=1)] if planned.size else planned
    if planned.shape[0] >= 2:
        points = [to_px(row[:3]) for row in planned]
        pygame.draw.lines(screen, (238, 184, 92), False, points, width=2)
        _draw_operator_dashed_polyline(pygame, screen, points, color=(255, 224, 142), width=1)
    marker_rows = np.asarray(markers, dtype=float).reshape(-1, 6) if np.asarray(markers).size else np.empty((0, 6))
    marker_rows = marker_rows[np.all(np.isfinite(marker_rows), axis=1)] if marker_rows.size else marker_rows
    for marker_idx, marker in enumerate(marker_rows, start=1):
        marker_px = to_px(marker[:3])
        _draw_operator_velocity_vector(
            pygame,
            screen,
            marker_px,
            marker[3:6],
            x_axis=x_axis,
            y_axis=y_axis,
            x_display_sign=x_display_sign,
            y_display_sign=y_display_sign,
            color=(86, 202, 245),
            length_px=30.0,
        )
        pygame.draw.circle(screen, OPERATOR_BURN_MARKER_COLOR, marker_px, 5)
        pygame.draw.circle(screen, (12, 16, 22), marker_px, 5, width=1)
        _text(screen, font, str(marker_idx), (marker_px[0] + 7, marker_px[1] - 14), OPERATOR_BURN_MARKER_COLOR)
    if selected_state_ric_km_s is not None:
        _draw_operator_probe_dot(
            pygame,
            screen,
            to_px(np.asarray(selected_state_ric_km_s, dtype=float)[:3]),
            label=_operator_probe_time_label(selected_time_s),
            font=font,
        )
    screen.set_clip(previous_clip)


def _new_operator_preview_dashboard(pygame: Any, screen: Any, *, font: Any, small_font: Any) -> Any:
    dashboard = object.__new__(PygameRPODashboard)
    for item in fields(PygameRPODashboard):
        if item.default_factory is not MISSING:  # type: ignore[attr-defined]
            value = item.default_factory()  # type: ignore[misc]
        elif item.default is not MISSING:
            value = item.default
        else:
            continue
        setattr(dashboard, item.name, value)
    dashboard.pygame = pygame
    dashboard.screen = screen
    dashboard.clock = None
    dashboard.font = font
    dashboard.small_font = small_font
    dashboard.large_font = font
    dashboard.closed = False
    dashboard.t_s = []
    dashboard.sample_wall_s = []
    dashboard.rel_hist = []
    dashboard.target_rel_hist = []
    dashboard.target_reference_rel_hist = []
    dashboard.target_eci_hist = []
    dashboard.chaser_eci_hist = []
    dashboard.thrust_hist = []
    dashboard.thrust_ric_hist = []
    max_rows = int(max(getattr(dashboard, "max_history", 900), 2))
    dashboard._rel_array = _new_history_ring(6, max_rows)
    dashboard._target_rel_array = _new_history_ring(6, max_rows)
    dashboard._target_reference_rel_array = _new_history_ring(6, max_rows)
    dashboard._target_eci_array = _new_history_ring(6, max_rows)
    dashboard._chaser_eci_array = _new_history_ring(6, max_rows)
    dashboard._thrust_ric_array = _new_history_ring(3, max_rows)
    dashboard.mean_motion_rad_s = None
    dashboard.reference_state_eci = None
    dashboard.target_orbit_reference_state_eci = None
    dashboard.target_true_anomaly_deg = None
    dashboard.briefing_scroll_px = 0
    dashboard.mission_banner_scroll_px = 0
    dashboard._frame_cache = {}
    dashboard._raw_frame_cache = {}
    dashboard._frame_cache_dirty = True
    dashboard._render_motion_enabled = False
    dashboard._render_wall_time_s = 0.0
    dashboard._render_speed_multiple = 1.0
    dashboard._prediction_cache = {}
    dashboard._briefing_layout_cache = {}
    dashboard._mission_banner_layout_cache = {}
    dashboard._text_cache = {}
    dashboard._operator_projection_transition = None
    dashboard._target_sprite = None
    dashboard._chaser_sprite = None
    dashboard._sprite_scale_cache = {}
    return dashboard


def _draw_operator_initial_plot(
    pygame: Any,
    screen: Any,
    rect: Any,
    *,
    title: str,
    plot_context: OperatorPlotContext,
    plan: OperatorBurnPlan,
    trajectory_probe: OperatorTrajectoryProbe | None,
    x_axis: int,
    y_axis: int,
    frame_convention: FrameConvention,
    font: Any,
    small_font: Any,
) -> None:
    if _draw_operator_pilot_first_frame_plot(
        pygame,
        screen,
        rect,
        title=title,
        plot_context=plot_context,
        plan=plan,
        trajectory_probe=trajectory_probe,
        x_axis=x_axis,
        y_axis=y_axis,
        frame_convention=frame_convention,
        font=font,
        small_font=small_font,
    ):
        return
    pygame.draw.rect(screen, (20, 27, 36), rect, border_radius=10)
    pygame.draw.rect(screen, (80, 92, 110), rect, width=1, border_radius=10)
    _text(screen, font, title, (rect.x + 14, rect.y + 10), (230, 235, 242))
    plot = rect.inflate(-48, -72)
    plot.y += 8
    pygame.draw.rect(screen, (8, 11, 16), plot)
    pygame.draw.rect(screen, (72, 84, 102), plot, width=1)
    rel6 = plot_context.initial_relative_ric_km_s
    if rel6 is None:
        _text(screen, font, "Initial RIC unavailable", (plot.x, plot.centery - 8), (245, 126, 126))
        return
    training_cfg = plot_context.training_config or RPOTrainingConfig(enabled=False)
    readout_state = (
        np.asarray(trajectory_probe.state_ric_km_s, dtype=float).reshape(6)
        if trajectory_probe is not None
        else np.asarray(rel6, dtype=float).reshape(6)
    )
    rel = np.array(readout_state[:3], dtype=float)
    initial_rel = np.array(rel6[:3], dtype=float)
    initial_rel_vel = np.array(rel6[3:6], dtype=float)
    coast = np.array(plot_context.initial_coast_ric_km_s, dtype=float).reshape(-1, 6)
    nmt = _operator_nmt_points(training_cfg)
    nmt_bounds = _operator_nmt_boundary_points(training_cfg)
    target_current = np.zeros(3, dtype=float)
    camera_center = _operator_camera_center_ric(
        plot_context,
        chaser_current=initial_rel,
        target_current=target_current,
        x_axis=x_axis,
        y_axis=y_axis,
    )
    projected_sets: list[np.ndarray] = [
        (initial_rel - camera_center).reshape(1, 3)[:, [int(x_axis), int(y_axis)]],
        (target_current - camera_center).reshape(1, 3)[:, [int(x_axis), int(y_axis)]],
    ]
    if coast.size:
        projected_sets.append((coast[:, :3] - camera_center.reshape(1, 3))[:, [int(x_axis), int(y_axis)]])
    if nmt.size and not nmt_bounds:
        projected_sets.append((nmt - camera_center.reshape(1, 3))[:, [int(x_axis), int(y_axis)]])
    projected_sets.extend(
        (boundary - camera_center.reshape(1, 3))[:, [int(x_axis), int(y_axis)]]
        for boundary in nmt_bounds
        if boundary.size
    )
    min_span = _operator_minimum_plot_span_km(
        plot_context,
        x_axis=x_axis,
        y_axis=y_axis,
        target_current=target_current,
        nmt=nmt,
        nmt_bounds=nmt_bounds,
    )
    scale_x, scale_y = _operator_axis_scales_for_plot(
        plot_context,
        plot,
        pts=projected_sets,
        min_span_km=min_span,
        x_axis=x_axis,
        y_axis=y_axis,
        screen_size=screen.get_size(),
    )
    center = (plot.centerx, plot.centery)
    x_display_sign = frame_convention_display_axis_sign(frame_convention, x_axis)
    y_display_sign = frame_convention_display_axis_sign(frame_convention, y_axis)

    def to_px(point: np.ndarray | tuple[float, float, float]) -> tuple[int, int]:
        arr = np.array(point, dtype=float).reshape(-1)[:3] - camera_center
        return (
            center[0] + int(round(float(arr[int(x_axis)]) * x_display_sign * scale_x)),
            center[1] - int(round(float(arr[int(y_axis)]) * y_display_sign * scale_y)),
        )

    def circle_rect(center_px: tuple[int, int], radius_km: float) -> Any:
        radius_x_px = max(1, int(round(float(radius_km) * scale_x)))
        radius_y_px = max(1, int(round(float(radius_km) * scale_y)))
        return pygame.Rect(center_px[0] - radius_x_px, center_px[1] - radius_y_px, radius_x_px * 2, radius_y_px * 2)

    previous_clip = screen.get_clip()
    screen.set_clip(plot)
    half_span = max(min_span, MIN_PLOT_SPAN_KM)
    if projected_sets:
        finite_span = [
            np.max(np.abs(np.array(points, dtype=float).reshape(-1, 2)))
            for points in projected_sets
            if np.array(points, dtype=float).size
        ]
        if finite_span:
            half_span = max(half_span, float(max(finite_span)) * 1.2)
    _draw_operator_grid(pygame, screen, plot, center=center, scale_x=scale_x, scale_y=scale_y, half_span_km=half_span)
    target_px = to_px(target_current)
    _draw_operator_forbidden_regions(
        pygame,
        screen,
        plot,
        training_cfg,
        x_axis=x_axis,
        y_axis=y_axis,
        to_px=to_px,
        offset=target_current,
    )
    plane = _plane_key_for_axes(x_axis=int(x_axis), y_axis=int(y_axis))
    if plane in plot_context.proximity_ring_plot_planes:
        if training_cfg.keepout_radius_km is not None and float(training_cfg.keepout_radius_km) > 0.0:
            pygame.draw.ellipse(
                screen,
                (190, 68, 68),
                circle_rect(target_px, float(training_cfg.keepout_radius_km)),
                width=2,
            )
        if training_cfg.goal_range_km is not None and float(training_cfg.goal_range_km) > 0.0:
            outer = float(training_cfg.goal_range_km)
            inner = outer
            if training_cfg.goal_range_tolerance_km is not None:
                tol = max(float(training_cfg.goal_range_tolerance_km), 0.0)
                inner = max(outer - tol, 0.0)
                outer += tol
            pygame.draw.ellipse(screen, (78, 178, 112), circle_rect(target_px, outer), width=2)
            if inner > 0.0 and not np.isclose(inner, outer):
                pygame.draw.ellipse(screen, (78, 178, 112), circle_rect(target_px, inner), width=2)
        if training_cfg.goal_radius_km is not None and float(training_cfg.goal_radius_km) > 0.0:
            goal_center = np.array(training_cfg.goal_relative_ric_km, dtype=float).reshape(3)
            pygame.draw.ellipse(
                screen,
                (78, 178, 112),
                circle_rect(to_px(goal_center), float(training_cfg.goal_radius_km)),
                width=2,
            )
        if (
            training_cfg.hard_speed_limit_radius_km is not None
            and float(training_cfg.hard_speed_limit_radius_km) > 0.0
        ):
            pygame.draw.ellipse(
                screen,
                (232, 194, 74),
                circle_rect(target_px, float(training_cfg.hard_speed_limit_radius_km)),
                width=2,
            )
    for boundary in nmt_bounds:
        boundary_points = _operator_rows_to_px(boundary, to_px=to_px)
        if len(boundary_points) >= 2:
            pygame.draw.lines(screen, (78, 178, 112), True, boundary_points, width=2)
    nmt_points = _operator_rows_to_px(nmt, to_px=to_px)
    if len(nmt_points) >= 2 and not nmt_bounds:
        _draw_operator_dashed_polyline(pygame, screen, nmt_points, color=(120, 236, 154), width=2)
    coast_points = _operator_rows_to_px(coast[:, :3], to_px=to_px) if coast.size else []
    if len(coast_points) >= 2:
        _draw_operator_dashed_polyline(pygame, screen, coast_points, color=(96, 174, 224), width=2)
    trajectory, markers = _operator_planned_trajectory(plot_context, plan)
    trajectory_points = _operator_rows_to_px(trajectory[:, :3], to_px=to_px) if trajectory.size else []
    if len(trajectory_points) >= 2:
        pygame.draw.lines(screen, (238, 184, 92), False, trajectory_points, width=2)
        _draw_operator_dashed_polyline(pygame, screen, trajectory_points, color=(255, 224, 142), width=1)
    if markers.size:
        for marker_idx, marker in enumerate(markers.reshape(-1, 6), start=1):
            marker_px = to_px(marker[:3])
            _draw_operator_velocity_vector(
                pygame,
                screen,
                marker_px,
                marker[3:6],
                x_axis=x_axis,
                y_axis=y_axis,
                x_display_sign=x_display_sign,
                y_display_sign=y_display_sign,
                color=(86, 202, 245),
                length_px=30.0,
            )
            pygame.draw.circle(screen, OPERATOR_BURN_MARKER_COLOR, marker_px, 5)
            pygame.draw.circle(screen, (12, 16, 22), marker_px, 5, width=1)
            _text(screen, font, str(marker_idx), (marker_px[0] + 7, marker_px[1] - 14), OPERATOR_BURN_MARKER_COLOR)
    if trajectory_probe is not None:
        _draw_operator_probe_dot(
            pygame,
            screen,
            to_px(rel),
            label=_operator_probe_time_label(trajectory_probe.time_s),
            font=font,
        )
    pygame.draw.line(screen, (90, 104, 124), (plot.left, center[1]), (plot.right, center[1]), width=1)
    pygame.draw.line(screen, (90, 104, 124), (center[0], plot.top), (center[0], plot.bottom), width=1)
    pygame.draw.circle(screen, (96, 174, 224), target_px, 5)
    chaser_px = to_px(initial_rel)
    pygame.draw.circle(screen, (245, 205, 92), chaser_px, 6)
    pygame.draw.circle(screen, (10, 14, 20), chaser_px, 6, width=1)
    _draw_operator_velocity_vector(
        pygame,
        screen,
        chaser_px,
        initial_rel_vel,
        x_axis=x_axis,
        y_axis=y_axis,
        x_display_sign=x_display_sign,
        y_display_sign=y_display_sign,
    )
    screen.set_clip(previous_clip)

    _draw_operator_signed_axis_labels(
        screen,
        plot,
        x_axis=x_axis,
        y_axis=y_axis,
        x_display_sign=x_display_sign,
        y_display_sign=y_display_sign,
        font=font,
    )
    _draw_operator_initial_state_readout(
        screen,
        rect,
        rel_state_ric_km_s=readout_state,
        x_axis=x_axis,
        y_axis=y_axis,
        font=font,
        color=(86, 202, 245) if trajectory_probe is not None else (162, 178, 198),
    )


def _draw_operator_initial_state_readout(
    screen: Any,
    rect: Any,
    *,
    rel_state_ric_km_s: np.ndarray,
    x_axis: int,
    y_axis: int,
    font: Any,
    color: tuple[int, int, int] = (162, 178, 198),
) -> None:
    state = np.asarray(rel_state_ric_km_s, dtype=float).reshape(-1)
    if state.size < 6:
        return
    rel = state[:3]
    rel_vel = state[3:6]
    show_velocity_readout = int(x_axis) == 2 and int(y_axis) == 0
    readout_labels = ("dR", "dI", "dC") if show_velocity_readout else ("R", "I", "C")
    readout_values = rel_vel * 1000.0 if show_velocity_readout else rel
    readout_unit = "m/s" if show_velocity_readout else "km"
    _text(
        screen,
        font,
        (
            f"{readout_labels[0]} {readout_values[0]:.2f} {readout_unit}  "
            f"{readout_labels[1]} {readout_values[1]:.2f} {readout_unit}  "
            f"{readout_labels[2]} {readout_values[2]:.2f} {readout_unit}"
        ),
        (rect.x + 12, rect.bottom - 22),
        color,
    )


def _operator_axis_symbol(axis: int) -> str:
    labels = ("R", "I", "C")
    idx = int(axis)
    return labels[idx] if 0 <= idx < len(labels) else ""


def _operator_signed_axis_label(axis: int, sign: int) -> str:
    symbol = _operator_axis_symbol(axis)
    if not symbol:
        return ""
    prefix = "+" if int(sign) >= 0 else "-"
    return f"{prefix}{symbol}"


def _draw_operator_signed_axis_labels(
    screen: Any,
    plot: Any,
    *,
    x_axis: int,
    y_axis: int,
    x_display_sign: float,
    y_display_sign: float,
    font: Any,
) -> None:
    color = (162, 178, 198)
    x_plus = _operator_signed_axis_label(x_axis, 1)
    x_minus = _operator_signed_axis_label(x_axis, -1)
    y_plus = _operator_signed_axis_label(y_axis, 1)
    y_minus = _operator_signed_axis_label(y_axis, -1)

    x_plus_right = float(x_display_sign) >= 0.0
    x_left_label = x_minus if x_plus_right else x_plus
    x_right_label = x_plus if x_plus_right else x_minus
    _text(screen, font, x_left_label, (plot.left + 8, plot.centery + 6), color)
    _text(screen, font, x_right_label, (plot.right - _text_width(font, x_right_label) - 8, plot.centery + 6), color)

    y_plus_top = float(y_display_sign) >= 0.0
    y_top_label = y_plus if y_plus_top else y_minus
    y_bottom_label = y_minus if y_plus_top else y_plus
    _text(screen, font, y_top_label, (plot.centerx + 8, plot.top + 4), color)
    _text(screen, font, y_bottom_label, (plot.centerx + 8, plot.bottom - 24), color)


def _launcher_mode_palette(mode: str) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    if _normalize_game_mode(mode) == "operator":
        return (62, 48, 28), (238, 184, 92)
    return (28, 48, 66), (96, 174, 224)


def _draw_clear_progress_button(pygame: Any, screen: Any, *, mode: str = "pilot", font: Any) -> None:
    rect = pygame.Rect(*CLEAR_PROGRESS_RECT)
    fill, stroke = _launcher_mode_palette(mode)
    pygame.draw.rect(screen, fill, rect, border_radius=6)
    pygame.draw.rect(screen, stroke, rect, width=1, border_radius=6)
    _text(screen, font, "Clear Progress", (rect.x + 15, rect.y + 8), (230, 238, 245))


def _draw_record_video_button(pygame: Any, screen: Any, *, enabled: bool, mode: str = "pilot", font: Any) -> None:
    rect = pygame.Rect(*RECORD_VIDEO_RECT)
    fill, stroke = _launcher_mode_palette(mode)
    pygame.draw.rect(screen, fill, rect, border_radius=6)
    pygame.draw.rect(screen, stroke, rect, width=1, border_radius=6)
    label = "Video: ON" if enabled else "Video: OFF"
    _text(screen, font, label, (rect.x + 18, rect.y + 8), (246, 238, 242))


def _draw_music_button(pygame: Any, screen: Any, *, enabled: bool, mode: str = "pilot", font: Any) -> None:
    rect = pygame.Rect(*MUSIC_RECT)
    fill, stroke = _launcher_mode_palette(mode)
    pygame.draw.rect(screen, fill, rect, border_radius=6)
    pygame.draw.rect(screen, stroke, rect, width=1, border_radius=6)
    label = "Music: ON" if enabled else "Music: OFF"
    _text(screen, font, label, (rect.x + 18, rect.y + 8), (238, 244, 250))


def _draw_mode_toggle(pygame: Any, screen: Any, *, mode: str, font: Any) -> None:
    width, height = screen.get_size()
    rect = pygame.Rect(*_mode_toggle_rect(width, height))
    mode_key = _normalize_game_mode(mode)
    is_operator = mode_key == "operator"
    fill = (62, 48, 28) if is_operator else (28, 52, 66)
    stroke = (238, 184, 92) if is_operator else (96, 174, 224)
    pygame.draw.rect(screen, fill, rect, border_radius=6)
    pygame.draw.rect(screen, stroke, rect, width=1, border_radius=6)
    label = "Operator Mode" if is_operator else "Pilot Mode"
    _text_centered(screen, font, label, rect.center, (246, 240, 226))


def _draw_settings_button(pygame: Any, screen: Any, *, mode: str) -> None:
    width, height = screen.get_size()
    rect = pygame.Rect(*_settings_button_rect(width, height))
    is_operator = _normalize_game_mode(mode) == "operator"
    fill = (62, 48, 28) if is_operator else (28, 52, 66)
    stroke = (238, 184, 92) if is_operator else (96, 174, 224)
    icon = (246, 240, 226)
    pygame.draw.rect(screen, fill, rect, border_radius=6)
    pygame.draw.rect(screen, stroke, rect, width=1, border_radius=6)
    center = rect.center
    for dx, dy in ((0, -9), (6, -6), (9, 0), (6, 6), (0, 9), (-6, 6), (-9, 0), (-6, -6)):
        pygame.draw.line(screen, icon, center, (center[0] + dx, center[1] + dy), width=1)
    pygame.draw.circle(screen, fill, center, 7)
    pygame.draw.circle(screen, icon, center, 7, width=2)
    pygame.draw.circle(screen, icon, center, 2)


def _draw_difficulty_picker(
    pygame: Any,
    screen: Any,
    *,
    selected_difficulty: str,
    selected_mode: str = "pilot",
    font: Any,
) -> None:
    operator_mode = _normalize_game_mode(selected_mode) == "operator"
    label = "Error" if operator_mode else "Assists"
    label_right = 574 + _text_width(font, "Error")
    label_x = 574 if operator_mode else label_right - _text_width(font, label)
    _text(screen, font, label, (label_x, 92), (172, 186, 206))
    for idx, difficulty in enumerate(DIFFICULTY_OPTIONS):
        rect = pygame.Rect(642 + idx * 86, 86, 76, 26)
        is_selected = difficulty == selected_difficulty
        if operator_mode:
            fill = (62, 48, 28) if is_selected else (18, 24, 32)
            stroke = (238, 184, 92) if is_selected else (112, 86, 46)
        else:
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
    mode: str = "pilot",
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
    for line in _wrapped_budget_lines(option, small_font, content_rect.width):
        _text(screen, small_font, line, (content_rect.x, y), (162, 178, 198))
        y += PREVIEW_LINE_HEIGHT
    y += PREVIEW_SECTION_GAP
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
    _draw_preview_scrollbar(pygame, screen, rect, content_height=content_height, scroll_px=scroll, mode=mode)


def _draw_preview_scrollbar(
    pygame: Any,
    screen: Any,
    rect: Any,
    *,
    content_height: int,
    scroll_px: int,
    mode: str = "pilot",
) -> None:
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
    thumb_color = (238, 184, 92) if _normalize_game_mode(mode) == "operator" else (96, 174, 224)
    pygame.draw.rect(screen, thumb_color, thumb, border_radius=2)


def _preview_content_height(option: GameScenarioOption, *, font: Any, small_font: Any, width_px: int) -> int:
    y = 0
    y += 34
    y += len(_wrapped_budget_lines(option, small_font, width_px)) * PREVIEW_LINE_HEIGHT
    y += PREVIEW_SECTION_GAP
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


def _wrapped_budget_lines(option: GameScenarioOption, font: Any, width_px: int) -> list[str]:
    return _wrap_text_px(_budget_line(option), font, width_px)


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


def _normalize_game_mode(value: Any) -> str:
    key = str(value or "pilot").strip().lower()
    if key in {"operator", "op", "script", "scripted"}:
        return "operator"
    return "pilot"


def _toggle_game_mode(value: Any) -> str:
    return "operator" if _normalize_game_mode(value) == "pilot" else "pilot"


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

__all__ = [name for name in globals() if not name.startswith("__")]
