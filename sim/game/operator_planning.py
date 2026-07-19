# ruff: noqa: F401,F403,F405,I001
from .launcher_common import *
from .launcher_models import *
from .scenario_catalog import *
from .launcher_persistence import *


def _operator_widget_renderer(name: str) -> Any:
    from . import launcher_widgets

    return _launcher_dep(name, getattr(launcher_widgets, name))


def _run_operator_plan_screen(
    pygame: Any,
    screen: Any,
    clock: Any,
    *,
    option: GameScenarioOption,
    font: Any,
    small_font: Any,
    title_font: Any,
    initial_plan: OperatorBurnPlan | None = None,
    difficulty: str = "easy",
    frame_convention: FrameConvention | dict[str, Any] | None = None,
    read_only: bool = False,
    demo_title: str = "",
    launch_label: str = "Launch",
) -> OperatorBurnPlan | None:
    frame_convention = normalize_frame_convention(frame_convention)
    seed_plan = initial_plan if initial_plan is not None else _load_saved_operator_burn_plan(option.scenario_id)
    rows: list[list[str]] = _operator_rows_from_plan(seed_plan)
    active_cell: tuple[int, int] = (0, 0)
    table_scroll_row = 0
    objectives_visible = True
    objectives_scroll_px = 0
    equation_sheet_visible = False
    equation_sheet_scroll_px = 0
    plot_context = _operator_plot_context(option.path, difficulty=difficulty)
    trajectory_probe: OperatorTrajectoryProbe | None = None
    draw_operator_plan_screen = _operator_widget_renderer("_draw_operator_plan_screen")
    while True:
        width, height = screen.get_size()
        button_gap = 10
        launch_rect = pygame.Rect(width - 52 - 126, height - 70, 126, 36)
        cancel_rect = pygame.Rect(launch_rect.x - button_gap - 120, height - 70, 120, 36)
        equation_sheet_rect = pygame.Rect(*_operator_equation_sheet_button_rect(width, height))
        objectives_rect = pygame.Rect(*_operator_objectives_button_rect(width, height))
        objectives_overlay_rect = pygame.Rect(*_operator_objectives_overlay_rect(width, height))
        objectives_content_rect = _operator_objectives_content_rect(objectives_overlay_rect)
        objectives_content_height = _operator_objectives_content_height(
            option,
            plot_context.training_config,
            font=small_font,
            width_px=objectives_content_rect[2],
        )
        objectives_scroll_px = _clamp_operator_objectives_scroll_px(
            objectives_scroll_px,
            content_height=objectives_content_height,
            viewport_height=objectives_content_rect[3],
        )
        equation_sheet_content_rect = _operator_objectives_content_rect(objectives_overlay_rect)
        equation_sheet_content_height = _operator_equation_sheet_content_height(
            small_font,
            width_px=equation_sheet_content_rect[2],
        )
        equation_sheet_scroll_px = _clamp_operator_objectives_scroll_px(
            equation_sheet_scroll_px,
            content_height=equation_sheet_content_height,
            viewport_height=equation_sheet_content_rect[3],
        )
        table_rect = _operator_burn_table_rect(width, height)
        add_burn_rect = _operator_add_burn_button_rect(pygame, table_rect)
        table_scroll_row = int(max(0, min(table_scroll_row, _operator_max_table_scroll_row(len(rows), table_rect))))
        field_rects = _operator_burn_field_rects(
            pygame,
            table_rect,
            row_count=len(rows),
            scroll_row=table_scroll_row,
        )
        delete_rects = _operator_burn_delete_rects(
            pygame,
            table_rect,
            row_count=len(rows),
            scroll_row=table_scroll_row,
        )
        plan, errors = _operator_plan_from_rows(rows, option=option)
        can_launch = not errors
        current_plan_key = _operator_planned_trajectory_cache_key(plot_context, plan) if can_launch else None
        if trajectory_probe is not None and trajectory_probe.plan_key != current_plan_key:
            trajectory_probe = None
        validation_message = errors[0] if errors else _operator_plan_status(plan, option=option)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
                if equation_sheet_visible and event.key in {getattr(pygame, "K_PAGEDOWN", object())}:
                    equation_sheet_scroll_px = _clamp_operator_objectives_scroll_px(
                        equation_sheet_scroll_px + PREVIEW_SCROLL_STEP_PX,
                        content_height=equation_sheet_content_height,
                        viewport_height=equation_sheet_content_rect[3],
                    )
                    continue
                if equation_sheet_visible and event.key in {getattr(pygame, "K_PAGEUP", object())}:
                    equation_sheet_scroll_px = _clamp_operator_objectives_scroll_px(
                        equation_sheet_scroll_px - PREVIEW_SCROLL_STEP_PX,
                        content_height=equation_sheet_content_height,
                        viewport_height=equation_sheet_content_rect[3],
                    )
                    continue
                if objectives_visible and event.key in {getattr(pygame, "K_PAGEDOWN", object())}:
                    objectives_scroll_px = _clamp_operator_objectives_scroll_px(
                        objectives_scroll_px + PREVIEW_SCROLL_STEP_PX,
                        content_height=objectives_content_height,
                        viewport_height=objectives_content_rect[3],
                    )
                    continue
                if objectives_visible and event.key in {getattr(pygame, "K_PAGEUP", object())}:
                    objectives_scroll_px = _clamp_operator_objectives_scroll_px(
                        objectives_scroll_px - PREVIEW_SCROLL_STEP_PX,
                        content_height=objectives_content_height,
                        viewport_height=objectives_content_rect[3],
                    )
                    continue
                if event.key == pygame.K_TAB:
                    if read_only:
                        continue
                    active_cell = _operator_next_cell(
                        active_cell,
                        rows=rows,
                        backwards=bool(getattr(event, "mod", 0) & pygame.KMOD_SHIFT),
                    )
                    table_scroll_row = _operator_scroll_for_active_row(
                        active_cell[0],
                        table_scroll_row,
                        row_count=len(rows),
                        table_rect=table_rect,
                    )
                    continue
                if event.key in {pygame.K_RIGHT, pygame.K_d}:
                    if read_only:
                        continue
                    active_cell = (active_cell[0], min(active_cell[1] + 1, 3))
                    continue
                if event.key in {pygame.K_LEFT, pygame.K_a}:
                    if read_only:
                        continue
                    active_cell = (active_cell[0], max(active_cell[1] - 1, 0))
                    continue
                if event.key in {pygame.K_DOWN, pygame.K_s}:
                    if read_only:
                        continue
                    active_cell = (min(active_cell[0] + 1, len(rows) - 1), active_cell[1])
                    table_scroll_row = _operator_scroll_for_active_row(
                        active_cell[0],
                        table_scroll_row,
                        row_count=len(rows),
                        table_rect=table_rect,
                    )
                    continue
                if event.key in {pygame.K_UP, pygame.K_w}:
                    if read_only:
                        continue
                    active_cell = (max(active_cell[0] - 1, 0), active_cell[1])
                    table_scroll_row = _operator_scroll_for_active_row(
                        active_cell[0],
                        table_scroll_row,
                        row_count=len(rows),
                        table_rect=table_rect,
                    )
                    continue
                if event.key in {pygame.K_RETURN, pygame.K_KP_ENTER}:
                    if can_launch:
                        if not read_only:
                            _save_operator_burn_plan(option.scenario_id, plan)
                        return plan
                    continue
                if event.key == pygame.K_BACKSPACE:
                    if read_only:
                        continue
                    row_idx, col_idx = active_cell
                    rows[row_idx][col_idx] = rows[row_idx][col_idx][:-1]
                    trajectory_probe = None
                    continue
                if event.key == pygame.K_DELETE:
                    if read_only:
                        continue
                    row_idx, col_idx = active_cell
                    rows[row_idx][col_idx] = ""
                    trajectory_probe = None
                    continue
                event_text = getattr(event, "unicode", "")
                if not read_only and event_text and _operator_field_accepts_text(event_text):
                    row_idx, col_idx = active_cell
                    rows[row_idx][col_idx] += event_text
                    trajectory_probe = None
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                if equation_sheet_visible and _pos_in_bounds(
                    mouse_pos,
                    (
                        objectives_overlay_rect.x,
                        objectives_overlay_rect.y,
                        objectives_overlay_rect.w,
                        objectives_overlay_rect.h,
                    ),
                ) and event.button in {4, 5}:
                    equation_sheet_scroll_px = _clamp_operator_objectives_scroll_px(
                        equation_sheet_scroll_px
                        + (PREVIEW_SCROLL_STEP_PX if event.button == 5 else -PREVIEW_SCROLL_STEP_PX),
                        content_height=equation_sheet_content_height,
                        viewport_height=equation_sheet_content_rect[3],
                    )
                    continue
                if objectives_visible and _pos_in_bounds(
                    mouse_pos,
                    (
                        objectives_overlay_rect.x,
                        objectives_overlay_rect.y,
                        objectives_overlay_rect.w,
                        objectives_overlay_rect.h,
                    ),
                ) and event.button in {4, 5}:
                    objectives_scroll_px = _clamp_operator_objectives_scroll_px(
                        objectives_scroll_px + (PREVIEW_SCROLL_STEP_PX if event.button == 5 else -PREVIEW_SCROLL_STEP_PX),
                        content_height=objectives_content_height,
                        viewport_height=objectives_content_rect[3],
                    )
                    continue
                if _pos_in_bounds(mouse_pos, table_rect) and event.button in {4, 5}:
                    table_scroll_row = int(
                        max(
                            0,
                            min(
                                table_scroll_row + (1 if event.button == 5 else -1),
                                _operator_max_table_scroll_row(len(rows), table_rect),
                            ),
                        )
                    )
                    continue
                if event.button != 1:
                    continue
                if _pos_in_bounds(mouse_pos, (objectives_rect.x, objectives_rect.y, objectives_rect.w, objectives_rect.h)):
                    objectives_visible = not objectives_visible
                    if objectives_visible:
                        equation_sheet_visible = False
                        objectives_scroll_px = 0
                    continue
                if _pos_in_bounds(
                    mouse_pos,
                    (equation_sheet_rect.x, equation_sheet_rect.y, equation_sheet_rect.w, equation_sheet_rect.h),
                ):
                    equation_sheet_visible = not equation_sheet_visible
                    if equation_sheet_visible:
                        objectives_visible = False
                        equation_sheet_scroll_px = 0
                    continue
                if _pos_in_bounds(mouse_pos, (launch_rect.x, launch_rect.y, launch_rect.w, launch_rect.h)) and can_launch:
                    if not read_only:
                        _save_operator_burn_plan(option.scenario_id, plan)
                    return plan
                if _pos_in_bounds(mouse_pos, (cancel_rect.x, cancel_rect.y, cancel_rect.w, cancel_rect.h)):
                    return None
                if can_launch and current_plan_key is not None and not objectives_visible and not equation_sheet_visible:
                    probe_handled, probe_state, probe_time_s = _operator_trajectory_probe_from_click(
                        plot_context,
                        plan,
                        mouse_pos,
                        selected_probe=trajectory_probe,
                    )
                    if probe_handled:
                        trajectory_probe = (
                            OperatorTrajectoryProbe(
                                state_ric_km_s=tuple(float(value) for value in probe_state.reshape(6)),
                                time_s=float(probe_time_s or 0.0),
                                plan_key=current_plan_key,
                            )
                            if probe_state is not None
                            else None
                        )
                        continue
                if (
                    _pos_in_bounds(mouse_pos, (add_burn_rect.x, add_burn_rect.y, add_burn_rect.w, add_burn_rect.h))
                    and len(rows) < OPERATOR_BURN_MAX_ROWS
                    and not read_only
                ):
                    rows.append(_operator_new_burn_row_from_probe(trajectory_probe))
                    trajectory_probe = None
                    active_cell = (len(rows) - 1, 0)
                    table_scroll_row = _operator_scroll_for_active_row(
                        active_cell[0],
                        table_scroll_row,
                        row_count=len(rows),
                        table_rect=table_rect,
                    )
                    continue
                clicked_delete_row = _operator_delete_row_at_pos(mouse_pos, delete_rects, table_rect=table_rect)
                if clicked_delete_row is not None and not read_only:
                    if len(rows) > 1:
                        row_idx = int(clicked_delete_row)
                        del rows[row_idx]
                        trajectory_probe = None
                        active_cell = (min(row_idx, len(rows) - 1), min(active_cell[1], 3))
                        table_scroll_row = _operator_scroll_for_active_row(
                            active_cell[0],
                            table_scroll_row,
                            row_count=len(rows),
                            table_rect=table_rect,
                        )
                    else:
                        rows[0] = ["", "", "", ""]
                        trajectory_probe = None
                        active_cell = (0, 0)
                        table_scroll_row = 0
                    continue
                clicked_cell = _operator_cell_at_pos(mouse_pos, field_rects, table_rect=table_rect)
                if clicked_cell is not None and not read_only:
                    active_cell = clicked_cell
            if event.type == getattr(pygame, "MOUSEWHEEL", object()):
                mouse_pos = pygame.mouse.get_pos()
                if equation_sheet_visible and _pos_in_bounds(
                    mouse_pos,
                    (
                        objectives_overlay_rect.x,
                        objectives_overlay_rect.y,
                        objectives_overlay_rect.w,
                        objectives_overlay_rect.h,
                    ),
                ):
                    equation_sheet_scroll_px = _clamp_operator_objectives_scroll_px(
                        equation_sheet_scroll_px - int(event.y) * PREVIEW_SCROLL_STEP_PX,
                        content_height=equation_sheet_content_height,
                        viewport_height=equation_sheet_content_rect[3],
                    )
                elif objectives_visible and _pos_in_bounds(
                    mouse_pos,
                    (
                        objectives_overlay_rect.x,
                        objectives_overlay_rect.y,
                        objectives_overlay_rect.w,
                        objectives_overlay_rect.h,
                    ),
                ):
                    objectives_scroll_px = _clamp_operator_objectives_scroll_px(
                        objectives_scroll_px - int(event.y) * PREVIEW_SCROLL_STEP_PX,
                        content_height=objectives_content_height,
                        viewport_height=objectives_content_rect[3],
                    )
                elif _pos_in_bounds(mouse_pos, table_rect):
                    table_scroll_row = int(
                        max(
                            0,
                            min(
                                table_scroll_row - int(event.y),
                                _operator_max_table_scroll_row(len(rows), table_rect),
                            ),
                        )
                    )

        draw_operator_plan_screen(
            pygame,
            screen,
            option=option,
            plan=plan if can_launch else OperatorBurnPlan(),
            rows=rows,
            active_cell=active_cell,
            field_rects=field_rects,
            delete_rects=delete_rects,
            table_scroll_row=table_scroll_row,
            plot_context=plot_context,
            trajectory_probe=trajectory_probe,
            frame_convention=frame_convention,
            validation_message=validation_message,
            can_launch=can_launch,
            launch_rect=launch_rect,
            cancel_rect=cancel_rect,
            add_burn_rect=add_burn_rect,
            objectives_rect=objectives_rect,
            equation_sheet_rect=equation_sheet_rect,
            equation_sheet_visible=equation_sheet_visible,
            equation_sheet_scroll_px=equation_sheet_scroll_px,
            objectives_visible=objectives_visible,
            objectives_scroll_px=objectives_scroll_px,
            read_only=read_only,
            demo_title=demo_title,
            launch_label=launch_label,
            font=font,
            small_font=small_font,
            title_font=title_font,
        )
        pygame.display.flip()
        clock.tick(60)


def _run_operator_prebrief_screen(
    pygame: Any,
    screen: Any,
    clock: Any,
    *,
    option: GameScenarioOption,
    font: Any,
    small_font: Any,
    title_font: Any,
) -> bool:
    scroll_px = 0
    draw_operator_prebrief_screen = _operator_widget_renderer("_draw_operator_prebrief_screen")
    while True:
        width, height = screen.get_size()
        continue_rect = pygame.Rect(width - 214, height - 70, 162, 36)
        cancel_rect = pygame.Rect(width - 354, height - 70, 120, 36)
        content_rect = _operator_prebrief_content_rect(width, height)
        content_height = _operator_prebrief_content_height(option, font=font, small_font=small_font, width_px=content_rect[2])
        max_scroll = max(content_height - content_rect[3], 0)
        scroll_px = int(max(0, min(scroll_px, max_scroll)))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key in {pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE}:
                    return True
                if event.key in {pygame.K_DOWN, pygame.K_s, getattr(pygame, "K_PAGEDOWN", object())}:
                    scroll_px += PREVIEW_SCROLL_STEP_PX
                if event.key in {pygame.K_UP, pygame.K_w, getattr(pygame, "K_PAGEUP", object())}:
                    scroll_px -= PREVIEW_SCROLL_STEP_PX
            if event.type == pygame.MOUSEWHEEL:
                scroll_px -= int(event.y) * PREVIEW_SCROLL_STEP_PX
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_pos = pygame.mouse.get_pos()
                if _pos_in_bounds(mouse_pos, (continue_rect.x, continue_rect.y, continue_rect.w, continue_rect.h)):
                    return True
                if _pos_in_bounds(mouse_pos, (cancel_rect.x, cancel_rect.y, cancel_rect.w, cancel_rect.h)):
                    return False
        draw_operator_prebrief_screen(
            pygame,
            screen,
            option=option,
            scroll_px=scroll_px,
            continue_rect=continue_rect,
            cancel_rect=cancel_rect,
            font=font,
            small_font=small_font,
            title_font=title_font,
        )
        pygame.display.flip()
        clock.tick(60)


def _operator_prebrief_content_rect(screen_width: int, screen_height: int) -> tuple[int, int, int, int]:
    panel = _operator_plan_panel_rect(screen_width, screen_height)
    return (panel[0] + 24, panel[1] + 24, max(panel[2] - 48, 1), max(panel[3] - 58, 1))


def _operator_prebrief_content_height(
    option: GameScenarioOption,
    *,
    font: Any,
    small_font: Any,
    width_px: int,
) -> int:
    y = 0
    y += 34
    y += len(_wrapped_budget_lines(option, small_font, width_px)) * PREVIEW_LINE_HEIGHT
    y += PREVIEW_SECTION_GAP
    y = _section_height(option.learning_goal, small_font, y, width_px)
    y = _section_height(option.player_brief or option.description, small_font, y + PREVIEW_SECTION_GAP, width_px)
    y = _bullets_height(option.pass_criteria, small_font, y + PREVIEW_SECTION_GAP, width_px)
    y = _bullets_height(option.instructor_notes, small_font, y + PREVIEW_SECTION_GAP, width_px)
    return max(y, _text_height(font))


def _operator_plan_from_text(text: str, *, option: GameScenarioOption) -> tuple[OperatorBurnPlan, tuple[str, ...]]:
    try:
        plan = parse_operator_burn_plan(text)
    except ValueError as exc:
        return OperatorBurnPlan(), (str(exc),)
    errors = validate_operator_burn_plan(
        plan,
        total_delta_v_budget_m_s=_operator_delta_v_budget_m_s(option),
        max_time_s=option.time_budget_s,
    )
    return plan, errors


def _operator_plan_from_rows(rows: list[list[str]], *, option: GameScenarioOption) -> tuple[OperatorBurnPlan, tuple[str, ...]]:
    burns: list[OperatorBurn] = []
    for row_idx, row in enumerate(rows, start=1):
        values = [str(row[col] if col < len(row) else "").strip() for col in range(4)]
        if not any(values):
            continue
        if not values[0]:
            return OperatorBurnPlan(), (f"Burn {row_idx}: enter a time.",)
        parsed: list[float] = []
        for col_idx, value in enumerate(values):
            if col_idx > 0 and not value:
                parsed.append(0.0)
                continue
            try:
                parsed.append(float(value))
            except ValueError:
                return OperatorBurnPlan(), (f"Burn {row_idx}: {OPERATOR_BURN_HEADERS[col_idx]} must be numeric.",)
        burns.append(
            OperatorBurn(
                time_s=float(parsed[0]),
                delta_v_ric_m_s=(float(parsed[1]), float(parsed[2]), float(parsed[3])),
            )
        )
    plan = OperatorBurnPlan(burns=tuple(sorted(burns, key=lambda burn: burn.time_s)))
    errors = validate_operator_burn_plan(
        plan,
        total_delta_v_budget_m_s=_operator_delta_v_budget_m_s(option),
        max_time_s=option.time_budget_s,
    )
    return plan, errors


def _operator_rows_from_plan(plan: OperatorBurnPlan | None) -> list[list[str]]:
    if plan is None or not plan.burns:
        return [["", "", "", ""]]
    rows: list[list[str]] = []
    for burn in plan.burns[:OPERATOR_BURN_MAX_ROWS]:
        r, i, c = burn.delta_v_ric_m_s
        rows.append([
            f"{float(burn.time_s):g}",
            f"{float(r):g}",
            f"{float(i):g}",
            f"{float(c):g}",
        ])
    return rows or [["", "", "", ""]]


def _operator_new_burn_row_from_probe(trajectory_probe: OperatorTrajectoryProbe | None = None) -> list[str]:
    if trajectory_probe is None:
        return ["", "", "", ""]
    return [f"{max(float(trajectory_probe.time_s), 0.0):g}", "", "", ""]


def _operator_plan_status(plan: OperatorBurnPlan, *, option: GameScenarioOption) -> str:
    budget = _operator_delta_v_budget_m_s(option)
    if budget is None:
        budget_text = "no total dV limit"
    else:
        budget_text = f"{budget:.1f} m/s total dV budget"
    return f"{len(plan.burns)} burns | {plan.total_delta_v_m_s:.2f} m/s planned | 5.0 m/s max per burn | {budget_text}"


def _operator_delta_v_budget_m_s(option: GameScenarioOption) -> float | None:
    controlled_id = str(option.controlled_object_id or "").strip()
    target_id = str(option.target_object_id or "").strip()
    if controlled_id and target_id and controlled_id == target_id and option.target_delta_v_budget_m_s is not None:
        return option.target_delta_v_budget_m_s
    return option.delta_v_budget_m_s


def _operator_field_accepts_text(value: str) -> bool:
    return all(char.isdigit() or char in ".-+" for char in str(value))


def _operator_next_cell(
    active_cell: tuple[int, int],
    *,
    rows: list[list[str]],
    backwards: bool = False,
) -> tuple[int, int]:
    row, col = active_cell
    linear = row * 4 + col + (-1 if backwards else 1)
    linear = max(0, min(linear, max(len(rows) * 4 - 1, 0)))
    return (linear // 4, linear % 4)


def _operator_cell_at_pos(
    pos: tuple[int, int],
    field_rects: list[list[Any]],
    *,
    table_rect: Any | None = None,
) -> tuple[int, int] | None:
    if table_rect is not None:
        table = _RectSpec(*table_rect) if not hasattr(table_rect, "height") else table_rect
        body_top = int(table.y) + 52
        if not _pos_in_bounds(pos, (int(table.x), body_top, int(table.width), max(int(table.height) - 74, 1))):
            return None
    for row_idx, row in enumerate(field_rects):
        for col_idx, rect in enumerate(row):
            if _pos_in_bounds(pos, (rect.x, rect.y, rect.w, rect.h)):
                return (row_idx, col_idx)
    return None


def _operator_burn_table_rect(screen_width: int, screen_height: int) -> tuple[int, int, int, int]:
    left, _right = _operator_game_plot_panel_rects(screen_width, screen_height)
    button_group_left = int(screen_width) - 52 - 126 - 10 - 120
    y = left.bottom + 10
    bottom_margin = 20
    min_height = 74 + OPERATOR_BURN_TABLE_MIN_VISIBLE_ROWS * OPERATOR_BURN_ROW_HEIGHT
    height = max(int(screen_height) - y - bottom_margin, min_height)
    width = max(min(button_group_left - left.x - 24, int(screen_width) - left.x - 72), left.width)
    return (left.x, y, width, height)


def _operator_add_burn_button_rect(pygame: Any, table_rect: Any) -> Any:
    table = pygame.Rect(*table_rect)
    return pygame.Rect(table.right - 34, table.y + 7, 24, 24)


def _operator_plan_panel_rect(screen_width: int, screen_height: int) -> tuple[int, int, int, int]:
    return (42, 104, max(int(screen_width) - 84, 620), max(int(screen_height) - 200, 360))


def _operator_game_plot_panel_rects(screen_width: int, screen_height: int) -> tuple[Any, Any]:
    left = _RectSpec(
        36,
        88,
        max((int(screen_width) - 108) // 2, 200),
        max(int(screen_height) - 256, 250),
    )
    right = _RectSpec(left.right + 36, left.y, left.width, left.height)
    return left, right


def _operator_objectives_button_rect(screen_width: int, screen_height: int) -> tuple[int, int, int, int]:
    del screen_height
    width = 196
    height = 36
    return (max(int(screen_width) - width - 52, 0), 34, width, height)


def _operator_equation_sheet_button_rect(screen_width: int, screen_height: int) -> tuple[int, int, int, int]:
    width = 120 + 10 + 126
    height = 36
    right = int(screen_width) - 52
    return (max(right - width, 0), max(int(screen_height) - 116, 0), width, height)


def _operator_objectives_overlay_rect(screen_width: int, screen_height: int) -> tuple[int, int, int, int]:
    left, right = _operator_game_plot_panel_rects(screen_width, screen_height)
    x = int(left.x) + 28
    y = int(left.y) + 34
    w = max(int(right.right) - x - 28, 320)
    h = max(min(int(left.height) - 68, int(screen_height) - y - 142), 220)
    return (x, y, w, h)


def _operator_objectives_content_rect(overlay_rect: Any) -> tuple[int, int, int, int]:
    rect = _RectSpec(*overlay_rect) if not hasattr(overlay_rect, "height") else overlay_rect
    inset = 22
    title_height = 52
    scrollbar_gutter = 18
    x = int(rect.x) + inset
    y = int(rect.y) + title_height
    width = max(int(rect.width) - inset * 2 - scrollbar_gutter, 60)
    height = max(int(rect.height) - title_height - 18, 40)
    return (x, y, width, height)


def _operator_objectives_content_height(
    option: GameScenarioOption,
    training_config: RPOTrainingConfig | None,
    *,
    font: Any,
    width_px: int,
) -> int:
    y = 0
    y += len(_wrapped_budget_lines(option, font, width_px)) * PREVIEW_LINE_HEIGHT
    if y:
        y += PREVIEW_SECTION_GAP
    y = _section_height(option.learning_goal, font, y, width_px)
    y = _section_height(option.player_brief or option.description, font, y + PREVIEW_SECTION_GAP, width_px)
    y = _bullets_height(option.pass_criteria, font, y + PREVIEW_SECTION_GAP, width_px)
    y = _bullets_height(option.instructor_notes, font, y + PREVIEW_SECTION_GAP, width_px)
    numeric_targets = _operator_objective_numeric_targets(option, training_config)
    if numeric_targets:
        y = _bullets_height(numeric_targets, font, y + PREVIEW_SECTION_GAP, width_px)
    return max(y, 1)


def _clamp_operator_objectives_scroll_px(scroll_px: int, *, content_height: int, viewport_height: int) -> int:
    max_scroll = max(int(content_height) - max(int(viewport_height), 1), 0)
    return int(max(0, min(int(scroll_px), max_scroll)))


def _operator_objective_numeric_targets(
    option: GameScenarioOption,
    training_config: RPOTrainingConfig | None,
) -> tuple[str, ...]:
    cfg = training_config
    targets: list[str] = []
    if cfg is not None and cfg.enabled:
        if _operator_is_finite(cfg.goal_nmt_radial_amplitude_km):
            targets.append(f"Desired radial amplitude: {format_distance_km(float(cfg.goal_nmt_radial_amplitude_km))}")
        if _operator_is_finite(cfg.goal_nmt_cross_track_amplitude_km) and abs(
            float(cfg.goal_nmt_cross_track_amplitude_km)
        ) > 0.0:
            targets.append(
                f"Desired cross-track amplitude: {format_distance_km(float(cfg.goal_nmt_cross_track_amplitude_km))}"
            )
        if _operator_is_finite(cfg.goal_nmt_element_tolerance_km):
            targets.append(f"NMT amplitude tolerance: {format_distance_km(float(cfg.goal_nmt_element_tolerance_km))}")
        if _operator_is_finite(cfg.goal_nmt_velocity_tolerance_km_s):
            targets.append(
                f"NMT velocity tolerance: {format_speed_km_s(float(cfg.goal_nmt_velocity_tolerance_km_s))}"
            )
        if _operator_is_finite(cfg.goal_nmt_cross_track_phase_deg) and abs(
            float(cfg.goal_nmt_cross_track_phase_deg)
        ) > 0.0:
            targets.append(f"Desired cross-track phase: {_operator_format_degrees(cfg.goal_nmt_cross_track_phase_deg)}")
        if _operator_is_finite(cfg.goal_range_km):
            targets.append(f"Goal range: {format_distance_km(float(cfg.goal_range_km))}")
        elif _operator_is_finite(option.goal_range_km):
            targets.append(f"Goal range: {format_distance_km(float(option.goal_range_km))}")
        if _operator_is_finite(cfg.goal_range_tolerance_km):
            targets.append(f"Goal range tolerance: {format_distance_km(float(cfg.goal_range_tolerance_km))}")
        if _operator_is_finite(cfg.goal_radius_km):
            targets.append(f"Goal radius: {format_distance_km(float(cfg.goal_radius_km))}")
        if _operator_is_finite(cfg.max_goal_speed_km_s):
            targets.append(f"Max goal speed: {format_speed_km_s(float(cfg.max_goal_speed_km_s))}")
        if _operator_is_finite(cfg.keepout_radius_km):
            targets.append(f"Keepout radius: {format_distance_km(float(cfg.keepout_radius_km))}")
        if _operator_is_finite(cfg.hard_speed_limit_radius_km) and _operator_is_finite(cfg.hard_speed_limit_km_s):
            targets.append(
                "Speed gate: "
                f"{format_speed_km_s(float(cfg.hard_speed_limit_km_s))} inside "
                f"{format_distance_km(float(cfg.hard_speed_limit_radius_km))}"
            )
        if _operator_is_finite(cfg.max_cross_track_amplitude_km):
            targets.append(f"Max cross-track amplitude: {format_distance_km(float(cfg.max_cross_track_amplitude_km))}")
        if _operator_is_finite(cfg.max_target_reference_range_km):
            targets.append(
                f"Max target reference range: {format_distance_km(float(cfg.max_target_reference_range_km))}"
            )
        for gate in cfg.approach_gates:
            gate_targets = [
                f"R {format_distance_km(float(gate.radial_ric_km))}",
                f"+/- {format_distance_km(float(gate.radial_tolerance_km))}",
            ]
            if _operator_is_finite(gate.max_abs_intrack_km):
                gate_targets.append(f"|I| <= {format_distance_km(float(gate.max_abs_intrack_km))}")
            if _operator_is_finite(gate.max_abs_cross_track_km):
                gate_targets.append(f"|C| <= {format_distance_km(float(gate.max_abs_cross_track_km))}")
            if _operator_is_finite(gate.max_total_speed_km_s):
                gate_targets.append(f"speed <= {format_speed_km_s(float(gate.max_total_speed_km_s))}")
            targets.append(f"{gate.name}: " + ", ".join(gate_targets))
        for gate in cfg.inspection_gates:
            center = ", ".join(format_distance_km(float(value)) for value in np.asarray(gate.center_ric_km).reshape(3))
            half_width = ", ".join(
                format_distance_km(float(value)) for value in np.asarray(gate.half_width_ric_km).reshape(3)
            )
            text = f"{gate.name}: center ({center}), half-width ({half_width})"
            if _operator_is_finite(gate.max_total_speed_km_s):
                text += f", speed <= {format_speed_km_s(float(gate.max_total_speed_km_s))}"
            targets.append(text)
        for constraint in cfg.sun_angle_constraints:
            text = f"{constraint.name}: half-angle {_operator_format_degrees(constraint.allowed_half_angle_deg)}"
            if _operator_is_finite(constraint.min_range_km):
                text += f", min range {format_distance_km(float(constraint.min_range_km))}"
            if _operator_is_finite(constraint.max_range_km):
                text += f", max range {format_distance_km(float(constraint.max_range_km))}"
            targets.append(text)
        for burn in cfg.required_phase_burns:
            targets.append(
                f"{burn.label}: |R| {format_distance_km(float(burn.radial_abs_km))} "
                f"+/- {format_distance_km(float(burn.radial_tolerance_km))}, "
                f"|I| <= {format_distance_km(float(burn.max_abs_intrack_km))}"
            )
        if _operator_is_finite(cfg.max_time_s):
            targets.append(f"Time budget: {_operator_format_seconds(cfg.max_time_s)}")
        if _operator_is_finite(cfg.max_delta_v_m_s):
            targets.append(f"Chaser delta-v budget: {format_speed_m_s(float(cfg.max_delta_v_m_s))}")
        if _operator_is_finite(cfg.max_target_delta_v_m_s):
            targets.append(f"Target delta-v budget: {format_speed_m_s(float(cfg.max_target_delta_v_m_s))}")
    else:
        if _operator_is_finite(option.goal_range_km):
            targets.append(f"Goal range: {format_distance_km(float(option.goal_range_km))}")
        if _operator_is_finite(option.goal_speed_km_s):
            targets.append(f"Max goal speed: {format_speed_km_s(float(option.goal_speed_km_s))}")
        if _operator_is_finite(option.time_budget_s):
            targets.append(f"Time budget: {_operator_format_seconds(float(option.time_budget_s))}")
        if _operator_is_finite(option.delta_v_budget_m_s):
            targets.append(f"Chaser delta-v budget: {format_speed_m_s(float(option.delta_v_budget_m_s))}")
        if _operator_is_finite(option.target_delta_v_budget_m_s):
            targets.append(f"Target delta-v budget: {format_speed_m_s(float(option.target_delta_v_budget_m_s))}")
    return tuple(dict.fromkeys(targets))


def _operator_equation_sheet_content_height(font: Any, *, width_px: int) -> int:
    text_width = _operator_equation_sheet_text_width(width_px)
    y = 0
    y = _section_height(
        "Circular-chief HCW intuition for the local RIC frame.",
        font,
        y,
        text_width,
    )
    y = _operator_lines_height(
        PygameRPODashboard._pause_overlay_equation_lines(),
        font,
        y + PREVIEW_SECTION_GAP,
        text_width,
    )
    y = _operator_lines_height(
        PygameRPODashboard._pause_overlay_takeaway_lines(),
        font,
        y + PREVIEW_SECTION_GAP,
        text_width,
    )
    return max(y, 1)


def _operator_equation_sheet_text_width(width_px: int) -> int:
    column_gap = 28
    if int(width_px) >= 760:
        return max((int(width_px) - column_gap) // 2, 240)
    return int(width_px)


def _operator_lines_height(items: tuple[str, ...], font: Any, y: int, width_px: int) -> int:
    y += PREVIEW_SECTION_TITLE_GAP
    for item in items:
        y += len(_wrap_text_px(item, font, width_px)) * PREVIEW_LINE_HEIGHT
    return y


def _operator_is_finite(value: Any) -> bool:
    return value is not None and bool(np.isfinite(float(value)))


def _operator_format_seconds(value_s: float) -> str:
    value = float(value_s)
    if not np.isfinite(value):
        return "--"
    return f"{value:.0f} s"


def _operator_format_degrees(value_deg: float) -> str:
    value = float(value_deg)
    if not np.isfinite(value):
        return "--"
    text = f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{text} deg"


def _operator_table_visible_rows(table_rect: Any) -> int:
    table = _RectSpec(*table_rect) if not hasattr(table_rect, "height") else table_rect
    return max(1, int((int(table.height) - 74) // OPERATOR_BURN_ROW_HEIGHT))


def _operator_max_table_scroll_row(row_count: int, table_rect: Any) -> int:
    return max(int(row_count) - _operator_table_visible_rows(table_rect), 0)


def _operator_scroll_for_active_row(
    active_row: int,
    scroll_row: int,
    *,
    row_count: int,
    table_rect: Any,
) -> int:
    max_scroll = _operator_max_table_scroll_row(row_count, table_rect)
    scroll = int(max(0, min(int(scroll_row), max_scroll)))
    visible = _operator_table_visible_rows(table_rect)
    if int(active_row) < scroll:
        return int(active_row)
    if int(active_row) >= scroll + visible:
        return int(min(max_scroll, int(active_row) - visible + 1))
    return scroll


def _operator_burn_field_rects(
    pygame: Any,
    table_rect: Any,
    *,
    row_count: int,
    scroll_row: int = 0,
) -> list[list[Any]]:
    table = pygame.Rect(*table_rect)
    gap = 8
    label_width = 24
    scrollbar_width = 14
    delete_width = 30
    usable = max(table.width - label_width - scrollbar_width - delete_width - gap * 4, 1)
    widths = [
        int(round(usable * 0.23)),
        int(round(usable * 0.25)),
        int(round(usable * 0.25)),
        usable - int(round(usable * 0.23)) - int(round(usable * 0.25)) * 2,
    ]
    rects: list[list[Any]] = []
    y = table.y + 52 - int(scroll_row) * OPERATOR_BURN_ROW_HEIGHT
    for _ in range(int(row_count)):
        x = table.x + label_width
        row: list[Any] = []
        for width in widths:
            row.append(pygame.Rect(x, y, max(int(width), 48), 28))
            x += int(width) + gap
        rects.append(row)
        y += OPERATOR_BURN_ROW_HEIGHT
    return rects


def _operator_burn_delete_rects(
    pygame: Any,
    table_rect: Any,
    *,
    row_count: int,
    scroll_row: int = 0,
) -> list[Any]:
    table = pygame.Rect(*table_rect)
    rects: list[Any] = []
    y = table.y + 54 - int(scroll_row) * OPERATOR_BURN_ROW_HEIGHT
    x = table.right - 42
    for _ in range(int(row_count)):
        rects.append(pygame.Rect(x, y, 22, 22))
        y += OPERATOR_BURN_ROW_HEIGHT
    return rects


def _operator_delete_row_at_pos(
    pos: tuple[int, int],
    delete_rects: list[Any],
    *,
    table_rect: Any | None = None,
) -> int | None:
    if table_rect is not None:
        table = _RectSpec(*table_rect) if not hasattr(table_rect, "height") else table_rect
        body_top = int(table.y) + 52
        if not _pos_in_bounds(pos, (int(table.x), body_top, int(table.width), max(int(table.height) - 74, 1))):
            return None
    for row_idx, rect in enumerate(delete_rects):
        if _pos_in_bounds(pos, (rect.x, rect.y, rect.w, rect.h)):
            return int(row_idx)
    return None


def _operator_initial_relative_ric_state(config_path: Path) -> tuple[float, float, float] | None:
    context = _operator_plot_context(config_path)
    if context.initial_relative_ric_km_s is None:
        return None
    return tuple(context.initial_relative_ric_km_s[:3])


def _operator_plot_context(config_path: Path, *, difficulty: str = "easy") -> OperatorPlotContext:
    try:
        with Path(config_path).open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except OSError:
        return OperatorPlotContext()
    metadata = dict(raw.get("metadata", {}) or {})
    try:
        training_config = RPOTrainingConfig.from_metadata(metadata)
        training_config = training_config_for_game_mode(training_config, game_mode="operator")
    except (TypeError, ValueError):
        training_config = None
    game = dict(metadata.get("game", {}) or {})
    training = dict(game.get("training", {}) or {})
    common_context: dict[str, Any] = {
        "training_config": training_config,
        "camera_mode": str(game.get("camera_mode", "reference") or "reference"),
        "coast_prediction_model": str(game.get("coast_prediction_model", "hcw") or "hcw").strip().lower(),
        "cr3bp_projection_mode": str(game.get("cr3bp_projection_mode", "nonlinear") or "nonlinear"),
        "cr3bp_coast_prediction_horizon_s": _operator_positive_float_or_none(
            game.get("cr3bp_coast_prediction_horizon_s")
        ),
        "cr3bp_coast_prediction_dt_s": _operator_positive_float_or_none(game.get("cr3bp_coast_prediction_dt_s")),
        "target_centered_plot_planes": _operator_game_plane_tuple(game.get("target_centered_plot_planes", ())),
        "target_centered_plot_axes": _operator_game_target_centered_plot_axes(game),
        "plot_overlays_in_zoom": bool(game.get("plot_overlays_in_zoom", True)),
        "plot_overlays_in_zoom_by_plane": _operator_game_plot_overlays_in_zoom_by_plane(game),
        "plot_axis_scale": _operator_game_plot_axis_scale(game),
        "plot_fixed_axis_half_span_km": _operator_game_plot_fixed_axis_half_span_km(game),
        "plot_equal_axis_scale_planes": _operator_game_plane_tuple(game.get("plot_equal_axis_scale_planes", ())),
        "proximity_ring_plot_planes": _operator_game_plane_tuple(
            game.get("proximity_ring_plot_planes", ("RI", "RC", "IC"))
        )
        or ("RI", "RC", "IC"),
    }
    preview_training_cfg, preview_snapshot, preview_dashboard_kwargs = _operator_pilot_first_frame_preview(
        Path(config_path),
        difficulty=difficulty,
    )
    if preview_training_cfg is not None:
        common_context["training_config"] = preview_training_cfg
    if preview_dashboard_kwargs:
        common_context["coast_prediction_model"] = str(
            preview_dashboard_kwargs.get("coast_prediction_model", common_context["coast_prediction_model"])
        )
        common_context["cr3bp_projection_mode"] = str(
            preview_dashboard_kwargs.get("cr3bp_projection_mode", common_context["cr3bp_projection_mode"])
        )
    reference_state = _operator_reference_state_from_preview(preview_snapshot, preview_dashboard_kwargs)
    if reference_state is not None:
        common_context["reference_state_eci_km_s"] = reference_state
    common_context["pilot_initial_snapshot"] = preview_snapshot
    common_context["pilot_dashboard_kwargs"] = preview_dashboard_kwargs
    chaser_id = str(training.get("chaser_object_id", game.get("controlled_object_id", "chaser")) or "chaser")
    objects = dict(raw.get("objects", {}) or {})
    chaser = dict(objects.get(chaser_id, {}) or {})
    initial = dict(chaser.get("initial_state", {}) or {})
    relative = dict(initial.get("relative_to_target_ric", {}) or initial.get("relative_ric", {}) or {})
    state = relative.get("state")
    if state is None:
        return OperatorPlotContext(**common_context)
    try:
        values = [float(value) for value in state[:6]]
    except (TypeError, ValueError):
        return OperatorPlotContext(**common_context)
    if len(values) < 3:
        return OperatorPlotContext(**common_context)
    while len(values) < 6:
        values.append(0.0)
    rel6 = (values[0], values[1], values[2], values[3], values[4], values[5])
    mean_motion = _operator_target_mean_motion_rad_s(raw, training_config=training_config)
    reference_state_for_coast = common_context.get("reference_state_eci_km_s")
    return OperatorPlotContext(
        **common_context,
        initial_relative_ric_km_s=rel6,
        mean_motion_rad_s=mean_motion,
        initial_coast_ric_km_s=_operator_initial_coast_path(
            rel6,
            mean_motion_rad_s=mean_motion,
            coast_prediction_model=str(common_context["coast_prediction_model"]),
            chief_state_eci=(
                None if reference_state_for_coast is None else np.asarray(reference_state_for_coast, dtype=float)
            ),
            cr3bp_projection_mode=str(common_context["cr3bp_projection_mode"]),
            cr3bp_horizon_s=common_context.get("cr3bp_coast_prediction_horizon_s"),
        ),
    )


def _operator_reference_state_from_preview(
    snapshot: Any | None,
    dashboard_kwargs: dict[str, Any],
) -> tuple[float, float, float, float, float, float] | None:
    if snapshot is None or not dashboard_kwargs:
        return None
    reference_id = str(dashboard_kwargs.get("reference_object_id", "") or dashboard_kwargs.get("target_object_id", ""))
    if not reference_id:
        return None
    truth = getattr(snapshot, "truth", {}) or {}
    try:
        state = truth.get(reference_id)
    except AttributeError:
        return None
    if state is None:
        return None
    arr = np.asarray(state, dtype=float).reshape(-1)
    if arr.size < 6 or not np.all(np.isfinite(arr[:6])):
        return None
    return tuple(float(value) for value in arr[:6])


def _operator_pilot_first_frame_preview(
    config_path: Path,
    *,
    difficulty: str = "easy",
) -> tuple[RPOTrainingConfig | None, Any | None, dict[str, Any]]:
    try:
        from sim.api import SimulationConfig
        from sim.game.manual import KeyboardCommandState
        from sim.game.runner import (
            _coast_prediction_orbit_fraction,
            _dashboard_object_ids,
            _force_game_acceleration_off_config,
            _game_camera_mode,
            _game_camera_rule_mode,
            _game_camera_rule_toggle_enabled,
            _game_chaser_sprite_diameter_km,
            _game_chaser_sprite_path,
            _game_coast_prediction_model,
            _game_control_mode,
            _game_controlled_object_id,
            _game_cr3bp_active_prediction_horizon_s,
            _game_cr3bp_coast_prediction_dt_s,
            _game_cr3bp_coast_prediction_horizon_mode,
            _game_cr3bp_coast_prediction_horizon_s,
            _game_cr3bp_projection_mode,
            _game_plot_axis_scale,
            _game_plot_equal_axis_scale_planes,
            _game_plot_fixed_axis_half_span_km,
            _game_plot_overlays_in_zoom,
            _game_plot_overlays_in_zoom_by_plane,
            _game_plot_prediction_full_trajectory_only,
            _game_plot_prediction_in_zoom,
            _game_plot_prediction_zoom_max_span_km,
            _game_proximity_ring_plot_planes,
            _game_relative_frame,
            _game_ric_reference_object_id,
            _game_show_target_hcw_path,
            _game_target_centered_plot_axes,
            _game_target_centered_plot_planes,
            _game_target_coast_prediction_dt_s,
            _game_target_coast_prediction_horizon_s,
            _game_target_sprite_diameter_km,
            _game_target_sprite_path,
            _game_timed_input_accumulator_enabled,
            _game_visual_extrapolation_enabled,
            _start_game_attempt,
            _training_config_with_sun_environment,
        )
    except Exception:
        return None, None, {}

    try:
        config = _force_game_acceleration_off_config(SimulationConfig.from_yaml(config_path))
        training_cfg = _training_config_with_sun_environment(
            RPOTrainingConfig.from_metadata(dict(config.scenario.metadata or {})),
            config,
        )
        training_cfg = training_config_for_game_mode(training_cfg, game_mode="operator")
        controlled_object_id = _game_controlled_object_id(config)
        command_state = KeyboardCommandState()
        command_state.use_timing_accumulator = _game_timed_input_accumulator_enabled(config)
        _session, _provider, snapshot = _start_game_attempt(
            config,
            command_state=command_state,
            training_cfg=training_cfg,
            controlled_object_id=controlled_object_id,
            attitude_rate_deg_s=45.0,
            control_mode=_game_control_mode(config),
            ric_reference_object_id=_game_ric_reference_object_id(config, training_cfg.target_object_id),
            operator_burn_plan=None,
        )
        anim_cfg = dict(config.scenario.outputs.animations or {})
        dashboard_target_id, dashboard_chaser_id = _dashboard_object_ids(training_cfg, anim_cfg)
        dashboard_kwargs = {
            "target_object_id": dashboard_target_id,
            "chaser_object_id": dashboard_chaser_id,
            "controlled_object_id": controlled_object_id,
            "reference_object_id": _game_ric_reference_object_id(config, training_cfg.target_object_id),
            "relative_frame": _game_relative_frame(config),
            "keepout_radius_km": training_cfg.keepout_radius_km,
            "goal_range_km": training_cfg.goal_range_km,
            "goal_range_tolerance_km": training_cfg.goal_range_tolerance_km,
            "goal_radius_km": training_cfg.goal_radius_km,
            "hard_speed_limit_radius_km": training_cfg.hard_speed_limit_radius_km,
            "hard_speed_limit_km_s": training_cfg.hard_speed_limit_km_s,
            "goal_relative_ric_km": training_cfg.goal_relative_ric_km,
            "goal_nmt_radial_amplitude_km": training_cfg.goal_nmt_radial_amplitude_km,
            "goal_nmt_cross_track_amplitude_km": training_cfg.goal_nmt_cross_track_amplitude_km,
            "goal_nmt_cross_track_phase_deg": training_cfg.goal_nmt_cross_track_phase_deg,
            "goal_nmt_center_ric_km": training_cfg.goal_nmt_center_ric_km,
            "goal_nmt_element_tolerance_km": training_cfg.goal_nmt_element_tolerance_km,
            "coast_prediction_orbit_fraction": _coast_prediction_orbit_fraction(difficulty),
            "coast_prediction_model": _game_coast_prediction_model(config),
            "cr3bp_projection_mode": _game_cr3bp_projection_mode(config),
            "cr3bp_coast_prediction_horizon_s": _game_cr3bp_coast_prediction_horizon_s(config) or 21600.0,
            "cr3bp_active_prediction_horizon_s": _game_cr3bp_active_prediction_horizon_s(config),
            "cr3bp_coast_prediction_horizon_mode": _game_cr3bp_coast_prediction_horizon_mode(config),
            "cr3bp_coast_prediction_dt_s": _game_cr3bp_coast_prediction_dt_s(config) or 300.0,
            "target_coast_prediction_horizon_s": _game_target_coast_prediction_horizon_s(config),
            "target_coast_prediction_dt_s": _game_target_coast_prediction_dt_s(config),
            "forbidden_regions": training_cfg.forbidden_regions,
            "approach_gates": training_cfg.approach_gates,
            "inspection_gates": training_cfg.inspection_gates,
            "sun_angle_constraints": training_cfg.sun_angle_constraints,
            "plot_overlays_in_zoom": _game_plot_overlays_in_zoom(config),
            "plot_overlays_in_zoom_by_plane": _game_plot_overlays_in_zoom_by_plane(config),
            "plot_prediction_in_zoom": _game_plot_prediction_in_zoom(config),
            "plot_prediction_zoom_max_span_km": _game_plot_prediction_zoom_max_span_km(config),
            "plot_prediction_full_trajectory_only": _game_plot_prediction_full_trajectory_only(config),
            "plot_axis_scale": _game_plot_axis_scale(config),
            "plot_fixed_axis_half_span_km": _game_plot_fixed_axis_half_span_km(config),
            "plot_equal_axis_scale_planes": _game_plot_equal_axis_scale_planes(config),
            "target_centered_plot_planes": _game_target_centered_plot_planes(config),
            "target_centered_plot_axes": _game_target_centered_plot_axes(config),
            "proximity_ring_plot_planes": _game_proximity_ring_plot_planes(config),
            "target_reference_object_id": training_cfg.target_reference_object_id,
            "camera_mode": _game_camera_mode(config),
            "camera_rule_mode": _game_camera_rule_mode(config),
            "camera_rule_toggle_enabled": _game_camera_rule_toggle_enabled(config),
            "target_sprite_path": _game_target_sprite_path(config),
            "chaser_sprite_path": _game_chaser_sprite_path(config),
            "target_sprite_diameter_km": _game_target_sprite_diameter_km(config),
            "chaser_sprite_diameter_km": _game_chaser_sprite_diameter_km(config),
            "show_target_coast_prediction": _game_show_target_hcw_path(config),
            "visual_extrapolation_enabled": _game_visual_extrapolation_enabled(config),
        }
        return training_cfg, snapshot, dashboard_kwargs
    except Exception:
        return None, None, {}


def _operator_game_plane_tuple(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, str):
        values = [raw]
    else:
        try:
            values = list(raw)
        except TypeError:
            return ()
    planes: list[str] = []
    for value in values:
        plane = str(value or "").strip().upper()
        if plane in {"RI", "RC", "IC"} and plane not in planes:
            planes.append(plane)
    return tuple(planes)


def _operator_game_target_centered_plot_axes(game: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    raw = game.get("target_centered_plot_axes", {}) or {}
    if not isinstance(raw, dict):
        return {}
    parsed: dict[str, tuple[str, ...]] = {}
    for plane, value in raw.items():
        key = str(plane or "").strip().upper()
        if key not in {"RI", "RC", "IC"}:
            continue
        if isinstance(value, str):
            values = [value]
        else:
            try:
                values = list(value)
            except TypeError:
                continue
        axes: list[str] = []
        for raw_axis in values:
            axis = str(raw_axis or "").strip().lower()
            if axis in {"x", "y"} and axis not in axes:
                axes.append(axis)
        if axes:
            parsed[key] = tuple(axes)
    return parsed


def _operator_game_plot_overlays_in_zoom_by_plane(game: dict[str, Any]) -> dict[str, bool]:
    raw = game.get("plot_overlays_in_zoom_by_plane", {}) or {}
    if not isinstance(raw, dict):
        return {}
    parsed: dict[str, bool] = {}
    for plane, value in raw.items():
        key = str(plane or "").strip().upper()
        if key in {"RI", "RC", "IC"}:
            parsed[key] = bool(value)
    return parsed


def _operator_game_plot_axis_scale(game: dict[str, Any]) -> dict[str, tuple[float, float]]:
    raw = game.get("plot_axis_scale", {}) or {}
    if not isinstance(raw, dict):
        return {}
    parsed: dict[str, tuple[float, float]] = {}
    for plane, value in raw.items():
        key = str(plane or "").strip().upper()
        if key not in {"RI", "RC", "IC"}:
            continue
        if isinstance(value, dict):
            pair = (value.get("x", 1.0), value.get("y", 1.0))
        else:
            try:
                pair = tuple(value)
            except TypeError:
                continue
            if len(pair) != 2:
                continue
        try:
            x_scale = float(pair[0])
            y_scale = float(pair[1])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(x_scale) or x_scale <= 0.0:
            x_scale = 1.0
        if not np.isfinite(y_scale) or y_scale <= 0.0:
            y_scale = 1.0
        parsed[key] = (x_scale, y_scale)
    return parsed


def _operator_game_plot_fixed_axis_half_span_km(game: dict[str, Any]) -> dict[str, tuple[float | None, float | None]]:
    raw = game.get("plot_fixed_axis_half_span_km", {}) or {}
    if not isinstance(raw, dict):
        return {}
    parsed: dict[str, tuple[float | None, float | None]] = {}
    for plane, value in raw.items():
        key = str(plane or "").strip().upper()
        if key not in {"RI", "RC", "IC"}:
            continue
        if isinstance(value, dict):
            pair = (value.get("x"), value.get("y"))
        else:
            try:
                pair = tuple(value)
            except TypeError:
                continue
            if len(pair) != 2:
                continue
        x_span = _operator_positive_float_or_none(pair[0])
        y_span = _operator_positive_float_or_none(pair[1])
        if x_span is not None or y_span is not None:
            parsed[key] = (x_span, y_span)
    return parsed


def _operator_positive_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result) or result <= 0.0:
        return None
    return result


def _operator_target_mean_motion_rad_s(
    raw_config: dict[str, Any],
    *,
    training_config: RPOTrainingConfig | None,
) -> float:
    objects = dict(raw_config.get("objects", {}) or {})
    target_id = "target"
    if training_config is not None and training_config.target_object_id:
        target_id = str(training_config.target_object_id)
    target = dict(objects.get(target_id, objects.get("target", {})) or {})
    initial = dict(target.get("initial_state", {}) or {})
    coes = dict(initial.get("coes", {}) or {})
    semi_major_axis_km = _optional_float(coes.get("a_km"))
    if semi_major_axis_km is None or not np.isfinite(semi_major_axis_km) or semi_major_axis_km <= 0.0:
        semi_major_axis_km = 7000.0
    return float(np.sqrt(EARTH_MU_KM3_S2 / (float(semi_major_axis_km) ** 3)))


def _operator_initial_coast_path(
    rel6: tuple[float, float, float, float, float, float],
    *,
    mean_motion_rad_s: float,
    coast_prediction_model: str = "hcw",
    chief_state_eci: np.ndarray | None = None,
    cr3bp_projection_mode: str = "nonlinear",
    cr3bp_horizon_s: float | None = None,
) -> tuple[tuple[float, float, float, float, float, float], ...]:
    n = float(mean_motion_rad_s)
    if not np.isfinite(n) or n <= 0.0:
        return ()
    cr3bp = _coast_prediction_model_key(coast_prediction_model) == "cr3bp" and chief_state_eci is not None
    horizon_s = (
        float(cr3bp_horizon_s or 21600.0)
        if cr3bp
        else float(2.0 * np.pi / n)
    )
    times = np.linspace(0.0, horizon_s, 241)
    rows, _ = _operator_planned_coast_states(
        np.array(rel6, dtype=float),
        times,
        mean_motion_rad_s=n,
        coast_prediction_model=coast_prediction_model,
        chief_state_eci=chief_state_eci,
        cr3bp_projection_mode=cr3bp_projection_mode,
        current_time_s=0.0,
    )
    rows = rows[np.all(np.isfinite(rows), axis=1)]
    return tuple(tuple(float(value) for value in row[:6]) for row in rows)


def _operator_planned_trajectory(
    plot_context: OperatorPlotContext,
    plan: OperatorBurnPlan,
) -> tuple[np.ndarray, np.ndarray]:
    if not plan.burns or plot_context.initial_relative_ric_km_s is None:
        return np.empty((0, 6), dtype=float), np.empty((0, 6), dtype=float)
    n = plot_context.mean_motion_rad_s
    if n is None or not np.isfinite(float(n)) or float(n) <= 0.0:
        return np.empty((0, 6), dtype=float), np.empty((0, 6), dtype=float)
    cache_key = _operator_planned_trajectory_cache_key(plot_context, plan)
    cached = plot_context._planned_trajectory_cache.get(cache_key)
    if cached is not None:
        return cached

    cr3bp = _coast_prediction_model_key(plot_context.coast_prediction_model) == "cr3bp"
    tail_horizon_s = (
        float(plot_context.cr3bp_coast_prediction_horizon_s or 21600.0)
        if cr3bp
        else float(2.0 * np.pi / float(n))
    )
    horizon_s = max(float(plan.burns[-1].time_s), 0.0) + tail_horizon_s
    sample_dt_s = (
        max(
            float(plot_context.cr3bp_coast_prediction_dt_s or 1.0),
            horizon_s / 480.0,
            1.0,
        )
        if cr3bp
        else max(min(tail_horizon_s / 240.0, 120.0), 5.0)
    )
    state = np.array(plot_context.initial_relative_ric_km_s, dtype=float).reshape(6)
    chief_state = (
        np.array(plot_context.reference_state_eci_km_s, dtype=float).reshape(6)
        if plot_context.reference_state_eci_km_s is not None
        else None
    )
    current_t_s = 0.0
    rows: list[np.ndarray] = [state.copy()]
    time_rows: list[float] = [0.0]
    marker_rows: list[np.ndarray] = []

    for burn in plan.burns:
        burn_t_s = max(float(burn.time_s), current_t_s)
        state, chief_state = _operator_append_planned_coast(
            rows,
            time_rows=time_rows,
            state=state,
            chief_state_eci=chief_state,
            start_t_s=current_t_s,
            stop_t_s=burn_t_s,
            sample_dt_s=sample_dt_s,
            mean_motion_rad_s=float(n),
            coast_prediction_model=plot_context.coast_prediction_model,
            cr3bp_projection_mode=plot_context.cr3bp_projection_mode,
        )
        delta_v_km_s = np.asarray(burn.delta_v_ric_m_s, dtype=float).reshape(3) / 1000.0
        if np.all(np.isfinite(delta_v_km_s)):
            state = state.copy()
            state[3:6] += delta_v_km_s
            rows.append(state.copy())
            time_rows.append(float(burn_t_s))
            marker_rows.append(state.copy())
        current_t_s = burn_t_s

    _operator_append_planned_coast(
        rows,
        time_rows=time_rows,
        state=state,
        chief_state_eci=chief_state,
        start_t_s=current_t_s,
        stop_t_s=horizon_s,
        sample_dt_s=sample_dt_s,
        mean_motion_rad_s=float(n),
        coast_prediction_model=plot_context.coast_prediction_model,
        cr3bp_projection_mode=plot_context.cr3bp_projection_mode,
    )
    trajectory = np.vstack(rows) if rows else np.empty((0, 6), dtype=float)
    trajectory_times = np.asarray(time_rows, dtype=float).reshape(-1)
    markers = np.vstack(marker_rows) if marker_rows else np.empty((0, 6), dtype=float)
    finite = np.all(np.isfinite(trajectory), axis=1) if trajectory.size else np.empty(0, dtype=bool)
    if trajectory.size:
        trajectory = trajectory[finite]
        trajectory_times = trajectory_times[finite] if trajectory_times.size == finite.size else np.empty(0, dtype=float)
    marker_finite = np.all(np.isfinite(markers), axis=1) if markers.size else np.empty(0, dtype=bool)
    markers = markers[marker_finite] if markers.size else markers
    plot_context._planned_trajectory_cache[cache_key] = (trajectory, markers)
    plot_context._planned_trajectory_time_cache[cache_key] = trajectory_times
    return trajectory, markers


def _operator_planned_trajectory_times(plot_context: OperatorPlotContext, plan: OperatorBurnPlan) -> np.ndarray:
    cache_key = _operator_planned_trajectory_cache_key(plot_context, plan)
    times = plot_context._planned_trajectory_time_cache.get(cache_key)
    if times is None:
        _operator_planned_trajectory(plot_context, plan)
        times = plot_context._planned_trajectory_time_cache.get(cache_key)
    return np.asarray(times, dtype=float).reshape(-1) if times is not None else np.empty(0, dtype=float)


def _operator_append_planned_coast(
    rows: list[np.ndarray],
    *,
    time_rows: list[float] | None = None,
    state: np.ndarray,
    chief_state_eci: np.ndarray | None,
    start_t_s: float,
    stop_t_s: float,
    sample_dt_s: float,
    mean_motion_rad_s: float,
    coast_prediction_model: str,
    cr3bp_projection_mode: str = "nonlinear",
) -> tuple[np.ndarray, np.ndarray | None]:
    state_arr = np.array(state, dtype=float).reshape(6)
    chief_arr = None if chief_state_eci is None else np.array(chief_state_eci, dtype=float).reshape(6)
    duration_s = max(float(stop_t_s) - float(start_t_s), 0.0)
    if duration_s <= 1.0e-9:
        return state_arr, chief_arr
    interior_count = max(int(np.ceil(duration_s / max(float(sample_dt_s), 1.0e-6))), 1)
    times = np.linspace(0.0, duration_s, interior_count + 1, dtype=float)[1:]
    coast, chief_rows = _operator_planned_coast_states(
        state_arr,
        times,
        mean_motion_rad_s=float(mean_motion_rad_s),
        coast_prediction_model=coast_prediction_model,
        chief_state_eci=chief_arr,
        cr3bp_projection_mode=cr3bp_projection_mode,
        current_time_s=float(start_t_s),
    )
    for row in coast:
        rows.append(np.array(row, dtype=float).reshape(6).copy())
    if time_rows is not None:
        for t_s in times:
            time_rows.append(float(start_t_s) + float(t_s))
    chief_end = chief_rows[-1].copy() if chief_rows.size else _operator_propagate_reference_state(chief_arr, duration_s)
    return np.array(coast[-1], dtype=float).reshape(6), chief_end


def _operator_planned_coast_states(
    state: np.ndarray,
    times_s: np.ndarray,
    *,
    mean_motion_rad_s: float,
    coast_prediction_model: str,
    chief_state_eci: np.ndarray | None,
    cr3bp_projection_mode: str = "nonlinear",
    current_time_s: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(times_s, dtype=float).reshape(-1)
    if times.size == 0:
        return np.empty((0, 6), dtype=float), np.empty((0, 6), dtype=float)
    model_key = _coast_prediction_model_key(coast_prediction_model)
    if model_key == "cr3bp" and chief_state_eci is not None:
        chief = np.asarray(chief_state_eci, dtype=float).reshape(6)
        if _cr3bp_projection_mode_key(cr3bp_projection_mode) == "linearized":
            result = _linearized_cr3bp_moon_ric_coast_prediction(
                state,
                target_state=chief,
                times=times,
                current_t_s=float(current_time_s),
            )
        else:
            result = _nonlinear_cr3bp_moon_ric_coast_prediction(
                state,
                target_state=chief,
                times=times,
                current_t_s=float(current_time_s),
            )
        chief_rows = _operator_cr3bp_reference_coast_states(
            chief,
            times,
            current_time_s=float(current_time_s),
        )
        return result, chief_rows
    chief_rows = _operator_reference_coast_states(chief_state_eci, times)
    if _operator_uses_ya_planned_coast(coast_prediction_model) and chief_state_eci is not None:
        try:
            rows = []
            for t_s, chief_end in zip(times, chief_rows, strict=True):
                phi = _launcher_dep("ya_closed_form_transition_matrix", ya_closed_form_transition_matrix)(float(t_s), chief_state_eci, chief_end)
                rows.append(phi @ np.asarray(state, dtype=float).reshape(6))
            result = np.vstack(rows)
            if result.shape == (times.size, 6) and np.all(np.isfinite(result)):
                return result, chief_rows
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            pass
    return _cw_coast_states(np.array(state, dtype=float).reshape(6), times, float(mean_motion_rad_s)), chief_rows


def _operator_cr3bp_reference_coast_states(
    chief_state_eci: np.ndarray,
    times_s: np.ndarray,
    *,
    current_time_s: float,
) -> np.ndarray:
    from sim.dynamics.orbit.cr3bp import propagate_cr3bp_state

    state = np.asarray(chief_state_eci, dtype=float).reshape(6).copy()
    rows: list[np.ndarray] = []
    elapsed_s = 0.0
    current_t_s = float(current_time_s)
    for target_t_s in np.asarray(times_s, dtype=float).reshape(-1):
        step_s = max(float(target_t_s) - elapsed_s, 0.0)
        if step_s > 0.0:
            state = propagate_cr3bp_state(state, step_s, current_t_s)
            current_t_s += step_s
        rows.append(state.copy())
        elapsed_s = max(float(target_t_s), elapsed_s)
    return np.vstack(rows) if rows else np.empty((0, 6), dtype=float)


def _operator_uses_ya_planned_coast(coast_prediction_model: str) -> bool:
    return _coast_prediction_model_key(coast_prediction_model) in {"elliptic_linear", "tschauner_hempel", "ts"}


def _operator_reference_coast_states(chief_state_eci: np.ndarray | None, times_s: np.ndarray) -> np.ndarray:
    if chief_state_eci is None:
        return np.empty((0, 6), dtype=float)
    times = np.asarray(times_s, dtype=float).reshape(-1)
    if times.size == 0:
        return np.empty((0, 6), dtype=float)
    state = np.asarray(chief_state_eci, dtype=float).reshape(6).copy()
    rows: list[np.ndarray] = []
    previous_t_s = 0.0
    for target_t_s in times:
        duration_s = max(float(target_t_s) - previous_t_s, 0.0)
        propagated = _operator_propagate_reference_state(state, duration_s)
        if propagated is not None:
            state = propagated
        rows.append(state.copy())
        previous_t_s = max(float(target_t_s), previous_t_s)
    return np.vstack(rows) if rows else np.empty((0, 6), dtype=float)


def _operator_propagate_reference_state(chief_state_eci: np.ndarray | None, dt_s: float) -> np.ndarray | None:
    if chief_state_eci is None:
        return None
    state = np.asarray(chief_state_eci, dtype=float).reshape(6).copy()
    remaining_s = max(float(dt_s), 0.0)
    if remaining_s <= 1.0e-9:
        return state
    max_step_s = 10.0
    zero_accel = np.zeros(3, dtype=float)
    while remaining_s > 1.0e-9:
        step_s = min(max_step_s, remaining_s)
        state = propagate_two_body_rk4(state, step_s, EARTH_MU_KM3_S2, zero_accel)
        remaining_s -= step_s
    return state


def _operator_planned_trajectory_cache_key(
    plot_context: OperatorPlotContext,
    plan: OperatorBurnPlan,
) -> tuple[Any, ...]:
    initial = tuple(round(float(value), 12) for value in (plot_context.initial_relative_ric_km_s or ()))
    n = None if plot_context.mean_motion_rad_s is None else round(float(plot_context.mean_motion_rad_s), 15)
    reference = tuple(round(float(value), 12) for value in (plot_context.reference_state_eci_km_s or ()))
    burns = tuple(
        (
            round(float(burn.time_s), 6),
            tuple(round(float(value), 9) for value in burn.delta_v_ric_m_s),
        )
        for burn in plan.burns
    )
    return (
        initial,
        n,
        _coast_prediction_model_key(plot_context.coast_prediction_model),
        _cr3bp_projection_mode_key(plot_context.cr3bp_projection_mode),
        plot_context.cr3bp_coast_prediction_horizon_s,
        plot_context.cr3bp_coast_prediction_dt_s,
        reference,
        burns,
    )


def _operator_nmt_points(training_cfg: RPOTrainingConfig) -> np.ndarray:
    if training_cfg.goal_nmt_radial_amplitude_km is None:
        return np.empty((0, 3), dtype=float)
    return nmt_curve_points_km(
        radial_amplitude_km=float(training_cfg.goal_nmt_radial_amplitude_km),
        cross_track_amplitude_km=float(training_cfg.goal_nmt_cross_track_amplitude_km),
        cross_track_phase_deg=float(training_cfg.goal_nmt_cross_track_phase_deg),
        center_ric_km=np.array(training_cfg.goal_nmt_center_ric_km, dtype=float).reshape(3),
        samples=241,
    )


def _operator_nmt_boundary_points(training_cfg: RPOTrainingConfig) -> tuple[np.ndarray, ...]:
    radial = training_cfg.goal_nmt_radial_amplitude_km
    tol = training_cfg.goal_nmt_element_tolerance_km or training_cfg.goal_nmt_tolerance_km
    if radial is None or tol is None:
        return ()
    outer = float(radial) + max(float(tol), 0.0)
    inner = max(float(radial) - max(float(tol), 0.0), 0.0)
    points = [
        nmt_curve_points_km(
            radial_amplitude_km=outer,
            cross_track_amplitude_km=float(training_cfg.goal_nmt_cross_track_amplitude_km) + max(float(tol), 0.0),
            cross_track_phase_deg=float(training_cfg.goal_nmt_cross_track_phase_deg),
            center_ric_km=np.array(training_cfg.goal_nmt_center_ric_km, dtype=float).reshape(3),
            samples=241,
        )
    ]
    if inner > 0.0:
        points.append(
            nmt_curve_points_km(
                radial_amplitude_km=inner,
                cross_track_amplitude_km=max(
                    float(training_cfg.goal_nmt_cross_track_amplitude_km) - max(float(tol), 0.0),
                    0.0,
                ),
                cross_track_phase_deg=float(training_cfg.goal_nmt_cross_track_phase_deg),
                center_ric_km=np.array(training_cfg.goal_nmt_center_ric_km, dtype=float).reshape(3),
                samples=241,
            )
        )
    return tuple(points)


def _operator_plot_half_span(
    rows: list[np.ndarray],
    *,
    x_axis: int,
    y_axis: int,
    ring_radii: list[float | None],
) -> float:
    max_abs = 0.5
    for row in rows:
        arr = np.array(row, dtype=float)
        if arr.size == 0:
            continue
        arr = arr.reshape(-1, arr.shape[-1])
        if arr.shape[1] <= max(int(x_axis), int(y_axis)):
            continue
        projected = arr[:, [int(x_axis), int(y_axis)]]
        finite = projected[np.all(np.isfinite(projected), axis=1)]
        if finite.size:
            max_abs = max(max_abs, float(np.max(np.abs(finite))))
    for radius in ring_radii:
        if radius is not None and np.isfinite(float(radius)):
            max_abs = max(max_abs, float(radius))
    return max_abs * 1.25


def _operator_camera_center_ric(
    plot_context: OperatorPlotContext,
    *,
    chaser_current: np.ndarray,
    target_current: np.ndarray,
    x_axis: int,
    y_axis: int,
) -> np.ndarray:
    mode = str(plot_context.camera_mode or "reference").strip().lower()
    target = np.array(target_current, dtype=float).reshape(3)
    chaser = np.array(chaser_current, dtype=float).reshape(3)
    if mode not in {"target_pair", "satellite_pair", "pair"}:
        return np.zeros(3, dtype=float)
    plane = _plane_key_for_axes(x_axis=int(x_axis), y_axis=int(y_axis))
    if plane in plot_context.target_centered_plot_planes:
        return target
    if {int(x_axis), int(y_axis)} == {0, 2}:
        return np.zeros(3, dtype=float)
    center = _satellite_pair_camera_center(
        chaser=chaser,
        target=target,
        x_axis=int(x_axis),
        y_axis=int(y_axis),
        keep_rc_reference_centered=True,
    )
    override_axes = tuple(str(value or "").strip().lower() for value in plot_context.target_centered_plot_axes.get(plane, ()))
    if "x" in override_axes:
        center[int(x_axis)] = float(target[int(x_axis)])
    if "y" in override_axes:
        center[int(y_axis)] = float(target[int(y_axis)])
    return center


def _operator_forbidden_region_projection_points(
    training_cfg: RPOTrainingConfig,
    *,
    x_axis: int,
    y_axis: int,
    offset: np.ndarray,
) -> list[np.ndarray]:
    pts: list[np.ndarray] = []
    for region in training_cfg.forbidden_regions:
        if not _region_visible_on_plane(region, x_axis=int(x_axis), y_axis=int(y_axis)):
            continue
        if region.kind == "annular_sector":
            polygon = region.sector_polygon_ric()
            if polygon.size:
                polygon = polygon.copy()
                polygon[:, :3] += np.array(offset, dtype=float).reshape(1, 3)
                pts.append(polygon[:, [int(x_axis), int(y_axis)]])
            continue
        if region.kind == "cylinder":
            polygon = _cylinder_projection_polygon_ric(region, x_axis=int(x_axis), y_axis=int(y_axis))
            if polygon.size:
                polygon = polygon.copy()
                polygon[:, :3] += np.array(offset, dtype=float).reshape(1, 3)
                pts.append(polygon[:, [int(x_axis), int(y_axis)]])
            continue
        if region.kind == "sphere":
            polygon = _sphere_projection_polygon_ric(region, x_axis=int(x_axis), y_axis=int(y_axis))
            if polygon.size:
                polygon = polygon.copy()
                polygon[:, :3] += np.array(offset, dtype=float).reshape(1, 3)
                pts.append(polygon[:, [int(x_axis), int(y_axis)]])
            continue
        bounds = _finite_projected_region_bounds(region, x_axis=int(x_axis), y_axis=int(y_axis))
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
        corners += np.array([offset[int(x_axis)], offset[int(y_axis)]], dtype=float).reshape(1, 2)
        pts.append(corners)
    return pts


def _operator_minimum_plot_span_km(
    plot_context: OperatorPlotContext,
    *,
    x_axis: int,
    y_axis: int,
    target_current: np.ndarray,
    nmt: np.ndarray,
    nmt_bounds: tuple[np.ndarray, ...],
) -> float:
    plane = _plane_key_for_axes(x_axis=int(x_axis), y_axis=int(y_axis))
    include_overlays = bool(
        plot_context.plot_overlays_in_zoom_by_plane.get(plane, plot_context.plot_overlays_in_zoom)
    )
    if not include_overlays:
        return MIN_PLOT_SPAN_KM
    origin_offset = np.array(target_current, dtype=float).reshape(3)
    overlay_points = _operator_forbidden_region_projection_points(
        plot_context.training_config or RPOTrainingConfig(enabled=False),
        x_axis=int(x_axis),
        y_axis=int(y_axis),
        offset=origin_offset,
    )
    if nmt.size:
        overlay_points.append(nmt[:, [int(x_axis), int(y_axis)]] + origin_offset[[int(x_axis), int(y_axis)]].reshape(1, 2))
    for boundary in nmt_bounds:
        if boundary.size:
            overlay_points.append(
                boundary[:, [int(x_axis), int(y_axis)]]
                + origin_offset[[int(x_axis), int(y_axis)]].reshape(1, 2)
            )
    finite: list[np.ndarray] = []
    for points in overlay_points:
        projected = np.array(points, dtype=float).reshape(-1, 2)
        projected = projected[np.all(np.isfinite(projected), axis=1)]
        if projected.size:
            finite.append(projected)
    span = MIN_PLOT_SPAN_KM
    if finite:
        span = max(span, float(np.max(np.abs(np.vstack(finite)))) * PLOT_OVERLAY_MARGIN)
    return span


def _operator_axis_scales_for_plot(
    plot_context: OperatorPlotContext,
    plot: Any,
    *,
    pts: list[np.ndarray],
    min_span_km: float,
    x_axis: int,
    y_axis: int,
    screen_size: tuple[int, int],
) -> tuple[float, float]:
    plane = _plane_key_for_axes(x_axis=int(x_axis), y_axis=int(y_axis))
    axis_scale_x, axis_scale_y = plot_context.plot_axis_scale.get(plane, (1.0, 1.0))
    fixed_x, fixed_y = plot_context.plot_fixed_axis_half_span_km.get(plane, (None, None))
    if fixed_x is None and fixed_y is None:
        base_scale = _operator_scale_for_plot(pts=pts, min_span_km=min_span_km, screen_size=screen_size)
        scale_x = base_scale * axis_scale_x
        scale_y = base_scale * axis_scale_y
    else:
        scale_x = _operator_scale_for_axis(pts=pts, axis_index=0, min_span_km=min_span_km, plot_px=plot.width)
        scale_y = _operator_scale_for_axis(pts=pts, axis_index=1, min_span_km=min_span_km, plot_px=plot.height)
        scale_x *= axis_scale_x
        scale_y *= axis_scale_y
        if fixed_x is not None:
            scale_x = _operator_scale_for_fixed_half_span(plot_px=plot.width, half_span_km=fixed_x)
        if fixed_y is not None:
            scale_y = _operator_scale_for_fixed_half_span(plot_px=plot.height, half_span_km=fixed_y)
    if plane in plot_context.plot_equal_axis_scale_planes:
        scale_x = scale_y
    return (scale_x, scale_y)


def _operator_scale_for_plot(*, pts: list[np.ndarray], min_span_km: float, screen_size: tuple[int, int]) -> float:
    finite: list[np.ndarray] = []
    for arr in pts:
        projected = np.array(arr, dtype=float).reshape(-1, 2)
        projected = projected[np.all(np.isfinite(projected), axis=1)]
        if projected.size:
            finite.append(projected)
    span = float(max(min_span_km, MIN_PLOT_SPAN_KM))
    if finite:
        span = max(float(np.max(np.abs(np.vstack(finite)))) * 1.2, span)
    width, height = screen_size
    px_span = max(min(width, height) * 0.28, 80.0)
    return float(px_span / max(span, 1.0e-9))


def _operator_scale_for_axis(*, pts: list[np.ndarray], axis_index: int, min_span_km: float, plot_px: float) -> float:
    finite: list[np.ndarray] = []
    for arr in pts:
        projected = np.array(arr, dtype=float).reshape(-1, 2)
        projected = projected[np.all(np.isfinite(projected), axis=1)]
        if projected.size:
            finite.append(projected[:, int(axis_index)])
    span = float(max(min_span_km, MIN_PLOT_SPAN_KM))
    if finite:
        span = max(float(np.max(np.abs(np.concatenate(finite)))) * 1.2, span)
    px_span = max(float(plot_px) * 0.42, 80.0)
    return float(px_span / max(span, 1.0e-9))


def _operator_scale_for_fixed_half_span(*, plot_px: float, half_span_km: float) -> float:
    span = float(half_span_km)
    if not np.isfinite(span) or span <= 0.0:
        span = MIN_PLOT_SPAN_KM
    return float(max(float(plot_px) * 0.5, 1.0) / span)


def _operator_rows_to_px(rows: np.ndarray, *, to_px: Any) -> list[tuple[int, int]]:
    arr = np.array(rows, dtype=float)
    if arr.size == 0:
        return []
    return [to_px(row[:3]) for row in arr.reshape(-1, arr.shape[-1]) if row.size >= 3]


def _operator_plot_transform_to_px(
    transform: dict[str, Any],
    point_ric_km: np.ndarray,
    *,
    x_axis: int,
    y_axis: int,
) -> tuple[int, int] | None:
    try:
        plot = transform["plot"]
        center_x = int(plot[0]) + int(plot[2]) // 2
        center_y = int(plot[1]) + int(plot[3]) // 2
        camera_center = np.asarray(transform["camera_center"], dtype=float).reshape(3)
        scale_x = float(transform["scale_x"])
        scale_y = float(transform["scale_y"])
        x_display_sign = float(transform["x_display_sign"])
        y_display_sign = float(transform["y_display_sign"])
    except (KeyError, TypeError, ValueError):
        return None
    shifted = np.asarray(point_ric_km, dtype=float).reshape(-1)[:3] - camera_center
    return (
        center_x + int(round(float(shifted[int(x_axis)]) * x_display_sign * scale_x)),
        center_y - int(round(float(shifted[int(y_axis)]) * y_display_sign * scale_y)),
    )


def _operator_trajectory_probe_from_click(
    plot_context: OperatorPlotContext,
    plan: OperatorBurnPlan,
    pos: tuple[int, int],
    *,
    selected_probe: OperatorTrajectoryProbe | None = None,
) -> tuple[bool, np.ndarray | None, float | None]:
    dashboard = plot_context._preview_dashboard
    transforms = getattr(dashboard, "_frame_cache", {}).get("plot_transforms", {}) if dashboard is not None else {}
    if not transforms:
        return False, None, None
    trajectory, _markers = _operator_planned_trajectory(plot_context, plan)
    trajectory_times = _operator_planned_trajectory_times(plot_context, plan)
    trajectory_rows = (
        np.asarray(trajectory, dtype=float).reshape(-1, 6)
        if np.asarray(trajectory).size
        else np.empty((0, 6), dtype=float)
    )
    if trajectory_rows.size:
        finite = np.all(np.isfinite(trajectory_rows), axis=1)
        trajectory_rows = trajectory_rows[finite]
        trajectory_times = trajectory_times[finite] if trajectory_times.size == finite.size else np.empty(0, dtype=float)
    if trajectory_rows.size == 0 and selected_probe is None:
        return False, None, None
    for x_axis, y_axis in ((1, 0), (2, 0)):
        transform = transforms.get((x_axis, y_axis))
        if not isinstance(transform, dict):
            continue
        try:
            plot = transform["plot"]
            plot_bounds = (int(plot[0]), int(plot[1]), int(plot[2]), int(plot[3]))
        except (KeyError, TypeError, ValueError):
            continue
        if not _pos_in_bounds(pos, plot_bounds):
            continue
        if selected_probe is not None:
            selected_px = _operator_plot_transform_to_px(
                transform,
                np.asarray(selected_probe.state_ric_km_s, dtype=float)[:3],
                x_axis=x_axis,
                y_axis=y_axis,
            )
            if selected_px is not None and _operator_pixel_distance(pos, selected_px) <= 10.0:
                return True, None, None
        if trajectory_rows.size == 0:
            return False, None, None
        projected_rows = [
            (px, row, time_s)
            for row, time_s in zip(trajectory_rows, trajectory_times, strict=False)
            if (px := _operator_plot_transform_to_px(transform, row[:3], x_axis=x_axis, y_axis=y_axis)) is not None
        ]
        if not projected_rows:
            return False, None, None
        points = np.array([px for px, _row, _time_s in projected_rows], dtype=float)
        distances = np.linalg.norm(points - np.array(pos, dtype=float).reshape(1, 2), axis=1)
        closest_idx = int(np.argmin(distances))
        if float(distances[closest_idx]) <= 10.0:
            _px, row, time_s = projected_rows[closest_idx]
            return True, np.asarray(row, dtype=float).reshape(6).copy(), float(time_s)
        return False, None, None
    return False, None, None


def _operator_pixel_distance(a: tuple[int, int], b: tuple[int, int]) -> float:
    return float(np.linalg.norm(np.array(a, dtype=float) - np.array(b, dtype=float)))


def _draw_operator_grid(
    pygame: Any,
    screen: Any,
    plot: Any,
    *,
    center: tuple[int, int],
    scale_x: float,
    scale_y: float,
    half_span_km: float,
) -> None:
    pygame.draw.rect(screen, (8, 11, 16), plot)
    spacing_km = _operator_grid_spacing_km(half_span_km)
    limit = int(np.ceil(float(half_span_km) / spacing_km))
    for idx in range(-limit, limit + 1):
        x = center[0] + int(round(idx * spacing_km * scale_x))
        y = center[1] - int(round(idx * spacing_km * scale_y))
        if plot.left <= x <= plot.right:
            pygame.draw.line(screen, (24, 32, 44), (x, plot.top), (x, plot.bottom), width=1)
        if plot.top <= y <= plot.bottom:
            pygame.draw.line(screen, (24, 32, 44), (plot.left, y), (plot.right, y), width=1)


def _operator_grid_spacing_km(half_span_km: float) -> float:
    raw = max(float(half_span_km) / 3.0, 0.1)
    magnitude = 10.0 ** np.floor(np.log10(raw))
    for multiplier in (1.0, 2.0, 5.0, 10.0):
        spacing = multiplier * magnitude
        if spacing >= raw:
            return float(spacing)
    return float(10.0 * magnitude)


def _draw_operator_dashed_polyline(
    pygame: Any,
    screen: Any,
    points: list[tuple[int, int]],
    *,
    color: tuple[int, int, int],
    width: int = 1,
) -> None:
    for idx in range(len(points) - 1):
        if idx % 2 == 0:
            pygame.draw.line(screen, color, points[idx], points[idx + 1], width=width)


def _draw_operator_velocity_vector(
    pygame: Any,
    screen: Any,
    origin: tuple[int, int],
    velocity_ric_km_s: np.ndarray,
    *,
    x_axis: int,
    y_axis: int,
    x_display_sign: float = 1.0,
    y_display_sign: float = 1.0,
    color: tuple[int, int, int] = (86, 202, 245),
    length_px: float = 28.0,
) -> None:
    end = _operator_velocity_vector_endpoint(
        origin,
        velocity_ric_km_s,
        x_axis=x_axis,
        y_axis=y_axis,
        x_display_sign=x_display_sign,
        y_display_sign=y_display_sign,
        length_px=length_px,
    )
    if end is None:
        return
    pygame.draw.line(screen, color, origin, end, width=2)
    _draw_operator_arrowhead(pygame, screen, origin=origin, end=end, color=color)


def _draw_operator_arrowhead(
    pygame: Any,
    screen: Any,
    *,
    origin: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    direction = np.array([float(end[0] - origin[0]), float(end[1] - origin[1])], dtype=float)
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= 1.0e-9:
        return
    unit = direction / norm
    normal = np.array([-unit[1], unit[0]], dtype=float)
    tip = np.array([float(end[0]), float(end[1])], dtype=float)
    wing_center = tip - unit * 8.0
    for sign in (-1.0, 1.0):
        wing = wing_center + normal * sign * 5.0
        pygame.draw.line(
            screen,
            color,
            (int(round(tip[0])), int(round(tip[1]))),
            (int(round(wing[0])), int(round(wing[1]))),
            width=2,
        )


def _draw_operator_probe_dot(
    pygame: Any,
    screen: Any,
    center: tuple[int, int],
    *,
    label: str | None = None,
    font: Any | None = None,
) -> None:
    pygame.draw.circle(screen, (86, 202, 245), center, 6)
    pygame.draw.circle(screen, (8, 11, 16), center, 6, width=1)
    if label and font is not None:
        _text(screen, font, label, (center[0] + 9, center[1] - 18), (86, 202, 245))


def _operator_probe_time_label(time_s: float | None) -> str | None:
    if time_s is None or not np.isfinite(float(time_s)):
        return None
    value = max(float(time_s), 0.0)
    return f"T={value:.0f}s"


def _operator_velocity_vector_endpoint(
    origin: tuple[int, int],
    velocity_ric_km_s: np.ndarray,
    *,
    x_axis: int,
    y_axis: int,
    x_display_sign: float = 1.0,
    y_display_sign: float = 1.0,
    length_px: float = 28.0,
) -> tuple[int, int] | None:
    vel = np.array(velocity_ric_km_s, dtype=float).reshape(3)
    projected = vel[[int(x_axis), int(y_axis)]]
    projected = projected * np.array([float(x_display_sign), float(y_display_sign)], dtype=float)
    norm = float(np.linalg.norm(projected))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        return None
    direction = projected / norm
    return (
        int(round(origin[0] + direction[0] * float(length_px))),
        int(round(origin[1] - direction[1] * float(length_px))),
    )


def _draw_operator_translucent_polygon(
    pygame: Any,
    screen: Any,
    plot: Any,
    points: list[tuple[int, int]],
    *,
    color: tuple[int, int, int, int],
) -> None:
    if len(points) < 3:
        return
    overlay = pygame.Surface((plot.width, plot.height), pygame.SRCALPHA)
    shifted = [(int(x - plot.x), int(y - plot.y)) for x, y in points]
    pygame.draw.polygon(overlay, color, shifted)
    screen.blit(overlay, plot.topleft)


def _draw_operator_forbidden_regions(
    pygame: Any,
    screen: Any,
    plot: Any,
    training_cfg: RPOTrainingConfig,
    *,
    x_axis: int,
    y_axis: int,
    to_px: Any,
    offset: np.ndarray,
) -> None:
    for region in training_cfg.forbidden_regions:
        if not _region_visible_on_plane(region, x_axis=int(x_axis), y_axis=int(y_axis)):
            continue
        if region.kind == "annular_sector":
            polygon = region.sector_polygon_ric()
            if not polygon.size:
                continue
            polygon = polygon.copy()
            polygon[:, :3] += np.array(offset, dtype=float).reshape(1, 3)
            points = [to_px(row) for row in polygon]
            if len(points) >= 3:
                _draw_operator_translucent_polygon(pygame, screen, plot, points, color=(168, 44, 54, 58))
                pygame.draw.lines(screen, (230, 80, 92), True, points, width=1)
            continue
        if region.kind == "cylinder":
            polygon = _cylinder_projection_polygon_ric(region, x_axis=int(x_axis), y_axis=int(y_axis))
            if not polygon.size:
                continue
            polygon = polygon.copy()
            polygon[:, :3] += np.array(offset, dtype=float).reshape(1, 3)
            points = [to_px(row) for row in polygon]
            if len(points) >= 3:
                _draw_operator_translucent_polygon(pygame, screen, plot, points, color=(168, 44, 54, 58))
                pygame.draw.lines(screen, (230, 80, 92), True, points, width=1)
            continue
        if region.kind == "sphere":
            polygon = _sphere_projection_polygon_ric(region, x_axis=int(x_axis), y_axis=int(y_axis))
            if not polygon.size:
                continue
            polygon = polygon.copy()
            polygon[:, :3] += np.array(offset, dtype=float).reshape(1, 3)
            points = [to_px(row) for row in polygon]
            if len(points) >= 3:
                _draw_operator_translucent_polygon(pygame, screen, plot, points, color=(168, 44, 54, 58))
                pygame.draw.lines(screen, (230, 80, 92), True, points, width=1)
            continue
        bounds = _finite_projected_region_bounds(region, x_axis=int(x_axis), y_axis=int(y_axis))
        if bounds is None:
            continue
        lo, hi = bounds
        p_min = np.zeros(3, dtype=float)
        p_max = np.zeros(3, dtype=float)
        p_min[int(x_axis)] = lo[0] + float(offset[int(x_axis)])
        p_min[int(y_axis)] = lo[1] + float(offset[int(y_axis)])
        p_max[int(x_axis)] = hi[0] + float(offset[int(x_axis)])
        p_max[int(y_axis)] = hi[1] + float(offset[int(y_axis)])
        a = to_px(p_min)
        b = to_px(p_max)
        rect = pygame.Rect(min(a[0], b[0]), min(a[1], b[1]), abs(a[0] - b[0]), abs(a[1] - b[1]))
        clipped = rect.clip(plot)
        if clipped.width <= 0 or clipped.height <= 0:
            continue
        fill = pygame.Surface((clipped.width, clipped.height), pygame.SRCALPHA)
        fill.fill((168, 44, 54, 58))
        screen.blit(fill, (clipped.x, clipped.y))
        pygame.draw.rect(screen, (230, 80, 92), clipped, width=1)

__all__ = [name for name in globals() if not name.startswith("__")]
