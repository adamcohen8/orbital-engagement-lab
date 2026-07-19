# ruff: noqa: F401,F403,F405,I001
from .launcher_common import *
from .launcher_models import *
from .scenario_catalog import *
from .launcher_persistence import *
from .operator_planning import *
from .launcher_widgets import *

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


def _max_scroll_offset(count: int, screen_height: int) -> int:
    return max(int(count) - _visible_option_count(screen_height), 0)


def _advance_launcher_selection(selected: int, step: int, *, count: int) -> int:
    if int(count) <= 0:
        return 0
    return int((int(selected) + int(step)) % int(count))


def _scroll_for_selection(selected: int, scroll_offset: int, *, count: int, screen_height: int) -> int:
    max_scroll = _max_scroll_offset(count, screen_height)
    visible = _visible_option_count(screen_height)
    scroll = int(max(0, min(scroll_offset, max_scroll)))
    if selected < scroll:
        return int(max(0, selected))
    if selected >= scroll + visible:
        return int(min(max_scroll, selected - visible + 1))
    return scroll


def _pos_in_bounds(pos: tuple[int, int], bounds: tuple[int, int, int, int]) -> bool:
    px, py = pos
    x, y, w, h = bounds
    return x <= px <= x + w and y <= py <= y + h


def _frame_convention_dialog_action(
    pos: tuple[int, int],
    *,
    width: int,
    height: int,
) -> str | None:
    for action, rect in _frame_convention_dialog_choice_rects(width, height).items():
        if _pos_in_bounds(pos, rect):
            return action
    checkbox = _frame_convention_dialog_checkbox_rect(width, height)
    if _pos_in_bounds(pos, (checkbox[0], checkbox[1], 190, checkbox[3])):
        return "dont_ask_again"
    if _pos_in_bounds(pos, _frame_convention_dialog_continue_rect(width, height)):
        return "continue"
    return None


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


def _mode_toggle_at_pos(pos: tuple[int, int], *, width: int, height: int) -> bool:
    return _pos_in_bounds(pos, _mode_toggle_rect(width, height))


def _settings_button_at_pos(pos: tuple[int, int], *, width: int, height: int) -> bool:
    return _pos_in_bounds(pos, _settings_button_rect(width, height))


def _enter_operator_fullscreen(pygame: Any, screen: Any) -> OperatorDisplayState:
    previous_size = tuple(int(value) for value in screen.get_size())
    flags = pygame.FULLSCREEN | pygame.SCALED
    _reset_pygame_display(pygame)
    pygame.display.set_mode((1280, 720), flags)
    pygame.display.set_caption("Orbital Engagement Lab - Operator Mode")
    pygame.mouse.set_visible(True)
    pygame.event.set_grab(False)
    return OperatorDisplayState(previous_size=(previous_size[0], previous_size[1]))


def _restore_operator_display(pygame: Any, state: OperatorDisplayState) -> None:
    width = max(int(state.previous_size[0]), 640)
    height = max(int(state.previous_size[1]), 480)
    _reset_pygame_display(pygame)
    pygame.display.set_mode((width, height), pygame.RESIZABLE)
    pygame.display.set_caption("Orbital Engagement Lab - Level Select")
    pygame.mouse.set_visible(True)
    pygame.event.set_grab(False)


def _reset_pygame_display(pygame: Any) -> None:
    pygame.display.quit()
    pygame.display.init()
    pygame.font.init()


def _selection_for_launch(
    pygame: Any,
    screen: Any,
    clock: Any,
    *,
    option: GameScenarioOption,
    difficulty: str,
    music_enabled: bool,
    record_video: bool,
    mode: str,
    frame_convention: FrameConvention,
    font: Any,
    small_font: Any,
    title_font: Any,
) -> GameLaunchSelection | None:
    mode_key = _normalize_game_mode(mode)
    operator_tutorial = mode_key == "operator" and option.scenario_id == "rpo_00_tutorial"
    return GameLaunchSelection(
        path=option.path,
        difficulty=difficulty,
        music_enabled=music_enabled,
        record_video=record_video,
        mode=mode_key,
        frame_convention=frame_convention,
        operator_burn_plan=None,
        skip_initial_briefing=mode_key == "operator" and not operator_tutorial,
    )

__all__ = [name for name in globals() if not name.startswith("__")]
