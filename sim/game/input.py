from __future__ import annotations

from typing import Any

from sim.game.manual import KeyboardCommandState
from sim.game.tuning import BRIEFING_SCROLL_STEP_PX


def poll_pygame_input(
    pygame: Any,
    state: KeyboardCommandState,
    *,
    control_mode: str = "attitude_thrust",
    briefing_open: bool = False,
) -> None:
    ric_mode = str(control_mode or "").strip().lower() in {"ric", "ric_translation", "translation"}
    state.briefing_scroll_px = 0
    state.open_debrief_requested = False
    focus_lost = False
    for event in pygame.event.get():
        if pygame_focus_lost(pygame, event):
            focus_lost = True
        elif event.type == pygame.QUIT:
            state.quit_requested = True
        elif briefing_open and event.type == getattr(pygame, "MOUSEWHEEL", object()):
            state.briefing_scroll_px -= int(getattr(event, "y", 0)) * BRIEFING_SCROLL_STEP_PX
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            state.quit_requested = True
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            state.restart_requested = True
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE and ric_mode:
            state.paused = not bool(state.paused)
        elif briefing_open and event.type == pygame.KEYDOWN and event.key == getattr(pygame, "K_PAGEUP", object()):
            state.briefing_scroll_px -= BRIEFING_SCROLL_STEP_PX * 4
        elif briefing_open and event.type == pygame.KEYDOWN and event.key == getattr(pygame, "K_PAGEDOWN", object()):
            state.briefing_scroll_px += BRIEFING_SCROLL_STEP_PX * 4
        elif briefing_open and event.type == pygame.KEYDOWN and event.key == getattr(pygame, "K_HOME", object()):
            state.briefing_scroll_px = -1000000
        elif briefing_open and event.type == pygame.KEYDOWN and event.key == getattr(pygame, "K_END", object()):
            state.briefing_scroll_px = 1000000
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_PERIOD:
            state.step_requested = True
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_m:
            state.music_toggle_requested = True
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_d:
            state.open_debrief_requested = True
        elif event.type == pygame.KEYDOWN and event.key == getattr(pygame, "K_c", object()):
            state.camera_rule_toggle_requested = True
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
            state.speed_multiplier_change += 1
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
            state.speed_multiplier_change -= 1

    if focus_lost:
        state.reset_axes()
        return

    keys = pygame.key.get_pressed()
    state.pitch = opposing_key_axis(keys, positive_key=pygame.K_w, negative_key=pygame.K_s)
    state.yaw = opposing_key_axis(keys, positive_key=pygame.K_d, negative_key=pygame.K_a)
    state.roll = opposing_key_axis(keys, positive_key=pygame.K_RIGHT, negative_key=pygame.K_LEFT)
    state.firing = False if ric_mode else bool(keys[pygame.K_SPACE])


def pygame_focus_lost(pygame: Any, event: Any) -> bool:
    event_type = getattr(event, "type", None)
    for focus_event_name in ("WINDOWFOCUSLOST", "WINDOWMINIMIZED", "WINDOWHIDDEN"):
        focus_event_type = getattr(pygame, focus_event_name, None)
        if focus_event_type is not None and event_type == focus_event_type:
            return True
    active_event_type = getattr(pygame, "ACTIVEEVENT", None)
    if active_event_type is None or event_type != active_event_type:
        return False
    return int(getattr(event, "gain", 1)) == 0 and bool(getattr(event, "state", 0))


def opposing_key_axis(keys: Any, *, positive_key: Any, negative_key: Any) -> float:
    return float(bool(keys[positive_key])) - float(bool(keys[negative_key]))
