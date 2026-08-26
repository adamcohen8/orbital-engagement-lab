from __future__ import annotations

from typing import Any

from sim.game.frame_convention import frame_convention_display_axis_sign
from sim.game.manual import DIRECT_CONTROL_MODES, KeyboardCommandState
from sim.game.manual import (
    TRANSLATION_CONTROL_MODES as _TRANSLATION_CONTROL_MODES,
)
from sim.game.tuning import BRIEFING_SCROLL_STEP_PX

# Compatibility export retained for callers that historically imported this
# constant through sim.game.input.
TRANSLATION_CONTROL_MODES = _TRANSLATION_CONTROL_MODES


def poll_pygame_input(
    pygame: Any,
    state: KeyboardCommandState,
    *,
    control_mode: str = "attitude_thrust",
    briefing_open: bool = False,
    terminal_open: bool = False,
    frame_convention: Any = None,
) -> None:
    direct_control_mode = str(control_mode or "").strip().lower() in DIRECT_CONTROL_MODES
    ric_translation_mode = str(control_mode or "").strip().lower() in {
        "ric",
        "ric_translation",
        "translation",
    }
    in_track_input_sign = (
        frame_convention_display_axis_sign(frame_convention, 1) if ric_translation_mode else 1.0
    )
    scrollable_overlay_open = bool(briefing_open or terminal_open)
    state.briefing_scroll_px = 0
    state.open_debrief_requested = False
    state.clip_record_toggle_requested = False
    state.clip_record_save_requested = False
    state.eci_ri_plot_toggle_requested = False
    state.eci_rc_plot_toggle_requested = False
    state.speed_multiplier_change = 0
    state.pitch_event_pulse = False
    state.yaw_event_pulse = False
    state.roll_event_pulse = False
    state.firing_event_pulse = False
    focus_lost = False
    pulse_pitch = 0.0
    pulse_yaw = 0.0
    pulse_roll = 0.0
    pulse_firing = False
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            pulse_pitch += _event_axis_value(pygame, event, positive_name="K_w", negative_name="K_s")
            pulse_yaw += _event_axis_value(pygame, event, positive_name="K_d", negative_name="K_a")
            pulse_roll += _event_axis_value(pygame, event, positive_name="K_RIGHT", negative_name="K_LEFT")
            if event.key == pygame.K_SPACE:
                pulse_firing = True
        if pygame_focus_lost(pygame, event):
            focus_lost = True
        elif event.type == pygame.QUIT:
            state.quit_requested = True
        elif scrollable_overlay_open and event.type == getattr(pygame, "MOUSEWHEEL", object()):
            state.briefing_scroll_px -= int(getattr(event, "y", 0)) * BRIEFING_SCROLL_STEP_PX
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            state.quit_requested = True
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            state.restart_requested = True
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE and direct_control_mode:
            state.paused = not bool(state.paused)
        elif (
            scrollable_overlay_open
            and event.type == pygame.KEYDOWN
            and event.key == getattr(pygame, "K_PAGEUP", object())
        ):
            state.briefing_scroll_px -= BRIEFING_SCROLL_STEP_PX * 4
        elif (
            scrollable_overlay_open
            and event.type == pygame.KEYDOWN
            and event.key == getattr(pygame, "K_PAGEDOWN", object())
        ):
            state.briefing_scroll_px += BRIEFING_SCROLL_STEP_PX * 4
        elif (
            scrollable_overlay_open
            and event.type == pygame.KEYDOWN
            and event.key == getattr(pygame, "K_HOME", object())
        ):
            state.briefing_scroll_px = -1000000
        elif (
            scrollable_overlay_open and event.type == pygame.KEYDOWN and event.key == getattr(pygame, "K_END", object())
        ):
            state.briefing_scroll_px = 1000000
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_m:
            state.music_toggle_requested = True
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_d:
            state.open_debrief_requested = True
        elif event.type == pygame.KEYDOWN and event.key in {
            getattr(pygame, "K_g", object()),
            getattr(pygame, "K_F9", object()),
        }:
            state.clip_record_toggle_requested = True
        elif (
            not scrollable_overlay_open
            and event.type == pygame.KEYDOWN
            and event.key in {getattr(pygame, "K_RETURN", object()), getattr(pygame, "K_KP_ENTER", object())}
        ):
            state.clip_record_save_requested = True
        elif event.type == pygame.KEYDOWN and event.key == getattr(pygame, "K_c", object()):
            state.camera_rule_toggle_requested = True
        elif event.type == pygame.KEYDOWN and event.key == getattr(pygame, "K_o", object()):
            state.eci_ri_plot_toggle_requested = True
        elif event.type == pygame.KEYDOWN and event.key == getattr(pygame, "K_p", object()):
            state.eci_rc_plot_toggle_requested = True
        elif event.type == pygame.KEYDOWN and _speed_increase_key(pygame, event):
            if (
                state.speed_multiplier_change == 0
                and not bool(state.speed_increase_held)
                and not bool(getattr(event, "repeat", False))
            ):
                state.speed_multiplier_change = 1
            state.speed_increase_held = True
        elif event.type == pygame.KEYDOWN and _speed_decrease_key(pygame, event):
            if (
                state.speed_multiplier_change == 0
                and not bool(state.speed_decrease_held)
                and not bool(getattr(event, "repeat", False))
            ):
                state.speed_multiplier_change = -1
            state.speed_decrease_held = True
        elif event.type == getattr(pygame, "KEYUP", object()) and _speed_increase_key(pygame, event):
            state.speed_increase_held = False
        elif event.type == getattr(pygame, "KEYUP", object()) and _speed_decrease_key(pygame, event):
            state.speed_decrease_held = False

    if focus_lost:
        state.reset_axes()
        return

    keys = pygame.key.get_pressed()
    if not _speed_increase_pressed(pygame, keys):
        state.speed_increase_held = False
    if not _speed_decrease_pressed(pygame, keys):
        state.speed_decrease_held = False
    held_pitch = opposing_key_axis(keys, positive_key=pygame.K_w, negative_key=pygame.K_s)
    held_yaw = opposing_key_axis(keys, positive_key=pygame.K_d, negative_key=pygame.K_a)
    held_roll = opposing_key_axis(keys, positive_key=pygame.K_RIGHT, negative_key=pygame.K_LEFT)
    held_firing = bool(keys[pygame.K_SPACE])
    state.pitch = _combine_axis_pulse(held_pitch, pulse_pitch)
    state.yaw = in_track_input_sign * _combine_axis_pulse(held_yaw, pulse_yaw)
    state.roll = _combine_axis_pulse(held_roll, pulse_roll)
    state.firing = False if direct_control_mode else bool(held_firing or pulse_firing)
    state.pitch_event_pulse = bool(abs(float(held_pitch)) <= 1.0e-12 and abs(float(pulse_pitch)) > 1.0e-12)
    state.yaw_event_pulse = bool(abs(float(held_yaw)) <= 1.0e-12 and abs(float(pulse_yaw)) > 1.0e-12)
    state.roll_event_pulse = bool(abs(float(held_roll)) <= 1.0e-12 and abs(float(pulse_roll)) > 1.0e-12)
    state.firing_event_pulse = bool((not held_firing) and pulse_firing)


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


def _event_key_or_text_matches(
    pygame: Any,
    event: Any,
    *,
    key_names: tuple[str, ...],
    text_values: tuple[str, ...],
) -> bool:
    key = getattr(event, "key", None)
    for key_name in key_names:
        sentinel = object()
        if key == getattr(pygame, key_name, sentinel):
            return True
    text = str(getattr(event, "unicode", "") or "")
    return bool(text and text in text_values)


def _event_axis_value(pygame: Any, event: Any, *, positive_name: str, negative_name: str) -> float:
    key = getattr(event, "key", None)
    if key == getattr(pygame, positive_name, object()):
        return 1.0
    if key == getattr(pygame, negative_name, object()):
        return -1.0
    return 0.0


def _combine_axis_pulse(held_axis: float, pulse_axis: float) -> float:
    value = float(held_axis) + float(pulse_axis)
    if value > 1.0:
        return 1.0
    if value < -1.0:
        return -1.0
    return value


def _speed_increase_key(pygame: Any, event: Any) -> bool:
    return _event_key_or_text_matches(
        pygame,
        event,
        key_names=("K_UP", "K_PLUS", "K_EQUALS", "K_KP_PLUS"),
        text_values=("+", "="),
    )


def _speed_decrease_key(pygame: Any, event: Any) -> bool:
    return _event_key_or_text_matches(
        pygame,
        event,
        key_names=("K_DOWN", "K_MINUS", "K_KP_MINUS"),
        text_values=("-", "_"),
    )


def _key_pressed_for_names(pygame: Any, keys: Any, key_names: tuple[str, ...]) -> bool:
    for key_name in key_names:
        key = getattr(pygame, key_name, None)
        if key is None:
            continue
        try:
            if bool(keys[key]):
                return True
        except Exception:
            continue
    return False


def _speed_increase_pressed(pygame: Any, keys: Any) -> bool:
    return _key_pressed_for_names(pygame, keys, ("K_UP", "K_PLUS", "K_EQUALS", "K_KP_PLUS"))


def _speed_decrease_pressed(pygame: Any, keys: Any) -> bool:
    return _key_pressed_for_names(pygame, keys, ("K_DOWN", "K_MINUS", "K_KP_MINUS"))


def opposing_key_axis(keys: Any, *, positive_key: Any, negative_key: Any) -> float:
    return float(bool(keys[positive_key])) - float(bool(keys[negative_key]))
