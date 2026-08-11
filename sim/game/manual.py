from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CISLUNAR_TRANSLATION_MODES = {"cislunar", "cislunar_translation", "cr3bp", "cr3bp_translation"}
MOON_RIC_TRANSLATION_MODES = {"moon_ric", "moon_ric_translation", "lunar_ric", "lunar_ric_translation"}
TRANSLATION_CONTROL_MODES = {"ric", "ric_translation", "translation", *CISLUNAR_TRANSLATION_MODES, *MOON_RIC_TRANSLATION_MODES}
AERODYNAMIC_CONTROL_MODES = {"aerodynamic", "aero", "aero_control", "aerodynamic_control"}
DIRECT_CONTROL_MODES = {*TRANSLATION_CONTROL_MODES, *AERODYNAMIC_CONTROL_MODES}
ATTITUDE_CONTROL_MODES = {"attitude_thrust", "attitude", "thrust"}


@dataclass
class KeyboardCommandState:
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    firing: bool = False
    throttle: float = 1.0
    reset_requested: bool = False
    restart_requested: bool = False
    paused: bool = False
    speed_multiplier_change: int = 0
    camera_rule_toggle_requested: bool = False
    eci_ri_plot_toggle_requested: bool = False
    eci_rc_plot_toggle_requested: bool = False
    music_toggle_requested: bool = False
    clip_record_toggle_requested: bool = False
    clip_record_save_requested: bool = False
    open_debrief_requested: bool = False
    briefing_scroll_px: int = 0
    quit_requested: bool = False
    speed_increase_held: bool = False
    speed_decrease_held: bool = False
    pitch_event_pulse: bool = False
    yaw_event_pulse: bool = False
    roll_event_pulse: bool = False
    firing_event_pulse: bool = False
    use_timing_accumulator: bool = False
    pitch_sim_s: float = 0.0
    yaw_sim_s: float = 0.0
    roll_sim_s: float = 0.0
    firing_sim_s: float = 0.0

    def reset_axes(self) -> None:
        self.pitch = 0.0
        self.yaw = 0.0
        self.roll = 0.0
        self.firing = False
        self.speed_increase_held = False
        self.speed_decrease_held = False
        self.clear_event_pulses()
        self.clear_timed_input()

    def clear_event_pulses(self) -> None:
        if bool(self.pitch_event_pulse):
            self.pitch = 0.0
        if bool(self.yaw_event_pulse):
            self.yaw = 0.0
        if bool(self.roll_event_pulse):
            self.roll = 0.0
        if bool(self.firing_event_pulse):
            self.firing = False
        self.pitch_event_pulse = False
        self.yaw_event_pulse = False
        self.roll_event_pulse = False
        self.firing_event_pulse = False

    def clear_timed_input(self) -> None:
        self.pitch_sim_s = 0.0
        self.yaw_sim_s = 0.0
        self.roll_sim_s = 0.0
        self.firing_sim_s = 0.0

    def accumulate_timed_input(
        self,
        wall_dt_s: float,
        *,
        speed_multiple: float,
        control_mode: str = "attitude_thrust",
        max_pending_sim_s: float | None = None,
    ) -> None:
        if not bool(self.use_timing_accumulator):
            return
        elapsed_sim_s = float(max(wall_dt_s, 0.0)) * float(max(speed_multiple, 0.0))
        if elapsed_sim_s <= 0.0:
            return
        mode = str(control_mode or "").strip().lower()
        if mode in TRANSLATION_CONTROL_MODES:
            if float(self.throttle) <= 0.0:
                return
            self.pitch_sim_s += float(np.clip(self.pitch, -1.0, 1.0)) * elapsed_sim_s
            self.yaw_sim_s += float(np.clip(self.yaw, -1.0, 1.0)) * elapsed_sim_s
            self.roll_sim_s += float(np.clip(self.roll, -1.0, 1.0)) * elapsed_sim_s
            cap = _positive_float_or_none(max_pending_sim_s)
            if cap is not None:
                self.pitch_sim_s = float(np.clip(self.pitch_sim_s, -cap, cap))
                self.yaw_sim_s = float(np.clip(self.yaw_sim_s, -cap, cap))
                self.roll_sim_s = float(np.clip(self.roll_sim_s, -cap, cap))
        elif bool(self.firing):
            self.firing_sim_s += elapsed_sim_s
            cap = _positive_float_or_none(max_pending_sim_s)
            if cap is not None:
                self.firing_sim_s = min(max(float(self.firing_sim_s), 0.0), cap)

    def consume_ric_duty_cycle(self, dt_s: float) -> np.ndarray:
        if not bool(self.use_timing_accumulator):
            return np.array(
                [
                    float(np.clip(self.pitch, -1.0, 1.0)),
                    float(np.clip(self.yaw, -1.0, 1.0)),
                    float(np.clip(self.roll, -1.0, 1.0)),
                ],
                dtype=float,
            )
        dt = float(max(dt_s, 1.0e-9))
        values = [self.pitch_sim_s, self.yaw_sim_s, self.roll_sim_s]
        consumed: list[float] = []
        remaining: list[float] = []
        for value in values:
            amount = np.sign(float(value)) * min(abs(float(value)), dt)
            consumed.append(float(amount) / dt)
            remaining.append(float(value) - float(amount))
        self.pitch_sim_s, self.yaw_sim_s, self.roll_sim_s = remaining
        return np.array(consumed, dtype=float)

    def consume_firing_duty_cycle(self, dt_s: float) -> float:
        if not bool(self.use_timing_accumulator):
            return 1.0 if bool(self.firing) else 0.0
        dt = float(max(dt_s, 1.0e-9))
        amount = min(max(float(self.firing_sim_s), 0.0), dt)
        self.firing_sim_s = max(float(self.firing_sim_s) - amount, 0.0)
        return float(amount) / dt


def _positive_float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) and parsed > 0.0 else None
