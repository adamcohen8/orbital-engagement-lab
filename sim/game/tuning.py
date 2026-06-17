from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GameTuning:
    speed_multiplier_options: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0)
    speed_dt_schedule: tuple[tuple[float, float], ...] = ((10.0, 2.0), (25.0, 2.0), (50.0, 5.0), (100.0, 10.0))
    max_realtime_steps_per_frame: int = 12
    dashboard_fps: float = 60.0
    game_recording_fps: float = 30.0
    high_speed_dashboard_fps: float = 30.0
    medium_high_speed_dashboard_fps: float = 45.0
    maneuver_control_speed: float = 10.0
    briefing_scroll_step_px: int = 48


DEFAULT_GAME_TUNING = GameTuning()

SPEED_MULTIPLIER_OPTIONS = DEFAULT_GAME_TUNING.speed_multiplier_options
SPEED_DT_SCHEDULE = DEFAULT_GAME_TUNING.speed_dt_schedule
MAX_REALTIME_STEPS_PER_FRAME = DEFAULT_GAME_TUNING.max_realtime_steps_per_frame
DASHBOARD_FPS = DEFAULT_GAME_TUNING.dashboard_fps
GAME_RECORDING_FPS = DEFAULT_GAME_TUNING.game_recording_fps
HIGH_SPEED_DASHBOARD_FPS = DEFAULT_GAME_TUNING.high_speed_dashboard_fps
MEDIUM_HIGH_SPEED_DASHBOARD_FPS = DEFAULT_GAME_TUNING.medium_high_speed_dashboard_fps
MANEUVER_CONTROL_SPEED = DEFAULT_GAME_TUNING.maneuver_control_speed
BRIEFING_SCROLL_STEP_PX = DEFAULT_GAME_TUNING.briefing_scroll_step_px
