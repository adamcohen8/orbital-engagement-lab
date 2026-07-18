# ruff: noqa: F401,I001
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from sim.api import SimulationConfig
from sim.config import object_section
from sim.game import input as game_input
from sim.game import recording_controller as game_recording
from sim.game.arcade import (
    _arcade_mission_metrics,
    _arcade_round_briefing_lines,
    _arcade_round_initial_state_rng,
    _arcade_round_is_boss,
    _arcade_round_music_track,
    _arcade_round_rng,
    _arcade_round_simulation_config,
    _arcade_round_time_bonus_s,
    _arcade_round_training_config,
    _arcade_round_weighted_score,
    _arcade_score,
    _game_arcade_enabled,
    _game_arcade_initial_time_s,
    _game_defensive_target_provider,
    _game_random_direction_defensive_target_provider,
    _new_arcade_seed,
    _score_time_used_s,
)
from sim.game.audio import (
    GAME_MUSIC_DIR,
    _stop_game_music,
)
from sim.game.audio_controller import GameAudioController
from sim.game.debrief import (
    game_debrief_path,
    next_game_debrief_attempt_index,
    open_game_debrief_folder,
    tracker_replay_history,
    write_game_debrief,
)
from sim.game.defensive_target import DefensiveTargetIntentProvider
from sim.game.formatting import format_distance_km, format_speed_km_s, format_speed_m_s
from sim.game.frame_convention import FrameConvention, normalize_frame_convention
from sim.game.launcher import plan_operator_burns_for_config
from sim.game.manual import (
    CISLUNAR_TRANSLATION_MODES,
    MOON_RIC_TRANSLATION_MODES,
    TRANSLATION_CONTROL_MODES,
    KeyboardCommandState,
    ManualGameCommandProvider,
)
from sim.game.operator import OperatorBurn, OperatorBurnCommandProvider, OperatorBurnPlan, operator_plan_summary
from sim.game.recording_controller import GameClipRecordingController, GameRecordingController
from sim.game.session import (
    GamePhysicsSession,
    _attempt_config_for_training_clock,
    _install_chaser_delta_v_limiter,
    _set_chaser_delta_v_limiter_dt,
)
from sim.game.state import (
    GamePhase,
    mission_state_for_dashboard,
    phase_from_score,
    phase_is_terminal,
    phase_shows_briefing,
)
from sim.game.training import RPOTrainingConfig, RPOTrainingTracker, training_config_for_game_mode
from sim.game.tuning import (
    BRIEFING_SCROLL_STEP_PX,
    DASHBOARD_FPS,
    GAME_RECORDING_FPS,
    HIGH_SPEED_DASHBOARD_FPS,
    MANEUVER_CONTROL_SPEED,
    MAX_REALTIME_STEPS_PER_FRAME,
    MEDIUM_HIGH_SPEED_DASHBOARD_FPS,
    SPEED_DT_SCHEDULE,
    SPEED_MULTIPLIER_OPTIONS,
    STATIC_DASHBOARD_FPS,
)
from sim.presets.thrusters import resolve_thruster_max_thrust_n_from_specs

FULL_ATTEMPT_RECORDING_PAD_S = 3.0
OPERATOR_SCRIPT_RECORDING_HOLD_S = 3.0
RIC_PRIMER_STAGE_COUNT = 3
GAME_BURN_TRACE_ENV = "OEL_GAME_BURN_TRACE"
OPERATOR_BURN_CINEMATIC_SPEED_MULTIPLE = 10.0
OPERATOR_BURN_CINEMATIC_LOOKAHEAD_S = 5.0
OPERATOR_BURN_VISUAL_DURATION_BASE_S = 1.0
OPERATOR_BURN_VISUAL_DURATION_PER_M_S = 0.2
OPERATOR_BURN_VISUAL_DURATION_MIN_S = 1.0
OPERATOR_BURN_VISUAL_DURATION_MAX_S = 2.0
OPERATOR_TUTORIAL_PLAYBACK_SPEED_MULTIPLE = 200.0
OPERATOR_TUTORIAL_STAGE_DURATION_S = 3000.0
OPERATOR_TUTORIAL_BURN_TIME_S = 50.0
OPERATOR_TUTORIAL_BURN_DELTA_V_M_S = 0.25
OPERATOR_ACTUATOR_ERROR_BY_DIFFICULTY: dict[str, float] = {
    "easy": 0.0,
    "medium": 0.01,
    "normal": 0.01,
    "hard": 0.025,
    "extreme": 0.05,
    "expert": 0.05,
}

def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _as_str_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value if str(item))
    return (str(value),)

__all__ = [name for name in globals() if not name.startswith("__")]
