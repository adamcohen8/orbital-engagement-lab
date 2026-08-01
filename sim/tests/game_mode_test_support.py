from __future__ import annotations

# Shared compatibility imports and deterministic builders for the owner-aligned
# game test modules.
# ruff: noqa: F401
import json
import os
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import numpy as np
import pytest
import yaml

import sim.game.arcade as game_arcade
import sim.game.audio as game_audio
import sim.game.debrief as game_debrief
import sim.game.launcher as game_launcher
import sim.game.pygame_dashboard as game_pygame_dashboard
import sim.game.recording_controller as game_recording_controller
import sim.game.runner as game_runner
import sim.game.session as game_session
import sim.game.training as game_training
from sim.acceleration.settings import ACCELERATION_ENV, acceleration_settings_from_config
from sim.api import SimulationConfig, SimulationSession, SimulationSnapshot
from sim.core.models import Command, StateBelief, StateTruth
from sim.dynamics.orbit.cr3bp import (
    EARTH_MOON_MEAN_MOTION_RAD_S,
    cr3bp_derivative_physical,
    cr3bp_halo_seed_state_km_s,
    cr3bp_jacobian_physical,
    cr3bp_l1_state_km_s,
    cr3bp_moon_state_km_s,
    propagate_cr3bp_reference_stm,
    propagate_cr3bp_state,
)
from sim.dynamics.orbit.elements import coes_mapping_to_rv_eci
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2
from sim.dynamics.orbit.two_body import propagate_two_body_rk4
from sim.game.audio import (
    ARCADE_ROUND_CLEAR_SOUND_PATH,
    LEVEL_MUSIC_PATHS,
    MISSION_FAILURE_MUSIC_PATH,
    MISSION_SUCCESS_MUSIC_PATH,
)
from sim.game.debrief import (
    game_debrief_path,
    next_game_debrief_attempt_index,
    open_game_debrief_folder,
    tracker_replay_history,
    write_game_debrief,
)
from sim.game.defensive_target import DefensiveTargetIntentProvider
from sim.game.formatting import format_distance_km, format_speed_km_s, format_speed_m_s
from sim.game.frame_convention import (
    FRAME_CONVENTION_PRESET_SPACE_FORCE,
    FrameConvention,
    frame_convention_display_axis_sign,
    frame_convention_from_preset,
)
from sim.game.launcher import (
    GameScenarioOption,
    GameSettings,
    OperatorTrajectoryProbe,
    clear_game_progress,
    discover_game_scenarios,
    discover_game_scenarios_for_mode,
    record_game_progress,
)
from sim.game.manual import KeyboardCommandState, ManualGameCommandProvider
from sim.game.operator import (
    OperatorBurn,
    OperatorBurnCommandProvider,
    OperatorBurnPlan,
    parse_operator_burn_plan,
    validate_operator_burn_plan,
)
from sim.game.pygame_dashboard import (
    MAX_ACTIVE_CR3BP_GHOST_DRAW_POINTS,
    MIN_PLOT_SPAN_KM,
    MOON_RADIUS_KM,
    PLOT_OVERLAY_MARGIN,
    PygameRPODashboard,
)
from sim.game.recording import (
    GameFrameRecorder,
    add_looped_audio_to_video,
    game_clip_recording_path,
    game_recording_path,
)
from sim.game.runner import (
    OPERATOR_TUTORIAL_BURN_DELTA_V_M_S,
    OPERATOR_TUTORIAL_BURN_TIME_S,
    OperatorBurnCinematicRuntime,
    OperatorTutorialRuntime,
    SandboxSetupValues,
)
from sim.game.session import GamePhysicsSession
from sim.game.training import (
    ApproachGateConfig,
    ForbiddenRegionConfig,
    GuidedTutorialBurnConfig,
    GuidedTutorialSpeedStepConfig,
    RequiredPhaseBurnConfig,
    RPOTrainingConfig,
    RPOTrainingScore,
    RPOTrainingTracker,
    nmt_curve_points_km,
    nmt_element_errors,
    nmt_position_error_km,
    nmt_velocity_error_km_s,
    relative_moon_ric_state_from_arrays,
    relative_state_from_arrays,
    training_config_for_game_mode,
)
from sim.game.tuning import SPEED_DT_SCHEDULE
from sim.resource_limits import SimulationMemoryBudgetError
from sim.utils.frames import ric_dcm_ir_from_rv, ric_rect_state_to_eci


def _knowledge_from_state6(state6: np.ndarray) -> StateBelief:
    return StateBelief(state=np.array(state6, dtype=float).reshape(6), covariance=np.eye(6), last_update_t_s=0.0)

def _game_config(tmp_path: Path) -> dict:
    with (Path(__file__).resolve().parents[2] / "sim" / "game" / "configs" / "game_mode_basic.yaml").open(
        "r", encoding="utf-8"
    ) as f:
        cfg = yaml.safe_load(f)
    cfg = deepcopy(cfg)
    cfg["simulator"]["duration_s"] = 1.0
    cfg["outputs"]["output_dir"] = str(tmp_path)
    cfg["outputs"]["stats"]["print_summary"] = False
    cfg["outputs"]["stats"]["save_json"] = False
    cfg["outputs"]["stats"]["save_full_log"] = False
    return cfg

class _FixedWidthFont:
    def size(self, text: str) -> tuple[int, int]:
        return (len(str(text)) * 8, 14)

    def get_height(self) -> int:
        return 14

def _launcher_option_with_long_preview() -> GameScenarioOption:
    return GameScenarioOption(
        path=Path("level.yaml"),
        scenario_id="rpo_test",
        title="Level Test",
        description="",
        learning_goal=" ".join(["Objective text"] * 16),
        player_brief=" ".join(["Brief text"] * 18),
        pass_criteria=tuple(" ".join(["Criterion"] * 10) for _ in range(4)),
        instructor_notes=tuple(" ".join(["Instructor note"] * 10) for _ in range(3)),
        difficulty="easy",
        time_budget_s=1200.0,
        delta_v_budget_m_s=4.0,
        goal_speed_km_s=0.001,
        target_delta_v_budget_m_s=None,
        completed_difficulties=(),
        high_score=12345,
        level_number=1,
    )


__all__ = [name for name in globals() if not name.startswith("__")]
