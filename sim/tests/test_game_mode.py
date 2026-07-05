from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import tomllib
import yaml

import sim.game.launcher as game_launcher
import sim.game.pygame_dashboard as game_pygame_dashboard
import sim.game.recording_controller as game_recording_controller
import sim.game.runner as game_runner
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
from sim.game.arcade import (
    _arcade_round_briefing_lines,
    _arcade_round_coast_prediction_model,
    _arcade_round_initial_state_rng,
    _arcade_round_is_boss,
    _arcade_round_music_track,
    _arcade_round_rng,
    _arcade_round_score_multiplier,
    _arcade_round_simulation_config,
    _arcade_round_time_bonus_s,
    _arcade_round_training_config,
    _arcade_round_weighted_score,
    _arcade_score,
    _difficulty_score_multiplier,
    _game_arcade_delta_v_bonus_time_per_m_s,
    _game_arcade_enabled,
    _game_arcade_goal_range_step_km,
    _game_arcade_initial_time_s,
    _game_arcade_min_goal_range_km,
    _game_arcade_round_bonus_time_s,
    _game_random_direction_defensive_target_provider,
    _score_time_used_s,
)
from sim.game.audio import (
    ARCADE_ROUND_CLEAR_SOUND_PATH,
    LEVEL_MUSIC_PATHS,
    MISSION_FAILURE_MUSIC_PATH,
    MISSION_SUCCESS_MUSIC_PATH,
    _level_music_path,
    _play_game_sound_effect,
    _result_music_path,
)
from sim.game.debrief import (
    _active_segments,
    _cumulative_delta_v_m_s,
    _debrief_display_title,
    _event_timeline,
    _plane_axes,
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
    _advance_launcher_selection,
    _clamp_operator_objectives_scroll_px,
    _clamp_preview_scroll_px,
    _clear_progress_at_pos,
    _difficulty_at_pos,
    _difficulty_index,
    _fit_text_px,
    _frame_convention_dialog_action,
    _frame_convention_dialog_checkbox_rect,
    _frame_convention_dialog_choice_rects,
    _frame_convention_dialog_continue_rect,
    _frame_convention_dialog_settings,
    _game_progress_path,
    _game_settings_path,
    _load_game_settings,
    _load_saved_operator_burn_plan,
    _music_at_pos,
    _operator_burn_table_rect,
    _operator_camera_center_ric,
    _operator_delete_row_at_pos,
    _operator_equation_sheet_button_rect,
    _operator_forbidden_region_projection_points,
    _operator_game_plot_panel_rects,
    _operator_initial_relative_ric_state,
    _operator_minimum_plot_span_km,
    _operator_objective_numeric_targets,
    _operator_objectives_button_rect,
    _operator_objectives_content_height,
    _operator_objectives_content_rect,
    _operator_objectives_overlay_rect,
    _operator_plan_from_rows,
    _operator_planned_trajectory,
    _operator_plot_context,
    _operator_plot_transform_to_px,
    _operator_probe_time_label,
    _operator_rows_from_plan,
    _operator_scroll_for_active_row,
    _operator_signed_axis_label,
    _operator_table_visible_rows,
    _operator_trajectory_probe_from_click,
    _operator_velocity_vector_endpoint,
    _option_index_at_pos,
    _preview_bounds,
    _preview_content_height,
    _progress_stars,
    _record_video_at_pos,
    _save_game_settings,
    _save_operator_burn_plan,
    _scroll_for_selection,
    _settings_button_at_pos,
    _show_progress_text,
    _start_artwork_rect,
    _start_screen_event_action,
    _wrap_text_px,
    _wrapped_budget_lines,
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
    _coast_prediction_model_key,
    _cr3bp_projection_mode_key,
    _cr3bp_reference_cache_valid,
    _cr3bp_state_to_moon_ric_rect,
    _cr3bp_states_to_moon_ric_rect_rows,
    _cw_coast_state,
    _cw_coast_states,
    _cw_forced_state,
    _dashboard_history_array,
    _elliptic_linear_coast_states,
    _front_loaded_prediction_times,
    _history_array_tail,
    _linearized_cr3bp_moon_ric_coast_prediction,
    _moon_ric_rect_state_to_cr3bp,
    _nonlinear_cr3bp_moon_ric_coast_prediction,
    _project_eci_positions_to_plane,
    _project_moon_rotating_yz_to_plane,
    _ric_primer_stage,
    _sample_rows,
    _satellite_marker_reticle_radii_px,
    _satellite_marker_size_px,
    _scaled_body_rect_tuple,
    _should_draw_cislunar_moon_background,
    _should_draw_nominal_nmt,
    _true_anomaly_deg_from_state,
    _two_body_coast_state,
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
    _add_level_music_to_recording,
    _adjust_speed_multiple,
    _apply_sandbox_setup_to_config,
    _clear_dashboard_tutorial_path,
    _clear_live_prediction_burn,
    _clear_two_rail_released_maneuver_input,
    _clip_recording_status,
    _coast_prediction_orbit_fraction,
    _coerce_speed_multiple,
    _command_status,
    _dashboard_fps_for_speed,
    _dashboard_object_ids,
    _effective_speed_multiple_for_control,
    _effective_speed_multiple_for_mode,
    _finish_game_recording,
    _game_active_tick_dt_s,
    _game_camera_mode,
    _game_camera_rule_mode,
    _game_camera_rule_toggle_enabled,
    _game_chaser_sprite_diameter_km,
    _game_chaser_sprite_path,
    _game_coast_chaser_after_delta_v_budget,
    _game_coast_prediction_model,
    _game_coast_speed_dt_schedule,
    _game_coast_tick_dt_s,
    _game_command_status,
    _game_control_mode,
    _game_controlled_object_id,
    _game_cr3bp_active_prediction_horizon_s,
    _game_cr3bp_coast_prediction_dt_s,
    _game_cr3bp_coast_prediction_horizon_mode,
    _game_cr3bp_coast_prediction_horizon_s,
    _game_cr3bp_prediction_coast_update_interval_s,
    _game_cr3bp_projection_mode,
    _game_dashboard_fps_cap,
    _game_dashboard_high_speed_fps,
    _game_dashboard_high_speed_fps_max_multiple,
    _game_debrief_enabled,
    _game_initial_speed_multiple,
    _game_level_title,
    _game_loop_should_exit,
    _game_maneuver_control_speed_multiple,
    _game_max_autonomy_step_s,
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
    _game_retained_history_samples,
    _game_ric_reference_object_id,
    _game_sandbox_enabled,
    _game_show_target_hcw_path,
    _game_speed_dt_schedule,
    _game_speed_multiplier_options,
    _game_target_centered_plot_axes,
    _game_target_centered_plot_planes,
    _game_target_coast_prediction_dt_s,
    _game_target_coast_prediction_horizon_s,
    _game_target_sprite_diameter_km,
    _game_target_sprite_path,
    _game_tick_dt_s,
    _game_two_rail_speed_control_enabled,
    _guided_tutorial_delta_v_m_s,
    _guided_tutorial_expected_key,
    _guided_tutorial_input_matches,
    _guided_tutorial_speed_step_follows_burn,
    _guided_tutorial_speed_step_hint,
    _guided_tutorial_speed_step_reached,
    _guided_tutorial_stage_hint,
    _guided_tutorial_target_path,
    _guided_tutorial_update_dashboard_path,
    _guided_tutorial_wrong_input_active,
    _live_prediction_accel_ric,
    _live_prediction_burn,
    _manual_maneuver_active_for_mode,
    _max_accel_from_config,
    _mission_checklist,
    _mission_metrics,
    _operator_actuator_error_fraction,
    _operator_burn_cinematic_hold_for_animation,
    _operator_burn_cinematic_should_arm,
    _operator_burn_cinematic_speed_multiple,
    _operator_burn_visual_duration_s,
    _operator_coast_prediction_orbit_fraction,
    _operator_next_burn_status,
    _operator_tutorial_demo_title,
    _operator_tutorial_enabled,
    _operator_tutorial_stages,
    _opposing_key_axis,
    _pause_teaching_overlay_enabled,
    _pending_maneuver_sim_s_for_mode,
    _phase_from_score_with_operator_animation,
    _poll_pygame_input,
    _realtime_steps_due,
    _reset_guided_tutorial_stage_attempt,
    _ric_primer_enabled,
    _run_sandbox_setup_form,
    _safe_capture_recording_frame,
    _sandbox_coast_prediction_model,
    _sandbox_setup_briefing_lines,
    _sandbox_setup_from_config,
    _sandbox_setup_from_text_values,
    _score_debrief_lines,
    _speed_after_maneuver_input,
    _split_game_step_dt,
    _start_game_attempt,
    _start_game_clip_recorder,
    _start_game_recorder,
    _step_game_attempt,
    _sync_dashboard_training_config,
    _sync_guided_tutorial_path_for_mode,
    _sync_live_prediction_burn_for_mode,
    _timed_maneuver_pending_sim_s,
    _training_briefing_lines,
    _trigger_operator_projection_transition,
    _update_operator_burn_cinematic,
    _wall_step_s,
)
from sim.game.session import GamePhysicsSession, _attempt_config_for_training_clock, _DeltaVLimitedOrbitController
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


def test_game_launcher_discovers_ordered_training_levels() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "game" / "configs"
    options = discover_game_scenarios(config_dir)

    assert [option.scenario_id for option in options] == [
        "rpo_00_tutorial",
        "rpo_01_coast_relative_motion",
        "rpo_02_vbar_approach",
        "rpo_03_rbar_approach",
        "rpo_04_rendezvous",
        "rpo_05_passive_cross_track_approach",
        "rpo_06_sun_angle_inspection",
        "rpo_07_elliptic_burn_then_approach",
        "rpo_08_elliptic_nmc",
        "rpo_09_elliptic_rendezvous",
        "rpo_11_evasive_target_survival",
        "rpo_10_defensive_target_demo",
        "rpo_bonus_cislunar_rendezvous",
        "rpo_sandbox",
    ]
    assert options[0].title == "Level 0 - Pilot Tutorial"
    assert options[1].title == "Level 1 - Relative Orbit"
    assert options[2].title == "Level 2 - V-Bar Approach"
    assert options[3].title == "Level 3 - R-Bar Approach"
    assert options[4].title == "Level 4 - Rendezvous"
    assert options[5].title == "Level 5 - Safe Inspection"
    assert options[0].player_brief
    assert options[0].pass_criteria
    assert options[0].instructor_notes
    assert options[0].time_budget_s == pytest.approx(18000.0)
    assert options[0].delta_v_budget_m_s == pytest.approx(12.0)
    assert options[0].path.name == "game_training_rpo_00_tutorial.yaml"
    assert options[5].path.name == "game_training_rpo_05_passive_cross_track_approach.yaml"
    assert options[6].path.name == "game_training_rpo_06_sun_angle_inspection.yaml"
    assert options[6].title == "Level 6 - Sun-Angle Inspection"
    assert options[7].path.name == "game_training_rpo_07_elliptic_burn_then_approach.yaml"
    assert options[7].title == "Level 7 - Elliptical Approach"
    assert options[8].path.name == "game_training_rpo_08_elliptic_nmc.yaml"
    assert options[8].title == "Level 8 - Elliptical NMC"
    operator_options = discover_game_scenarios_for_mode(config_dir, mode="operator")
    assert operator_options[0].title == "Level 0 - Operator Tutorial"
    operator_ids = {option.scenario_id for option in operator_options}
    assert "rpo_10_defensive_target_demo" not in operator_ids
    assert "rpo_arcade_pursuit" not in operator_ids
    assert "rpo_11_evasive_target_survival" in operator_ids
    assert options[9].path.name == "game_training_rpo_09_elliptic_rendezvous.yaml"
    assert options[9].title == "Level 9 - Elliptical Rendezvous"
    assert options[10].path.name == "game_training_rpo_11_evasive_target_survival.yaml"
    assert options[10].title == "Level 10 - Evasion"
    assert options[11].path.name == "game_training_rpo_10_defensive_target_demo.yaml"
    assert options[11].title == "Level 11 - Pursuit"
    assert options[10].delta_v_budget_m_s == pytest.approx(25.0)
    assert options[10].target_delta_v_budget_m_s == pytest.approx(1.0)
    assert options[11].target_delta_v_budget_m_s == pytest.approx(0.1)
    assert options[12].title == "Bonus Level - Cislunar Rendezvous"
    assert options[12].path.name == "game_training_rpo_bonus_cislunar_rendezvous.yaml"
    assert options[13].scenario_id == "rpo_sandbox"
    assert options[12].time_budget_s == pytest.approx(259200.0)
    assert options[12].delta_v_budget_m_s == pytest.approx(75.0)
    assert options[13].title == "Sandbox"
    assert options[13].path.name == "game_training_rpo_sandbox.yaml"
    assert options[13].time_budget_s == pytest.approx(20000.0)
    assert options[13].delta_v_budget_m_s is None


@pytest.mark.parametrize(
    ("config_name", "expected_title"),
    [
        ("game_training_rpo_02_vbar_approach.yaml", "RPO Trainer Level 2 - V-Bar Approach Debrief"),
        ("game_training_rpo_03_rbar_approach.yaml", "RPO Trainer Level 3 - R-Bar Approach Debrief"),
        ("game_training_rpo_04_rendezvous.yaml", "RPO Trainer Level 4 - Rendezvous Debrief"),
    ],
)
def test_game_debrief_titles_use_level_names_for_levels_2_to_4(config_name: str, expected_title: str) -> None:
    sim_cfg = SimulationConfig.from_yaml(Path(__file__).resolve().parents[1] / "game" / "configs" / config_name)
    training_cfg = RPOTrainingConfig.from_metadata(dict(sim_cfg.scenario.metadata or {}))

    assert _debrief_display_title(config=training_cfg, score=object()) == expected_title


def test_bonus_cislunar_rendezvous_uses_cr3bp_frame() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "sim"
        / "game"
        / "configs"
        / "game_training_rpo_bonus_cislunar_rendezvous.yaml"
    )
    config = SimulationConfig.from_yaml(path)

    assert _game_control_mode(config) == "moon_ric_translation"
    assert _game_relative_frame(config) == "moon_ric"
    assert _game_coast_prediction_model(config) == "cr3bp"
    assert _game_cr3bp_projection_mode(config) == "linearized"
    assert _game_camera_mode(config) == "rule_toggle_pair"
    assert _game_camera_rule_mode(config) == "current_pair"
    assert _game_camera_rule_toggle_enabled(config) is True
    assert _game_chaser_sprite_path(config) == Path("cislunar_chaser_sprite.png")
    assert _game_target_sprite_path(config) == Path("cislunar_target_sprite.png")
    assert (Path(__file__).resolve().parents[1] / "game" / "assets" / _game_chaser_sprite_path(config)).is_file()
    assert (Path(__file__).resolve().parents[1] / "game" / "assets" / _game_target_sprite_path(config)).is_file()
    assert _game_chaser_sprite_diameter_km(config) == pytest.approx(0.05)
    assert _game_target_sprite_diameter_km(config) == pytest.approx(0.12)
    assert _game_dashboard_fps_cap(config) == pytest.approx(45.0)
    assert _game_dashboard_high_speed_fps(config) == pytest.approx(45.0)
    assert _game_dashboard_high_speed_fps_max_multiple(config) == pytest.approx(100.0)
    assert _dashboard_fps_for_speed(
        100.0,
        fps_cap=_game_dashboard_fps_cap(config),
        high_speed_fps=_game_dashboard_high_speed_fps(config),
        high_speed_fps_max_multiple=_game_dashboard_high_speed_fps_max_multiple(config),
    ) == pytest.approx(45.0)
    assert _dashboard_fps_for_speed(
        2000.0,
        fps_cap=_game_dashboard_fps_cap(config),
        high_speed_fps=_game_dashboard_high_speed_fps(config),
        high_speed_fps_max_multiple=_game_dashboard_high_speed_fps_max_multiple(config),
    ) == pytest.approx(30.0)
    assert _game_target_centered_plot_planes(config) == ("RI", "RC")
    assert _game_target_centered_plot_axes(config) == {}
    assert _game_plot_prediction_full_trajectory_only(config) is True
    assert _game_initial_speed_multiple(config, None) == pytest.approx(10.0)
    assert _game_initial_speed_multiple(config, 1.0) == pytest.approx(10.0)
    assert _game_maneuver_control_speed_multiple(config) == pytest.approx(100.0)
    assert _game_two_rail_speed_control_enabled(config) is True
    assert _game_speed_dt_schedule(config) == ((10.0, 0.5), (25.0, 1.0), (50.0, 1.0), (100.0, 1.0))
    assert _game_coast_speed_dt_schedule(config) == (
        (10.0, 0.5),
        (25.0, 1.0),
        (50.0, 2.0),
        (100.0, 2.0),
        (200.0, 10.0),
        (500.0, 25.0),
        (1000.0, 50.0),
        (2000.0, 100.0),
    )
    assert _game_tick_dt_s(config, 10.0) == pytest.approx(0.5)
    assert _game_tick_dt_s(config, 25.0) == pytest.approx(1.0)
    assert _game_tick_dt_s(config, 50.0) == pytest.approx(1.0)
    assert _game_tick_dt_s(config, 100.0) == pytest.approx(1.0)
    assert _game_tick_dt_s(config, 1000.0) == pytest.approx(1.0)
    assert _game_coast_tick_dt_s(config, 100.0) == pytest.approx(2.0)
    assert _game_coast_tick_dt_s(config, 200.0) == pytest.approx(10.0)
    assert _game_coast_tick_dt_s(config, 1000.0) == pytest.approx(50.0)
    assert _game_coast_tick_dt_s(config, 2000.0) == pytest.approx(100.0)
    speed_options = _game_speed_multiplier_options(config)
    assert speed_options[:2] == pytest.approx((10.0, 25.0))
    assert 1.0 not in speed_options
    assert 2.0 not in speed_options
    assert 5.0 not in speed_options
    assert speed_options[-3:] == pytest.approx((500.0, 1000.0, 2000.0))
    assert _adjust_speed_multiple(200.0, 1, options=speed_options) == pytest.approx(500.0)
    assert _adjust_speed_multiple(500.0, 1, options=speed_options) == pytest.approx(1000.0)
    coast_state = KeyboardCommandState()
    burn_state = KeyboardCommandState(pitch=1.0)
    assert _effective_speed_multiple_for_control(
        config,
        1000.0,
        coast_state,
        control_mode="moon_ric_translation",
        options=speed_options,
    ) == pytest.approx(1000.0)
    assert _effective_speed_multiple_for_control(
        config,
        1000.0,
        burn_state,
        control_mode="moon_ric_translation",
        options=speed_options,
    ) == pytest.approx(100.0)
    assert _game_active_tick_dt_s(config, 1000.0, maneuver_active=False) == pytest.approx(50.0)
    assert _game_active_tick_dt_s(
        config,
        _effective_speed_multiple_for_control(
            config,
            1000.0,
            burn_state,
            control_mode="moon_ric_translation",
            options=speed_options,
        ),
        maneuver_active=True,
    ) == pytest.approx(1.0)
    assert _adjust_speed_multiple(1000.0, 1, options=speed_options) == pytest.approx(2000.0)
    assert _adjust_speed_multiple(2000.0, 1, options=speed_options) == pytest.approx(2000.0)
    assert _game_cr3bp_coast_prediction_horizon_s(config) == pytest.approx(21600.0)
    assert _game_cr3bp_active_prediction_horizon_s(config) is None
    assert _game_cr3bp_coast_prediction_horizon_mode(config) == "default"
    assert _game_cr3bp_coast_prediction_dt_s(config) == pytest.approx(1.0)
    assert _game_cr3bp_prediction_coast_update_interval_s(config) == pytest.approx(300.0)
    assert _game_show_target_hcw_path(config) is False
    assert _game_target_coast_prediction_horizon_s(config) == pytest.approx(1127210.360660)
    assert _game_target_coast_prediction_dt_s(config) == pytest.approx(600.0)
    assert _max_accel_from_config(config, "chaser") == pytest.approx(1.25e-6)
    assert config.scenario.simulator.dt_s == pytest.approx(10.0)
    assert config.scenario.simulator.dynamics["orbit"]["model"] == "cr3bp"
    assert config.scenario.simulator.dynamics["orbit"]["orbit_substep_s"] == pytest.approx(10.0)
    halo_initial = config.scenario.objects["target"].initial_state["cr3bp_halo"]
    assert halo_initial["family"] == "l2_nrho_southern"
    assert halo_initial["phase_time_s"] == pytest.approx(843600.0)
    assert halo_initial["phase_substep_s"] == pytest.approx(120.0)
    training_cfg = RPOTrainingConfig.from_metadata(dict(config.scenario.metadata or {}))
    assert training_cfg.relative_frame == "moon_ric"
    assert training_cfg.max_goal_speed_km_s == pytest.approx(0.0001)
    assert training_cfg.hard_speed_limit_km_s == pytest.approx(0.0001)
    chaser_initial = config.scenario.objects["chaser"].initial_state["relative_to_target_ric"]
    assert chaser_initial["reference_frame"] == "moon_ric"

    session = SimulationSession.from_config(config)
    snapshot = session.reset()
    assert snapshot is not None
    chaser0 = np.array(snapshot.truth["chaser"], dtype=float).reshape(-1)[:6]
    target0 = np.array(snapshot.truth["target"], dtype=float).reshape(-1)[:6]
    rel0 = relative_moon_ric_state_from_arrays(target0, chaser0)
    target_moon0 = target0 - cr3bp_moon_state_km_s()
    assert target_moon0 == pytest.approx(
        [
            -120.88579680700822,
            -2920.2783735755906,
            2434.1051443969734,
            -0.052740452172463344,
            1.4146334563164559,
            0.6881647620057171,
        ]
    )
    assert np.linalg.norm(target_moon0[:3]) == pytest.approx(3803.6176212945975)
    assert rel0 == pytest.approx([-3.0, 4.0, 0.5, 0.0, 0.0, 2.0e-6])
    assert np.linalg.norm(rel0[:3]) == pytest.approx(5.024937810560445)

    stepped = session.step()
    target_motion = np.linalg.norm(
        np.array(stepped.truth["target"], dtype=float).reshape(-1)[:3]
        - np.array(snapshot.truth["target"], dtype=float).reshape(-1)[:3]
    )
    assert target_motion > 0.0


def test_level3_rbar_approach_uses_iss_target_sprite() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "sim"
        / "game"
        / "configs"
        / "game_training_rpo_03_rbar_approach.yaml"
    )
    config = SimulationConfig.from_yaml(path)
    sprite_path = _game_target_sprite_path(config)

    assert sprite_path == Path("rpo_iss_target_sprite.png")
    assert (Path(__file__).resolve().parents[1] / "game" / "assets" / sprite_path).is_file()
    assert _game_target_sprite_diameter_km(config) == pytest.approx(0.11)
    assert RPOTrainingConfig.from_metadata(dict(config.scenario.metadata or {})).keepout_radius_km == pytest.approx(0.15)


def test_cr3bp_large_l1_halo_seed_is_available_for_cislunar_game() -> None:
    state = cr3bp_halo_seed_state_km_s(family="l1_northern_large")

    assert state - cr3bp_l1_state_km_s() == pytest.approx(
        [-4288.472449806286, 0.0, 30752.0, 0.0, 0.198451917044, 0.0]
    )


def test_cr3bp_l2_nrho_seed_is_available_for_cislunar_game() -> None:
    state = cr3bp_halo_seed_state_km_s(family="l2_nrho_southern")

    assert state - cr3bp_l1_state_km_s() == pytest.approx(
        [70894.61952879478, 0.0, -69817.0344, 0.0, -0.1042749723224868, 0.0]
    )


def test_cr3bp_moon_ric_transform_round_trips_for_nrho_target() -> None:
    target = cr3bp_halo_seed_state_km_s(family="l2_nrho_southern")
    rel = np.array([5.0, -20.0, -5.0, 0.001, -0.002, 0.003], dtype=float)
    chaser = _moon_ric_rect_state_to_cr3bp(rel, target)

    assert _cr3bp_state_to_moon_ric_rect(chaser, target) == pytest.approx(rel)


def test_cr3bp_moon_ric_batched_transform_matches_scalar_rows() -> None:
    target0 = cr3bp_halo_seed_state_km_s(family="l2_nrho_southern")
    references = np.vstack(
        [
            target0,
            propagate_cr3bp_state(target0, 600.0, 0.0),
            propagate_cr3bp_state(target0, 3600.0, 0.0),
        ]
    )
    rel_rows = np.array(
        [
            [5.0, -20.0, -5.0, 0.001, -0.002, 0.003],
            [4.0, -18.0, -4.0, 0.0008, -0.0015, 0.0025],
            [2.0, -12.0, -3.0, 0.0004, -0.0007, 0.0010],
        ],
        dtype=float,
    )
    deputies = np.vstack(
        [_moon_ric_rect_state_to_cr3bp(rel, reference) for rel, reference in zip(rel_rows, references)]
    )
    scalar = np.vstack(
        [_cr3bp_state_to_moon_ric_rect(deputy, reference) for deputy, reference in zip(deputies, references)]
    )

    batched = _cr3bp_states_to_moon_ric_rect_rows(deputies, references)

    np.testing.assert_allclose(batched, scalar, rtol=1.0e-12, atol=1.0e-12)


def test_cr3bp_physical_jacobian_matches_force_finite_difference() -> None:
    state = cr3bp_halo_seed_state_km_s(family="l2_nrho_southern") + np.array(
        [100.0, -50.0, 20.0, 1.0e-4, -2.0e-4, 1.5e-4],
        dtype=float,
    )
    analytic = cr3bp_jacobian_physical(state)
    finite_difference = np.zeros((6, 6), dtype=float)
    perturbations = np.array([1.0e-2, 1.0e-2, 1.0e-2, 1.0e-8, 1.0e-8, 1.0e-8], dtype=float)

    for idx, step in enumerate(perturbations):
        delta = np.zeros(6, dtype=float)
        delta[idx] = step
        finite_difference[:, idx] = (
            cr3bp_derivative_physical(state + delta) - cr3bp_derivative_physical(state - delta)
        ) / (2.0 * step)

    assert analytic == pytest.approx(finite_difference, abs=1.0e-9)


def test_cr3bp_reference_stm_matches_propagated_finite_difference() -> None:
    reference0 = cr3bp_halo_seed_state_km_s(family="l2_nrho_southern") + np.array(
        [100.0, -50.0, 20.0, 1.0e-4, -2.0e-4, 1.5e-4],
        dtype=float,
    )
    delta0 = np.array([1.0e-3, -2.0e-3, 5.0e-4, 1.0e-8, -2.0e-8, 1.5e-8], dtype=float)

    reference, stm = propagate_cr3bp_reference_stm(reference0, np.eye(6, dtype=float), 600.0, 0.0)
    deputy = propagate_cr3bp_state(reference0 + delta0, 600.0, 0.0)

    assert deputy - reference == pytest.approx(stm @ delta0, abs=1.0e-10)


def test_linearized_cr3bp_moon_ric_projection_tracks_nonlinear_for_small_offsets() -> None:
    target = cr3bp_halo_seed_state_km_s(family="l2_nrho_southern")
    rel0 = np.array([0.1, -0.2, 0.05, 1.0e-5, -2.0e-5, 1.5e-5], dtype=float)
    times = np.linspace(0.0, 600.0, 7, dtype=float)

    nonlinear = _nonlinear_cr3bp_moon_ric_coast_prediction(rel0, target_state=target, times=times, current_t_s=0.0)
    linearized = _linearized_cr3bp_moon_ric_coast_prediction(rel0, target_state=target, times=times, current_t_s=0.0)

    assert linearized.shape == nonlinear.shape
    assert linearized[0] == pytest.approx(rel0)
    assert linearized == pytest.approx(nonlinear, abs=2.0e-4)


def test_game_configs_and_optional_music_packaging_contract() -> None:
    def music_tracks(value: object) -> set[str]:
        if isinstance(value, dict):
            tracks = {str(track) for key, track in value.items() if key == "music_track" and track}
            for nested in value.values():
                tracks.update(music_tracks(nested))
            return tracks
        if isinstance(value, list):
            tracks: set[str] = set()
            for nested in value:
                tracks.update(music_tracks(nested))
            return tracks
        return set()

    root = Path(__file__).resolve().parents[2]
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    pyproject = tomllib.loads(text)
    setuptools_cfg = pyproject["tool"]["setuptools"]
    package_data = set(setuptools_cfg["package-data"]["sim"])
    exclude_package_data = set(setuptools_cfg["exclude-package-data"]["sim"])
    expected_music = {path.name for path in LEVEL_MUSIC_PATHS.values()}
    expected_music.update(
        {
            game_launcher.LAUNCHER_MUSIC_PATH.name,
            MISSION_SUCCESS_MUSIC_PATH.name,
            MISSION_FAILURE_MUSIC_PATH.name,
            ARCADE_ROUND_CLEAR_SOUND_PATH.name,
        }
    )
    for config_path in (root / "sim/game/configs").glob("*.yaml"):
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        expected_music.update(music_tracks(cfg))
    package_music = {Path(pattern).name for pattern in package_data if pattern.startswith("game/music/") and pattern.endswith(".wav")}

    assert '"game/configs/*.yaml"' in text
    assert '"game/assets/*.png"' in text
    assert '"game/music/*.md"' in text
    assert "include-package-data = false" in text
    assert "game/music/*.wav" not in package_data
    assert "game/music/*.wav" in exclude_package_data
    assert package_music == set()
    try:
        from tools.export_public import DEFAULT_GAME_MUSIC_FILES
    except ModuleNotFoundError:
        DEFAULT_GAME_MUSIC_FILES = None
    if DEFAULT_GAME_MUSIC_FILES is not None:
        public_export_music = {Path(path).name for path in DEFAULT_GAME_MUSIC_FILES}
        assert public_export_music == expected_music


def test_training_game_configs_default_to_ric_translation(tmp_path: Path) -> None:
    cfg = _game_config(tmp_path)
    cfg["metadata"]["game"]["training"] = {
        "enabled": True,
        "scenario_id": "custom_training",
        "target_object_id": "training_target",
        "chaser_object_id": "training_chaser",
    }
    cfg["metadata"]["game"].pop("control_mode", None)
    path = tmp_path / "training_default.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    assert _game_control_mode(SimulationConfig.from_yaml(path)) == "ric_translation"


def test_non_training_game_configs_default_to_attitude_thrust(tmp_path: Path) -> None:
    cfg = _game_config(tmp_path)
    cfg["metadata"]["game"].pop("training", None)
    cfg["metadata"]["game"].pop("control_mode", None)
    path = tmp_path / "legacy_default.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    assert _game_control_mode(SimulationConfig.from_yaml(path)) == "attitude_thrust"


def test_public_manual_rpo_example_uses_ric_translation_controls() -> None:
    path = Path(__file__).resolve().parents[2] / "examples" / "configs" / "public_manual_rpo_training.yaml"

    assert _game_control_mode(SimulationConfig.from_yaml(path)) == "ric_translation"


def test_dashboard_object_ids_follow_training_defaults() -> None:
    training_cfg = RPOTrainingConfig(
        enabled=True,
        target_object_id="training_target",
        chaser_object_id="training_chaser",
    )

    assert _dashboard_object_ids(training_cfg, {}) == ("training_target", "training_chaser")
    assert _dashboard_object_ids(
        training_cfg,
        {
            "battlespace_dashboard_target_object_id": "visual_target",
            "battlespace_dashboard_chaser_object_id": "visual_chaser",
        },
    ) == ("visual_target", "visual_chaser")


def test_training_briefing_lines_include_objective_and_assists() -> None:
    config_path = (
        Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_01_coast_relative_motion.yaml"
    )
    config = SimulationConfig.from_yaml(config_path)
    training_cfg = RPOTrainingConfig.from_metadata(dict(config.scenario.metadata or {}))

    lines = _training_briefing_lines(config, training_cfg, difficulty="hard")

    assert lines[0] == "rpo_01_coast_relative_motion"
    assert "Assists: Hard" in lines
    assert any(line.startswith("Objective:") for line in lines)
    assert any(line.startswith("Gate:") for line in lines)


def test_sandbox_config_is_open_ended_setup_mode() -> None:
    config_path = Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_sandbox.yaml"
    config = SimulationConfig.from_yaml(config_path)
    training_cfg = RPOTrainingConfig.from_metadata(dict(config.scenario.metadata or {}))
    score = RPOTrainingScore(
        scenario_id=training_cfg.scenario_id,
        learning_goal=training_cfg.learning_goal,
        samples=2,
        elapsed_s=12.0,
        closest_approach_km=3.0,
        final_range_km=3.0,
        final_goal_error_km=3.0,
        final_relative_speed_km_s=0.0,
        time_inside_keepout_s=0.0,
        approximate_delta_v_m_s=1.25,
        target_delta_v_m_s=0.0,
        burn_axes_satisfied=(),
        phase_burns_satisfied=(),
        speed_multiplier_changed=False,
        coast_after_burn_satisfied=False,
        coast_after_burn_s=0.0,
        guided_tutorial_burns_satisfied=(),
        guided_tutorial_burns_total=0,
        guided_tutorial_speed_satisfied=True,
        guided_tutorial_speed_target=None,
        achieved_time_s=None,
        min_goal_error_km=3.0,
        final_nmt_radial_amplitude_km=float("nan"),
        final_nmt_cross_track_amplitude_km=float("nan"),
        final_nmt_radial_amplitude_error_km=float("nan"),
        final_nmt_cross_track_amplitude_error_km=float("nan"),
        final_nmt_drift_velocity_error_km_s=float("nan"),
        goal_met=False,
        level_passed=False,
        level_failed=False,
        pass_fail_reasons=("Sandbox active; no pass/fail objective.",),
        keepout_violation=False,
        hard_speed_limit_violation=False,
        forbidden_region_violation=False,
        forbidden_region_names=(),
        approach_gate_violation=False,
        approach_gate_names=(),
        approach_gates_satisfied=0,
        approach_gates_total=0,
        inspection_gates_satisfied=0,
        inspection_gates_total=0,
        inspection_gate_names=(),
        hints=(),
    )

    assert _game_sandbox_enabled(config) is True
    assert _game_camera_rule_mode(config) == "full_trajectory"
    assert training_cfg.sandbox_mode is True
    assert training_cfg.max_time_s == pytest.approx(20000.0)
    assert training_cfg.max_delta_v_m_s is None
    assert config.scenario.simulator.duration_s == pytest.approx(20000.0)
    assert config.scenario.simulator.dt_s == pytest.approx(1.0)
    assert _game_target_centered_plot_planes(config) == ("RI", "RC")
    assert _sandbox_coast_prediction_model(_sandbox_setup_from_config(config)) == "hcw"
    assert "INFO dV Used 1.250 m/s" in _mission_metrics(training_cfg, score)
    assert _mission_checklist(training_cfg, score) == ("INFO Experiment Freely",)


def test_training_tracker_sandbox_stays_open_until_time_limit_then_succeeds() -> None:
    cfg = RPOTrainingConfig(
        enabled=True,
        scenario_id="sandbox-unit",
        sandbox_mode=True,
        max_time_s=1.0,
    )
    tracker = RPOTrainingTracker(cfg)
    target_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.54605329, 0.0], dtype=float)
    rel_ric = np.array([0.0, -3.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    chaser_state = ric_rect_state_to_eci(rel_ric, target_state[:3], target_state[3:])

    for time_s in (0.0, 2.0):
        tracker.record(
            snapshot=type(
                "Snapshot",
                (),
                {
                    "time_s": time_s,
                    "truth": {"target": target_state, "chaser": chaser_state},
                    "applied_thrust": {"chaser": np.zeros(3, dtype=float)},
                },
            )()
        )

    score = tracker.score()

    assert score.level_passed is True
    assert score.level_failed is False
    assert "Sandbox complete; time limit reached." in score.pass_fail_reasons

    active_cfg = RPOTrainingConfig(
        enabled=True,
        scenario_id="sandbox-active-unit",
        sandbox_mode=True,
        max_time_s=10.0,
    )
    active_tracker = RPOTrainingTracker(active_cfg)
    for time_s in (0.0, 2.0):
        active_tracker.record(
            snapshot=type(
                "Snapshot",
                (),
                {
                    "time_s": time_s,
                    "truth": {"target": target_state, "chaser": chaser_state},
                    "applied_thrust": {"chaser": np.zeros(3, dtype=float)},
                },
            )()
        )
    active_score = active_tracker.score()

    assert active_score.level_passed is False
    assert active_score.level_failed is False
    assert active_score.pass_fail_reasons == ("Sandbox active; no pass/fail objective.",)
    assert active_tracker.current_hint() == "Sandbox: Maneuver freely, coast, and watch the relative orbit respond."


def test_sandbox_setup_form_values_update_runtime_config() -> None:
    config_path = Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_sandbox.yaml"
    config = SimulationConfig.from_yaml(config_path)
    values = [
        "1.2",
        "-4.5",
        "0.25",
        "0.3",
        "-0.4",
        "0.5",
        "7200",
        "0.1",
        "45",
    ]

    setup, error = _sandbox_setup_from_text_values(values)
    assert error == ""
    assert setup == SandboxSetupValues(
        radial_km=1.2,
        in_track_km=-4.5,
        cross_track_km=0.25,
        radial_rate_m_s=0.3,
        in_track_rate_m_s=-0.4,
        cross_track_rate_m_s=0.5,
        target_a_km=7200.0,
        target_ecc=0.1,
        target_true_anomaly_deg=45.0,
    )

    updated = _apply_sandbox_setup_to_config(config, setup)
    chaser = updated.scenario.objects["chaser"]
    target = updated.scenario.objects["target"]
    training_cfg = RPOTrainingConfig.from_metadata(dict(updated.scenario.metadata or {}))

    assert chaser.initial_state["relative_to_target_ric"]["state"] == pytest.approx(
        [1.2, -4.5, 0.25, 0.0003, -0.0004, 0.0005]
    )
    assert target.initial_state["coes"]["a_km"] == pytest.approx(7200.0)
    assert target.initial_state["coes"]["ecc"] == pytest.approx(0.1)
    assert target.initial_state["coes"]["true_anomaly_deg"] == pytest.approx(45.0)
    assert _game_coast_prediction_model(updated) == "tschauner_hempel"
    assert _game_camera_rule_mode(updated) == "full_trajectory"
    assert _game_target_centered_plot_planes(updated) == ("RI", "RC")
    assert training_cfg.sandbox_mode is True
    assert training_cfg.max_time_s == pytest.approx(20000.0)
    assert training_cfg.max_delta_v_m_s is None
    assert updated.scenario.simulator.duration_s == pytest.approx(20000.0)
    assert updated.scenario.simulator.dt_s == pytest.approx(1.0)

    attempt_config = _attempt_config_for_training_clock(updated, training_cfg)
    assert attempt_config.scenario.simulator.duration_s == pytest.approx(20000.0)
    assert attempt_config.scenario.simulator.dt_s == pytest.approx(1.0)

    session = SimulationSession.from_config(attempt_config)
    snapshot = session.step()
    assert {"target", "chaser"}.issubset(snapshot.truth)


def test_sandbox_setup_form_validation_and_lines() -> None:
    setup, error = _sandbox_setup_from_text_values(["0"] * 9)
    assert setup is None
    assert error == "Target Semimajor Axis must be positive."

    setup, error = _sandbox_setup_from_text_values(["0", "0", "0", "0", "0", "0", "7000", "1", "0"])
    assert setup is None
    assert error == "Target Eccentricity must satisfy 0 <= e < 1."

    lines = _sandbox_setup_briefing_lines(["0"] * 9, active_index=2, error="Target Eccentricity must satisfy 0 <= e < 1.")

    assert lines[0] == "Sandbox Setup"
    assert "> Cross-Track C: 0 km" in lines
    assert any(line.startswith("Input Error:") for line in lines)


def test_sandbox_setup_form_supports_briefing_scroll() -> None:
    class FakeEventSource:
        def __init__(self, batches: list[list[object]]) -> None:
            self._batches = list(batches)

        def get(self) -> list[object]:
            if not self._batches:
                return []
            return self._batches.pop(0)

    class FakePygame:
        QUIT = "quit"
        KEYDOWN = "keydown"
        KEYUP = "keyup"
        MOUSEWHEEL = "mousewheel"
        K_ESCAPE = "escape"
        K_RETURN = "return"
        K_KP_ENTER = "kp_enter"
        K_SPACE = "space"
        K_TAB = "tab"
        K_DOWN = "down"
        K_UP = "up"
        K_PAGEUP = "pageup"
        K_PAGEDOWN = "pagedown"
        K_HOME = "home"
        K_END = "end"
        K_BACKSPACE = "backspace"
        K_DELETE = "delete"

        def __init__(self, batches: list[list[object]]) -> None:
            self.event = FakeEventSource(batches)

    class FakeDashboard:
        closed = False

        def __init__(self, batches: list[list[object]]) -> None:
            self.pygame = FakePygame(batches)
            self.scrolls: list[int] = []
            self.draws = 0

        def scroll_briefing(self, delta_px: int) -> None:
            self.scrolls.append(int(delta_px))

        def draw(self, **_: object) -> None:
            self.draws += 1

        def tick(self, _: float) -> None:
            return None

    wheel = type("WheelEvent", (), {"type": FakePygame.MOUSEWHEEL, "y": -2})()
    page_down = type("KeyEvent", (), {"type": FakePygame.KEYDOWN, "key": FakePygame.K_PAGEDOWN, "unicode": ""})()
    enter = type("KeyEvent", (), {"type": FakePygame.KEYDOWN, "key": FakePygame.K_RETURN, "unicode": ""})()
    dashboard = FakeDashboard([[wheel, page_down], [enter]])
    config_path = Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_sandbox.yaml"
    config = SimulationConfig.from_yaml(config_path)

    setup = _run_sandbox_setup_form(
        dashboard,
        config=config,
        speed_multiple=1.0,
        level_title="Sandbox",
    )

    assert setup == _sandbox_setup_from_config(config)
    assert dashboard.scrolls == [96, 192]
    assert dashboard.draws == 1


def test_launcher_hit_test_accounts_for_scroll_offset() -> None:
    assert _option_index_at_pos((60, 120), count=12, scroll_offset=4) is None
    assert _option_index_at_pos((60, 140), count=12, scroll_offset=0) == 0
    assert _option_index_at_pos((60, 140), count=12, scroll_offset=4) == 4
    assert _option_index_at_pos((60, 200), count=12, scroll_offset=4) == 4
    assert _option_index_at_pos((60, 214), count=12, scroll_offset=4) == 5
    assert _option_index_at_pos((60, 204), count=12, scroll_offset=4) is None


def test_launcher_scroll_tracks_keyboard_selection() -> None:
    assert _scroll_for_selection(0, 0, count=12, screen_height=680) == 0
    assert _scroll_for_selection(6, 0, count=12, screen_height=680) == 1
    assert _scroll_for_selection(11, 1, count=12, screen_height=680) == 6
    assert _scroll_for_selection(4, 6, count=12, screen_height=680) == 4


def test_launcher_keyboard_selection_wraps_at_edges() -> None:
    assert _advance_launcher_selection(0, -1, count=12) == 11
    assert _advance_launcher_selection(11, 1, count=12) == 0
    assert _advance_launcher_selection(4, 1, count=12) == 5
    assert _advance_launcher_selection(4, -1, count=12) == 3
    assert _advance_launcher_selection(0, -1, count=4) == 3
    assert _advance_launcher_selection(3, 1, count=4) == 0
    assert _advance_launcher_selection(0, -1, count=0) == 0


def test_launcher_difficulty_helpers_support_picker() -> None:
    assert _difficulty_index("easy") == 0
    assert _difficulty_index("normal") == 1
    assert _difficulty_index("expert") == 3
    assert _difficulty_index("unknown") == 0
    assert _difficulty_at_pos((650, 94)) == "easy"
    assert _difficulty_at_pos((908, 94)) == "extreme"
    assert _difficulty_at_pos((500, 94)) is None


def test_launcher_progress_helpers_persist_user_state_without_mutating_yaml(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OEL_GAME_PROGRESS_PATH", str(tmp_path / "progress.yaml"))
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    path = config_dir / "game_training_rpo_01_demo.yaml"
    cfg = _game_config(tmp_path)
    cfg["metadata"]["game"]["training"] = {"scenario_id": "rpo_01_demo"}
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    original_yaml = path.read_text(encoding="utf-8")

    record_game_progress(path, "hard")
    options = discover_game_scenarios(config_dir)

    assert path.read_text(encoding="utf-8") == original_yaml
    assert _game_progress_path().exists()
    assert options[0].completed_difficulties == ("hard",)
    assert options[0].high_score == 0
    assert _progress_stars(options[0].completed_difficulties) == "★★★☆"

    record_game_progress(path, "medium", score=1200)
    record_game_progress(path, "easy", score=900)
    options = discover_game_scenarios(config_dir)

    assert options[0].completed_difficulties == ("easy", "medium", "hard")
    assert options[0].high_score == 1200

    record_game_progress(path, "extreme", score=2500, completed=False)
    options = discover_game_scenarios(config_dir)

    assert options[0].completed_difficulties == ("easy", "medium", "hard")
    assert options[0].high_score == 2500

    clear_game_progress(config_dir)
    options = discover_game_scenarios(config_dir)

    assert path.read_text(encoding="utf-8") == original_yaml
    assert options[0].completed_difficulties == ()
    assert options[0].high_score == 0
    assert _progress_stars(options[0].completed_difficulties) == "☆☆☆☆"


def test_launcher_progress_is_separate_for_operator_mode(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OEL_GAME_PROGRESS_PATH", str(tmp_path / "progress.yaml"))
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    path = config_dir / "game_training_rpo_02_demo.yaml"
    cfg = _game_config(tmp_path)
    cfg["metadata"]["game"]["training"] = {"scenario_id": "rpo_02_demo"}
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    record_game_progress(path, "hard", score=100, mode="pilot")
    record_game_progress(path, "easy", score=700, mode="operator")
    pilot = discover_game_scenarios_for_mode(config_dir, mode="pilot")[0]
    operator = discover_game_scenarios_for_mode(config_dir, mode="operator")[0]

    assert pilot.completed_difficulties == ("hard",)
    assert pilot.high_score == 100
    assert operator.completed_difficulties == ("easy",)
    assert operator.high_score == 700


def test_launcher_settings_persist_frame_convention(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OEL_GAME_SETTINGS_PATH", str(tmp_path / "settings.yaml"))
    plan = parse_operator_burn_plan("T= 50 s, 2.0 m/s R, 1.0 m/s I, 0.2 m/s C")

    _save_game_settings(
        GameSettings(
            frame_convention=FrameConvention(
                positive_in_track="left",
                positive_cross_track="clockwise",
            ),
            ask_frame_convention_on_launch=False,
            last_game_mode="operator",
            operator_burn_scripts={"rpo_01_relative_orbit": plan},
        )
    )
    loaded = _load_game_settings()

    assert _game_settings_path().exists()
    assert loaded.frame_convention == FrameConvention(
        positive_in_track="left",
        positive_cross_track="clockwise",
    )
    assert loaded.ask_frame_convention_on_launch is False
    assert loaded.last_game_mode == "operator"
    assert loaded.operator_burn_scripts["rpo_01_relative_orbit"].burns == plan.burns


def test_frame_convention_dialog_settings_preserve_saved_operator_scripts() -> None:
    plan = parse_operator_burn_plan("T= 50 s, 2.0 m/s R")
    settings = GameSettings(
        frame_convention=FrameConvention(),
        operator_burn_scripts={"rpo_01_relative_orbit": plan},
    )

    updated = _frame_convention_dialog_settings(
        settings,
        frame_convention=frame_convention_from_preset(FRAME_CONVENTION_PRESET_SPACE_FORCE),
        dont_ask_again=True,
        selected_mode="operator",
    )

    assert updated.ask_frame_convention_on_launch is False
    assert updated.last_game_mode == "operator"
    assert updated.operator_burn_scripts["rpo_01_relative_orbit"].burns == plan.burns


def test_operator_burn_scripts_are_saved_by_scenario_id(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OEL_GAME_SETTINGS_PATH", str(tmp_path / "settings.yaml"))
    plan = parse_operator_burn_plan("T= 10 s, 0.5 m/s R\nT= 40 s, -0.2 m/s I, 0.1 m/s C")

    assert _load_saved_operator_burn_plan("rpo_02_vbar_approach") is None

    _save_operator_burn_plan("rpo_02_vbar_approach", plan)
    loaded = _load_saved_operator_burn_plan("rpo_02_vbar_approach")

    assert loaded is not None
    assert loaded.burns == plan.burns

    _save_operator_burn_plan("rpo_02_vbar_approach", OperatorBurnPlan())

    assert _load_saved_operator_burn_plan("rpo_02_vbar_approach") == OperatorBurnPlan()


def test_launcher_settings_persist_frame_convention_preset(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OEL_GAME_SETTINGS_PATH", str(tmp_path / "settings.yaml"))
    settings_path = _game_settings_path()
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(
        "frame_convention:\n"
        f"  preset: {FRAME_CONVENTION_PRESET_SPACE_FORCE}\n"
        "ask_frame_convention_on_launch: false\n",
        encoding="utf-8",
    )

    loaded = _load_game_settings()

    assert loaded.frame_convention == FrameConvention(
        positive_in_track="left",
        positive_cross_track="clockwise",
    )
    assert loaded.ask_frame_convention_on_launch is False
    assert loaded.last_game_mode is None


def test_space_force_frame_mirrors_only_display_in_track_axis() -> None:
    convention = frame_convention_from_preset(FRAME_CONVENTION_PRESET_SPACE_FORCE)

    assert frame_convention_display_axis_sign(convention, 0) == pytest.approx(1.0)
    assert frame_convention_display_axis_sign(convention, 1) == pytest.approx(-1.0)
    assert frame_convention_display_axis_sign(convention, 2) == pytest.approx(1.0)


def test_dashboard_frame_convention_does_not_flip_cislunar_axes() -> None:
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.frame_convention = frame_convention_from_preset(FRAME_CONVENTION_PRESET_SPACE_FORCE)
    dashboard.relative_frame = "ric"

    assert dashboard._axis_display_sign(1) == pytest.approx(-1.0)

    dashboard.relative_frame = "cislunar_l1"

    assert dashboard._axis_display_sign(1) == pytest.approx(1.0)


def test_dashboard_signed_axis_labels_include_positive_and_negative_directions() -> None:
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.relative_frame = "ric"

    assert dashboard._signed_axis_label_for_plot(0, 1) == "+R"
    assert dashboard._signed_axis_label_for_plot(0, -1) == "-R"
    assert dashboard._signed_axis_label_for_plot(1, 1) == "+I"
    assert dashboard._signed_axis_label_for_plot(2, -1) == "-C"


def test_operator_preview_signed_axis_labels_include_positive_and_negative_directions() -> None:
    assert _operator_signed_axis_label(0, 1) == "+R"
    assert _operator_signed_axis_label(1, -1) == "-I"
    assert _operator_signed_axis_label(2, 1) == "+C"


def test_frame_convention_dialog_hit_testing() -> None:
    choices = _frame_convention_dialog_choice_rects(1040, 680)
    checkbox = _frame_convention_dialog_checkbox_rect(1040, 680)
    continue_rect = _frame_convention_dialog_continue_rect(1040, 680)

    assert (
        _frame_convention_dialog_action(
            (choices["oel_default"][0] + 2, choices["oel_default"][1] + 2),
            width=1040,
            height=680,
        )
        == "oel_default"
    )
    assert (
        _frame_convention_dialog_action(
            (choices["space_force"][0] + 2, choices["space_force"][1] + 2),
            width=1040,
            height=680,
        )
        == "space_force"
    )
    assert _frame_convention_dialog_action((checkbox[0] + 3, checkbox[1] + 3), width=1040, height=680) == "dont_ask_again"
    assert _frame_convention_dialog_action((continue_rect[0] + 3, continue_rect[1] + 3), width=1040, height=680) == "continue"
    assert _frame_convention_dialog_action((10, 10), width=1040, height=680) is None


def test_clear_progress_suppresses_legacy_yaml_progress_without_mutating_yaml(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OEL_GAME_PROGRESS_PATH", str(tmp_path / "progress.yaml"))
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    path = config_dir / "game_training_rpo_01_demo.yaml"
    cfg = _game_config(tmp_path)
    cfg["metadata"]["game"]["training"] = {"scenario_id": "rpo_01_demo"}
    cfg["metadata"]["game"]["progress"] = {"completed_difficulties": ["hard"]}
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    original_yaml = path.read_text(encoding="utf-8")

    assert discover_game_scenarios(config_dir)[0].completed_difficulties == ("hard",)

    clear_game_progress(config_dir)
    options = discover_game_scenarios(config_dir)

    assert path.read_text(encoding="utf-8") == original_yaml
    assert options[0].completed_difficulties == ()


def test_clear_progress_button_hit_test() -> None:
    assert _clear_progress_at_pos((860, 44)) is True
    assert _clear_progress_at_pos((800, 44)) is False


def test_record_video_button_hit_test() -> None:
    assert _record_video_at_pos((700, 44)) is True
    assert _record_video_at_pos((650, 44)) is False


def test_music_button_hit_test() -> None:
    assert _music_at_pos((536, 44)) is True
    assert _music_at_pos((500, 44)) is False


def test_settings_button_hit_test() -> None:
    assert _settings_button_at_pos((980, 642), width=1040, height=680) is True
    assert _settings_button_at_pos((1008, 642), width=1040, height=680) is False


def test_start_screen_event_action_begins_on_any_non_escape_key() -> None:
    class FakePygame:
        QUIT = "quit"
        KEYDOWN = "keydown"
        K_ESCAPE = "escape"

    ordinary_key = type("Event", (), {"type": FakePygame.KEYDOWN, "key": "return"})()
    escape_key = type("Event", (), {"type": FakePygame.KEYDOWN, "key": FakePygame.K_ESCAPE})()
    quit_event = type("Event", (), {"type": FakePygame.QUIT})()
    mouse_event = type("Event", (), {"type": "mouse"})()

    assert _start_screen_event_action(FakePygame, ordinary_key) == "begin"
    assert _start_screen_event_action(FakePygame, escape_key) == "quit"
    assert _start_screen_event_action(FakePygame, quit_event) == "quit"
    assert _start_screen_event_action(FakePygame, mouse_event) == "ignore"


def test_choose_game_launch_can_skip_start_screen(monkeypatch) -> None:
    calls: list[bool] = []

    monkeypatch.setattr(game_launcher, "_load_game_settings", lambda: GameSettings())
    monkeypatch.setattr(game_launcher, "discover_game_scenarios_for_mode", lambda config_dir=None, *, mode="pilot": ("option",))

    def fake_run_launcher(options, *, show_start_screen=True, initial_mode="pilot"):
        calls.append(bool(show_start_screen))
        return None

    monkeypatch.setattr(game_launcher, "_run_launcher", fake_run_launcher)

    game_launcher.choose_game_launch(show_start_screen=False)

    assert calls == [False]


def test_choose_game_launch_opens_in_initial_mode(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(game_launcher, "_load_game_settings", lambda: GameSettings())

    def fake_discover(config_dir=None, *, mode="pilot"):
        calls.append(("discover", mode))
        return ("option",)

    def fake_run_launcher(options, *, show_start_screen=True, initial_mode="pilot"):
        calls.append(("run", initial_mode))
        return None

    monkeypatch.setattr(game_launcher, "discover_game_scenarios_for_mode", fake_discover)
    monkeypatch.setattr(game_launcher, "_run_launcher", fake_run_launcher)

    game_launcher.choose_game_launch(show_start_screen=False, initial_mode="operator")

    assert calls == [("discover", "operator"), ("run", "operator")]


def test_choose_game_launch_prefers_saved_last_mode(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(game_launcher, "_load_game_settings", lambda: GameSettings(last_game_mode="operator"))

    def fake_discover(config_dir=None, *, mode="pilot"):
        calls.append(("discover", mode))
        return ("option",)

    def fake_run_launcher(options, *, show_start_screen=True, initial_mode="pilot"):
        calls.append(("run", initial_mode))
        return None

    monkeypatch.setattr(game_launcher, "discover_game_scenarios_for_mode", fake_discover)
    monkeypatch.setattr(game_launcher, "_run_launcher", fake_run_launcher)

    game_launcher.choose_game_launch(show_start_screen=False, initial_mode="pilot")

    assert calls == [("discover", "operator"), ("run", "operator")]


def test_operator_selection_defers_script_screen_to_gameplay(monkeypatch) -> None:
    config_dir = Path(__file__).resolve().parents[1] / "game" / "configs"
    option = next(
        option for option in discover_game_scenarios_for_mode(config_dir, mode="operator")
        if option.scenario_id == "rpo_01_coast_relative_motion"
    )
    calls: list[str] = []

    class FakePygame:
        pass

    def fail_prebrief(*_args, **_kwargs):
        raise AssertionError("operator launch should not show the prebrief screen")

    def fail_plan_screen(*_args, **_kwargs):
        calls.append("plan")
        raise AssertionError("operator launch should defer the script screen to gameplay")

    monkeypatch.setattr(game_launcher, "_enter_operator_fullscreen", lambda pygame, screen: None)
    monkeypatch.setattr(game_launcher, "_run_operator_prebrief_screen", fail_prebrief)
    monkeypatch.setattr(game_launcher, "_run_operator_plan_screen", fail_plan_screen)

    selection = game_launcher._selection_for_launch(
        FakePygame(),
        object(),
        object(),
        option=option,
        difficulty="easy",
        music_enabled=False,
        record_video=False,
        mode="operator",
        frame_convention=FrameConvention(),
        font=_FixedWidthFont(),
        small_font=_FixedWidthFont(),
        title_font=_FixedWidthFont(),
    )

    assert calls == []
    assert selection is not None
    assert selection.operator_burn_plan is None
    assert selection.skip_initial_briefing is True


def test_start_screen_artwork_rect_fits_screen_without_distortion() -> None:
    rect = _start_artwork_rect((1672, 941), (1040, 680))
    x, y, width, height = rect

    assert x >= 0
    assert y >= 0
    assert width <= 1040
    assert height <= 680
    assert width == 1040
    assert height < 680
    assert width / height == pytest.approx(1672 / 941, rel=0.02)


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


def test_launcher_hides_tutorial_progress_text() -> None:
    option = _launcher_option_with_long_preview()
    tutorial = replace(option, scenario_id="rpo_00_tutorial", title="Level 0 - Pilot Tutorial", level_number=0)

    assert _show_progress_text(tutorial) is False
    assert _show_progress_text(option) is True


def test_launcher_preview_wraps_text_to_pixel_width() -> None:
    font = _FixedWidthFont()
    lines = _wrap_text_px(
        "Use small pulses and long coast arcs to shape the relative orbit.",
        font,
        160,
    )

    assert len(lines) > 1
    assert all(font.size(line)[0] <= 160 for line in lines)


def test_launcher_preview_wraps_long_budget_line() -> None:
    font = _FixedWidthFont()
    option = replace(
        _launcher_option_with_long_preview(),
        time_budget_s=7200.0,
        delta_v_budget_m_s=1.0,
        goal_speed_km_s=0.00005,
        target_delta_v_budget_m_s=2.0,
    )

    lines = _wrapped_budget_lines(option, font, 300)

    assert len(lines) > 1
    assert all(font.size(line)[0] <= 300 for line in lines)
    assert not lines[-1].endswith("...")


def test_launcher_preview_truncates_long_unbroken_words_to_pixel_width() -> None:
    font = _FixedWidthFont()
    text = _fit_text_px("supercalifragilisticexpialidocious", font, 80)

    assert text.endswith("...")
    assert font.size(text)[0] <= 80


def test_launcher_preview_scroll_clamps_to_scrollable_content() -> None:
    font = _FixedWidthFont()
    option = _launcher_option_with_long_preview()
    bounds = _preview_bounds(1040, 320)
    content_height = _preview_content_height(option, font=font, small_font=font, width_px=bounds[2] - 40)

    scroll = _clamp_preview_scroll_px(100_000, option=option, font=font, small_font=font, preview_bounds=bounds)

    assert content_height > bounds[3] - 40
    assert scroll == content_height - (bounds[3] - 40)


def test_launcher_preview_scroll_is_zero_when_content_fits() -> None:
    font = _FixedWidthFont()
    option = GameScenarioOption(
        path=Path("level.yaml"),
        scenario_id="rpo_test",
        title="Level Test",
        description="",
        learning_goal="Short objective.",
        player_brief="Short brief.",
        pass_criteria=("Short pass.",),
        instructor_notes=(),
        difficulty="easy",
        time_budget_s=None,
        delta_v_budget_m_s=None,
        goal_speed_km_s=None,
        target_delta_v_budget_m_s=None,
        completed_difficulties=(),
        high_score=0,
        level_number=1,
    )

    scroll = _clamp_preview_scroll_px(120, option=option, font=font, small_font=font, preview_bounds=(490, 124, 420, 480))

    assert scroll == 0


def test_game_recording_path_uses_scenario_difficulty_and_attempt(tmp_path: Path) -> None:
    path = game_recording_path(
        scenario_name="RPO 09 Defensive Target Demo",
        difficulty="Hard",
        attempt_index=3,
        output_dir=tmp_path,
        timestamp=datetime(2026, 5, 14, 12, 30, 45),
    )

    assert path == tmp_path / "rpo_09_defensive_target_demo_hard_20260514_123045_attempt03.mp4"


def test_game_clip_recording_path_uses_clip_folder_and_index(tmp_path: Path) -> None:
    path = game_clip_recording_path(
        scenario_name="RPO 01 Relative Orbit",
        difficulty="Easy",
        clip_index=2,
        output_dir=tmp_path,
        timestamp=datetime(2026, 5, 27, 9, 8, 7),
    )

    assert path == tmp_path / "clips" / "rpo_01_relative_orbit_easy_20260527_090807_clip02.mp4"


def test_game_debrief_writer_exports_summary_and_replay(tmp_path: Path) -> None:
    cfg = RPOTrainingConfig(
        enabled=True,
        scenario_id="unit-debrief",
        learning_goal="test",
        goal_range_km=0.25,
    )
    tracker = RPOTrainingTracker(cfg)
    target_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.54605329, 0.0], dtype=float)
    rel_ric = np.array([0.0, -0.2, 0.0, 0.0, 0.0, 0.0], dtype=float)
    chaser_state = ric_rect_state_to_eci(rel_ric, target_state[:3], target_state[3:])
    tracker.record(
        snapshot=type(
            "Snapshot",
            (),
            {
                "time_s": 0.0,
                "truth": {"target": target_state, "chaser": chaser_state},
                "applied_thrust": {"chaser": np.zeros(3, dtype=float), "target": np.zeros(3, dtype=float)},
            },
        )()
    )
    score = tracker.score()
    path = game_debrief_path(
        scenario_id=cfg.scenario_id,
        difficulty="easy",
        attempt_index=2,
        output_dir=tmp_path,
        timestamp=datetime(2026, 5, 14, 12, 30, 45),
    )

    out = write_game_debrief(
        path,
        config=cfg,
        score=score,
        difficulty="easy",
        objective_checklist=_mission_checklist(cfg, score),
        arcade_score=123,
        arcade_seed=456,
        arcade_round_index=7,
        recording_path=tmp_path / "attempt.mp4",
        replay_history=tracker_replay_history(tracker),
    )
    summary_path = out.parent / "summary.json"
    payload = yaml.safe_load(summary_path.read_text(encoding="utf-8"))

    assert out == tmp_path / "unit_debrief" / "attempt_002_easy_20260514_123045" / "report.md"
    assert summary_path == out.parent / "summary.json"
    assert payload["scenario_id"] == "unit-debrief"
    assert payload["level_passed"] is True
    assert payload["score"]["arcade_score"] == 123
    assert payload["score"]["arcade_seed"] == 456
    assert payload["score"]["arcade_round_index"] == 7
    assert payload["artifacts"]["recording_path"].endswith("attempt.mp4")
    assert payload["artifacts"]["report_path"].endswith("report.md")
    assert payload["artifacts"]["summary_path"].endswith("summary.json")
    assert "ric_2d" in payload["artifacts"]["plot_paths"]
    assert "mission_timeline" in payload["artifacts"]["plot_paths"]
    assert (out.parent / "plots" / "ric_2d_plots.png").exists()
    assert (out.parent / "plots" / "mission_timeline.png").exists()
    assert "Pass/Failure" in out.read_text(encoding="utf-8")
    assert "# Unit Debrief" in out.read_text(encoding="utf-8")
    assert "## Event Timeline" not in out.read_text(encoding="utf-8")
    assert "![Mission Timeline](plots/mission_timeline.png)" in out.read_text(encoding="utf-8")
    assert "![2D RIC Plots](plots/ric_2d_plots.png)" in out.read_text(encoding="utf-8")
    assert payload["replay"]["time_s"] == [0.0]
    assert payload["replay"]["relative_ric"][0][:3] == pytest.approx([0.0, -0.2, 0.0])


def test_game_debrief_header_uses_rpo_trainer_level_name(tmp_path: Path) -> None:
    sim_cfg = SimulationConfig.from_yaml(
        Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_06_sun_angle_inspection.yaml"
    )
    cfg = RPOTrainingConfig.from_metadata(dict(sim_cfg.scenario.metadata or {}))
    score = type(
        "Score",
        (),
        {
            "scenario_id": cfg.scenario_id,
            "learning_goal": cfg.learning_goal,
            "samples": 1,
            "elapsed_s": 0.0,
            "closest_approach_km": 1.0,
            "final_range_km": 1.0,
            "final_goal_error_km": 1.0,
            "final_relative_speed_km_s": 0.0,
            "time_inside_keepout_s": 0.0,
            "approximate_delta_v_m_s": 0.0,
            "level_passed": True,
            "level_failed": False,
            "pass_fail_reasons": ("All pass criteria satisfied.",),
        },
    )()
    out = write_game_debrief(
        game_debrief_path(
            scenario_id=cfg.scenario_id,
            difficulty="easy",
            attempt_index=1,
            output_dir=tmp_path,
            timestamp=datetime(2026, 5, 22, 12, 0, 0),
        ),
        config=cfg,
        score=score,
        difficulty="easy",
        objective_checklist=(),
        replay_history={
            "time_s": [0.0],
            "relative_ric": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            "chaser_thrust_ric_km_s2": [[0.0, 0.0, 0.0]],
        },
    )
    payload = yaml.safe_load((out.parent / "summary.json").read_text(encoding="utf-8"))

    assert cfg.level_name == "Level 6 - Sun-Angle Inspection"
    assert payload["display_title"] == "RPO Trainer Level 6 - Sun-Angle Inspection Debrief"
    assert out.read_text(encoding="utf-8").startswith("# RPO Trainer Level 6 - Sun-Angle Inspection Debrief\n")


def test_tracker_replay_history_uses_array_backed_replay_stream() -> None:
    cfg = RPOTrainingConfig(
        enabled=True,
        scenario_id="unit-replay",
        learning_goal="test",
        goal_range_km=0.25,
    )
    tracker = RPOTrainingTracker(cfg)
    target_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.54605329, 0.0], dtype=float)
    rel_ric = np.array([0.0, -0.2, 0.1, 0.0, 0.0, 0.0], dtype=float)
    chaser_state = ric_rect_state_to_eci(rel_ric, target_state[:3], target_state[3:])
    tracker.record(
        snapshot=type(
            "Snapshot",
            (),
            {
                "time_s": 12.0,
                "truth": {"target": target_state, "chaser": chaser_state},
                "applied_thrust": {
                    "chaser": np.array([0.0, 1.0e-6, 0.0], dtype=float),
                    "target": np.array([0.0, 0.0, 2.0e-6], dtype=float),
                },
            },
        )()
    )

    expected = tracker_replay_history(tracker)
    tracker.t_s.clear()
    tracker.rel_ric_hist.clear()
    tracker.thrust_hist.clear()
    tracker.thrust_ric_hist.clear()
    tracker.target_thrust_hist.clear()
    replay = tracker_replay_history(tracker)

    assert replay["time_s"] == expected["time_s"] == [12.0]
    np.testing.assert_allclose(replay["relative_ric"], expected["relative_ric"])
    np.testing.assert_allclose(replay["chaser_thrust_ric_km_s2"], expected["chaser_thrust_ric_km_s2"])
    np.testing.assert_allclose(replay["target_thrust_eci_km_s2"], [[0.0, 0.0, 2.0e-6]])


def test_game_debrief_attempt_index_counts_level_folders(tmp_path: Path) -> None:
    first = game_debrief_path(
        scenario_id="Unit Debrief",
        difficulty="easy",
        attempt_index=1,
        output_dir=tmp_path,
        timestamp=datetime(2026, 5, 14, 12, 30, 45),
    )
    first.parent.mkdir(parents=True)

    assert next_game_debrief_attempt_index(scenario_id="Unit Debrief", output_dir=tmp_path) == 2


def test_game_debrief_is_disabled_for_sandbox_and_arcade_modes() -> None:
    sandbox_path = Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_sandbox.yaml"
    sandbox_config = SimulationConfig.from_yaml(sandbox_path)
    sandbox_training = RPOTrainingConfig.from_metadata(dict(sandbox_config.scenario.metadata or {}))

    arcade_path = Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_arcade_pursuit.yaml"
    arcade_config = SimulationConfig.from_yaml(arcade_path)
    arcade_training = RPOTrainingConfig.from_metadata(dict(arcade_config.scenario.metadata or {}))

    normal_path = Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_01_coast_relative_motion.yaml"
    normal_config = SimulationConfig.from_yaml(normal_path)
    normal_cfg = RPOTrainingConfig.from_metadata(dict(normal_config.scenario.metadata or {}))

    assert _game_debrief_enabled(sandbox_config, sandbox_training, arcade_enabled=False) is False
    assert _game_debrief_enabled(arcade_config, arcade_training, arcade_enabled=True) is False
    assert _game_debrief_enabled(normal_config, normal_cfg, arcade_enabled=False) is True


@pytest.mark.parametrize(
    "config_name,expected_enabled",
    [
        ("game_training_rpo_00_tutorial.yaml", True),
        ("game_training_rpo_01_coast_relative_motion.yaml", True),
        ("game_training_rpo_02_vbar_approach.yaml", True),
        ("game_training_rpo_03_rbar_approach.yaml", True),
        ("game_training_rpo_04_rendezvous.yaml", True),
        ("game_training_rpo_05_passive_cross_track_approach.yaml", True),
        ("game_training_rpo_06_sun_angle_inspection.yaml", True),
        ("game_training_rpo_07_elliptic_burn_then_approach.yaml", True),
        ("game_training_rpo_08_elliptic_nmc.yaml", True),
        ("game_training_rpo_09_elliptic_rendezvous.yaml", True),
        ("game_training_rpo_10_defensive_target_demo.yaml", True),
        ("game_training_rpo_11_evasive_target_survival.yaml", True),
        ("game_training_rpo_arcade_pursuit.yaml", False),
        ("game_training_rpo_sandbox.yaml", False),
    ],
)
def test_game_debrief_policy_and_writer_across_levels(
    config_name: str,
    expected_enabled: bool,
    tmp_path: Path,
) -> None:
    config = SimulationConfig.from_yaml(Path(__file__).resolve().parents[1] / "game" / "configs" / config_name)
    training_cfg = RPOTrainingConfig.from_metadata(dict(config.scenario.metadata or {}))
    enabled = _game_debrief_enabled(
        config,
        training_cfg,
        arcade_enabled=_game_arcade_enabled(config),
    )

    assert enabled is expected_enabled
    if not enabled:
        return

    score = RPOTrainingScore(
        scenario_id=training_cfg.scenario_id,
        learning_goal=training_cfg.learning_goal,
        samples=3,
        elapsed_s=2.0,
        closest_approach_km=0.25,
        final_range_km=0.3,
        final_goal_error_km=0.05,
        final_relative_speed_km_s=0.0001,
        time_inside_keepout_s=0.0,
        approximate_delta_v_m_s=0.04,
        target_delta_v_m_s=0.0,
        burn_axes_satisfied=tuple(training_cfg.required_burn_axes),
        phase_burns_satisfied=tuple(burn.name for burn in training_cfg.required_phase_burns),
        speed_multiplier_changed=bool(training_cfg.require_speed_multiplier_change),
        coast_after_burn_satisfied=training_cfg.required_coast_after_burn_s is None,
        coast_after_burn_s=0.0,
        guided_tutorial_burns_satisfied=tuple(burn.name for burn in training_cfg.guided_tutorial_burns),
        guided_tutorial_burns_total=len(training_cfg.guided_tutorial_burns),
        guided_tutorial_speed_satisfied=True,
        guided_tutorial_speed_target=(
            None
            if training_cfg.guided_tutorial_speed_step is None
            else training_cfg.guided_tutorial_speed_step.target_speed_multiplier
        ),
        achieved_time_s=2.0,
        min_goal_error_km=0.04,
        final_nmt_radial_amplitude_km=training_cfg.goal_nmt_radial_amplitude_km or 0.0,
        final_nmt_cross_track_amplitude_km=training_cfg.goal_nmt_cross_track_amplitude_km,
        final_nmt_radial_amplitude_error_km=0.0,
        final_nmt_cross_track_amplitude_error_km=0.0,
        final_nmt_drift_velocity_error_km_s=0.0,
        goal_met=True,
        level_passed=True,
        level_failed=False,
        pass_fail_reasons=("All pass criteria satisfied.",),
        keepout_violation=False,
        hard_speed_limit_violation=False,
        forbidden_region_violation=False,
        forbidden_region_names=(),
        approach_gate_violation=False,
        approach_gate_names=(),
        approach_gates_satisfied=len(training_cfg.approach_gates),
        approach_gates_total=len(training_cfg.approach_gates),
        inspection_gates_satisfied=len(training_cfg.inspection_gates),
        inspection_gates_total=len(training_cfg.inspection_gates),
        inspection_gate_names=tuple(gate.name for gate in training_cfg.inspection_gates),
        hints=(),
    )
    replay = {
        "time_s": [0.0, 1.0, 2.0],
        "relative_ric": [
            [0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, -0.7, 0.05, 0.0001, 0.0, 0.0],
            [0.0, -0.3, 0.0, 0.0, 0.0001, 0.0],
        ],
        "chaser_thrust_ric_km_s2": [
            [0.0, 0.0, 0.0],
            [0.00001, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        "target_thrust_eci_km_s2": [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
    }
    out = write_game_debrief(
        game_debrief_path(
            scenario_id=training_cfg.scenario_id,
            difficulty="easy",
            attempt_index=1,
            output_dir=tmp_path,
            timestamp=datetime(2026, 5, 22, 12, 0, 0),
        ),
        config=training_cfg,
        score=score,
        difficulty="easy",
        objective_checklist=_mission_checklist(training_cfg, score),
        replay_history=replay,
    )

    assert out.exists()
    assert (out.parent / "summary.json").exists()
    assert (out.parent / "plots" / "mission_timeline.png").exists()
    assert (out.parent / "plots" / "ric_2d_plots.png").exists()
    assert "## Stats Summary" in out.read_text(encoding="utf-8")


def test_tutorial_debrief_history_can_scope_to_final_free_maneuver_phase() -> None:
    cfg = RPOTrainingConfig(
        enabled=True,
        scenario_id="rpo_00_tutorial",
        guided_tutorial_burns=(
            GuidedTutorialBurnConfig(
                name="plus_in_track",
                axis="in_track",
                sign=1,
                delta_v_m_s=0.25,
            ),
        ),
    )
    tracker = RPOTrainingTracker(cfg)
    target_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.54605329, 0.0], dtype=float)
    guided_rel = np.array([0.0, -0.8, 0.0, 0.0, 0.0, 0.0], dtype=float)
    final_rel = np.array([0.0, -0.25, 0.0, 0.0, 0.0, 0.0], dtype=float)

    for time_s, rel_ric in ((1.0, guided_rel),):
        tracker.record(
            snapshot=type(
                "Snapshot",
                (),
                {
                    "time_s": time_s,
                    "truth": {
                        "target": target_state,
                        "chaser": ric_rect_state_to_eci(rel_ric, target_state[:3], target_state[3:]),
                    },
                    "applied_thrust": {"chaser": np.zeros(3, dtype=float)},
                },
            )()
        )
    tracker.mark_guided_tutorial_burn_complete("plus_in_track")
    tracker.clear(reset_guided_tutorial_progress=False)
    tracker.record(
        snapshot=type(
            "Snapshot",
            (),
            {
                "time_s": 99.0,
                "truth": {
                    "target": target_state,
                    "chaser": ric_rect_state_to_eci(final_rel, target_state[:3], target_state[3:]),
                },
                "applied_thrust": {"chaser": np.zeros(3, dtype=float)},
            },
        )()
    )

    replay = tracker_replay_history(tracker)

    assert tracker.guided_tutorial_burns_satisfied() == ("plus_in_track",)
    assert replay["time_s"] == [99.0]
    assert replay["relative_ric"][0][:3] == pytest.approx([0.0, -0.25, 0.0])


def test_game_debrief_ric_plot_axes_keep_radial_vertical() -> None:
    assert _plane_axes("RI") == (1, 0, "I", "R")
    assert _plane_axes("RC") == (2, 0, "C", "R")
    assert _plane_axes("IC") == (1, 2, "I", "C")


def test_game_debrief_cumulative_delta_v_matches_sampled_accel_integral() -> None:
    thrust_km_s2 = np.array(
        [
            [0.001, 0.0, 0.0],
            [0.0, 0.002, 0.0],
            [np.nan, 0.0, 0.0],
            [0.003, 0.0, 0.0],
        ],
        dtype=float,
    )
    t_s = np.array([0.0, 2.0, 5.0, 8.0], dtype=float)

    cumulative = _cumulative_delta_v_m_s(thrust_km_s2, t_s)

    assert cumulative == pytest.approx([0.0, 2.0, 8.0, 8.0])


def test_game_debrief_timeline_uses_burn_intervals() -> None:
    cfg = RPOTrainingConfig(enabled=True, scenario_id="timeline-unit")
    score = RPOTrainingScore(
        scenario_id=cfg.scenario_id,
        learning_goal="",
        samples=5,
        elapsed_s=4.0,
        closest_approach_km=0.2,
        final_range_km=0.3,
        final_goal_error_km=0.0,
        final_relative_speed_km_s=0.0,
        time_inside_keepout_s=0.0,
        approximate_delta_v_m_s=0.0,
        target_delta_v_m_s=0.0,
        burn_axes_satisfied=(),
        phase_burns_satisfied=(),
        speed_multiplier_changed=False,
        coast_after_burn_satisfied=True,
        coast_after_burn_s=0.0,
        guided_tutorial_burns_satisfied=(),
        guided_tutorial_burns_total=0,
        guided_tutorial_speed_satisfied=True,
        guided_tutorial_speed_target=None,
        achieved_time_s=4.0,
        min_goal_error_km=0.0,
        final_nmt_radial_amplitude_km=0.0,
        final_nmt_cross_track_amplitude_km=0.0,
        final_nmt_radial_amplitude_error_km=0.0,
        final_nmt_cross_track_amplitude_error_km=0.0,
        final_nmt_drift_velocity_error_km_s=0.0,
        goal_met=True,
        level_passed=True,
        level_failed=False,
        pass_fail_reasons=("All pass criteria satisfied.",),
        keepout_violation=False,
        hard_speed_limit_violation=False,
        forbidden_region_violation=False,
        forbidden_region_names=(),
        approach_gate_violation=False,
        approach_gate_names=(),
        approach_gates_satisfied=0,
        approach_gates_total=0,
        inspection_gates_satisfied=0,
        inspection_gates_total=0,
        inspection_gate_names=(),
        hints=(),
    )
    replay = {
        "time_s": [0.0, 1.0, 2.0, 3.0, 4.0],
        "relative_ric": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 5,
        "chaser_thrust_ric_km_s2": [
            [0.0, 0.0, 0.0],
            [0.001, 0.0, 0.0],
            [0.001, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.001, 0.0, 0.0],
        ],
    }

    events = _event_timeline(config=cfg, score=score, replay_history=replay)
    burn_events = [event for event in events if event.get("kind") == "interval"]

    assert _active_segments(np.array([False, True, True, False, True])) == [(1, 2), (4, 4)]
    assert burn_events[0]["start_time_s"] == pytest.approx(1.0)
    assert burn_events[0]["end_time_s"] == pytest.approx(3.0)
    assert burn_events[0]["label"] == "Control input"
    assert burn_events[1]["start_time_s"] == pytest.approx(4.0)
    assert burn_events[1]["end_time_s"] == pytest.approx(4.0)


def test_open_game_debrief_folder_opens_report_parent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []
    report = tmp_path / "attempt_001" / "report.md"
    report.parent.mkdir()
    report.write_text("# Debrief\n", encoding="utf-8")

    monkeypatch.setattr("sim.game.debrief.sys.platform", "darwin")
    monkeypatch.setattr("sim.game.debrief.subprocess.Popen", lambda cmd: calls.append(list(cmd)))

    assert open_game_debrief_folder(report) is True
    assert calls == [["open", str(report.parent)]]


def test_game_frame_recorder_finishes_or_discards_with_fake_writer(tmp_path: Path) -> None:
    class FakeWriter:
        def __init__(self, path: Path):
            self.path = path
            self.frames: list[np.ndarray] = []
            self.closed = False

        def append_data(self, frame: np.ndarray) -> None:
            self.frames.append(np.array(frame, dtype=np.uint8))

        def close(self) -> None:
            self.closed = True
            self.path.write_bytes(b"fake-mp4")

    writers: list[FakeWriter] = []

    def factory(path: Path, fps: float) -> FakeWriter:
        assert fps == 12.0
        writer = FakeWriter(path)
        writers.append(writer)
        return writer

    path = tmp_path / "attempt.mp4"
    recorder = GameFrameRecorder.start(path, fps=12.0, writer_factory=factory)
    recorder.capture_frame(np.zeros((2, 3, 3), dtype=np.uint8))

    assert recorder.finish() == path
    assert recorder.saved is True
    assert recorder.frames_written == 1
    assert writers[-1].closed is True
    assert path.exists()

    recorder = GameFrameRecorder.start(path, fps=12.0, writer_factory=factory)
    recorder.capture_frame(np.zeros((2, 3, 4), dtype=np.uint8))
    recorder.discard()

    assert recorder.saved is False
    assert writers[-1].closed is True
    assert not path.exists()


def test_add_looped_audio_to_video_muxes_audio_with_ffmpeg(tmp_path: Path) -> None:
    video = tmp_path / "attempt.mp4"
    audio = tmp_path / "level.wav"
    video.write_bytes(b"silent-video")
    audio.write_bytes(b"level-audio")
    captured: dict[str, list[str]] = {}

    def fake_runner(cmd, **kwargs):
        captured["cmd"] = [str(part) for part in cmd]
        assert kwargs["check"] is True
        Path(cmd[-1]).write_bytes(b"muxed-video")

    out = add_looped_audio_to_video(video, audio, ffmpeg_exe="/usr/local/bin/ffmpeg", runner=fake_runner)

    assert out == video
    assert video.read_bytes() == b"muxed-video"
    cmd = captured["cmd"]
    assert cmd[:6] == ["/usr/local/bin/ffmpeg", "-y", "-i", str(video), "-stream_loop", "-1"]
    assert cmd[6:8] == ["-i", str(audio)]
    assert "-shortest" in cmd
    assert cmd[cmd.index("-c:v") + 1] == "copy"
    assert cmd[cmd.index("-c:a") + 1] == "aac"


def test_add_level_music_to_recording_uses_training_level_track(tmp_path: Path, monkeypatch) -> None:
    video = tmp_path / "attempt.mp4"
    video.write_bytes(b"silent-video")
    calls: list[tuple[Path, Path]] = []

    def fake_add_audio(recording_path: Path, music_path: Path) -> Path:
        calls.append((recording_path, music_path))
        return recording_path

    monkeypatch.setattr(game_recording_controller, "add_looped_audio_to_video", fake_add_audio)
    cfg = RPOTrainingConfig(enabled=True, scenario_id="rpo_00_tutorial")

    assert _add_level_music_to_recording(video, cfg) == video
    assert calls == [(video, LEVEL_MUSIC_PATHS["rpo_00_tutorial"])]


def test_add_level_music_to_recording_prefers_arcade_override(tmp_path: Path, monkeypatch) -> None:
    video = tmp_path / "attempt.mp4"
    video.write_bytes(b"silent-video")
    calls: list[tuple[Path, Path]] = []

    def fake_add_audio(recording_path: Path, music_path: Path) -> Path:
        calls.append((recording_path, music_path))
        return recording_path

    monkeypatch.setattr(game_recording_controller, "add_looped_audio_to_video", fake_add_audio)
    cfg = RPOTrainingConfig(enabled=True, scenario_id="rpo_arcade_pursuit")
    boss_track = game_runner.GAME_MUSIC_DIR / "28_high_shred_boss_riff.wav"

    assert _add_level_music_to_recording(video, cfg, override_level_path=boss_track) == video
    assert calls == [(video, boss_track)]


def test_game_recording_defaults_to_recording_fps() -> None:
    assert game_runner.run_game_mode.__kwdefaults__["recording_fps"] == game_runner.GAME_RECORDING_FPS


def test_start_game_recorder_disables_when_writer_start_fails(tmp_path: Path, monkeypatch) -> None:
    config = SimulationConfig.from_dict(_game_config(tmp_path))
    monkeypatch.setattr(
        game_recording_controller.GameFrameRecorder,
        "start",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("writer unavailable")),
    )

    recorder = _start_game_recorder(
        enabled=True,
        config=config,
        difficulty="easy",
        attempt_index=1,
        output_dir=tmp_path,
        fps=30.0,
    )

    assert recorder is None


def test_start_game_clip_recorder_uses_clip_path(tmp_path: Path, monkeypatch) -> None:
    config = SimulationConfig.from_dict(_game_config(tmp_path))
    starts: list[Path] = []

    def fake_start(path, **kwargs):
        starts.append(Path(path))
        return "recorder"

    monkeypatch.setattr(game_recording_controller.GameFrameRecorder, "start", fake_start)

    recorder = _start_game_clip_recorder(
        enabled=True,
        config=config,
        difficulty="easy",
        clip_index=4,
        output_dir=tmp_path,
        fps=30.0,
    )

    assert recorder == "recorder"
    assert starts[0].parent == tmp_path / "clips"
    assert starts[0].name.startswith(f"{config.scenario.scenario_name}_easy_")
    assert starts[0].name.endswith("_clip04.mp4")


def test_recording_hold_frame_count_uses_duration_and_fps() -> None:
    assert game_recording_controller.recording_hold_frame_count(duration_s=3.0, fps=30.0) == 90
    assert game_recording_controller.recording_hold_frame_count(duration_s=2.5, fps=24.0) == 60
    assert game_recording_controller.recording_hold_frame_count(duration_s=-1.0, fps=30.0) == 0


def test_next_available_recording_path_avoids_existing_clip(tmp_path: Path) -> None:
    base = tmp_path / "clip.mp4"
    base.write_bytes(b"existing")

    assert game_recording_controller.next_available_recording_path(base) == tmp_path / "clip_02.mp4"

    (tmp_path / "clip_02.mp4").write_bytes(b"existing")

    assert game_recording_controller.next_available_recording_path(base) == tmp_path / "clip_03.mp4"


def test_recording_controller_capture_hold_repeats_current_frame(tmp_path: Path, monkeypatch) -> None:
    config = SimulationConfig.from_dict(_game_config(tmp_path))
    captures: list[object] = []

    class FakeRecorder:
        saved = False

    def fake_capture(recorder, dashboard):
        captures.append(dashboard)
        return recorder

    monkeypatch.setattr(game_recording_controller, "safe_capture_recording_frame", fake_capture)
    controller = game_recording_controller.GameRecordingController(
        enabled=True,
        config=config,
        difficulty="easy",
        fps=10.0,
        recorder=FakeRecorder(),
    )
    dashboard = object()

    assert controller.capture_hold(dashboard, duration_s=3.0) is controller.recorder
    assert captures == [dashboard] * 30


def test_safe_capture_recording_frame_discards_and_disables_on_capture_failure(monkeypatch) -> None:
    class FakeRecorder:
        saved = False

        def __init__(self) -> None:
            self.discarded = False

        def discard(self) -> None:
            self.discarded = True

    recorder = FakeRecorder()
    monkeypatch.setattr(
        game_recording_controller,
        "capture_recording_frame",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("capture failed")),
    )

    returned = _safe_capture_recording_frame(recorder, type("Dashboard", (), {"screen": object()})())

    assert returned is None
    assert recorder.discarded is True


def test_finish_game_recording_discards_and_disables_on_finalize_failure() -> None:
    class FakeRecorder:
        def __init__(self) -> None:
            self.discarded = False

        def finish(self) -> Path:
            raise RuntimeError("encode failed")

        def discard(self) -> None:
            self.discarded = True

    recorder = FakeRecorder()

    returned = _finish_game_recording(recorder, RPOTrainingConfig(enabled=True))

    assert returned is None
    assert recorder.discarded is True


def test_recording_controller_restart_does_not_delete_saved_recording(tmp_path: Path, monkeypatch) -> None:
    class FakeRecorder:
        saved = True

        def __init__(self) -> None:
            self.discarded = False

        def discard(self) -> None:
            self.discarded = True

    saved_recorder = FakeRecorder()
    starts: list[int] = []

    def fake_start(**kwargs):
        starts.append(int(kwargs["attempt_index"]))
        return None

    monkeypatch.setattr(game_recording_controller, "start_game_recorder", fake_start)
    controller = game_recording_controller.GameRecordingController(
        enabled=True,
        config=SimulationConfig.from_dict(_game_config(tmp_path)),
        difficulty="easy",
        output_dir=tmp_path,
        recorder=saved_recorder,
    )

    controller.restart()

    assert saved_recorder.discarded is False
    assert starts == [2]
    assert controller.attempt_index == 2


def test_defensive_target_provider_pulses_on_unsafe_closure() -> None:
    provider = DefensiveTargetIntentProvider(
        trigger_range_km=1.2,
        trigger_closing_speed_km_s=0.00025,
        max_accel_km_s2=7.5e-6,
    )
    target_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.54605329, 0.0], dtype=float)
    close_rel_ric = np.array([0.0, -1.0, 0.0, 0.0, 0.0005, 0.0], dtype=float)
    far_rel_ric = np.array([0.0, -2.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    close_chaser_state = ric_rect_state_to_eci(close_rel_ric, target_state[:3], target_state[3:])
    far_chaser_state = ric_rect_state_to_eci(far_rel_ric, target_state[:3], target_state[3:])
    target = StateTruth(
        position_eci_km=target_state[:3],
        velocity_eci_km_s=target_state[3:],
        attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        angular_rate_body_rad_s=np.zeros(3, dtype=float),
        mass_kg=1800.0,
        t_s=0.0,
    )
    active = provider(truth=target, own_knowledge={"chaser": _knowledge_from_state6(close_chaser_state)}, t_s=10.0)
    inactive = provider(truth=target, own_knowledge={"chaser": _knowledge_from_state6(far_chaser_state)}, t_s=10.0)

    assert np.isclose(np.linalg.norm(active["thrust_eci_km_s2"]), 7.5e-6)
    assert active["command_mode_flags"]["target_defensive"] is True
    assert np.allclose(inactive["thrust_eci_km_s2"], np.zeros(3), atol=1e-15)
    assert inactive["command_mode_flags"]["target_defensive"] is False


def test_defensive_target_provider_caps_delta_v_budget() -> None:
    provider = DefensiveTargetIntentProvider(
        trigger_range_km=1.2,
        trigger_closing_speed_km_s=0.00025,
        max_accel_km_s2=1.0e-3,
        max_delta_v_m_s=5.0,
    )
    target_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.54605329, 0.0], dtype=float)
    rel_ric = np.array([0.0, -1.0, 0.0, 0.0, 0.0005, 0.0], dtype=float)
    chaser_state = ric_rect_state_to_eci(rel_ric, target_state[:3], target_state[3:])
    target = StateTruth(
        position_eci_km=target_state[:3],
        velocity_eci_km_s=target_state[3:],
        attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        angular_rate_body_rad_s=np.zeros(3, dtype=float),
        mass_kg=1800.0,
        t_s=0.0,
    )
    own_knowledge = {"chaser": _knowledge_from_state6(chaser_state)}
    first = provider(truth=target, own_knowledge=own_knowledge, t_s=0.0)
    second = provider(truth=target, own_knowledge=own_knowledge, t_s=10.0)
    third = provider(truth=target, own_knowledge=own_knowledge, t_s=20.0)

    assert np.isclose(np.linalg.norm(first["thrust_eci_km_s2"]), 1.0e-3)
    assert np.isclose(provider.used_delta_v_m_s, 5.0)
    assert np.isclose(np.linalg.norm(second["thrust_eci_km_s2"]), 5.0e-4)
    assert np.allclose(third["thrust_eci_km_s2"], np.zeros(3), atol=1e-15)
    assert third["command_mode_flags"]["target_defensive_budget_exhausted"] is True


def test_defensive_target_provider_charges_first_timed_pulse() -> None:
    provider = DefensiveTargetIntentProvider(
        trigger_range_km=1.2,
        trigger_closing_speed_km_s=0.00025,
        max_accel_km_s2=1.0e-3,
        max_delta_v_m_s=5.0,
    )
    target_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.54605329, 0.0], dtype=float)
    rel_ric = np.array([0.0, -1.0, 0.0, 0.0, 0.0005, 0.0], dtype=float)
    chaser_state = ric_rect_state_to_eci(rel_ric, target_state[:3], target_state[3:])
    target = StateTruth(
        position_eci_km=target_state[:3],
        velocity_eci_km_s=target_state[3:],
        attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        angular_rate_body_rad_s=np.zeros(3, dtype=float),
        mass_kg=1800.0,
        t_s=0.0,
    )
    own_knowledge = {"chaser": _knowledge_from_state6(chaser_state)}
    first = provider(truth=target, own_knowledge=own_knowledge, t_s=10.0, dt_s=10.0)
    second = provider(truth=target, own_knowledge=own_knowledge, t_s=20.0, dt_s=10.0)

    assert np.isclose(np.linalg.norm(first["thrust_eci_km_s2"]), 5.0e-4)
    assert np.isclose(provider.used_delta_v_m_s, 5.0)
    assert np.allclose(second["thrust_eci_km_s2"], np.zeros(3), atol=1e-15)
    assert second["command_mode_flags"]["target_defensive_budget_exhausted"] is True


def test_level_nine_uses_target_reference_for_game_ric_frame() -> None:
    config_path = (
        Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_10_defensive_target_demo.yaml"
    )
    config = SimulationConfig.from_yaml(config_path)
    session = SimulationSession.from_config(config)
    snap = session.reset()

    assert _game_ric_reference_object_id(config, "target") == "target_reference"
    assert snap is not None
    assert "target_reference" in snap.truth
    assert snap.truth["target_reference"].shape[0] >= 6


def test_level_nine_ric_translation_commands_use_target_reference_frame() -> None:
    config_path = (
        Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_10_defensive_target_demo.yaml"
    )
    config = SimulationConfig.from_yaml(config_path)
    training_cfg = RPOTrainingConfig.from_metadata(dict(config.scenario.metadata or {}))
    state = KeyboardCommandState(yaw=1.0)

    session, _, snap0 = _start_game_attempt(
        config,
        command_state=state,
        training_cfg=training_cfg,
        controlled_object_id="chaser",
        attitude_rate_deg_s=45.0,
        control_mode="ric_translation",
        ric_reference_object_id="target_reference",
    )
    snap1 = session.step()

    ref_state = np.array(snap0.truth["target_reference"][:6], dtype=float)
    c_ir = ric_dcm_ir_from_rv(ref_state[:3], ref_state[3:6])
    expected = c_ir @ np.array([0.0, 1.5e-5, 0.0], dtype=float)
    assert np.allclose(snap1.applied_thrust["chaser"], expected, atol=1e-12)


def test_game_attempt_forces_acceleration_off(monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_00_tutorial.yaml"
    config = SimulationConfig.from_yaml(config_path).with_value("simulator.acceleration.mode", "auto")
    training_cfg = RPOTrainingConfig.from_metadata(dict(config.scenario.metadata or {}))
    monkeypatch.setenv(ACCELERATION_ENV, "auto")

    session, _, _ = _start_game_attempt(
        config,
        command_state=KeyboardCommandState(),
        training_cfg=training_cfg,
        controlled_object_id="chaser",
        attitude_rate_deg_s=45.0,
        control_mode="ric_translation",
        ric_reference_object_id="target",
    )

    assert os.environ[ACCELERATION_ENV] == "auto"
    assert session.config.scenario.simulator.acceleration.mode == "off"
    assert session.config.scenario.simulator.acceleration.warmup is False
    assert session.config.scenario.simulator.acceleration["env_override"] is False
    assert acceleration_settings_from_config(session.config.scenario).enabled is False


def test_level_eleven_game_attempt_uses_dynamic_history() -> None:
    config_path = (
        Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_06_sun_angle_inspection.yaml"
    )
    config = SimulationConfig.from_yaml(config_path)
    training_cfg = RPOTrainingConfig.from_metadata(dict(config.scenario.metadata or {}))

    session, _, snapshot = _start_game_attempt(
        config,
        command_state=KeyboardCommandState(),
        training_cfg=training_cfg,
        controlled_object_id="chaser",
        attitude_rate_deg_s=45.0,
        control_mode="ric_translation",
        ric_reference_object_id="target",
    )

    engine = session._engine
    assert snapshot is not None
    assert isinstance(session, GamePhysicsSession)
    assert engine is not None
    assert engine.history_mode == "dynamic"
    assert engine.planned_samples == 86401
    assert engine.n < engine.planned_samples
    assert engine.allocated_history_samples == 4096
    assert engine.retained_start_step == 0
    assert engine.retained_end_step == 0
    assert engine.truth_hist["chaser"].shape[0] == engine.n
    assert engine.history_memory_estimate.estimated_peak_mb < 1024.0
    with pytest.raises(RuntimeError, match="Dynamic history mode"):
        session._session.run()

    for _ in range(engine.n + 5):
        snapshot = session.step(dt_s=1.0)

    assert snapshot.step_index == engine.n + 5
    assert snapshot.time_s == pytest.approx(float(engine.n + 5))
    assert engine.history_mode == "dynamic"
    assert engine.n < engine.planned_samples
    assert engine.sample_offset > 0
    assert engine.current_index < engine.n
    assert engine.retained_start_step > 0
    assert engine.retained_end_step == snapshot.step_index
    assert engine.retained_sample_count >= (engine.allocated_history_samples * 3) // 4
    assert engine.truth_hist["chaser"].shape[0] == engine.n
    assert engine.history_memory_estimate.estimated_peak_mb < 1024.0
    assert engine.snapshot(snapshot.step_index)["time_s"] == pytest.approx(snapshot.time_s)
    with pytest.raises(IndexError):
        engine.snapshot(0)


def test_game_attempt_dynamic_history_avoids_full_history_memory_budget() -> None:
    config_path = (
        Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_06_sun_angle_inspection.yaml"
    )
    config = (
        SimulationConfig.from_yaml(config_path)
        .with_value("simulator.dynamics.orbit.orbit_substep_s", 0.1)
        .with_value("simulator.dt_s", 0.1)
        .with_value("outputs.resource_limits.max_history_memory_mb", 128.0)
    )
    training_cfg = RPOTrainingConfig.from_metadata(dict(config.scenario.metadata or {}))

    with pytest.raises(SimulationMemoryBudgetError):
        SimulationSession.from_config(config).reset()

    session, _, snapshot = _start_game_attempt(
        config,
        command_state=KeyboardCommandState(),
        training_cfg=training_cfg,
        controlled_object_id="chaser",
        attitude_rate_deg_s=45.0,
        control_mode="ric_translation",
        ric_reference_object_id="target",
    )

    engine = session._engine
    assert snapshot is not None
    assert engine is not None
    assert engine.history_mode == "dynamic"
    assert engine.planned_samples == 864001
    assert engine.allocated_history_samples == 4096
    assert engine.history_memory_estimate.estimated_peak_mb < 128.0


def test_game_attempt_uses_configured_retained_history_samples(tmp_path: Path) -> None:
    cfg = _game_config(tmp_path)
    cfg["simulator"]["duration_s"] = 20.0
    game_meta = cfg.setdefault("metadata", {}).setdefault("game", {})
    game_meta["retained_history_samples"] = 8
    config = SimulationConfig.from_dict(cfg)
    training_cfg = RPOTrainingConfig.from_metadata(dict(config.scenario.metadata or {}))

    session, _, snapshot = _start_game_attempt(
        config,
        command_state=KeyboardCommandState(),
        training_cfg=training_cfg,
        controlled_object_id="chaser",
        attitude_rate_deg_s=45.0,
        control_mode="ric_translation",
        ric_reference_object_id="target",
    )

    engine = session._engine
    assert snapshot is not None
    assert _game_retained_history_samples(config) == 8
    assert engine is not None
    assert engine.history_mode == "dynamic"
    assert engine.planned_samples > 8
    assert engine.allocated_history_samples == 8


def test_level_nine_restart_gets_fresh_defensive_target_provider() -> None:
    config_path = (
        Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_10_defensive_target_demo.yaml"
    )
    config = SimulationConfig.from_yaml(config_path)
    training_cfg = RPOTrainingConfig.from_metadata(dict(config.scenario.metadata or {}))
    state = KeyboardCommandState()

    session1, _, _ = _start_game_attempt(
        config,
        command_state=state,
        training_cfg=training_cfg,
        controlled_object_id="chaser",
        attitude_rate_deg_s=45.0,
        control_mode="ric_translation",
        ric_reference_object_id="target_reference",
    )
    session2, _, _ = _start_game_attempt(
        config,
        command_state=state,
        training_cfg=training_cfg,
        controlled_object_id="chaser",
        attitude_rate_deg_s=45.0,
        control_mode="ric_translation",
        ric_reference_object_id="target_reference",
    )

    target_provider1 = session1._external_intent_providers["target"]
    target_provider2 = session2._external_intent_providers["target"]
    assert target_provider1 is not target_provider2
    assert target_provider2.used_delta_v_m_s == 0.0


def test_level_nine_goal_is_100_meter_close_approach() -> None:
    config_path = (
        Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_10_defensive_target_demo.yaml"
    )
    config = SimulationConfig.from_yaml(config_path)
    training_cfg = RPOTrainingConfig.from_metadata(dict(config.scenario.metadata or {}))
    defensive_target = config.scenario.metadata["game"]["defensive_target"]
    pass_criteria = tuple(config.scenario.metadata["game"]["training"]["pass_criteria"])

    assert training_cfg.goal_range_km == pytest.approx(0.1)
    assert training_cfg.goal_range_tolerance_km is None
    assert training_cfg.goal_radius_km is None
    assert training_cfg.keepout_radius_km is None
    assert training_cfg.max_goal_speed_km_s is None
    assert training_cfg.max_time_s == pytest.approx(14400.0)
    assert config.scenario.simulator.duration_s == pytest.approx(14400.0)
    assert _game_camera_mode(config) == "target_pair"
    assert defensive_target["keepout_radius_km"] == pytest.approx(0.1)
    assert defensive_target["max_delta_v_m_s"] == pytest.approx(0.1)
    assert any("100 m" in item for item in pass_criteria)


def test_pursuit_arcade_uses_level_nine_shape_with_arcade_clock() -> None:
    config_path = (
        Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_arcade_pursuit.yaml"
    )
    config = SimulationConfig.from_yaml(config_path)
    training_cfg = RPOTrainingConfig.from_metadata(dict(config.scenario.metadata or {}))
    defensive_target = config.scenario.metadata["game"]["defensive_target"]

    assert _game_arcade_enabled(config) is True
    assert _game_arcade_initial_time_s(config, training_cfg) == pytest.approx(12000.0)
    assert _game_arcade_round_bonus_time_s(config) == pytest.approx(0.0)
    assert _game_arcade_delta_v_bonus_time_per_m_s(config) == pytest.approx(1000.0)
    assert _game_arcade_goal_range_step_km(config) == pytest.approx(0.005)
    assert _game_arcade_min_goal_range_km(config) == pytest.approx(0.005)
    assert config.scenario.metadata["game"]["level_name"] == "Pursuit Arcade"
    assert training_cfg.scenario_id == "rpo_arcade_pursuit"
    assert training_cfg.goal_range_km == pytest.approx(0.1)
    assert training_cfg.max_time_s == pytest.approx(12000.0)
    assert training_cfg.max_delta_v_m_s == pytest.approx(3.0)
    assert config.scenario.simulator.duration_s == pytest.approx(12000.0)
    assert config.scenario.simulator.dt_s == pytest.approx(1.0)
    assert defensive_target["max_delta_v_m_s"] == pytest.approx(0.1)
    assert defensive_target["delta_v_ramp_after_round"] == 20
    assert defensive_target["delta_v_ramp_step_m_s"] == pytest.approx(0.01)


def test_level_zero_tutorial_is_passive_close_range_intro() -> None:
    config_path = (
        Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_00_tutorial.yaml"
    )
    config = SimulationConfig.from_yaml(config_path)
    training_cfg = RPOTrainingConfig.from_metadata(dict(config.scenario.metadata or {}))
    chaser_initial = config.scenario.objects["chaser"].initial_state["relative_to_target_ric"]

    assert training_cfg.scenario_id == "rpo_00_tutorial"
    assert config.scenario.metadata["game"]["level_name"] == "Level 0 - Pilot Tutorial"
    assert _game_control_mode(config) == "ric_translation"
    assert _game_camera_mode(config) == "target_pair"
    assert _game_target_centered_plot_planes(config) == ("RI", "RC", "IC")
    assert _game_plot_overlays_in_zoom(config) is False
    assert training_cfg.goal_range_km == pytest.approx(0.25)
    assert training_cfg.goal_radius_km is None
    assert training_cfg.keepout_radius_km is None
    assert training_cfg.max_goal_speed_km_s == pytest.approx(0.0003)
    assert training_cfg.max_time_s == pytest.approx(18000.0)
    assert training_cfg.max_delta_v_m_s == pytest.approx(12.0)
    assert training_cfg.required_burn_axes == ()
    assert training_cfg.require_speed_multiplier_change is False
    assert training_cfg.required_coast_after_burn_s is None
    assert [burn.display_label for burn in training_cfg.guided_tutorial_burns] == [
        "+I burn",
        "-I burn",
        "+R burn",
        "-R burn",
        "+C burn",
        "-C burn",
    ]
    assert {burn.axis for burn in training_cfg.guided_tutorial_burns} == {"radial", "in_track", "cross_track"}
    assert all(burn.delta_v_m_s == pytest.approx(0.25) for burn in training_cfg.guided_tutorial_burns)
    assert training_cfg.guided_tutorial_speed_step is not None
    assert training_cfg.guided_tutorial_speed_step.after_burn_name == "plus_in_track"
    assert training_cfg.guided_tutorial_speed_step.target_speed_multiplier == pytest.approx(10.0)
    assert "Want to go faster" in training_cfg.guided_tutorial_speed_step.hint
    assert "toward or away from Earth" in training_cfg.axis_descriptions["radial"]
    assert "Stage 2" in training_cfg.tutorial_stage_hints["in_track"]
    assert config.scenario.simulator.duration_s == pytest.approx(18000.0)
    assert chaser_initial["frame"] == "rect"
    assert np.allclose(chaser_initial["state"], np.array([0.0, -0.8, 0.0, 0.0, 0.0, 0.0]))


def test_level_six_combines_elliptic_burn_familiarization_and_approach() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "game"
        / "configs"
        / "game_training_rpo_07_elliptic_burn_then_approach.yaml"
    )
    config = SimulationConfig.from_yaml(config_path)
    training_cfg = RPOTrainingConfig.from_metadata(dict(config.scenario.metadata or {}))
    target_coes = config.scenario.objects["target"].initial_state["coes"]

    assert training_cfg.scenario_id == "rpo_07_elliptic_burn_then_approach"
    assert config.scenario.metadata["game"]["level_name"] == "Level 7 - Elliptical Approach"
    assert _game_control_mode(config) == "ric_translation"
    assert _game_coast_prediction_model(config) == "tschauner_hempel"
    assert _game_camera_mode(config) == "rule_toggle_pair"
    assert _game_camera_rule_mode(config) == "current_pair"
    assert _game_camera_rule_toggle_enabled(config) is True
    assert _game_target_centered_plot_planes(config) == ("RI", "RC")
    assert _game_plot_prediction_full_trajectory_only(config) is True
    assert training_cfg.survival_goal is False
    assert training_cfg.goal_radius_km == pytest.approx(0.18)
    assert np.allclose(training_cfg.goal_relative_ric_km, np.array([0.0, -0.75, 0.0]))
    assert training_cfg.max_goal_speed_km_s == pytest.approx(0.00035)
    assert training_cfg.keepout_radius_km == pytest.approx(0.25)
    assert training_cfg.max_time_s == pytest.approx(9000.0)
    assert training_cfg.max_delta_v_m_s == pytest.approx(8.0)
    assert "radial" in training_cfg.required_burn_axes
    assert "in_track" in training_cfg.required_burn_axes
    assert training_cfg.require_speed_multiplier_change is True
    assert target_coes["ecc"] == pytest.approx(0.25)
    assert target_coes["true_anomaly_deg"] == pytest.approx(60.0)


def test_level_seven_rule_toggle_keeps_ri_and_rc_target_centered() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "game"
        / "configs"
        / "game_training_rpo_07_elliptic_burn_then_approach.yaml"
    )
    config = SimulationConfig.from_yaml(config_path)
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.camera_mode = _game_camera_mode(config)
    dashboard.camera_rule_mode = _game_camera_rule_mode(config)
    dashboard.target_centered_plot_planes = _game_target_centered_plot_planes(config)
    dashboard.target_centered_plot_axes = _game_target_centered_plot_axes(config)
    chaser_current = np.array([0.0, -4.5, 0.2], dtype=float)
    target_current = np.zeros(3, dtype=float)

    ri_center = dashboard._camera_center_ric(
        chaser_current=chaser_current,
        target_current=target_current,
        x_axis=1,
        y_axis=0,
    )
    rc_center = dashboard._camera_center_ric(
        chaser_current=chaser_current,
        target_current=target_current,
        x_axis=2,
        y_axis=0,
    )

    assert dashboard.camera_mode == "rule_toggle_pair"
    assert dashboard._camera_rule_mode_key() == "current_pair"
    assert dashboard.target_centered_plot_planes == ("RI", "RC")
    assert ri_center == pytest.approx(target_current)
    assert rc_center == pytest.approx(target_current)


def test_level_seven_operator_mode_relaxes_required_burn_axes() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "game"
        / "configs"
        / "game_training_rpo_07_elliptic_burn_then_approach.yaml"
    )
    config = SimulationConfig.from_yaml(config_path)
    training_cfg = RPOTrainingConfig.from_metadata(dict(config.scenario.metadata or {}))

    assert training_config_for_game_mode(training_cfg, game_mode="pilot").required_burn_axes == (
        "radial",
        "in_track",
    )
    assert training_config_for_game_mode(training_cfg, game_mode="operator").required_burn_axes == ()


def test_operator_level_seven_brief_omits_required_burn_axis_criteria() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "game" / "configs"
    option = next(
        option for option in discover_game_scenarios_for_mode(config_dir, mode="operator")
        if option.scenario_id == "rpo_07_elliptic_burn_then_approach"
    )

    assert not any("radial burn" in criterion.lower() for criterion in option.pass_criteria)
    assert not any("in-track burn" in criterion.lower() for criterion in option.pass_criteria)
    assert "First test radial and in-track burns" not in option.player_brief


def test_level_seven_is_elliptic_nmc_lesson() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "game"
        / "configs"
        / "game_training_rpo_08_elliptic_nmc.yaml"
    )
    config = SimulationConfig.from_yaml(config_path)
    training_cfg = RPOTrainingConfig.from_metadata(dict(config.scenario.metadata or {}))
    target_coes = config.scenario.objects["target"].initial_state["coes"]

    assert training_cfg.scenario_id == "rpo_08_elliptic_nmc"
    assert config.scenario.metadata["game"]["level_name"] == "Level 8 - Elliptical NMC"
    assert _game_control_mode(config) == "ric_translation"
    assert _game_coast_prediction_model(config) == "tschauner_hempel"
    assert _game_target_centered_plot_planes(config) == ("RI",)
    assert _game_plot_fixed_axis_half_span_km(config) == {"RI": (3.25, 1.6)}
    assert training_cfg.goal_nmt_radial_amplitude_km == pytest.approx(1.2)
    assert training_cfg.goal_nmt_cross_track_amplitude_km == pytest.approx(0.8)
    assert training_cfg.goal_nmt_cross_track_phase_deg == pytest.approx(90.0)
    assert training_cfg.goal_nmt_element_tolerance_km == pytest.approx(0.2)
    assert training_cfg.goal_nmt_velocity_tolerance_km_s == pytest.approx(0.0008)
    assert len(training_cfg.required_phase_burns) == 1
    phase_burn = training_cfg.required_phase_burns[0]
    assert phase_burn.name == "Cross-track phase burn"
    assert phase_burn.axis == "cross_track"
    assert phase_burn.radial_abs_km == pytest.approx(1.2)
    assert phase_burn.radial_tolerance_km == pytest.approx(0.25)
    assert phase_burn.max_abs_intrack_km == pytest.approx(0.5)
    assert training_cfg.keepout_radius_km == pytest.approx(0.25)
    assert training_cfg.max_time_s == pytest.approx(10800.0)
    assert training_cfg.max_delta_v_m_s == pytest.approx(10.0)
    assert target_coes["ecc"] == pytest.approx(0.25)
    assert target_coes["true_anomaly_deg"] == pytest.approx(140.0)


def test_level_eight_is_elliptic_rendezvous() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "game"
        / "configs"
        / "game_training_rpo_09_elliptic_rendezvous.yaml"
    )
    config = SimulationConfig.from_yaml(config_path)
    training_cfg = RPOTrainingConfig.from_metadata(dict(config.scenario.metadata or {}))
    target_coes = config.scenario.objects["target"].initial_state["coes"]

    assert training_cfg.scenario_id == "rpo_09_elliptic_rendezvous"
    assert config.scenario.metadata["game"]["level_name"] == "Level 9 - Elliptical Rendezvous"
    assert _game_control_mode(config) == "ric_translation"
    assert _game_camera_mode(config) == "target_pair"
    assert _game_coast_prediction_model(config) == "tschauner_hempel"
    assert _game_target_centered_plot_planes(config) == ("RI",)
    assert _game_plot_prediction_in_zoom(config) is False
    assert _game_plot_prediction_zoom_max_span_km(config) is None
    assert training_cfg.goal_radius_km == pytest.approx(0.01)
    assert training_cfg.max_goal_speed_km_s == pytest.approx(0.0001)
    assert training_cfg.hard_speed_limit_radius_km == pytest.approx(0.025)
    assert training_cfg.hard_speed_limit_km_s == pytest.approx(0.0001)
    assert training_cfg.max_time_s == pytest.approx(14400.0)
    assert training_cfg.max_delta_v_m_s == pytest.approx(3.0)
    assert target_coes["a_km"] == pytest.approx(9000.0)
    assert target_coes["ecc"] == pytest.approx(0.25)
    assert target_coes["true_anomaly_deg"] == pytest.approx(100.0)


def test_active_top_bar_uses_configured_level_title() -> None:
    root = Path(__file__).resolve().parents[1] / "game" / "configs"
    config = SimulationConfig.from_yaml(root / "game_training_rpo_04_rendezvous.yaml")

    assert _game_level_title(config) == "Level 4 - Rendezvous"
    assert PygameRPODashboard._top_bar_label("active", _game_level_title(config)) == "LEVEL 4 - RENDEZVOUS"
    assert PygameRPODashboard._top_bar_label("active", "") == "LEVEL ACTIVE"


def test_arcade_attempt_config_uses_current_training_clock_for_duration() -> None:
    config_path = (
        Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_arcade_pursuit.yaml"
    )
    config = SimulationConfig.from_yaml(config_path)
    training_cfg = RPOTrainingConfig.from_metadata(dict(config.scenario.metadata or {}))

    next_round = replace(training_cfg, max_time_s=8500.25)
    attempt_cfg = _attempt_config_for_training_clock(config, next_round)

    assert config.scenario.simulator.duration_s == pytest.approx(12000.0)
    assert attempt_cfg.scenario.simulator.duration_s == pytest.approx(8501.0)


def test_pursuit_arcade_goal_range_tightens_each_round_to_floor() -> None:
    config_path = (
        Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_arcade_pursuit.yaml"
    )
    config = SimulationConfig.from_yaml(config_path)
    base_cfg = RPOTrainingConfig.from_metadata(dict(config.scenario.metadata or {}))

    round1 = _arcade_round_training_config(config, base_cfg, round_index=1, max_time_s=12000.0)
    round2 = _arcade_round_training_config(config, base_cfg, round_index=2, max_time_s=11900.0)
    round20 = _arcade_round_training_config(config, base_cfg, round_index=20, max_time_s=8000.0)
    round99 = _arcade_round_training_config(config, base_cfg, round_index=99, max_time_s=5000.0)

    assert round1.goal_range_km == pytest.approx(0.100)
    assert round2.goal_range_km == pytest.approx(0.095)
    assert round20.goal_range_km == pytest.approx(0.005)
    assert round99.goal_range_km == pytest.approx(0.005)
    assert round2.max_time_s == pytest.approx(11900.0)


def test_pursuit_arcade_boss_rounds_use_elliptical_target_and_random_anomaly() -> None:
    config_path = (
        Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_arcade_pursuit.yaml"
    )
    config = SimulationConfig.from_yaml(config_path)
    base_cfg = RPOTrainingConfig.from_metadata(dict(config.scenario.metadata or {}))

    round4 = _arcade_round_simulation_config(
        config,
        base_cfg,
        round_index=4,
        rng=_arcade_round_initial_state_rng(1234, 4),
    )
    round5 = _arcade_round_simulation_config(
        config,
        base_cfg,
        round_index=5,
        rng=_arcade_round_initial_state_rng(1234, 5),
    )
    round5_repeat = _arcade_round_simulation_config(
        config,
        base_cfg,
        round_index=5,
        rng=_arcade_round_initial_state_rng(1234, 5),
    )
    round10 = _arcade_round_simulation_config(
        config,
        base_cfg,
        round_index=10,
        rng=_arcade_round_initial_state_rng(1234, 10),
    )
    round25 = _arcade_round_simulation_config(
        config,
        base_cfg,
        round_index=25,
        rng=_arcade_round_initial_state_rng(1234, 25),
    )
    round30 = _arcade_round_simulation_config(
        config,
        base_cfg,
        round_index=30,
        rng=_arcade_round_initial_state_rng(1234, 30),
    )

    normal_coes = round4.scenario.objects["target"].initial_state["coes"]
    boss_coes = round5.scenario.objects["target"].initial_state["coes"]
    boss_repeat_coes = round5_repeat.scenario.objects["target"].initial_state["coes"]
    boss_10_coes = round10.scenario.objects["target"].initial_state["coes"]
    boss_25_coes = round25.scenario.objects["target"].initial_state["coes"]
    boss_30_coes = round30.scenario.objects["target"].initial_state["coes"]

    assert _arcade_round_is_boss(config, 4) is False
    assert _arcade_round_is_boss(config, 5) is True
    assert _arcade_round_is_boss(config, 10) is True
    assert normal_coes["ecc"] == pytest.approx(0.0)
    assert boss_coes["a_km"] == pytest.approx(9000.0)
    assert boss_coes["ecc"] == pytest.approx(0.05)
    assert boss_10_coes["ecc"] == pytest.approx(0.10)
    assert boss_25_coes["ecc"] == pytest.approx(0.20)
    assert boss_30_coes["ecc"] == pytest.approx(0.20)
    assert 0.0 <= float(boss_coes["true_anomaly_deg"]) < 360.0
    assert boss_coes["true_anomaly_deg"] == pytest.approx(boss_repeat_coes["true_anomaly_deg"])
    assert boss_coes["true_anomaly_deg"] != pytest.approx(boss_10_coes["true_anomaly_deg"])
    assert round5.scenario.objects["target"].reference_orbit["enabled"] is True
    assert _game_coast_prediction_model(round5) == "tschauner_hempel"


def test_pursuit_arcade_boss_rounds_keep_energy_matched_random_start() -> None:
    config_path = (
        Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_arcade_pursuit.yaml"
    )
    config = SimulationConfig.from_yaml(config_path)
    base_cfg = RPOTrainingConfig.from_metadata(dict(config.scenario.metadata or {}))

    round5 = _arcade_round_simulation_config(
        config,
        base_cfg,
        round_index=5,
        rng=_arcade_round_initial_state_rng(1234, 5),
    )

    state = np.array(round5.scenario.objects["chaser"].initial_state["relative_to_target_ric"]["state"], dtype=float)
    target_r, target_v = coes_mapping_to_rv_eci(round5.scenario.objects["target"].initial_state["coes"])
    chaser_state = ric_rect_state_to_eci(state, target_r, target_v)
    target_energy = 0.5 * float(np.dot(target_v, target_v)) - EARTH_MU_KM3_S2 / float(np.linalg.norm(target_r))
    chaser_energy = 0.5 * float(np.dot(chaser_state[3:], chaser_state[3:])) - EARTH_MU_KM3_S2 / float(
        np.linalg.norm(chaser_state[:3])
    )

    assert float(np.linalg.norm(state[:3])) >= 5.0
    assert chaser_energy == pytest.approx(target_energy, abs=1e-12)


def test_pursuit_arcade_boss_round_music_and_bonuses() -> None:
    config_path = (
        Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_arcade_pursuit.yaml"
    )
    config = SimulationConfig.from_yaml(config_path)
    training_cfg = RPOTrainingConfig.from_metadata(dict(config.scenario.metadata or {}))
    score = type("Score", (), {"level_passed": True, "achieved_time_s": 100.0, "approximate_delta_v_m_s": 2.5})()

    assert _arcade_round_music_track(config, 4) is None
    assert _arcade_round_music_track(config, 5) == "28_high_shred_boss_riff.wav"
    assert _arcade_round_coast_prediction_model(config, 5) == "tschauner_hempel"
    assert _arcade_round_score_multiplier(config, 5) == pytest.approx(2.0)
    assert _arcade_round_time_bonus_s(config, training_cfg, score, round_index=4) == pytest.approx(500.0)
    assert _arcade_round_time_bonus_s(config, training_cfg, score, round_index=5) == pytest.approx(5500.0)
    assert _arcade_round_weighted_score(
        training_cfg,
        score,
        difficulty="easy",
        round_index=5,
        arcade_config=config,
    ) == 124000


def test_web_pursuit_arcade_default_record_matches_desktop_scoring_policy() -> None:
    root = Path(__file__).resolve().parents[2]
    config = SimulationConfig.from_yaml(root / "sim" / "game" / "configs" / "game_training_rpo_arcade_pursuit.yaml")
    web_record = json.loads(
        (root / "web" / "rpo-trainer-preview" / "fixtures" / "default-challenge-record.json").read_text(
            encoding="utf-8"
        )
    )

    desktop_arcade = config.scenario.metadata["game"]["arcade"]
    web_arcade = web_record["config"]["arcade"]
    desktop_boss = desktop_arcade["boss"]
    web_boss = web_arcade["boss"]

    assert web_record["challenge_id"] == "rpo_arcade_pursuit"
    assert web_arcade["initial_time_s"] == pytest.approx(desktop_arcade["initial_time_s"])
    assert web_arcade["delta_v_bonus_time_per_m_s"] == pytest.approx(desktop_arcade["delta_v_bonus_time_per_m_s"])
    assert web_arcade["goal_range_step_km"] == pytest.approx(desktop_arcade["goal_range_step_km"])
    assert web_arcade["min_goal_range_km"] == pytest.approx(desktop_arcade["min_goal_range_km"])
    assert web_arcade["boss_round_interval"] == desktop_arcade["boss_round_interval"]
    assert web_boss["bonus_time_s"] == pytest.approx(desktop_boss["bonus_time_s"])
    assert web_boss["score_multiplier"] == pytest.approx(desktop_boss["score_multiplier"])
    assert web_boss["coast_prediction_model"] == desktop_boss["coast_prediction_model"]
    assert web_boss["music_track"] == desktop_boss["music_track"]
    assert web_boss["target_coes"] == desktop_boss["target_coes"]


def test_web_preview_assets_byte_match_source_game_assets() -> None:
    root = Path(__file__).resolve().parents[2]
    source_asset_dir = root / "sim" / "game" / "assets"
    web_sprite_dir = root / "web" / "rpo-trainer-preview" / "assets" / "sprites"

    if web_sprite_dir.exists():
        sprite_assets = sorted(web_sprite_dir.iterdir())
    else:
        sprite_assets = []
    for web_asset in sprite_assets:
        if not web_asset.is_file():
            continue
        source_asset = source_asset_dir / web_asset.name
        assert source_asset.is_file(), web_asset.name
        assert web_asset.read_bytes() == source_asset.read_bytes(), web_asset.name

    source_music_dir = root / "sim" / "game" / "music"
    web_music_dir = root / "web" / "rpo-trainer-preview" / "assets"
    for web_music in sorted(web_music_dir.glob("*.wav")):
        source_music = source_music_dir / web_music.name
        assert source_music.is_file(), web_music.name
        assert web_music.read_bytes() == source_music.read_bytes(), web_music.name


def test_pursuit_arcade_keeps_round_one_initial_state() -> None:
    config_path = (
        Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_arcade_pursuit.yaml"
    )
    config = SimulationConfig.from_yaml(config_path)
    base_cfg = RPOTrainingConfig.from_metadata(dict(config.scenario.metadata or {}))
    base_state = config.scenario.objects["chaser"].initial_state["relative_to_target_ric"]["state"]

    round1 = _arcade_round_simulation_config(
        config,
        base_cfg,
        round_index=1,
        rng=_arcade_round_initial_state_rng(1234, 1),
    )

    round1_state = round1.scenario.objects["chaser"].initial_state["relative_to_target_ric"]["state"]
    assert round1_state == base_state


def test_pursuit_arcade_randomizes_round_two_initial_state_with_energy_match() -> None:
    config_path = (
        Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_arcade_pursuit.yaml"
    )
    config = SimulationConfig.from_yaml(config_path)
    base_cfg = RPOTrainingConfig.from_metadata(dict(config.scenario.metadata or {}))

    round2 = _arcade_round_simulation_config(
        config,
        base_cfg,
        round_index=2,
        rng=_arcade_round_initial_state_rng(1234, 2),
    )
    round2_repeat = _arcade_round_simulation_config(
        config,
        base_cfg,
        round_index=2,
        rng=_arcade_round_initial_state_rng(1234, 2),
    )
    round3 = _arcade_round_simulation_config(
        config,
        base_cfg,
        round_index=3,
        rng=_arcade_round_initial_state_rng(1234, 3),
    )

    state = np.array(round2.scenario.objects["chaser"].initial_state["relative_to_target_ric"]["state"], dtype=float)
    repeat_state = np.array(
        round2_repeat.scenario.objects["chaser"].initial_state["relative_to_target_ric"]["state"], dtype=float
    )
    round3_state = np.array(
        round3.scenario.objects["chaser"].initial_state["relative_to_target_ric"]["state"], dtype=float
    )
    target_r, target_v = coes_mapping_to_rv_eci(config.scenario.objects["target"].initial_state["coes"])
    chaser_state = ric_rect_state_to_eci(state, target_r, target_v)

    target_energy = 0.5 * float(np.dot(target_v, target_v)) - EARTH_MU_KM3_S2 / float(np.linalg.norm(target_r))
    chaser_energy = 0.5 * float(np.dot(chaser_state[3:], chaser_state[3:])) - EARTH_MU_KM3_S2 / float(
        np.linalg.norm(chaser_state[:3])
    )

    assert -1.0 <= state[0] <= 1.0
    assert -10.0 <= state[1] <= 10.0
    assert -1.0 <= state[2] <= 1.0
    assert float(np.linalg.norm(state[:3])) >= 5.0
    assert state[3] == pytest.approx(0.0)
    assert -0.001 <= state[5] <= 0.001
    assert chaser_energy == pytest.approx(target_energy, abs=1e-12)
    assert np.allclose(state, repeat_state)
    assert not np.allclose(state, round3_state)


def test_dashboard_goal_overlay_syncs_with_arcade_round_range() -> None:
    config_path = (
        Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_arcade_pursuit.yaml"
    )
    config = SimulationConfig.from_yaml(config_path)
    base_cfg = RPOTrainingConfig.from_metadata(dict(config.scenario.metadata or {}))
    round2 = _arcade_round_training_config(config, base_cfg, round_index=2, max_time_s=11900.0)
    round2 = replace(
        round2,
        hard_speed_limit_radius_km=0.025,
        hard_speed_limit_km_s=0.00005,
        max_target_reference_range_km=1.0,
    )
    dashboard = type("Dashboard", (), {"goal_range_km": base_cfg.goal_range_km, "_frame_cache_dirty": False})()

    _sync_dashboard_training_config(dashboard, round2)

    assert dashboard.goal_range_km == pytest.approx(0.095)
    assert dashboard.hard_speed_limit_radius_km == pytest.approx(0.025)
    assert dashboard.hard_speed_limit_km_s == pytest.approx(0.00005)
    assert dashboard.max_target_reference_range_km == pytest.approx(1.0)
    assert dashboard.target_reference_object_id == "target_reference"
    assert dashboard._frame_cache_dirty is True


def test_pursuit_arcade_random_target_provider_sets_fixed_direction() -> None:
    config_path = (
        Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_arcade_pursuit.yaml"
    )
    config = SimulationConfig.from_yaml(config_path)
    first = _game_random_direction_defensive_target_provider(config, rng=np.random.default_rng(1))
    second = _game_random_direction_defensive_target_provider(config, rng=np.random.default_rng(2))

    assert first is not None
    assert second is not None
    assert first.fixed_direction_ric is not None
    assert second.fixed_direction_ric is not None
    assert np.linalg.norm(first.fixed_direction_ric) == pytest.approx(1.0)
    assert not np.allclose(first.fixed_direction_ric, second.fixed_direction_ric)


def test_pursuit_arcade_target_delta_v_budget_ramps_after_round_twenty() -> None:
    config_path = (
        Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_arcade_pursuit.yaml"
    )
    config = SimulationConfig.from_yaml(config_path)

    round1 = _game_random_direction_defensive_target_provider(config, round_index=1, rng=np.random.default_rng(1))
    round20 = _game_random_direction_defensive_target_provider(config, round_index=20, rng=np.random.default_rng(1))
    round21 = _game_random_direction_defensive_target_provider(config, round_index=21, rng=np.random.default_rng(1))
    round25 = _game_random_direction_defensive_target_provider(config, round_index=25, rng=np.random.default_rng(1))

    assert round1 is not None
    assert round20 is not None
    assert round21 is not None
    assert round25 is not None
    assert round1.max_delta_v_m_s == pytest.approx(0.10)
    assert round20.max_delta_v_m_s == pytest.approx(0.10)
    assert round21.max_delta_v_m_s == pytest.approx(0.11)
    assert round25.max_delta_v_m_s == pytest.approx(0.15)


def test_pursuit_arcade_round_rng_varies_by_session_seed() -> None:
    config_path = (
        Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_arcade_pursuit.yaml"
    )
    config = SimulationConfig.from_yaml(config_path)
    seeded_a = _game_random_direction_defensive_target_provider(config, rng=_arcade_round_rng(1001, 1))
    seeded_a_repeat = _game_random_direction_defensive_target_provider(config, rng=_arcade_round_rng(1001, 1))
    seeded_b = _game_random_direction_defensive_target_provider(config, rng=_arcade_round_rng(2002, 1))

    assert seeded_a is not None
    assert seeded_a_repeat is not None
    assert seeded_b is not None
    assert seeded_a.fixed_direction_ric is not None
    assert seeded_a_repeat.fixed_direction_ric is not None
    assert seeded_b.fixed_direction_ric is not None
    assert np.allclose(seeded_a.fixed_direction_ric, seeded_a_repeat.fixed_direction_ric)
    assert not np.allclose(seeded_a.fixed_direction_ric, seeded_b.fixed_direction_ric)


def test_level_ten_is_player_target_survival_against_ric_pd_chaser() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "game"
        / "configs"
        / "game_training_rpo_11_evasive_target_survival.yaml"
    )
    config = SimulationConfig.from_yaml(config_path)
    training_cfg = RPOTrainingConfig.from_metadata(dict(config.scenario.metadata or {}))
    chaser = config.scenario.objects["chaser"]
    target = config.scenario.objects["target"]
    chaser_initial = chaser.initial_state["relative_to_target_ric"]
    source_config = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / "configs" / "ric_pd_10km_experiment.yaml").read_text(
            encoding="utf-8"
        )
    )
    source_control = source_config["objects"]["chaser"]["orbit_control"]

    assert _game_controlled_object_id(config) == "target"
    assert _game_control_mode(config) == "ric_translation"
    assert _game_camera_mode(config) == "rule_toggle_pair"
    assert _game_camera_rule_mode(config) == "current_pair"
    assert _game_camera_rule_toggle_enabled(config) is True
    assert _game_show_target_hcw_path(config) is True
    assert _game_speed_multiplier_options(config) == pytest.approx((1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0))
    assert np.array(_game_speed_dt_schedule(config), dtype=float) == pytest.approx(
        np.array(((1.0, 0.1), (10.0, 0.5), (25.0, 1.0), (50.0, 5.0), (100.0, 10.0)), dtype=float)
    )
    assert _game_tick_dt_s(config, 1.0) == pytest.approx(0.1)
    assert _game_tick_dt_s(config, 5.0) == pytest.approx(0.1)
    assert _game_tick_dt_s(config, 10.0) == pytest.approx(0.5)
    assert _game_tick_dt_s(config, 25.0) == pytest.approx(1.0)
    assert _game_tick_dt_s(config, 100.0) == pytest.approx(1.0)
    assert _game_active_tick_dt_s(config, 100.0, maneuver_active=False) == pytest.approx(10.0)
    assert _game_max_autonomy_step_s(config) == pytest.approx(1.0)
    assert training_cfg.scenario_id == "rpo_11_evasive_target_survival"
    assert training_cfg.survival_goal is True
    assert training_cfg.keepout_radius_km == pytest.approx(0.1)
    assert training_cfg.goal_range_km is None
    assert training_cfg.max_goal_speed_km_s is None
    assert training_cfg.max_time_s == pytest.approx(9000.0)
    assert training_cfg.target_reference_object_id == "target_reference"
    assert training_cfg.max_target_reference_range_km == pytest.approx(1.0)
    assert training_cfg.max_target_delta_v_m_s == pytest.approx(1.0)
    assert training_cfg.max_delta_v_m_s == pytest.approx(25.0)
    assert config.scenario.simulator.duration_s == pytest.approx(9000.0)
    assert training_cfg.fail_on_delta_v_budget is False
    assert _game_coast_chaser_after_delta_v_budget(training_cfg) is True
    assert chaser.orbit_control.module == "sim.control.orbit.ric_pd"
    assert chaser.orbit_control.class_name == "RICPDTransferController"
    assert chaser.orbit_control.module == source_control["module"]
    assert chaser.orbit_control.class_name == source_control["class_name"]
    assert chaser.orbit_control.params == source_control["params"]
    assert chaser.attitude_control.module == "sim.control.attitude.zero_torque"
    assert chaser.attitude_control.class_name == "ZeroTorqueController"
    assert chaser.knowledge["estimation"]["type"] == "measured_state"
    assert target.knowledge["estimation"]["type"] == "measured_state"
    assert chaser.mission_execution is None
    assert config.scenario.simulator.dynamics["attitude"] == {"enabled": False}
    assert chaser_initial["frame"] == "curv"
    assert np.allclose(chaser_initial["state"], np.array([0.0, -5.0, 0.0, 0.0, 0.0, 0.001]))


def test_level_ten_player_controls_target_while_chaser_controller_runs() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "game"
        / "configs"
        / "game_training_rpo_11_evasive_target_survival.yaml"
    )
    config = SimulationConfig.from_yaml(config_path)
    training_cfg = RPOTrainingConfig.from_metadata(dict(config.scenario.metadata or {}))
    state = KeyboardCommandState(yaw=1.0)

    session, _, snap0 = _start_game_attempt(
        config,
        command_state=state,
        training_cfg=training_cfg,
        controlled_object_id=_game_controlled_object_id(config),
        attitude_rate_deg_s=45.0,
        control_mode=_game_control_mode(config),
        ric_reference_object_id=_game_ric_reference_object_id(config, training_cfg.target_object_id),
    )
    snap1 = session.step()

    assert "target" in session._external_intent_providers
    assert np.linalg.norm(snap1.applied_thrust["target"]) > 0.0
    assert np.linalg.norm(snap1.applied_thrust["chaser"]) > 0.0
    assert not np.allclose(snap1.truth["target"][:6], snap0.truth["target"][:6])


def test_delta_v_limited_orbit_controller_coasts_after_budget() -> None:
    class ConstantController:
        def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
            return Command(
                thrust_eci_km_s2=np.array([0.001, 0.0, 0.0], dtype=float),
                torque_body_nm=np.zeros(3, dtype=float),
                mode_flags={"mode": "constant"},
            )

    controller = _DeltaVLimitedOrbitController(
        base=ConstantController(),
        max_delta_v_m_s=1.5,
        dt_s=1.0,
    )
    belief = StateBelief(state=np.zeros(6, dtype=float), covariance=np.eye(6), last_update_t_s=0.0)

    first = controller.act(belief, 0.0, 1.0)
    second = controller.act(belief, 1.0, 1.0)
    third = controller.act(belief, 2.0, 1.0)

    assert np.allclose(first.thrust_eci_km_s2, np.array([0.001, 0.0, 0.0]))
    assert np.allclose(second.thrust_eci_km_s2, np.array([0.0005, 0.0, 0.0]))
    assert np.allclose(third.thrust_eci_km_s2, np.zeros(3))
    assert controller.used_delta_v_m_s == pytest.approx(1.5)
    assert third.mode_flags["delta_v_limit_exhausted"] is True


def test_terminal_mission_state_keeps_game_loop_open_after_session_done() -> None:
    passed = type("Score", (), {"level_passed": True, "level_failed": False})()
    failed = type("Score", (), {"level_passed": False, "level_failed": True})()
    active = type("Score", (), {"level_passed": False, "level_failed": False})()

    assert _game_loop_should_exit(session_done=True, score=passed) is False
    assert _game_loop_should_exit(session_done=True, score=failed) is False
    assert _game_loop_should_exit(session_done=True, score=active) is True


def test_result_music_paths_follow_terminal_mission_state() -> None:
    passed = type("Score", (), {"level_passed": True, "level_failed": False})()
    failed = type("Score", (), {"level_passed": False, "level_failed": True})()
    active = type("Score", (), {"level_passed": False, "level_failed": False})()

    assert _result_music_path(passed) == MISSION_SUCCESS_MUSIC_PATH
    assert _result_music_path(failed) == MISSION_FAILURE_MUSIC_PATH
    assert _result_music_path(active) is None
    assert MISSION_SUCCESS_MUSIC_PATH.name == "05_final_burn_victory_loop.wav"
    assert MISSION_FAILURE_MUSIC_PATH.name == "15_mission_failed_lament_credits.wav"


def test_level_music_maps_rendezvous_vector_to_level_2() -> None:
    tutorial = RPOTrainingConfig(enabled=True, scenario_id="rpo_00_tutorial")
    level1 = RPOTrainingConfig(enabled=True, scenario_id="rpo_01_coast_relative_motion")
    level2 = RPOTrainingConfig(enabled=True, scenario_id="rpo_02_vbar_approach")
    level3 = RPOTrainingConfig(enabled=True, scenario_id="rpo_03_rbar_approach")
    level4 = RPOTrainingConfig(enabled=True, scenario_id="rpo_04_rendezvous")
    level5 = RPOTrainingConfig(enabled=True, scenario_id="rpo_05_passive_cross_track_approach")
    level6 = RPOTrainingConfig(enabled=True, scenario_id="rpo_06_sun_angle_inspection")
    level7 = RPOTrainingConfig(enabled=True, scenario_id="rpo_07_elliptic_burn_then_approach")
    level8 = RPOTrainingConfig(enabled=True, scenario_id="rpo_08_elliptic_nmc")
    level9 = RPOTrainingConfig(enabled=True, scenario_id="rpo_09_elliptic_rendezvous")
    level10 = RPOTrainingConfig(enabled=True, scenario_id="rpo_10_defensive_target_demo")
    level11 = RPOTrainingConfig(enabled=True, scenario_id="rpo_11_evasive_target_survival")
    cislunar = RPOTrainingConfig(enabled=True, scenario_id="rpo_bonus_cislunar_rendezvous")
    arcade = RPOTrainingConfig(enabled=True, scenario_id="rpo_arcade_pursuit")
    unmapped = RPOTrainingConfig(enabled=True, scenario_id="rpo_11_unmapped")

    assert _level_music_path(tutorial) == LEVEL_MUSIC_PATHS["rpo_00_tutorial"]
    assert _level_music_path(tutorial).name == "10_training_grid_sunrise.wav"
    assert _level_music_path(level1) == LEVEL_MUSIC_PATHS["rpo_01_coast_relative_motion"]
    assert _level_music_path(level1).name == "07_starfield_attract_mode.wav"
    assert _level_music_path(level2) == LEVEL_MUSIC_PATHS["rpo_02_vbar_approach"]
    assert _level_music_path(level2).name == "02_rendezvous_vector.wav"
    assert _level_music_path(level3) == LEVEL_MUSIC_PATHS["rpo_03_rbar_approach"]
    assert _level_music_path(level3).name == "18_keepout_zone_accelerando.wav"
    assert _level_music_path(level4) == LEVEL_MUSIC_PATHS["rpo_04_rendezvous"]
    assert _level_music_path(level4).name == "06_casting_the_orbit_line.wav"
    assert _level_music_path(level5) == LEVEL_MUSIC_PATHS["rpo_05_passive_cross_track_approach"]
    assert _level_music_path(level5).name == "19_cross_track_ghost_orbit.wav"
    assert _level_music_path(level6) == LEVEL_MUSIC_PATHS["rpo_06_sun_angle_inspection"]
    assert _level_music_path(level6).name == "33_amber_terminator_demo.wav"
    assert _level_music_path(level7) == LEVEL_MUSIC_PATHS["rpo_07_elliptic_burn_then_approach"]
    assert _level_music_path(level7).name == "08_silent_running_radar.wav"
    assert _level_music_path(level8) == LEVEL_MUSIC_PATHS["rpo_08_elliptic_nmc"]
    assert _level_music_path(level8).name == "04_docking_bay_neon.wav"
    assert _level_music_path(level9) == LEVEL_MUSIC_PATHS["rpo_09_elliptic_rendezvous"]
    assert _level_music_path(level9).name == "23_elliptic_final_burn_cinematic.wav"
    assert _level_music_path(level10) == LEVEL_MUSIC_PATHS["rpo_10_defensive_target_demo"]
    assert _level_music_path(level10).name == "17_orbital_boss_metal.wav"
    assert _level_music_path(level11) == LEVEL_MUSIC_PATHS["rpo_11_evasive_target_survival"]
    assert _level_music_path(level11).name == "09_defender_boss_vector.wav"
    assert _level_music_path(cislunar) == LEVEL_MUSIC_PATHS["rpo_bonus_cislunar_rendezvous"]
    assert _level_music_path(cislunar).name == "30_far_side_navigation_demo.wav"
    assert _level_music_path(arcade) == LEVEL_MUSIC_PATHS["rpo_arcade_pursuit"]
    assert _level_music_path(arcade).name == "21_pursuit_arcade_overdrive_no_siren_demo.wav"
    assert _level_music_path(unmapped) is None
    assert ARCADE_ROUND_CLEAR_SOUND_PATH.name == "22_arcade_round_clear_flyover.wav"
    assert ARCADE_ROUND_CLEAR_SOUND_PATH.parent.name == "music"


def test_play_game_sound_effect_uses_pygame_sound_channel() -> None:
    calls: list[tuple[str, object]] = []

    class FakeSound:
        def __init__(self, path: str) -> None:
            calls.append(("load", path))

        def set_volume(self, volume: float) -> None:
            calls.append(("volume", volume))

        def play(self) -> None:
            calls.append(("play", None))

    class FakeMixer:
        @staticmethod
        def get_init() -> bool:
            return True

        Sound = FakeSound

    class FakePygame:
        mixer = FakeMixer()

    assert _play_game_sound_effect(FakePygame(), ARCADE_ROUND_CLEAR_SOUND_PATH, volume=1.5) is True
    assert calls[0] == ("load", str(ARCADE_ROUND_CLEAR_SOUND_PATH))
    assert calls[1] == ("volume", 1.0)
    assert calls[2] == ("play", None)


def test_arcade_score_uses_remaining_time_delta_v_and_difficulty() -> None:
    cfg = RPOTrainingConfig(enabled=True, max_time_s=120.0, max_delta_v_m_s=3.0)
    score = type(
        "Score",
        (),
        {
            "level_passed": True,
            "achieved_time_s": 20.0,
            "elapsed_s": 30.0,
            "approximate_delta_v_m_s": 1.25,
        },
    )()

    assert _difficulty_score_multiplier("easy") == 1
    assert _difficulty_score_multiplier("medium") == 2
    assert _difficulty_score_multiplier("hard") == 3
    assert _difficulty_score_multiplier("extreme") == 4
    assert _difficulty_score_multiplier("expert") == 4
    assert _arcade_score(cfg, score, difficulty="hard") == 5550
    assert _arcade_round_weighted_score(cfg, score, difficulty="hard", round_index=3) == 16650
    assert _score_time_used_s(score) == pytest.approx(20.0)


def test_arcade_round_time_bonus_adds_delta_v_remaining() -> None:
    config_path = (
        Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_arcade_pursuit.yaml"
    )
    config = SimulationConfig.from_yaml(config_path)
    training_cfg = RPOTrainingConfig(enabled=True, max_delta_v_m_s=3.0)
    score = type("Score", (), {"approximate_delta_v_m_s": 2.5})()

    assert _arcade_round_time_bonus_s(config, training_cfg, score) == pytest.approx(500.0)
    assert _arcade_round_time_bonus_s(config, training_cfg, score, round_index=5) == pytest.approx(5500.0)


def test_arcade_round_briefing_lines_show_transition_summary() -> None:
    lines = _arcade_round_briefing_lines(
        cleared_round_index=2,
        next_round_index=3,
        round_score=4200,
        total_score=7000,
        time_used_s=1234.4,
        bonus_time_s=850.0,
        next_time_budget_s=5000.6,
        next_goal_range_km=0.095,
    )

    assert lines[0] == "Round 2 Cleared"
    assert "Round score: 4,200" in lines[1]
    assert "Total score: 7,000" in lines[1]
    assert "Bonus awarded: 850 s" in lines[2]
    assert "Round 3 starts with 5001 s" in lines[3]
    assert "Next pursuit target: close within 95.00 m." in lines[4]
    assert "Fuel resets" in lines[5]


def test_arcade_score_is_zero_until_success() -> None:
    cfg = RPOTrainingConfig(enabled=True, max_time_s=120.0, max_delta_v_m_s=3.0)
    score = type(
        "Score",
        (),
        {
            "level_passed": False,
            "achieved_time_s": None,
            "elapsed_s": 30.0,
            "approximate_delta_v_m_s": 1.25,
        },
    )()

    assert _arcade_score(cfg, score, difficulty="extreme") == 0


def test_game_ui_formatters_use_engineering_units_and_sig_figs() -> None:
    assert format_distance_km(0.9999) == "999.9 m"
    assert format_distance_km(1.001) == "1.001 km"
    assert format_distance_km(0.000025) == "25.00 mm"
    assert format_speed_m_s(0.9999) == "999.9 mm/s"
    assert format_speed_km_s(0.0001) == "100.0 mm/s"
    assert format_speed_km_s(0.001001) == "1.001 m/s"


def test_mission_metrics_show_delta_v_remaining_with_engineering_units() -> None:
    cfg = RPOTrainingConfig(enabled=True, max_delta_v_m_s=8.0, max_target_delta_v_m_s=1.0)
    score = type(
        "Score",
        (),
        {
            "approximate_delta_v_m_s": 1.234,
            "target_delta_v_m_s": 0.123,
        },
    )()

    metrics = _mission_metrics(cfg, score)

    assert "OK Chaser dV 6.766 m/s" in metrics
    assert "OK Target dV 877.0 mm/s" in metrics


def test_manual_game_provider_commands_attitude_target_and_thrust(tmp_path: Path) -> None:
    state = KeyboardCommandState(roll=1.0, firing=True)
    provider = ManualGameCommandProvider(
        command_state=state,
        max_accel_km_s2=2.0e-5,
        attitude_rate_deg_s=30.0,
        controlled_object_id="chaser",
    )
    session = SimulationSession.from_config(SimulationConfig.from_dict(_game_config(tmp_path)))
    session.set_external_intent_provider("chaser", provider)
    snap0 = session.reset()
    assert snap0 is not None

    snap1 = session.step()

    assert np.linalg.norm(snap1.applied_thrust["chaser"]) > 0.0
    assert provider.desired_attitude_quat_bn is not None
    assert not np.allclose(provider.desired_attitude_quat_bn, snap0.truth["chaser"][6:10])
    assert np.linalg.norm(snap1.applied_torque["chaser"]) > 0.0


def test_external_intent_provider_can_be_removed(tmp_path: Path) -> None:
    state = KeyboardCommandState(firing=True)
    provider = ManualGameCommandProvider(command_state=state, max_accel_km_s2=2.0e-5)
    session = SimulationSession.from_config(SimulationConfig.from_dict(_game_config(tmp_path)))
    session.set_external_intent_provider("chaser", provider)
    snap0 = session.reset()
    assert snap0 is not None

    session.set_external_intent_provider("chaser", None)
    snap1 = session.step()

    assert np.allclose(snap1.applied_thrust["chaser"], np.zeros(3), atol=1e-15)


def test_ric_translation_provider_commands_direct_ric_thrust() -> None:
    state = KeyboardCommandState(pitch=1.0, yaw=0.0, roll=0.0)
    provider = ManualGameCommandProvider(
        command_state=state,
        max_accel_km_s2=2.0e-5,
        control_mode="ric_translation",
        reference_object_id="target",
    )
    target = StateTruth(
        position_eci_km=np.array([7000.0, 0.0, 0.0], dtype=float),
        velocity_eci_km_s=np.array([0.0, 7.5, 0.0], dtype=float),
        attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        angular_rate_body_rad_s=np.zeros(3, dtype=float),
        mass_kg=100.0,
        t_s=0.0,
    )

    out = provider(
        truth=target,
        t_s=0.0,
        dt_s=1.0,
        object_id="chaser",
        own_knowledge={"target": _knowledge_from_state6(np.hstack((target.position_eci_km, target.velocity_eci_km_s)))},
    )

    assert out["command_mode_flags"]["player_control_mode"] == "ric_translation"
    assert np.allclose(out["thrust_eci_km_s2"], np.array([2.0e-5, 0.0, 0.0]), atol=1e-12)


def test_operator_burn_plan_parser_and_validation() -> None:
    plan = parse_operator_burn_plan("T= 50 s, 2.0 m/s R, 1.0 m/s I, 0.2 m/s C")

    assert len(plan.burns) == 1
    assert plan.burns[0].time_s == pytest.approx(50.0)
    assert plan.burns[0].delta_v_ric_m_s == pytest.approx((2.0, 1.0, 0.2))
    assert validate_operator_burn_plan(plan, total_delta_v_budget_m_s=5.0) == ()

    too_large = parse_operator_burn_plan("T= 1 s, 6.0 m/s R")

    assert validate_operator_burn_plan(too_large, total_delta_v_budget_m_s=20.0) == (
        "Burn 1: delta-v exceeds 5.0 m/s.",
    )

    too_close = parse_operator_burn_plan("T=1 s, 0.5 m/s R\nT=10.5 s, 0.5 m/s I")

    assert validate_operator_burn_plan(too_close, total_delta_v_budget_m_s=20.0) == (
        "Burn 2: time must be at least 10 seconds after Burn 1.",
    )

    exactly_spaced = parse_operator_burn_plan("T=1 s, 0.5 m/s R\nT=11 s, 0.5 m/s I")

    assert validate_operator_burn_plan(exactly_spaced, total_delta_v_budget_m_s=20.0) == ()


def test_operator_burn_rows_build_plan_with_blank_components(tmp_path: Path) -> None:
    option = GameScenarioOption(
        path=tmp_path / "game_training_rpo_01_demo.yaml",
        scenario_id="rpo_01_demo",
        title="Demo",
        description="",
        learning_goal="",
        player_brief="",
        pass_criteria=(),
        instructor_notes=(),
        difficulty="easy",
        time_budget_s=100.0,
        delta_v_budget_m_s=5.0,
        goal_speed_km_s=None,
        target_delta_v_budget_m_s=None,
        completed_difficulties=(),
        high_score=0,
        level_number=1,
    )

    plan, errors = _operator_plan_from_rows([["50", "2", "", "0.2"], ["", "", "", ""]], option=option)

    assert errors == ()
    assert len(plan.burns) == 1
    assert plan.burns[0].time_s == pytest.approx(50.0)
    assert plan.burns[0].delta_v_ric_m_s == pytest.approx((2.0, 0.0, 0.2))


def test_operator_burn_rows_use_target_budget_when_target_controlled(tmp_path: Path) -> None:
    option = GameScenarioOption(
        path=tmp_path / "game_training_rpo_11_demo.yaml",
        scenario_id="rpo_11_demo",
        title="Demo",
        description="",
        learning_goal="",
        player_brief="",
        pass_criteria=(),
        instructor_notes=(),
        difficulty="easy",
        time_budget_s=100.0,
        delta_v_budget_m_s=25.0,
        goal_speed_km_s=None,
        target_delta_v_budget_m_s=1.0,
        completed_difficulties=(),
        high_score=0,
        level_number=10,
        controlled_object_id="target",
        target_object_id="target",
    )

    _plan, errors = _operator_plan_from_rows([["50", "2", "", "0"]], option=option)

    assert errors == ("Plan total delta-v exceeds 1.0 m/s budget.",)


def test_operator_burn_rows_prefill_from_existing_plan() -> None:
    plan = OperatorBurnPlan(
        burns=(
            OperatorBurn(time_s=50.0, delta_v_ric_m_s=(2.0, 1.0, 0.2)),
            OperatorBurn(time_s=120.0, delta_v_ric_m_s=(-0.5, 0.0, 0.0)),
        )
    )

    assert _operator_rows_from_plan(plan) == [["50", "2", "1", "0.2"], ["120", "-0.5", "0", "0"]]


def test_operator_tutorial_scripts_six_locked_burn_demos() -> None:
    config_path = (
        Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_00_tutorial.yaml"
    )
    training_cfg = RPOTrainingConfig.from_metadata(dict(SimulationConfig.from_yaml(config_path).scenario.metadata or {}))
    stages = _operator_tutorial_stages()

    assert _operator_tutorial_enabled("operator", training_cfg)
    assert not _operator_tutorial_enabled("pilot", training_cfg)
    assert [stage.display_label for stage in stages] == [
        "+I Burn",
        "-I Burn",
        "+R Burn",
        "-R Burn",
        "+C Burn",
        "-C Burn",
    ]
    assert _operator_tutorial_demo_title(OperatorTutorialRuntime(stage_index=2)) == "Demo 3/6: +R Burn"

    expected = (
        (1, 1),
        (1, -1),
        (0, 1),
        (0, -1),
        (2, 1),
        (2, -1),
    )
    for stage, (axis_index, sign) in zip(stages, expected, strict=True):
        assert len(stage.plan.burns) == 1
        burn = stage.plan.burns[0]
        expected_delta_v = np.zeros(3, dtype=float)
        expected_delta_v[axis_index] = sign * OPERATOR_TUTORIAL_BURN_DELTA_V_M_S
        assert burn.time_s == pytest.approx(OPERATOR_TUTORIAL_BURN_TIME_S)
        assert burn.delta_v_ric_m_s == pytest.approx(tuple(expected_delta_v))


def test_operator_tutorial_uses_no_helper_trajectory() -> None:
    dashboard = type(
        "Dashboard",
        (),
        {
            "tutorial_target_path_ric": np.ones((3, 6), dtype=float),
            "_frame_cache_dirty": False,
        },
    )()

    _clear_dashboard_tutorial_path(dashboard)

    assert dashboard.tutorial_target_path_ric.shape == (0, 6)
    assert dashboard._frame_cache_dirty is True


def test_operator_plan_layout_uses_game_plot_positions_and_scrollable_burn_table() -> None:
    left, right = _operator_game_plot_panel_rects(1280, 720)
    table = _operator_burn_table_rect(1280, 720)

    assert (left.x, left.y, left.width, left.height) == (36, 88, 586, 464)
    assert (right.x, right.y, right.width, right.height) == (658, 88, 586, 464)
    assert table[0] == left.x
    assert table[1] > left.bottom
    assert table[2] > 800
    assert _operator_table_visible_rows(table) == 2
    assert _operator_table_visible_rows(table) < 6
    assert _operator_scroll_for_active_row(5, 0, row_count=8, table_rect=table) > 0


def test_operator_objectives_button_and_overlay_cover_plot_area() -> None:
    button = _operator_objectives_button_rect(1280, 720)
    equation_button = _operator_equation_sheet_button_rect(1280, 720)
    overlay = _operator_objectives_overlay_rect(1280, 720)
    left, right = _operator_game_plot_panel_rects(1280, 720)
    table = _operator_burn_table_rect(1280, 720)

    assert button[0] + button[2] == 1228
    assert button[1] == 34
    assert equation_button == (972, 604, 256, 36)
    assert overlay[0] > left.x
    assert overlay[1] > left.y
    assert overlay[0] + overlay[2] < right.right
    assert overlay[1] + overlay[3] < table[1]


def test_operator_objectives_numeric_targets_include_level_thresholds() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "game" / "configs"
    option = next(
        option for option in discover_game_scenarios_for_mode(config_dir, mode="operator")
        if option.scenario_id == "rpo_01_coast_relative_motion"
    )
    plot_context = _operator_plot_context(option.path)

    targets = _operator_objective_numeric_targets(option, plot_context.training_config)

    assert "Desired radial amplitude: 1.500 km" in targets
    assert "Desired cross-track amplitude: 1.500 km" in targets
    assert "NMT amplitude tolerance: 150.0 m" in targets
    assert "Chaser delta-v budget: 8.000 m/s" in targets


def test_operator_objectives_scroll_clamps_to_overflow_content() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "game" / "configs"
    option = next(
        option for option in discover_game_scenarios_for_mode(config_dir, mode="operator")
        if option.scenario_id == "rpo_01_coast_relative_motion"
    )
    plot_context = _operator_plot_context(option.path)
    overlay = _operator_objectives_overlay_rect(1280, 720)
    content_rect = _operator_objectives_content_rect(overlay)
    content_height = _operator_objectives_content_height(
        option,
        plot_context.training_config,
        font=_FixedWidthFont(),
        width_px=content_rect[2],
    )

    assert content_height > content_rect[3]
    assert _clamp_operator_objectives_scroll_px(
        9999,
        content_height=content_height,
        viewport_height=content_rect[3],
    ) == content_height - content_rect[3]


def test_operator_plan_delete_row_hit_testing_is_table_scoped() -> None:
    rect = type("Rect", (), {})
    first = rect()
    first.x, first.y, first.w, first.h = 900, 600, 22, 22
    second = rect()
    second.x, second.y, second.w, second.h = 900, 636, 22, 22

    assert _operator_delete_row_at_pos((905, 605), [first, second], table_rect=(36, 548, 912, 172)) == 0
    assert _operator_delete_row_at_pos((905, 641), [first, second], table_rect=(36, 548, 912, 172)) == 1
    assert _operator_delete_row_at_pos((905, 605), [first, second], table_rect=(36, 650, 912, 120)) is None


def test_operator_initial_relative_state_reads_level_yaml(tmp_path: Path) -> None:
    path = tmp_path / "game_training_rpo_01_demo.yaml"
    cfg = _game_config(tmp_path)
    cfg["metadata"]["game"]["training"] = {"scenario_id": "rpo_01_demo", "chaser_object_id": "chaser"}
    cfg["objects"]["chaser"]["initial_state"]["relative_to_target_ric"] = {
        "frame": "rect",
        "state": [-7.5, -15.0, 1.0, 0.0, 0.011, 0.0],
    }
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    assert _operator_initial_relative_ric_state(path) == pytest.approx((-7.5, -15.0, 1.0))


def test_operator_plot_context_uses_initial_coast_not_goal_nmt_for_level_one() -> None:
    path = Path("sim/game/configs/game_training_rpo_01_coast_relative_motion.yaml")

    context = _operator_plot_context(path)
    coast = np.array(context.initial_coast_ric_km_s, dtype=float).reshape(-1, 6)

    assert coast.shape[0] > 10
    assert coast[:, 0] == pytest.approx(0.0)
    assert coast[:, 1] == pytest.approx(-3.0)
    assert coast[:, 2] == pytest.approx(0.0)


def test_operator_planned_trajectory_applies_burns_and_extends_one_orbit() -> None:
    path = Path("sim/game/configs/game_training_rpo_01_coast_relative_motion.yaml")
    context = _operator_plot_context(path)
    assert context.mean_motion_rad_s is not None
    plan = OperatorBurnPlan(
        burns=(
            OperatorBurn(time_s=0.0, delta_v_ric_m_s=(0.0, 0.1, 0.0)),
            OperatorBurn(time_s=2000.0, delta_v_ric_m_s=(0.0, -0.1, 0.0)),
        )
    )

    trajectory, markers = _operator_planned_trajectory(context, plan)
    period_s = 2.0 * np.pi / float(context.mean_motion_rad_s)
    initial = np.asarray(context.initial_relative_ric_km_s, dtype=float)

    assert trajectory.shape[0] > 200
    assert markers.shape == (2, 6)
    assert trajectory[0] == pytest.approx(initial)
    assert markers[0, 3:6] == pytest.approx(initial[3:6] + np.array([0.0, 0.0001, 0.0]))
    expected_second_pre_burn = _cw_coast_states(markers[0], np.array([2000.0]), float(context.mean_motion_rad_s))[0]
    assert markers[1, :3] == pytest.approx(expected_second_pre_burn[:3], abs=1.0e-9)
    assert markers[1, 3:6] == pytest.approx(expected_second_pre_burn[3:6] + np.array([0.0, -0.0001, 0.0]))
    expected_final = _cw_coast_states(
        markers[-1],
        np.array([period_s]),
        float(context.mean_motion_rad_s),
    )[0]
    assert trajectory[-1] == pytest.approx(expected_final, abs=1.0e-9)


def test_operator_burn_velocity_vector_uses_post_burn_velocity_direction() -> None:
    path = Path("sim/game/configs/game_training_rpo_01_coast_relative_motion.yaml")
    context = _operator_plot_context(path)
    plan = OperatorBurnPlan(burns=(OperatorBurn(time_s=0.0, delta_v_ric_m_s=(0.0, 0.1, 0.0)),))

    _trajectory, markers = _operator_planned_trajectory(context, plan)
    endpoint = _operator_velocity_vector_endpoint(
        (100, 100),
        markers[0, 3:6],
        x_axis=1,
        y_axis=0,
        length_px=30.0,
    )

    assert markers[0, 3:6] == pytest.approx(np.array(context.initial_relative_ric_km_s[3:6]) + np.array([0.0, 0.0001, 0.0]))
    assert endpoint == (130, 100)


def test_operator_trajectory_probe_click_selects_and_clears_state() -> None:
    path = Path("sim/game/configs/game_training_rpo_01_coast_relative_motion.yaml")
    context = _operator_plot_context(path)
    plan = OperatorBurnPlan(burns=(OperatorBurn(time_s=0.0, delta_v_ric_m_s=(0.0, 0.1, 0.0)),))
    trajectory, _markers = _operator_planned_trajectory(context, plan)
    transform = {
        "plot": (0, 0, 500, 400),
        "camera_center": (0.0, 0.0, 0.0),
        "scale_x": 50.0,
        "scale_y": 50.0,
        "x_display_sign": 1.0,
        "y_display_sign": 1.0,
    }
    dashboard = type(
        "Dashboard",
        (),
        {"_frame_cache": {"plot_transforms": {(1, 0): transform}}},
    )()
    object.__setattr__(context, "_preview_dashboard", dashboard)
    target_state = np.asarray(trajectory[12], dtype=float).reshape(6)
    click_pos = _operator_plot_transform_to_px(transform, target_state[:3], x_axis=1, y_axis=0)
    assert click_pos is not None

    handled, selected_state, selected_time_s = _operator_trajectory_probe_from_click(context, plan, click_pos)

    assert handled is True
    assert selected_state is not None
    selected_px = _operator_plot_transform_to_px(transform, selected_state[:3], x_axis=1, y_axis=0)
    assert selected_px is not None
    assert np.linalg.norm(np.array(selected_px, dtype=float) - np.array(click_pos, dtype=float)) <= 10.0
    assert any(np.allclose(selected_state, row) for row in trajectory)
    assert selected_time_s is not None
    assert _operator_probe_time_label(selected_time_s).startswith("T=")
    assert _operator_probe_time_label(selected_time_s).endswith("s")

    selected_probe = OperatorTrajectoryProbe(
        state_ric_km_s=tuple(float(value) for value in selected_state),
        time_s=float(selected_time_s),
        plan_key=("test",),
    )
    handled, selected_state, selected_time_s = _operator_trajectory_probe_from_click(
        context,
        plan,
        click_pos,
        selected_probe=selected_probe,
    )

    assert handled is True
    assert selected_state is None
    assert selected_time_s is None


def test_operator_planned_trajectory_uses_ya_for_elliptic_levels(monkeypatch: pytest.MonkeyPatch) -> None:
    path = Path("sim/game/configs/game_training_rpo_07_elliptic_burn_then_approach.yaml")
    context = _operator_plot_context(path)
    assert _coast_prediction_model_key(context.coast_prediction_model) == "tschauner_hempel"
    assert context.reference_state_eci_km_s is not None
    calls: list[float] = []
    original = game_launcher.ya_closed_form_transition_matrix

    def spy_ya(dt_s: float, chief_start_eci_km_s: np.ndarray, chief_end_eci_km_s: np.ndarray, **kwargs: Any) -> Any:
        calls.append(float(dt_s))
        return original(dt_s, chief_start_eci_km_s, chief_end_eci_km_s, **kwargs)

    monkeypatch.setattr(game_launcher, "ya_closed_form_transition_matrix", spy_ya)
    plan = OperatorBurnPlan(burns=(OperatorBurn(time_s=0.0, delta_v_ric_m_s=(0.0, 0.1, 0.0)),))

    trajectory, markers = _operator_planned_trajectory(context, plan)

    assert calls
    assert max(calls) > 1000.0
    assert trajectory.shape[0] > 100
    assert markers.shape == (1, 6)


def test_operator_vbar_script_preview_uses_live_level_camera_and_forbidden_regions() -> None:
    path = Path("sim/game/configs/game_training_rpo_02_vbar_approach.yaml")
    context = _operator_plot_context(path)
    training_cfg = context.training_config
    assert training_cfg is not None
    rel = np.array(context.initial_relative_ric_km_s[:3], dtype=float)
    target = np.zeros(3, dtype=float)

    ri_center = _operator_camera_center_ric(context, chaser_current=rel, target_current=target, x_axis=1, y_axis=0)
    rc_center = _operator_camera_center_ric(context, chaser_current=rel, target_current=target, x_axis=2, y_axis=0)
    ri_fr = _operator_forbidden_region_projection_points(training_cfg, x_axis=1, y_axis=0, offset=target)
    rc_fr = _operator_forbidden_region_projection_points(training_cfg, x_axis=2, y_axis=0, offset=target)

    assert context.camera_mode == "target_pair"
    assert context.target_centered_plot_axes == {"RI": ("y",)}
    assert context.plot_overlays_in_zoom is False
    assert context.plot_overlays_in_zoom_by_plane == {"RC": True}
    assert context.proximity_ring_plot_planes == ("RI",)
    assert ri_center == pytest.approx(np.array([0.0, -2.5, 0.0]))
    assert rc_center == pytest.approx(np.zeros(3, dtype=float))
    assert len(ri_fr) == 3
    assert len(rc_fr) == 2
    assert _operator_minimum_plot_span_km(
        context,
        x_axis=1,
        y_axis=0,
        target_current=target,
        nmt=np.empty((0, 3), dtype=float),
        nmt_bounds=(),
    ) == pytest.approx(MIN_PLOT_SPAN_KM)
    assert _operator_minimum_plot_span_km(
        context,
        x_axis=2,
        y_axis=0,
        target_current=target,
        nmt=np.empty((0, 3), dtype=float),
        nmt_bounds=(),
    ) == pytest.approx(5.8 * PLOT_OVERLAY_MARGIN)


def test_operator_rbar_script_preview_stays_target_centered_without_fixed_spans() -> None:
    path = Path("sim/game/configs/game_training_rpo_03_rbar_approach.yaml")
    context = _operator_plot_context(path)
    training_cfg = context.training_config
    assert training_cfg is not None
    rel = np.array(context.initial_relative_ric_km_s[:3], dtype=float)
    target = np.zeros(3, dtype=float)

    ri_center = _operator_camera_center_ric(context, chaser_current=rel, target_current=target, x_axis=1, y_axis=0)
    rc_center = _operator_camera_center_ric(context, chaser_current=rel, target_current=target, x_axis=2, y_axis=0)

    assert context.camera_mode == "target_pair"
    assert context.target_centered_plot_planes == ("RI", "RC")
    assert context.plot_overlays_in_zoom is False
    assert context.plot_fixed_axis_half_span_km == {}
    assert ri_center == pytest.approx(np.zeros(3, dtype=float))
    assert rc_center == pytest.approx(np.zeros(3, dtype=float))
    assert len(_operator_forbidden_region_projection_points(training_cfg, x_axis=1, y_axis=0, offset=target)) == 1
    assert len(_operator_forbidden_region_projection_points(training_cfg, x_axis=2, y_axis=0, offset=target)) == 1


def test_operator_burn_provider_executes_single_ric_impulse() -> None:
    plan = parse_operator_burn_plan("T= 10 s, 2.0 m/s R, 1.0 m/s I, 0.2 m/s C")
    provider = OperatorBurnCommandProvider(
        plan,
        controlled_object_id="chaser",
        reference_object_id="target",
    )
    target = StateTruth(
        position_eci_km=np.array([7000.0, 0.0, 0.0], dtype=float),
        velocity_eci_km_s=np.array([0.0, 7.5, 0.0], dtype=float),
        attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        angular_rate_body_rad_s=np.zeros(3, dtype=float),
        mass_kg=100.0,
        t_s=0.0,
    )
    own_knowledge = {"target": _knowledge_from_state6(np.hstack((target.position_eci_km, target.velocity_eci_km_s)))}

    before = provider(
        truth=target,
        t_s=8.0,
        dt_s=1.0,
        object_id="chaser",
        own_knowledge=own_knowledge,
    )
    burn = provider(
        truth=target,
        t_s=9.5,
        dt_s=1.0,
        object_id="chaser",
        own_knowledge=own_knowledge,
    )
    burn_delta_v_ric = provider.last_executed_delta_v_ric_m_s
    after = provider(
        truth=target,
        t_s=10.5,
        dt_s=1.0,
        object_id="chaser",
        own_knowledge=own_knowledge,
    )

    assert np.allclose(before["thrust_eci_km_s2"], np.zeros(3), atol=1e-15)
    assert np.allclose(burn["thrust_eci_km_s2"], np.array([0.002, 0.001, 0.0002]), atol=1e-12)
    assert burn["command_mode_flags"]["operator_burn_index"] == 1
    assert burn_delta_v_ric == pytest.approx((2.0, 1.0, 0.2))
    assert provider.executed_delta_v_m_s == pytest.approx(np.sqrt(2.0**2 + 1.0**2 + 0.2**2))
    assert np.allclose(after["thrust_eci_km_s2"], np.zeros(3), atol=1e-15)
    assert provider.last_executed_delta_v_ric_m_s is None


def test_operator_burn_provider_uses_moon_ric_frame_for_cislunar_impulse() -> None:
    plan = parse_operator_burn_plan("T= 0 s, 1.0 m/s R")
    provider = OperatorBurnCommandProvider(
        plan,
        controlled_object_id="chaser",
        reference_object_id="target",
        control_mode="moon_ric_translation",
        relative_frame="moon_ric",
    )
    moon = cr3bp_moon_state_km_s()
    target_state = moon + np.array([1000.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=float)
    target = StateTruth(
        position_eci_km=target_state[:3],
        velocity_eci_km_s=target_state[3:],
        attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        angular_rate_body_rad_s=np.zeros(3, dtype=float),
        mass_kg=100.0,
        t_s=0.0,
    )
    own_knowledge = {"target": _knowledge_from_state6(target_state)}

    burn = provider(
        truth=target,
        t_s=0.0,
        dt_s=1.0,
        object_id="chaser",
        own_knowledge=own_knowledge,
    )

    assert np.allclose(burn["thrust_eci_km_s2"], np.array([0.001, 0.0, 0.0]), atol=1e-12)
    assert burn["command_mode_flags"]["operator_burn_delta_v_ric_m_s"] == pytest.approx((1.0, 0.0, 0.0))


def test_operator_burn_provider_applies_actuator_error_to_impulse() -> None:
    plan = parse_operator_burn_plan("T= 10 s, 2.0 m/s R, 1.0 m/s I, 0.2 m/s C")
    provider = OperatorBurnCommandProvider(
        plan,
        controlled_object_id="chaser",
        reference_object_id="target",
        actuator_error_fraction=0.025,
    )
    target = StateTruth(
        position_eci_km=np.array([7000.0, 0.0, 0.0], dtype=float),
        velocity_eci_km_s=np.array([0.0, 7.5, 0.0], dtype=float),
        attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        angular_rate_body_rad_s=np.zeros(3, dtype=float),
        mass_kg=100.0,
        t_s=0.0,
    )
    own_knowledge = {"target": _knowledge_from_state6(np.hstack((target.position_eci_km, target.velocity_eci_km_s)))}

    burn = provider(
        truth=target,
        t_s=9.5,
        dt_s=1.0,
        object_id="chaser",
        own_knowledge=own_knowledge,
    )

    expected_delta_v_ric_m_s = np.array([2.0, 1.0, 0.2], dtype=float) * 1.025
    assert burn["command_mode_flags"]["operator_burn_planned_delta_v_ric_m_s"] == pytest.approx((2.0, 1.0, 0.2))
    assert burn["command_mode_flags"]["operator_burn_delta_v_ric_m_s"] == pytest.approx(expected_delta_v_ric_m_s)
    assert burn["command_mode_flags"]["operator_actuator_error_fraction"] == pytest.approx(0.025)
    assert provider.last_executed_delta_v_ric_m_s == pytest.approx(expected_delta_v_ric_m_s)
    assert provider.executed_delta_v_m_s == pytest.approx(np.linalg.norm(expected_delta_v_ric_m_s))


def test_operator_projection_transition_uses_pre_and_post_burn_ric_velocity() -> None:
    class FakeDashboard:
        def __init__(self) -> None:
            self.rel_hist = [
                np.array([1.0, -2.0, 0.5, 0.010, -0.020, 0.003], dtype=float),
            ]
            self.transitions: list[tuple[np.ndarray, np.ndarray, float]] = []

        def set_operator_projection_transition(
            self,
            pre_burn_rel: np.ndarray,
            post_burn_rel: np.ndarray,
            *,
            duration_s: float | None = None,
        ) -> None:
            self.transitions.append(
                (
                    np.array(pre_burn_rel, dtype=float),
                    np.array(post_burn_rel, dtype=float),
                    float(duration_s or 0.0),
                )
            )

    provider = type(
        "Provider",
        (),
        {
            "last_executed_burn": object(),
            "last_executed_delta_v_ric_m_s": (2.0, -1.0, 0.5),
        },
    )()
    dashboard = FakeDashboard()

    duration_s = _trigger_operator_projection_transition(dashboard, provider)

    assert len(dashboard.transitions) == 1
    pre_burn, post_burn, transition_duration_s = dashboard.transitions[0]
    assert np.allclose(post_burn, dashboard.rel_hist[-1])
    assert np.allclose(pre_burn[:3], post_burn[:3])
    assert np.allclose(pre_burn[3:6], post_burn[3:6] - np.array([0.002, -0.001, 0.0005]))
    assert duration_s == pytest.approx(_operator_burn_visual_duration_s(np.linalg.norm([2.0, -1.0, 0.5])))
    assert transition_duration_s == pytest.approx(duration_s)


def test_manual_translation_bypasses_orbital_command_latch_for_tap_burns(tmp_path: Path) -> None:
    state = KeyboardCommandState(yaw=-1.0, use_timing_accumulator=True)
    provider = ManualGameCommandProvider(
        command_state=state,
        max_accel_km_s2=2.0e-5,
        control_mode="ric_translation",
        reference_object_id="target",
    )
    session = SimulationSession.from_config(SimulationConfig.from_dict(_game_config(tmp_path)))
    session.set_external_intent_provider("chaser", provider)
    session.reset()
    state.accumulate_timed_input(0.02, speed_multiple=2.0, control_mode="ric_translation")
    state.yaw = 0.0

    burn = session.step(dt_s=0.04)
    coast = session.step(dt_s=0.04)

    assert np.linalg.norm(burn.applied_thrust["chaser"]) == pytest.approx(2.0e-5)
    assert np.linalg.norm(coast.applied_thrust["chaser"]) == pytest.approx(0.0)


def test_moon_ric_translation_provider_commands_target_about_moon_frame() -> None:
    state = KeyboardCommandState(pitch=1.0, yaw=0.0, roll=0.0)
    provider = ManualGameCommandProvider(
        command_state=state,
        max_accel_km_s2=2.5e-4,
        control_mode="moon_ric_translation",
        reference_object_id="target",
    )
    target_state = cr3bp_halo_seed_state_km_s(family="l2_nrho_southern")
    target = StateTruth(
        position_eci_km=target_state[:3],
        velocity_eci_km_s=target_state[3:],
        attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        angular_rate_body_rad_s=np.zeros(3, dtype=float),
        mass_kg=100.0,
        t_s=0.0,
    )

    out = provider(
        truth=target,
        t_s=0.0,
        dt_s=1.0,
        object_id="chaser",
        own_knowledge={"target": _knowledge_from_state6(target_state)},
    )

    target_moon = target_state - cr3bp_moon_state_km_s()
    expected = ric_dcm_ir_from_rv(target_moon[:3], target_moon[3:]) @ np.array([2.5e-4, 0.0, 0.0])
    assert out["command_mode_flags"]["player_control_mode"] == "moon_ric_translation"
    assert np.allclose(out["thrust_eci_km_s2"], expected, atol=1e-12)


def test_ric_translation_provider_uses_timed_input_duty_cycle() -> None:
    state = KeyboardCommandState(yaw=1.0, use_timing_accumulator=True)
    provider = ManualGameCommandProvider(
        command_state=state,
        max_accel_km_s2=2.0e-5,
        control_mode="ric_translation",
        reference_object_id="target",
    )
    target = StateTruth(
        position_eci_km=np.array([7000.0, 0.0, 0.0], dtype=float),
        velocity_eci_km_s=np.array([0.0, 7.5, 0.0], dtype=float),
        attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        angular_rate_body_rad_s=np.zeros(3, dtype=float),
        mass_kg=100.0,
        t_s=0.0,
    )
    state.accumulate_timed_input(0.05, speed_multiple=10.0, control_mode="ric_translation")

    out = provider(
        truth=target,
        t_s=0.0,
        dt_s=2.0,
        object_id="chaser",
        own_knowledge={},
    )
    empty = provider(
        truth=target,
        t_s=2.0,
        dt_s=2.0,
        object_id="chaser",
        own_knowledge={},
    )

    assert np.allclose(out["thrust_eci_km_s2"], np.array([0.0, 5.0e-6, 0.0]), atol=1e-12)
    assert np.allclose(empty["thrust_eci_km_s2"], np.zeros(3, dtype=float), atol=1e-15)


def test_timed_translation_input_caps_pending_burn_to_one_step() -> None:
    state = KeyboardCommandState(yaw=1.0, use_timing_accumulator=True)
    state.accumulate_timed_input(
        1.0,
        speed_multiple=10.0,
        control_mode="ric_translation",
        max_pending_sim_s=2.0,
    )

    first = state.consume_ric_duty_cycle(2.0)
    second = state.consume_ric_duty_cycle(2.0)

    np.testing.assert_allclose(first, np.array([0.0, 1.0, 0.0], dtype=float))
    np.testing.assert_allclose(second, np.zeros(3, dtype=float))


def test_timed_translation_tap_exposes_pending_step_duration() -> None:
    state = KeyboardCommandState(yaw=1.0, use_timing_accumulator=True)
    state.accumulate_timed_input(
        0.008,
        speed_multiple=5.0,
        control_mode="ric_translation",
        max_pending_sim_s=0.1,
    )

    assert _timed_maneuver_pending_sim_s(state, control_mode="ric_translation") == pytest.approx(0.04)


def test_attitude_thrust_provider_uses_timed_firing_duty_cycle() -> None:
    state = KeyboardCommandState(firing=True, use_timing_accumulator=True)
    provider = ManualGameCommandProvider(
        command_state=state,
        max_accel_km_s2=2.0e-5,
        control_mode="attitude_thrust",
    )
    target = StateTruth(
        position_eci_km=np.array([7000.0, 0.0, 0.0], dtype=float),
        velocity_eci_km_s=np.array([0.0, 7.5, 0.0], dtype=float),
        attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        angular_rate_body_rad_s=np.zeros(3, dtype=float),
        mass_kg=100.0,
        t_s=0.0,
    )
    state.accumulate_timed_input(0.05, speed_multiple=10.0, control_mode="attitude_thrust")

    out = provider(truth=target, t_s=0.0, dt_s=2.0, object_id="chaser")
    empty = provider(truth=target, t_s=2.0, dt_s=2.0, object_id="chaser")

    assert out["mission_mode"]["firing_duty_cycle"] == pytest.approx(0.25)
    assert np.allclose(out["thrust_eci_km_s2"], np.array([5.0e-6, 0.0, 0.0]), atol=1e-12)
    assert empty["mission_mode"]["firing_duty_cycle"] == pytest.approx(0.0)
    assert np.allclose(empty["thrust_eci_km_s2"], np.zeros(3, dtype=float), atol=1e-15)


def test_timed_firing_input_caps_pending_burn_to_one_step() -> None:
    state = KeyboardCommandState(firing=True, use_timing_accumulator=True)
    state.accumulate_timed_input(
        1.0,
        speed_multiple=10.0,
        control_mode="attitude_thrust",
        max_pending_sim_s=2.0,
    )

    first = state.consume_firing_duty_cycle(2.0)
    second = state.consume_firing_duty_cycle(2.0)

    assert first == pytest.approx(1.0)
    assert second == pytest.approx(0.0)


def test_training_tracker_scores_keepout_and_goal() -> None:
    cfg = RPOTrainingConfig(
        enabled=True,
        scenario_id="unit",
        learning_goal="test",
        keepout_radius_km=0.5,
        goal_radius_km=0.25,
        goal_relative_ric_km=np.array([0.0, -1.0, 0.0], dtype=float),
        max_goal_speed_km_s=0.01,
    )
    tracker = RPOTrainingTracker(cfg)
    target = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])
    chaser0 = np.array([7000.0, -1.5, 0.0, 0.0, 7.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])
    chaser1 = np.array([7000.0, -1.0, 0.0, 0.0, 7.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])
    for idx, chaser in enumerate((chaser0, chaser1)):
        tracker.record(
            snapshot=type(
                "Snapshot",
                (),
                {
                    "time_s": float(idx),
                    "truth": {"target": target, "chaser": chaser},
                    "applied_thrust": {"chaser": np.zeros(3, dtype=float)},
                },
            )()
        )

    score = tracker.score()

    assert score.goal_met is True
    assert score.keepout_violation is False
    assert score.final_goal_error_km <= 0.25


def test_training_tracker_requires_tutorial_burn_axes_and_speed_change() -> None:
    cfg = RPOTrainingConfig(
        enabled=True,
        scenario_id="tutorial-unit",
        learning_goal="test",
        goal_range_km=0.25,
        max_time_s=100.0,
        max_delta_v_m_s=1.0,
        required_burn_axes=("radial", "in_track", "cross_track"),
        require_speed_multiplier_change=True,
    )
    tracker = RPOTrainingTracker(cfg)
    target_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.54605329, 0.0], dtype=float)
    rel_ric = np.array([0.0, -0.2, 0.0, 0.0, 0.0, 0.0], dtype=float)
    chaser_state = ric_rect_state_to_eci(rel_ric, target_state[:3], target_state[3:])

    tracker.record(
        snapshot=type(
            "Snapshot",
            (),
            {
                "time_s": 0.0,
                "truth": {"target": target_state, "chaser": chaser_state},
                "applied_thrust": {"chaser": np.zeros(3, dtype=float)},
            },
        )()
    )

    score = tracker.score()
    metrics = _mission_metrics(cfg, score)
    checklist = _mission_checklist(cfg, score)

    assert score.level_passed is False
    assert score.level_failed is False
    assert score.achieved_time_s is None
    assert score.burn_axes_satisfied == ()
    assert score.speed_multiplier_changed is False
    assert "Radial burn required." in score.pass_fail_reasons
    assert "In-track burn required." in score.pass_fail_reasons
    assert "Cross-track burn required." in score.pass_fail_reasons
    assert "Speed multiplier change required." in score.pass_fail_reasons
    assert "WARN Burns R-/I-/C-" in metrics
    assert "WARN Speed X" in metrics
    assert checklist[:4] == (
        "WARN Radial burn",
        "WARN In-track burn",
        "WARN Cross-track burn",
        "WARN Change speed",
    )
    assert "Tutorial checklist" in tracker.current_hint()


def test_training_tracker_ignores_small_burn_axis_leakage() -> None:
    cfg = RPOTrainingConfig(
        enabled=True,
        scenario_id="tutorial-unit",
        learning_goal="test",
        goal_range_km=0.25,
        required_burn_axes=("radial", "in_track"),
    )
    tracker = RPOTrainingTracker(cfg)
    target_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.54605329, 0.0], dtype=float)
    rel_ric = np.array([0.0, -0.2, 0.0, 0.0, 0.0, 0.0], dtype=float)
    chaser_state = ric_rect_state_to_eci(rel_ric, target_state[:3], target_state[3:])
    c_ir = ric_dcm_ir_from_rv(target_state[:3], target_state[3:])
    radial_with_frame_leakage = c_ir @ np.array([1.0e-5, 2.0e-8, 0.0], dtype=float)

    tracker.record(
        snapshot=type(
            "Snapshot",
            (),
            {
                "time_s": 0.0,
                "truth": {"target": target_state, "chaser": chaser_state},
                "applied_thrust": {"chaser": radial_with_frame_leakage},
            },
        )()
    )

    score = tracker.score()

    assert score.burn_axes_satisfied == ("radial",)
    assert score.level_passed is False
    assert "In-track burn required." in score.pass_fail_reasons


def test_training_tracker_requires_mostly_single_axis_tutorial_burns() -> None:
    cfg = RPOTrainingConfig(
        enabled=True,
        scenario_id="tutorial-unit",
        learning_goal="test",
        goal_range_km=0.25,
        required_burn_axes=("radial", "in_track"),
        required_burn_axis_min_component_fraction=0.75,
    )
    tracker = RPOTrainingTracker(cfg)
    target_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.54605329, 0.0], dtype=float)
    rel_ric = np.array([0.0, -0.2, 0.0, 0.0, 0.0, 0.0], dtype=float)
    chaser_state = ric_rect_state_to_eci(rel_ric, target_state[:3], target_state[3:])
    c_ir = ric_dcm_ir_from_rv(target_state[:3], target_state[3:])
    diagonal_burn = c_ir @ np.array([1.0e-5, 1.0e-5, 0.0], dtype=float)

    tracker.record(
        snapshot=type(
            "Snapshot",
            (),
            {
                "time_s": 0.0,
                "truth": {"target": target_state, "chaser": chaser_state},
                "applied_thrust": {"chaser": diagonal_burn},
            },
        )()
    )

    score = tracker.score()

    assert score.burn_axes_satisfied == ()
    assert "Radial burn required." in score.pass_fail_reasons
    assert "In-track burn required." in score.pass_fail_reasons


def test_training_tracker_passes_tutorial_after_required_controls() -> None:
    cfg = RPOTrainingConfig(
        enabled=True,
        scenario_id="tutorial-unit",
        learning_goal="test",
        goal_range_km=0.25,
        max_time_s=100.0,
        max_delta_v_m_s=1.0,
        required_burn_axes=("radial", "in_track", "cross_track"),
        require_speed_multiplier_change=True,
    )
    tracker = RPOTrainingTracker(cfg)
    target_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.54605329, 0.0], dtype=float)
    rel_ric = np.array([0.0, -0.2, 0.0, 0.0, 0.0, 0.0], dtype=float)
    chaser_state = ric_rect_state_to_eci(rel_ric, target_state[:3], target_state[3:])
    c_ir = ric_dcm_ir_from_rv(target_state[:3], target_state[3:])

    for idx, thrust_ric in enumerate(
        (
            np.array([1.0e-5, 0.0, 0.0], dtype=float),
            np.array([0.0, 1.0e-5, 0.0], dtype=float),
            np.array([0.0, 0.0, 1.0e-5], dtype=float),
        )
    ):
        tracker.record(
            snapshot=type(
                "Snapshot",
                (),
                {
                    "time_s": float(idx),
                    "truth": {"target": target_state, "chaser": chaser_state},
                    "applied_thrust": {"chaser": c_ir @ thrust_ric},
                },
            )()
        )
    tracker.record_speed_multiplier_change()

    score = tracker.score()
    metrics = _mission_metrics(cfg, score)
    checklist = _mission_checklist(cfg, score)

    assert score.level_passed is True
    assert score.level_failed is False
    assert score.achieved_time_s == pytest.approx(2.0)
    assert score.burn_axes_satisfied == ("radial", "in_track", "cross_track")
    assert score.speed_multiplier_changed is True
    assert "OK Burns R+/I+/C+" in metrics
    assert "OK Speed X" in metrics
    assert checklist[:4] == (
        "OK Radial burn",
        "OK In-track burn",
        "OK Cross-track burn",
        "OK Change speed",
    )


def test_training_tracker_requires_tutorial_coast_after_burn() -> None:
    cfg = RPOTrainingConfig(
        enabled=True,
        scenario_id="tutorial-unit",
        learning_goal="test",
        goal_range_km=0.25,
        required_burn_axes=("radial",),
        required_coast_after_burn_s=10.0,
        tutorial_stage_hints={
            "radial": "Tap radial.",
            "coast": "Coast after a pulse.",
            "final_approach": "Settle gently.",
        },
    )
    tracker = RPOTrainingTracker(cfg)
    target_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.54605329, 0.0], dtype=float)
    rel_ric = np.array([0.0, -0.2, 0.0, 0.0, 0.0, 0.0], dtype=float)
    chaser_state = ric_rect_state_to_eci(rel_ric, target_state[:3], target_state[3:])
    c_ir = ric_dcm_ir_from_rv(target_state[:3], target_state[3:])

    for time_s, thrust_ric in (
        (0.0, np.array([1.0e-5, 0.0, 0.0], dtype=float)),
        (5.0, np.zeros(3, dtype=float)),
    ):
        tracker.record(
            snapshot=type(
                "Snapshot",
                (),
                {
                    "time_s": time_s,
                    "truth": {"target": target_state, "chaser": chaser_state},
                    "applied_thrust": {"chaser": c_ir @ thrust_ric},
                },
            )()
        )

    score = tracker.score()

    assert score.level_passed is False
    assert score.coast_after_burn_satisfied is False
    assert "Coast for 10 s after a burn required." in score.pass_fail_reasons
    assert "Coast after a pulse." in tracker.current_hint()

    for time_s in (10.0, 15.0):
        tracker.record(
            snapshot=type(
                "Snapshot",
                (),
                {
                    "time_s": time_s,
                    "truth": {"target": target_state, "chaser": chaser_state},
                    "applied_thrust": {"chaser": np.zeros(3, dtype=float)},
                },
            )()
        )

    score = tracker.score()

    assert score.level_passed is True
    assert score.coast_after_burn_satisfied is True
    assert score.coast_after_burn_s == pytest.approx(10.0)


def test_ric_primer_only_enables_for_level_zero_guided_tutorial() -> None:
    burn = GuidedTutorialBurnConfig(name="plus_i", axis="in_track", sign=1, delta_v_m_s=0.25)
    tutorial = RPOTrainingConfig(
        enabled=True,
        scenario_id="rpo_00_tutorial",
        guided_tutorial_burns=(burn,),
    )

    assert _ric_primer_enabled(tutorial) is True
    assert _ric_primer_enabled(replace(tutorial, scenario_id="rpo_01_coast_relative_motion")) is False
    assert _ric_primer_enabled(replace(tutorial, sandbox_mode=True)) is False
    assert _ric_primer_enabled(tutorial, arcade_enabled=True) is False
    assert _ric_primer_stage(0)["id"] == "radial"
    assert _ric_primer_stage(1)["id"] == "in_track"
    assert _ric_primer_stage(2)["id"] == "cross_track"


def test_guided_tutorial_input_matches_only_requested_axis() -> None:
    stage = GuidedTutorialBurnConfig(name="plus_i", axis="in_track", sign=1, delta_v_m_s=0.25)
    state = KeyboardCommandState(yaw=1.0)

    assert _guided_tutorial_input_matches(state, stage) is True
    assert _guided_tutorial_wrong_input_active(state, stage) is False

    state.pitch = 1.0
    assert _guided_tutorial_input_matches(state, stage) is False
    assert _guided_tutorial_wrong_input_active(state, stage) is True

    state.pitch = 0.0
    state.yaw = -1.0
    assert _guided_tutorial_input_matches(state, stage) is False
    assert _guided_tutorial_wrong_input_active(state, stage) is True

    state.yaw = 0.0
    assert _guided_tutorial_wrong_input_active(state, stage) is False


def test_guided_tutorial_wrong_key_hint_names_expected_control() -> None:
    stage = GuidedTutorialBurnConfig(name="plus_i", axis="in_track", sign=1, delta_v_m_s=0.25)
    runtime = game_runner.GuidedTutorialRuntime(wrong_key_active=True)

    assert _guided_tutorial_expected_key(stage) == "D"
    assert _guided_tutorial_stage_hint(stage, runtime) == "Wrong key - hold D for +I burn."


def test_guided_tutorial_target_path_applies_requested_burn() -> None:
    rel0 = np.array([0.0, -0.8, 0.0, 0.0, 0.0, 0.0], dtype=float)
    stage = GuidedTutorialBurnConfig(name="minus_c", axis="cross_track", sign=-1, delta_v_m_s=0.25)

    path = _guided_tutorial_target_path(rel0, 0.001, stage, samples=5)

    assert path.shape == (5, 6)
    assert path[0, 5] == pytest.approx(-0.00025)
    assert np.allclose(path[0, :5], rel0[:5])


def test_guided_tutorial_dashboard_path_uses_tracker_history() -> None:
    cfg = RPOTrainingConfig(
        enabled=True,
        scenario_id="tutorial-guided",
        guided_tutorial_burns=(
            GuidedTutorialBurnConfig(name="plus_i", axis="in_track", sign=1, delta_v_m_s=0.25),
        ),
    )
    tracker = RPOTrainingTracker(cfg)
    tracker.rel_ric_hist.append(np.array([0.0, -0.8, 0.0, 0.0, 0.0, 0.0], dtype=float))
    tracker.mean_motion_hist.append(0.001)
    runtime = game_runner.GuidedTutorialRuntime()
    dashboard = type("Dashboard", (), {"tutorial_target_path_ric": np.empty((0, 6)), "_frame_cache_dirty": False})()

    _guided_tutorial_update_dashboard_path(dashboard, tracker, cfg, runtime)

    assert dashboard.tutorial_target_path_ric.shape[1] == 6
    assert dashboard.tutorial_target_path_ric[0, 4] == pytest.approx(0.00025)
    assert dashboard._frame_cache_dirty is True


def test_operator_mode_skips_guided_tutorial_path_sync() -> None:
    cfg = RPOTrainingConfig(
        enabled=True,
        scenario_id="tutorial-guided",
        guided_tutorial_burns=(
            GuidedTutorialBurnConfig(name="plus_i", axis="in_track", sign=1, delta_v_m_s=0.25),
        ),
    )
    tracker = RPOTrainingTracker(cfg)
    tracker.rel_ric_hist.append(np.array([0.0, -0.8, 0.0, 0.0, 0.0, 0.0], dtype=float))
    tracker.mean_motion_hist.append(0.001)
    runtime = game_runner.GuidedTutorialRuntime()
    initial_path = np.ones((3, 6), dtype=float)
    dashboard = type(
        "Dashboard",
        (),
        {"tutorial_target_path_ric": initial_path.copy(), "_frame_cache_dirty": False},
    )()

    _sync_guided_tutorial_path_for_mode(
        dashboard,
        tracker,
        cfg,
        runtime,
        game_mode="operator",
    )

    assert np.array_equal(dashboard.tutorial_target_path_ric, initial_path)
    assert dashboard._frame_cache_dirty is False

    _sync_guided_tutorial_path_for_mode(
        dashboard,
        tracker,
        cfg,
        runtime,
        game_mode="pilot",
    )

    assert dashboard.tutorial_target_path_ric.shape[1] == 6
    assert dashboard.tutorial_target_path_ric[0, 4] == pytest.approx(0.00025)
    assert dashboard._frame_cache_dirty is True


def test_guided_tutorial_speed_step_helpers() -> None:
    burn = GuidedTutorialBurnConfig(name="plus_i", axis="in_track", sign=1, delta_v_m_s=0.25)
    cfg = RPOTrainingConfig(
        enabled=True,
        scenario_id="tutorial-guided",
        guided_tutorial_burns=(burn,),
        guided_tutorial_speed_step=GuidedTutorialSpeedStepConfig(
            name="speed_to_10x",
            after_burn_name="plus_i",
            target_speed_multiplier=10.0,
            hint="Want to go faster? Hit the up arrow key.",
        ),
    )

    assert _guided_tutorial_speed_step_follows_burn(cfg, burn) is True
    assert _guided_tutorial_speed_step_reached(cfg, 5.0) is False
    assert _guided_tutorial_speed_step_reached(cfg, 10.0) is True
    assert "Current speed: 5x" in _guided_tutorial_speed_step_hint(cfg, 5.0)


def test_guided_tutorial_speed_step_reset_uses_zero_relative_velocity() -> None:
    config_path = (
        Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_00_tutorial.yaml"
    )
    config = SimulationConfig.from_yaml(config_path)
    training_cfg = RPOTrainingConfig.from_metadata(dict(config.scenario.metadata or {}))
    command_state = KeyboardCommandState(yaw=1.0)
    trainer = RPOTrainingTracker(training_cfg)
    trainer.mark_guided_tutorial_burn_complete("plus_in_track")

    class FakeDashboard:
        def __init__(self) -> None:
            self.snapshots: list[object] = []
            self._frame_cache_dirty = False

        def clear(self) -> None:
            self.snapshots.clear()

        def push_snapshot(self, snapshot: object) -> None:
            self.snapshots.append(snapshot)

    dashboard = FakeDashboard()

    _reset_guided_tutorial_stage_attempt(
        attempt_config=config,
        command_state=command_state,
        trainer=trainer,
        dashboard=dashboard,
        training_cfg=training_cfg,
        controlled_object_id="chaser",
        attitude_rate_deg_s=45.0,
        control_mode="ric_translation",
        ric_reference_object_id="target",
    )

    assert command_state.yaw == pytest.approx(0.0)
    assert trainer.guided_tutorial_burns_satisfied() == ("plus_in_track",)
    assert np.allclose(trainer.rel_ric_hist[-1][3:], np.zeros(3, dtype=float))
    assert dashboard.snapshots


def test_training_tracker_requires_guided_tutorial_speed_step() -> None:
    burn = GuidedTutorialBurnConfig(name="plus_i", axis="in_track", sign=1, delta_v_m_s=0.25)
    cfg = RPOTrainingConfig(
        enabled=True,
        scenario_id="tutorial-guided-speed",
        goal_range_km=1.0,
        guided_tutorial_burns=(burn,),
        guided_tutorial_speed_step=GuidedTutorialSpeedStepConfig(
            name="speed_to_10x",
            after_burn_name="plus_i",
            target_speed_multiplier=10.0,
            label="Speed 10x",
        ),
    )
    tracker = RPOTrainingTracker(cfg)
    target_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.54605329, 0.0], dtype=float)
    tracker.record(
        snapshot=type(
            "Snapshot",
            (),
            {
                "time_s": 0.0,
                "truth": {"target": target_state, "chaser": target_state},
                "applied_thrust": {"chaser": np.zeros(3, dtype=float)},
            },
        )()
    )
    tracker.mark_guided_tutorial_burn_complete("plus_i")

    score = tracker.score()

    assert score.level_passed is False
    assert score.guided_tutorial_speed_satisfied is False
    assert "Speed 10x tutorial step required." in score.pass_fail_reasons

    tracker.mark_guided_tutorial_speed_complete()
    score = tracker.score()

    assert score.level_passed is True
    assert score.guided_tutorial_speed_satisfied is True


def test_guided_tutorial_delta_v_tracks_signed_axis_progress() -> None:
    cfg = RPOTrainingConfig(
        enabled=True,
        scenario_id="tutorial-guided",
        guided_tutorial_burns=(
            GuidedTutorialBurnConfig(name="plus_r", axis="radial", sign=1, delta_v_m_s=0.25),
        ),
    )
    tracker = RPOTrainingTracker(cfg)
    target_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.54605329, 0.0], dtype=float)
    chaser_state = ric_rect_state_to_eci(np.zeros(6, dtype=float), target_state[:3], target_state[3:])
    c_ir = ric_dcm_ir_from_rv(target_state[:3], target_state[3:])

    for time_s, thrust_ric in (
        (0.0, np.zeros(3, dtype=float)),
        (10.0, np.array([1.0e-5, 0.0, 0.0], dtype=float)),
        (20.0, np.array([1.0e-5, 0.0, 0.0], dtype=float)),
        (30.0, np.array([-1.0e-5, 0.0, 0.0], dtype=float)),
    ):
        tracker.record(
            snapshot=type(
                "Snapshot",
                (),
                {
                    "time_s": time_s,
                    "truth": {"target": target_state, "chaser": chaser_state},
                    "applied_thrust": {"chaser": c_ir @ thrust_ric},
                },
            )()
        )

    assert _guided_tutorial_delta_v_m_s(tracker, cfg.guided_tutorial_burns[0]) == pytest.approx(0.2)


def test_nmt_goal_uses_two_to_one_intrack_radial_shape() -> None:
    center = np.zeros(3, dtype=float)
    on_ellipse = np.array(
        [
            [1.5, 0.0, 0.0],
            [0.0, -3.0, 0.0],
            [-1.5, 0.0, 0.0],
            [0.0, 3.0, 0.0],
        ],
        dtype=float,
    )
    circular_but_not_nmt = np.array([[0.0, -1.5, 0.0]], dtype=float)

    assert np.allclose(nmt_position_error_km(on_ellipse, radial_amplitude_km=1.5, center_ric_km=center), 0.0)
    assert nmt_position_error_km(circular_but_not_nmt, radial_amplitude_km=1.5, center_ric_km=center)[0] > 0.0


def test_nmt_velocity_goal_matches_passive_hcw_relationship() -> None:
    n = 0.001
    state = np.array([0.0, -3.0, 0.0, -0.0015, 0.0, 0.0], dtype=float)
    stopped_on_ellipse = np.array([0.0, -3.0, 0.0, 0.0, 0.0, 0.0], dtype=float)

    assert (
        nmt_velocity_error_km_s(state, mean_motion_rad_s=n, radial_amplitude_km=1.5, center_ric_km=np.zeros(3))
        < 1.0e-12
    )
    assert (
        nmt_velocity_error_km_s(
            stopped_on_ellipse,
            mean_motion_rad_s=n,
            radial_amplitude_km=1.5,
            center_ric_km=np.zeros(3),
        )
        > 0.0
    )


def test_nmt_cross_track_phase_creates_rc_ellipse() -> None:
    curve = nmt_curve_points_km(
        radial_amplitude_km=1.5,
        cross_track_amplitude_km=1.0,
        cross_track_phase_deg=45.0,
        center_ric_km=np.zeros(3),
    )

    assert np.ptp(curve[:, 2]) > 1.9
    assert abs(np.corrcoef(curve[:, 0], curve[:, 2])[0, 1]) > 0.5
    assert (
        nmt_position_error_km(
            curve[[25, 180, 320]],
            radial_amplitude_km=1.5,
            cross_track_amplitude_km=1.0,
            cross_track_phase_deg=45.0,
            center_ric_km=np.zeros(3),
        ).max()
        == 0.0
    )


def test_level_one_uses_circular_rc_phase_objective() -> None:
    root = Path(__file__).resolve().parents[1] / "game" / "configs"
    config = SimulationConfig.from_yaml(root / "game_training_rpo_01_coast_relative_motion.yaml")
    training_cfg = RPOTrainingConfig.from_metadata(dict(config.scenario.metadata or {}))
    phase_burn = training_cfg.required_phase_burns[0]

    assert training_cfg.goal_nmt_radial_amplitude_km == pytest.approx(1.5)
    assert training_cfg.goal_nmt_cross_track_amplitude_km == pytest.approx(1.5)
    assert training_cfg.goal_nmt_cross_track_phase_deg == pytest.approx(90.0)
    assert phase_burn.name == "Cross-track phase burn"
    assert phase_burn.axis == "cross_track"
    assert phase_burn.radial_abs_km == pytest.approx(1.5)
    assert phase_burn.max_abs_intrack_km == pytest.approx(0.5)

    curve = nmt_curve_points_km(
        radial_amplitude_km=float(training_cfg.goal_nmt_radial_amplitude_km),
        cross_track_amplitude_km=float(training_cfg.goal_nmt_cross_track_amplitude_km),
        cross_track_phase_deg=float(training_cfg.goal_nmt_cross_track_phase_deg),
        center_ric_km=training_cfg.goal_nmt_center_ric_km,
    )
    radial = curve[:, 0]
    cross_track = curve[:, 2]

    assert np.max(np.abs(radial**2 + cross_track**2 - 1.5**2)) < 1.0e-6


def test_nmt_element_errors_ignore_phase_but_enforce_amplitudes_and_drift() -> None:
    n = 0.001
    state = np.array([0.0, -3.0, 0.0, -0.0015, 0.0, 0.001], dtype=float)

    errors = nmt_element_errors(
        state,
        mean_motion_rad_s=n,
        radial_amplitude_km=1.5,
        cross_track_amplitude_km=1.0,
        center_ric_km=np.zeros(3),
    )

    assert errors["radial_amplitude_error_km"][0] < 1.0e-12
    assert errors["cross_track_amplitude_error_km"][0] < 1.0e-12
    assert errors["drift_velocity_error_km_s"][0] < 1.0e-12


def test_training_tracker_drift_metric_is_constant_without_burns_for_nmt_levels() -> None:
    root = Path(__file__).resolve().parents[1] / "game" / "configs"
    level_configs = (
        "game_training_rpo_01_coast_relative_motion.yaml",
        "game_training_rpo_08_elliptic_nmc.yaml",
    )

    for level_config in level_configs:
        sim_cfg = SimulationConfig.from_yaml(root / level_config)
        training_cfg = RPOTrainingConfig.from_metadata(dict(sim_cfg.scenario.metadata or {}))
        session = SimulationSession.from_config(_attempt_config_for_training_clock(sim_cfg, training_cfg))
        snapshot = session.reset()
        tracker = RPOTrainingTracker(training_cfg)
        drift_errors = []
        for step_idx in range(240):
            if step_idx:
                snapshot = session.step()
            tracker.record(snapshot)
            drift_errors.append(tracker.score().final_nmt_drift_velocity_error_km_s)

        assert np.ptp(np.array(drift_errors, dtype=float)) < 1.0e-10


def test_training_tracker_level_passes_when_nmt_elements_met_within_budget() -> None:
    cfg = RPOTrainingConfig(
        enabled=True,
        scenario_id="unit-level",
        learning_goal="test",
        keepout_radius_km=0.25,
        goal_nmt_radial_amplitude_km=1.5,
        goal_nmt_cross_track_amplitude_km=1.0,
        goal_nmt_element_tolerance_km=0.05,
        goal_nmt_velocity_tolerance_km_s=0.00005,
        max_time_s=100.0,
        max_delta_v_m_s=1.0,
    )
    tracker = RPOTrainingTracker(cfg)
    target_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.54605329, 0.0], dtype=float)
    n = float(np.sqrt(398600.4418 / (7000.0**3)))
    rel_ric = np.array([0.0, -3.0, 1.0, -1.5 * n, 0.0, 0.0], dtype=float)
    chaser_state = ric_rect_state_to_eci(rel_ric, target_state[:3], target_state[3:])
    target = np.hstack((target_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))
    chaser = np.hstack((chaser_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))
    tracker.record(
        snapshot=type(
            "Snapshot",
            (),
            {
                "time_s": 10.0,
                "truth": {"target": target, "chaser": chaser},
                "applied_thrust": {"chaser": np.zeros(3, dtype=float)},
            },
        )()
    )

    score = tracker.score()

    assert score.level_passed is True
    assert score.achieved_time_s == 0.0
    assert score.final_nmt_radial_amplitude_error_km < 0.05
    assert score.final_nmt_cross_track_amplitude_error_km < 0.05


def test_training_tracker_requires_cross_track_phase_burn_at_radial_extremum() -> None:
    cfg = RPOTrainingConfig(
        enabled=True,
        scenario_id="unit-phase",
        learning_goal="test",
        keepout_radius_km=0.25,
        goal_nmt_radial_amplitude_km=1.5,
        goal_nmt_cross_track_amplitude_km=1.5,
        goal_nmt_cross_track_phase_deg=90.0,
        goal_nmt_element_tolerance_km=0.05,
        goal_nmt_velocity_tolerance_km_s=0.00005,
        max_time_s=100.0,
        max_delta_v_m_s=1.0,
        required_phase_burns=(
            RequiredPhaseBurnConfig(
                name="Cross-track phase burn",
                axis="cross_track",
                radial_abs_km=1.5,
                radial_tolerance_km=0.2,
                max_abs_intrack_km=0.35,
            ),
        ),
    )
    tracker = RPOTrainingTracker(cfg)
    target_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.54605329, 0.0], dtype=float)
    n = float(np.sqrt(398600.4418 / (7000.0**3)))
    c_ir = ric_dcm_ir_from_rv(target_state[:3], target_state[3:])
    target = np.hstack((target_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))
    phase_rel_ric = np.array([1.5, 0.0, 0.0, 0.0, -3.0 * n, -1.5 * n], dtype=float)
    chaser = np.hstack(
        (
            ric_rect_state_to_eci(phase_rel_ric, target_state[:3], target_state[3:]),
            np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0]),
        )
    )

    tracker.record(
        snapshot=type(
            "Snapshot",
            (),
            {
                "time_s": 0.0,
                "truth": {"target": target, "chaser": chaser},
                "applied_thrust": {"chaser": c_ir @ np.array([0.0, 0.0, 1.0e-5], dtype=float)},
            },
        )()
    )

    score = tracker.score()
    metrics = _mission_metrics(cfg, score)
    checklist = _mission_checklist(cfg, score)

    assert score.level_passed is True
    assert score.phase_burns_satisfied == ("Cross-track phase burn",)
    assert "OK Phase 1/1" in metrics
    assert "OK Cross-track phase burn" in checklist


def test_training_tracker_rejects_cross_track_phase_burn_away_from_radial_extremum() -> None:
    cfg = RPOTrainingConfig(
        enabled=True,
        scenario_id="unit-phase",
        learning_goal="test",
        keepout_radius_km=0.25,
        goal_nmt_radial_amplitude_km=1.5,
        goal_nmt_cross_track_amplitude_km=1.5,
        goal_nmt_cross_track_phase_deg=90.0,
        goal_nmt_element_tolerance_km=0.05,
        goal_nmt_velocity_tolerance_km_s=0.00005,
        max_time_s=100.0,
        max_delta_v_m_s=1.0,
        required_phase_burns=(
            RequiredPhaseBurnConfig(
                name="Cross-track phase burn",
                axis="cross_track",
                radial_abs_km=1.5,
                radial_tolerance_km=0.2,
                max_abs_intrack_km=0.35,
            ),
        ),
    )
    tracker = RPOTrainingTracker(cfg)
    target_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.54605329, 0.0], dtype=float)
    n = float(np.sqrt(398600.4418 / (7000.0**3)))
    c_ir = ric_dcm_ir_from_rv(target_state[:3], target_state[3:])
    target = np.hstack((target_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))
    bad_phase_rel_ric = np.array([0.0, -3.0, -1.5, -1.5 * n, 0.0, 0.0], dtype=float)
    good_final_rel_ric = np.array([1.5, 0.0, 0.0, 0.0, -3.0 * n, -1.5 * n], dtype=float)

    for idx, (rel_ric, thrust_ric) in enumerate(
        (
            (bad_phase_rel_ric, np.array([0.0, 0.0, 1.0e-5], dtype=float)),
            (good_final_rel_ric, np.zeros(3, dtype=float)),
        )
    ):
        chaser = np.hstack(
            (
                ric_rect_state_to_eci(rel_ric, target_state[:3], target_state[3:]),
                np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0]),
            )
        )
        tracker.record(
            snapshot=type(
                "Snapshot",
                (),
                {
                    "time_s": float(idx),
                    "truth": {"target": target, "chaser": chaser},
                    "applied_thrust": {"chaser": c_ir @ thrust_ric},
                },
            )()
        )

    score = tracker.score()
    metrics = _mission_metrics(cfg, score)
    checklist = _mission_checklist(cfg, score)

    assert score.level_passed is False
    assert score.level_failed is False
    assert score.phase_burns_satisfied == ()
    assert "Cross-track phase burn required." in score.pass_fail_reasons
    assert "WARN Phase 0/1" in metrics
    assert "WARN Cross-track phase burn" in checklist


def test_training_tracker_stationkeeping_goal_passes_with_goal_and_speed() -> None:
    cfg = RPOTrainingConfig(
        enabled=True,
        scenario_id="unit-vbar",
        learning_goal="test",
        keepout_radius_km=0.25,
        goal_radius_km=0.15,
        goal_relative_ric_km=np.array([0.0, -0.75, 0.0], dtype=float),
        max_goal_speed_km_s=0.0003,
        max_time_s=100.0,
        max_delta_v_m_s=1.0,
    )
    tracker = RPOTrainingTracker(cfg)
    target_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.54605329, 0.0], dtype=float)
    rel_ric = np.array([0.02, -0.80, 0.01, 0.0, 0.0001, 0.0], dtype=float)
    chaser_state = ric_rect_state_to_eci(rel_ric, target_state[:3], target_state[3:])
    target = np.hstack((target_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))
    chaser = np.hstack((chaser_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))
    tracker.record(
        snapshot=type(
            "Snapshot",
            (),
            {
                "time_s": 10.0,
                "truth": {"target": target, "chaser": chaser},
                "applied_thrust": {"chaser": np.zeros(3, dtype=float)},
            },
        )()
    )

    score = tracker.score()
    metrics = _mission_metrics(cfg, score)

    assert score.level_passed is True
    assert score.level_failed is False
    assert any("KO " in item for item in metrics)
    assert any("Goal " in item for item in metrics)
    assert any("Speed " in item for item in metrics)
    assert not any("NMT" in reason for reason in score.pass_fail_reasons)


def test_training_tracker_can_require_cross_track_amplitude_for_bar_approaches() -> None:
    cfg = RPOTrainingConfig(
        enabled=True,
        scenario_id="unit-vbar",
        goal_radius_km=0.15,
        goal_relative_ric_km=np.array([0.0, -0.75, 0.0], dtype=float),
        max_goal_speed_km_s=0.0003,
        max_cross_track_amplitude_km=0.02,
        max_delta_v_m_s=1.0,
    )
    target_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.54605329, 0.0], dtype=float)
    target = np.hstack((target_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))

    def score_for(rel_ric: np.ndarray) -> RPOTrainingScore:
        tracker = RPOTrainingTracker(cfg)
        chaser_state = ric_rect_state_to_eci(rel_ric, target_state[:3], target_state[3:])
        chaser = np.hstack((chaser_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))
        tracker.record(
            snapshot=type(
                "Snapshot",
                (),
                {
                    "time_s": 10.0,
                    "truth": {"target": target, "chaser": chaser},
                    "applied_thrust": {"chaser": np.zeros(3, dtype=float)},
                },
            )()
        )
        return tracker.score()

    high_c_amp = score_for(np.array([0.02, -0.80, 0.01, 0.0, 0.0001, 0.0001], dtype=float))
    low_c_amp = score_for(np.array([0.02, -0.80, 0.01, 0.0, 0.0001, 0.0], dtype=float))
    metrics = _mission_metrics(cfg, high_c_amp)

    assert high_c_amp.level_passed is False
    assert high_c_amp.final_nmt_cross_track_amplitude_km > 0.02
    assert any("Cross-track amplitude above" in reason for reason in high_c_amp.pass_fail_reasons)
    assert any("C Amp" in item for item in metrics)
    assert low_c_amp.level_passed is True
    assert low_c_amp.final_nmt_cross_track_amplitude_km <= 0.02


def test_training_configs_load_forbidden_regions_for_bar_approaches() -> None:
    root = Path(__file__).resolve().parents[1] / "game" / "configs"
    vbar_sim_cfg = SimulationConfig.from_yaml(root / "game_training_rpo_02_vbar_approach.yaml")
    rbar_sim_cfg = SimulationConfig.from_yaml(root / "game_training_rpo_03_rbar_approach.yaml")
    vbar = RPOTrainingConfig.from_metadata(dict(vbar_sim_cfg.scenario.metadata or {}))
    rbar = RPOTrainingConfig.from_metadata(dict(rbar_sim_cfg.scenario.metadata or {}))

    assert len(vbar.forbidden_regions) == 5
    assert any("V-bar" in region.name for region in vbar.forbidden_regions)
    assert {region.plot_planes for region in vbar.forbidden_regions} == {("RI",), ("RC",)}
    assert {region.kind for region in vbar.forbidden_regions} == {"box", "annular_sector"}
    vbar_ri_arch = next(
        region for region in vbar.forbidden_regions if region.kind == "annular_sector" and region.plane == "RI"
    )
    vbar_ri_boxes = [region for region in vbar.forbidden_regions if region.kind == "box" and region.plot_planes == ("RI",)]
    vbar_rc_boxes = [region for region in vbar.forbidden_regions if region.kind == "box" and region.plot_planes == ("RC",)]
    assert len(vbar_ri_boxes) == 2
    assert len(vbar_rc_boxes) == 2
    assert bool(vbar_ri_boxes[0].contains_positions(np.array([[1.0, -2.0, 0.0]], dtype=float))[0]) is True
    assert bool(vbar_ri_boxes[1].contains_positions(np.array([[-1.0, -2.0, 0.0]], dtype=float))[0]) is True
    assert bool(vbar_ri_arch.contains_positions(np.array([[0.0, 1.0, 0.0]], dtype=float))[0]) is True
    assert bool(vbar_ri_arch.contains_positions(np.array([[0.0, -0.75, 0.0]], dtype=float))[0]) is False
    assert bool(vbar_rc_boxes[0].contains_positions(np.array([[1.0, -2.0, 0.0]], dtype=float))[0]) is True
    assert bool(vbar_rc_boxes[1].contains_positions(np.array([[-1.0, -2.0, 0.0]], dtype=float))[0]) is True
    assert bool(vbar_rc_boxes[0].contains_positions(np.array([[0.0, -2.0, 1.0]], dtype=float))[0]) is False
    assert bool(vbar_rc_boxes[1].contains_positions(np.array([[0.0, -2.0, -1.0]], dtype=float))[0]) is False
    assert len(rbar.forbidden_regions) == 2
    assert any("R-bar" in region.name for region in rbar.forbidden_regions)
    assert {region.plot_planes for region in rbar.forbidden_regions} == {("RI",), ("RC",)}
    ri_arch = next(region for region in rbar.forbidden_regions if region.plane == "RI")
    rc_arch = next(region for region in rbar.forbidden_regions if region.plane == "RC")
    assert {region.kind for region in rbar.forbidden_regions} == {"annular_sector"}
    assert bool(ri_arch.contains_positions(np.array([[1.0, 0.0, 0.0]], dtype=float))[0]) is True
    assert bool(ri_arch.contains_positions(np.array([[-0.75, 0.0, 0.0]], dtype=float))[0]) is False
    assert bool(ri_arch.contains_positions(np.array([[1.0, 0.0, 1.1]], dtype=float))[0]) is False
    assert bool(rc_arch.contains_positions(np.array([[1.0, 0.0, 1.0]], dtype=float))[0]) is True
    assert bool(rc_arch.contains_positions(np.array([[-0.75, 0.0, 0.0]], dtype=float))[0]) is False
    assert bool(rc_arch.contains_positions(np.array([[1.0, 1.3, 1.0]], dtype=float))[0]) is False
    assert rbar.goal_range_km == 0.75
    assert rbar.goal_range_tolerance_km is None
    assert rbar.goal_radius_km is None
    assert vbar.max_delta_v_m_s == pytest.approx(1.0)
    assert vbar.max_cross_track_amplitude_km == pytest.approx(0.02)
    assert rbar.max_cross_track_amplitude_km is None
    vbar_initial_dc = vbar_sim_cfg.scenario.objects["chaser"].initial_state["relative_to_target_ric"]["state"][5]
    assert 0.0 < vbar_initial_dc <= 0.001
    assert rbar.approach_gates == ()
    assert _game_camera_mode(rbar_sim_cfg) == "target_pair"
    assert _game_target_centered_plot_planes(rbar_sim_cfg) == ("RI", "RC")
    assert _game_plot_overlays_in_zoom(rbar_sim_cfg) is False
    assert _game_plot_fixed_axis_half_span_km(rbar_sim_cfg) == {}


def test_passive_cross_track_level_uses_intrack_cylinder_forbidden_region() -> None:
    root = Path(__file__).resolve().parents[1] / "game" / "configs"
    sim_cfg = SimulationConfig.from_yaml(root / "game_training_rpo_05_passive_cross_track_approach.yaml")
    cfg = RPOTrainingConfig.from_metadata(
        dict(sim_cfg.scenario.metadata or {})
    )

    assert cfg.scenario_id == "rpo_05_passive_cross_track_approach"
    assert cfg.goal_range_km is None
    assert cfg.goal_range_tolerance_km is None
    assert cfg.max_time_s == 28800.0
    assert len(cfg.inspection_gates) == 4
    assert [gate.name for gate in cfg.inspection_gates] == [
        "left/front RC inspection gate",
        "upper RC inspection gate",
        "aft-right RC inspection gate",
        "lower RC inspection gate",
    ]
    assert np.allclose(cfg.inspection_gates[0].center_ric_km, np.array([0.0, 1.125, -0.75], dtype=float))
    assert np.allclose(cfg.inspection_gates[1].center_ric_km, np.array([0.75, 0.375, 0.0], dtype=float))
    assert np.allclose(cfg.inspection_gates[2].center_ric_km, np.array([0.0, -1.5, 0.75], dtype=float))
    assert np.allclose(cfg.inspection_gates[3].center_ric_km, np.array([-0.75, -0.375, 0.0], dtype=float))
    assert np.allclose(cfg.inspection_gates[0].half_width_ric_km, np.array([0.25, 0.25, 0.25], dtype=float))
    assert np.allclose(cfg.inspection_gates[1].half_width_ric_km, np.array([0.25, 0.25, 0.25], dtype=float))
    assert np.allclose(cfg.inspection_gates[2].half_width_ric_km, np.array([0.25, 0.25, 0.25], dtype=float))
    assert np.allclose(cfg.inspection_gates[3].half_width_ric_km, np.array([0.25, 0.25, 0.25], dtype=float))
    assert {gate.max_total_speed_km_s for gate in cfg.inspection_gates} == {None}
    chaser_state = sim_cfg.scenario.objects["chaser"].initial_state["relative_to_target_ric"]["state"]
    assert np.allclose(chaser_state, np.array([0.0, 3.0, 0.0, 0.0, 0.0, 0.0], dtype=float))
    assert len(cfg.forbidden_regions) == 1
    cylinder = cfg.forbidden_regions[0]
    assert cylinder.kind == "cylinder"
    assert cylinder.axis == "I"
    assert cylinder.radius_km == 0.5
    assert cylinder.height_km == 3.0
    assert cylinder.plot_planes == ("RI", "RC")
    assert bool(cylinder.contains_positions(np.array([[0.25, 0.0, 0.25]], dtype=float))[0]) is True
    assert bool(cylinder.contains_positions(np.array([[0.45, 1.6, 0.0]], dtype=float))[0]) is False
    assert bool(cylinder.contains_positions(np.array([[0.6, 0.0, 0.0]], dtype=float))[0]) is False
    assert bool(cylinder.contains_positions(np.array([[0.0, -0.75, 0.6]], dtype=float))[0]) is False
    assert bool(cylinder.contains_positions(np.array([[0.0, -0.75, 0.5]], dtype=float))[0]) is True


@pytest.mark.parametrize(
    "config_filename",
    [
        "game_training_rpo_01_coast_relative_motion.yaml",
        "game_training_rpo_02_vbar_approach.yaml",
        "game_training_rpo_03_rbar_approach.yaml",
        "game_training_rpo_04_rendezvous.yaml",
        "game_training_rpo_05_passive_cross_track_approach.yaml",
        "game_training_rpo_06_sun_angle_inspection.yaml",
        "game_training_rpo_07_elliptic_burn_then_approach.yaml",
        "game_training_rpo_08_elliptic_nmc.yaml",
        "game_training_rpo_09_elliptic_rendezvous.yaml",
        "game_training_rpo_10_defensive_target_demo.yaml",
        "game_training_rpo_11_evasive_target_survival.yaml",
        "game_training_rpo_sandbox.yaml",
        "game_training_rpo_arcade_pursuit.yaml",
    ],
)
def test_leo_training_sandbox_and_arcade_use_two_rail_timing(config_filename: str) -> None:
    root = Path(__file__).resolve().parents[1] / "game" / "configs"
    sim_cfg = SimulationConfig.from_yaml(root / config_filename)
    speed_options = _game_speed_multiplier_options(sim_cfg)
    burn_state = KeyboardCommandState(pitch=1.0)
    burn_speed = _effective_speed_multiple_for_control(
        sim_cfg,
        1000.0,
        burn_state,
        control_mode=_game_control_mode(sim_cfg),
        options=speed_options,
    )

    assert _game_initial_speed_multiple(sim_cfg, None) == pytest.approx(10.0)
    expected_speed_options = (
        (1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0)
        if config_filename == "game_training_rpo_11_evasive_target_survival.yaml"
        else (1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 500.0, 1000.0)
    )
    assert speed_options == pytest.approx(expected_speed_options)
    assert _game_maneuver_control_speed_multiple(sim_cfg) is None
    assert _game_speed_dt_schedule(sim_cfg) == SPEED_DT_SCHEDULE
    assert _game_coast_speed_dt_schedule(sim_cfg) == SPEED_DT_SCHEDULE
    assert _game_two_rail_speed_control_enabled(sim_cfg) is True
    assert _game_tick_dt_s(sim_cfg, 10.0) == pytest.approx(0.5)
    assert _game_tick_dt_s(sim_cfg, 5.0) == pytest.approx(0.1)
    assert _game_tick_dt_s(sim_cfg, 1000.0) == pytest.approx(1.0)
    assert _game_coast_tick_dt_s(sim_cfg, 1000.0) == pytest.approx(10.0)
    assert burn_speed == pytest.approx(10.0)
    assert _game_active_tick_dt_s(sim_cfg, burn_speed, maneuver_active=True) == pytest.approx(0.5)


def test_two_rail_release_clears_pending_burn_before_coast() -> None:
    root = Path(__file__).resolve().parents[1] / "game" / "configs"
    sim_cfg = SimulationConfig.from_yaml(root / "game_training_rpo_06_sun_angle_inspection.yaml")
    state = KeyboardCommandState(pitch=1.0, use_timing_accumulator=True)
    state.accumulate_timed_input(0.04, speed_multiple=10.0, control_mode="ric_translation")

    assert state.pitch_sim_s > 0.0
    assert _effective_speed_multiple_for_control(
        sim_cfg,
        1000.0,
        state,
        control_mode="ric_translation",
        options=_game_speed_multiplier_options(sim_cfg),
    ) == pytest.approx(10.0)

    state.pitch = 0.0

    assert _clear_two_rail_released_maneuver_input(sim_cfg, state, control_mode="ric_translation") is True
    assert state.pitch_sim_s == pytest.approx(0.0)
    assert state.yaw_sim_s == pytest.approx(0.0)
    assert state.roll_sim_s == pytest.approx(0.0)
    assert _effective_speed_multiple_for_control(
        sim_cfg,
        1000.0,
        state,
        control_mode="ric_translation",
        options=_game_speed_multiplier_options(sim_cfg),
    ) == pytest.approx(1000.0)
    np.testing.assert_allclose(state.consume_ric_duty_cycle(10.0), np.zeros(3, dtype=float))


def test_two_rail_release_clear_is_opt_in() -> None:
    cfg = SimulationConfig.from_dict(_game_config(Path("/tmp")))
    state = KeyboardCommandState(use_timing_accumulator=True)
    state.pitch_sim_s = 0.25

    assert _clear_two_rail_released_maneuver_input(cfg, state, control_mode="ric_translation") is False
    assert state.pitch_sim_s == pytest.approx(0.25)


def test_sun_angle_inspection_level_uses_geo_beam_constraint() -> None:
    root = Path(__file__).resolve().parents[1] / "game" / "configs"
    sim_cfg = SimulationConfig.from_yaml(root / "game_training_rpo_06_sun_angle_inspection.yaml")
    cfg = RPOTrainingConfig.from_metadata(dict(sim_cfg.scenario.metadata or {}))
    safe_sim_cfg = SimulationConfig.from_yaml(root / "game_training_rpo_05_passive_cross_track_approach.yaml")
    safe_cfg = RPOTrainingConfig.from_metadata(dict(safe_sim_cfg.scenario.metadata or {}))

    assert cfg.scenario_id == "rpo_06_sun_angle_inspection"
    assert sim_cfg.scenario.objects["target"].initial_state["coes"]["a_km"] == pytest.approx(42164.169451)
    assert sim_cfg.scenario.simulator.duration_s == pytest.approx(86400.0)
    assert sim_cfg.scenario.simulator.dt_s == pytest.approx(1.0)
    assert sim_cfg.scenario.simulator.dynamics["orbit"]["orbit_substep_s"] == pytest.approx(1.0)
    assert _max_accel_from_config(sim_cfg, "chaser") == pytest.approx(1.0e-5)
    assert _game_initial_speed_multiple(sim_cfg, None) == pytest.approx(10.0)
    assert _game_speed_multiplier_options(sim_cfg) == pytest.approx(
        (1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 500.0, 1000.0)
    )
    assert _game_maneuver_control_speed_multiple(sim_cfg) is None
    assert _game_speed_dt_schedule(sim_cfg) == SPEED_DT_SCHEDULE
    assert _game_coast_speed_dt_schedule(sim_cfg) == SPEED_DT_SCHEDULE
    assert _game_two_rail_speed_control_enabled(sim_cfg) is True
    assert _game_tick_dt_s(sim_cfg, 10.0) == pytest.approx(0.5)
    assert _game_tick_dt_s(sim_cfg, 1000.0) == pytest.approx(1.0)
    assert _game_coast_tick_dt_s(sim_cfg, 1000.0) == pytest.approx(10.0)
    burn_state = KeyboardCommandState(pitch=1.0)
    assert _effective_speed_multiple_for_control(
        sim_cfg,
        1000.0,
        burn_state,
        control_mode="ric_translation",
        options=_game_speed_multiplier_options(sim_cfg),
    ) == pytest.approx(10.0)
    assert _game_active_tick_dt_s(
        sim_cfg,
        _effective_speed_multiple_for_control(
            sim_cfg,
            1000.0,
            burn_state,
            control_mode="ric_translation",
            options=_game_speed_multiplier_options(sim_cfg),
        ),
        maneuver_active=True,
    ) == pytest.approx(0.5)
    assert _game_dashboard_fps_cap(sim_cfg) is None
    assert _dashboard_fps_for_speed(10.0, fps_cap=_game_dashboard_fps_cap(sim_cfg)) == pytest.approx(60.0)
    assert [gate.name for gate in cfg.inspection_gates] == [
        "upper RC inspection gate",
        "lower RC inspection gate",
    ]
    assert np.allclose(cfg.inspection_gates[0].center_ric_km, np.array([0.75, 0.0, 0.0], dtype=float))
    assert np.allclose(cfg.inspection_gates[0].half_width_ric_km, np.array([0.25, 0.75, 0.75], dtype=float))
    assert np.allclose(cfg.inspection_gates[1].center_ric_km, np.array([-0.75, 0.0, 0.0], dtype=float))
    assert np.allclose(cfg.inspection_gates[1].half_width_ric_km, np.array([0.25, 0.75, 0.75], dtype=float))
    assert len(cfg.forbidden_regions) == 1
    assert len(safe_cfg.forbidden_regions) == 1
    sphere = cfg.forbidden_regions[0]
    assert sphere.name == "forbidden proximity sphere"
    assert sphere.kind == "sphere"
    assert sphere.radius_km == pytest.approx(safe_cfg.forbidden_regions[0].radius_km)
    assert sphere.height_km is None
    assert bool(sphere.contains_positions(np.array([[0.25, 0.0, 0.25]], dtype=float))[0]) is True
    assert bool(sphere.contains_positions(np.array([[0.0, 0.75, 0.0]], dtype=float))[0]) is False
    assert bool(sphere.contains_positions(np.array([[0.0, -0.75, 0.0]], dtype=float))[0]) is False
    assert bool(sphere.contains_positions(np.array([[0.0, 0.0, 0.5]], dtype=float))[0]) is True
    assert np.allclose(
        sim_cfg.scenario.objects["chaser"].initial_state["relative_to_target_ric"]["state"],
        safe_sim_cfg.scenario.objects["chaser"].initial_state["relative_to_target_ric"]["state"],
    )
    assert len(cfg.sun_angle_constraints) == 1
    beam = cfg.sun_angle_constraints[0]
    assert beam.name == "GEO anti-Sun inspection beam"
    assert beam.dynamic_sun is True
    assert beam.allowed_center_mode == "anti_sun"
    assert beam.allowed_half_angle_deg == pytest.approx(32.0)
    assert beam.min_range_km == pytest.approx(0.5)
    assert beam.max_range_km == pytest.approx(4.0)
    assert beam.beam_radius_km == pytest.approx(12.0)
    assert bool(beam.samples_satisfying_constraint(np.array([[0.0, -4.0, 0.0]], dtype=float))[0]) is True
    assert bool(beam.samples_satisfying_constraint(np.array([[0.0, 4.0, 0.0]], dtype=float))[0]) is False
    assert float(beam.sun_angles_deg(np.array([[0.0, -4.0, 0.0]], dtype=float))[0]) == pytest.approx(180.0)


def test_sun_angle_inspection_gates_overlap_sun_corridor_during_level_day() -> None:
    root = Path(__file__).resolve().parents[1] / "game" / "configs"
    sim_cfg = SimulationConfig.from_yaml(root / "game_training_rpo_06_sun_angle_inspection.yaml")
    cfg = RPOTrainingConfig.from_metadata(dict(sim_cfg.scenario.metadata or {}))
    constraint = cfg.sun_angle_constraints[0].with_sun_environment(
        {
            **dict(sim_cfg.scenario.simulator.environment or {}),
            "jd_utc_start": sim_cfg.scenario.simulator.initial_jd_utc,
        }
    )
    r_eci_km, v_eci_km_s = coes_mapping_to_rv_eci(sim_cfg.scenario.objects["target"].initial_state["coes"])
    target_state = np.hstack([r_eci_km, v_eci_km_s])
    target_states: list[tuple[float, np.ndarray]] = []
    for time_s in range(0, int(sim_cfg.scenario.simulator.duration_s) + 1, 600):
        target_states.append((float(time_s), target_state.copy()))
        target_state = propagate_two_body_rk4(target_state, 600.0, EARTH_MU_KM3_S2, np.zeros(3, dtype=float))

    for gate in cfg.inspection_gates:
        axes = [
            np.linspace(center - half_width, center + half_width, 5)
            for center, half_width in zip(gate.center_ric_km, gate.half_width_ric_km)
        ]
        samples = np.array(np.meshgrid(*axes, indexing="ij"), dtype=float).reshape(3, -1).T
        assert any(
            bool(
                np.any(
                    constraint.samples_satisfying_constraint(
                        samples,
                        target_state_eci=state,
                        time_s=time_s,
                    )
                )
            )
            for time_s, state in target_states
        ), gate.name


def test_training_tracker_requires_sun_angle_for_gate_completion() -> None:
    root = Path(__file__).resolve().parents[1] / "game" / "configs"
    base = RPOTrainingConfig.from_metadata(
        dict(SimulationConfig.from_yaml(root / "game_training_rpo_06_sun_angle_inspection.yaml").scenario.metadata or {})
    )
    cfg = replace(
        base,
        inspection_gates=(
            game_training.InspectionGateConfig(
                name="sun gate",
                center_ric_km=np.array([0.0, 5.0, 0.0], dtype=float),
                half_width_ric_km=np.array([0.5, 0.5, 0.5], dtype=float),
            ),
        ),
        sun_angle_constraints=(
            game_training.SunAngleConstraintConfig(
                name="static anti-sun beam",
                sun_direction_ric=np.array([0.0, 1.0, 0.0], dtype=float),
                allowed_center_ric=np.array([0.0, -1.0, 0.0], dtype=float),
                allowed_half_angle_deg=20.0,
            ),
        ),
        goal_radius_km=None,
        max_time_s=100.0,
        max_delta_v_m_s=None,
    )
    tracker = RPOTrainingTracker(cfg)
    tracker.t_s = [0.0, 10.0]
    tracker.rel_ric_hist = [
        np.array([0.0, -5.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
        np.array([0.0, 5.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
    ]
    tracker.thrust_hist = [np.zeros(3, dtype=float), np.zeros(3, dtype=float)]
    tracker.target_thrust_hist = [np.zeros(3, dtype=float), np.zeros(3, dtype=float)]
    tracker.mean_motion_hist = [1.0e-4, 1.0e-4]
    tracker._record_inspection_gate_sample(np.array([0.0, 5.0, 0.0, 0.0, 0.0, 0.0], dtype=float))

    score = tracker.score()

    assert score.sun_angle_violation is True
    assert score.sun_angle_constraint_names == ("static anti-sun beam",)
    assert score.inspection_gates_satisfied == 0
    assert score.level_failed is False
    assert score.level_passed is False
    assert score.min_sun_angle_deg == pytest.approx(0.0)


def test_passive_cross_track_level_passes_after_ordered_inspection_gates() -> None:
    root = Path(__file__).resolve().parents[1] / "game" / "configs"
    cfg = RPOTrainingConfig.from_metadata(
        dict(
            SimulationConfig.from_yaml(
                root / "game_training_rpo_05_passive_cross_track_approach.yaml"
            ).scenario.metadata
            or {}
        )
    )
    tracker = RPOTrainingTracker(cfg)
    target_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.54605329, 0.0], dtype=float)
    target = np.hstack((target_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))

    gate_states = (
        np.array([0.0, 1.125, -0.75, 0.0, 0.0001, 0.0], dtype=float),
        np.array([0.75, 0.375, 0.0, 0.0, 0.0001, 0.0], dtype=float),
        np.array([0.0, -1.5, 0.75, 0.0, 0.0001, 0.0], dtype=float),
        np.array([-0.75, -0.375, 0.0, 0.0, 0.0001, 0.0], dtype=float),
    )
    for idx, rel_ric in enumerate(gate_states):
        chaser_state = ric_rect_state_to_eci(rel_ric, target_state[:3], target_state[3:])
        chaser = np.hstack((chaser_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))
        tracker.record(
            snapshot=type(
                "Snapshot",
                (),
                {
                    "time_s": float(idx) * 60.0,
                    "truth": {"target": target, "chaser": chaser},
                    "applied_thrust": {"chaser": np.zeros(3, dtype=float)},
                },
            )()
        )

    score = tracker.score()
    metrics = _mission_metrics(cfg, score)

    assert score.level_passed is True
    assert score.inspection_gates_satisfied == 4
    assert score.inspection_gates_total == 4
    assert score.forbidden_region_violation is False
    assert "OK Inspect 4/4" in metrics


def test_passive_cross_track_level_counts_swept_gate_crossings() -> None:
    root = Path(__file__).resolve().parents[1] / "game" / "configs"
    cfg = RPOTrainingConfig.from_metadata(
        dict(
            SimulationConfig.from_yaml(
                root / "game_training_rpo_05_passive_cross_track_approach.yaml"
            ).scenario.metadata
            or {}
        )
    )
    tracker = RPOTrainingTracker(cfg)
    target_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.54605329, 0.0], dtype=float)
    target = np.hstack((target_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))

    gate_passage_segments = (
        (
            np.array([0.35, 1.125, -0.75, 0.0, 0.0001, 0.0], dtype=float),
            np.array([-0.35, 1.125, -0.75, 0.0, 0.0001, 0.0], dtype=float),
        ),
        (
            np.array([0.75, 0.375, 0.35, 0.0, 0.0001, 0.0], dtype=float),
            np.array([0.75, 0.375, -0.35, 0.0, 0.0001, 0.0], dtype=float),
        ),
        (
            np.array([-0.35, -1.5, 0.75, 0.0, 0.0001, 0.0], dtype=float),
            np.array([0.35, -1.5, 0.75, 0.0, 0.0001, 0.0], dtype=float),
        ),
        (
            np.array([-0.75, -0.375, -0.35, 0.0, 0.0001, 0.0], dtype=float),
            np.array([-0.75, -0.375, 0.35, 0.0, 0.0001, 0.0], dtype=float),
        ),
    )
    idx = 0
    for start_rel_ric, end_rel_ric in gate_passage_segments:
        for rel_ric in (start_rel_ric, end_rel_ric):
            chaser_state = ric_rect_state_to_eci(rel_ric, target_state[:3], target_state[3:])
            chaser = np.hstack((chaser_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))
            tracker.record(
                snapshot=type(
                    "Snapshot",
                    (),
                    {
                        "time_s": float(idx) * 60.0,
                        "truth": {"target": target, "chaser": chaser},
                        "applied_thrust": {"chaser": np.zeros(3, dtype=float)},
                    },
                )()
            )
            idx += 1

    score = tracker.score()

    assert score.level_passed is True
    assert score.inspection_gates_satisfied == 4
    assert score.inspection_gates_total == 4


def test_passive_cross_track_gates_register_without_hidden_speed_limit() -> None:
    root = Path(__file__).resolve().parents[1] / "game" / "configs"
    cfg = RPOTrainingConfig.from_metadata(
        dict(
            SimulationConfig.from_yaml(
                root / "game_training_rpo_05_passive_cross_track_approach.yaml"
            ).scenario.metadata
            or {}
        )
    )
    tracker = RPOTrainingTracker(cfg)
    target_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.54605329, 0.0], dtype=float)
    target = np.hstack((target_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))

    gate_states = (
        np.array([0.0, 1.125, -0.75, 0.0, 0.0020, 0.0], dtype=float),
        np.array([0.75, 0.375, 0.0, 0.0, 0.0020, 0.0], dtype=float),
        np.array([0.0, -1.5, 0.75, 0.0, 0.0020, 0.0], dtype=float),
        np.array([-0.75, -0.375, 0.0, 0.0, 0.0020, 0.0], dtype=float),
    )
    for idx, rel_ric in enumerate(gate_states):
        chaser_state = ric_rect_state_to_eci(rel_ric, target_state[:3], target_state[3:])
        chaser = np.hstack((chaser_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))
        tracker.record(
            snapshot=type(
                "Snapshot",
                (),
                {
                    "time_s": float(idx) * 60.0,
                    "truth": {"target": target, "chaser": chaser},
                    "applied_thrust": {"chaser": np.zeros(3, dtype=float)},
                },
            )()
        )

    score = tracker.score()

    assert score.inspection_gates_satisfied == 4
    assert score.inspection_gates_total == 4


def test_passive_cross_track_gate_progress_is_incremental(monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(__file__).resolve().parents[1] / "game" / "configs"
    cfg = RPOTrainingConfig.from_metadata(
        dict(
            SimulationConfig.from_yaml(
                root / "game_training_rpo_05_passive_cross_track_approach.yaml"
            ).scenario.metadata
            or {}
        )
    )
    tracker = RPOTrainingTracker(cfg)
    target_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.54605329, 0.0], dtype=float)
    target = np.hstack((target_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))

    gate_states = (
        np.array([0.0, 1.125, -0.75, 0.0, 0.0001, 0.0], dtype=float),
        np.array([0.75, 0.375, 0.0, 0.0, 0.0001, 0.0], dtype=float),
        np.array([0.0, -1.5, 0.75, 0.0, 0.0001, 0.0], dtype=float),
        np.array([-0.75, -0.375, 0.0, 0.0, 0.0001, 0.0], dtype=float),
    )
    for idx, rel_ric in enumerate(gate_states):
        chaser_state = ric_rect_state_to_eci(rel_ric, target_state[:3], target_state[3:])
        chaser = np.hstack((chaser_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))
        tracker.record(
            snapshot=type(
                "Snapshot",
                (),
                {
                    "time_s": float(idx) * 60.0,
                    "truth": {"target": target, "chaser": chaser},
                    "applied_thrust": {"chaser": np.zeros(3, dtype=float)},
                },
            )()
        )

    def fail_full_history_gate_scan(*_: object, **__: object) -> dict[str, object]:
        raise AssertionError("score should use incremental inspection gate state")

    monkeypatch.setattr(game_training, "_inspection_gate_status", fail_full_history_gate_scan)

    score = tracker.score()

    assert score.level_passed is True
    assert score.inspection_gates_satisfied == 4
    assert score.inspection_gate_names == tuple(gate.name for gate in cfg.inspection_gates)
    assert tracker.current_hint() == "All inspection gates complete: level should complete."


def test_passive_cross_track_gates_register_out_of_order_passes() -> None:
    root = Path(__file__).resolve().parents[1] / "game" / "configs"
    cfg = RPOTrainingConfig.from_metadata(
        dict(
            SimulationConfig.from_yaml(
                root / "game_training_rpo_05_passive_cross_track_approach.yaml"
            ).scenario.metadata
            or {}
        )
    )
    tracker = RPOTrainingTracker(cfg)
    target_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.54605329, 0.0], dtype=float)
    target = np.hstack((target_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))

    gate_states = (
        np.array([0.0, -1.5, 0.75, 0.0, 0.0001, 0.0], dtype=float),
        np.array([-0.75, -0.375, 0.0, 0.0, 0.0001, 0.0], dtype=float),
        np.array([0.0, 1.125, -0.75, 0.0, 0.0001, 0.0], dtype=float),
        np.array([0.75, 0.375, 0.0, 0.0, 0.0001, 0.0], dtype=float),
    )
    for idx, rel_ric in enumerate(gate_states):
        chaser_state = ric_rect_state_to_eci(rel_ric, target_state[:3], target_state[3:])
        chaser = np.hstack((chaser_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))
        tracker.record(
            snapshot=type(
                "Snapshot",
                (),
                {
                    "time_s": float(idx) * 60.0,
                    "truth": {"target": target, "chaser": chaser},
                    "applied_thrust": {"chaser": np.zeros(3, dtype=float)},
                },
            )()
        )

    score = tracker.score()

    assert score.level_passed is True
    assert score.inspection_gates_satisfied == 4
    assert set(score.inspection_gate_names) == {gate.name for gate in cfg.inspection_gates}


def test_level_four_rendezvous_uses_ten_meter_goal_and_proximity_speed_gate() -> None:
    root = Path(__file__).resolve().parents[1] / "game" / "configs"
    cfg = RPOTrainingConfig.from_metadata(
        dict(SimulationConfig.from_yaml(root / "game_training_rpo_04_rendezvous.yaml").scenario.metadata or {})
    )

    assert cfg.scenario_id == "rpo_04_rendezvous"
    assert cfg.goal_radius_km == pytest.approx(0.01)
    assert cfg.max_goal_speed_km_s == pytest.approx(0.00005)
    assert cfg.hard_speed_limit_radius_km == pytest.approx(0.025)
    assert cfg.hard_speed_limit_km_s == pytest.approx(0.00005)
    assert cfg.max_delta_v_m_s == pytest.approx(1.0)


def test_rbar_metrics_prioritize_range_and_speed_before_advisory_gates() -> None:
    root = Path(__file__).resolve().parents[1] / "game" / "configs"
    rbar = RPOTrainingConfig.from_metadata(
        dict(SimulationConfig.from_yaml(root / "game_training_rpo_03_rbar_approach.yaml").scenario.metadata or {})
    )
    score = type(
        "Score",
        (),
        {
            "elapsed_s": 100.0,
            "approximate_delta_v_m_s": 1.0,
            "final_goal_error_km": 0.2,
            "final_relative_speed_km_s": 0.0004,
            "final_range_km": 0.95,
            "forbidden_region_violation": False,
            "approach_gate_violation": False,
            "approach_gates_satisfied": 0,
            "approach_gates_total": len(rbar.approach_gates),
        },
    )()

    metrics = _mission_metrics(rbar, score)

    range_idx = next(idx for idx, item in enumerate(metrics) if " Range " in item)
    speed_idx = next(idx for idx, item in enumerate(metrics) if " Speed " in item)
    assert range_idx < len(metrics)
    assert speed_idx < len(metrics)
    assert not any(" Gates " in item for item in metrics)
    assert any(" Range " in item for item in metrics[:5])
    assert any(" Speed " in item for item in metrics[:5])


def test_training_tracker_hard_fails_on_forbidden_region_violation() -> None:
    cfg = RPOTrainingConfig(
        enabled=True,
        scenario_id="unit-forbidden",
        goal_radius_km=0.15,
        goal_relative_ric_km=np.array([0.0, -0.75, 0.0], dtype=float),
        max_goal_speed_km_s=0.0003,
        forbidden_regions=(
            ForbiddenRegionConfig(
                name="off-axis test region",
                min_ric_km=np.array([0.4, -5.0, -0.5], dtype=float),
                max_ric_km=np.array([5.0, -0.5, 0.5], dtype=float),
            ),
        ),
    )
    tracker = RPOTrainingTracker(cfg)
    target_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.54605329, 0.0], dtype=float)
    target = np.hstack((target_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))
    for idx, rel_ric in enumerate(
        (
            np.array([0.0, -4.5, 0.0, 0.0, 0.0, 0.0], dtype=float),
            np.array([0.6, -3.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
        )
    ):
        chaser_state = ric_rect_state_to_eci(rel_ric, target_state[:3], target_state[3:])
        chaser = np.hstack((chaser_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))
        tracker.record(
            snapshot=type(
                "Snapshot",
                (),
                {
                    "time_s": float(idx),
                    "truth": {"target": target, "chaser": chaser},
                    "applied_thrust": {"chaser": np.zeros(3, dtype=float)},
                },
            )()
        )

    score = tracker.score()
    metrics = _mission_metrics(cfg, score)

    assert score.level_failed is True
    assert score.forbidden_region_violation is True
    assert score.forbidden_region_names == ("off-axis test region",)
    assert any("Forbidden region" in reason for reason in score.pass_fail_reasons)
    assert "FAIL FR Violated" in metrics


def test_training_tracker_enforces_rbar_approach_gates() -> None:
    gate = ApproachGateConfig(
        name="gate",
        radial_ric_km=-2.0,
        radial_tolerance_km=0.1,
        max_abs_intrack_km=0.25,
        max_abs_cross_track_km=0.2,
        max_abs_radial_rate_km_s=0.0005,
    )
    cfg = RPOTrainingConfig(
        enabled=True,
        scenario_id="unit-rbar-gate",
        goal_radius_km=0.15,
        goal_relative_ric_km=np.array([-0.75, 0.0, 0.0], dtype=float),
        max_goal_speed_km_s=0.0003,
        approach_gates=(gate,),
    )
    target_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.54605329, 0.0], dtype=float)
    target = np.hstack((target_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))

    def score_for(states: tuple[np.ndarray, ...]):
        tracker = RPOTrainingTracker(cfg)
        for idx, rel_ric in enumerate(states):
            chaser_state = ric_rect_state_to_eci(rel_ric, target_state[:3], target_state[3:])
            chaser = np.hstack((chaser_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))
            tracker.record(
                snapshot=type(
                    "Snapshot",
                    (),
                    {
                        "time_s": float(idx),
                        "truth": {"target": target, "chaser": chaser},
                        "applied_thrust": {"chaser": np.zeros(3, dtype=float)},
                    },
                )()
            )
        return tracker.score()

    passed = score_for(
        (
            np.array([-2.0, 0.05, 0.0, 0.0003, 0.0, 0.0], dtype=float),
            np.array([-0.78, 0.0, 0.0, 0.0001, 0.0, 0.0], dtype=float),
        )
    )
    failed = score_for(
        (
            np.array([-2.0, 0.6, 0.0, 0.0003, 0.0, 0.0], dtype=float),
            np.array([-0.78, 0.0, 0.0, 0.0001, 0.0, 0.0], dtype=float),
        )
    )
    not_terminal_yet = score_for((np.array([-2.0, 0.6, 0.0, 0.0003, 0.0, 0.0], dtype=float),))

    assert passed.level_passed is True
    assert passed.approach_gates_satisfied == 1
    assert failed.level_failed is True
    assert failed.approach_gate_violation is True
    assert failed.approach_gate_names == ("gate",)
    assert any("R-bar approach gate" in reason for reason in failed.pass_fail_reasons)
    assert not_terminal_yet.level_failed is False
    assert not_terminal_yet.approach_gate_violation is False


def test_training_tracker_counts_gate_hit_on_terminal_goal_sample() -> None:
    gate = ApproachGateConfig(
        name="terminal gate",
        radial_ric_km=-0.75,
        radial_tolerance_km=0.05,
        max_abs_intrack_km=0.1,
        max_abs_cross_track_km=0.1,
        max_total_speed_km_s=0.0003,
    )
    cfg = RPOTrainingConfig(
        enabled=True,
        scenario_id="unit-terminal-gate",
        goal_radius_km=0.02,
        goal_relative_ric_km=np.array([-0.75, 0.0, 0.0], dtype=float),
        max_goal_speed_km_s=0.0003,
        approach_gates=(gate,),
    )
    tracker = RPOTrainingTracker(cfg)
    target_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.54605329, 0.0], dtype=float)
    target = np.hstack((target_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))

    for idx, rel_ric in enumerate(
        (
            np.array([-1.2, 0.0, 0.0, 0.0001, 0.0, 0.0], dtype=float),
            np.array([-0.75, 0.0, 0.0, 0.0001, 0.0, 0.0], dtype=float),
        )
    ):
        chaser_state = ric_rect_state_to_eci(rel_ric, target_state[:3], target_state[3:])
        chaser = np.hstack((chaser_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))
        tracker.record(
            snapshot=type(
                "Snapshot",
                (),
                {
                    "time_s": float(idx),
                    "truth": {"target": target, "chaser": chaser},
                    "applied_thrust": {"chaser": np.zeros(3, dtype=float)},
                },
            )()
        )

    score = tracker.score()

    assert score.level_passed is True
    assert score.approach_gates_satisfied == 1
    assert score.approach_gate_violation is False


def test_training_tracker_allows_advisory_approach_gate_misses() -> None:
    gate = ApproachGateConfig(
        name="advisory gate",
        radial_ric_km=-2.0,
        radial_tolerance_km=0.1,
        max_abs_intrack_km=0.25,
        required=False,
    )
    cfg = RPOTrainingConfig(
        enabled=True,
        scenario_id="unit-advisory-rbar-gate",
        goal_radius_km=0.15,
        goal_relative_ric_km=np.array([-0.75, 0.0, 0.0], dtype=float),
        max_goal_speed_km_s=0.0003,
        approach_gates=(gate,),
    )
    tracker = RPOTrainingTracker(cfg)
    target_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.54605329, 0.0], dtype=float)
    target = np.hstack((target_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))

    for idx, rel_ric in enumerate(
        (
            np.array([-3.0, 0.8, 0.0, 0.0003, 0.0, 0.0], dtype=float),
            np.array([-0.78, 0.0, 0.0, 0.0001, 0.0, 0.0], dtype=float),
        )
    ):
        chaser_state = ric_rect_state_to_eci(rel_ric, target_state[:3], target_state[3:])
        chaser = np.hstack((chaser_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))
        tracker.record(
            snapshot=type(
                "Snapshot",
                (),
                {
                    "time_s": float(idx),
                    "truth": {"target": target, "chaser": chaser},
                    "applied_thrust": {"chaser": np.zeros(3, dtype=float)},
                },
            )()
        )

    score = tracker.score()

    assert score.level_passed is True
    assert score.approach_gates_satisfied == 0
    assert score.approach_gate_violation is False
    assert score.approach_gate_names == ()


def test_training_tracker_range_goal_passes_at_target_range_with_low_speed() -> None:
    cfg = RPOTrainingConfig(
        enabled=True,
        scenario_id="unit-range-goal",
        goal_range_km=0.75,
        max_goal_speed_km_s=0.0003,
    )
    tracker = RPOTrainingTracker(cfg)
    target_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.54605329, 0.0], dtype=float)
    rel_ric = np.array([-0.45, -0.60, 0.0, 0.0001, 0.0, 0.0], dtype=float)
    chaser_state = ric_rect_state_to_eci(rel_ric, target_state[:3], target_state[3:])
    target = np.hstack((target_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))
    chaser = np.hstack((chaser_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))
    tracker.record(
        snapshot=type(
            "Snapshot",
            (),
            {
                "time_s": 10.0,
                "truth": {"target": target, "chaser": chaser},
                "applied_thrust": {"chaser": np.zeros(3, dtype=float)},
            },
        )()
    )

    score = tracker.score()
    metrics = _mission_metrics(cfg, score)

    assert score.level_passed is True
    assert score.final_range_km == pytest.approx(0.75)
    assert score.final_goal_error_km == pytest.approx(0.0)
    assert "OK Range 750.0 m/750.0 m" in metrics


def test_training_tracker_range_goal_fails_inside_keepout() -> None:
    cfg = RPOTrainingConfig(
        enabled=True,
        scenario_id="unit-range-keepout",
        keepout_radius_km=0.25,
        goal_range_km=0.75,
        max_goal_speed_km_s=0.0003,
    )
    tracker = RPOTrainingTracker(cfg)
    target_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.54605329, 0.0], dtype=float)
    rel_ric = np.array([-0.20, 0.0, 0.0, 0.0001, 0.0, 0.0], dtype=float)
    chaser_state = ric_rect_state_to_eci(rel_ric, target_state[:3], target_state[3:])
    target = np.hstack((target_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))
    chaser = np.hstack((chaser_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))
    tracker.record(
        snapshot=type(
            "Snapshot",
            (),
            {
                "time_s": 10.0,
                "truth": {"target": target, "chaser": chaser},
                "applied_thrust": {"chaser": np.zeros(3, dtype=float)},
            },
        )()
    )

    score = tracker.score()

    assert score.level_passed is False
    assert score.level_failed is True
    assert score.keepout_violation is True


def test_training_tracker_survival_goal_passes_at_time_without_keepout_violation() -> None:
    cfg = RPOTrainingConfig(
        enabled=True,
        scenario_id="unit-survival",
        keepout_radius_km=0.1,
        survival_goal=True,
        max_time_s=6000.0,
    )
    tracker = RPOTrainingTracker(cfg)
    target_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.54605329, 0.0], dtype=float)
    target = np.hstack((target_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))
    rel_ric = np.array([0.0, -0.2, 0.0, 0.0, 0.0, 0.0], dtype=float)
    chaser_state = ric_rect_state_to_eci(rel_ric, target_state[:3], target_state[3:])
    chaser = np.hstack((chaser_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))

    for t_s in (0.0, 6000.0):
        tracker.record(
            snapshot=type(
                "Snapshot",
                (),
                {
                    "time_s": t_s,
                    "truth": {"target": target, "chaser": chaser},
                    "applied_thrust": {"chaser": np.zeros(3, dtype=float)},
                },
            )()
        )

    score = tracker.score()

    assert score.level_passed is True
    assert score.achieved_time_s == pytest.approx(6000.0)
    assert score.keepout_violation is False


def test_training_tracker_survival_goal_fails_on_keepout_violation() -> None:
    cfg = RPOTrainingConfig(
        enabled=True,
        scenario_id="unit-survival-fail",
        keepout_radius_km=0.1,
        survival_goal=True,
        max_time_s=6000.0,
    )
    tracker = RPOTrainingTracker(cfg)
    target_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.54605329, 0.0], dtype=float)
    target = np.hstack((target_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))
    rel_ric = np.array([0.0, -0.05, 0.0, 0.0, 0.0, 0.0], dtype=float)
    chaser_state = ric_rect_state_to_eci(rel_ric, target_state[:3], target_state[3:])
    chaser = np.hstack((chaser_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))

    for t_s in (0.0, 6000.0):
        tracker.record(
            snapshot=type(
                "Snapshot",
                (),
                {
                    "time_s": t_s,
                    "truth": {"target": target, "chaser": chaser},
                    "applied_thrust": {"chaser": np.zeros(3, dtype=float)},
                },
            )()
        )

    score = tracker.score()

    assert score.level_passed is False
    assert score.level_failed is True
    assert score.keepout_violation is True


def test_training_tracker_survival_goal_enforces_target_delta_v_budget() -> None:
    cfg = RPOTrainingConfig(
        enabled=True,
        scenario_id="unit-survival-target-dv",
        keepout_radius_km=0.1,
        survival_goal=True,
        max_time_s=10.0,
        max_target_delta_v_m_s=1.0,
    )
    tracker = RPOTrainingTracker(cfg)
    target_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.54605329, 0.0], dtype=float)
    target = np.hstack((target_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))
    rel_ric = np.array([0.0, -0.2, 0.0, 0.0, 0.0, 0.0], dtype=float)
    chaser_state = ric_rect_state_to_eci(rel_ric, target_state[:3], target_state[3:])
    chaser = np.hstack((chaser_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))

    for t_s in (0.0, 10.0):
        tracker.record(
            snapshot=type(
                "Snapshot",
                (),
                {
                    "time_s": t_s,
                    "truth": {"target": target, "chaser": chaser},
                    "applied_thrust": {
                        "target": np.array([0.0002, 0.0, 0.0], dtype=float),
                        "chaser": np.zeros(3, dtype=float),
                    },
                },
            )()
        )

    score = tracker.score()

    assert score.target_delta_v_m_s == pytest.approx(2.0)
    assert score.level_passed is False
    assert score.level_failed is True
    assert any("Target delta-v budget exceeded" in reason for reason in score.pass_fail_reasons)


def test_training_tracker_survival_goal_enforces_target_reference_range() -> None:
    cfg = RPOTrainingConfig(
        enabled=True,
        scenario_id="unit-survival-target-reference-range",
        keepout_radius_km=0.1,
        survival_goal=True,
        max_time_s=10.0,
        target_reference_object_id="target_reference",
        max_target_reference_range_km=1.0,
    )
    tracker = RPOTrainingTracker(cfg)
    reference_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.54605329, 0.0], dtype=float)
    reference = np.hstack((reference_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))
    target_rel_ric = np.array([0.0, 1.1, 0.0, 0.0, 0.0, 0.0], dtype=float)
    target_state = ric_rect_state_to_eci(target_rel_ric, reference_state[:3], reference_state[3:])
    target = np.hstack((target_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))
    chaser_rel_ric = np.array([0.0, -0.2, 0.0, 0.0, 0.0, 0.0], dtype=float)
    chaser_state = ric_rect_state_to_eci(chaser_rel_ric, target_state[:3], target_state[3:])
    chaser = np.hstack((chaser_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))

    for t_s in (0.0, 10.0):
        tracker.record(
            snapshot=type(
                "Snapshot",
                (),
                {
                    "time_s": t_s,
                    "truth": {"target_reference": reference, "target": target, "chaser": chaser},
                    "applied_thrust": {"target": np.zeros(3, dtype=float), "chaser": np.zeros(3, dtype=float)},
                },
            )()
        )

    score = tracker.score()

    assert score.final_target_reference_range_km == pytest.approx(1.1)
    assert score.target_reference_range_violation is True
    assert score.level_passed is False
    assert score.level_failed is True
    assert any("Mission-capable radius exceeded" in reason for reason in score.pass_fail_reasons)


def test_training_tracker_target_reference_range_is_three_dimensional() -> None:
    cfg = RPOTrainingConfig(
        enabled=True,
        scenario_id="unit-survival-target-reference-range-3d",
        keepout_radius_km=0.1,
        survival_goal=True,
        max_time_s=10.0,
        target_reference_object_id="target_reference",
        max_target_reference_range_km=1.0,
    )
    tracker = RPOTrainingTracker(cfg)
    reference_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.54605329, 0.0], dtype=float)
    reference = np.hstack((reference_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))
    target_rel_ric = np.array([0.0, 0.8, 0.8, 0.0, 0.0, 0.0], dtype=float)
    target_state = ric_rect_state_to_eci(target_rel_ric, reference_state[:3], reference_state[3:])
    target = np.hstack((target_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))
    chaser_rel_ric = np.array([0.0, -0.2, 0.0, 0.0, 0.0, 0.0], dtype=float)
    chaser_state = ric_rect_state_to_eci(chaser_rel_ric, target_state[:3], target_state[3:])
    chaser = np.hstack((chaser_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))

    tracker.record(
        snapshot=type(
            "Snapshot",
            (),
            {
                "time_s": 10.0,
                "truth": {"target_reference": reference, "target": target, "chaser": chaser},
                "applied_thrust": {"target": np.zeros(3, dtype=float), "chaser": np.zeros(3, dtype=float)},
            },
        )()
    )

    score = tracker.score()

    assert np.linalg.norm(target_rel_ric[[0, 1]]) < 1.0
    assert np.linalg.norm(target_rel_ric[[0, 2]]) < 1.0
    assert score.final_target_reference_range_km == pytest.approx(np.sqrt(0.8**2 + 0.8**2))
    assert score.target_reference_range_violation is True


def test_dashboard_tracks_mission_capable_reference_separately_from_camera_reference() -> None:
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.target_object_id = "target"
    dashboard.chaser_object_id = "chaser"
    dashboard.reference_object_id = "camera_reference"
    dashboard.target_reference_object_id = "target_reference"
    dashboard.relative_frame = "ric"
    dashboard.coast_prediction_model = "hcw"
    dashboard.max_history = 10
    dashboard.t_s = []
    dashboard.rel_hist = []
    dashboard.target_rel_hist = []
    dashboard.target_reference_rel_hist = []
    dashboard.thrust_hist = []
    dashboard.thrust_ric_hist = []
    dashboard._rel_array = np.zeros((0, 6), dtype=float)
    dashboard._target_rel_array = np.zeros((0, 6), dtype=float)
    dashboard._target_reference_rel_array = np.zeros((0, 6), dtype=float)
    dashboard._thrust_ric_array = np.zeros((0, 3), dtype=float)
    dashboard.target_true_anomaly_deg = None
    camera_reference_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.54605329, 0.0], dtype=float)
    camera_reference = np.hstack(
        (camera_reference_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0]))
    )
    target_reference_rel_ric = np.array([0.0, 0.4, 0.3, 0.0, 0.0, 0.0], dtype=float)
    target_reference_state = ric_rect_state_to_eci(
        target_reference_rel_ric,
        camera_reference_state[:3],
        camera_reference_state[3:],
    )
    target_reference = np.hstack(
        (target_reference_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0]))
    )
    target_rel_to_reference = np.array([0.0, 0.8, 0.0, 0.0, 0.0, 0.0], dtype=float)
    target_state = ric_rect_state_to_eci(
        target_rel_to_reference,
        target_reference_state[:3],
        target_reference_state[3:],
    )
    target = np.hstack((target_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))
    chaser_rel_to_target = np.array([0.0, -0.2, 0.0, 0.0, 0.0, 0.0], dtype=float)
    chaser_state = ric_rect_state_to_eci(chaser_rel_to_target, target_state[:3], target_state[3:])
    chaser = np.hstack((chaser_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))

    dashboard.push_snapshot(
        type(
            "Snapshot",
            (),
            {
                "time_s": 0.0,
                "truth": {
                    "camera_reference": camera_reference,
                    "target_reference": target_reference,
                    "target": target,
                    "chaser": chaser,
                },
                "applied_thrust": {"chaser": np.zeros(3, dtype=float)},
            },
        )()
    )

    assert dashboard.target_reference_rel_hist[-1][:3] == pytest.approx(target_reference_rel_ric[:3])
    assert not np.allclose(dashboard.target_rel_hist[-1][:3], target_reference_rel_ric[:3])


def test_training_tracker_survival_goal_can_ignore_chaser_delta_v_budget_failure() -> None:
    cfg = RPOTrainingConfig(
        enabled=True,
        scenario_id="unit-survival-chaser-dv",
        keepout_radius_km=0.1,
        survival_goal=True,
        max_time_s=10.0,
        max_delta_v_m_s=5.0,
        fail_on_delta_v_budget=False,
    )
    tracker = RPOTrainingTracker(cfg)
    target_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.54605329, 0.0], dtype=float)
    target = np.hstack((target_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))
    rel_ric = np.array([0.0, -0.2, 0.0, 0.0, 0.0, 0.0], dtype=float)
    chaser_state = ric_rect_state_to_eci(rel_ric, target_state[:3], target_state[3:])
    chaser = np.hstack((chaser_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))

    for t_s in (0.0, 10.0):
        tracker.record(
            snapshot=type(
                "Snapshot",
                (),
                {
                    "time_s": t_s,
                    "truth": {"target": target, "chaser": chaser},
                    "applied_thrust": {
                        "target": np.zeros(3, dtype=float),
                        "chaser": np.array([0.0006, 0.0, 0.0], dtype=float),
                    },
                },
            )()
        )

    score = tracker.score()

    assert score.approximate_delta_v_m_s == pytest.approx(6.0)
    assert score.level_passed is True
    assert score.level_failed is False
    assert not any("Delta-v budget exceeded" in reason for reason in score.pass_fail_reasons)


def test_training_tracker_hint_calls_out_hold_box_speed_blocker() -> None:
    cfg = RPOTrainingConfig(
        enabled=True,
        scenario_id="unit-hint",
        goal_radius_km=0.15,
        goal_relative_ric_km=np.array([-0.75, 0.0, 0.0], dtype=float),
        max_goal_speed_km_s=0.0003,
    )
    tracker = RPOTrainingTracker(cfg)
    tracker.rel_ric_hist.append(np.array([-0.75, 0.0, 0.0, 0.0005, 0.0, 0.0], dtype=float))

    assert tracker.current_hint() == "Inside hold box: slow below 300.0 mm/s to finish."


def test_training_tracker_hint_calls_out_range_goal_speed_blocker() -> None:
    cfg = RPOTrainingConfig(
        enabled=True,
        scenario_id="unit-range-hint",
        goal_range_km=0.75,
        max_goal_speed_km_s=0.0003,
    )
    tracker = RPOTrainingTracker(cfg)
    tracker.rel_ric_hist.append(np.array([-0.45, -0.60, 0.0, 0.0005, 0.0, 0.0], dtype=float))

    assert tracker.current_hint() == "Inside green circle: slow below 300.0 mm/s to finish."


def test_training_tracker_score_cache_reuses_score_until_new_sample() -> None:
    cfg = RPOTrainingConfig(
        enabled=True,
        scenario_id="unit-cache",
        goal_range_km=1.0,
    )
    tracker = RPOTrainingTracker(cfg)

    empty_score = tracker.score()
    assert tracker.score() is empty_score

    target_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.54605329, 0.0], dtype=float)
    rel_ric = np.array([1.0, 0.0, 0.0, 0.0001, 0.0, 0.0], dtype=float)
    chaser_state = ric_rect_state_to_eci(rel_ric, target_state[:3], target_state[3:])
    target = np.hstack((target_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))
    chaser = np.hstack((chaser_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))
    tracker.record(
        snapshot=type(
            "Snapshot",
            (),
            {
                "time_s": 10.0,
                "truth": {"target": target, "chaser": chaser},
                "applied_thrust": {"chaser": np.zeros(3, dtype=float)},
            },
        )()
    )

    score = tracker.score()
    assert score is tracker.score()
    assert score is not empty_score
    assert score.samples == 1


def test_training_tracker_rendezvous_metrics_use_close_approach_units() -> None:
    cfg = RPOTrainingConfig(
        enabled=True,
        scenario_id="unit-rendezvous",
        learning_goal="test",
        goal_radius_km=0.025,
        goal_relative_ric_km=np.zeros(3, dtype=float),
        max_goal_speed_km_s=0.001,
    )
    tracker = RPOTrainingTracker(cfg)
    target_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.54605329, 0.0], dtype=float)
    rel_ric = np.array([0.02, 0.0, 0.0, 0.0, 0.0005, 0.0], dtype=float)
    chaser_state = ric_rect_state_to_eci(rel_ric, target_state[:3], target_state[3:])
    target = np.hstack((target_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))
    chaser = np.hstack((chaser_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))
    tracker.record(
        snapshot=type(
            "Snapshot",
            (),
            {
                "time_s": 10.0,
                "truth": {"target": target, "chaser": chaser},
                "applied_thrust": {"chaser": np.zeros(3, dtype=float)},
            },
        )()
    )

    score = tracker.score()
    metrics = _mission_metrics(cfg, score)

    assert score.level_passed is True
    assert "WARN Goal 20.00 m/25.00 m" in metrics
    assert "OK Speed 500.0 mm/s/1.000 m/s" in metrics


def test_training_tracker_fails_fast_rendezvous_inside_proximity_gate() -> None:
    cfg = RPOTrainingConfig(
        enabled=True,
        scenario_id="unit-rendezvous-hard-speed",
        learning_goal="test",
        goal_radius_km=0.01,
        goal_relative_ric_km=np.zeros(3, dtype=float),
        max_goal_speed_km_s=0.0001,
        hard_speed_limit_radius_km=0.025,
        hard_speed_limit_km_s=0.0001,
    )
    tracker = RPOTrainingTracker(cfg)
    target_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.54605329, 0.0], dtype=float)
    rel_ric = np.array([0.02, 0.0, 0.0, 0.0, 0.0002, 0.0], dtype=float)
    chaser_state = ric_rect_state_to_eci(rel_ric, target_state[:3], target_state[3:])
    target = np.hstack((target_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))
    chaser = np.hstack((chaser_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))
    tracker.record(
        snapshot=type(
            "Snapshot",
            (),
            {
                "time_s": 10.0,
                "truth": {"target": target, "chaser": chaser},
                "applied_thrust": {"chaser": np.zeros(3, dtype=float)},
            },
        )()
    )

    score = tracker.score()
    metrics = _mission_metrics(cfg, score)

    assert score.level_failed is True
    assert score.hard_speed_limit_violation is True
    assert any("Hard speed limit" in reason for reason in score.pass_fail_reasons)
    assert "FAIL Prox V <= 100.0 mm/s" in metrics


def test_training_tracker_fails_fast_rendezvous_swept_through_proximity_gate() -> None:
    cfg = RPOTrainingConfig(
        enabled=True,
        scenario_id="unit-rendezvous-hard-speed-sweep",
        learning_goal="test",
        goal_radius_km=0.01,
        goal_relative_ric_km=np.zeros(3, dtype=float),
        max_goal_speed_km_s=0.0001,
        hard_speed_limit_radius_km=0.025,
        hard_speed_limit_km_s=0.0001,
    )
    tracker = RPOTrainingTracker(cfg)
    target_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.54605329, 0.0], dtype=float)
    target = np.hstack((target_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))
    for t_s, rel_ric in (
        (0.0, np.array([0.03, 0.0, 0.0, -0.0002, 0.0, 0.0], dtype=float)),
        (10.0, np.array([-0.03, 0.0, 0.0, -0.0002, 0.0, 0.0], dtype=float)),
    ):
        chaser_state = ric_rect_state_to_eci(rel_ric, target_state[:3], target_state[3:])
        chaser = np.hstack((chaser_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))
        tracker.record(
            snapshot=type(
                "Snapshot",
                (),
                {
                    "time_s": t_s,
                    "truth": {"target": target, "chaser": chaser},
                    "applied_thrust": {"chaser": np.zeros(3, dtype=float)},
                },
            )()
        )

    score = tracker.score()

    assert score.level_failed is True
    assert score.hard_speed_limit_violation is True


def test_score_debrief_lines_show_after_terminal_mission_state() -> None:
    cfg = RPOTrainingConfig(
        enabled=True,
        scenario_id="unit-debrief",
        goal_radius_km=0.025,
        goal_relative_ric_km=np.zeros(3, dtype=float),
        max_goal_speed_km_s=0.001,
        max_time_s=100.0,
        max_delta_v_m_s=2.0,
    )
    tracker = RPOTrainingTracker(cfg)
    target_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.54605329, 0.0], dtype=float)
    rel_ric = np.array([0.02, 0.0, 0.0, 0.0, 0.0005, 0.0], dtype=float)
    chaser_state = ric_rect_state_to_eci(rel_ric, target_state[:3], target_state[3:])
    target = np.hstack((target_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))
    chaser = np.hstack((chaser_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))
    tracker.record(
        snapshot=type(
            "Snapshot",
            (),
            {
                "time_s": 10.0,
                "truth": {"target": target, "chaser": chaser},
                "applied_thrust": {"chaser": np.zeros(3, dtype=float)},
            },
        )()
    )

    score = tracker.score()
    debrief = _score_debrief_lines(score, config=cfg, difficulty="medium")

    assert score.level_passed is True
    assert any("Scenario" in line and "unit-debrief" in line for line in debrief)
    assert any("Score" in line and "4,200" in line for line in debrief)
    assert any(line.startswith("Final Range") for line in debrief)
    assert any(line.startswith("Final Speed") for line in debrief)


def test_score_debrief_lines_preserve_terminal_failure_details() -> None:
    cfg = RPOTrainingConfig(enabled=True, scenario_id="unit-failure")
    score = type(
        "Score",
        (),
        {
            "level_passed": False,
            "level_failed": True,
            "scenario_id": "unit-failure",
            "elapsed_s": 42.0,
            "final_range_km": 5.0,
            "final_goal_error_km": 4.0,
            "final_relative_speed_km_s": 0.002,
            "approximate_delta_v_m_s": 0.5,
            "pass_fail_reasons": (
                "Forbidden region violated: upper radial rail of V-bar U, lower radial floor in RC.",
                "Delta-v budget exceeded (1.0 m/s).",
                "Time budget exceeded (18000 s).",
            ),
        },
    )()

    debrief = _score_debrief_lines(score, config=cfg, difficulty="medium")

    assert any("Closest App" in line for line in debrief)
    assert any("Keepout Time" in line for line in debrief)
    assert any("Target dV" in line for line in debrief)
    assert any("upper radial rail" in line for line in debrief)
    assert any("Delta-v budget exceeded" in line for line in debrief)
    assert any("Time budget exceeded" in line for line in debrief)


def test_training_tracker_hard_fails_on_keepout_or_expired_time() -> None:
    cfg = RPOTrainingConfig(
        enabled=True,
        scenario_id="unit-fail",
        keepout_radius_km=0.25,
        goal_nmt_radial_amplitude_km=1.5,
        goal_nmt_cross_track_amplitude_km=1.0,
        goal_nmt_element_tolerance_km=0.05,
        max_time_s=1.0,
        max_delta_v_m_s=1.0,
    )
    tracker = RPOTrainingTracker(cfg)
    target_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.54605329, 0.0], dtype=float)
    target = np.hstack((target_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))
    for idx, rel_ric in enumerate((np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0]), np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0]))):
        chaser_state = ric_rect_state_to_eci(rel_ric, target_state[:3], target_state[3:])
        chaser = np.hstack((chaser_state, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])))
        tracker.record(
            snapshot=type(
                "Snapshot",
                (),
                {
                    "time_s": float(idx + 1),
                    "truth": {"target": target, "chaser": chaser},
                    "applied_thrust": {"chaser": np.zeros(3, dtype=float)},
                },
            )()
        )

    score = tracker.score()

    assert score.level_failed is True
    assert score.level_passed is False
    assert any("Keepout" in reason for reason in score.pass_fail_reasons)
    assert any("Time budget" in reason for reason in score.pass_fail_reasons)
    assert _mission_metrics(cfg, score)


def test_pygame_input_mapping_sets_ric_axes_and_quit() -> None:
    class FakeEvent:
        def __init__(self, type_value, key=None, *, unicode="", repeat=False):
            self.type = type_value
            self.key = key
            self.unicode = unicode
            self.repeat = repeat

    class FakeKeys:
        def __getitem__(self, key):
            return key in {"w", "d", "right", "space"}

    class FakePygame:
        QUIT = "quit"
        KEYDOWN = "keydown"
        KEYUP = "keyup"
        MOUSEWHEEL = "mousewheel"
        K_ESCAPE = "escape"
        K_r = "r"
        K_PERIOD = "."
        K_m = "m"
        K_c = "c"
        K_o = "o"
        K_p = "p"
        K_g = "g"
        K_F9 = "f9"
        K_RETURN = "return"
        K_KP_ENTER = "kp_enter"
        K_PAGEUP = "pageup"
        K_PAGEDOWN = "pagedown"
        K_HOME = "home"
        K_END = "end"
        K_w = "w"
        K_s = "s"
        K_d = "d"
        K_a = "a"
        K_RIGHT = "right"
        K_LEFT = "left"
        K_UP = "up"
        K_DOWN = "down"
        K_PLUS = "plus"
        K_EQUALS = "equals"
        K_MINUS = "minus"
        K_KP_PLUS = "kp_plus"
        K_KP_MINUS = "kp_minus"
        K_SPACE = "space"

        class event:
            @staticmethod
            def get():
                return []

        class key:
            @staticmethod
            def get_pressed():
                return FakeKeys()

    state = KeyboardCommandState()

    _poll_pygame_input(FakePygame, state, control_mode="ric_translation")

    assert state.pitch == 1.0
    assert state.yaw == 1.0
    assert state.roll == 1.0
    assert state.firing is False

    class QuitPygame(FakePygame):
        class event:
            @staticmethod
            def get():
                return [FakeEvent(FakePygame.KEYDOWN, FakePygame.K_ESCAPE)]

    _poll_pygame_input(QuitPygame, state, control_mode="ric_translation")

    assert state.quit_requested is True

    class PauseSpeedPygame(FakePygame):
        class event:
            @staticmethod
            def get():
                return [
                    FakeEvent(FakePygame.KEYDOWN, FakePygame.K_SPACE),
                    FakeEvent(FakePygame.KEYDOWN, FakePygame.K_PERIOD),
                    FakeEvent(FakePygame.KEYDOWN, FakePygame.K_UP),
                    FakeEvent(FakePygame.KEYDOWN, FakePygame.K_r),
                    FakeEvent(FakePygame.KEYDOWN, FakePygame.K_m),
                    FakeEvent(FakePygame.KEYDOWN, FakePygame.K_d),
                    FakeEvent(FakePygame.KEYDOWN, FakePygame.K_c),
                    FakeEvent(FakePygame.KEYDOWN, FakePygame.K_o),
                    FakeEvent(FakePygame.KEYDOWN, FakePygame.K_p),
                    FakeEvent(FakePygame.KEYDOWN, FakePygame.K_g),
                    FakeEvent(FakePygame.KEYDOWN, FakePygame.K_RETURN),
                ]

    state = KeyboardCommandState()

    _poll_pygame_input(PauseSpeedPygame, state, control_mode="ric_translation")

    assert state.paused is True
    assert not hasattr(state, "step_requested")
    assert state.speed_multiplier_change == 1
    assert state.restart_requested is True
    assert state.music_toggle_requested is True
    assert state.open_debrief_requested is True
    assert state.camera_rule_toggle_requested is True
    assert state.eci_ri_plot_toggle_requested is True
    assert state.eci_rc_plot_toggle_requested is True
    assert state.clip_record_toggle_requested is True
    assert state.clip_record_save_requested is True

    class SlowDownPygame(FakePygame):
        class event:
            @staticmethod
            def get():
                return [FakeEvent(FakePygame.KEYDOWN, FakePygame.K_DOWN)]

    state = KeyboardCommandState()

    _poll_pygame_input(SlowDownPygame, state, control_mode="ric_translation")

    assert state.speed_multiplier_change == -1

    class PlusMinusPygame(FakePygame):
        class event:
            @staticmethod
            def get():
                return [
                    FakeEvent(FakePygame.KEYDOWN, FakePygame.K_EQUALS, unicode="+"),
                    FakeEvent(FakePygame.KEYDOWN, FakePygame.K_PLUS, repeat=True),
                ]

    state = KeyboardCommandState()

    _poll_pygame_input(PlusMinusPygame, state, control_mode="ric_translation")

    assert state.speed_multiplier_change == 1

    class ReleasedPlusPygame(FakePygame):
        class event:
            @staticmethod
            def get():
                return [
                    FakeEvent(FakePygame.KEYUP, FakePygame.K_EQUALS, unicode="+"),
                    FakeEvent(FakePygame.KEYDOWN, FakePygame.K_KP_PLUS),
                ]

    _poll_pygame_input(ReleasedPlusPygame, state, control_mode="ric_translation")

    assert state.speed_multiplier_change == 1

    class RepeatedMinusPygame(FakePygame):
        class event:
            @staticmethod
            def get():
                return [
                    FakeEvent(FakePygame.KEYDOWN, FakePygame.K_MINUS, unicode="-"),
                    FakeEvent(FakePygame.KEYDOWN, FakePygame.K_KP_MINUS, repeat=True),
                ]

    state = KeyboardCommandState()

    _poll_pygame_input(RepeatedMinusPygame, state, control_mode="ric_translation")

    assert state.speed_multiplier_change == -1

    class BurstPlusPygame(FakePygame):
        class event:
            @staticmethod
            def get():
                return [
                    FakeEvent(FakePygame.KEYDOWN, FakePygame.K_EQUALS, unicode="+"),
                    FakeEvent(FakePygame.KEYUP, FakePygame.K_EQUALS, unicode="+"),
                    FakeEvent(FakePygame.KEYDOWN, FakePygame.K_EQUALS, unicode="+"),
                    FakeEvent(FakePygame.KEYUP, FakePygame.K_EQUALS, unicode="+"),
                    FakeEvent(FakePygame.KEYDOWN, FakePygame.K_EQUALS, unicode="+"),
                ]

    state = KeyboardCommandState()

    _poll_pygame_input(BurstPlusPygame, state, control_mode="ric_translation")

    assert state.speed_multiplier_change == 1

    class NoHeldKeys:
        def __getitem__(self, key):
            return False

    class QuickBurnTapPygame(FakePygame):
        class event:
            @staticmethod
            def get():
                return [
                    FakeEvent(FakePygame.KEYDOWN, FakePygame.K_d),
                    FakeEvent(FakePygame.KEYUP, FakePygame.K_d),
                ]

        class key:
            @staticmethod
            def get_pressed():
                return NoHeldKeys()

    state = KeyboardCommandState()

    _poll_pygame_input(QuickBurnTapPygame, state, control_mode="ric_translation")

    assert state.pitch == 0.0
    assert state.yaw == 1.0
    assert state.roll == 0.0
    assert state.yaw_event_pulse is True
    state.clear_event_pulses()
    assert state.yaw == 0.0
    assert state.yaw_event_pulse is False

    class BriefingScrollPygame(FakePygame):
        class event:
            @staticmethod
            def get():
                return [
                    type("WheelEvent", (), {"type": FakePygame.MOUSEWHEEL, "y": -2})(),
                    FakeEvent(FakePygame.KEYDOWN, FakePygame.K_PAGEDOWN),
                    FakeEvent(FakePygame.KEYDOWN, FakePygame.K_RETURN),
                ]

    state = KeyboardCommandState()

    _poll_pygame_input(BriefingScrollPygame, state, control_mode="ric_translation", briefing_open=True)

    assert state.briefing_scroll_px == 288
    assert state.clip_record_save_requested is False

    terminal_state = KeyboardCommandState()

    _poll_pygame_input(BriefingScrollPygame, terminal_state, control_mode="ric_translation", terminal_open=True)

    assert terminal_state.briefing_scroll_px == 288
    assert terminal_state.clip_record_save_requested is False


def test_pygame_focus_loss_clears_live_axes() -> None:
    class FakeKeys:
        def __getitem__(self, key):
            return key in {"w", "d", "right", "space"}

    class FakePygame:
        QUIT = "quit"
        KEYDOWN = "keydown"
        WINDOWFOCUSLOST = "focuslost"
        K_w = "w"
        K_s = "s"
        K_d = "d"
        K_a = "a"
        K_RIGHT = "right"
        K_LEFT = "left"
        K_SPACE = "space"

        class event:
            @staticmethod
            def get():
                return [type("FocusEvent", (), {"type": FakePygame.WINDOWFOCUSLOST})()]

        class key:
            @staticmethod
            def get_pressed():
                return FakeKeys()

    state = KeyboardCommandState(pitch=1.0, yaw=-1.0, roll=1.0, firing=True)

    _poll_pygame_input(FakePygame, state, control_mode="attitude_thrust")

    assert state.pitch == 0.0
    assert state.yaw == 0.0
    assert state.roll == 0.0
    assert state.firing is False


def test_opposing_ric_translation_keys_cancel_axis() -> None:
    class FakeKeys:
        def __init__(self, pressed: set[str]) -> None:
            self.pressed = pressed

        def __getitem__(self, key):
            return key in self.pressed

    assert _opposing_key_axis(FakeKeys({"d"}), positive_key="d", negative_key="a") == 1.0
    assert _opposing_key_axis(FakeKeys({"a"}), positive_key="d", negative_key="a") == -1.0
    assert _opposing_key_axis(FakeKeys({"d", "a"}), positive_key="d", negative_key="a") == 0.0


def test_speed_multiple_converts_sim_dt_to_wall_step() -> None:
    assert _wall_step_s(10.0, 10.0) == 1.0
    assert _wall_step_s(0.25, 2.0) == 0.125


def test_shared_game_tick_schedule_clamps_to_level_base_dt(tmp_path: Path) -> None:
    leo_like = deepcopy(_game_config(tmp_path / "leo"))
    leo_like["simulator"]["duration_s"] = 2.0
    leo_like["simulator"]["dt_s"] = 1.0
    leo_cfg = SimulationConfig.from_dict(leo_like)

    assert _game_speed_dt_schedule(leo_cfg) == (
        (1.0, 0.1),
        (10.0, 0.5),
        (25.0, 1.0),
        (50.0, 5.0),
        (100.0, 10.0),
    )
    assert _game_tick_dt_s(leo_cfg, 1.0) == pytest.approx(0.1)
    assert _game_tick_dt_s(leo_cfg, 5.0) == pytest.approx(0.1)
    assert _game_tick_dt_s(leo_cfg, 10.0) == pytest.approx(0.5)
    assert _game_tick_dt_s(leo_cfg, 25.0) == pytest.approx(1.0)
    assert _game_tick_dt_s(leo_cfg, 200.0) == pytest.approx(1.0)

    cislunar_like = deepcopy(_game_config(tmp_path / "cislunar_like"))
    cislunar_like["simulator"]["duration_s"] = 20.0
    cislunar_like["simulator"]["dt_s"] = 10.0
    cislunar_cfg = SimulationConfig.from_dict(cislunar_like)

    assert _game_tick_dt_s(cislunar_cfg, 1.0) == pytest.approx(0.1)
    assert _game_tick_dt_s(cislunar_cfg, 5.0) == pytest.approx(0.1)
    assert _game_tick_dt_s(cislunar_cfg, 10.0) == pytest.approx(0.5)
    assert _game_tick_dt_s(cislunar_cfg, 25.0) == pytest.approx(1.0)
    assert _game_tick_dt_s(cislunar_cfg, 50.0) == pytest.approx(5.0)
    assert _game_tick_dt_s(cislunar_cfg, 200.0) == pytest.approx(10.0)


def test_simulation_session_step_accepts_smaller_game_tick(tmp_path: Path) -> None:
    cfg_dict = _game_config(tmp_path)
    cfg_dict["simulator"]["duration_s"] = 1.0
    cfg_dict["simulator"]["dt_s"] = 0.25
    config = SimulationConfig.from_dict(cfg_dict)
    session = SimulationSession.from_config(config)

    snap0 = session.reset()
    assert snap0 is not None

    snapshots = [session.step(dt_s=0.1) for _ in range(6)]

    assert snapshots[0].step_index == 1
    assert snapshots[0].time_s == pytest.approx(0.1)
    assert snapshots[-1].step_index == 6
    assert snapshots[-1].time_s == pytest.approx(0.6)
    assert session.done is False


def test_speed_multiple_adjustment_uses_allowed_options() -> None:
    assert _coerce_speed_multiple(3.0) == 2.0
    assert _adjust_speed_multiple(1.0, -1) == 1.0
    assert _adjust_speed_multiple(1.0, 1) == 2.0
    assert _adjust_speed_multiple(2.0, 1) == 5.0
    assert _adjust_speed_multiple(10.0, 1) == 25.0
    assert _adjust_speed_multiple(25.0, 1) == 50.0
    assert _adjust_speed_multiple(50.0, 1) == 100.0
    assert _adjust_speed_multiple(100.0, 1) == 200.0
    assert _adjust_speed_multiple(200.0, 1) == 200.0
    assert _adjust_speed_multiple(50.0, -2) == 10.0


def test_operator_burn_cinematic_arms_near_next_burn_and_caps_speed() -> None:
    runtime = OperatorBurnCinematicRuntime()
    provider = type("Provider", (), {"next_burn_time_s": lambda self: 103.0})()

    assert _operator_burn_cinematic_should_arm(provider, current_sim_time_s=97.0, dt_s=1.0) is False
    _update_operator_burn_cinematic(runtime, provider, now_wall_s=10.0, current_sim_time_s=99.0, dt_s=1.0)

    assert runtime.active is True
    assert runtime.hold_until_wall_s is None
    assert _operator_burn_cinematic_speed_multiple(100.0, runtime) == pytest.approx(10.0)
    assert _operator_burn_cinematic_speed_multiple(5.0, runtime) == pytest.approx(5.0)


def test_operator_burn_cinematic_arms_for_high_speed_frame_horizon() -> None:
    runtime = OperatorBurnCinematicRuntime()
    provider = type("Provider", (), {"next_burn_time_s": lambda self: 125.0})()

    assert _operator_burn_cinematic_should_arm(provider, current_sim_time_s=100.0, dt_s=10.0) is False
    _update_operator_burn_cinematic(
        runtime,
        provider,
        now_wall_s=10.0,
        current_sim_time_s=100.0,
        dt_s=10.0,
        frame_horizon_s=30.0,
    )

    assert runtime.active is True
    assert _operator_burn_cinematic_speed_multiple(1000.0, runtime, options=(1.0, 10.0, 1000.0)) == pytest.approx(
        10.0
    )


def test_operator_burn_visual_duration_scales_with_delta_v() -> None:
    assert _operator_burn_visual_duration_s(0.0) == pytest.approx(1.0)
    assert _operator_burn_visual_duration_s(0.5) == pytest.approx(1.1)
    assert _operator_burn_visual_duration_s(2.0) == pytest.approx(1.4)
    assert _operator_burn_visual_duration_s(5.0) == pytest.approx(2.0)
    assert _operator_burn_visual_duration_s(20.0) == pytest.approx(2.0)


def test_operator_burn_cinematic_holds_until_animation_finishes() -> None:
    runtime = OperatorBurnCinematicRuntime(active=True)
    provider = type("Provider", (), {"next_burn_time_s": lambda self: None})()

    _operator_burn_cinematic_hold_for_animation(runtime, now_wall_s=20.0, duration_s=1.15)
    _update_operator_burn_cinematic(runtime, provider, now_wall_s=21.0, current_sim_time_s=110.0, dt_s=1.0)

    assert runtime.active is True
    assert _operator_burn_cinematic_speed_multiple(50.0, runtime) == pytest.approx(10.0)

    _update_operator_burn_cinematic(runtime, provider, now_wall_s=21.16, current_sim_time_s=110.0, dt_s=1.0)

    assert runtime.active is False
    assert _operator_burn_cinematic_speed_multiple(50.0, runtime) == pytest.approx(50.0)


def test_operator_terminal_phase_waits_for_burn_animation() -> None:
    score = type("Score", (), {"level_passed": True, "level_failed": False})()
    runtime = OperatorBurnCinematicRuntime(active=True)

    assert (
        _phase_from_score_with_operator_animation(
            score,
            paused=True,
            game_mode="operator",
            operator_burn_cinematic=runtime,
        )
        == game_runner.GamePhase.PLAYING
    )

    runtime.reset()

    assert (
        _phase_from_score_with_operator_animation(
            score,
            paused=True,
            game_mode="operator",
            operator_burn_cinematic=runtime,
        )
        == game_runner.GamePhase.PASSED
    )


def test_command_status_uses_capitalized_indicators() -> None:
    ric_status = _command_status(KeyboardCommandState(paused=True, yaw=1.0), control_mode="ric_translation")
    attitude_status = _command_status(KeyboardCommandState(firing=False), control_mode="attitude_thrust")

    assert ric_status == "W/S R  A/D I  Left/Right C  C Camera  O/P ECI  M Music"
    assert "M Music" in ric_status
    assert "PAUSED" not in ric_status
    assert "Throttle=" not in ric_status
    assert "W/S Pitch" in attitude_status
    assert "Space Fire" in attitude_status
    assert "Thrust=Coast" in attitude_status


def test_operator_command_status_shows_next_burn_instead_of_keyboard_controls() -> None:
    provider = OperatorBurnCommandProvider(
        parse_operator_burn_plan("T= 50 s, 2.0 m/s R, 1.0 m/s I, 0.2 m/s C"),
        controlled_object_id="chaser",
        reference_object_id="target",
    )

    status = _operator_next_burn_status(provider)
    operator_status = _game_command_status(
        KeyboardCommandState(),
        control_mode="ric_translation",
        game_mode="operator",
        command_provider=provider,
    )
    pilot_status = _game_command_status(
        KeyboardCommandState(),
        control_mode="ric_translation",
        game_mode="pilot",
        command_provider=provider,
    )

    assert status == "Next Burn: T+50s | 2 m/s R, 1 m/s I, 0.2 m/s C"
    assert operator_status == status
    assert "W/S R" not in operator_status
    assert "W/S R" in pilot_status


def test_operator_command_status_shows_none_after_final_burn() -> None:
    provider = OperatorBurnCommandProvider(
        parse_operator_burn_plan("T= 0 s, 1.0 m/s R"),
        controlled_object_id="chaser",
        reference_object_id="target",
    )
    target_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0], dtype=float)
    target = StateTruth(
        position_eci_km=target_state[:3],
        velocity_eci_km_s=target_state[3:],
        attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        angular_rate_body_rad_s=np.zeros(3, dtype=float),
        mass_kg=100.0,
        t_s=0.0,
    )

    provider(
        truth=target,
        t_s=0.0,
        dt_s=1.0,
        object_id="chaser",
        own_knowledge={"target": _knowledge_from_state6(target_state)},
    )

    assert _operator_next_burn_status(provider) == "Next Burn: None"


def test_operator_mode_clears_and_skips_pilot_live_prediction_sync() -> None:
    class FakeDashboard:
        def __init__(self) -> None:
            self.calls: list[tuple[np.ndarray, float]] = []

        def set_live_prediction_burn(self, accel_ric_km_s2: np.ndarray, elapsed_s: float) -> None:
            self.calls.append((np.array(accel_ric_km_s2, dtype=float), float(elapsed_s)))

    dashboard = FakeDashboard()
    active_state = KeyboardCommandState(pitch=1.0)

    _clear_live_prediction_burn(dashboard)
    _sync_live_prediction_burn_for_mode(
        dashboard,
        active_state,
        game_mode="operator",
        control_mode="ric_translation",
        max_accel_km_s2=1.0e-6,
        elapsed_wall_s=1.0,
        speed_multiple=10.0,
        dt_s=10.0,
    )

    assert len(dashboard.calls) == 1
    assert dashboard.calls[0][0] == pytest.approx(np.zeros(3, dtype=float))
    assert dashboard.calls[0][1] == pytest.approx(0.0)

    _sync_live_prediction_burn_for_mode(
        dashboard,
        active_state,
        game_mode="pilot",
        control_mode="ric_translation",
        max_accel_km_s2=1.0e-6,
        elapsed_wall_s=1.0,
        speed_multiple=10.0,
        dt_s=10.0,
    )

    assert len(dashboard.calls) == 2
    assert dashboard.calls[1][0] == pytest.approx(np.array([1.0e-6, 0.0, 0.0], dtype=float))
    assert dashboard.calls[1][1] == pytest.approx(10.0)


def test_maneuver_input_above_control_speed_drops_to_control_speed() -> None:
    ric_state = KeyboardCommandState(pitch=1.0)
    coasting_state = KeyboardCommandState()
    no_throttle_state = KeyboardCommandState(pitch=1.0, throttle=0.0)

    assert _speed_after_maneuver_input(200.0, ric_state, control_mode="ric_translation") == 10.0
    assert _speed_after_maneuver_input(100.0, ric_state, control_mode="ric_translation") == 10.0
    assert _speed_after_maneuver_input(50.0, ric_state, control_mode="ric_translation") == 10.0
    assert _speed_after_maneuver_input(25.0, ric_state, control_mode="ric_translation") == 10.0
    assert _speed_after_maneuver_input(10.0, ric_state, control_mode="ric_translation") == 10.0
    assert _speed_after_maneuver_input(5.0, ric_state, control_mode="ric_translation") == 5.0
    assert _speed_after_maneuver_input(200.0, coasting_state, control_mode="ric_translation") == 200.0
    assert _speed_after_maneuver_input(200.0, no_throttle_state, control_mode="ric_translation") == 200.0


def test_operator_mode_bypasses_manual_maneuver_speed_bookkeeping(tmp_path: Path) -> None:
    cfg_dict = _game_config(tmp_path)
    cfg_dict["metadata"]["game"]["two_rail_speed_control"] = True
    config = SimulationConfig.from_dict(cfg_dict)
    active_state = KeyboardCommandState(pitch=1.0, use_timing_accumulator=True)
    active_state.accumulate_timed_input(0.5, speed_multiple=100.0, control_mode="ric_translation")

    assert _manual_maneuver_active_for_mode("operator", active_state, control_mode="ric_translation") is False
    assert _pending_maneuver_sim_s_for_mode("operator", active_state, control_mode="ric_translation") == pytest.approx(
        0.0
    )
    assert (
        _effective_speed_multiple_for_mode(
            config,
            100.0,
            active_state,
            game_mode="operator",
            control_mode="ric_translation",
        )
        == pytest.approx(100.0)
    )

    assert _manual_maneuver_active_for_mode("pilot", active_state, control_mode="ric_translation") is True
    assert _pending_maneuver_sim_s_for_mode("pilot", active_state, control_mode="ric_translation") > 0.0
    assert (
        _effective_speed_multiple_for_mode(
            config,
            100.0,
            active_state,
            game_mode="pilot",
            control_mode="ric_translation",
        )
        == pytest.approx(10.0)
    )


def test_maneuver_control_speed_cap_uses_current_selected_speed() -> None:
    selected_speed = 100.0
    active_state = KeyboardCommandState(pitch=1.0)
    released_state = KeyboardCommandState()

    capped = _speed_after_maneuver_input(selected_speed, active_state, control_mode="ric_translation")
    unchanged = _speed_after_maneuver_input(capped, released_state, control_mode="ric_translation")

    assert capped == pytest.approx(10.0)
    assert unchanged == pytest.approx(10.0)


def test_maneuver_input_can_use_configured_control_speed_cap() -> None:
    ric_state = KeyboardCommandState(pitch=1.0)
    options = (10.0, 25.0, 50.0, 100.0, 200.0, 500.0)

    assert (
        _speed_after_maneuver_input(
            10.0,
            ric_state,
            control_mode="ric_translation",
            options=options,
            maneuver_control_speed_multiple=100.0,
        )
        == 10.0
    )
    assert (
        _speed_after_maneuver_input(
            50.0,
            ric_state,
            control_mode="ric_translation",
            options=options,
            maneuver_control_speed_multiple=100.0,
        )
        == 50.0
    )
    assert (
        _speed_after_maneuver_input(
            200.0,
            ric_state,
            control_mode="ric_translation",
            options=options,
            maneuver_control_speed_multiple=100.0,
        )
        == 100.0
    )


def test_attitude_or_thrust_input_above_control_speed_drops_to_control_speed() -> None:
    rotate_state = KeyboardCommandState(yaw=1.0)
    firing_state = KeyboardCommandState(firing=True)
    coasting_state = KeyboardCommandState()

    assert _speed_after_maneuver_input(200.0, rotate_state, control_mode="attitude_thrust") == 10.0
    assert _speed_after_maneuver_input(100.0, rotate_state, control_mode="attitude_thrust") == 10.0
    assert _speed_after_maneuver_input(50.0, firing_state, control_mode="attitude_thrust") == 10.0
    assert _speed_after_maneuver_input(25.0, firing_state, control_mode="attitude_thrust") == 10.0
    assert _speed_after_maneuver_input(10.0, firing_state, control_mode="attitude_thrust") == 10.0
    assert _speed_after_maneuver_input(5.0, firing_state, control_mode="attitude_thrust") == 5.0
    assert _speed_after_maneuver_input(200.0, coasting_state, control_mode="attitude_thrust") == 200.0


def test_dashboard_fps_drops_at_high_speed_unless_recording() -> None:
    assert _dashboard_fps_for_speed(10.0) == 60.0
    assert _dashboard_fps_for_speed(10.0, fps_cap=30.0) == 30.0
    assert _dashboard_fps_for_speed(10.0, static_screen=True) == game_runner.STATIC_DASHBOARD_FPS
    assert _dashboard_fps_for_speed(100.0, static_screen=True, high_speed_fps=60.0) == game_runner.STATIC_DASHBOARD_FPS
    assert _dashboard_fps_for_speed(50.0) == 45.0
    assert _dashboard_fps_for_speed(100.0) == 30.0
    assert _dashboard_fps_for_speed(200.0) == 30.0
    assert _dashboard_fps_for_speed(100.0, high_speed_fps=60.0) == 60.0
    assert _dashboard_fps_for_speed(100.0, high_speed_fps=60.0, high_speed_fps_max_multiple=100.0) == 60.0
    assert _dashboard_fps_for_speed(200.0, high_speed_fps=60.0, high_speed_fps_max_multiple=100.0) == 30.0
    assert _dashboard_fps_for_speed(100.0, fps_cap=45.0, high_speed_fps=60.0) == 45.0
    assert _dashboard_fps_for_speed(200.0, recording=True) == game_runner.GAME_RECORDING_FPS
    assert _dashboard_fps_for_speed(10.0, recording=True, fps_cap=20.0) == game_runner.GAME_RECORDING_FPS
    assert _dashboard_fps_for_speed(200.0, recording=True, recording_fps=24.0) == 24.0


def test_clip_recording_status_shows_elapsed_and_recent_messages(tmp_path: Path) -> None:
    class FakeRecorder:
        saved = False

    controller = game_recording_controller.GameClipRecordingController(
        config=SimulationConfig.from_dict(_game_config(tmp_path)),
        difficulty="easy",
        recorder=FakeRecorder(),
    )

    assert _clip_recording_status(controller, started_wall_s=10.0, now_wall_s=75.0) == (
        "REC 01:05  G/F9 discard  Enter save"
    )

    controller.recorder = None

    assert (
        _clip_recording_status(
            controller,
            started_wall_s=None,
            now_wall_s=12.0,
            status_message="Clip saved",
            status_until_wall_s=13.0,
        )
        == "Clip saved"
    )
    assert (
        _clip_recording_status(
            controller,
            started_wall_s=None,
            now_wall_s=14.0,
            status_message="Clip saved",
            status_until_wall_s=13.0,
        )
        == ""
    )


def test_step_game_attempt_stops_on_first_terminal_score() -> None:
    active = type("Score", (), {"level_passed": False, "level_failed": False})()
    terminal = type("Score", (), {"level_passed": True, "level_failed": False})()

    class FakeSession:
        done = False

        def __init__(self) -> None:
            self.snapshots = ["first", "second", "third"]
            self.step_count = 0

        def step(self) -> str:
            snapshot = self.snapshots[self.step_count]
            self.step_count += 1
            return snapshot

    class FakeDashboard:
        def __init__(self) -> None:
            self.snapshots: list[str] = []

        def push_snapshot(self, snapshot: str) -> None:
            self.snapshots.append(snapshot)

    class FakeTrainer:
        def __init__(self) -> None:
            self.snapshots: list[str] = []

        def record(self, snapshot: str) -> None:
            self.snapshots.append(snapshot)

        def score(self):
            return terminal if self.snapshots else active

    session = FakeSession()
    dashboard = FakeDashboard()
    trainer = FakeTrainer()

    score = _step_game_attempt(
        session=session,
        dashboard=dashboard,
        trainer=trainer,
        steps_to_run=3,
        initial_score=active,
    )

    assert score is terminal
    assert session.step_count == 1
    assert dashboard.snapshots == ["first"]
    assert trainer.snapshots == ["first"]


def test_step_game_attempt_splits_large_autonomy_tick() -> None:
    active = type("Score", (), {"level_passed": False, "level_failed": False})()

    class FakeSession:
        done = False

        def __init__(self) -> None:
            self.step_dts: list[float] = []

        def step(self, *, dt_s: float) -> str:
            self.step_dts.append(float(dt_s))
            return f"snapshot-{len(self.step_dts)}"

    class FakeDashboard:
        def __init__(self) -> None:
            self.snapshots: list[str] = []

        def push_snapshot(self, snapshot: str) -> None:
            self.snapshots.append(snapshot)

    class FakeTrainer:
        config = RPOTrainingConfig(enabled=True)

        def __init__(self) -> None:
            self.snapshots: list[str] = []

        def record(self, snapshot: str) -> None:
            self.snapshots.append(snapshot)

        def score(self):
            return active

    session = FakeSession()
    dashboard = FakeDashboard()
    trainer = FakeTrainer()

    score = _step_game_attempt(
        session=session,
        dashboard=dashboard,
        trainer=trainer,
        steps_to_run=1,
        initial_score=active,
        dt_s=3.25,
        max_step_dt_s=1.0,
    )

    assert score is active
    assert session.step_dts == pytest.approx([1.0, 1.0, 1.0, 0.25])
    assert dashboard.snapshots == ["snapshot-1", "snapshot-2", "snapshot-3", "snapshot-4"]
    assert trainer.snapshots == dashboard.snapshots
    assert _split_game_step_dt(0.75, max_step_dt_s=1.0) == pytest.approx((0.75,))
    assert _split_game_step_dt(2.5, max_step_dt_s=1.0) == pytest.approx((1.0, 1.0, 0.5))


def test_realtime_steps_due_supports_multi_step_catchup_for_100x() -> None:
    steps, next_wall = _realtime_steps_due(now_s=10.016, last_step_wall_s=10.0, wall_step_s=0.01)

    assert steps == 1
    assert np.isclose(next_wall, 10.01)

    steps, next_wall = _realtime_steps_due(now_s=10.033, last_step_wall_s=10.01, wall_step_s=0.01)

    assert steps == 2
    assert np.isclose(next_wall, 10.03)


def test_realtime_steps_due_caps_stall_catchup() -> None:
    steps, next_wall = _realtime_steps_due(now_s=20.0, last_step_wall_s=10.0, wall_step_s=0.01, max_steps=12)

    assert steps == 12
    assert next_wall == 20.0


def test_cw_coast_state_zero_time_returns_initial_state() -> None:
    x0 = np.array([0.1, -1.0, 0.2, 0.0, 0.001, -0.001], dtype=float)

    out = _cw_coast_state(x0, 0.0, 0.001)

    assert np.allclose(out, x0)


def test_cw_coast_states_matches_scalar_propagation() -> None:
    x0 = np.array([0.1, -1.0, 0.2, 0.0002, 0.001, -0.001], dtype=float)
    times = np.linspace(0.0, 120.0, 25)
    expected = np.vstack([_cw_coast_state(x0, float(t), 0.001) for t in times])

    out = _cw_coast_states(x0, times, 0.001)

    np.testing.assert_allclose(out, expected)


def test_cw_forced_state_advances_short_visual_burn() -> None:
    x0 = np.zeros(6, dtype=float)
    accel = np.array([1.0e-5, 0.0, 0.0], dtype=float)

    out = _cw_forced_state(x0, accel, 0.5, 0.001)

    assert out[0] > 0.0
    assert out[3] == pytest.approx(5.0e-6, rel=1.0e-3)


def test_live_prediction_accel_ric_matches_manual_translation_scaling() -> None:
    state = KeyboardCommandState(pitch=1.0, yaw=1.0, roll=0.0, throttle=0.5)

    accel = _live_prediction_accel_ric(
        state,
        control_mode="ric_translation",
        max_accel_km_s2=2.0e-5,
    )

    assert np.linalg.norm(accel) == pytest.approx(1.0e-5)
    assert accel[0] == pytest.approx(accel[1])


def test_live_prediction_accel_ric_clears_when_paused() -> None:
    state = KeyboardCommandState(pitch=1.0, yaw=0.0, roll=0.0, throttle=1.0, paused=True)

    accel = _live_prediction_accel_ric(
        state,
        control_mode="ric_translation",
        max_accel_km_s2=2.0e-5,
    )

    assert np.allclose(accel, np.zeros(3, dtype=float))


def test_live_prediction_burn_uses_current_timed_input() -> None:
    state = KeyboardCommandState(pitch=1.0, yaw=1.0, throttle=1.0)
    state.use_timing_accumulator = True
    state.pitch_sim_s = 0.25
    state.yaw_sim_s = 0.10

    accel, elapsed = _live_prediction_burn(
        state,
        control_mode="ric_translation",
        max_accel_km_s2=2.0e-5,
        elapsed_wall_s=0.75,
        speed_multiple=10.0,
        dt_s=1.0,
    )

    assert elapsed == pytest.approx(1.0)
    expected = np.array([1.0, 1.0, 0.0], dtype=float)
    expected = expected / np.linalg.norm(expected) * 2.0e-5
    np.testing.assert_allclose(accel, expected)
    assert accel[2] == pytest.approx(0.0)


def test_live_prediction_burn_uses_current_input_before_timed_residual_accumulates() -> None:
    state = KeyboardCommandState(pitch=1.0, throttle=1.0)
    state.use_timing_accumulator = True

    accel, elapsed = _live_prediction_burn(
        state,
        control_mode="ric_translation",
        max_accel_km_s2=2.0e-5,
        elapsed_wall_s=0.02,
        speed_multiple=10.0,
        dt_s=1.0,
    )

    np.testing.assert_allclose(accel, np.array([2.0e-5, 0.0, 0.0], dtype=float))
    assert elapsed == pytest.approx(0.2)


def test_live_prediction_burn_ignores_pending_residual_after_key_release() -> None:
    state = KeyboardCommandState(throttle=1.0)
    state.use_timing_accumulator = True
    state.pitch_sim_s = 0.25
    state.yaw_sim_s = 0.10

    accel, elapsed = _live_prediction_burn(
        state,
        control_mode="ric_translation",
        max_accel_km_s2=2.0e-5,
        elapsed_wall_s=0.75,
        speed_multiple=10.0,
        dt_s=1.0,
    )

    np.testing.assert_allclose(accel, np.zeros(3, dtype=float))
    assert elapsed == pytest.approx(0.0)


def test_live_prediction_burn_matches_manual_duty_cycle_shape() -> None:
    state = KeyboardCommandState(pitch=1.0, yaw=1.0, throttle=1.0)
    state.use_timing_accumulator = True
    state.pitch_sim_s = 2.0
    state.yaw_sim_s = 1.0

    accel, elapsed = _live_prediction_burn(
        state,
        control_mode="ric_translation",
        max_accel_km_s2=2.0e-5,
        elapsed_wall_s=0.0,
        speed_multiple=10.0,
        dt_s=1.0,
    )

    expected_duty = np.array([1.0, 1.0, 0.0], dtype=float)
    expected_duty /= np.linalg.norm(expected_duty)
    assert elapsed == pytest.approx(0.0)
    np.testing.assert_allclose(accel, expected_duty * 2.0e-5)


def test_live_prediction_burn_falls_back_to_wall_elapsed_without_accumulator() -> None:
    state = KeyboardCommandState(pitch=1.0, throttle=1.0)

    accel, elapsed = _live_prediction_burn(
        state,
        control_mode="ric_translation",
        max_accel_km_s2=2.0e-5,
        elapsed_wall_s=0.2,
        speed_multiple=3.0,
        dt_s=1.0,
    )

    assert elapsed == pytest.approx(0.6)
    assert accel[0] == pytest.approx(2.0e-5)
    assert accel[1] == pytest.approx(0.0)


def test_coast_prediction_model_aliases_support_elliptic_levels() -> None:
    assert _coast_prediction_model_key("HCW") == "hcw"
    assert _coast_prediction_model_key("elliptic") == "elliptic_linear"
    assert _coast_prediction_model_key("Tschauner-Hempel") == "tschauner_hempel"
    assert _cr3bp_projection_mode_key("STM") == "linearized"
    assert _cr3bp_projection_mode_key("linearized") == "linearized"
    assert _cr3bp_projection_mode_key("nonlinear") == "nonlinear"


def test_elliptic_dashboard_true_anomaly_indicator_uses_target_state() -> None:
    target_r, target_v = coes_mapping_to_rv_eci(
        {
            "a_km": 9000.0,
            "ecc": 0.25,
            "inc_deg": 45.0,
            "raan_deg": 0.0,
            "argp_deg": 0.0,
            "true_anomaly_deg": 140.0,
        }
    )
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.coast_prediction_model = "tschauner_hempel"
    dashboard.target_true_anomaly_deg = _true_anomaly_deg_from_state(np.hstack((target_r, target_v)))

    assert dashboard.target_true_anomaly_deg == pytest.approx(140.0)
    assert dashboard._true_anomaly_indicator_text() == "Target ν=140.0 deg"


def test_true_anomaly_indicator_is_hidden_for_hcw_dashboard() -> None:
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.coast_prediction_model = "hcw"
    dashboard.target_true_anomaly_deg = 140.0

    assert dashboard._true_anomaly_indicator_text() == ""


def test_hcw_dashboard_does_not_compute_true_anomaly(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_anomaly_compute(_: np.ndarray) -> float:
        raise AssertionError("HCW dashboard should not compute target true anomaly")

    monkeypatch.setattr("sim.game.pygame_dashboard._true_anomaly_deg_from_state", fail_anomaly_compute)
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.target_object_id = "target"
    dashboard.chaser_object_id = "chaser"
    dashboard.reference_object_id = "target"
    dashboard.coast_prediction_model = "hcw"
    dashboard.max_history = 10
    dashboard.t_s = []
    dashboard.rel_hist = []
    dashboard.target_rel_hist = []
    dashboard.thrust_hist = []
    dashboard.thrust_ric_hist = []
    dashboard._rel_array = np.zeros((0, 6), dtype=float)
    dashboard._target_rel_array = np.zeros((0, 6), dtype=float)
    dashboard._thrust_ric_array = np.zeros((0, 3), dtype=float)
    dashboard.target_true_anomaly_deg = 123.0
    target = np.array([7000.0, 0.0, 0.0, 0.0, 7.54605329, 0.0], dtype=float)
    chaser = ric_rect_state_to_eci(np.array([0.0, -1.0, 0.0, 0.0, 0.0, 0.0]), target[:3], target[3:])
    snapshot = type(
        "Snapshot",
        (),
        {
            "time_s": 0.0,
            "truth": {"target": target, "chaser": chaser},
            "applied_thrust": {"chaser": np.zeros(3, dtype=float)},
        },
    )()

    dashboard.push_snapshot(snapshot)

    assert dashboard.target_true_anomaly_deg is None


def test_cislunar_dashboard_accepts_full_truth_arrays() -> None:
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.target_object_id = "target"
    dashboard.chaser_object_id = "chaser"
    dashboard.reference_object_id = "target"
    dashboard.relative_frame = "cislunar_l1"
    dashboard.coast_prediction_model = "cr3bp"
    dashboard.max_history = 10
    dashboard.t_s = []
    dashboard.rel_hist = []
    dashboard.target_rel_hist = []
    dashboard.thrust_hist = []
    dashboard.thrust_ric_hist = []
    dashboard._rel_array = np.zeros((0, 6), dtype=float)
    dashboard._target_rel_array = np.zeros((0, 6), dtype=float)
    dashboard._thrust_ric_array = np.zeros((0, 3), dtype=float)
    dashboard.target_true_anomaly_deg = None
    origin = cr3bp_l1_state_km_s()
    target_state6 = origin + np.array([1.0, 2.0, 3.0, 0.001, 0.002, 0.003], dtype=float)
    chaser_state6 = target_state6 + np.array([0.0, -25.0, 10.0, 0.0, 0.0, 0.0], dtype=float)
    target_truth14 = np.hstack((target_state6, [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1800.0]))
    chaser_truth14 = np.hstack((chaser_state6, [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [200.0]))
    snapshot = type(
        "Snapshot",
        (),
        {
            "time_s": 0.0,
            "truth": {"target": target_truth14, "chaser": chaser_truth14},
            "applied_thrust": {"chaser": np.array([0.0, 1.0e-9, 0.0], dtype=float)},
        },
    )()

    dashboard.push_snapshot(snapshot)

    assert dashboard.target_rel_hist[-1] == pytest.approx(target_state6 - origin)
    assert dashboard.rel_hist[-1] == pytest.approx(chaser_state6 - origin)
    assert dashboard.thrust_ric_hist[-1] == pytest.approx([0.0, 1.0e-9, 0.0])


def test_cislunar_bonus_uses_target_centered_moon_ric_panels() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "sim"
        / "game"
        / "configs"
        / "game_training_rpo_bonus_cislunar_rendezvous.yaml"
    )
    config = SimulationConfig.from_yaml(path)
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.camera_mode = _game_camera_mode(config)
    dashboard.camera_rule_mode = _game_camera_rule_mode(config)
    dashboard.relative_frame = _game_relative_frame(config)
    dashboard.target_centered_plot_planes = _game_target_centered_plot_planes(config)
    dashboard.target_centered_plot_axes = _game_target_centered_plot_axes(config)
    chaser_current = np.array([-3.0, -4.0, 0.5], dtype=float)
    target_current = np.zeros(3, dtype=float)
    left_panel, right_panel = dashboard._plot_panel_specs()

    ri_center = dashboard._camera_center_ric(
        chaser_current=chaser_current,
        target_current=target_current,
        x_axis=left_panel[1],
        y_axis=left_panel[2],
    )
    right_center = dashboard._camera_center_ric(
        chaser_current=chaser_current,
        target_current=target_current,
        x_axis=right_panel[1],
        y_axis=right_panel[2],
    )

    assert dashboard.camera_mode == "rule_toggle_pair"
    assert dashboard.target_centered_plot_planes == ("RI", "RC")
    assert left_panel == ("RI Plane: In-Track Vs Radial", 1, 0)
    assert right_panel == ("RC Plane: Cross-Track Vs Radial", 2, 0)
    assert dashboard._axis_label_for_plot(0) == "R km"
    assert dashboard._axis_label_for_plot(1) == "I km"
    assert dashboard._axis_label_for_plot(2) == "C km"
    assert ri_center == pytest.approx(np.zeros(3, dtype=float))
    assert right_center == pytest.approx(np.zeros(3, dtype=float))

    dashboard.toggle_camera_rule_mode()

    assert dashboard._camera_rule_mode_key() == "full_trajectory"
    assert dashboard._camera_center_ric(
        chaser_current=chaser_current,
        target_current=target_current,
        x_axis=left_panel[1],
        y_axis=left_panel[2],
    ) == pytest.approx(np.zeros(3, dtype=float))


def test_cislunar_moon_background_is_right_plot_only_and_to_scale() -> None:
    assert _should_draw_cislunar_moon_background(relative_frame="cislunar_l1", x_axis=1, y_axis=2) is True
    assert _should_draw_cislunar_moon_background(relative_frame="cislunar_l1", x_axis=1, y_axis=0) is False
    assert _should_draw_cislunar_moon_background(relative_frame="ric", x_axis=1, y_axis=2) is False

    rect = _scaled_body_rect_tuple(
        center_px=(100, 200),
        radius_km=MOON_RADIUS_KM,
        scale_x=0.01,
        scale_y=0.02,
    )

    assert rect == (
        100 - round(MOON_RADIUS_KM * 0.01),
        200 - round(MOON_RADIUS_KM * 0.02),
        2 * round(MOON_RADIUS_KM * 0.01),
        2 * round(MOON_RADIUS_KM * 0.02),
    )


def test_cislunar_dashboard_uses_bounded_cached_cr3bp_prediction(monkeypatch: pytest.MonkeyPatch) -> None:
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.coast_prediction_model = "cr3bp"
    dashboard.coast_prediction_orbit_fraction = 1.0
    dashboard.coast_prediction_horizon_s = 300.0
    dashboard.coast_prediction_dt_s = 10.0
    dashboard.cr3bp_coast_prediction_horizon_s = 21600.0
    dashboard.cr3bp_coast_prediction_dt_s = 300.0
    dashboard.mean_motion_rad_s = EARTH_MOON_MEAN_MOTION_RAD_S
    dashboard.t_s = [0.0]
    dashboard.reference_state_eci = cr3bp_l1_state_km_s()
    dashboard._prediction_cache = {}

    calls = 0
    original = PygameRPODashboard._coast_prediction_from

    def counted(self: PygameRPODashboard, rel0: np.ndarray, **kwargs: float | None) -> np.ndarray:
        nonlocal calls
        calls += 1
        return original(self, rel0, **kwargs)

    monkeypatch.setattr(PygameRPODashboard, "_coast_prediction_from", counted)
    rel0 = np.array([1.0, 2.0, 3.0, 1.0e-3, 2.0e-3, 3.0e-3], dtype=float)

    first = dashboard._coast_prediction_from_cached("chaser", rel0, active_burn=False)
    second = dashboard._coast_prediction_from_cached("chaser", rel0, active_burn=False)

    assert calls == 1
    assert first.shape == (73, 6)
    assert second == pytest.approx(first)


def test_cr3bp_coast_prediction_cache_uses_configured_update_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.coast_prediction_model = "cr3bp"
    dashboard.cr3bp_prediction_coast_update_interval_s = 300.0
    dashboard.t_s = [0.0]
    dashboard.reference_state_eci = cr3bp_l1_state_km_s()
    dashboard._prediction_cache = {}
    calls = []

    def fake_prediction(self: PygameRPODashboard, rel0: np.ndarray, **kwargs: float | None) -> np.ndarray:
        calls.append(np.array(rel0, dtype=float).copy())
        return np.full((2, 6), float(len(calls)), dtype=float)

    monkeypatch.setattr(PygameRPODashboard, "_coast_prediction_from", fake_prediction)
    rel0 = np.array([1.0, 2.0, 3.0, 1.0e-3, 2.0e-3, 3.0e-3], dtype=float)

    first = dashboard._coast_prediction_from_cached("chaser", rel0, active_burn=False)
    dashboard.t_s = [100.0]
    second = dashboard._coast_prediction_from_cached("chaser", rel0, active_burn=False)
    dashboard.t_s = [301.0]
    third = dashboard._coast_prediction_from_cached("chaser", rel0, active_burn=False)

    assert len(calls) == 2
    assert np.all(first == 1.0)
    assert np.all(second == 1.0)
    assert np.all(third == 2.0)


def test_cr3bp_prediction_cache_refreshes_active_burns(monkeypatch: pytest.MonkeyPatch) -> None:
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.coast_prediction_model = "cr3bp"
    dashboard.t_s = [0.0]
    dashboard.reference_state_eci = cr3bp_l1_state_km_s()
    dashboard._prediction_cache = {}
    calls = []

    def fake_prediction(self: PygameRPODashboard, rel0: np.ndarray, **kwargs: float | None) -> np.ndarray:
        calls.append(np.array(rel0, dtype=float).copy())
        return np.full((2, 6), float(len(calls)), dtype=float)

    monkeypatch.setattr(PygameRPODashboard, "_coast_prediction_from", fake_prediction)
    rel0 = np.array([1.0, 2.0, 3.0, 1.0e-3, 2.0e-3, 3.0e-3], dtype=float)

    first = dashboard._coast_prediction_from_cached("chaser", rel0, active_burn=True)
    second = dashboard._coast_prediction_from_cached("chaser", rel0, active_burn=True)

    assert len(calls) == 2
    assert np.all(first == 1.0)
    assert np.all(second == 2.0)


def test_cr3bp_active_burn_projection_uses_smaller_point_budget() -> None:
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.coast_prediction_model = "cr3bp"
    dashboard.cr3bp_projection_mode = "linearized"
    dashboard.relative_frame = "moon_ric"
    dashboard.coast_prediction_orbit_fraction = None
    dashboard.coast_prediction_horizon_s = 604800.0
    dashboard.cr3bp_coast_prediction_horizon_s = 604800.0
    dashboard.cr3bp_coast_prediction_dt_s = 1.0
    dashboard.coast_prediction_dt_s = 10.0
    dashboard.mean_motion_rad_s = EARTH_MOON_MEAN_MOTION_RAD_S
    dashboard.t_s = [0.0]
    dashboard.reference_state_eci = cr3bp_halo_seed_state_km_s(family="l2_nrho_southern")
    dashboard._prediction_cache = {}
    rel0 = np.array([-3.0, 4.0, 0.5, 0.0, 0.0, 2.0e-6], dtype=float)

    active = dashboard._coast_prediction_from_cached("chaser", rel0, active_burn=True)
    dashboard._prediction_cache = {}
    coast = dashboard._coast_prediction_from_cached("chaser", rel0, active_burn=False)

    assert active.shape[0] == MAX_ACTIVE_CR3BP_GHOST_DRAW_POINTS
    assert coast.shape[0] == 120
    np.testing.assert_allclose(active[:6, 0], coast[:6, 0])


def test_cr3bp_time_remaining_horizon_overrides_local_orbit_fraction(monkeypatch: pytest.MonkeyPatch) -> None:
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.coast_prediction_model = "cr3bp"
    dashboard.cr3bp_projection_mode = "linearized"
    dashboard.relative_frame = "moon_ric"
    dashboard.coast_prediction_orbit_fraction = 1.0
    dashboard.mean_motion_rad_s = 2.0 * np.pi / 1000.0
    dashboard.coast_prediction_horizon_s = 1000.0
    dashboard.cr3bp_coast_prediction_horizon_s = 604800.0
    dashboard.cr3bp_coast_prediction_horizon_mode = "time_remaining"
    dashboard.cr3bp_coast_prediction_dt_s = 1.0
    dashboard.coast_prediction_dt_s = 10.0
    dashboard.mission_time_budget_s = 259200.0
    dashboard.t_s = [1000.0, 87400.0]
    dashboard.reference_state_eci = cr3bp_halo_seed_state_km_s(family="l2_nrho_southern")
    captured: dict[str, np.ndarray] = {}

    def fake_projection(
        self: PygameRPODashboard,
        rel0: np.ndarray,
        *,
        target_state: np.ndarray,
        times: np.ndarray,
        current_t_s: float,
    ) -> np.ndarray:
        captured["times"] = np.array(times, dtype=float).copy()
        return np.zeros((len(times), 6), dtype=float)

    monkeypatch.setattr(PygameRPODashboard, "_linearized_cr3bp_moon_ric_coast_prediction_cached", fake_projection)

    prediction = dashboard._coast_prediction_from(np.zeros(6, dtype=float))

    assert prediction.shape[0] == 120
    assert captured["times"][-1] == pytest.approx(172800.0)


def test_cr3bp_configured_horizon_overrides_local_orbit_fraction(monkeypatch: pytest.MonkeyPatch) -> None:
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.coast_prediction_model = "cr3bp"
    dashboard.cr3bp_projection_mode = "linearized"
    dashboard.relative_frame = "moon_ric"
    dashboard.coast_prediction_orbit_fraction = 1.0
    dashboard.mean_motion_rad_s = 2.0 * np.pi / 1000.0
    dashboard.coast_prediction_horizon_s = 1000.0
    dashboard.cr3bp_coast_prediction_horizon_s = 21600.0
    dashboard.cr3bp_coast_prediction_horizon_mode = "default"
    dashboard.cr3bp_coast_prediction_dt_s = 1.0
    dashboard.coast_prediction_dt_s = 10.0
    dashboard.t_s = [0.0]
    dashboard.reference_state_eci = cr3bp_halo_seed_state_km_s(family="l2_nrho_southern")
    captured: dict[str, np.ndarray] = {}

    def fake_projection(
        self: PygameRPODashboard,
        rel0: np.ndarray,
        *,
        target_state: np.ndarray,
        times: np.ndarray,
        current_t_s: float,
    ) -> np.ndarray:
        captured["times"] = np.array(times, dtype=float).copy()
        return np.zeros((len(times), 6), dtype=float)

    monkeypatch.setattr(PygameRPODashboard, "_linearized_cr3bp_moon_ric_coast_prediction_cached", fake_projection)

    prediction = dashboard._coast_prediction_from(np.zeros(6, dtype=float))

    assert prediction.shape[0] == 120
    assert captured["times"][-1] == pytest.approx(21600.0)


def test_cr3bp_active_prediction_horizon_can_cap_live_burn_only(monkeypatch: pytest.MonkeyPatch) -> None:
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.coast_prediction_model = "cr3bp"
    dashboard.cr3bp_projection_mode = "linearized"
    dashboard.relative_frame = "moon_ric"
    dashboard.coast_prediction_orbit_fraction = 1.0
    dashboard.mean_motion_rad_s = 2.0 * np.pi / 1000.0
    dashboard.coast_prediction_horizon_s = 1000.0
    dashboard.cr3bp_coast_prediction_horizon_s = 604800.0
    dashboard.cr3bp_active_prediction_horizon_s = 21600.0
    dashboard.cr3bp_coast_prediction_horizon_mode = "time_remaining"
    dashboard.cr3bp_coast_prediction_dt_s = 1.0
    dashboard.coast_prediction_dt_s = 10.0
    dashboard.cr3bp_prediction_coast_update_interval_s = 300.0
    dashboard.mission_time_budget_s = 259200.0
    dashboard.t_s = [0.0]
    dashboard.reference_state_eci = cr3bp_halo_seed_state_km_s(family="l2_nrho_southern")
    dashboard._prediction_cache = {}
    captured: list[np.ndarray] = []

    def fake_projection(
        self: PygameRPODashboard,
        rel0: np.ndarray,
        *,
        target_state: np.ndarray,
        times: np.ndarray,
        current_t_s: float,
    ) -> np.ndarray:
        captured.append(np.array(times, dtype=float).copy())
        return np.zeros((len(times), 6), dtype=float)

    monkeypatch.setattr(PygameRPODashboard, "_linearized_cr3bp_moon_ric_coast_prediction_cached", fake_projection)

    active = dashboard._coast_prediction_from_cached("active", np.zeros(6, dtype=float), active_burn=True)
    coast = dashboard._coast_prediction_from_cached("coast", np.zeros(6, dtype=float), active_burn=False)

    assert active.shape[0] == MAX_ACTIVE_CR3BP_GHOST_DRAW_POINTS
    assert coast.shape[0] == 120
    assert captured[0][-1] == pytest.approx(21600.0)
    assert captured[1][-1] == pytest.approx(259200.0)


def test_linearized_cr3bp_stm_table_reuses_for_new_live_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.coast_prediction_model = "cr3bp"
    dashboard.cr3bp_projection_mode = "linearized"
    dashboard.relative_frame = "moon_ric"
    dashboard.coast_prediction_orbit_fraction = None
    dashboard.coast_prediction_horizon_s = 604800.0
    dashboard.cr3bp_coast_prediction_horizon_s = 604800.0
    dashboard.cr3bp_coast_prediction_dt_s = 1.0
    dashboard.coast_prediction_dt_s = 10.0
    dashboard.mean_motion_rad_s = EARTH_MOON_MEAN_MOTION_RAD_S
    dashboard.t_s = [0.0]
    dashboard.reference_state_eci = cr3bp_halo_seed_state_km_s(family="l2_nrho_southern")
    dashboard._prediction_cache = {}
    calls = 0
    original = game_pygame_dashboard.propagate_cr3bp_reference_stm

    def counted(reference_state, stm, dt_s, t_s, **kwargs):
        nonlocal calls
        calls += 1
        return original(reference_state, stm, dt_s, t_s, **kwargs)

    monkeypatch.setattr(game_pygame_dashboard, "propagate_cr3bp_reference_stm", counted)
    rel0 = np.array([-3.0, 4.0, 0.5, 0.0, 0.0, 2.0e-6], dtype=float)
    rel1 = rel0.copy()
    rel1[4] += 1.0e-5

    first = dashboard._coast_prediction_from_cached("chaser", rel0, active_burn=True)
    first_calls = calls
    second = dashboard._coast_prediction_from_cached("chaser", rel1, active_burn=True)

    assert first_calls > 0
    assert calls == first_calls
    assert first.shape == second.shape == (MAX_ACTIVE_CR3BP_GHOST_DRAW_POINTS, 6)
    assert not np.allclose(first, second)


def test_linearized_cr3bp_stm_table_invalidates_when_reference_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.coast_prediction_model = "cr3bp"
    dashboard.cr3bp_projection_mode = "linearized"
    dashboard.relative_frame = "moon_ric"
    dashboard.coast_prediction_orbit_fraction = None
    dashboard.coast_prediction_horizon_s = 120.0
    dashboard.cr3bp_coast_prediction_horizon_s = 120.0
    dashboard.cr3bp_coast_prediction_dt_s = 1.0
    dashboard.coast_prediction_dt_s = 10.0
    dashboard.mean_motion_rad_s = EARTH_MOON_MEAN_MOTION_RAD_S
    dashboard.t_s = [0.0]
    dashboard.reference_state_eci = cr3bp_halo_seed_state_km_s(family="l2_nrho_southern")
    dashboard._prediction_cache = {}
    calls = 0
    original = game_pygame_dashboard.propagate_cr3bp_reference_stm

    def counted(reference_state, stm, dt_s, t_s, **kwargs):
        nonlocal calls
        calls += 1
        return original(reference_state, stm, dt_s, t_s, **kwargs)

    monkeypatch.setattr(game_pygame_dashboard, "propagate_cr3bp_reference_stm", counted)
    rel0 = np.array([-3.0, 4.0, 0.5, 0.0, 0.0, 2.0e-6], dtype=float)

    dashboard._coast_prediction_from_cached("chaser", rel0, active_burn=True)
    first_calls = calls
    dashboard.reference_state_eci = dashboard.reference_state_eci.copy()
    dashboard.reference_state_eci[0] += 0.01
    dashboard._coast_prediction_from_cached("chaser", rel0, active_burn=True)

    assert first_calls > 0
    assert calls > first_calls


def test_briefing_body_wraps_all_lines_for_scrollable_card() -> None:
    class FakeFont:
        def size(self, text):
            return (len(str(text)) * 8, 18)

    dashboard = object.__new__(PygameRPODashboard)
    dashboard.font = FakeFont()

    lines = tuple(f"Instruction {idx} " + "burn coast observe " * 6 for idx in range(12))

    wrapped = dashboard._briefing_body_lines(lines, width_px=180)

    assert any("Instruction 11" in line for line in wrapped)
    assert len(wrapped) > len(lines)
    assert PygameRPODashboard._briefing_footer_text(scrollable=True).startswith("Scroll To Read.")


def test_mission_banner_wraps_long_failure_lines_inside_fixed_box() -> None:
    class FakeFont:
        def size(self, text):
            return (len(str(text)) * 8, 18)

    dashboard = object.__new__(PygameRPODashboard)
    dashboard.font = FakeFont()
    dashboard._mission_banner_layout_cache = {}

    lines = (
        "Result        Forbidden region violated: lower radial rail of V-bar U, lower radial floor in RC.",
    )

    wrapped = dashboard._mission_banner_body_lines(lines, width_px=360)

    assert len(wrapped) > 1
    assert wrapped[0].startswith("Result")
    assert all(FakeFont().size(line)[0] <= 360 for line in wrapped)
    assert any(line.startswith("              ") for line in wrapped[1:])


def test_mission_banner_scroll_footer_stays_compact() -> None:
    footer = PygameRPODashboard._mission_banner_footer_text("D Debrief  R Retry  Esc Quit", scrollable=True)

    assert footer == "Scroll/Page  D Debrief  R Retry  Esc Quit"
    assert "Press" not in footer
    assert len(footer) < 48


def test_pause_overlay_teaches_hcw_and_ric_frame() -> None:
    equations = PygameRPODashboard._pause_overlay_equation_lines()
    takeaways = PygameRPODashboard._pause_overlay_takeaway_lines()

    assert "R'' = 3 n² R + 2 n I' + a_R" in equations
    assert "I'' = -2 n R' + a_I" in equations
    assert "C'' = -n² C + a_C" in equations
    assert any("R is radial" in line for line in takeaways)
    assert any("I is in-track" in line for line in takeaways)
    assert any("C is cross-track" in line for line in takeaways)


def test_pause_teaching_overlay_skips_guided_tutorial_prompts() -> None:
    training_cfg = RPOTrainingConfig(
        enabled=True,
        scenario_id="rpo_00_tutorial",
        guided_tutorial_burns=(
            GuidedTutorialBurnConfig(name="radial", axis="radial", sign=1, delta_v_m_s=0.01),
        ),
    )
    runtime = game_runner.GuidedTutorialRuntime()

    assert _pause_teaching_overlay_enabled(game_runner.GamePhase.PAUSED, training_cfg, runtime) is False
    assert _pause_teaching_overlay_enabled(
        game_runner.GamePhase.PAUSED,
        replace(training_cfg, guided_tutorial_burns=()),
        runtime,
    )
    runtime.awaiting_speed_step = True
    assert _pause_teaching_overlay_enabled(
        game_runner.GamePhase.PAUSED,
        replace(training_cfg, guided_tutorial_burns=()),
        runtime,
    ) is False


def test_elliptic_linear_coast_matches_hcw_for_circular_chief() -> None:
    rel0 = np.array([0.1, -1.0, 0.2, 0.0, 0.001, -0.001], dtype=float)
    chief = np.array([7000.0, 0.0, 0.0, 0.0, np.sqrt(398600.4418 / 7000.0), 0.0], dtype=float)
    n = np.sqrt(398600.4418 / 7000.0**3)
    times = np.array([0.0, 30.0, 60.0, 120.0], dtype=float)

    elliptic = _elliptic_linear_coast_states(rel0, times, chief)
    circular = np.vstack([_cw_coast_state(rel0, float(t), n) for t in times])

    assert np.allclose(elliptic, circular, atol=2.0e-5)


def test_coast_prediction_difficulty_maps_to_orbit_fraction() -> None:
    assert _coast_prediction_orbit_fraction("easy") == 1.0
    assert _coast_prediction_orbit_fraction("medium") == 0.5
    assert _coast_prediction_orbit_fraction("hard") == 0.25
    assert _coast_prediction_orbit_fraction("extreme") == 0.0


def test_operator_difficulty_maps_to_actuator_error_and_full_projection() -> None:
    assert _operator_actuator_error_fraction("easy") == pytest.approx(0.0)
    assert _operator_actuator_error_fraction("medium") == pytest.approx(0.01)
    assert _operator_actuator_error_fraction("hard") == pytest.approx(0.025)
    assert _operator_actuator_error_fraction("extreme") == pytest.approx(0.05)
    assert _operator_coast_prediction_orbit_fraction("operator", "extreme") == pytest.approx(1.0)
    assert _operator_coast_prediction_orbit_fraction("pilot", "extreme") == pytest.approx(0.0)


def test_coast_prediction_horizon_uses_orbital_period() -> None:
    n = 0.001
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.coast_prediction_horizon_s = 300.0
    dashboard.coast_prediction_orbit_fraction = 0.5

    assert np.isclose(dashboard._coast_prediction_horizon_s(n), np.pi / n)

    dashboard.coast_prediction_orbit_fraction = 0.0
    assert dashboard._coast_prediction_horizon_s(n) == 0.0


def test_dashboard_samples_long_polylines_for_drawing() -> None:
    rows = np.arange(1000.0).reshape(500, 2)

    sampled = _sample_rows(rows, 120)

    assert sampled.shape[0] <= 120
    assert np.allclose(sampled[0], rows[0])
    assert np.allclose(sampled[-1], rows[-1])


def test_dashboard_eci_plot_toggle_swaps_ri_and_rc_panels() -> None:
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.relative_frame = "ric"
    dashboard.plot_view_modes = {}
    dashboard._frame_cache_dirty = False

    assert dashboard._plot_view_mode_for_axes(x_axis=1, y_axis=0) == "ric"
    assert dashboard.toggle_eci_plot("RI") == "eci"
    assert dashboard._plot_view_mode_for_axes(x_axis=1, y_axis=0) == "eci"
    assert dashboard._plot_view_mode_for_axes(x_axis=2, y_axis=0) == "ric"
    assert dashboard._frame_cache_dirty is True

    dashboard._frame_cache_dirty = False
    assert dashboard.toggle_eci_plot("RC") == "eci"
    assert dashboard._plot_view_mode_for_axes(x_axis=2, y_axis=0) == "eci"
    assert dashboard.toggle_eci_plot("RI") == "ric"
    assert dashboard._plot_view_mode_for_axes(x_axis=1, y_axis=0) == "ric"


def test_dashboard_eci_plot_is_disabled_for_cislunar_frame() -> None:
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.relative_frame = "cislunar_l1"
    dashboard.plot_view_modes = {"RI": "eci", "RC": "eci"}

    assert dashboard._plot_view_mode_for_axes(x_axis=1, y_axis=0) == "ric"
    assert dashboard._plot_view_mode_for_axes(x_axis=1, y_axis=2) == "ric"


def test_dashboard_moon_ric_swap_uses_moon_view_title() -> None:
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.relative_frame = "moon_ric"
    dashboard.plot_view_modes = {"RI": "eci"}

    title = dashboard._panel_title_for_axes("RI Plane: In-Track Vs Radial", x_axis=1, y_axis=0)

    assert title == "Moon View (RI Swap): Tangential Vs Normal"
    assert dashboard._plot_view_mode_for_axes(x_axis=1, y_axis=0) == "eci"


def test_dashboard_moon_ric_swap_uses_real_cr3bp_target_orbit() -> None:
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.relative_frame = "moon_ric"
    dashboard.coast_prediction_model = "cr3bp"
    dashboard.reference_state_eci = cr3bp_halo_seed_state_km_s(family="l2_nrho_southern")
    dashboard.target_coast_prediction_horizon_s = 86400.0
    dashboard.target_coast_prediction_dt_s = 21600.0
    dashboard.cr3bp_coast_prediction_horizon_s = 86400.0
    dashboard.cr3bp_coast_prediction_dt_s = 21600.0
    dashboard.coast_prediction_orbit_fraction = None
    dashboard.coast_prediction_horizon_s = 86400.0
    dashboard._prediction_cache = {}
    dashboard.t_s = [0.0]

    orbit = dashboard._cr3bp_target_orbit_prediction()
    moon = cr3bp_moon_state_km_s()
    moon_centered_radii = np.linalg.norm(orbit[:, :3] - moon[:3], axis=1)
    moon_view_xy = _project_moon_rotating_yz_to_plane(orbit[:, :3] - moon[:3])

    assert orbit.shape == (5, 6)
    assert float(np.ptp(moon_centered_radii)) > 1.0
    assert float(np.min(np.linalg.norm(moon_view_xy, axis=1))) > MOON_RADIUS_KM


def test_dashboard_target_orbit_render_path_uses_cache_only(monkeypatch: pytest.MonkeyPatch) -> None:
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.relative_frame = "moon_ric"
    dashboard.coast_prediction_model = "cr3bp"
    reference = cr3bp_halo_seed_state_km_s(family="l2_nrho_southern")
    dashboard.reference_state_eci = reference.copy()
    dashboard.target_orbit_reference_state_eci = reference.copy()
    dashboard.target_coast_prediction_horizon_s = 86400.0
    dashboard.target_coast_prediction_dt_s = 21600.0
    dashboard.cr3bp_coast_prediction_horizon_s = 86400.0
    dashboard.cr3bp_coast_prediction_dt_s = 21600.0
    dashboard.coast_prediction_orbit_fraction = None
    dashboard.coast_prediction_horizon_s = 86400.0
    dashboard._prediction_cache = {}
    dashboard.t_s = [0.0]

    def fail_propagation(*args, **kwargs):
        raise AssertionError("render path must not build target CR3BP orbit")

    monkeypatch.setattr(game_pygame_dashboard, "propagate_cr3bp_state", fail_propagation)

    assert dashboard._cr3bp_target_orbit_prediction(allow_build=False).size == 0

    cached_prediction = np.vstack([reference, reference + np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])])
    dashboard._prediction_cache["target_absolute_cr3bp_orbit"] = {
        "time_s": 0.0,
        "prediction": cached_prediction,
        "reference": reference.copy(),
        "horizon_s": 86400.0,
        "dt_s": 21600.0,
    }

    np.testing.assert_allclose(
        dashboard._cr3bp_target_orbit_prediction(allow_build=False),
        cached_prediction,
    )


def test_dashboard_moon_ric_target_orbit_is_prepropagated_from_initial_state() -> None:
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.relative_frame = "moon_ric"
    dashboard.coast_prediction_model = "cr3bp"
    initial_target = cr3bp_halo_seed_state_km_s(family="l2_nrho_southern")
    dashboard.reference_state_eci = initial_target.copy()
    dashboard.target_orbit_reference_state_eci = initial_target.copy()
    dashboard.target_coast_prediction_horizon_s = 43200.0
    dashboard.target_coast_prediction_dt_s = 10800.0
    dashboard.cr3bp_coast_prediction_horizon_s = 43200.0
    dashboard.cr3bp_coast_prediction_dt_s = 10800.0
    dashboard.coast_prediction_orbit_fraction = None
    dashboard.coast_prediction_horizon_s = 43200.0
    dashboard._prediction_cache = {}
    dashboard.t_s = [0.0]

    orbit_from_initial = dashboard._cr3bp_target_orbit_prediction()
    dashboard.reference_state_eci = propagate_cr3bp_state(initial_target, 3600.0, 0.0)
    dashboard.t_s = [3600.0]
    dashboard._prediction_cache = {}
    orbit_after_current_state_changes = dashboard._cr3bp_target_orbit_prediction()

    np.testing.assert_allclose(orbit_after_current_state_changes, orbit_from_initial)


def test_dashboard_push_snapshot_prewarms_cr3bp_target_orbit() -> None:
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.target_object_id = "target"
    dashboard.chaser_object_id = "chaser"
    dashboard.reference_object_id = "target"
    dashboard.target_reference_object_id = None
    dashboard.relative_frame = "moon_ric"
    dashboard.coast_prediction_model = "cr3bp"
    dashboard.target_coast_prediction_horizon_s = 600.0
    dashboard.target_coast_prediction_dt_s = 300.0
    dashboard.cr3bp_coast_prediction_horizon_s = 600.0
    dashboard.cr3bp_coast_prediction_dt_s = 300.0
    dashboard.coast_prediction_orbit_fraction = None
    dashboard.coast_prediction_horizon_s = 600.0
    dashboard.max_history = 10
    dashboard.t_s = []
    dashboard.sample_wall_s = []
    dashboard.rel_hist = []
    dashboard.target_rel_hist = []
    dashboard.target_reference_rel_hist = []
    dashboard.target_eci_hist = []
    dashboard.chaser_eci_hist = []
    dashboard.thrust_hist = []
    dashboard.thrust_ric_hist = []
    dashboard._rel_array = np.zeros((0, 6), dtype=float)
    dashboard._target_rel_array = np.zeros((0, 6), dtype=float)
    dashboard._target_reference_rel_array = np.zeros((0, 6), dtype=float)
    dashboard._target_eci_array = np.zeros((0, 6), dtype=float)
    dashboard._chaser_eci_array = np.zeros((0, 6), dtype=float)
    dashboard._thrust_ric_array = np.zeros((0, 3), dtype=float)
    dashboard.target_orbit_reference_state_eci = None
    dashboard.target_true_anomaly_deg = None
    dashboard._prediction_cache = {}
    target = cr3bp_halo_seed_state_km_s(family="l2_nrho_southern")
    chaser = _moon_ric_rect_state_to_cr3bp(
        np.array([-1.0, 2.0, 0.5, 0.0, 0.0, 1.0e-6], dtype=float),
        target,
    )
    snapshot = SimulationSnapshot(
        step_index=0,
        time_s=0.0,
        truth={"target": target, "chaser": chaser},
        belief={},
        applied_thrust={"chaser": np.zeros(3, dtype=float)},
        applied_torque={},
    )

    dashboard.push_snapshot(snapshot)

    cached = dashboard._prediction_cache.get("target_absolute_cr3bp_orbit")
    assert cached is not None
    assert np.array(cached["prediction"], dtype=float).shape == (3, 6)
    np.testing.assert_allclose(
        dashboard._cr3bp_target_orbit_prediction(allow_build=False),
        cached["prediction"],
    )


def test_cr3bp_reference_cache_accepts_propagated_reference_motion() -> None:
    reference = cr3bp_halo_seed_state_km_s(family="l2_nrho_southern")
    current = propagate_cr3bp_state(reference, 10.0, 0.0)

    assert _cr3bp_reference_cache_valid(reference, current, elapsed_s=10.0) is True
    assert _cr3bp_reference_cache_valid(reference, current) is False


def test_dashboard_eci_projection_uses_target_orbit_plane() -> None:
    dashboard = object.__new__(PygameRPODashboard)
    target_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0], dtype=float)

    basis = dashboard._eci_target_plane_basis(target_state)

    assert basis is not None
    i_hat, r_hat, c_hat = basis
    assert i_hat == pytest.approx([0.0, 1.0, 0.0])
    assert r_hat == pytest.approx([1.0, 0.0, 0.0])
    assert c_hat == pytest.approx([0.0, 0.0, 1.0])
    projected = _project_eci_positions_to_plane(
        np.array([[7000.0, 0.0, 0.0], [7000.0, 1.0, 0.5]], dtype=float),
        x_hat=i_hat,
        y_hat=r_hat,
    )
    np.testing.assert_allclose(projected, np.array([[0.0, 7000.0], [1.0, 7000.0]], dtype=float))


def test_satellite_marker_size_uses_physical_plot_scale() -> None:
    assert _satellite_marker_size_px(100.0, 100.0) == 1
    assert _satellite_marker_size_px(1000.0, 1000.0) == 6
    assert _satellite_marker_size_px(5000.0, 5000.0) == 30
    assert _satellite_marker_size_px(100000.0, 100000.0) == 600
    assert _satellite_marker_size_px(100.0, 100.0, diameter_km=0.05) == 5
    assert _satellite_marker_size_px(100.0, 100.0, diameter_km=0.11) == 11
    assert _satellite_marker_size_px(100.0, 100.0, diameter_km=0.12) == 12


def test_satellite_marker_reticle_scales_with_sprite_size() -> None:
    assert _satellite_marker_reticle_radii_px(0) == (0, 0)
    assert _satellite_marker_reticle_radii_px(5) == (2, 4)
    assert _satellite_marker_reticle_radii_px(11) == (2, 4)
    assert _satellite_marker_reticle_radii_px(20) == (2, 4)
    assert _satellite_marker_reticle_radii_px(60) == (3, 6)


def test_dashboard_vectors_match_web_preview_scaling() -> None:
    rel = np.array([0.0, -5.0, 0.0, 0.001, -0.002, 0.003], dtype=float)
    thrust = np.array([0.0, 3.0e-5, 4.0e-5], dtype=float)

    velocity_px = PygameRPODashboard._web_velocity_vector_px(rel, x_axis=1, y_axis=0)
    thrust_px = PygameRPODashboard._web_thrust_vector_px(thrust, x_axis=1, y_axis=2)
    hidden_axis_thrust_px = PygameRPODashboard._web_thrust_vector_px(thrust, x_axis=1, y_axis=0)

    assert velocity_px == pytest.approx([-150.0, 75.0])
    assert thrust_px == pytest.approx([25.2, 33.6])
    assert np.linalg.norm(thrust_px) == pytest.approx(42.0)
    assert hidden_axis_thrust_px == pytest.approx([25.2, 0.0])
    assert PygameRPODashboard._web_thrust_vector_px(thrust, x_axis=1, y_axis=2, threshold=1.0e-4) == pytest.approx(
        [0.0, 0.0]
    )


def test_dashboard_visual_extrapolation_advances_latest_row_only() -> None:
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.visual_extrapolation_enabled = True
    dashboard.visual_extrapolation_max_sim_s = 1.0
    dashboard._render_motion_enabled = True
    dashboard._render_speed_multiple = 1.0
    dashboard._render_wall_time_s = 10.5
    dashboard.sample_wall_s = [9.0, 10.0]
    dashboard.t_s = [0.0, 1.0]
    rows = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0, 0.2, -0.4, 0.6],
        ],
        dtype=float,
    )

    visual = dashboard._visual_state_rows(rows)

    assert visual[0] == pytest.approx(rows[0])
    assert visual[1, :3] == pytest.approx([1.1, 1.8, 3.3])
    assert visual[1, 3:6] == pytest.approx(rows[1, 3:6])
    assert rows[1, :3] == pytest.approx([1.0, 2.0, 3.0])


def test_dashboard_visual_extrapolation_can_be_disabled_and_is_capped() -> None:
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.visual_extrapolation_enabled = True
    dashboard.visual_extrapolation_max_sim_s = 5.0
    dashboard._render_motion_enabled = True
    dashboard._render_speed_multiple = 10.0
    dashboard._render_wall_time_s = 12.0
    dashboard.sample_wall_s = [10.0]
    dashboard.t_s = [0.0]
    rows = np.array([[1.0, 2.0, 3.0, 0.2, 0.0, 0.0]], dtype=float)

    visual = dashboard._visual_state_rows(rows)
    dashboard.visual_extrapolation_enabled = False
    disabled = dashboard._visual_state_rows(rows)

    assert visual[0, :3] == pytest.approx([2.0, 2.0, 3.0])
    assert disabled[0, :3] == pytest.approx([1.0, 2.0, 3.0])


def test_dashboard_frame_cache_samples_shared_draw_rows_once() -> None:
    dashboard = object.__new__(PygameRPODashboard)
    rows = 500
    rel = np.zeros((rows, 6), dtype=float)
    rel[:, 1] = np.linspace(-10.0, 10.0, rows)
    target_rel = np.zeros((rows, 6), dtype=float)
    target_rel[:, 0] = np.linspace(0.0, 1.0, rows)
    thrust = np.zeros((rows, 3), dtype=float)
    thrust[::3, 1] = 1.0e-5
    ghost = np.zeros((300, 6), dtype=float)
    ghost[:, 1] = np.linspace(0.0, 30.0, 300)
    target_ghost = np.zeros((240, 6), dtype=float)
    target_ghost[:, 0] = np.linspace(0.0, 12.0, 240)
    dashboard.rel_hist = [row for row in rel]
    dashboard.target_rel_hist = [row for row in target_rel]
    dashboard.thrust_ric_hist = [row for row in thrust]
    dashboard._rel_array = rel
    dashboard._target_rel_array = target_rel
    dashboard._thrust_ric_array = thrust
    dashboard.max_history = 900
    dashboard.burn_marker_threshold_km_s2 = 1.0e-12
    dashboard._frame_cache = {}
    dashboard._frame_cache_dirty = True
    dashboard._coast_prediction_from_cached = lambda *_, **__: ghost
    dashboard._target_coast_prediction = lambda *_: target_ghost
    dashboard._nmt_points = lambda: np.empty((0, 3), dtype=float)
    dashboard._nmt_boundary_points = lambda: ()

    dashboard._prepare_frame_cache()

    assert dashboard._frame_cache_dirty is False
    assert dashboard._frame_cache["rel_trail"].shape[0] <= 260
    assert dashboard._frame_cache["target_trail"].shape[0] <= 260
    assert dashboard._frame_cache["ghost_sample"].shape[0] <= 120
    assert dashboard._frame_cache["target_ghost_sample"].shape[0] <= 120
    assert dashboard._frame_cache["burn_marker_rel"].shape[0] <= 80
    assert np.allclose(dashboard._frame_cache["rel_trail"][0], rel[0])
    assert np.allclose(dashboard._frame_cache["rel_trail"][-1], rel[-1])


def test_dashboard_frame_cache_reuses_raw_predictions_between_visual_frames() -> None:
    dashboard = object.__new__(PygameRPODashboard)
    rel = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0, 0.2, -0.4, 0.6],
        ],
        dtype=float,
    )
    target_rel = np.zeros((2, 6), dtype=float)
    target_reference_rel = np.zeros((2, 6), dtype=float)
    thrust = np.zeros((2, 3), dtype=float)
    ghost = np.zeros((2, 6), dtype=float)
    target_ghost = np.ones((2, 6), dtype=float)
    dashboard.rel_hist = [row for row in rel]
    dashboard.target_rel_hist = [row for row in target_rel]
    dashboard.target_reference_rel_hist = [row for row in target_reference_rel]
    dashboard.thrust_ric_hist = [row for row in thrust]
    dashboard._rel_array = rel
    dashboard._target_rel_array = target_rel
    dashboard._target_reference_rel_array = target_reference_rel
    dashboard._thrust_ric_array = thrust
    dashboard.max_history = 900
    dashboard.burn_marker_threshold_km_s2 = 1.0e-12
    dashboard.visual_extrapolation_enabled = True
    dashboard.visual_extrapolation_max_sim_s = 1.0
    dashboard._render_motion_enabled = True
    dashboard._render_speed_multiple = 1.0
    dashboard.sample_wall_s = [9.0, 10.0]
    dashboard.t_s = [0.0, 1.0]
    dashboard.live_prediction_accel_ric_km_s2 = np.zeros(3, dtype=float)
    dashboard.live_prediction_elapsed_s = 0.0
    dashboard._frame_cache = {}
    dashboard._raw_frame_cache = {}
    dashboard._frame_cache_dirty = True
    calls = {"target_ghost": 0, "nmt": 0, "nmt_bounds": 0, "ghost": 0}

    def fake_ghost(*_, **__):
        calls["ghost"] += 1
        return ghost

    def fake_target_ghost(*_):
        calls["target_ghost"] += 1
        return target_ghost

    def fake_nmt():
        calls["nmt"] += 1
        return np.empty((0, 3), dtype=float)

    def fake_nmt_bounds():
        calls["nmt_bounds"] += 1
        return ()

    dashboard._coast_prediction_from_cached = fake_ghost
    dashboard._target_coast_prediction = fake_target_ghost
    dashboard._nmt_points = fake_nmt
    dashboard._nmt_boundary_points = fake_nmt_bounds

    dashboard._render_wall_time_s = 10.25
    dashboard._prepare_frame_cache()
    first_latest = dashboard._frame_cache["rel"][-1, :3].copy()
    dashboard._render_wall_time_s = 10.75
    dashboard._prepare_frame_cache()

    assert calls == {"target_ghost": 1, "nmt": 1, "nmt_bounds": 1, "ghost": 2}
    assert first_latest == pytest.approx([1.05, 1.9, 3.15])
    assert dashboard._frame_cache["rel"][-1, :3] == pytest.approx([1.15, 1.7, 3.45])
    assert dashboard._raw_frame_cache["raw_rel"][-1, :3] == pytest.approx([1.0, 2.0, 3.0])


def test_dashboard_frame_cache_reuses_prepared_static_frame() -> None:
    dashboard = object.__new__(PygameRPODashboard)
    rel = np.array([[1.0, 2.0, 3.0, 0.2, -0.4, 0.6]], dtype=float)
    target_rel = np.zeros((1, 6), dtype=float)
    target_reference_rel = np.zeros((1, 6), dtype=float)
    thrust = np.zeros((1, 3), dtype=float)
    dashboard.rel_hist = [rel[0]]
    dashboard.target_rel_hist = [target_rel[0]]
    dashboard.target_reference_rel_hist = [target_reference_rel[0]]
    dashboard.thrust_ric_hist = [thrust[0]]
    dashboard._rel_array = rel
    dashboard._target_rel_array = target_rel
    dashboard._target_reference_rel_array = target_reference_rel
    dashboard._thrust_ric_array = thrust
    dashboard.max_history = 900
    dashboard.burn_marker_threshold_km_s2 = 1.0e-12
    dashboard._render_motion_enabled = False
    dashboard.live_prediction_accel_ric_km_s2 = np.zeros(3, dtype=float)
    dashboard.live_prediction_elapsed_s = 0.0
    dashboard._frame_cache = {}
    dashboard._raw_frame_cache = {}
    dashboard._frame_cache_dirty = True
    calls = {"ghost": 0}

    def fake_ghost(*_, **__):
        calls["ghost"] += 1
        return np.zeros((2, 6), dtype=float)

    dashboard._coast_prediction_from_cached = fake_ghost
    dashboard._target_coast_prediction = lambda *_: np.empty((0, 6), dtype=float)
    dashboard._nmt_points = lambda: np.empty((0, 3), dtype=float)
    dashboard._nmt_boundary_points = lambda: ()

    dashboard._prepare_frame_cache()
    first_cache = dashboard._frame_cache
    dashboard._prepare_frame_cache()

    assert calls == {"ghost": 1}
    assert dashboard._frame_cache is first_cache


def test_dashboard_history_array_tail_keeps_chronological_ring_tail() -> None:
    dashboard = object.__new__(PygameRPODashboard)
    history: object = np.zeros((0, 6), dtype=float)
    for idx in range(5):
        row = np.full(6, float(idx), dtype=float)
        history = _history_array_tail(history, row, width=6, max_rows=3)
    dashboard._rel_array = history

    rows = _dashboard_history_array(dashboard, "_rel_array", [], width=6)

    assert rows.shape == (3, 6)
    assert rows[:, 0].tolist() == [2.0, 3.0, 4.0]


def test_dashboard_live_prediction_seed_moves_ghost_without_moving_truth() -> None:
    dashboard = object.__new__(PygameRPODashboard)
    rel = np.zeros((1, 6), dtype=float)
    target_rel = np.zeros((1, 6), dtype=float)
    thrust = np.zeros((1, 3), dtype=float)
    dashboard.rel_hist = [rel[0]]
    dashboard.target_rel_hist = [target_rel[0]]
    dashboard.thrust_ric_hist = [thrust[0]]
    dashboard._rel_array = rel
    dashboard._target_rel_array = target_rel
    dashboard._thrust_ric_array = thrust
    dashboard.max_history = 900
    dashboard.burn_marker_threshold_km_s2 = 1.0e-12
    dashboard.mean_motion_rad_s = 0.001
    dashboard.live_prediction_accel_ric_km_s2 = np.array([0.0, 1.0e-5, 0.0], dtype=float)
    dashboard.live_prediction_elapsed_s = 0.5
    dashboard._frame_cache = {}
    dashboard._frame_cache_dirty = True
    seeds = []

    def fake_prediction(cache_name, rel0, *, active_burn):
        seeds.append((cache_name, np.array(rel0, dtype=float), active_burn))
        return np.zeros((2, 6), dtype=float)

    dashboard._coast_prediction_from_cached = fake_prediction
    dashboard._target_coast_prediction = lambda *_: np.empty((0, 6), dtype=float)
    dashboard._nmt_points = lambda: np.empty((0, 3), dtype=float)
    dashboard._nmt_boundary_points = lambda: ()

    dashboard._prepare_frame_cache()

    assert np.allclose(dashboard._frame_cache["rel"][-1], rel[-1])
    assert seeds[0][0] == "chaser"
    assert seeds[0][1][4] > 0.0
    assert seeds[0][2] is True
    assert dashboard._frame_cache["ghost_active_burn"] is True


def test_dashboard_cr3bp_live_prediction_seed_moves_with_partial_burn() -> None:
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.coast_prediction_model = "cr3bp"
    dashboard.relative_frame = "moon_ric"
    dashboard.rel_hist = [np.zeros(6, dtype=float)]
    dashboard.target_rel_hist = [np.zeros(6, dtype=float)]
    dashboard.thrust_ric_hist = [np.zeros(3, dtype=float)]
    dashboard._rel_array = np.zeros((1, 6), dtype=float)
    dashboard._target_rel_array = np.zeros((1, 6), dtype=float)
    dashboard._thrust_ric_array = np.zeros((1, 3), dtype=float)
    dashboard.max_history = 900
    dashboard.burn_marker_threshold_km_s2 = 1.0e-12
    dashboard.mean_motion_rad_s = EARTH_MOON_MEAN_MOTION_RAD_S
    dashboard.reference_state_eci = cr3bp_halo_seed_state_km_s(family="l2_nrho_southern")
    dashboard.live_prediction_accel_ric_km_s2 = np.array([0.0, 1.0e-5, 0.0], dtype=float)
    dashboard.live_prediction_elapsed_s = 0.5
    dashboard._frame_cache = {}
    dashboard._raw_frame_cache = {}
    dashboard._frame_cache_dirty = True
    seeds = []

    def fake_prediction(cache_name, rel0, *, active_burn):
        seeds.append((cache_name, np.array(rel0, dtype=float), active_burn))
        return np.zeros((2, 6), dtype=float)

    dashboard._coast_prediction_from_cached = fake_prediction
    dashboard._target_coast_prediction = lambda *_: np.empty((0, 6), dtype=float)
    dashboard._nmt_points = lambda: np.empty((0, 3), dtype=float)
    dashboard._nmt_boundary_points = lambda: ()

    dashboard._prepare_frame_cache()

    assert seeds[0][2] is True
    assert seeds[0][1][1] == pytest.approx(0.5 * 1.0e-5 * 0.5 * 0.5)
    assert seeds[0][1][4] == pytest.approx(1.0e-5 * 0.5)


def test_dashboard_stale_snapshot_thrust_does_not_keep_live_burn_active() -> None:
    dashboard = object.__new__(PygameRPODashboard)
    rel = np.zeros((1, 6), dtype=float)
    target_rel = np.zeros((1, 6), dtype=float)
    thrust = np.array([[0.0, 1.0e-5, 0.0]], dtype=float)
    dashboard.rel_hist = [rel[0]]
    dashboard.target_rel_hist = [target_rel[0]]
    dashboard.thrust_ric_hist = [thrust[0]]
    dashboard._rel_array = rel
    dashboard._target_rel_array = target_rel
    dashboard._thrust_ric_array = thrust
    dashboard.max_history = 900
    dashboard.burn_marker_threshold_km_s2 = 1.0e-12
    dashboard.mean_motion_rad_s = 0.001
    dashboard.live_prediction_accel_ric_km_s2 = np.zeros(3, dtype=float)
    dashboard.live_prediction_elapsed_s = 0.0
    dashboard._frame_cache = {}
    dashboard._raw_frame_cache = {}
    dashboard._frame_cache_dirty = True
    calls = []

    def fake_prediction(cache_name, rel0, *, active_burn):
        calls.append((cache_name, np.array(rel0, dtype=float), active_burn))
        return np.zeros((2, 6), dtype=float)

    dashboard._coast_prediction_from_cached = fake_prediction
    dashboard._target_coast_prediction = lambda *_: np.empty((0, 6), dtype=float)
    dashboard._nmt_points = lambda: np.empty((0, 3), dtype=float)
    dashboard._nmt_boundary_points = lambda: ()

    dashboard._prepare_frame_cache()

    assert calls[0][2] is False
    assert dashboard._frame_cache["ghost_active_burn"] is False


def test_dashboard_live_prediction_seed_anchors_to_committed_sample() -> None:
    dashboard = object.__new__(PygameRPODashboard)
    rel = np.array([[1.0, 2.0, 3.0, 0.2, -0.4, 0.6]], dtype=float)
    target_rel = np.zeros((1, 6), dtype=float)
    thrust = np.zeros((1, 3), dtype=float)
    dashboard.rel_hist = [rel[0]]
    dashboard.target_rel_hist = [target_rel[0]]
    dashboard.thrust_ric_hist = [thrust[0]]
    dashboard._rel_array = rel
    dashboard._target_rel_array = target_rel
    dashboard._thrust_ric_array = thrust
    dashboard.max_history = 900
    dashboard.burn_marker_threshold_km_s2 = 1.0e-12
    dashboard.mean_motion_rad_s = 0.001
    dashboard.live_prediction_accel_ric_km_s2 = np.array([0.0, 1.0e-5, 0.0], dtype=float)
    dashboard.live_prediction_elapsed_s = 0.5
    dashboard.visual_extrapolation_enabled = True
    dashboard.visual_extrapolation_max_sim_s = 1.0
    dashboard._render_motion_enabled = True
    dashboard._render_speed_multiple = 1.0
    dashboard.sample_wall_s = [10.0]
    dashboard.t_s = [0.0]
    dashboard._render_wall_time_s = 10.75
    dashboard._frame_cache = {}
    dashboard._raw_frame_cache = {}
    dashboard._frame_cache_dirty = True
    seeds = []

    def fake_prediction(cache_name, rel0, *, active_burn):
        seeds.append((cache_name, np.array(rel0, dtype=float), active_burn))
        return np.zeros((2, 6), dtype=float)

    dashboard._coast_prediction_from_cached = fake_prediction
    dashboard._target_coast_prediction = lambda *_: np.empty((0, 6), dtype=float)
    dashboard._nmt_points = lambda: np.empty((0, 3), dtype=float)
    dashboard._nmt_boundary_points = lambda: ()

    dashboard._prepare_frame_cache()

    visual_latest = dashboard._frame_cache["rel"][-1]
    expected_seed = _cw_forced_state(
        rel[0],
        dashboard.live_prediction_accel_ric_km_s2,
        dashboard.live_prediction_elapsed_s,
        dashboard.mean_motion_rad_s,
    )

    assert visual_latest[:3] == pytest.approx([1.15, 1.7, 3.45])
    assert seeds[0][1] == pytest.approx(expected_seed)
    visual_seed = _cw_forced_state(
        visual_latest,
        dashboard.live_prediction_accel_ric_km_s2,
        dashboard.live_prediction_elapsed_s,
        dashboard.mean_motion_rad_s,
    )
    assert float(np.linalg.norm(seeds[0][1] - visual_seed)) > 1.0e-6


def test_set_live_prediction_burn_marks_frame_cache_dirty() -> None:
    dashboard = object.__new__(PygameRPODashboard)
    dashboard._frame_cache_dirty = False
    dashboard.live_prediction_accel_ric_km_s2 = np.zeros(3, dtype=float)
    dashboard.live_prediction_elapsed_s = 0.0

    dashboard.set_live_prediction_burn(np.array([1.0e-5, 0.0, 0.0], dtype=float), 0.25)

    assert dashboard._frame_cache_dirty is True


def test_set_live_prediction_burn_marks_frame_cache_dirty_on_elapsed_change() -> None:
    dashboard = object.__new__(PygameRPODashboard)
    dashboard._frame_cache_dirty = False
    dashboard.live_prediction_accel_ric_km_s2 = np.array([1.0e-5, 0.0, 0.0], dtype=float)
    dashboard.live_prediction_elapsed_s = 0.25

    dashboard.set_live_prediction_burn(np.array([1.0e-5, 0.0, 0.0], dtype=float), 0.5)

    assert dashboard._frame_cache_dirty is True


def test_coast_prediction_caps_draw_points() -> None:
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.mean_motion_rad_s = 0.001
    dashboard.coast_prediction_orbit_fraction = 1.0
    dashboard.coast_prediction_horizon_s = 300.0
    dashboard.coast_prediction_dt_s = 1.0

    prediction = dashboard._coast_prediction_from(np.array([0.1, -1.0, 0.2, 0.0, 0.001, -0.001], dtype=float))

    assert 2 <= prediction.shape[0] <= 120


def test_front_loaded_prediction_times_preserve_near_term_resolution() -> None:
    times = _front_loaded_prediction_times(259200.0, 1.0, max_points=120)

    assert times.shape == (120,)
    np.testing.assert_allclose(times[:6], np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float))
    assert times[-1] == pytest.approx(259200.0)
    assert np.all(np.diff(times) > 0.0)
    assert np.max(np.diff(times[:80])) == pytest.approx(1.0)


def test_front_loaded_prediction_times_support_long_cislunar_horizon() -> None:
    times = _front_loaded_prediction_times(604800.0, 1.0, max_points=120)

    assert times.shape == (120,)
    np.testing.assert_allclose(times[:6], np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float))
    assert times[-1] == pytest.approx(604800.0)
    assert np.max(np.diff(times[:80])) == pytest.approx(1.0)


def test_dashboard_uses_elliptic_prediction_model_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.mean_motion_rad_s = 0.001
    dashboard.reference_state_eci = np.array([9000.0, 0.0, 0.0, 0.0, 6.0, 0.0], dtype=float)
    dashboard.coast_prediction_orbit_fraction = None
    dashboard.coast_prediction_horizon_s = 120.0
    dashboard.coast_prediction_dt_s = 60.0
    dashboard.coast_prediction_model = "tschauner_hempel"
    rel0 = np.array([0.1, -1.0, 0.2, 0.0, 0.001, -0.001], dtype=float)
    calls: list[float] = []
    original = game_pygame_dashboard.ya_closed_form_transition_matrix

    def spy_ya(dt_s: float, chief_start_eci_km_s: np.ndarray, chief_end_eci_km_s: np.ndarray, **kwargs: Any) -> Any:
        calls.append(float(dt_s))
        return original(dt_s, chief_start_eci_km_s, chief_end_eci_km_s, **kwargs)

    monkeypatch.setattr(game_pygame_dashboard, "ya_closed_form_transition_matrix", spy_ya)

    prediction = dashboard._coast_prediction_from(rel0)

    assert calls == pytest.approx([0.0, 60.0, 120.0])
    assert prediction.shape == (3, 6)
    assert np.allclose(prediction[0], rel0)
    assert not np.allclose(prediction[-1], _cw_coast_state(rel0, 120.0, 0.001))


def test_elliptic_prediction_cache_throttles_coast_but_refreshes_burns() -> None:
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.coast_prediction_model = "tschauner_hempel"
    dashboard.t_s = [0.0]
    reference = np.array([9000.0, 0.0, 0.0, 0.0, 6.0, 0.0], dtype=float)
    dashboard.reference_state_eci = reference.copy()
    dashboard._prediction_cache = {}
    calls = []

    def fake_prediction(rel0):
        calls.append(np.array(rel0, dtype=float).copy())
        return np.full((2, 6), float(len(calls)), dtype=float)

    dashboard._coast_prediction_from = fake_prediction
    rel0 = np.array([0.1, -1.0, 0.2, 0.0, 0.001, -0.001], dtype=float)

    first = dashboard._coast_prediction_from_cached("chaser", rel0, active_burn=False)
    dashboard.t_s = [10.0]
    dashboard.reference_state_eci = _two_body_coast_state(reference, 10.0)
    coasting = dashboard._coast_prediction_from_cached("chaser", rel0 + 1.0, active_burn=False)
    dashboard.t_s = [11.0]
    burning = dashboard._coast_prediction_from_cached("chaser", rel0 + 2.0, active_burn=True)

    assert len(calls) == 2
    assert np.all(first == 1.0)
    assert np.all(coasting == 1.0)
    assert np.all(burning == 2.0)


def test_elliptic_prediction_cache_refreshes_when_reference_maneuvers() -> None:
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.coast_prediction_model = "tschauner_hempel"
    dashboard.t_s = [0.0]
    reference = np.array([9000.0, 0.0, 0.0, 0.0, 6.0, 0.0], dtype=float)
    dashboard.reference_state_eci = reference.copy()
    dashboard._prediction_cache = {}
    calls = []

    def fake_prediction(rel0):
        calls.append(np.array(rel0, dtype=float).copy())
        return np.full((2, 6), float(len(calls)), dtype=float)

    dashboard._coast_prediction_from = fake_prediction
    rel0 = np.array([0.1, -1.0, 0.2, 0.0, 0.001, -0.001], dtype=float)

    first = dashboard._coast_prediction_from_cached("chaser", rel0, active_burn=False)
    dashboard.t_s = [10.0]
    maneuvered_reference = _two_body_coast_state(reference, 10.0)
    maneuvered_reference[3] += 2.0e-5
    dashboard.reference_state_eci = maneuvered_reference
    refreshed = dashboard._coast_prediction_from_cached("chaser", rel0 + 1.0, active_burn=False)

    assert len(calls) == 2
    assert np.all(first == 1.0)
    assert np.all(refreshed == 2.0)


def test_target_hcw_path_uses_target_relative_state_when_enabled() -> None:
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.mean_motion_rad_s = 0.001
    dashboard.coast_prediction_orbit_fraction = None
    dashboard.coast_prediction_horizon_s = 20.0
    dashboard.coast_prediction_dt_s = 10.0
    dashboard.show_target_coast_prediction = True
    target_rel = np.array([[0.1, -1.0, 0.2, 0.0, 0.001, -0.001]], dtype=float)

    prediction = dashboard._target_coast_prediction(target_rel)

    assert prediction.shape[0] == 3
    assert np.allclose(prediction[0], target_rel[0])


def test_cislunar_target_path_uses_target_specific_full_horizon() -> None:
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.coast_prediction_model = "cr3bp"
    dashboard.coast_prediction_orbit_fraction = 1.0
    dashboard.coast_prediction_horizon_s = 300.0
    dashboard.coast_prediction_dt_s = 10.0
    dashboard.cr3bp_coast_prediction_horizon_s = 21600.0
    dashboard.cr3bp_coast_prediction_dt_s = 300.0
    dashboard.target_coast_prediction_horizon_s = 172800.0
    dashboard.target_coast_prediction_dt_s = 1800.0
    dashboard.show_target_coast_prediction = True
    dashboard.mean_motion_rad_s = EARTH_MOON_MEAN_MOTION_RAD_S
    dashboard.t_s = [0.0]
    target_rel = np.array([[100.0, 0.0, 50.0, 0.0, 0.01, 0.0]], dtype=float)

    prediction = dashboard._target_coast_prediction(target_rel)

    assert prediction.shape == (97, 6)
    assert prediction[0] == pytest.approx(target_rel[0])


def test_target_hcw_path_is_disabled_by_default() -> None:
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.show_target_coast_prediction = False
    dashboard.target_rel_hist = [np.array([0.1, -1.0, 0.2, 0.0, 0.001, -0.001], dtype=float)]

    assert dashboard._target_coast_prediction().size == 0


def test_plot_scale_uses_current_satellites_not_old_trail_or_overlays() -> None:
    class FakeScreen:
        @staticmethod
        def get_size():
            return (1280, 720)

    dashboard = object.__new__(PygameRPODashboard)
    dashboard.screen = FakeScreen()
    dashboard.goal_nmt_radial_amplitude_km = None
    dashboard.goal_radius_km = 0.025
    dashboard.goal_range_km = 2.0
    dashboard.goal_range_tolerance_km = 0.5
    dashboard.keepout_radius_km = None
    dashboard.goal_relative_ric_km = np.zeros(3, dtype=float)
    dashboard.target_rel_hist = []

    close_rel = np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    far_trail = np.array([[0.0, -0.75], [0.0, 0.0], [0.0, 0.02]], dtype=float)
    close_scale = dashboard._scale_for_plot(pts=[close_rel[:2].reshape(1, 2), np.zeros((1, 2), dtype=float)])
    history_scale = dashboard._scale_for_plot(pts=[far_trail])

    assert close_scale > history_scale
    assert close_scale == pytest.approx(dashboard._scale_for_plot(pts=[close_rel[:2].reshape(1, 2)]))


def test_camera_rule_toggle_switches_between_current_pair_and_full_trajectory_scale() -> None:
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.camera_rule_mode = "current_pair"
    dashboard._frame_cache_dirty = False
    dashboard.plot_prediction_in_zoom = True
    dashboard.plot_prediction_zoom_max_span_km = None
    rel = np.array(
        [
            [0.0, -0.2, 0.0, 0.0, 0.0, 0.0],
            [0.0, -4.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    target_rel = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    ghost = np.array(
        [
            [0.0, -0.2, 0.0, 0.0, 0.0, 0.0],
            [0.0, -8.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    target_ghost = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 12.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    assert dashboard._camera_rule_mode_key() == "current_pair"
    current_points = dashboard._camera_rule_scale_points(
        rel=rel,
        target_rel=target_rel,
        ghost=ghost,
        target_ghost=target_ghost,
        x_axis=1,
        y_axis=0,
        camera_center=np.zeros(3, dtype=float),
    )
    assert len(current_points) == 1
    assert np.max(np.abs(np.vstack(current_points))) == pytest.approx(8.0)
    assert dashboard.toggle_camera_rule_mode() == "full_trajectory"
    assert dashboard._frame_cache_dirty is True

    points = dashboard._camera_rule_scale_points(
        rel=rel,
        target_rel=target_rel,
        ghost=ghost,
        target_ghost=target_ghost,
        x_axis=1,
        y_axis=0,
        camera_center=np.zeros(3, dtype=float),
    )

    assert dashboard._camera_rule_mode_key() == "full_trajectory"
    assert len(points) == 4
    assert np.max(np.abs(np.vstack(points))) == pytest.approx(12.0)
    assert dashboard.toggle_camera_rule_mode() == "current_pair"


def test_default_camera_rule_preserves_prediction_zoom_scaling() -> None:
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.camera_rule_mode = "default"
    dashboard.plot_prediction_in_zoom = True
    dashboard.plot_prediction_zoom_max_span_km = 5.0
    ghost = np.array(
        [
            [0.0, -2.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -8.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    points = dashboard._camera_rule_scale_points(
        rel=np.empty((0, 6), dtype=float),
        target_rel=np.empty((0, 6), dtype=float),
        ghost=ghost,
        x_axis=1,
        y_axis=0,
        camera_center=np.zeros(3, dtype=float),
    )

    assert dashboard._camera_rule_mode_key() == "default"
    assert len(points) == 1
    assert np.max(np.abs(points[0])) == pytest.approx(5.0)


def test_full_trajectory_only_prediction_scales_only_in_full_trajectory_camera() -> None:
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.camera_mode = "rule_toggle_pair"
    dashboard.camera_rule_mode = "current_pair"
    dashboard.plot_prediction_full_trajectory_only = True

    assert dashboard._prediction_scales_current_camera() is False

    dashboard.camera_rule_mode = "full_trajectory"

    assert dashboard._prediction_scales_current_camera() is True


def test_target_pair_camera_centers_ri_between_current_satellites() -> None:
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.camera_mode = "target_pair"
    dashboard.target_centered_plot_planes = ()
    dashboard.target_centered_plot_axes = {}

    center = dashboard._camera_center_ric(
        chaser_current=np.array([0.2, -1.0, 0.4], dtype=float),
        target_current=np.array([-0.4, 0.5, -0.2], dtype=float),
        x_axis=1,
        y_axis=0,
    )

    assert np.allclose(center, np.array([-0.1, -0.25, 0.0], dtype=float))


def test_level_zero_target_pair_camera_keeps_target_centered() -> None:
    config_path = (
        Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_00_tutorial.yaml"
    )
    config = SimulationConfig.from_yaml(config_path)
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.camera_mode = _game_camera_mode(config)
    dashboard.target_centered_plot_planes = _game_target_centered_plot_planes(config)
    dashboard.target_centered_plot_axes = _game_target_centered_plot_axes(config)
    chaser = np.array([0.2, -1.0, 0.4], dtype=float)
    target = np.array([-0.4, 0.5, -0.2], dtype=float)

    ri_center = dashboard._camera_center_ric(
        chaser_current=chaser,
        target_current=target,
        x_axis=1,
        y_axis=0,
    )
    rc_center = dashboard._camera_center_ric(
        chaser_current=chaser,
        target_current=target,
        x_axis=2,
        y_axis=0,
    )

    assert ri_center == pytest.approx(target)
    assert rc_center == pytest.approx(target)


def test_target_pair_camera_keeps_reference_centered_for_rc() -> None:
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.camera_mode = "target_pair"
    dashboard.target_centered_plot_planes = ()
    dashboard.target_centered_plot_axes = {}

    center = dashboard._camera_center_ric(
        chaser_current=np.array([0.2, -1.0, 0.4], dtype=float),
        target_current=np.array([-0.4, 0.5, -0.2], dtype=float),
        x_axis=2,
        y_axis=0,
    )

    assert np.allclose(center, np.zeros(3, dtype=float))


def test_rule_toggle_pair_camera_switches_between_midpoint_and_reference() -> None:
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.camera_mode = "rule_toggle_pair"
    dashboard.camera_rule_mode = "current_pair"
    chaser = np.array([0.2, -1.0, 0.4], dtype=float)
    target = np.array([-0.4, 0.5, -0.2], dtype=float)

    current_center = dashboard._camera_center_ric(
        chaser_current=chaser,
        target_current=target,
        x_axis=2,
        y_axis=0,
    )
    dashboard.camera_rule_mode = "full_trajectory"
    full_center = dashboard._camera_center_ric(
        chaser_current=chaser,
        target_current=target,
        x_axis=2,
        y_axis=0,
    )

    assert current_center == pytest.approx(np.array([-0.1, 0.0, 0.1], dtype=float))
    assert full_center == pytest.approx(np.zeros(3, dtype=float))


def test_target_centered_plane_override_keeps_ri_on_target() -> None:
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.camera_mode = "target_pair"
    dashboard.target_centered_plot_planes = ("RI",)
    dashboard.target_centered_plot_axes = {}

    center = dashboard._camera_center_ric(
        chaser_current=np.array([0.2, -1.0, 0.4], dtype=float),
        target_current=np.array([-0.4, 0.5, -0.2], dtype=float),
        x_axis=1,
        y_axis=0,
    )
    rc_center = dashboard._camera_center_ric(
        chaser_current=np.array([0.2, -1.0, 0.4], dtype=float),
        target_current=np.array([-0.4, 0.5, -0.2], dtype=float),
        x_axis=2,
        y_axis=0,
    )

    assert np.allclose(center, np.array([-0.4, 0.5, -0.2], dtype=float))
    assert np.allclose(rc_center, np.zeros(3, dtype=float))


def test_target_centered_axis_override_locks_only_requested_plot_axis() -> None:
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.camera_mode = "target_pair"
    dashboard.target_centered_plot_planes = ()
    dashboard.target_centered_plot_axes = {"RI": ("y",)}

    center = dashboard._camera_center_ric(
        chaser_current=np.array([0.2, -1.0, 0.4], dtype=float),
        target_current=np.array([-0.4, 0.5, -0.2], dtype=float),
        x_axis=1,
        y_axis=0,
    )

    assert np.allclose(center, np.array([-0.4, -0.25, 0.0], dtype=float))


def test_reference_camera_keeps_reference_origin_centered_by_default() -> None:
    dashboard = object.__new__(PygameRPODashboard)

    center = dashboard._camera_center_ric(
        chaser_current=np.array([0.2, -1.0, 0.4], dtype=float),
        target_current=np.array([-0.4, 0.5, -0.2], dtype=float),
    )

    assert np.allclose(center, np.zeros(3, dtype=float))


def test_level5_plot_scale_keeps_forbidden_region_and_gates_visible() -> None:
    class FakeScreen:
        @staticmethod
        def get_size():
            return (1280, 720)

    root = Path(__file__).resolve().parents[1] / "game" / "configs"
    cfg = RPOTrainingConfig.from_metadata(
        dict(
            SimulationConfig.from_yaml(
                root / "game_training_rpo_05_passive_cross_track_approach.yaml"
            ).scenario.metadata
            or {}
        )
    )
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.screen = FakeScreen()
    dashboard.forbidden_regions = cfg.forbidden_regions
    dashboard.approach_gates = ()
    dashboard.inspection_gates = cfg.inspection_gates

    rc_min_span = dashboard._minimum_plot_span_km(x_axis=2, y_axis=0, offset=np.zeros(3, dtype=float))
    ri_min_span = dashboard._minimum_plot_span_km(x_axis=1, y_axis=0, offset=np.zeros(3, dtype=float))
    close_scale = dashboard._scale_for_plot(
        pts=[np.array([[0.0, 0.0]], dtype=float)],
        min_span_km=rc_min_span,
    )
    wide_scale = dashboard._scale_for_plot(
        pts=[np.array([[2.5, 0.0]], dtype=float)],
        min_span_km=rc_min_span,
    )

    assert rc_min_span == pytest.approx(1.0 * 1.18)
    assert ri_min_span == pytest.approx(1.75 * 1.18)
    assert close_scale == pytest.approx(dashboard._scale_for_plot(pts=[], min_span_km=rc_min_span))
    assert wide_scale < close_scale


def test_level6_plot_scale_includes_capped_projection() -> None:
    root = Path(__file__).resolve().parents[1] / "game" / "configs"
    sim_cfg = SimulationConfig.from_yaml(root / "game_training_rpo_07_elliptic_burn_then_approach.yaml")
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.plot_prediction_zoom_max_span_km = _game_plot_prediction_zoom_max_span_km(sim_cfg)
    projection = np.array(
        [
            [0.0, -2.0, 0.0, 0.0, 0.0, 0.0],
            [12.0, 20.0, -14.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    ri = dashboard._capped_projection_points_for_zoom(
        projection,
        x_axis=1,
        y_axis=0,
        camera_center=np.zeros(3, dtype=float),
    )
    rc = dashboard._capped_projection_points_for_zoom(
        projection,
        x_axis=2,
        y_axis=0,
        camera_center=np.zeros(3, dtype=float),
    )

    assert _game_plot_prediction_in_zoom(sim_cfg) is True
    assert _game_plot_prediction_zoom_max_span_km(sim_cfg) == pytest.approx(8.0)
    assert np.max(np.abs(ri)) == pytest.approx(8.0)
    assert np.max(np.abs(rc)) == pytest.approx(8.0)


def test_level7_plot_scale_keeps_goal_nmc_visible() -> None:
    root = Path(__file__).resolve().parents[1] / "game" / "configs"
    sim_cfg = SimulationConfig.from_yaml(root / "game_training_rpo_08_elliptic_nmc.yaml")
    cfg = RPOTrainingConfig.from_metadata(dict(sim_cfg.scenario.metadata or {}))
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.goal_nmt_radial_amplitude_km = cfg.goal_nmt_radial_amplitude_km
    dashboard.goal_nmt_cross_track_amplitude_km = cfg.goal_nmt_cross_track_amplitude_km
    dashboard.goal_nmt_cross_track_phase_deg = cfg.goal_nmt_cross_track_phase_deg
    dashboard.goal_nmt_center_ric_km = cfg.goal_nmt_center_ric_km
    dashboard.goal_nmt_element_tolerance_km = cfg.goal_nmt_element_tolerance_km
    dashboard.forbidden_regions = ()
    dashboard.approach_gates = ()
    dashboard.inspection_gates = ()
    dashboard.plot_overlays_in_zoom = _game_plot_overlays_in_zoom(sim_cfg)
    dashboard.plot_overlays_in_zoom_by_plane = _game_plot_overlays_in_zoom_by_plane(sim_cfg)

    ri_min_span = dashboard._minimum_plot_span_km(x_axis=1, y_axis=0, offset=np.zeros(3, dtype=float))
    rc_min_span = dashboard._minimum_plot_span_km(x_axis=2, y_axis=0, offset=np.zeros(3, dtype=float))

    assert _game_plot_overlays_in_zoom(sim_cfg) is True
    assert ri_min_span == pytest.approx(2.8 * PLOT_OVERLAY_MARGIN)
    assert rc_min_span == pytest.approx(1.4 * PLOT_OVERLAY_MARGIN)


def test_sun_angle_beam_does_not_drive_plot_zoom() -> None:
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.forbidden_regions = ()
    dashboard.approach_gates = ()
    dashboard.inspection_gates = ()
    dashboard.sun_angle_constraints = (
        game_training.SunAngleConstraintConfig(
            name="wide beam",
            sun_direction_ric=np.array([0.0, 1.0, 0.0], dtype=float),
            allowed_center_ric=np.array([0.0, -1.0, 0.0], dtype=float),
            allowed_half_angle_deg=45.0,
            beam_radius_km=100.0,
        ),
    )
    dashboard.plot_overlays_in_zoom = True
    dashboard.plot_overlays_in_zoom_by_plane = {}

    min_span = dashboard._minimum_plot_span_km(x_axis=1, y_axis=0, offset=np.zeros(3, dtype=float))

    assert min_span == pytest.approx(MIN_PLOT_SPAN_KM)


def test_nmc_overlay_boundaries_use_element_tolerance() -> None:
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.goal_nmt_radial_amplitude_km = 1.2
    dashboard.goal_nmt_cross_track_amplitude_km = 0.8
    dashboard.goal_nmt_cross_track_phase_deg = 90.0
    dashboard.goal_nmt_center_ric_km = np.zeros(3, dtype=float)
    dashboard.goal_nmt_element_tolerance_km = 0.2

    nominal = dashboard._nmt_points()
    lower, upper = dashboard._nmt_boundary_points()

    assert np.max(np.abs(nominal[:, 0])) == pytest.approx(1.2)
    assert np.max(np.abs(nominal[:, 1])) == pytest.approx(2.4)
    assert np.max(np.abs(nominal[:, 2])) == pytest.approx(0.8)
    assert np.max(np.abs(lower[:, 0])) == pytest.approx(1.0)
    assert np.max(np.abs(lower[:, 1])) == pytest.approx(2.0)
    assert np.max(np.abs(lower[:, 2])) == pytest.approx(0.6)
    assert np.max(np.abs(upper[:, 0])) == pytest.approx(1.4)
    assert np.max(np.abs(upper[:, 1])) == pytest.approx(2.8)
    assert np.max(np.abs(upper[:, 2])) == pytest.approx(1.0)


def test_nmc_nominal_curve_is_hidden_when_boundaries_exist() -> None:
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.goal_nmt_radial_amplitude_km = 1.2
    dashboard.goal_nmt_cross_track_amplitude_km = 0.8
    dashboard.goal_nmt_cross_track_phase_deg = 90.0
    dashboard.goal_nmt_center_ric_km = np.zeros(3, dtype=float)
    dashboard.goal_nmt_element_tolerance_km = 0.2

    assert dashboard._nmt_points().size > 0
    assert _should_draw_nominal_nmt(dashboard._nmt_points(), dashboard._nmt_boundary_points()) is False
    assert _should_draw_nominal_nmt(dashboard._nmt_points(), ()) is True


def test_level2_plot_scale_can_ignore_forbidden_region_zoom_extent() -> None:
    root = Path(__file__).resolve().parents[1] / "game" / "configs"
    sim_cfg = SimulationConfig.from_yaml(root / "game_training_rpo_02_vbar_approach.yaml")
    training_cfg = RPOTrainingConfig.from_metadata(dict(sim_cfg.scenario.metadata or {}))
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.forbidden_regions = training_cfg.forbidden_regions
    dashboard.approach_gates = ()
    dashboard.inspection_gates = ()
    dashboard.plot_overlays_in_zoom = _game_plot_overlays_in_zoom(sim_cfg)
    dashboard.plot_overlays_in_zoom_by_plane = _game_plot_overlays_in_zoom_by_plane(sim_cfg)
    dashboard.plot_axis_scale = _game_plot_axis_scale(sim_cfg)
    dashboard.plot_fixed_axis_half_span_km = _game_plot_fixed_axis_half_span_km(sim_cfg)
    dashboard.plot_equal_axis_scale_planes = _game_plot_equal_axis_scale_planes(sim_cfg)
    dashboard.proximity_ring_plot_planes = _game_proximity_ring_plot_planes(sim_cfg)

    ri_ignored_span = dashboard._minimum_plot_span_km(x_axis=1, y_axis=0, offset=np.zeros(3, dtype=float))
    rc_fixed_span = dashboard._minimum_plot_span_km(x_axis=2, y_axis=0, offset=np.zeros(3, dtype=float))
    dashboard.plot_overlays_in_zoom = True
    dashboard.plot_overlays_in_zoom_by_plane = {}
    ri_full_overlay_span = dashboard._minimum_plot_span_km(x_axis=1, y_axis=0, offset=np.zeros(3, dtype=float))

    assert _game_camera_mode(sim_cfg) == "target_pair"
    assert _game_plot_overlays_in_zoom(sim_cfg) is False
    assert _game_plot_overlays_in_zoom_by_plane(sim_cfg) == {"RC": True}
    assert dashboard._axis_scale_for_plane(x_axis=1, y_axis=0) == pytest.approx((1.2, 1.0))
    assert dashboard._axis_scale_for_plane(x_axis=2, y_axis=0) == pytest.approx((1.0, 1.0))
    assert _game_plot_fixed_axis_half_span_km(sim_cfg) == {"RI": (None, 0.75), "RC": (None, 0.75)}
    assert _game_target_centered_plot_axes(sim_cfg) == {"RI": ("y",)}
    assert _game_plot_equal_axis_scale_planes(sim_cfg) == ("RC",)
    assert dashboard._fixed_axis_half_span_for_plane(x_axis=1, y_axis=0) == (None, 0.75)
    assert dashboard._fixed_axis_half_span_for_plane(x_axis=2, y_axis=0) == (None, 0.75)
    assert dashboard._equal_axis_scale_for_plane(x_axis=2, y_axis=0) is True
    plot = type("Plot", (), {"width": 500, "height": 300})()
    near_radial_pts = [np.array([[-2.0, 0.0], [2.0, 0.0]], dtype=float)]
    wide_radial_pts = [np.array([[-2.0, -0.7], [2.0, 0.7]], dtype=float)]
    near_intrack_pts = [np.array([[-0.5, 0.0], [0.5, 0.0]], dtype=float)]
    ri_near_scale = dashboard._axis_scales_for_plot(
        plot,
        pts=near_radial_pts,
        min_span_km=0.05,
        x_axis=1,
        y_axis=0,
    )
    ri_close_intrack_scale = dashboard._axis_scales_for_plot(
        plot,
        pts=near_intrack_pts,
        min_span_km=0.05,
        x_axis=1,
        y_axis=0,
    )
    ri_wide_scale = dashboard._axis_scales_for_plot(
        plot,
        pts=wide_radial_pts,
        min_span_km=0.05,
        x_axis=1,
        y_axis=0,
    )
    rc_scale = dashboard._axis_scales_for_plot(
        plot,
        pts=near_radial_pts,
        min_span_km=rc_fixed_span,
        x_axis=2,
        y_axis=0,
    )
    assert ri_near_scale[0] == pytest.approx(ri_wide_scale[0])
    assert ri_near_scale[1] == pytest.approx(ri_wide_scale[1])
    assert ri_close_intrack_scale[0] > ri_near_scale[0]
    assert ri_close_intrack_scale[1] == pytest.approx(ri_near_scale[1])
    assert ri_near_scale[1] == pytest.approx(300 * 0.5 / 0.75)
    assert rc_scale == pytest.approx((ri_near_scale[1], ri_near_scale[1]))
    assert _game_proximity_ring_plot_planes(sim_cfg) == ("RI",)
    assert dashboard._show_proximity_rings_for_plane(x_axis=1, y_axis=0) is True
    assert dashboard._show_proximity_rings_for_plane(x_axis=2, y_axis=0) is False
    assert ri_ignored_span == pytest.approx(0.005)
    assert rc_fixed_span > 5.0
    assert ri_full_overlay_span > 5.0
