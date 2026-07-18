# ruff: noqa: F401,F403,F405,I001
from .runner_common import *
from .runner_models import *
from .runner_config import *

def _start_game_recorder(
    *,
    enabled: bool,
    config: SimulationConfig,
    difficulty: str,
    attempt_index: int,
    output_dir: str | Path | None,
    fps: float,
) -> Any:
    return game_recording.start_game_recorder(
        enabled=enabled,
        config=config,
        difficulty=difficulty,
        attempt_index=attempt_index,
        output_dir=output_dir,
        fps=fps,
    )


def _start_game_clip_recorder(
    *,
    enabled: bool,
    config: SimulationConfig,
    difficulty: str,
    clip_index: int,
    output_dir: str | Path | None,
    fps: float,
) -> Any:
    return game_recording.start_game_clip_recorder(
        enabled=enabled,
        config=config,
        difficulty=difficulty,
        clip_index=clip_index,
        output_dir=output_dir,
        fps=fps,
    )


def _capture_recording_frame(recorder: Any, dashboard: Any) -> None:
    game_recording.capture_recording_frame(recorder, dashboard)


def _safe_capture_recording_frame(recorder: Any, dashboard: Any) -> Any:
    return game_recording.safe_capture_recording_frame(recorder, dashboard)


def _finish_game_recording(
    recorder: Any,
    training_cfg: RPOTrainingConfig,
    *,
    override_level_path: Path | None = None,
) -> Path | None:
    return game_recording.finish_game_recording(recorder, training_cfg, override_level_path=override_level_path)


def _discard_recorder_safely(recorder: Any) -> None:
    game_recording.discard_recorder_safely(recorder)


def _add_level_music_to_recording(
    recording_path: Path,
    training_cfg: RPOTrainingConfig,
    *,
    override_level_path: Path | None = None,
) -> Path:
    return game_recording.add_level_music_to_recording(
        recording_path,
        training_cfg,
        override_level_path=override_level_path,
    )


def _sync_dashboard_training_config(dashboard: Any, training_cfg: RPOTrainingConfig) -> None:
    dashboard.keepout_radius_km = training_cfg.keepout_radius_km
    dashboard.goal_range_km = training_cfg.goal_range_km
    dashboard.goal_range_tolerance_km = training_cfg.goal_range_tolerance_km
    dashboard.goal_radius_km = training_cfg.goal_radius_km
    dashboard.hard_speed_limit_radius_km = training_cfg.hard_speed_limit_radius_km
    dashboard.hard_speed_limit_km_s = training_cfg.hard_speed_limit_km_s
    dashboard.max_target_reference_range_km = training_cfg.max_target_reference_range_km
    dashboard.target_reference_object_id = training_cfg.target_reference_object_id
    dashboard.goal_relative_ric_km = training_cfg.goal_relative_ric_km
    dashboard.goal_nmt_radial_amplitude_km = training_cfg.goal_nmt_radial_amplitude_km
    dashboard.goal_nmt_cross_track_amplitude_km = training_cfg.goal_nmt_cross_track_amplitude_km
    dashboard.goal_nmt_cross_track_phase_deg = training_cfg.goal_nmt_cross_track_phase_deg
    dashboard.goal_nmt_center_ric_km = training_cfg.goal_nmt_center_ric_km
    dashboard.goal_nmt_element_tolerance_km = training_cfg.goal_nmt_element_tolerance_km
    dashboard.sun_angle_constraints = training_cfg.sun_angle_constraints
    dashboard.mission_time_budget_s = training_cfg.max_time_s
    if hasattr(dashboard, "_frame_cache_dirty"):
        dashboard._frame_cache_dirty = True


def _sync_dashboard_round_config(dashboard: Any, config: SimulationConfig) -> None:
    dashboard.coast_prediction_model = _game_coast_prediction_model(config)
    dashboard.cr3bp_projection_mode = _game_cr3bp_projection_mode(config)
    dashboard.relative_frame = _game_relative_frame(config)
    dashboard.visual_extrapolation_enabled = _game_visual_extrapolation_enabled(config)
    dashboard.camera_rule_mode = _game_camera_rule_mode(config)
    dashboard.camera_rule_toggle_enabled = _game_camera_rule_toggle_enabled(config)
    dashboard.plot_prediction_full_trajectory_only = _game_plot_prediction_full_trajectory_only(config)
    cr3bp_horizon_s = _game_cr3bp_coast_prediction_horizon_s(config)
    if cr3bp_horizon_s is not None:
        dashboard.cr3bp_coast_prediction_horizon_s = cr3bp_horizon_s
    dashboard.cr3bp_active_prediction_horizon_s = _game_cr3bp_active_prediction_horizon_s(config)
    dashboard.cr3bp_coast_prediction_horizon_mode = _game_cr3bp_coast_prediction_horizon_mode(config)
    cr3bp_dt_s = _game_cr3bp_coast_prediction_dt_s(config)
    if cr3bp_dt_s is not None:
        dashboard.cr3bp_coast_prediction_dt_s = cr3bp_dt_s
    cr3bp_coast_update_interval_s = _game_cr3bp_prediction_coast_update_interval_s(config)
    if cr3bp_coast_update_interval_s is not None:
        dashboard.cr3bp_prediction_coast_update_interval_s = cr3bp_coast_update_interval_s
    target_horizon_s = _game_target_coast_prediction_horizon_s(config)
    dashboard.target_coast_prediction_horizon_s = target_horizon_s
    target_dt_s = _game_target_coast_prediction_dt_s(config)
    dashboard.target_coast_prediction_dt_s = target_dt_s
    if hasattr(dashboard, "_prediction_cache"):
        dashboard._prediction_cache = {}
    if hasattr(dashboard, "_frame_cache_dirty"):
        dashboard._frame_cache_dirty = True


def _arcade_round_music_path(config: SimulationConfig, round_index: int) -> Path | None:
    track = _arcade_round_music_track(config, round_index)
    if track is None:
        return None
    path = Path(track)
    return path if path.is_absolute() else GAME_MUSIC_DIR / path

__all__ = [name for name in globals() if not name.startswith("__")]
