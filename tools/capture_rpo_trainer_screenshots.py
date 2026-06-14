from __future__ import annotations

import os
import shutil
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame  # noqa: E402

from sim.api import SimulationConfig  # noqa: E402
from sim.game.launcher import _draw_launcher, discover_game_scenarios  # noqa: E402
from sim.game.manual import KeyboardCommandState  # noqa: E402
from sim.game.pygame_dashboard import PygameRPODashboard  # noqa: E402
from sim.game.runner import (  # noqa: E402
    _coast_prediction_orbit_fraction,
    _command_status,
    _dashboard_object_ids,
    _force_game_acceleration_off_config,
    _game_camera_mode,
    _game_camera_rule_mode,
    _game_coast_prediction_model,
    _game_control_mode,
    _game_difficulty,
    _game_level_title,
    _game_plot_axis_scale,
    _game_plot_equal_axis_scale_planes,
    _game_plot_fixed_axis_half_span_km,
    _game_plot_overlays_in_zoom,
    _game_plot_overlays_in_zoom_by_plane,
    _game_plot_prediction_in_zoom,
    _game_plot_prediction_zoom_max_span_km,
    _game_proximity_ring_plot_planes,
    _game_ric_reference_object_id,
    _game_show_target_hcw_path,
    _game_target_centered_plot_axes,
    _game_target_centered_plot_planes,
    _mission_metrics,
    _start_game_attempt,
    _step_game_attempt,
    _sync_dashboard_round_config,
    _sync_dashboard_training_config,
)
from sim.game.training import RPOTrainingConfig, RPOTrainingTracker  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "docs" / "assets" / "rpo-trainer"


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        ROOT / "sim" / "game" / "assets" / "OEL_RPO_Trainer.png",
        OUTDIR / "rpo-trainer-landing.png",
    )
    _capture_level_selector(OUTDIR / "rpo-trainer-level-selector.png")
    _capture_level(
        ROOT / "sim" / "game" / "configs" / "game_training_rpo_03_rbar_approach.yaml",
        OUTDIR / "rpo-level-03-rbar-approach.png",
        schedule=((90, 1.0, 0.0, 0.0), (170, 0.0, 0.0, 0.0), (55, -1.0, 0.0, 0.0), (80, 0.0, 0.0, 0.0)),
        hint="R-bar approach: pulse toward the target, coast, then brake in stages.",
    )
    _capture_level(
        ROOT / "sim" / "game" / "configs" / "game_training_rpo_05_passive_cross_track_approach.yaml",
        OUTDIR / "rpo-level-05-safe-inspection.png",
        schedule=((120, 0.0, 0.0, 1.0), (220, 0.0, 0.0, 0.0), (45, 0.0, -1.0, 0.0), (120, 0.0, 0.0, 0.0)),
        hint="Safe inspection: build cross-track separation before drifting through the gates.",
    )


def _capture_level_selector(out_path: Path) -> None:
    options = tuple(
        replace(option, completed_difficulties=(), high_score=0)
        for option in discover_game_scenarios(ROOT / "sim" / "game" / "configs")
    )
    selected = min(3, max(len(options) - 1, 0))
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((1040, 680), pygame.RESIZABLE)
    font = pygame.font.SysFont("Menlo", 18) or pygame.font.Font(None, 18)
    small_font = pygame.font.SysFont("Menlo", 14) or pygame.font.Font(None, 14)
    title_font = pygame.font.SysFont("Menlo", 30) or pygame.font.Font(None, 30)
    try:
        _draw_launcher(
            pygame,
            screen,
            options=options,
            selected=selected,
            scroll_offset=0,
            selected_difficulty=options[selected].difficulty if options else "easy",
            music_enabled=True,
            preview_scroll_px=0,
            record_video=False,
            font=font,
            small_font=small_font,
            title_font=title_font,
        )
        pygame.display.flip()
        pygame.image.save(screen, str(out_path))
    finally:
        pygame.display.quit()
        pygame.quit()


def _capture_level(
    config_path: Path,
    out_path: Path,
    *,
    schedule: tuple[tuple[int, float, float, float], ...],
    hint: str,
) -> None:
    config = _force_game_acceleration_off_config(SimulationConfig.from_yaml(config_path))
    training_cfg = RPOTrainingConfig.from_metadata(dict(config.scenario.metadata or {}))
    control_mode = _game_control_mode(config)
    difficulty = _game_difficulty(config)
    ric_reference_object_id = _game_ric_reference_object_id(config, training_cfg.target_object_id)
    command_state = KeyboardCommandState()
    command_state.paused = False
    session, _, snapshot = _start_game_attempt(
        config,
        command_state=command_state,
        training_cfg=training_cfg,
        controlled_object_id=training_cfg.chaser_object_id,
        attitude_rate_deg_s=45.0,
        control_mode=control_mode,
        ric_reference_object_id=ric_reference_object_id,
    )
    anim_cfg = dict(config.scenario.outputs.animations or {})
    dashboard_target_id, dashboard_chaser_id = _dashboard_object_ids(training_cfg, anim_cfg)
    dashboard = PygameRPODashboard(
        target_object_id=dashboard_target_id,
        chaser_object_id=dashboard_chaser_id,
        reference_object_id=ric_reference_object_id,
        keepout_radius_km=training_cfg.keepout_radius_km,
        goal_range_km=training_cfg.goal_range_km,
        goal_range_tolerance_km=training_cfg.goal_range_tolerance_km,
        goal_radius_km=training_cfg.goal_radius_km,
        hard_speed_limit_radius_km=training_cfg.hard_speed_limit_radius_km,
        hard_speed_limit_km_s=training_cfg.hard_speed_limit_km_s,
        goal_relative_ric_km=training_cfg.goal_relative_ric_km,
        goal_nmt_radial_amplitude_km=training_cfg.goal_nmt_radial_amplitude_km,
        goal_nmt_cross_track_amplitude_km=training_cfg.goal_nmt_cross_track_amplitude_km,
        goal_nmt_cross_track_phase_deg=training_cfg.goal_nmt_cross_track_phase_deg,
        goal_nmt_center_ric_km=training_cfg.goal_nmt_center_ric_km,
        goal_nmt_element_tolerance_km=training_cfg.goal_nmt_element_tolerance_km,
        coast_prediction_orbit_fraction=_coast_prediction_orbit_fraction(difficulty),
        coast_prediction_model=_game_coast_prediction_model(config),
        forbidden_regions=training_cfg.forbidden_regions,
        approach_gates=training_cfg.approach_gates,
        inspection_gates=training_cfg.inspection_gates,
        plot_overlays_in_zoom=_game_plot_overlays_in_zoom(config),
        plot_overlays_in_zoom_by_plane=_game_plot_overlays_in_zoom_by_plane(config),
        plot_prediction_in_zoom=_game_plot_prediction_in_zoom(config),
        plot_prediction_zoom_max_span_km=_game_plot_prediction_zoom_max_span_km(config),
        plot_axis_scale=_game_plot_axis_scale(config),
        plot_fixed_axis_half_span_km=_game_plot_fixed_axis_half_span_km(config),
        plot_equal_axis_scale_planes=_game_plot_equal_axis_scale_planes(config),
        target_centered_plot_planes=_game_target_centered_plot_planes(config),
        target_centered_plot_axes=_game_target_centered_plot_axes(config),
        proximity_ring_plot_planes=_game_proximity_ring_plot_planes(config),
        camera_mode=_game_camera_mode(config),
        camera_rule_mode=_game_camera_rule_mode(config),
        show_target_coast_prediction=_game_show_target_hcw_path(config),
        fullscreen=False,
    )
    try:
        _sync_dashboard_training_config(dashboard, training_cfg)
        _sync_dashboard_round_config(dashboard, config)
        trainer = RPOTrainingTracker(training_cfg)
        dashboard.push_snapshot(snapshot)
        trainer.record(snapshot)
        score = trainer.score()
        for steps, pitch, yaw, roll in schedule:
            command_state.pitch = float(pitch)
            command_state.yaw = float(yaw)
            command_state.roll = float(roll)
            score = _step_game_attempt(
                session=session,
                dashboard=dashboard,
                trainer=trainer,
                steps_to_run=int(steps),
                initial_score=score,
            )
        command_state.reset_axes()
        dashboard.draw(
            command_status=_command_status(command_state, control_mode=control_mode),
            coach_hint=hint,
            mission_state="active",
            level_title=_game_level_title(config),
            mission_metrics=_mission_metrics(training_cfg, score),
            speed_multiple=10.0,
        )
        pygame.image.save(dashboard.screen, str(out_path))
    finally:
        dashboard.close()


if __name__ == "__main__":
    main()
