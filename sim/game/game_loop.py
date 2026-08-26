# ruff: noqa: F401,F403,F405,I001
from .runner_common import *
from .runner_models import *
from .runner_config import *
from .tutorial_runtime import *
from .recording_runtime import *
from .attempt_lifecycle import *
from .mission_metrics import *


def run_game_mode(
    config_path: str | Path,
    *,
    controlled_object_id: str | None = None,
    attitude_rate_deg_s: float = 45.0,
    realtime: bool = True,
    speed_multiple: float | None = None,
    difficulty_override: str | None = None,
    music_enabled: bool = True,
    record_video: bool = False,
    game_mode: str = "pilot",
    frame_convention: FrameConvention | dict[str, Any] | None = None,
    operator_burn_plan: OperatorBurnPlan | None = None,
    skip_initial_briefing: bool = False,
    recording_output_dir: str | Path | None = None,
    recording_fps: float = GAME_RECORDING_FPS,
    arcade_seed: int | None = None,
    debrief_output_dir: str | Path | None = None,
    presentation_mode: str | None = None,
    presentation_fps_cap: float | None = None,
    presentation_refresh_hz: float | None = None,
    presentation_vsync: str | None = None,
    presentation_diagnostics: bool | None = None,
    presentation_diagnostics_output: str | Path | None = None,
) -> GameRunResult:
    from sim.game.pygame_dashboard import PygameRPODashboard

    config = _force_game_acceleration_off_config(SimulationConfig.from_yaml(config_path))
    configured_controlled_object_id = _game_controlled_object_id(config, default="chaser")
    controlled_object_id = str(controlled_object_id or configured_controlled_object_id)
    config = _select_game_controlled_object(
        config,
        controlled_object_id=controlled_object_id,
        configured_object_id=configured_controlled_object_id,
    )
    control_mode = _game_control_mode(config)
    difficulty = str(difficulty_override or _game_difficulty(config)).strip().lower()
    game_mode = _normalize_game_mode(game_mode)
    operator_playback_mode = game_mode == "operator"
    frame_convention = normalize_frame_convention(frame_convention)
    presentation_settings = _game_presentation_settings(
        config,
        mode=presentation_mode,
        fps_cap=presentation_fps_cap,
        refresh_rate_hz=presentation_refresh_hz,
        vsync=presentation_vsync,
        diagnostics=presentation_diagnostics,
        diagnostics_output=presentation_diagnostics_output,
    )
    initial_operator_burn_plan = operator_burn_plan
    operator_burn_plan = (operator_burn_plan or OperatorBurnPlan()) if operator_playback_mode else None
    skip_initial_briefing = bool(skip_initial_briefing and operator_playback_mode)
    operator_actuator_error_fraction = _operator_actuator_error_fraction(difficulty) if operator_playback_mode else 0.0
    training_cfg = _training_config_with_sun_environment(
        RPOTrainingConfig.from_metadata(dict(config.scenario.metadata or {})),
        config,
    )
    training_cfg = training_config_for_game_mode(training_cfg, game_mode=game_mode)
    base_training_cfg = training_cfg
    arcade_enabled = _game_arcade_enabled(config)
    arcade_seed_value = _new_arcade_seed() if arcade_enabled and arcade_seed is None else arcade_seed
    arcade_round_index = 1
    arcade_total_score = 0
    arcade_remaining_time_s = _game_arcade_initial_time_s(config, training_cfg) if arcade_enabled else None
    if arcade_enabled:
        training_cfg = _arcade_round_training_config(
            config,
            training_cfg,
            round_index=arcade_round_index,
            max_time_s=arcade_remaining_time_s,
        )
        training_cfg = _training_config_with_sun_environment(training_cfg, config)
        training_cfg = training_config_for_game_mode(training_cfg, game_mode=game_mode)
    operator_tutorial_enabled = _operator_tutorial_enabled(game_mode, training_cfg, arcade_enabled=arcade_enabled)
    operator_tutorial = OperatorTutorialRuntime() if operator_tutorial_enabled else None
    initial_operator_plan_needed = bool(
        game_mode == "operator" and not operator_tutorial_enabled and initial_operator_burn_plan is None
    )
    if operator_tutorial_enabled:
        operator_burn_plan = OperatorBurnPlan()
    debrief_enabled = _game_debrief_enabled(
        config,
        training_cfg,
        arcade_enabled=arcade_enabled,
    )
    attempt_config = _arcade_round_simulation_config(
        config,
        training_cfg,
        round_index=arcade_round_index,
        rng=(
            _arcade_round_initial_state_rng(int(arcade_seed_value), arcade_round_index)
            if arcade_enabled and arcade_seed_value is not None
            else None
        ),
    )
    ric_reference_object_id = _game_ric_reference_object_id(config, training_cfg.target_object_id)
    level_title = _game_level_title(config)
    if operator_tutorial_enabled:
        level_title = "Level 0 - Operator Tutorial"
    speed_multiplier_options = _game_speed_multiplier_options(config)
    current_speed_multiple = _game_initial_speed_multiple(config, speed_multiple)
    if operator_tutorial_enabled:
        current_speed_multiple = _coerce_speed_multiple(
            OPERATOR_TUTORIAL_PLAYBACK_SPEED_MULTIPLE,
            options=speed_multiplier_options,
        )
    effective_speed_multiple = current_speed_multiple
    maneuver_control_speed_multiple = _game_maneuver_control_speed_multiple(config)
    two_rail_speed_control = _game_two_rail_speed_control_enabled(config)
    burn_trace_enabled = _game_burn_trace_enabled()
    trainer = RPOTrainingTracker(training_cfg)
    command_state = KeyboardCommandState()
    command_state.use_timing_accumulator = _game_timed_input_accumulator_enabled(config)
    player_max_accel_km_s2 = _max_accel_from_config(attempt_config, controlled_object_id)
    guided_tutorial = GuidedTutorialRuntime()
    ric_primer = RICPrimerRuntime()
    ric_primer_enabled = _ric_primer_enabled(training_cfg, arcade_enabled=arcade_enabled) or operator_tutorial_enabled
    operator_burn_cinematic = OperatorBurnCinematicRuntime()
    command_state.paused = bool(training_cfg.enabled and not skip_initial_briefing)
    phase = GamePhase.BRIEFING if training_cfg.enabled and not skip_initial_briefing else GamePhase.PLAYING
    if operator_tutorial_enabled:
        command_state.paused = True
        phase = GamePhase.PRIMER
    briefing_lines = _training_briefing_lines(
        config,
        training_cfg,
        difficulty=difficulty,
        game_mode=game_mode,
        operator_burn_plan=operator_burn_plan,
    )
    session, command_provider, snapshot = _start_game_attempt(
        attempt_config,
        command_state=command_state,
        training_cfg=training_cfg,
        controlled_object_id=controlled_object_id,
        attitude_rate_deg_s=attitude_rate_deg_s,
        control_mode=control_mode,
        ric_reference_object_id=ric_reference_object_id,
        operator_burn_plan=operator_burn_plan,
        operator_actuator_error_fraction=operator_actuator_error_fraction,
        defensive_target_profile=(
            _game_random_direction_defensive_target_profile(
                config,
                round_index=arcade_round_index,
                rng=_arcade_round_rng(int(arcade_seed_value), arcade_round_index),
            )
            if arcade_enabled and arcade_seed_value is not None
            else None
        ),
    )

    anim_cfg = dict(config.scenario.outputs.animations or {})
    dashboard_target_id, dashboard_chaser_id = _dashboard_object_ids(training_cfg, anim_cfg)
    dashboard = PygameRPODashboard(
        target_object_id=dashboard_target_id,
        chaser_object_id=dashboard_chaser_id,
        controlled_object_id=controlled_object_id,
        reference_object_id=ric_reference_object_id,
        relative_frame=_game_relative_frame(config),
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
        coast_prediction_orbit_fraction=_operator_coast_prediction_orbit_fraction(game_mode, difficulty),
        coast_prediction_model=_game_coast_prediction_model(attempt_config),
        show_coast_prediction=_game_show_coast_prediction(config),
        cr3bp_projection_mode=_game_cr3bp_projection_mode(config),
        cr3bp_coast_prediction_horizon_s=_game_cr3bp_coast_prediction_horizon_s(config) or 21600.0,
        cr3bp_active_prediction_horizon_s=_game_cr3bp_active_prediction_horizon_s(config),
        cr3bp_coast_prediction_horizon_mode=_game_cr3bp_coast_prediction_horizon_mode(config),
        cr3bp_coast_prediction_dt_s=_game_cr3bp_coast_prediction_dt_s(config) or 300.0,
        target_coast_prediction_horizon_s=_game_target_coast_prediction_horizon_s(config),
        target_coast_prediction_dt_s=_game_target_coast_prediction_dt_s(config),
        forbidden_regions=training_cfg.forbidden_regions,
        approach_gates=training_cfg.approach_gates,
        inspection_gates=training_cfg.inspection_gates,
        sun_angle_constraints=training_cfg.sun_angle_constraints,
        plot_overlays_in_zoom=_game_plot_overlays_in_zoom(config),
        plot_overlays_in_zoom_by_plane=_game_plot_overlays_in_zoom_by_plane(config),
        plot_prediction_in_zoom=_game_plot_prediction_in_zoom(config),
        plot_prediction_zoom_max_span_km=_game_plot_prediction_zoom_max_span_km(config),
        plot_prediction_full_trajectory_only=_game_plot_prediction_full_trajectory_only(config),
        plot_axis_scale=_game_plot_axis_scale(config),
        plot_fixed_axis_half_span_km=_game_plot_fixed_axis_half_span_km(config),
        plot_equal_axis_scale_planes=_game_plot_equal_axis_scale_planes(config),
        target_centered_plot_planes=_game_target_centered_plot_planes(config),
        target_centered_plot_axes=_game_target_centered_plot_axes(config),
        proximity_ring_plot_planes=_game_proximity_ring_plot_planes(config),
        target_reference_object_id=training_cfg.target_reference_object_id,
        camera_mode=_game_camera_mode(config),
        camera_rule_mode=_game_camera_rule_mode(config),
        camera_rule_toggle_enabled=_game_camera_rule_toggle_enabled(config),
        target_sprite_path=_game_target_sprite_path(config),
        chaser_sprite_path=_game_chaser_sprite_path(config),
        chaser_sprite_ri_path=_game_chaser_plane_sprite_path(config, "ri"),
        chaser_sprite_rc_path=_game_chaser_plane_sprite_path(config, "rc"),
        target_sprite_diameter_km=_game_target_sprite_diameter_km(config),
        chaser_sprite_diameter_km=_game_chaser_sprite_diameter_km(config),
        chaser_sprite_ri_size_scale=_game_chaser_sprite_ri_size_scale(config),
        show_target_coast_prediction=_game_show_target_hcw_path(config),
        frame_convention=frame_convention,
        fullscreen=True,
        presentation_mode=presentation_settings.mode,
        presentation_vsync=presentation_settings.vsync,
    )
    presentation_controller = create_presentation_controller(
        dashboard.pygame,
        dashboard,
        presentation_settings,
    )
    aero_cfg = _game_aerodynamic_control_config(config)
    dashboard.aerodynamic_ri_pitch_max_deg = aero_cfg["ri_pitch_max_deg"]
    _sync_dashboard_aerodynamic_control(dashboard, command_provider)
    if operator_playback_mode:
        _clear_live_prediction_burn(dashboard)

    def hold_operator_burn_cinematic_for_animation(duration_s: float) -> None:
        _operator_burn_cinematic_hold_for_animation(
            operator_burn_cinematic,
            now_wall_s=perf_counter(),
            duration_s=duration_s,
        )

    _sync_dashboard_training_config(dashboard, training_cfg)
    _sync_dashboard_round_config(dashboard, attempt_config)
    recording_attempt = 1
    recording_path: Path | None = None
    debrief_path: Path | None = None
    debrief_folder_to_open: Path | None = None
    recording_controller = GameRecordingController(
        enabled=record_video,
        config=config,
        difficulty=difficulty,
        attempt_index=recording_attempt,
        output_dir=recording_output_dir,
        fps=recording_fps,
    )
    clip_recording_controller = GameClipRecordingController(
        config=config,
        difficulty=difficulty,
        output_dir=recording_output_dir,
        fps=recording_fps,
    )
    clip_recording_started_wall: float | None = None
    clip_recording_status_message = ""
    clip_recording_status_until = 0.0
    audio_controller: GameAudioController | None = None
    operator_tutorial_level_passed = False

    def restart_attempt_for_operator_plan(
        plan: OperatorBurnPlan,
        *,
        tutorial_stage: OperatorTutorialStage | None = None,
    ) -> None:
        nonlocal session, command_provider, snapshot, trainer, guided_tutorial, ric_primer, ric_primer_enabled
        nonlocal operator_burn_plan, operator_burn_cinematic
        operator_burn_plan = plan
        session, command_provider, snapshot = _start_game_attempt(
            attempt_config,
            command_state=command_state,
            training_cfg=training_cfg,
            controlled_object_id=controlled_object_id,
            attitude_rate_deg_s=attitude_rate_deg_s,
            control_mode=control_mode,
            ric_reference_object_id=ric_reference_object_id,
            operator_burn_plan=operator_burn_plan,
            operator_actuator_error_fraction=operator_actuator_error_fraction,
            defensive_target_profile=(
                _game_random_direction_defensive_target_profile(
                    config,
                    round_index=arcade_round_index,
                    rng=_arcade_round_rng(int(arcade_seed_value), arcade_round_index),
                )
                if arcade_enabled and arcade_seed_value is not None
                else None
            ),
        )
        trainer = RPOTrainingTracker(training_cfg)
        guided_tutorial = GuidedTutorialRuntime()
        ric_primer = RICPrimerRuntime()
        ric_primer_enabled = _ric_primer_enabled(training_cfg, arcade_enabled=arcade_enabled) or bool(
            operator_tutorial_enabled
        )
        operator_burn_cinematic.reset()
        dashboard.clear()
        if operator_playback_mode:
            _clear_live_prediction_burn(dashboard)
        _sync_dashboard_training_config(dashboard, training_cfg)
        _sync_dashboard_round_config(dashboard, attempt_config)
        dashboard.push_snapshot(snapshot)
        trainer.record(snapshot, control_telemetry_provider=command_provider)
        if tutorial_stage is None:
            _sync_guided_tutorial_path_for_mode(
                dashboard,
                trainer,
                training_cfg,
                guided_tutorial,
                game_mode=game_mode,
            )
        else:
            _clear_dashboard_tutorial_path(dashboard)

    initial_snapshot_recorded = False
    try:
        if _game_sandbox_enabled(config):
            dashboard.push_snapshot(snapshot)
            setup = _run_sandbox_setup_form(
                dashboard,
                config=config,
                speed_multiple=current_speed_multiple,
                level_title=level_title,
            )
            if setup is None:
                training_cfg = RPOTrainingConfig(enabled=False)
                return GameRunResult(
                    config_path=Path(config_path),
                    difficulty=difficulty,
                    level_passed=False,
                    mode=game_mode,
                    frame_convention=frame_convention,
                    arcade_score=0,
                    arcade_seed=arcade_seed_value if arcade_enabled else None,
                )
            config = _apply_sandbox_setup_to_config(config, setup)
            training_cfg = _training_config_with_sun_environment(
                RPOTrainingConfig.from_metadata(dict(config.scenario.metadata or {})),
                config,
            )
            training_cfg = training_config_for_game_mode(training_cfg, game_mode=game_mode)
            base_training_cfg = training_cfg
            debrief_enabled = _game_debrief_enabled(
                config,
                training_cfg,
                arcade_enabled=arcade_enabled,
            )
            attempt_config = _arcade_round_simulation_config(
                config,
                training_cfg,
                round_index=arcade_round_index,
                rng=(
                    _arcade_round_initial_state_rng(int(arcade_seed_value), arcade_round_index)
                    if arcade_enabled and arcade_seed_value is not None
                    else None
                ),
            )
            player_max_accel_km_s2 = _max_accel_from_config(attempt_config, controlled_object_id)
            ric_reference_object_id = _game_ric_reference_object_id(config, training_cfg.target_object_id)
            speed_multiplier_options = _game_speed_multiplier_options(config)
            effective_speed_multiple = current_speed_multiple
            level_title = _game_level_title(config)
            two_rail_speed_control = _game_two_rail_speed_control_enabled(config)
            trainer = RPOTrainingTracker(training_cfg)
            guided_tutorial = GuidedTutorialRuntime()
            ric_primer = RICPrimerRuntime()
            ric_primer_enabled = _ric_primer_enabled(training_cfg, arcade_enabled=arcade_enabled)
            operator_burn_cinematic.reset()
            command_state.paused = bool(training_cfg.enabled and not skip_initial_briefing)
            phase = GamePhase.BRIEFING if training_cfg.enabled and not skip_initial_briefing else GamePhase.PLAYING
            briefing_lines = _training_briefing_lines(
                config,
                training_cfg,
                difficulty=difficulty,
                game_mode=game_mode,
                operator_burn_plan=operator_burn_plan,
            )
            session, command_provider, snapshot = _start_game_attempt(
                attempt_config,
                command_state=command_state,
                training_cfg=training_cfg,
                controlled_object_id=controlled_object_id,
                attitude_rate_deg_s=attitude_rate_deg_s,
                control_mode=control_mode,
                ric_reference_object_id=ric_reference_object_id,
                operator_burn_plan=operator_burn_plan,
                operator_actuator_error_fraction=operator_actuator_error_fraction,
                defensive_target_profile=(
                    _game_random_direction_defensive_target_profile(
                        config,
                        round_index=arcade_round_index,
                        rng=_arcade_round_rng(int(arcade_seed_value), arcade_round_index),
                    )
                    if arcade_enabled and arcade_seed_value is not None
                    else None
                ),
            )
            initial_snapshot_recorded = False
            dashboard.clear()
            if operator_playback_mode:
                _clear_live_prediction_burn(dashboard)
            dashboard.reference_object_id = ric_reference_object_id
            _sync_dashboard_training_config(dashboard, training_cfg)
            _sync_dashboard_round_config(dashboard, attempt_config)
            dashboard.camera_rule_mode = _game_camera_rule_mode(config)
            if recording_controller.recorder is None:
                recording_controller = GameRecordingController(
                    enabled=record_video,
                    config=config,
                    difficulty=difficulty,
                    attempt_index=recording_attempt,
                    output_dir=recording_output_dir,
                    fps=recording_fps,
                )
            else:
                recording_controller.config = config
            clip_recording_controller = GameClipRecordingController(
                config=config,
                difficulty=difficulty,
                output_dir=recording_output_dir,
                fps=recording_fps,
            )
            clip_recording_started_wall = None
            clip_recording_status_message = ""
            clip_recording_status_until = 0.0
        if initial_operator_plan_needed:
            selected_plan = plan_operator_burns_for_config(
                dashboard.pygame,
                dashboard.screen,
                dashboard.clock,
                config_path,
                font=dashboard.font,
                small_font=dashboard.small_font,
                title_font=dashboard.large_font,
                initial_plan=None,
                difficulty=difficulty,
                frame_convention=frame_convention,
                config_override=config,
            )
            if selected_plan is None:
                return GameRunResult(
                    config_path=Path(config_path),
                    difficulty=difficulty,
                    level_passed=False,
                    mode=game_mode,
                    frame_convention=frame_convention,
                    arcade_score=0,
                    arcade_seed=arcade_seed_value if arcade_enabled else None,
                )
            if record_video:
                recording_controller.start()
                recording_controller.capture_hold(dashboard, duration_s=OPERATOR_SCRIPT_RECORDING_HOLD_S)
            restart_attempt_for_operator_plan(selected_plan)
            command_state.reset_axes()
            command_state.paused = False
            phase = GamePhase.PLAYING
            briefing_lines = ()
            initial_snapshot_recorded = True
        if recording_controller.recorder is None:
            recording_controller.start()
        if not initial_snapshot_recorded:
            dashboard.push_snapshot(snapshot)
            trainer.record(snapshot, control_telemetry_provider=command_provider)
            _sync_guided_tutorial_path_for_mode(
                dashboard,
                trainer,
                training_cfg,
                guided_tutorial,
                game_mode=game_mode,
            )
        score = trainer.score()
        if phase != GamePhase.PRIMER:
            phase = phase_from_score(score, briefing_open=phase_shows_briefing(phase), paused=command_state.paused)
        _sync_live_prediction_burn_for_mode(
            dashboard,
            command_state,
            game_mode=game_mode,
            control_mode=control_mode,
            max_accel_km_s2=player_max_accel_km_s2,
            elapsed_wall_s=0.0,
            speed_multiple=current_speed_multiple,
            dt_s=float(attempt_config.scenario.simulator.dt_s),
        )
        _sync_dashboard_aerodynamic_control(dashboard, command_provider)
        dashboard.draw(
            command_status=_game_command_status(
                command_state,
                control_mode=control_mode,
                game_mode=game_mode,
                command_provider=command_provider,
                frame_convention=frame_convention,
            ),
            coach_hint=_coach_hint_with_camera_rule(
                _guided_tutorial_stage_hint(
                    _guided_tutorial_current_stage(training_cfg, guided_tutorial),
                    guided_tutorial,
                    frame_convention=frame_convention,
                )
                or trainer.current_hint(),
                dashboard,
                training_cfg,
            ),
            mission_state=mission_state_for_dashboard(phase),
            level_title=level_title,
            mission_metrics=_arcade_mission_metrics(
                _mission_metrics(training_cfg, score),
                enabled=arcade_enabled,
                round_index=arcade_round_index,
                total_score=arcade_total_score,
                is_boss=_arcade_round_is_boss(config, arcade_round_index),
            ),
            objective_checklist=_mission_checklist(training_cfg, score),
            speed_multiple=current_speed_multiple,
            selected_speed_multiple=current_speed_multiple,
            recording_status=_clip_recording_status(
                clip_recording_controller,
                started_wall_s=clip_recording_started_wall,
                now_wall_s=perf_counter(),
                status_message=clip_recording_status_message,
                status_until_wall_s=clip_recording_status_until,
            ),
            briefing_lines=briefing_lines if phase_shows_briefing(phase) else (),
            debrief_lines=_score_debrief_lines(score, config=training_cfg, difficulty=difficulty),
            debrief_available=debrief_enabled,
            render_motion=not command_state.paused and not phase_shows_briefing(phase) and not phase_is_terminal(phase),
            pause_overlay=False,
        )
        recording_controller.capture(dashboard)
        if phase_shows_briefing(phase):
            recording_controller.capture_hold(dashboard, duration_s=FULL_ATTEMPT_RECORDING_PAD_S)
        clip_recording_controller.capture(dashboard)

        pygame = dashboard.pygame
        audio_controller = GameAudioController(pygame=pygame, music_enabled=music_enabled)
        audio_controller.sync(
            score,
            training_cfg=training_cfg,
            override_level_path=_arcade_round_music_path(config, arcade_round_index) if arcade_enabled else None,
        )
        dashboard_fps_cap = _game_dashboard_fps_cap(config)
        dashboard_high_speed_fps = _game_dashboard_high_speed_fps(config)
        dashboard_high_speed_fps_max_multiple = _game_dashboard_high_speed_fps_max_multiple(config)
        effective_speed_multiple = current_speed_multiple
        dt_s = _game_active_tick_dt_s(config, effective_speed_multiple, maneuver_active=False)
        wall_step_s = _wall_step_s(dt_s, effective_speed_multiple)
        last_step_wall = perf_counter()
        last_input_wall = last_step_wall
        while (not command_state.quit_requested) and (not dashboard.closed):
            briefing_open = phase_shows_briefing(phase)
            debrief_hotkey_enabled = phase_is_terminal(phase)
            input_now = perf_counter()
            frame_started_wall = input_now
            input_elapsed_wall = max(float(input_now) - float(last_input_wall), 0.0)
            last_input_wall = input_now
            _poll_pygame_input(
                pygame,
                command_state,
                control_mode=control_mode,
                briefing_open=briefing_open,
                terminal_open=debrief_hotkey_enabled,
                frame_convention=frame_convention,
            )
            if not operator_playback_mode:
                _clear_two_rail_released_maneuver_input(config, command_state, control_mode=control_mode)
                _request_pilot_input_poll_for_transition(
                    session,
                    command_provider,
                    command_state,
                    controlled_object_id=controlled_object_id,
                )
            if not debrief_hotkey_enabled:
                command_state.open_debrief_requested = False
            if command_state.quit_requested:
                break
            if briefing_open and command_state.briefing_scroll_px:
                dashboard.scroll_briefing(command_state.briefing_scroll_px)
            elif debrief_hotkey_enabled and command_state.briefing_scroll_px:
                dashboard.scroll_mission_banner(command_state.briefing_scroll_px)
            if briefing_open and not command_state.paused:
                if ric_primer_enabled:
                    phase = GamePhase.PRIMER
                    ric_primer.reset()
                    command_state.paused = True
                    command_state.reset_axes()
                else:
                    phase = GamePhase.PLAYING
                dashboard.reset_briefing_scroll()
                last_step_wall = perf_counter()
                last_input_wall = last_step_wall
            if command_state.music_toggle_requested:
                command_state.music_toggle_requested = False
                audio_controller.toggle(
                    trainer.score(),
                    training_cfg=training_cfg,
                    override_level_path=(
                        _arcade_round_music_path(config, arcade_round_index) if arcade_enabled else None
                    ),
                )
            if phase == GamePhase.PRIMER:
                if command_state.restart_requested:
                    ric_primer.reset()
                    command_state.restart_requested = False
                    command_state.paused = True
                elif not command_state.paused:
                    ric_primer.stage_index += 1
                    ric_primer.elapsed_s = 0.0
                    if ric_primer.stage_index >= RIC_PRIMER_STAGE_COUNT:
                        phase = GamePhase.PLAYING
                        command_state.paused = (
                            _guided_tutorial_current_stage(
                                training_cfg,
                                guided_tutorial,
                            )
                            is not None
                        )
                    else:
                        command_state.paused = True
                if phase == GamePhase.PRIMER:
                    ric_primer.elapsed_s += input_elapsed_wall
                    command_state.reset_axes()
                    command_state.speed_multiplier_change = 0
                    command_state.camera_rule_toggle_requested = False
                    command_state.eci_ri_plot_toggle_requested = False
                    command_state.eci_rc_plot_toggle_requested = False
                    command_state.clip_record_toggle_requested = False
                    command_state.clip_record_save_requested = False
                    command_state.open_debrief_requested = False
                    dashboard.draw_ric_primer(
                        stage_index=ric_primer.stage_index,
                        elapsed_s=ric_primer.elapsed_s,
                        recording_status=_clip_recording_status(
                            clip_recording_controller,
                            started_wall_s=clip_recording_started_wall,
                            now_wall_s=perf_counter(),
                            status_message=clip_recording_status_message,
                            status_until_wall_s=clip_recording_status_until,
                        ),
                    )
                    recording_controller.capture(dashboard)
                    clip_recording_controller.capture(dashboard)
                    primer_target_fps = _presentation_fps_for_frame(
                        presentation_controller,
                        current_speed_multiple,
                        static_screen=True,
                        recording=(
                            recording_controller.recorder is not None or clip_recording_controller.recording
                        ),
                        recording_fps=recording_fps,
                        fps_cap=dashboard_fps_cap,
                        high_speed_fps=dashboard_high_speed_fps,
                        high_speed_fps_max_multiple=dashboard_high_speed_fps_max_multiple,
                    )
                    if presentation_controller is not None:
                        presentation_controller.observe_frame(
                            work_s=perf_counter() - frame_started_wall,
                            authoritative_steps=0,
                            snapshot_age_s=_dashboard_snapshot_age_s(dashboard),
                        )
                    dashboard.tick(primer_target_fps)
                    continue
            if (
                operator_tutorial is not None
                and not operator_tutorial.completed
                and operator_tutorial.awaiting_script
                and phase != GamePhase.PRIMER
            ):
                stage = _operator_tutorial_current_stage(operator_tutorial)
                if stage is None:
                    operator_tutorial.completed = True
                    operator_tutorial_level_passed = True
                    score = _operator_tutorial_complete_score(score)
                    command_state.paused = True
                    phase = GamePhase.PASSED
                    continue
                command_state.reset_axes()
                command_state.paused = True
                current_speed_multiple = _coerce_speed_multiple(
                    OPERATOR_TUTORIAL_PLAYBACK_SPEED_MULTIPLE,
                    options=speed_multiplier_options,
                )
                effective_speed_multiple = current_speed_multiple
                dt_s = _game_active_tick_dt_s(config, effective_speed_multiple, maneuver_active=False)
                wall_step_s = _wall_step_s(dt_s, effective_speed_multiple)
                selected_plan = plan_operator_burns_for_config(
                    dashboard.pygame,
                    dashboard.screen,
                    dashboard.clock,
                    config_path,
                    font=dashboard.font,
                    small_font=dashboard.small_font,
                    title_font=dashboard.large_font,
                    initial_plan=stage.plan,
                    difficulty=difficulty,
                    frame_convention=frame_convention,
                    read_only=True,
                    demo_title=_operator_tutorial_demo_title(operator_tutorial),
                    launch_label="Launch Demo",
                    config_override=config,
                )
                last_step_wall = perf_counter()
                last_input_wall = last_step_wall
                if selected_plan is None:
                    command_state.quit_requested = True
                    break
                restart_attempt_for_operator_plan(stage.plan, tutorial_stage=stage)
                operator_tutorial.awaiting_script = False
                operator_tutorial.stage_start_sim_s = float(dashboard.t_s[-1]) if getattr(dashboard, "t_s", ()) else 0.0
                command_state.paused = False
                phase = GamePhase.PLAYING
                current_speed_multiple = _coerce_speed_multiple(
                    OPERATOR_TUTORIAL_PLAYBACK_SPEED_MULTIPLE,
                    options=speed_multiplier_options,
                )
                effective_speed_multiple = current_speed_multiple
                dt_s = _game_active_tick_dt_s(config, effective_speed_multiple, maneuver_active=False)
                wall_step_s = _wall_step_s(dt_s, effective_speed_multiple)
                last_step_wall = perf_counter()
                last_input_wall = last_step_wall
                continue
            if command_state.speed_multiplier_change:
                previous_speed_multiple = current_speed_multiple
                speed_step_change = int(np.sign(command_state.speed_multiplier_change))
                current_speed_multiple = _adjust_speed_multiple(
                    current_speed_multiple,
                    speed_step_change,
                    options=speed_multiplier_options,
                )
                if not np.isclose(current_speed_multiple, previous_speed_multiple):
                    trainer.record_speed_multiplier_change()
                effective_speed_multiple = _effective_speed_multiple_for_mode(
                    config,
                    current_speed_multiple,
                    command_state,
                    game_mode=game_mode,
                    control_mode=control_mode,
                    options=speed_multiplier_options,
                )
                dt_s = _game_active_tick_dt_s(
                    config,
                    effective_speed_multiple,
                    maneuver_active=_manual_maneuver_active_for_mode(
                        game_mode,
                        command_state,
                        control_mode=control_mode,
                    ),
                )
                wall_step_s = _wall_step_s(dt_s, effective_speed_multiple)
                command_state.speed_multiplier_change = 0
                last_step_wall = perf_counter()
                last_input_wall = last_step_wall
            if command_state.camera_rule_toggle_requested:
                if _camera_rule_toggle_enabled_for_dashboard(dashboard, training_cfg) and hasattr(
                    dashboard, "toggle_camera_rule_mode"
                ):
                    dashboard.toggle_camera_rule_mode()
                command_state.camera_rule_toggle_requested = False
            if command_state.eci_ri_plot_toggle_requested:
                if hasattr(dashboard, "toggle_eci_plot"):
                    dashboard.toggle_eci_plot("RI")
                command_state.eci_ri_plot_toggle_requested = False
            if command_state.eci_rc_plot_toggle_requested:
                if hasattr(dashboard, "toggle_eci_plot"):
                    dashboard.toggle_eci_plot("RC")
                command_state.eci_rc_plot_toggle_requested = False
            if operator_tutorial is not None and not operator_tutorial.awaiting_script:
                current_speed_multiple = _coerce_speed_multiple(
                    OPERATOR_TUTORIAL_PLAYBACK_SPEED_MULTIPLE,
                    options=speed_multiplier_options,
                )
            if (
                not operator_playback_mode
                and guided_tutorial.awaiting_speed_step
                and _guided_tutorial_speed_step_reached(
                    training_cfg,
                    current_speed_multiple,
                )
            ):
                trainer.mark_guided_tutorial_speed_complete()
                guided_tutorial.awaiting_speed_step = False
                session, snapshot = _reset_guided_tutorial_stage_attempt(
                    attempt_config=attempt_config,
                    command_state=command_state,
                    trainer=trainer,
                    dashboard=dashboard,
                    training_cfg=training_cfg,
                    controlled_object_id=controlled_object_id,
                    attitude_rate_deg_s=attitude_rate_deg_s,
                    control_mode=control_mode,
                    ric_reference_object_id=ric_reference_object_id,
                )
                command_state.paused = _guided_tutorial_current_stage(training_cfg, guided_tutorial) is not None
                _guided_tutorial_update_dashboard_path(dashboard, trainer, training_cfg, guided_tutorial)
                last_step_wall = perf_counter()
                last_input_wall = last_step_wall
            if not operator_playback_mode:
                maneuver_speed_multiple = _speed_after_maneuver_input(
                    current_speed_multiple,
                    command_state,
                    control_mode=control_mode,
                    options=speed_multiplier_options,
                    maneuver_control_speed_multiple=maneuver_control_speed_multiple,
                )
                if two_rail_speed_control:
                    if not np.isclose(maneuver_speed_multiple, effective_speed_multiple):
                        effective_speed_multiple = maneuver_speed_multiple
                        dt_s = _game_active_tick_dt_s(
                            config,
                            effective_speed_multiple,
                            maneuver_active=_manual_maneuver_active_for_mode(
                                game_mode,
                                command_state,
                                control_mode=control_mode,
                            ),
                        )
                        wall_step_s = _wall_step_s(dt_s, effective_speed_multiple)
                elif not np.isclose(maneuver_speed_multiple, current_speed_multiple):
                    current_speed_multiple = maneuver_speed_multiple
                    effective_speed_multiple = current_speed_multiple
                    dt_s = _game_active_tick_dt_s(
                        config,
                        effective_speed_multiple,
                        maneuver_active=_manual_maneuver_active_for_mode(
                            game_mode,
                            command_state,
                            control_mode=control_mode,
                        ),
                    )
                    wall_step_s = _wall_step_s(dt_s, effective_speed_multiple)
                    last_step_wall = perf_counter()
                    last_input_wall = last_step_wall
            if command_state.clip_record_toggle_requested:
                command_state.clip_record_toggle_requested = False
                if clip_recording_controller.recording:
                    clip_recording_controller.discard()
                    clip_recording_started_wall = None
                    clip_recording_status_message = "Clip discarded"
                    clip_recording_status_until = perf_counter() + 2.5
                elif phase_shows_briefing(phase) or phase_is_terminal(phase):
                    clip_recording_status_message = "Clip starts during play"
                    clip_recording_status_until = perf_counter() + 2.5
                else:
                    recorder = clip_recording_controller.start_next()
                    if recorder is not None:
                        clip_recording_started_wall = perf_counter()
                        clip_recording_status_message = ""
                        clip_recording_status_until = 0.0
                    else:
                        clip_recording_started_wall = None
                        clip_recording_status_message = "Clip recording unavailable"
                        clip_recording_status_until = perf_counter() + 2.5
            if command_state.clip_record_save_requested:
                command_state.clip_record_save_requested = False
                if clip_recording_controller.recording:
                    clip_path = clip_recording_controller.finish(
                        base_training_cfg,
                        override_level_path=(
                            _arcade_round_music_path(config, arcade_round_index) if arcade_enabled else None
                        ),
                    )
                    clip_recording_started_wall = None
                    if clip_path is not None:
                        print(f"Saved game clip recording: {clip_path}")
                        clip_recording_status_message = "Clip saved"
                    else:
                        clip_recording_status_message = "Clip save failed"
                    clip_recording_status_until = perf_counter() + 2.5
                else:
                    clip_recording_status_message = "No active clip"
                    clip_recording_status_until = perf_counter() + 2.5
            if command_state.restart_requested:
                if operator_tutorial is not None and not operator_tutorial.completed:
                    command_state.restart_requested = False
                    command_state.reset_axes()
                    operator_tutorial.awaiting_script = True
                    operator_tutorial.stage_start_sim_s = None
                    command_state.paused = True
                    phase = GamePhase.PAUSED
                    last_step_wall = perf_counter()
                    last_input_wall = last_step_wall
                    continue
                if game_mode == "operator":
                    command_state.restart_requested = False
                    command_state.paused = True
                    revised_plan = plan_operator_burns_for_config(
                        dashboard.pygame,
                        dashboard.screen,
                        dashboard.clock,
                        config_path,
                        font=dashboard.font,
                        small_font=dashboard.small_font,
                        title_font=dashboard.large_font,
                        initial_plan=operator_burn_plan,
                        frame_convention=frame_convention,
                        config_override=config,
                    )
                    last_step_wall = perf_counter()
                    last_input_wall = last_step_wall
                    if revised_plan is None:
                        command_state.reset_axes()
                        continue
                    operator_burn_plan = revised_plan
                recording_controller.discard()
                clip_recording_controller.discard()
                clip_recording_started_wall = None
                clip_recording_status_message = ""
                clip_recording_status_until = 0.0
                audio_controller.stop()
                if arcade_enabled:
                    arcade_round_index = 1
                    arcade_total_score = 0
                    arcade_remaining_time_s = _game_arcade_initial_time_s(config, base_training_cfg)
                    training_cfg = _arcade_round_training_config(
                        config,
                        base_training_cfg,
                        round_index=arcade_round_index,
                        max_time_s=arcade_remaining_time_s,
                    )
                    training_cfg = _training_config_with_sun_environment(training_cfg, config)
                    training_cfg = training_config_for_game_mode(training_cfg, game_mode=game_mode)
                    attempt_config = _arcade_round_simulation_config(
                        config,
                        training_cfg,
                        round_index=arcade_round_index,
                        rng=(
                            _arcade_round_initial_state_rng(int(arcade_seed_value), arcade_round_index)
                            if arcade_seed_value is not None
                            else None
                        ),
                    )
                    player_max_accel_km_s2 = _max_accel_from_config(attempt_config, controlled_object_id)
                    trainer = RPOTrainingTracker(training_cfg)
                    guided_tutorial = GuidedTutorialRuntime()
                    ric_primer = RICPrimerRuntime()
                    ric_primer_enabled = _ric_primer_enabled(training_cfg, arcade_enabled=arcade_enabled)
                recording_controller.restart()
                recording_attempt = recording_controller.attempt_index
                recording_path = None
                debrief_path = None
                session, command_provider, snapshot = _start_game_attempt(
                    attempt_config,
                    command_state=command_state,
                    training_cfg=training_cfg,
                    controlled_object_id=controlled_object_id,
                    attitude_rate_deg_s=attitude_rate_deg_s,
                    control_mode=control_mode,
                    ric_reference_object_id=ric_reference_object_id,
                    operator_burn_plan=operator_burn_plan,
                    operator_actuator_error_fraction=operator_actuator_error_fraction,
                    defensive_target_profile=(
                        _game_random_direction_defensive_target_profile(
                            config,
                            round_index=arcade_round_index,
                            rng=_arcade_round_rng(int(arcade_seed_value), arcade_round_index),
                        )
                        if arcade_enabled and arcade_seed_value is not None
                        else None
                    ),
                )
                trainer.clear()
                guided_tutorial = GuidedTutorialRuntime()
                ric_primer = RICPrimerRuntime()
                ric_primer_enabled = _ric_primer_enabled(training_cfg, arcade_enabled=arcade_enabled)
                operator_burn_cinematic.reset()
                dashboard.clear()
                if operator_playback_mode:
                    _clear_live_prediction_burn(dashboard)
                _sync_dashboard_training_config(dashboard, training_cfg)
                _sync_dashboard_round_config(dashboard, attempt_config)
                dashboard.push_snapshot(snapshot)
                trainer.record(snapshot, control_telemetry_provider=command_provider)
                _sync_guided_tutorial_path_for_mode(
                    dashboard,
                    trainer,
                    training_cfg,
                    guided_tutorial,
                    game_mode=game_mode,
                )
                command_state.restart_requested = False
                command_state.speed_multiplier_change = 0
                command_state.camera_rule_toggle_requested = False
                command_state.eci_ri_plot_toggle_requested = False
                command_state.eci_rc_plot_toggle_requested = False
                command_state.music_toggle_requested = False
                command_state.clip_record_toggle_requested = False
                command_state.clip_record_save_requested = False
                command_state.open_debrief_requested = False
                restart_skips_briefing = bool(game_mode == "operator")
                command_state.paused = bool(training_cfg.enabled and not restart_skips_briefing)
                phase = GamePhase.BRIEFING if training_cfg.enabled and not restart_skips_briefing else GamePhase.PLAYING
                dashboard.reset_briefing_scroll()
                dashboard.reset_mission_banner_scroll()
                effective_speed_multiple = current_speed_multiple
                dt_s = _game_active_tick_dt_s(config, effective_speed_multiple, maneuver_active=False)
                wall_step_s = _wall_step_s(dt_s, effective_speed_multiple)
                last_step_wall = perf_counter()
                last_input_wall = last_step_wall
            now = perf_counter()
            pre_score = trainer.score()
            if operator_tutorial_level_passed:
                pre_score = _operator_tutorial_complete_score(pre_score)
            phase = _phase_from_score_with_operator_animation(
                pre_score,
                briefing_open=phase_shows_briefing(phase),
                paused=command_state.paused,
                game_mode=game_mode,
                operator_burn_cinematic=operator_burn_cinematic,
            )
            mission_decided = phase_is_terminal(phase)
            if _game_loop_should_exit(session_done=session.done, score=pre_score):
                break
            if mission_decided:
                command_state.paused = True
            guided_stage = (
                None if operator_playback_mode else _guided_tutorial_current_stage(training_cfg, guided_tutorial)
            )
            if (
                guided_stage is not None
                and not briefing_open
                and not mission_decided
                and not session.done
                and operator_tutorial is None
            ):
                guided_input_ok = _guided_tutorial_input_matches(command_state, guided_stage)
                guided_tutorial.wrong_key_active = _guided_tutorial_wrong_input_active(command_state, guided_stage)
                command_state.paused = not guided_input_ok
            elif (
                not operator_playback_mode
                and guided_tutorial.awaiting_speed_step
                and not briefing_open
                and not mission_decided
            ):
                guided_tutorial.wrong_key_active = False
                command_state.reset_axes()
                command_state.paused = True
            else:
                guided_tutorial.wrong_key_active = False
            maneuver_active = (
                not briefing_open
                and not mission_decided
                and not session.done
                and not command_state.paused
                and _manual_maneuver_active_for_mode(
                    game_mode,
                    command_state,
                    control_mode=control_mode,
                )
            )
            base_next_effective_speed_multiple = _effective_speed_multiple_for_mode(
                config,
                current_speed_multiple,
                command_state,
                game_mode=game_mode,
                control_mode=control_mode,
                options=speed_multiplier_options,
            )
            base_next_dt_s = _game_active_tick_dt_s(
                config,
                base_next_effective_speed_multiple,
                maneuver_active=maneuver_active,
            )
            base_frame_horizon_s = base_next_dt_s
            if realtime:
                base_wall_step_s = _wall_step_s(base_next_dt_s, base_next_effective_speed_multiple)
                base_steps_due, _ = _realtime_steps_due(
                    now_s=now,
                    last_step_wall_s=last_step_wall,
                    wall_step_s=base_wall_step_s,
                )
                base_frame_horizon_s = base_next_dt_s * max(int(base_steps_due), 1)
            if game_mode == "operator":
                _update_operator_burn_cinematic(
                    operator_burn_cinematic,
                    command_provider,
                    now_wall_s=now,
                    current_sim_time_s=float(dashboard.t_s[-1]) if getattr(dashboard, "t_s", ()) else 0.0,
                    dt_s=base_next_dt_s,
                    frame_horizon_s=base_frame_horizon_s,
                )
            next_effective_speed_multiple = (
                _operator_burn_cinematic_speed_multiple(
                    base_next_effective_speed_multiple,
                    operator_burn_cinematic,
                    options=speed_multiplier_options,
                )
                if game_mode == "operator"
                else base_next_effective_speed_multiple
            )
            next_dt_s = _game_active_tick_dt_s(
                config,
                next_effective_speed_multiple,
                maneuver_active=maneuver_active,
            )
            if not np.isclose(next_effective_speed_multiple, effective_speed_multiple) or not np.isclose(
                next_dt_s, dt_s
            ):
                effective_speed_multiple = next_effective_speed_multiple
                dt_s = next_dt_s
                wall_step_s = _wall_step_s(dt_s, effective_speed_multiple)
                if not two_rail_speed_control:
                    last_step_wall = now
            if (
                not briefing_open
                and not mission_decided
                and not session.done
                and not command_state.paused
                and not operator_playback_mode
            ):
                command_state.accumulate_timed_input(
                    input_elapsed_wall,
                    speed_multiple=effective_speed_multiple,
                    control_mode=control_mode,
                    max_pending_sim_s=float(dt_s) * float(_game_maneuver_input_max_pending_steps(config)),
                )
                command_state.clear_event_pulses()
            if command_state.paused:
                command_state.clear_timed_input()
            if command_state.paused:
                last_step_wall = now
            static_screen = bool(
                command_state.paused or phase_shows_briefing(phase) or phase_is_terminal(phase)
            )
            recording_active = bool(
                recording_controller.recorder is not None or clip_recording_controller.recording
            )
            target_presentation_fps = _presentation_fps_for_frame(
                presentation_controller,
                effective_speed_multiple,
                static_screen=static_screen,
                recording=recording_active,
                recording_fps=recording_fps,
                fps_cap=dashboard_fps_cap,
                high_speed_fps=dashboard_high_speed_fps,
                high_speed_fps_max_multiple=dashboard_high_speed_fps_max_multiple,
            )
            steps_to_run = 0
            discarded_backlog_steps = 0
            step_dt_s = dt_s
            pending_maneuver_sim_s = _pending_maneuver_sim_s_for_mode(
                game_mode,
                command_state,
                control_mode=control_mode,
            )
            if not mission_decided and not session.done and not command_state.paused:
                if realtime:
                    if presentation_controller is None or not presentation_controller.trajectory_aware:
                        steps_to_run, last_step_wall = _realtime_steps_due(
                            now_s=now,
                            last_step_wall_s=last_step_wall,
                            wall_step_s=wall_step_s,
                        )
                    else:
                        max_presentation_steps = presentation_controller.authoritative_step_limit(
                            wall_step_s=wall_step_s,
                            hard_limit=MAX_REALTIME_STEPS_PER_FRAME,
                        )
                        steps_to_run, last_step_wall, discarded_backlog_steps = (
                            _realtime_steps_due_with_backlog(
                                now_s=now,
                                last_step_wall_s=last_step_wall,
                                wall_step_s=wall_step_s,
                                max_steps=max_presentation_steps,
                            )
                        )
                    if steps_to_run <= 0 and pending_maneuver_sim_s > 1.0e-9:
                        step_dt_s = min(float(dt_s), float(pending_maneuver_sim_s))
                        steps_to_run = 1
                        last_step_wall = now
                else:
                    steps_to_run = 1
                    last_step_wall = now
            burn_trace_interesting = (
                bool(burn_trace_enabled)
                and not briefing_open
                and (
                    abs(float(getattr(command_state, "pitch", 0.0))) > 1.0e-12
                    or abs(float(getattr(command_state, "yaw", 0.0))) > 1.0e-12
                    or abs(float(getattr(command_state, "roll", 0.0))) > 1.0e-12
                    or bool(getattr(command_state, "firing", False))
                    or bool(getattr(command_state, "pitch_event_pulse", False))
                    or bool(getattr(command_state, "yaw_event_pulse", False))
                    or bool(getattr(command_state, "roll_event_pulse", False))
                    or bool(getattr(command_state, "firing_event_pulse", False))
                    or pending_maneuver_sim_s > 1.0e-12
                    or steps_to_run > 0
                )
            )
            if burn_trace_interesting:
                _trace_burn_loop(
                    "pre "
                    f"t={float(getattr(snapshot, 'time_s', 0.0)):.6f} "
                    f"input_wall={input_elapsed_wall:.6f} "
                    f"axes=({float(command_state.pitch):+.0f},{float(command_state.yaw):+.0f},{float(command_state.roll):+.0f}) "
                    f"pulses=({int(bool(command_state.pitch_event_pulse))},{int(bool(command_state.yaw_event_pulse))},{int(bool(command_state.roll_event_pulse))}) "
                    f"pending={pending_maneuver_sim_s:.9f} "
                    f"speed={effective_speed_multiple:g}x "
                    f"dt={dt_s:.6f} step_dt={step_dt_s:.9f} steps={steps_to_run}"
                )
            score = _step_game_attempt(
                session=session,
                dashboard=dashboard,
                trainer=trainer,
                steps_to_run=steps_to_run,
                initial_score=pre_score,
                dt_s=step_dt_s,
                max_step_dt_s=_game_max_autonomy_step_s(config),
                control_telemetry_provider=command_provider,
                operator_command_provider=command_provider if game_mode == "operator" else None,
                operator_burn_transition_callback=(
                    hold_operator_burn_cinematic_for_animation if game_mode == "operator" else None
                ),
            )
            if burn_trace_interesting:
                engine = session._engine
                applied_norm = 0.0
                sim_t = float(getattr(snapshot, "time_s", 0.0))
                if engine is not None:
                    k_trace = int(getattr(engine, "current_index", 0))
                    try:
                        sim_t = float(engine.t_s[k_trace])
                    except Exception:
                        sim_t = float(getattr(snapshot, "time_s", 0.0))
                    try:
                        thrust_hist = getattr(engine, "thrust_hist", {})
                        thrust = np.array(thrust_hist.get(str(controlled_object_id))[k_trace], dtype=float).reshape(3)
                        applied_norm = float(np.linalg.norm(thrust))
                    except Exception:
                        applied_norm = 0.0
                _trace_burn_loop(
                    "post "
                    f"t={sim_t:.6f} "
                    f"applied_norm={applied_norm:.9e} "
                    f"pending={_pending_maneuver_sim_s_for_mode(game_mode, command_state, control_mode=control_mode):.9f}"
                )
            guided_stage_completed = False
            completed_guided_stage: Any | None = None
            if (
                steps_to_run > 0
                and guided_stage is not None
                and not bool(getattr(score, "level_failed", False))
                and operator_tutorial is None
            ):
                completed_guided_stage = guided_stage
                guided_stage_completed = _guided_tutorial_complete_active_stage(
                    trainer,
                    training_cfg,
                    guided_tutorial,
                )
                score = trainer.score()
            if guided_stage_completed:
                if (
                    _guided_tutorial_speed_step_follows_burn(training_cfg, completed_guided_stage)
                    and not trainer.guided_tutorial_speed_satisfied()
                ):
                    guided_tutorial.awaiting_speed_step = True
                    guided_tutorial.wrong_key_active = False
                    session, _ = _reset_guided_tutorial_stage_attempt(
                        attempt_config=attempt_config,
                        command_state=command_state,
                        trainer=trainer,
                        dashboard=dashboard,
                        training_cfg=training_cfg,
                        controlled_object_id=controlled_object_id,
                        attitude_rate_deg_s=attitude_rate_deg_s,
                        control_mode=control_mode,
                        ric_reference_object_id=ric_reference_object_id,
                    )
                    command_state.reset_axes()
                    command_state.paused = True
                    _guided_tutorial_update_dashboard_path(dashboard, trainer, training_cfg, guided_tutorial)
                    score = trainer.score()
                    last_step_wall = perf_counter()
                    last_input_wall = last_step_wall
                else:
                    session, snapshot = _reset_guided_tutorial_stage_attempt(
                        attempt_config=attempt_config,
                        command_state=command_state,
                        trainer=trainer,
                        dashboard=dashboard,
                        training_cfg=training_cfg,
                        controlled_object_id=controlled_object_id,
                        attitude_rate_deg_s=attitude_rate_deg_s,
                        control_mode=control_mode,
                        ric_reference_object_id=ric_reference_object_id,
                    )
                    command_state.paused = _guided_tutorial_current_stage(training_cfg, guided_tutorial) is not None
                    guided_tutorial.wrong_key_active = False
                    _guided_tutorial_update_dashboard_path(dashboard, trainer, training_cfg, guided_tutorial)
                    score = trainer.score()
                    last_step_wall = perf_counter()
                    last_input_wall = last_step_wall
            if (
                operator_tutorial is not None
                and not operator_tutorial.completed
                and not operator_tutorial.awaiting_script
                and not bool(getattr(score, "level_failed", False))
            ):
                current_sim_s = float(dashboard.t_s[-1]) if getattr(dashboard, "t_s", ()) else 0.0
                stage_start_s = (
                    current_sim_s
                    if operator_tutorial.stage_start_sim_s is None
                    else float(operator_tutorial.stage_start_sim_s)
                )
                if current_sim_s - stage_start_s >= OPERATOR_TUTORIAL_STAGE_DURATION_S:
                    operator_tutorial.stage_index += 1
                    operator_tutorial.stage_start_sim_s = None
                    operator_tutorial.awaiting_script = True
                    command_state.reset_axes()
                    command_state.paused = True
                    if _operator_tutorial_current_stage(operator_tutorial) is None:
                        operator_tutorial.completed = True
                        operator_tutorial_level_passed = True
                        score = _operator_tutorial_complete_score(score)
                        phase = GamePhase.PASSED
                        last_step_wall = perf_counter()
                        last_input_wall = last_step_wall
                        continue
                    phase = GamePhase.PAUSED
                    last_step_wall = perf_counter()
                    last_input_wall = last_step_wall
                    continue
            if score.level_passed or score.level_failed:
                command_state.paused = True
                phase = _phase_from_score_with_operator_animation(
                    score,
                    paused=True,
                    game_mode=game_mode,
                    operator_burn_cinematic=operator_burn_cinematic,
                )
            else:
                phase = _phase_from_score_with_operator_animation(
                    score,
                    briefing_open=phase_shows_briefing(phase),
                    paused=command_state.paused,
                    game_mode=game_mode,
                    operator_burn_cinematic=operator_burn_cinematic,
                )
            if arcade_enabled and bool(getattr(score, "level_passed", False)):
                audio_controller.play_round_clear()
                round_score = _arcade_round_weighted_score(
                    training_cfg,
                    score,
                    difficulty=difficulty,
                    round_index=arcade_round_index,
                    arcade_config=config,
                )
                arcade_total_score += int(round_score)
                time_used = _score_time_used_s(score)
                round_bonus_s = _arcade_round_time_bonus_s(
                    config,
                    training_cfg,
                    score,
                    round_index=arcade_round_index,
                )
                assert arcade_remaining_time_s is not None
                arcade_remaining_time_s = max(float(arcade_remaining_time_s) - time_used, 0.0) + round_bonus_s
                cleared_round_index = arcade_round_index
                arcade_round_index += 1
                training_cfg = _arcade_round_training_config(
                    config,
                    base_training_cfg,
                    round_index=arcade_round_index,
                    max_time_s=arcade_remaining_time_s,
                )
                training_cfg = _training_config_with_sun_environment(training_cfg, config)
                training_cfg = training_config_for_game_mode(training_cfg, game_mode=game_mode)
                attempt_config = _arcade_round_simulation_config(
                    config,
                    training_cfg,
                    round_index=arcade_round_index,
                    rng=_arcade_round_initial_state_rng(int(arcade_seed_value), arcade_round_index),
                )
                player_max_accel_km_s2 = _max_accel_from_config(attempt_config, controlled_object_id)
                trainer = RPOTrainingTracker(training_cfg)
                guided_tutorial = GuidedTutorialRuntime()
                ric_primer = RICPrimerRuntime()
                ric_primer_enabled = _ric_primer_enabled(training_cfg, arcade_enabled=arcade_enabled)
                operator_burn_cinematic.reset()
                session, command_provider, snapshot = _start_game_attempt(
                    attempt_config,
                    command_state=command_state,
                    training_cfg=training_cfg,
                    controlled_object_id=controlled_object_id,
                    attitude_rate_deg_s=attitude_rate_deg_s,
                    control_mode=control_mode,
                    ric_reference_object_id=ric_reference_object_id,
                    operator_burn_plan=operator_burn_plan,
                    operator_actuator_error_fraction=operator_actuator_error_fraction,
                    defensive_target_profile=_game_random_direction_defensive_target_profile(
                        config,
                        round_index=arcade_round_index,
                        rng=_arcade_round_rng(int(arcade_seed_value), arcade_round_index),
                    ),
                )
                dashboard.clear()
                if operator_playback_mode:
                    _clear_live_prediction_burn(dashboard)
                _sync_dashboard_training_config(dashboard, training_cfg)
                _sync_dashboard_round_config(dashboard, attempt_config)
                dashboard.push_snapshot(snapshot)
                trainer.record(snapshot, control_telemetry_provider=command_provider)
                _sync_guided_tutorial_path_for_mode(
                    dashboard,
                    trainer,
                    training_cfg,
                    guided_tutorial,
                    game_mode=game_mode,
                )
                score = trainer.score()
                briefing_lines = _arcade_round_briefing_lines(
                    cleared_round_index=cleared_round_index,
                    next_round_index=arcade_round_index,
                    round_score=int(round_score),
                    total_score=arcade_total_score,
                    time_used_s=time_used,
                    bonus_time_s=round_bonus_s,
                    next_time_budget_s=arcade_remaining_time_s,
                    next_goal_range_km=training_cfg.goal_range_km,
                    next_is_boss=_arcade_round_is_boss(config, arcade_round_index),
                )
                command_state.paused = True
                command_state.restart_requested = False
                command_state.speed_multiplier_change = 0
                command_state.music_toggle_requested = False
                command_state.open_debrief_requested = False
                phase = GamePhase.ARCADE_TRANSITION
                dashboard.reset_briefing_scroll()
                audio_controller.clear_active_path()
                last_step_wall = perf_counter()
                last_input_wall = last_step_wall
            recording_music_path = _arcade_round_music_path(config, arcade_round_index) if arcade_enabled else None
            audio_controller.sync(
                score,
                training_cfg=training_cfg,
                override_level_path=recording_music_path,
            )
            phase = _phase_from_score_with_operator_animation(
                score,
                briefing_open=phase_shows_briefing(phase),
                paused=command_state.paused,
                game_mode=game_mode,
                operator_burn_cinematic=operator_burn_cinematic,
            )
            if operator_tutorial is None:
                _sync_guided_tutorial_path_for_mode(
                    dashboard,
                    trainer,
                    training_cfg,
                    guided_tutorial,
                    game_mode=game_mode,
                )
            _sync_live_prediction_burn_for_mode(
                dashboard,
                command_state,
                game_mode=game_mode,
                control_mode=control_mode,
                max_accel_km_s2=player_max_accel_km_s2,
                elapsed_wall_s=max(float(now) - float(last_step_wall), float(input_elapsed_wall), 0.0),
                speed_multiple=effective_speed_multiple,
                dt_s=dt_s,
            )
            _sync_dashboard_aerodynamic_control(dashboard, command_provider)
            dashboard.draw(
                command_status=_game_command_status(
                    command_state,
                    control_mode=control_mode,
                    game_mode=game_mode,
                    command_provider=command_provider,
                    frame_convention=frame_convention,
                ),
                coach_hint=_coach_hint_with_camera_rule(
                    (_operator_tutorial_status(operator_tutorial) if operator_tutorial is not None else "")
                    or (
                        _guided_tutorial_speed_step_hint(training_cfg, current_speed_multiple)
                        if guided_tutorial.awaiting_speed_step
                        else ""
                    )
                    or _guided_tutorial_stage_hint(
                        _guided_tutorial_current_stage(training_cfg, guided_tutorial),
                        guided_tutorial,
                        frame_convention=frame_convention,
                    )
                    or trainer.current_hint(),
                    dashboard,
                    training_cfg,
                ),
                mission_state=mission_state_for_dashboard(phase),
                level_title=level_title,
                mission_metrics=_arcade_mission_metrics(
                    _mission_metrics(training_cfg, score),
                    enabled=arcade_enabled,
                    round_index=arcade_round_index,
                    total_score=arcade_total_score,
                    is_boss=_arcade_round_is_boss(config, arcade_round_index),
                ),
                objective_checklist=_mission_checklist(training_cfg, score),
                speed_multiple=effective_speed_multiple,
                selected_speed_multiple=current_speed_multiple,
                recording_status=_clip_recording_status(
                    clip_recording_controller,
                    started_wall_s=clip_recording_started_wall,
                    now_wall_s=perf_counter(),
                    status_message=clip_recording_status_message,
                    status_until_wall_s=clip_recording_status_until,
                ),
                briefing_lines=briefing_lines if phase_shows_briefing(phase) else (),
                debrief_lines=_score_debrief_lines(score, config=training_cfg, difficulty=difficulty),
                debrief_available=debrief_enabled,
                render_motion=not command_state.paused
                and not phase_shows_briefing(phase)
                and not phase_is_terminal(phase),
                pause_overlay=operator_tutorial is None
                and _pause_teaching_overlay_enabled(phase, training_cfg, guided_tutorial),
            )
            recording_controller.capture(dashboard)
            clip_recording_controller.capture(dashboard)
            terminal_screen_ready = phase_is_terminal(phase)
            recorder = recording_controller.recorder
            if recorder is not None and terminal_screen_ready and not recorder.saved:
                recording_controller.capture_hold(dashboard, duration_s=FULL_ATTEMPT_RECORDING_PAD_S)
                recording_path = recording_controller.finish(
                    base_training_cfg,
                    override_level_path=recording_music_path,
                )
                if recording_path is not None:
                    print(f"Saved game recording: {recording_path}")
                else:
                    recorder = None
            if debrief_enabled and terminal_screen_ready and debrief_path is None:
                debrief_attempt = next_game_debrief_attempt_index(
                    scenario_id=training_cfg.scenario_id,
                    output_dir=debrief_output_dir,
                )
                debrief_path = write_game_debrief(
                    game_debrief_path(
                        scenario_id=training_cfg.scenario_id,
                        difficulty=difficulty,
                        attempt_index=debrief_attempt,
                        output_dir=debrief_output_dir,
                    ),
                    config=training_cfg,
                    score=score,
                    difficulty=difficulty,
                    objective_checklist=_mission_checklist(training_cfg, score),
                    arcade_score=arcade_total_score
                    if arcade_enabled
                    else _arcade_score(training_cfg, score, difficulty=difficulty),
                    arcade_seed=arcade_seed_value if arcade_enabled else None,
                    arcade_round_index=arcade_round_index if arcade_enabled else None,
                    recording_path=recording_path,
                    replay_history=tracker_replay_history(trainer),
                )
                print(f"Saved game debrief: {debrief_path}")
            if command_state.open_debrief_requested and phase_is_terminal(phase) and debrief_path is not None:
                debrief_folder_to_open = debrief_path.parent
                command_state.quit_requested = True
                break
            if presentation_controller is not None:
                presentation_controller.observe_frame(
                    work_s=perf_counter() - frame_started_wall,
                    authoritative_steps=steps_to_run,
                    discarded_backlog_steps=discarded_backlog_steps,
                    snapshot_age_s=_dashboard_snapshot_age_s(dashboard),
                )
            dashboard.tick(target_presentation_fps)
    finally:
        recorder = recording_controller.recorder
        if recorder is not None and not recorder.saved:
            recording_controller.discard()
        if clip_recording_controller.recording:
            clip_recording_controller.discard()
        if audio_controller is not None:
            audio_controller.stop()
        else:
            _stop_game_music(getattr(dashboard, "pygame", None))
        if presentation_controller is not None:
            diagnostics_path = presentation_controller.write_summary()
            if diagnostics_path is not None:
                print(f"Saved game presentation diagnostics: {diagnostics_path}")
        dashboard.close()
        if debrief_folder_to_open is not None:
            opened = open_game_debrief_folder(debrief_folder_to_open)
            if opened:
                print(f"Opened game debrief folder: {debrief_folder_to_open}")
            else:
                print(f"Game debrief folder: {debrief_folder_to_open}")
        if training_cfg.enabled:
            print(trainer.debrief_text())
    final_arcade_score = arcade_total_score
    if arcade_enabled and bool(getattr(score, "level_passed", False)):
        final_arcade_score += _arcade_round_weighted_score(
            training_cfg,
            score,
            difficulty=difficulty,
            round_index=arcade_round_index,
            arcade_config=config,
        )
    return GameRunResult(
        config_path=Path(config_path),
        difficulty=difficulty,
        level_passed=bool(score.level_passed) or bool(operator_tutorial_level_passed),
        mode=game_mode,
        frame_convention=frame_convention,
        arcade_score=final_arcade_score
        if arcade_enabled
        else _arcade_score(training_cfg, score, difficulty=difficulty),
        arcade_seed=arcade_seed_value if arcade_enabled else None,
        recording_path=recording_path,
        debrief_path=debrief_path,
    )
