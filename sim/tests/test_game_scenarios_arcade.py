from __future__ import annotations

# These owner-aligned tests share deterministic builders and compatibility
# imports from the adjacent support module.
# ruff: noqa: F403, F405
from sim.tests.game_mode_test_support import *


def test_level_nine_uses_target_reference_for_game_ric_frame() -> None:
    config_path = (
        Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_10_defensive_target_demo.yaml"
    )
    config = SimulationConfig.from_yaml(config_path)
    session = SimulationSession.from_config(config)
    snap = session.reset()

    assert game_runner._game_ric_reference_object_id(config, "target") == "target_reference"
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

    session, _, snap0 = game_runner._start_game_attempt(
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
    # The v2 physical path maps the requested inertial vector through the
    # sampled body/actuator attitude before applying it.  A tiny orthogonal
    # component is therefore expected as attitude propagates between the
    # sensor sample and realization; the RIC direction and magnitude must
    # remain equivalent at game-control precision.
    applied = snap1.applied_thrust["chaser"]
    assert np.linalg.norm(applied) == pytest.approx(np.linalg.norm(expected), rel=1e-6)
    assert np.dot(applied, expected) / (np.linalg.norm(applied) * np.linalg.norm(expected)) > 0.9999999


def test_game_attempt_forces_acceleration_off(monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_00_tutorial.yaml"
    config = SimulationConfig.from_yaml(config_path).with_value("simulator.acceleration.mode", "auto")
    training_cfg = RPOTrainingConfig.from_metadata(dict(config.scenario.metadata or {}))
    monkeypatch.setenv(ACCELERATION_ENV, "auto")

    session, _, _ = game_runner._start_game_attempt(
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

    session, _, snapshot = game_runner._start_game_attempt(
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

    session, _, snapshot = game_runner._start_game_attempt(
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

    session, _, snapshot = game_runner._start_game_attempt(
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
    assert game_runner._game_retained_history_samples(config) == 8
    assert engine is not None
    assert engine.history_mode == "dynamic"
    assert engine.planned_samples > 8
    assert engine.allocated_history_samples == 8
    assert session._observer_samples.maxlen == 8
    assert session._scoring_events.maxlen == 8


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
    assert game_runner._game_camera_mode(config) == "target_pair"
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

    assert game_arcade._game_arcade_enabled(config) is True
    assert game_arcade._game_arcade_initial_time_s(config, training_cfg) == pytest.approx(12000.0)
    assert game_arcade._game_arcade_round_bonus_time_s(config) == pytest.approx(0.0)
    assert game_arcade._game_arcade_delta_v_bonus_time_per_m_s(config) == pytest.approx(1000.0)
    assert game_arcade._game_arcade_goal_range_step_km(config) == pytest.approx(0.005)
    assert game_arcade._game_arcade_min_goal_range_km(config) == pytest.approx(0.005)
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
    assert game_runner._game_control_mode(config) == "ric_translation"
    assert game_runner._game_camera_mode(config) == "target_pair"
    assert game_runner._game_target_centered_plot_planes(config) == ("RI", "RC", "IC")
    assert game_runner._game_plot_overlays_in_zoom(config) is False
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
    assert game_runner._game_control_mode(config) == "ric_translation"
    assert game_runner._game_coast_prediction_model(config) == "tschauner_hempel"
    assert game_runner._game_camera_mode(config) == "rule_toggle_pair"
    assert game_runner._game_camera_rule_mode(config) == "current_pair"
    assert game_runner._game_camera_rule_toggle_enabled(config) is True
    assert game_runner._game_target_centered_plot_planes(config) == ("RI", "RC")
    assert game_runner._game_plot_prediction_full_trajectory_only(config) is True
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
    dashboard.camera_mode = game_runner._game_camera_mode(config)
    dashboard.camera_rule_mode = game_runner._game_camera_rule_mode(config)
    dashboard.target_centered_plot_planes = game_runner._game_target_centered_plot_planes(config)
    dashboard.target_centered_plot_axes = game_runner._game_target_centered_plot_axes(config)
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
    assert game_runner._game_control_mode(config) == "ric_translation"
    assert game_runner._game_coast_prediction_model(config) == "tschauner_hempel"
    assert game_runner._game_target_centered_plot_planes(config) == ("RI",)
    assert game_runner._game_plot_fixed_axis_half_span_km(config) == {"RI": (3.25, 1.6)}
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
    assert game_runner._game_control_mode(config) == "ric_translation"
    assert game_runner._game_camera_mode(config) == "target_pair"
    assert game_runner._game_coast_prediction_model(config) == "tschauner_hempel"
    assert game_runner._game_target_centered_plot_planes(config) == ("RI",)
    assert game_runner._game_plot_prediction_in_zoom(config) is False
    assert game_runner._game_plot_prediction_zoom_max_span_km(config) is None
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

    assert game_runner._game_level_title(config) == "Level 4 - Rendezvous"
    assert PygameRPODashboard._top_bar_label("active", game_runner._game_level_title(config)) == "LEVEL 4 - RENDEZVOUS"
    assert PygameRPODashboard._top_bar_label("active", "") == "LEVEL ACTIVE"


def test_arcade_attempt_config_uses_current_training_clock_for_duration() -> None:
    config_path = (
        Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_arcade_pursuit.yaml"
    )
    config = SimulationConfig.from_yaml(config_path)
    training_cfg = RPOTrainingConfig.from_metadata(dict(config.scenario.metadata or {}))

    next_round = replace(training_cfg, max_time_s=8500.25)
    attempt_cfg = game_session._attempt_config_for_training_clock(config, next_round)

    assert config.scenario.simulator.duration_s == pytest.approx(12000.0)
    assert attempt_cfg.scenario.simulator.duration_s == pytest.approx(8501.0)


def test_pursuit_arcade_goal_range_tightens_each_round_to_floor() -> None:
    config_path = (
        Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_arcade_pursuit.yaml"
    )
    config = SimulationConfig.from_yaml(config_path)
    base_cfg = RPOTrainingConfig.from_metadata(dict(config.scenario.metadata or {}))

    round1 = game_arcade._arcade_round_training_config(config, base_cfg, round_index=1, max_time_s=12000.0)
    round2 = game_arcade._arcade_round_training_config(config, base_cfg, round_index=2, max_time_s=11900.0)
    round20 = game_arcade._arcade_round_training_config(config, base_cfg, round_index=20, max_time_s=8000.0)
    round99 = game_arcade._arcade_round_training_config(config, base_cfg, round_index=99, max_time_s=5000.0)

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

    round4 = game_arcade._arcade_round_simulation_config(
        config,
        base_cfg,
        round_index=4,
        rng=game_arcade._arcade_round_initial_state_rng(1234, 4),
    )
    round5 = game_arcade._arcade_round_simulation_config(
        config,
        base_cfg,
        round_index=5,
        rng=game_arcade._arcade_round_initial_state_rng(1234, 5),
    )
    round5_repeat = game_arcade._arcade_round_simulation_config(
        config,
        base_cfg,
        round_index=5,
        rng=game_arcade._arcade_round_initial_state_rng(1234, 5),
    )
    round10 = game_arcade._arcade_round_simulation_config(
        config,
        base_cfg,
        round_index=10,
        rng=game_arcade._arcade_round_initial_state_rng(1234, 10),
    )
    round25 = game_arcade._arcade_round_simulation_config(
        config,
        base_cfg,
        round_index=25,
        rng=game_arcade._arcade_round_initial_state_rng(1234, 25),
    )
    round30 = game_arcade._arcade_round_simulation_config(
        config,
        base_cfg,
        round_index=30,
        rng=game_arcade._arcade_round_initial_state_rng(1234, 30),
    )

    normal_coes = round4.scenario.objects["target"].initial_state["coes"]
    boss_coes = round5.scenario.objects["target"].initial_state["coes"]
    boss_repeat_coes = round5_repeat.scenario.objects["target"].initial_state["coes"]
    boss_10_coes = round10.scenario.objects["target"].initial_state["coes"]
    boss_25_coes = round25.scenario.objects["target"].initial_state["coes"]
    boss_30_coes = round30.scenario.objects["target"].initial_state["coes"]

    assert game_arcade._arcade_round_is_boss(config, 4) is False
    assert game_arcade._arcade_round_is_boss(config, 5) is True
    assert game_arcade._arcade_round_is_boss(config, 10) is True
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
    assert game_runner._game_coast_prediction_model(round5) == "tschauner_hempel"


def test_pursuit_arcade_boss_rounds_keep_energy_matched_random_start() -> None:
    config_path = (
        Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_arcade_pursuit.yaml"
    )
    config = SimulationConfig.from_yaml(config_path)
    base_cfg = RPOTrainingConfig.from_metadata(dict(config.scenario.metadata or {}))

    round5 = game_arcade._arcade_round_simulation_config(
        config,
        base_cfg,
        round_index=5,
        rng=game_arcade._arcade_round_initial_state_rng(1234, 5),
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

    assert game_arcade._arcade_round_music_track(config, 4) is None
    assert game_arcade._arcade_round_music_track(config, 5) == "28_high_shred_boss_riff.wav"
    assert game_arcade._arcade_round_coast_prediction_model(config, 5) == "tschauner_hempel"
    assert game_arcade._arcade_round_score_multiplier(config, 5) == pytest.approx(2.0)
    assert game_arcade._arcade_round_time_bonus_s(config, training_cfg, score, round_index=4) == pytest.approx(500.0)
    assert game_arcade._arcade_round_time_bonus_s(config, training_cfg, score, round_index=5) == pytest.approx(5500.0)
    assert game_arcade._arcade_round_weighted_score(
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


def test_rpo_duel_music_byte_matches_source_game_asset() -> None:
    root = Path(__file__).resolve().parents[2]
    filename = "39_perigee_afterburner_demo.wav"
    source_music = root / "sim" / "game" / "music" / filename
    duel_music = root / "web" / "rpo-duel-prototype" / "public" / "assets" / filename

    assert duel_music.read_bytes() == source_music.read_bytes()


def test_pursuit_arcade_keeps_round_one_initial_state() -> None:
    config_path = (
        Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_arcade_pursuit.yaml"
    )
    config = SimulationConfig.from_yaml(config_path)
    base_cfg = RPOTrainingConfig.from_metadata(dict(config.scenario.metadata or {}))
    base_state = config.scenario.objects["chaser"].initial_state["relative_to_target_ric"]["state"]

    round1 = game_arcade._arcade_round_simulation_config(
        config,
        base_cfg,
        round_index=1,
        rng=game_arcade._arcade_round_initial_state_rng(1234, 1),
    )

    round1_state = round1.scenario.objects["chaser"].initial_state["relative_to_target_ric"]["state"]
    assert round1_state == base_state


def test_pursuit_arcade_randomizes_round_two_initial_state_with_energy_match() -> None:
    config_path = (
        Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_arcade_pursuit.yaml"
    )
    config = SimulationConfig.from_yaml(config_path)
    base_cfg = RPOTrainingConfig.from_metadata(dict(config.scenario.metadata or {}))

    round2 = game_arcade._arcade_round_simulation_config(
        config,
        base_cfg,
        round_index=2,
        rng=game_arcade._arcade_round_initial_state_rng(1234, 2),
    )
    round2_repeat = game_arcade._arcade_round_simulation_config(
        config,
        base_cfg,
        round_index=2,
        rng=game_arcade._arcade_round_initial_state_rng(1234, 2),
    )
    round3 = game_arcade._arcade_round_simulation_config(
        config,
        base_cfg,
        round_index=3,
        rng=game_arcade._arcade_round_initial_state_rng(1234, 3),
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
    round2 = game_arcade._arcade_round_training_config(config, base_cfg, round_index=2, max_time_s=11900.0)
    round2 = replace(
        round2,
        hard_speed_limit_radius_km=0.025,
        hard_speed_limit_km_s=0.00005,
        max_target_reference_range_km=1.0,
    )
    dashboard = type("Dashboard", (), {"goal_range_km": base_cfg.goal_range_km, "_frame_cache_dirty": False})()

    game_runner._sync_dashboard_training_config(dashboard, round2)

    assert dashboard.goal_range_km == pytest.approx(0.095)
    assert dashboard.hard_speed_limit_radius_km == pytest.approx(0.025)
    assert dashboard.hard_speed_limit_km_s == pytest.approx(0.00005)
    assert dashboard.max_target_reference_range_km == pytest.approx(1.0)
    assert dashboard.target_reference_object_id == "target_reference"
    assert dashboard._frame_cache_dirty is True


def test_terminal_mission_state_keeps_game_loop_open_after_session_done() -> None:
    passed = type("Score", (), {"level_passed": True, "level_failed": False})()
    failed = type("Score", (), {"level_passed": False, "level_failed": True})()
    active = type("Score", (), {"level_passed": False, "level_failed": False})()

    assert game_runner._game_loop_should_exit(session_done=True, score=passed) is False
    assert game_runner._game_loop_should_exit(session_done=True, score=failed) is False
    assert game_runner._game_loop_should_exit(session_done=True, score=active) is True


def test_result_music_paths_follow_terminal_mission_state() -> None:
    passed = type("Score", (), {"level_passed": True, "level_failed": False})()
    failed = type("Score", (), {"level_passed": False, "level_failed": True})()
    active = type("Score", (), {"level_passed": False, "level_failed": False})()

    assert game_audio._result_music_path(passed) == MISSION_SUCCESS_MUSIC_PATH
    assert game_audio._result_music_path(failed) == MISSION_FAILURE_MUSIC_PATH
    assert game_audio._result_music_path(active) is None
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

    assert game_audio._level_music_path(tutorial) == LEVEL_MUSIC_PATHS["rpo_00_tutorial"]
    assert game_audio._level_music_path(tutorial).name == "10_training_grid_sunrise.wav"
    assert game_audio._level_music_path(level1) == LEVEL_MUSIC_PATHS["rpo_01_coast_relative_motion"]
    assert game_audio._level_music_path(level1).name == "07_starfield_attract_mode.wav"
    assert game_audio._level_music_path(level2) == LEVEL_MUSIC_PATHS["rpo_02_vbar_approach"]
    assert game_audio._level_music_path(level2).name == "02_rendezvous_vector.wav"
    assert game_audio._level_music_path(level3) == LEVEL_MUSIC_PATHS["rpo_03_rbar_approach"]
    assert game_audio._level_music_path(level3).name == "18_keepout_zone_accelerando.wav"
    assert game_audio._level_music_path(level4) == LEVEL_MUSIC_PATHS["rpo_04_rendezvous"]
    assert game_audio._level_music_path(level4).name == "06_casting_the_orbit_line.wav"
    assert game_audio._level_music_path(level5) == LEVEL_MUSIC_PATHS["rpo_05_passive_cross_track_approach"]
    assert game_audio._level_music_path(level5).name == "19_cross_track_ghost_orbit.wav"
    assert game_audio._level_music_path(level6) == LEVEL_MUSIC_PATHS["rpo_06_sun_angle_inspection"]
    assert game_audio._level_music_path(level6).name == "33_amber_terminator_demo.wav"
    assert game_audio._level_music_path(level7) == LEVEL_MUSIC_PATHS["rpo_07_elliptic_burn_then_approach"]
    assert game_audio._level_music_path(level7).name == "08_silent_running_radar.wav"
    assert game_audio._level_music_path(level8) == LEVEL_MUSIC_PATHS["rpo_08_elliptic_nmc"]
    assert game_audio._level_music_path(level8).name == "04_docking_bay_neon.wav"
    assert game_audio._level_music_path(level9) == LEVEL_MUSIC_PATHS["rpo_09_elliptic_rendezvous"]
    assert game_audio._level_music_path(level9).name == "23_elliptic_final_burn_cinematic.wav"
    assert game_audio._level_music_path(level10) == LEVEL_MUSIC_PATHS["rpo_10_defensive_target_demo"]
    assert game_audio._level_music_path(level10).name == "17_orbital_boss_metal.wav"
    assert game_audio._level_music_path(level11) == LEVEL_MUSIC_PATHS["rpo_11_evasive_target_survival"]
    assert game_audio._level_music_path(level11).name == "09_defender_boss_vector.wav"
    assert game_audio._level_music_path(cislunar) == LEVEL_MUSIC_PATHS["rpo_bonus_cislunar_rendezvous"]
    assert game_audio._level_music_path(cislunar).name == "30_far_side_navigation_demo.wav"
    assert game_audio._level_music_path(arcade) == LEVEL_MUSIC_PATHS["rpo_arcade_pursuit"]
    assert game_audio._level_music_path(arcade).name == "21_pursuit_arcade_overdrive_no_siren_demo.wav"
    assert game_audio._level_music_path(unmapped) is None
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

    assert game_audio._play_game_sound_effect(FakePygame(), ARCADE_ROUND_CLEAR_SOUND_PATH, volume=1.5) is True
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

    assert game_arcade._difficulty_score_multiplier("easy") == 1
    assert game_arcade._difficulty_score_multiplier("medium") == 2
    assert game_arcade._difficulty_score_multiplier("hard") == 3
    assert game_arcade._difficulty_score_multiplier("extreme") == 4
    assert game_arcade._difficulty_score_multiplier("expert") == 4
    assert game_arcade._arcade_score(cfg, score, difficulty="hard") == 5550
    assert game_arcade._arcade_round_weighted_score(cfg, score, difficulty="hard", round_index=3) == 16650
    assert game_arcade._score_time_used_s(score) == pytest.approx(20.0)


def test_arcade_round_time_bonus_adds_delta_v_remaining() -> None:
    config_path = (
        Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_arcade_pursuit.yaml"
    )
    config = SimulationConfig.from_yaml(config_path)
    training_cfg = RPOTrainingConfig(enabled=True, max_delta_v_m_s=3.0)
    score = type("Score", (), {"approximate_delta_v_m_s": 2.5})()

    assert game_arcade._arcade_round_time_bonus_s(config, training_cfg, score) == pytest.approx(500.0)
    assert game_arcade._arcade_round_time_bonus_s(config, training_cfg, score, round_index=5) == pytest.approx(5500.0)


def test_arcade_round_briefing_lines_show_transition_summary() -> None:
    lines = game_arcade._arcade_round_briefing_lines(
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

    assert game_arcade._arcade_score(cfg, score, difficulty="extreme") == 0


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

    metrics = game_runner._mission_metrics(cfg, score)

    assert "OK Chaser dV 6.766 m/s" in metrics
    assert "OK Target dV 877.0 mm/s" in metrics


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

    plan, errors = game_launcher._operator_plan_from_rows([["50", "2", "", "0.2"], ["", "", "", ""]], option=option)

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

    _plan, errors = game_launcher._operator_plan_from_rows([["50", "2", "", "0"]], option=option)

    assert errors == ("Plan total delta-v exceeds 1.0 m/s budget.",)


def test_operator_burn_rows_prefill_from_existing_plan() -> None:
    plan = OperatorBurnPlan(
        burns=(
            OperatorBurn(time_s=50.0, delta_v_ric_m_s=(2.0, 1.0, 0.2)),
            OperatorBurn(time_s=120.0, delta_v_ric_m_s=(-0.5, 0.0, 0.0)),
        )
    )

    assert game_launcher._operator_rows_from_plan(plan) == [["50", "2", "1", "0.2"], ["120", "-0.5", "0", "0"]]


def test_operator_tutorial_scripts_six_locked_burn_demos() -> None:
    config_path = (
        Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_00_tutorial.yaml"
    )
    training_cfg = RPOTrainingConfig.from_metadata(dict(SimulationConfig.from_yaml(config_path).scenario.metadata or {}))
    stages = game_runner._operator_tutorial_stages()

    assert game_runner._operator_tutorial_enabled("operator", training_cfg)
    assert not game_runner._operator_tutorial_enabled("pilot", training_cfg)
    assert [stage.display_label for stage in stages] == [
        "+I Burn",
        "-I Burn",
        "+R Burn",
        "-R Burn",
        "+C Burn",
        "-C Burn",
    ]
    assert game_runner._operator_tutorial_demo_title(OperatorTutorialRuntime(stage_index=2)) == "Demo 3/6: +R Burn"

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

    game_runner._clear_dashboard_tutorial_path(dashboard)

    assert dashboard.tutorial_target_path_ric.shape == (0, 6)
    assert dashboard._frame_cache_dirty is True


def test_operator_plan_layout_uses_game_plot_positions_and_scrollable_burn_table() -> None:
    left, right = game_launcher._operator_game_plot_panel_rects(1280, 720)
    table = game_launcher._operator_burn_table_rect(1280, 720)

    assert (left.x, left.y, left.width, left.height) == (36, 88, 586, 464)
    assert (right.x, right.y, right.width, right.height) == (658, 88, 586, 464)
    assert table[0] == left.x
    assert table[1] > left.bottom
    assert table[2] > 800
    assert game_launcher._operator_table_visible_rows(table) == 2
    assert game_launcher._operator_table_visible_rows(table) < 6
    assert game_launcher._operator_scroll_for_active_row(5, 0, row_count=8, table_rect=table) > 0


def test_operator_objectives_button_and_overlay_cover_plot_area() -> None:
    button = game_launcher._operator_objectives_button_rect(1280, 720)
    equation_button = game_launcher._operator_equation_sheet_button_rect(1280, 720)
    overlay = game_launcher._operator_objectives_overlay_rect(1280, 720)
    left, right = game_launcher._operator_game_plot_panel_rects(1280, 720)
    table = game_launcher._operator_burn_table_rect(1280, 720)

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
    plot_context = game_launcher._operator_plot_context(option.path)

    targets = game_launcher._operator_objective_numeric_targets(option, plot_context.training_config)

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
    plot_context = game_launcher._operator_plot_context(option.path)
    overlay = game_launcher._operator_objectives_overlay_rect(1280, 720)
    content_rect = game_launcher._operator_objectives_content_rect(overlay)
    content_height = game_launcher._operator_objectives_content_height(
        option,
        plot_context.training_config,
        font=_FixedWidthFont(),
        width_px=content_rect[2],
    )

    assert content_height > content_rect[3]
    assert game_launcher._clamp_operator_objectives_scroll_px(
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

    assert game_launcher._operator_delete_row_at_pos((905, 605), [first, second], table_rect=(36, 548, 912, 172)) == 0
    assert game_launcher._operator_delete_row_at_pos((905, 641), [first, second], table_rect=(36, 548, 912, 172)) == 1
    assert game_launcher._operator_delete_row_at_pos((905, 605), [first, second], table_rect=(36, 650, 912, 120)) is None


def test_operator_initial_relative_state_reads_level_yaml(tmp_path: Path) -> None:
    path = tmp_path / "game_training_rpo_01_demo.yaml"
    cfg = _game_config(tmp_path)
    cfg["metadata"]["game"]["training"] = {"scenario_id": "rpo_01_demo", "chaser_object_id": "chaser"}
    cfg["objects"]["chaser"]["initial_state"]["relative_to_target_ric"] = {
        "frame": "rect",
        "state": [-7.5, -15.0, 1.0, 0.0, 0.011, 0.0],
    }
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    assert game_launcher._operator_initial_relative_ric_state(path) == pytest.approx((-7.5, -15.0, 1.0))


def test_operator_plot_context_uses_initial_coast_not_goal_nmt_for_level_one() -> None:
    path = Path("sim/game/configs/game_training_rpo_01_coast_relative_motion.yaml")

    context = game_launcher._operator_plot_context(path)
    coast = np.array(context.initial_coast_ric_km_s, dtype=float).reshape(-1, 6)

    assert coast.shape[0] > 10
    assert coast[:, 0] == pytest.approx(0.0)
    assert coast[:, 1] == pytest.approx(-3.0)
    assert coast[:, 2] == pytest.approx(0.0)


def test_operator_planned_trajectory_applies_burns_and_extends_one_orbit() -> None:
    path = Path("sim/game/configs/game_training_rpo_01_coast_relative_motion.yaml")
    context = game_launcher._operator_plot_context(path)
    assert context.mean_motion_rad_s is not None
    plan = OperatorBurnPlan(
        burns=(
            OperatorBurn(time_s=0.0, delta_v_ric_m_s=(0.0, 0.1, 0.0)),
            OperatorBurn(time_s=2000.0, delta_v_ric_m_s=(0.0, -0.1, 0.0)),
        )
    )

    trajectory, markers = game_launcher._operator_planned_trajectory(context, plan)
    period_s = 2.0 * np.pi / float(context.mean_motion_rad_s)
    initial = np.asarray(context.initial_relative_ric_km_s, dtype=float)

    assert trajectory.shape[0] > 200
    assert markers.shape == (2, 6)
    assert trajectory[0] == pytest.approx(initial)
    assert markers[0, 3:6] == pytest.approx(initial[3:6] + np.array([0.0, 0.0001, 0.0]))
    expected_second_pre_burn = game_pygame_dashboard._cw_coast_states(markers[0], np.array([2000.0]), float(context.mean_motion_rad_s))[0]
    assert markers[1, :3] == pytest.approx(expected_second_pre_burn[:3], abs=1.0e-9)
    assert markers[1, 3:6] == pytest.approx(expected_second_pre_burn[3:6] + np.array([0.0, -0.0001, 0.0]))
    expected_final = game_pygame_dashboard._cw_coast_states(
        markers[-1],
        np.array([period_s]),
        float(context.mean_motion_rad_s),
    )[0]
    assert trajectory[-1] == pytest.approx(expected_final, abs=1.0e-9)


def test_operator_cislunar_preview_uses_configured_cr3bp_projection() -> None:
    path = Path("sim/game/configs/game_training_rpo_bonus_cislunar_rendezvous.yaml")
    context = game_launcher._operator_plot_context(path)
    plan = OperatorBurnPlan(burns=(OperatorBurn(time_s=0.0, delta_v_ric_m_s=(0.0, 0.0, 0.0)),))

    trajectory, _ = game_launcher._operator_planned_trajectory(context, plan)
    times = game_launcher._operator_planned_trajectory_times(context, plan)
    initial = np.asarray(context.initial_relative_ric_km_s, dtype=float)
    target = np.asarray(context.reference_state_eci_km_s, dtype=float)
    expected = game_pygame_dashboard._linearized_cr3bp_moon_ric_coast_prediction(
        initial,
        target_state=target,
        times=times,
        current_t_s=0.0,
    )

    assert context.coast_prediction_model == "cr3bp"
    assert context.cr3bp_projection_mode == "linearized"
    assert times[-1] == pytest.approx(21600.0)
    assert trajectory == pytest.approx(expected, abs=1.0e-9)


def test_operator_sandbox_preview_uses_preflight_setup_state() -> None:
    path = Path("sim/game/configs/game_training_rpo_sandbox.yaml")
    config = SimulationConfig.from_yaml(path)
    setup = SandboxSetupValues(
        target_a_km=7600.0,
        target_ecc=0.08,
        target_inc_deg=32.0,
        target_raan_deg=18.0,
        target_argp_deg=27.0,
        target_true_anomaly_deg=41.0,
        radial_km=1.25,
        in_track_km=-7.5,
        cross_track_km=0.4,
        radial_rate_m_s=0.2,
        in_track_rate_m_s=-0.3,
        cross_track_rate_m_s=0.1,
    )
    updated = game_runner._apply_sandbox_setup_to_config(config, setup)

    context = game_launcher._operator_plot_context(path, config_override=updated)
    target_coes = updated.scenario.objects["target"].initial_state["coes"]
    expected_target = np.hstack(coes_mapping_to_rv_eci(target_coes))

    assert context.initial_relative_ric_km_s == pytest.approx(setup.relative_ric_state_km_s)
    assert context.mean_motion_rad_s == pytest.approx(np.sqrt(EARTH_MU_KM3_S2 / setup.target_a_km**3))
    assert context.coast_prediction_model == "tschauner_hempel"
    assert context.reference_state_eci_km_s == pytest.approx(expected_target)


def test_operator_burn_velocity_vector_uses_post_burn_velocity_direction() -> None:
    path = Path("sim/game/configs/game_training_rpo_01_coast_relative_motion.yaml")
    context = game_launcher._operator_plot_context(path)
    plan = OperatorBurnPlan(burns=(OperatorBurn(time_s=0.0, delta_v_ric_m_s=(0.0, 0.1, 0.0)),))

    _trajectory, markers = game_launcher._operator_planned_trajectory(context, plan)
    endpoint = game_launcher._operator_velocity_vector_endpoint(
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
    context = game_launcher._operator_plot_context(path)
    plan = OperatorBurnPlan(burns=(OperatorBurn(time_s=0.0, delta_v_ric_m_s=(0.0, 0.1, 0.0)),))
    trajectory, _markers = game_launcher._operator_planned_trajectory(context, plan)
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
    click_pos = game_launcher._operator_plot_transform_to_px(transform, target_state[:3], x_axis=1, y_axis=0)
    assert click_pos is not None

    handled, selected_state, selected_time_s = game_launcher._operator_trajectory_probe_from_click(context, plan, click_pos)

    assert handled is True
    assert selected_state is not None
    selected_px = game_launcher._operator_plot_transform_to_px(transform, selected_state[:3], x_axis=1, y_axis=0)
    assert selected_px is not None
    assert np.linalg.norm(np.array(selected_px, dtype=float) - np.array(click_pos, dtype=float)) <= 10.0
    assert any(np.allclose(selected_state, row) for row in trajectory)
    assert selected_time_s is not None
    assert game_launcher._operator_probe_time_label(selected_time_s).startswith("T=")
    assert game_launcher._operator_probe_time_label(selected_time_s).endswith("s")

    selected_probe = OperatorTrajectoryProbe(
        state_ric_km_s=tuple(float(value) for value in selected_state),
        time_s=float(selected_time_s),
        plan_key=("test",),
    )
    handled, selected_state, selected_time_s = game_launcher._operator_trajectory_probe_from_click(
        context,
        plan,
        click_pos,
        selected_probe=selected_probe,
    )

    assert handled is True
    assert selected_state is None
    assert selected_time_s is None


def test_operator_add_burn_row_uses_selected_probe_time() -> None:
    probe = OperatorTrajectoryProbe(
        state_ric_km_s=(0.0, 1.0, 0.0, 0.0, 0.0001, 0.0),
        time_s=1234.5,
        plan_key=("test",),
    )

    assert game_launcher._operator_new_burn_row_from_probe(probe) == ["1234.5", "", "", ""]
    assert game_launcher._operator_new_burn_row_from_probe(None) == ["", "", "", ""]


def test_operator_planned_trajectory_uses_ya_for_elliptic_levels(monkeypatch: pytest.MonkeyPatch) -> None:
    path = Path("sim/game/configs/game_training_rpo_07_elliptic_burn_then_approach.yaml")
    context = game_launcher._operator_plot_context(path)
    assert game_pygame_dashboard._coast_prediction_model_key(context.coast_prediction_model) == "tschauner_hempel"
    assert context.reference_state_eci_km_s is not None
    calls: list[float] = []
    original = game_launcher.ya_closed_form_transition_matrix

    def spy_ya(dt_s: float, chief_start_eci_km_s: np.ndarray, chief_end_eci_km_s: np.ndarray, **kwargs: Any) -> Any:
        calls.append(float(dt_s))
        return original(dt_s, chief_start_eci_km_s, chief_end_eci_km_s, **kwargs)

    monkeypatch.setattr(game_launcher, "ya_closed_form_transition_matrix", spy_ya)
    plan = OperatorBurnPlan(burns=(OperatorBurn(time_s=0.0, delta_v_ric_m_s=(0.0, 0.1, 0.0)),))

    trajectory, markers = game_launcher._operator_planned_trajectory(context, plan)

    assert calls
    assert max(calls) > 1000.0
    assert trajectory.shape[0] > 100
    assert markers.shape == (1, 6)


def test_operator_vbar_script_preview_uses_live_level_camera_and_forbidden_regions() -> None:
    path = Path("sim/game/configs/game_training_rpo_02_vbar_approach.yaml")
    context = game_launcher._operator_plot_context(path)
    training_cfg = context.training_config
    assert training_cfg is not None
    rel = np.array(context.initial_relative_ric_km_s[:3], dtype=float)
    target = np.zeros(3, dtype=float)

    ri_center = game_launcher._operator_camera_center_ric(context, chaser_current=rel, target_current=target, x_axis=1, y_axis=0)
    rc_center = game_launcher._operator_camera_center_ric(context, chaser_current=rel, target_current=target, x_axis=2, y_axis=0)
    ri_fr = game_launcher._operator_forbidden_region_projection_points(training_cfg, x_axis=1, y_axis=0, offset=target)
    rc_fr = game_launcher._operator_forbidden_region_projection_points(training_cfg, x_axis=2, y_axis=0, offset=target)

    assert context.camera_mode == "target_pair"
    assert context.target_centered_plot_axes == {"RI": ("y",)}
    assert context.plot_overlays_in_zoom is False
    assert context.plot_overlays_in_zoom_by_plane == {"RC": True}
    assert context.proximity_ring_plot_planes == ("RI",)
    assert ri_center == pytest.approx(np.array([0.0, -2.5, 0.0]))
    assert rc_center == pytest.approx(np.zeros(3, dtype=float))
    assert len(ri_fr) == 3
    assert len(rc_fr) == 2
    assert game_launcher._operator_minimum_plot_span_km(
        context,
        x_axis=1,
        y_axis=0,
        target_current=target,
        nmt=np.empty((0, 3), dtype=float),
        nmt_bounds=(),
    ) == pytest.approx(MIN_PLOT_SPAN_KM)
    assert game_launcher._operator_minimum_plot_span_km(
        context,
        x_axis=2,
        y_axis=0,
        target_current=target,
        nmt=np.empty((0, 3), dtype=float),
        nmt_bounds=(),
    ) == pytest.approx(5.8 * PLOT_OVERLAY_MARGIN)


def test_operator_rbar_script_preview_stays_target_centered_without_fixed_spans() -> None:
    path = Path("sim/game/configs/game_training_rpo_03_rbar_approach.yaml")
    context = game_launcher._operator_plot_context(path)
    training_cfg = context.training_config
    assert training_cfg is not None
    rel = np.array(context.initial_relative_ric_km_s[:3], dtype=float)
    target = np.zeros(3, dtype=float)

    ri_center = game_launcher._operator_camera_center_ric(context, chaser_current=rel, target_current=target, x_axis=1, y_axis=0)
    rc_center = game_launcher._operator_camera_center_ric(context, chaser_current=rel, target_current=target, x_axis=2, y_axis=0)

    assert context.camera_mode == "target_pair"
    assert context.target_centered_plot_planes == ("RI", "RC")
    assert context.plot_overlays_in_zoom is False
    assert context.plot_fixed_axis_half_span_km == {}
    assert ri_center == pytest.approx(np.zeros(3, dtype=float))
    assert rc_center == pytest.approx(np.zeros(3, dtype=float))
    assert len(game_launcher._operator_forbidden_region_projection_points(training_cfg, x_axis=1, y_axis=0, offset=target)) == 1
    assert len(game_launcher._operator_forbidden_region_projection_points(training_cfg, x_axis=2, y_axis=0, offset=target)) == 1


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

    duration_s = game_runner._trigger_operator_projection_transition(dashboard, provider)

    assert len(dashboard.transitions) == 1
    pre_burn, post_burn, transition_duration_s = dashboard.transitions[0]
    assert np.allclose(post_burn, dashboard.rel_hist[-1])
    assert np.allclose(pre_burn[:3], post_burn[:3])
    assert np.allclose(pre_burn[3:6], post_burn[3:6] - np.array([0.002, -0.001, 0.0005]))
    assert duration_s == pytest.approx(game_runner._operator_burn_visual_duration_s(np.linalg.norm([2.0, -1.0, 0.5])))
    assert transition_duration_s == pytest.approx(duration_s)
def test_defensive_levels_use_onboard_rpo_stack_without_external_provider() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "game"
        / "configs"
        / "game_training_rpo_10_defensive_target_demo.yaml"
    )
    config = SimulationConfig.from_yaml(config_path)
    training_cfg = RPOTrainingConfig.from_metadata(dict(config.scenario.metadata or {}))
    first, _, _ = game_runner._start_game_attempt(
        config,
        command_state=KeyboardCommandState(),
        training_cfg=training_cfg,
        controlled_object_id="chaser",
        attitude_rate_deg_s=45.0,
        control_mode="ric_translation",
        ric_reference_object_id="target_reference",
    )
    second, _, _ = game_runner._start_game_attempt(
        config,
        command_state=KeyboardCommandState(),
        training_cfg=training_cfg,
        controlled_object_id="chaser",
        attitude_rate_deg_s=45.0,
        control_mode="ric_translation",
        ric_reference_object_id="target_reference",
    )

    first_runtime = first._engine.agents["target"].flight_software_runtime
    second_runtime = second._engine.agents["target"].flight_software_runtime
    assert first_runtime.stack.identity.stack_id == "fsw.rpo_reference"
    assert second_runtime.stack.identity.stack_id == "fsw.rpo_reference"
    assert first_runtime is not second_runtime


def test_arcade_defensive_profile_is_seeded_presentation_configuration() -> None:
    config = SimulationConfig.from_yaml(
        Path(__file__).resolve().parents[1] / "game/configs/game_training_rpo_arcade_pursuit.yaml"
    )
    first = game_arcade._game_random_direction_defensive_target_profile(config, rng=np.random.default_rng(1))
    repeated = game_arcade._game_random_direction_defensive_target_profile(config, rng=np.random.default_rng(1))
    different = game_arcade._game_random_direction_defensive_target_profile(config, rng=np.random.default_rng(2))

    assert first == repeated
    assert first["fixed_direction_ric"] != different["fixed_direction_ric"]
    assert np.linalg.norm(first["fixed_direction_ric"]) == pytest.approx(1.0)


def test_evasion_level_runs_player_and_ai_through_complete_stacks() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "game/configs/game_training_rpo_11_evasive_target_survival.yaml"
    )
    config = SimulationConfig.from_yaml(config_path)
    training_cfg = RPOTrainingConfig.from_metadata(dict(config.scenario.metadata or {}))
    session, _, _ = game_runner._start_game_attempt(
        config,
        command_state=KeyboardCommandState(yaw=1.0),
        training_cfg=training_cfg,
        controlled_object_id="target",
        attitude_rate_deg_s=45.0,
        control_mode="ric_translation",
        ric_reference_object_id="target_reference",
    )
    session.step()

    chaser_runtime = session._engine.agents["chaser"].flight_software_runtime
    target_runtime = session._engine.agents["target"].flight_software_runtime
    assert chaser_runtime.stack.identity.stack_id == "fsw.rpo_reference"
    assert target_runtime.stack.identity.stack_id == "fsw.game_pilot_reference"
    assert chaser_runtime.evidence.invocations
    assert target_runtime.evidence.invocations
