from __future__ import annotations

# These owner-aligned tests share deterministic builders and compatibility
# imports from the adjacent support module.
# ruff: noqa: F403, F405
from dataclasses import asdict

from sim.tests.game_mode_test_support import *


def test_training_tracker_records_cislunar_mean_motion_after_decomposition() -> None:
    tracker = RPOTrainingTracker(
        RPOTrainingConfig(
            enabled=True,
            relative_frame="cislunar",
            target_object_id="target",
            chaser_object_id="chaser",
        )
    )
    target = np.zeros(14, dtype=float)
    chaser = np.zeros(14, dtype=float)
    chaser[0] = 1.0

    tracker.record(
        snapshot=type(
            "Snapshot",
            (),
            {
                "time_s": 0.0,
                "truth": {"target": target, "chaser": chaser},
                "applied_thrust": {},
            },
        )()
    )

    assert tracker.mean_motion_hist == [EARTH_MOON_MEAN_MOTION_RAD_S]


def test_game_scoring_evidence_preserves_exact_score_mapping() -> None:
    score = RPOTrainingTracker(RPOTrainingConfig(enabled=True)).score()
    session = object.__new__(GamePhysicsSession)
    session._scoring_policy = "configured_training.v1"
    session._controlled_object_id = "chaser"
    session._observer_samples = [{"time_ns": 1_250_000_000}]
    session._scoring_events = []

    session.record_scoring(score)

    assert session._scoring_events == [
        {
            "object_id": "chaser",
            "time_ns": 1_250_000_000,
            "scoring_policy": "configured_training.v1",
            "event_type": "failed",
            "detail": asdict(score),
        }
    ]


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

    assert game_runner._timed_maneuver_pending_sim_s(state, control_mode="ric_translation") == pytest.approx(0.04)


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
    metrics = game_runner._mission_metrics(cfg, score)
    checklist = game_runner._mission_checklist(cfg, score)

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
    metrics = game_runner._mission_metrics(cfg, score)
    checklist = game_runner._mission_checklist(cfg, score)

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

    assert game_runner._ric_primer_enabled(tutorial) is True
    assert game_runner._ric_primer_enabled(replace(tutorial, scenario_id="rpo_01_coast_relative_motion")) is False
    assert game_runner._ric_primer_enabled(replace(tutorial, sandbox_mode=True)) is False
    assert game_runner._ric_primer_enabled(tutorial, arcade_enabled=True) is False
    assert game_pygame_dashboard._ric_primer_stage(0)["id"] == "radial"
    assert game_pygame_dashboard._ric_primer_stage(1)["id"] == "in_track"
    assert game_pygame_dashboard._ric_primer_stage(2)["id"] == "cross_track"


def test_guided_tutorial_input_matches_only_requested_axis() -> None:
    stage = GuidedTutorialBurnConfig(name="plus_i", axis="in_track", sign=1, delta_v_m_s=0.25)
    state = KeyboardCommandState(yaw=1.0)

    assert game_runner._guided_tutorial_input_matches(state, stage) is True
    assert game_runner._guided_tutorial_wrong_input_active(state, stage) is False

    state.pitch = 1.0
    assert game_runner._guided_tutorial_input_matches(state, stage) is False
    assert game_runner._guided_tutorial_wrong_input_active(state, stage) is True

    state.pitch = 0.0
    state.yaw = -1.0
    assert game_runner._guided_tutorial_input_matches(state, stage) is False
    assert game_runner._guided_tutorial_wrong_input_active(state, stage) is True

    state.yaw = 0.0
    assert game_runner._guided_tutorial_wrong_input_active(state, stage) is False


def test_guided_tutorial_wrong_key_hint_names_expected_control() -> None:
    stage = GuidedTutorialBurnConfig(name="plus_i", axis="in_track", sign=1, delta_v_m_s=0.25)
    runtime = game_runner.GuidedTutorialRuntime(wrong_key_active=True)

    assert game_runner._guided_tutorial_expected_key(stage) == "D"
    assert game_runner._guided_tutorial_stage_hint(stage, runtime) == "Wrong key - hold D for +I burn."

    space_force = frame_convention_from_preset(FRAME_CONVENTION_PRESET_SPACE_FORCE)
    assert game_runner._guided_tutorial_expected_key(stage, frame_convention=space_force) == "A"
    assert (
        game_runner._guided_tutorial_stage_hint(stage, runtime, frame_convention=space_force)
        == "Wrong key - hold A for +I burn."
    )


def test_guided_tutorial_stage_hint_swaps_in_track_key_for_space_force() -> None:
    stage = GuidedTutorialBurnConfig(
        name="plus_i",
        axis="in_track",
        sign=1,
        delta_v_m_s=0.25,
        hint="Hold D for +I, then coast.",
    )
    runtime = game_runner.GuidedTutorialRuntime()
    space_force = frame_convention_from_preset(FRAME_CONVENTION_PRESET_SPACE_FORCE)

    assert game_runner._guided_tutorial_stage_hint(stage, runtime).startswith("Hold D for +I")
    assert game_runner._guided_tutorial_stage_hint(stage, runtime, frame_convention=space_force).startswith(
        "Hold A for +I"
    )


def test_guided_tutorial_target_path_applies_requested_burn() -> None:
    rel0 = np.array([0.0, -0.8, 0.0, 0.0, 0.0, 0.0], dtype=float)
    stage = GuidedTutorialBurnConfig(name="minus_c", axis="cross_track", sign=-1, delta_v_m_s=0.25)

    path = game_runner._guided_tutorial_target_path(rel0, 0.001, stage, samples=5)

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

    game_runner._guided_tutorial_update_dashboard_path(dashboard, tracker, cfg, runtime)

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

    game_runner._sync_guided_tutorial_path_for_mode(
        dashboard,
        tracker,
        cfg,
        runtime,
        game_mode="operator",
    )

    assert np.array_equal(dashboard.tutorial_target_path_ric, initial_path)
    assert dashboard._frame_cache_dirty is False

    game_runner._sync_guided_tutorial_path_for_mode(
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

    assert game_runner._guided_tutorial_speed_step_follows_burn(cfg, burn) is True
    assert game_runner._guided_tutorial_speed_step_reached(cfg, 5.0) is False
    assert game_runner._guided_tutorial_speed_step_reached(cfg, 10.0) is True
    assert "Current speed: 5x" in game_runner._guided_tutorial_speed_step_hint(cfg, 5.0)


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

    game_runner._reset_guided_tutorial_stage_attempt(
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

    assert game_runner._guided_tutorial_delta_v_m_s(tracker, cfg.guided_tutorial_burns[0]) == pytest.approx(0.2)


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
        session = SimulationSession.from_config(game_session._attempt_config_for_training_clock(sim_cfg, training_cfg))
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
    metrics = game_runner._mission_metrics(cfg, score)
    checklist = game_runner._mission_checklist(cfg, score)

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
    metrics = game_runner._mission_metrics(cfg, score)
    checklist = game_runner._mission_checklist(cfg, score)

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
    metrics = game_runner._mission_metrics(cfg, score)

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
    metrics = game_runner._mission_metrics(cfg, high_c_amp)

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
    assert game_runner._game_camera_mode(rbar_sim_cfg) == "target_pair"
    assert game_runner._game_target_centered_plot_planes(rbar_sim_cfg) == ("RI", "RC")
    assert game_runner._game_plot_overlays_in_zoom(rbar_sim_cfg) is False
    assert game_runner._game_plot_fixed_axis_half_span_km(rbar_sim_cfg) == {}


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
    speed_options = game_runner._game_speed_multiplier_options(sim_cfg)
    burn_state = KeyboardCommandState(pitch=1.0)
    burn_speed = game_runner._effective_speed_multiple_for_control(
        sim_cfg,
        1000.0,
        burn_state,
        control_mode=game_runner._game_control_mode(sim_cfg),
        options=speed_options,
    )

    assert game_runner._game_initial_speed_multiple(sim_cfg, None) == pytest.approx(10.0)
    expected_speed_options = (
        (1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0)
        if config_filename == "game_training_rpo_11_evasive_target_survival.yaml"
        else (1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 500.0, 1000.0)
    )
    assert speed_options == pytest.approx(expected_speed_options)
    assert game_runner._game_maneuver_control_speed_multiple(sim_cfg) is None
    assert game_runner._game_speed_dt_schedule(sim_cfg) == SPEED_DT_SCHEDULE
    assert game_runner._game_coast_speed_dt_schedule(sim_cfg) == SPEED_DT_SCHEDULE
    assert game_runner._game_two_rail_speed_control_enabled(sim_cfg) is True
    assert game_runner._game_tick_dt_s(sim_cfg, 10.0) == pytest.approx(0.5)
    assert game_runner._game_tick_dt_s(sim_cfg, 5.0) == pytest.approx(0.1)
    assert game_runner._game_tick_dt_s(sim_cfg, 1000.0) == pytest.approx(1.0)
    assert game_runner._game_coast_tick_dt_s(sim_cfg, 1000.0) == pytest.approx(10.0)
    assert burn_speed == pytest.approx(10.0)
    assert game_runner._game_active_tick_dt_s(sim_cfg, burn_speed, maneuver_active=True) == pytest.approx(0.5)


def test_two_rail_release_clears_pending_burn_before_coast() -> None:
    root = Path(__file__).resolve().parents[1] / "game" / "configs"
    sim_cfg = SimulationConfig.from_yaml(root / "game_training_rpo_06_sun_angle_inspection.yaml")
    state = KeyboardCommandState(pitch=1.0, use_timing_accumulator=True)
    state.accumulate_timed_input(0.04, speed_multiple=10.0, control_mode="ric_translation")

    assert state.pitch_sim_s > 0.0
    assert game_runner._effective_speed_multiple_for_control(
        sim_cfg,
        1000.0,
        state,
        control_mode="ric_translation",
        options=game_runner._game_speed_multiplier_options(sim_cfg),
    ) == pytest.approx(10.0)

    state.pitch = 0.0

    assert game_runner._clear_two_rail_released_maneuver_input(sim_cfg, state, control_mode="ric_translation") is True
    assert state.pitch_sim_s == pytest.approx(0.0)
    assert state.yaw_sim_s == pytest.approx(0.0)
    assert state.roll_sim_s == pytest.approx(0.0)
    assert game_runner._effective_speed_multiple_for_control(
        sim_cfg,
        1000.0,
        state,
        control_mode="ric_translation",
        options=game_runner._game_speed_multiplier_options(sim_cfg),
    ) == pytest.approx(1000.0)
    np.testing.assert_allclose(state.consume_ric_duty_cycle(10.0), np.zeros(3, dtype=float))


def test_two_rail_release_clear_is_opt_in() -> None:
    cfg = SimulationConfig.from_dict(_game_config(Path("/tmp")))
    state = KeyboardCommandState(use_timing_accumulator=True)
    state.pitch_sim_s = 0.25

    assert game_runner._clear_two_rail_released_maneuver_input(cfg, state, control_mode="ric_translation") is False
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
    assert game_runner._max_accel_from_config(sim_cfg, "chaser") == pytest.approx(1.0e-5)
    assert game_runner._game_initial_speed_multiple(sim_cfg, None) == pytest.approx(10.0)
    assert game_runner._game_speed_multiplier_options(sim_cfg) == pytest.approx(
        (1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 500.0, 1000.0)
    )
    assert game_runner._game_maneuver_control_speed_multiple(sim_cfg) is None
    assert game_runner._game_speed_dt_schedule(sim_cfg) == SPEED_DT_SCHEDULE
    assert game_runner._game_coast_speed_dt_schedule(sim_cfg) == SPEED_DT_SCHEDULE
    assert game_runner._game_two_rail_speed_control_enabled(sim_cfg) is True
    assert game_runner._game_tick_dt_s(sim_cfg, 10.0) == pytest.approx(0.5)
    assert game_runner._game_tick_dt_s(sim_cfg, 1000.0) == pytest.approx(1.0)
    assert game_runner._game_coast_tick_dt_s(sim_cfg, 1000.0) == pytest.approx(10.0)
    burn_state = KeyboardCommandState(pitch=1.0)
    assert game_runner._effective_speed_multiple_for_control(
        sim_cfg,
        1000.0,
        burn_state,
        control_mode="ric_translation",
        options=game_runner._game_speed_multiplier_options(sim_cfg),
    ) == pytest.approx(10.0)
    assert game_runner._game_active_tick_dt_s(
        sim_cfg,
        game_runner._effective_speed_multiple_for_control(
            sim_cfg,
            1000.0,
            burn_state,
            control_mode="ric_translation",
            options=game_runner._game_speed_multiplier_options(sim_cfg),
        ),
        maneuver_active=True,
    ) == pytest.approx(0.5)
    assert game_runner._game_dashboard_fps_cap(sim_cfg) is None
    assert game_runner._dashboard_fps_for_speed(10.0, fps_cap=game_runner._game_dashboard_fps_cap(sim_cfg)) == pytest.approx(60.0)
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
        forbidden_regions=(),
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
    metrics = game_runner._mission_metrics(cfg, score)

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
    tracker = RPOTrainingTracker(replace(cfg, forbidden_regions=()))
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

    metrics = game_runner._mission_metrics(rbar, score)

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
    metrics = game_runner._mission_metrics(cfg, score)

    assert score.level_failed is True
    assert score.forbidden_region_violation is True
    assert score.forbidden_region_names == ("off-axis test region",)
    assert any("Forbidden region" in reason for reason in score.pass_fail_reasons)
    assert "FAIL FR Violated" in metrics


def test_training_tracker_detects_forbidden_region_crossing_between_samples() -> None:
    region = ForbiddenRegionConfig(
        name="no-fly sphere",
        kind="sphere",
        center_ric_km=np.zeros(3, dtype=float),
        radius_km=1.0,
        min_ric_km=np.full(3, -np.inf),
        max_ric_km=np.full(3, np.inf),
    )
    cfg = RPOTrainingConfig(
        enabled=True,
        scenario_id="unit-forbidden-crossing",
        goal_range_km=10.0,
        forbidden_regions=(region,),
    )
    tracker = RPOTrainingTracker(cfg)
    tracker.t_s = [0.0, 10.0]
    tracker.rel_ric_hist = [
        np.array([-2.0, 0.0, 0.0, 0.4, 0.0, 0.0], dtype=float),
        np.array([2.0, 0.0, 0.0, 0.4, 0.0, 0.0], dtype=float),
    ]
    tracker.thrust_hist = [np.zeros(3, dtype=float), np.zeros(3, dtype=float)]
    tracker.target_thrust_hist = [np.zeros(3, dtype=float), np.zeros(3, dtype=float)]
    tracker.mean_motion_hist = [0.001, 0.001]

    score = tracker.score()

    assert not np.any(region.contains_positions(np.array([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]])))
    assert region.intersects_segment(np.array([-2.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0]))
    assert score.forbidden_region_violation is True
    assert score.level_failed is True


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
    metrics = game_runner._mission_metrics(cfg, score)

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
    metrics = game_runner._mission_metrics(cfg, score)

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
    metrics = game_runner._mission_metrics(cfg, score)

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
    debrief = game_runner._score_debrief_lines(score, config=cfg, difficulty="medium")

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

    debrief = game_runner._score_debrief_lines(score, config=cfg, difficulty="medium")

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
    assert game_runner._mission_metrics(cfg, score)


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

    game_runner._poll_pygame_input(FakePygame, state, control_mode="ric_translation")

    assert state.pitch == 1.0
    assert state.yaw == 1.0
    assert state.roll == 1.0
    assert state.firing is False

    class QuitPygame(FakePygame):
        class event:
            @staticmethod
            def get():
                return [FakeEvent(FakePygame.KEYDOWN, FakePygame.K_ESCAPE)]

    game_runner._poll_pygame_input(QuitPygame, state, control_mode="ric_translation")

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

    game_runner._poll_pygame_input(PauseSpeedPygame, state, control_mode="ric_translation")

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

    game_runner._poll_pygame_input(SlowDownPygame, state, control_mode="ric_translation")

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

    game_runner._poll_pygame_input(PlusMinusPygame, state, control_mode="ric_translation")

    assert state.speed_multiplier_change == 1

    class ReleasedPlusPygame(FakePygame):
        class event:
            @staticmethod
            def get():
                return [
                    FakeEvent(FakePygame.KEYUP, FakePygame.K_EQUALS, unicode="+"),
                    FakeEvent(FakePygame.KEYDOWN, FakePygame.K_KP_PLUS),
                ]

    game_runner._poll_pygame_input(ReleasedPlusPygame, state, control_mode="ric_translation")

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

    game_runner._poll_pygame_input(RepeatedMinusPygame, state, control_mode="ric_translation")

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

    game_runner._poll_pygame_input(BurstPlusPygame, state, control_mode="ric_translation")

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

    game_runner._poll_pygame_input(QuickBurnTapPygame, state, control_mode="ric_translation")

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

    game_runner._poll_pygame_input(BriefingScrollPygame, state, control_mode="ric_translation", briefing_open=True)

    assert state.briefing_scroll_px == 288
    assert state.clip_record_save_requested is False

    terminal_state = KeyboardCommandState()

    game_runner._poll_pygame_input(BriefingScrollPygame, terminal_state, control_mode="ric_translation", terminal_open=True)

    assert terminal_state.briefing_scroll_px == 288
    assert terminal_state.clip_record_save_requested is False
