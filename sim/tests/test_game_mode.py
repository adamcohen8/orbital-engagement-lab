from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from sim.api import SimulationConfig, SimulationSession
from sim.core.models import StateBelief, StateTruth
from sim.game.defensive_target import DefensiveTargetIntentProvider
from sim.game.launcher import discover_game_scenarios
from sim.game.manual import KeyboardCommandState, ManualGameCommandProvider
from sim.game.pygame_dashboard import PygameRPODashboard, _cw_coast_state
from sim.game.runner import (
    _adjust_speed_multiple,
    _coast_prediction_orbit_fraction,
    _coerce_speed_multiple,
    _game_loop_should_exit,
    _game_ric_reference_object_id,
    _mission_metrics,
    _poll_pygame_input,
    _score_debrief_lines,
    _start_game_attempt,
    _wall_step_s,
)
from sim.game.training import (
    RPOTrainingConfig,
    RPOTrainingTracker,
    nmt_curve_points_km,
    nmt_element_errors,
    nmt_position_error_km,
    nmt_velocity_error_km_s,
)
from sim.utils.frames import ric_rect_state_to_eci


def _knowledge_from_state6(state6: np.ndarray) -> StateBelief:
    return StateBelief(state=np.array(state6, dtype=float).reshape(6), covariance=np.eye(6), last_update_t_s=0.0)


def _game_config(tmp_path: Path) -> dict:
    with (Path(__file__).resolve().parents[2] / "sim" / "game" / "configs" / "game_mode_basic.yaml").open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = deepcopy(cfg)
    cfg["simulator"]["duration_s"] = 1.0
    cfg["outputs"]["output_dir"] = str(tmp_path)
    cfg["outputs"]["stats"]["print_summary"] = False
    cfg["outputs"]["stats"]["save_json"] = False
    cfg["outputs"]["stats"]["save_full_log"] = False
    return cfg


def test_game_launcher_discovers_ordered_training_levels() -> None:
    options = discover_game_scenarios(Path(__file__).resolve().parents[1] / "game" / "configs")

    assert [option.scenario_id for option in options] == [
        "rpo_01_coast_relative_motion",
        "rpo_02_vbar_approach",
        "rpo_03_rbar_approach",
        "rpo_04_rendezvous",
        "rpo_05_keepout_recovery",
        "rpo_06_defensive_target_demo",
    ]
    assert options[0].title == "Level 1 - Coast Relative Motion"
    assert options[0].player_brief
    assert options[0].pass_criteria
    assert options[0].instructor_notes
    assert options[-1].path.name == "game_training_rpo_06_defensive_target_demo.yaml"
    assert options[-1].target_delta_v_budget_m_s == 5.0


def test_game_configs_are_packaged_with_sim_package() -> None:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"

    assert '"game/configs/*.yaml"' in pyproject.read_text(encoding="utf-8")


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
    close_chaser = StateTruth(
        position_eci_km=close_chaser_state[:3],
        velocity_eci_km_s=close_chaser_state[3:],
        attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        angular_rate_body_rad_s=np.zeros(3, dtype=float),
        mass_kg=200.0,
        t_s=0.0,
    )
    far_chaser = StateTruth(
        position_eci_km=far_chaser_state[:3],
        velocity_eci_km_s=far_chaser_state[3:],
        attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        angular_rate_body_rad_s=np.zeros(3, dtype=float),
        mass_kg=200.0,
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
    chaser = StateTruth(
        position_eci_km=chaser_state[:3],
        velocity_eci_km_s=chaser_state[3:],
        attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        angular_rate_body_rad_s=np.zeros(3, dtype=float),
        mass_kg=200.0,
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
    chaser = StateTruth(
        position_eci_km=chaser_state[:3],
        velocity_eci_km_s=chaser_state[3:],
        attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        angular_rate_body_rad_s=np.zeros(3, dtype=float),
        mass_kg=200.0,
        t_s=0.0,
    )

    own_knowledge = {"chaser": _knowledge_from_state6(chaser_state)}
    first = provider(truth=target, own_knowledge=own_knowledge, t_s=10.0, dt_s=10.0)
    second = provider(truth=target, own_knowledge=own_knowledge, t_s=20.0, dt_s=10.0)

    assert np.isclose(np.linalg.norm(first["thrust_eci_km_s2"]), 5.0e-4)
    assert np.isclose(provider.used_delta_v_m_s, 5.0)
    assert np.allclose(second["thrust_eci_km_s2"], np.zeros(3), atol=1e-15)
    assert second["command_mode_flags"]["target_defensive_budget_exhausted"] is True


def test_level_six_uses_target_reference_for_game_ric_frame() -> None:
    config_path = Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_06_defensive_target_demo.yaml"
    config = SimulationConfig.from_yaml(config_path)
    session = SimulationSession.from_config(config)
    snap = session.reset()

    assert _game_ric_reference_object_id(config, "target") == "target_reference"
    assert snap is not None
    assert "target_reference" in snap.truth
    assert snap.truth["target_reference"].shape[0] >= 6


def test_level_six_restart_gets_fresh_defensive_target_provider() -> None:
    config_path = Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_06_defensive_target_demo.yaml"
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
    assert getattr(target_provider2, "used_delta_v_m_s") == 0.0


def test_terminal_mission_state_keeps_game_loop_open_after_session_done() -> None:
    passed = type("Score", (), {"level_passed": True, "level_failed": False})()
    failed = type("Score", (), {"level_passed": False, "level_failed": True})()
    active = type("Score", (), {"level_passed": False, "level_failed": False})()

    assert _game_loop_should_exit(session_done=True, score=passed) is False
    assert _game_loop_should_exit(session_done=True, score=failed) is False
    assert _game_loop_should_exit(session_done=True, score=active) is True


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

    assert nmt_velocity_error_km_s(state, mean_motion_rad_s=n, radial_amplitude_km=1.5, center_ric_km=np.zeros(3)) < 1.0e-12
    assert nmt_velocity_error_km_s(
        stopped_on_ellipse,
        mean_motion_rad_s=n,
        radial_amplitude_km=1.5,
        center_ric_km=np.zeros(3),
    ) > 0.0


def test_nmt_cross_track_phase_creates_rc_ellipse() -> None:
    curve = nmt_curve_points_km(
        radial_amplitude_km=1.5,
        cross_track_amplitude_km=1.0,
        cross_track_phase_deg=45.0,
        center_ric_km=np.zeros(3),
    )

    assert np.ptp(curve[:, 2]) > 1.9
    assert abs(np.corrcoef(curve[:, 0], curve[:, 2])[0, 1]) > 0.5
    assert nmt_position_error_km(
        curve[[25, 180, 320]],
        radial_amplitude_km=1.5,
        cross_track_amplitude_km=1.0,
        cross_track_phase_deg=45.0,
        center_ric_km=np.zeros(3),
    ).max() == 0.0


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
    assert any(item.startswith("KO ") for item in metrics)
    assert any(item.startswith("Goal ") for item in metrics)
    assert any(item.startswith("Speed ") for item in metrics)
    assert not any("NMT" in reason for reason in score.pass_fail_reasons)


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
    assert "Goal 20 m/25 m" in metrics
    assert "Speed 0.50 m/s/1.00 m/s" in metrics


def test_score_debrief_lines_show_after_terminal_mission_state() -> None:
    cfg = RPOTrainingConfig(
        enabled=True,
        scenario_id="unit-debrief",
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
    debrief = _score_debrief_lines(score)

    assert score.level_passed is True
    assert any("Scenario" in line and "unit-debrief" in line for line in debrief)
    assert any(line.startswith("Final Range") for line in debrief)
    assert any(line.startswith("Final Speed") for line in debrief)


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
        def __init__(self, type_value, key=None):
            self.type = type_value
            self.key = key

    class FakeKeys:
        def __getitem__(self, key):
            return key in {"w", "d", "right", "space"}

    class FakePygame:
        QUIT = "quit"
        KEYDOWN = "keydown"
        K_ESCAPE = "escape"
        K_r = "r"
        K_PERIOD = "."
        K_w = "w"
        K_s = "s"
        K_d = "d"
        K_a = "a"
        K_RIGHT = "right"
        K_LEFT = "left"
        K_UP = "up"
        K_DOWN = "down"
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

    class PauseStepPygame(FakePygame):
        class event:
            @staticmethod
            def get():
                return [
                    FakeEvent(FakePygame.KEYDOWN, FakePygame.K_SPACE),
                    FakeEvent(FakePygame.KEYDOWN, FakePygame.K_PERIOD),
                    FakeEvent(FakePygame.KEYDOWN, FakePygame.K_UP),
                    FakeEvent(FakePygame.KEYDOWN, FakePygame.K_r),
                ]

    state = KeyboardCommandState()

    _poll_pygame_input(PauseStepPygame, state, control_mode="ric_translation")

    assert state.paused is True
    assert state.step_requested is True
    assert state.speed_multiplier_change == 1
    assert state.restart_requested is True

    class SlowDownPygame(FakePygame):
        class event:
            @staticmethod
            def get():
                return [FakeEvent(FakePygame.KEYDOWN, FakePygame.K_DOWN)]

    state = KeyboardCommandState()

    _poll_pygame_input(SlowDownPygame, state, control_mode="ric_translation")

    assert state.speed_multiplier_change == -1


def test_speed_multiple_converts_sim_dt_to_wall_step() -> None:
    assert _wall_step_s(10.0, 10.0) == 1.0
    assert _wall_step_s(0.25, 2.0) == 0.125


def test_speed_multiple_adjustment_uses_allowed_options() -> None:
    assert _coerce_speed_multiple(3.0) == 2.0
    assert _adjust_speed_multiple(1.0, -1) == 1.0
    assert _adjust_speed_multiple(1.0, 1) == 2.0
    assert _adjust_speed_multiple(2.0, 1) == 5.0
    assert _adjust_speed_multiple(10.0, 1) == 25.0
    assert _adjust_speed_multiple(25.0, 1) == 50.0
    assert _adjust_speed_multiple(50.0, 1) == 50.0
    assert _adjust_speed_multiple(50.0, -2) == 10.0


def test_cw_coast_state_zero_time_returns_initial_state() -> None:
    x0 = np.array([0.1, -1.0, 0.2, 0.0, 0.001, -0.001], dtype=float)

    out = _cw_coast_state(x0, 0.0, 0.001)

    assert np.allclose(out, x0)


def test_coast_prediction_difficulty_maps_to_orbit_fraction() -> None:
    assert _coast_prediction_orbit_fraction("easy") == 1.0
    assert _coast_prediction_orbit_fraction("medium") == 0.5
    assert _coast_prediction_orbit_fraction("hard") == 0.25
    assert _coast_prediction_orbit_fraction("extreme") == 0.0


def test_coast_prediction_horizon_uses_orbital_period() -> None:
    n = 0.001
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.coast_prediction_horizon_s = 300.0
    dashboard.coast_prediction_orbit_fraction = 0.5

    assert np.isclose(dashboard._coast_prediction_horizon_s(n), np.pi / n)

    dashboard.coast_prediction_orbit_fraction = 0.0
    assert dashboard._coast_prediction_horizon_s(n) == 0.0


def test_close_rendezvous_zoom_ignores_old_trail_for_scale() -> None:
    class FakeScreen:
        @staticmethod
        def get_size():
            return (1280, 720)

    dashboard = object.__new__(PygameRPODashboard)
    dashboard.screen = FakeScreen()
    dashboard.goal_nmt_radial_amplitude_km = None
    dashboard.goal_radius_km = 0.025
    dashboard.keepout_radius_km = None
    dashboard.goal_relative_ric_km = np.zeros(3, dtype=float)
    dashboard.target_rel_hist = []

    close_rel = np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    far_trail = np.array([[0.0, -0.75], [0.0, 0.0], [0.0, 0.02]], dtype=float)
    close_scale = dashboard._scale_for_plot(pts=[close_rel[:2].reshape(1, 2), np.zeros((1, 2), dtype=float)])
    history_scale = dashboard._scale_for_plot(pts=[far_trail])

    assert dashboard._use_close_goal_zoom(close_rel) is True
    assert close_scale > history_scale
    assert close_scale * dashboard.goal_radius_km >= 45.0
