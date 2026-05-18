from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import yaml

import sim.game.runner as game_runner
import sim.game.training as game_training
from sim.api import SimulationConfig, SimulationSession
from sim.core.models import Command, StateBelief, StateTruth
from sim.game.arcade import (
    _arcade_round_briefing_lines,
    _arcade_round_rng,
    _arcade_round_time_bonus_s,
    _arcade_round_weighted_score,
    _arcade_score,
    _difficulty_score_multiplier,
    _game_arcade_delta_v_bonus_time_per_m_s,
    _game_arcade_enabled,
    _game_arcade_initial_time_s,
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
from sim.game.debrief import game_debrief_path, tracker_replay_history, write_game_debrief
from sim.game.defensive_target import DefensiveTargetIntentProvider
from sim.game.launcher import (
    GameScenarioOption,
    _clamp_preview_scroll_px,
    _clear_progress_at_pos,
    _difficulty_at_pos,
    _difficulty_index,
    _fit_text_px,
    _game_progress_path,
    _music_at_pos,
    _option_index_at_pos,
    _preview_bounds,
    _preview_content_height,
    _progress_stars,
    _record_video_at_pos,
    _scroll_for_selection,
    _start_artwork_rect,
    _start_screen_event_action,
    _wrap_text_px,
    clear_game_progress,
    discover_game_scenarios,
    record_game_progress,
)
from sim.game.manual import KeyboardCommandState, ManualGameCommandProvider
from sim.game.pygame_dashboard import PygameRPODashboard, _cw_coast_state, _sample_rows
from sim.game.recording import GameFrameRecorder, game_recording_path
from sim.game.runner import (
    _adjust_speed_multiple,
    _coast_prediction_orbit_fraction,
    _coerce_speed_multiple,
    _dashboard_fps_for_speed,
    _dashboard_object_ids,
    _game_camera_mode,
    _game_coast_chaser_after_delta_v_budget,
    _game_control_mode,
    _game_controlled_object_id,
    _game_loop_should_exit,
    _game_plot_axis_scale,
    _game_plot_equal_axis_scale_planes,
    _game_plot_fixed_axis_half_span_km,
    _game_plot_overlays_in_zoom,
    _game_plot_overlays_in_zoom_by_plane,
    _game_proximity_ring_plot_planes,
    _game_ric_reference_object_id,
    _game_show_target_hcw_path,
    _mission_checklist,
    _mission_metrics,
    _poll_pygame_input,
    _realtime_steps_due,
    _score_debrief_lines,
    _speed_after_maneuver_input,
    _start_game_attempt,
    _step_game_attempt,
    _training_briefing_lines,
    _wall_step_s,
)
from sim.game.session import _attempt_config_for_training_clock, _DeltaVLimitedOrbitController
from sim.game.training import (
    ApproachGateConfig,
    ForbiddenRegionConfig,
    RequiredPhaseBurnConfig,
    RPOTrainingConfig,
    RPOTrainingTracker,
    nmt_curve_points_km,
    nmt_element_errors,
    nmt_position_error_km,
    nmt_velocity_error_km_s,
)
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
    options = discover_game_scenarios(Path(__file__).resolve().parents[1] / "game" / "configs")

    assert [option.scenario_id for option in options] == [
        "rpo_00_tutorial",
        "rpo_01_coast_relative_motion",
        "rpo_02_vbar_approach",
        "rpo_03_rbar_approach",
        "rpo_04_rendezvous",
        "rpo_05_passive_cross_track_approach",
        "rpo_06_defensive_target_demo",
        "rpo_07_evasive_target_survival",
        "rpo_arcade_pursuit",
    ]
    assert options[0].title == "Level 0 - Tutorial"
    assert options[1].title == "Level 1 - Relative Orbit"
    assert options[5].title == "Level 5 - Safe Inspection"
    assert options[0].player_brief
    assert options[0].pass_criteria
    assert options[0].instructor_notes
    assert options[0].time_budget_s == pytest.approx(18000.0)
    assert options[0].delta_v_budget_m_s == pytest.approx(12.0)
    assert options[0].path.name == "game_training_rpo_00_tutorial.yaml"
    assert options[5].path.name == "game_training_rpo_05_passive_cross_track_approach.yaml"
    assert options[6].path.name == "game_training_rpo_06_defensive_target_demo.yaml"
    assert options[7].path.name == "game_training_rpo_07_evasive_target_survival.yaml"
    assert options[7].title == "Level 7 - Evasive Target"
    assert options[7].delta_v_budget_m_s == pytest.approx(25.0)
    assert options[7].target_delta_v_budget_m_s == pytest.approx(1.0)
    assert options[6].target_delta_v_budget_m_s == pytest.approx(0.1)
    assert options[8].title == "Pursuit Arcade"
    assert options[8].time_budget_s == pytest.approx(12000.0)
    assert options[8].delta_v_budget_m_s == pytest.approx(8.0)
    assert options[8].target_delta_v_budget_m_s == pytest.approx(0.1)


def test_game_configs_are_packaged_and_music_is_optional() -> None:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")

    assert '"game/configs/*.yaml"' in text
    assert '"game/assets/*.png"' in text
    assert '"game/music/' not in text


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


def test_start_screen_artwork_rect_covers_screen_without_distortion() -> None:
    rect = _start_artwork_rect((1672, 941), (1040, 680))
    x, y, width, height = rect

    assert x <= 0
    assert y <= 0
    assert width >= 1040
    assert height >= 680
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


def test_launcher_preview_wraps_text_to_pixel_width() -> None:
    font = _FixedWidthFont()
    lines = _wrap_text_px(
        "Use small pulses and long coast arcs to shape the relative orbit.",
        font,
        160,
    )

    assert len(lines) > 1
    assert all(font.size(line)[0] <= 160 for line in lines)


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
        scenario_name="RPO 06 Defensive Target Demo",
        difficulty="Hard",
        attempt_index=3,
        output_dir=tmp_path,
        timestamp=datetime(2026, 5, 14, 12, 30, 45),
    )

    assert path == tmp_path / "rpo_06_defensive_target_demo_hard_20260514_123045_attempt03.mp4"


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
    payload = yaml.safe_load(out.read_text(encoding="utf-8"))

    assert out == tmp_path / "unit_debrief_easy_20260514_123045_attempt02.json"
    assert payload["scenario_id"] == "unit-debrief"
    assert payload["level_passed"] is True
    assert payload["score"]["arcade_score"] == 123
    assert payload["score"]["arcade_seed"] == 456
    assert payload["score"]["arcade_round_index"] == 7
    assert payload["artifacts"]["recording_path"].endswith("attempt.mp4")
    assert payload["replay"]["time_s"] == [0.0]
    assert payload["replay"]["relative_ric"][0][:3] == pytest.approx([0.0, -0.2, 0.0])


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


def test_game_recording_defaults_to_dashboard_fps() -> None:
    assert game_runner.run_game_mode.__kwdefaults__["recording_fps"] == game_runner.DASHBOARD_FPS


def test_run_game_mode_discards_recorder_when_initial_capture_fails(tmp_path: Path, monkeypatch) -> None:
    import sim.game.pygame_dashboard as dashboard_module

    class FakeDashboard:
        instances: list[FakeDashboard] = []

        def __init__(self, *args, **kwargs):
            self.screen = object()
            self.closed = False
            FakeDashboard.instances.append(self)

        def push_snapshot(self, snapshot) -> None:
            pass

        def draw(self, **kwargs) -> None:
            pass

        def close(self) -> None:
            self.closed = True

    class FakeRecorder:
        saved = False

        def __init__(self) -> None:
            self.discarded = False

        def discard(self) -> None:
            self.discarded = True

    recorder = FakeRecorder()
    cfg = _game_config(tmp_path)
    path = tmp_path / "game.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    monkeypatch.setattr(dashboard_module, "PygameRPODashboard", FakeDashboard)
    monkeypatch.setattr(game_runner, "_start_game_recorder", lambda **kwargs: recorder)
    monkeypatch.setattr(
        game_runner,
        "_capture_recording_frame",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("capture failed")),
    )

    with pytest.raises(RuntimeError, match="capture failed"):
        game_runner.run_game_mode(path, record_video=True)

    assert recorder.discarded is True
    assert FakeDashboard.instances[-1].closed is True


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


def test_level_six_uses_target_reference_for_game_ric_frame() -> None:
    config_path = (
        Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_06_defensive_target_demo.yaml"
    )
    config = SimulationConfig.from_yaml(config_path)
    session = SimulationSession.from_config(config)
    snap = session.reset()

    assert _game_ric_reference_object_id(config, "target") == "target_reference"
    assert snap is not None
    assert "target_reference" in snap.truth
    assert snap.truth["target_reference"].shape[0] >= 6


def test_level_six_ric_translation_commands_use_target_reference_frame() -> None:
    config_path = (
        Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_06_defensive_target_demo.yaml"
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


def test_level_six_restart_gets_fresh_defensive_target_provider() -> None:
    config_path = (
        Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_06_defensive_target_demo.yaml"
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


def test_level_six_goal_is_100_meter_close_approach() -> None:
    config_path = (
        Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_06_defensive_target_demo.yaml"
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


def test_pursuit_arcade_uses_level_six_shape_with_arcade_clock() -> None:
    config_path = (
        Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_arcade_pursuit.yaml"
    )
    config = SimulationConfig.from_yaml(config_path)
    training_cfg = RPOTrainingConfig.from_metadata(dict(config.scenario.metadata or {}))
    defensive_target = config.scenario.metadata["game"]["defensive_target"]

    assert _game_arcade_enabled(config) is True
    assert _game_arcade_initial_time_s(config, training_cfg) == pytest.approx(12000.0)
    assert _game_arcade_round_bonus_time_s(config) == pytest.approx(500.0)
    assert _game_arcade_delta_v_bonus_time_per_m_s(config) == pytest.approx(100.0)
    assert config.scenario.metadata["game"]["level_name"] == "Pursuit Arcade"
    assert training_cfg.scenario_id == "rpo_arcade_pursuit"
    assert training_cfg.goal_range_km == pytest.approx(0.1)
    assert training_cfg.max_time_s == pytest.approx(12000.0)
    assert training_cfg.max_delta_v_m_s == pytest.approx(8.0)
    assert config.scenario.simulator.duration_s == pytest.approx(12000.0)
    assert defensive_target["max_delta_v_m_s"] == pytest.approx(0.1)


def test_level_zero_tutorial_is_passive_close_range_intro() -> None:
    config_path = (
        Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_00_tutorial.yaml"
    )
    config = SimulationConfig.from_yaml(config_path)
    training_cfg = RPOTrainingConfig.from_metadata(dict(config.scenario.metadata or {}))
    chaser_initial = config.scenario.objects["chaser"].initial_state["relative_to_target_ric"]

    assert training_cfg.scenario_id == "rpo_00_tutorial"
    assert config.scenario.metadata["game"]["level_name"] == "Level 0 - Tutorial"
    assert _game_control_mode(config) == "ric_translation"
    assert _game_camera_mode(config) == "target_pair"
    assert _game_plot_overlays_in_zoom(config) is False
    assert training_cfg.goal_range_km == pytest.approx(0.25)
    assert training_cfg.goal_radius_km is None
    assert training_cfg.keepout_radius_km is None
    assert training_cfg.max_time_s == pytest.approx(18000.0)
    assert training_cfg.max_delta_v_m_s == pytest.approx(12.0)
    assert training_cfg.required_burn_axes == ("radial", "in_track", "cross_track")
    assert training_cfg.require_speed_multiplier_change is True
    assert config.scenario.simulator.duration_s == pytest.approx(18000.0)
    assert chaser_initial["frame"] == "rect"
    assert np.allclose(chaser_initial["state"], np.array([0.0, -0.8, 0.0, 0.0, 0.0, 0.0]))


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


def test_level_seven_is_player_target_survival_against_hcw_pd_chaser() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "game"
        / "configs"
        / "game_training_rpo_07_evasive_target_survival.yaml"
    )
    config = SimulationConfig.from_yaml(config_path)
    training_cfg = RPOTrainingConfig.from_metadata(dict(config.scenario.metadata or {}))
    chaser = config.scenario.objects["chaser"]
    chaser_initial = chaser.initial_state["relative_to_target_ric"]
    source_config = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / "configs" / "hcw_pd_10km_experiment.yaml").read_text(
            encoding="utf-8"
        )
    )
    source_control = source_config["objects"]["chaser"]["orbit_control"]

    assert _game_controlled_object_id(config) == "target"
    assert _game_control_mode(config) == "ric_translation"
    assert _game_camera_mode(config) == "target_pair"
    assert _game_show_target_hcw_path(config) is True
    assert training_cfg.scenario_id == "rpo_07_evasive_target_survival"
    assert training_cfg.survival_goal is True
    assert training_cfg.keepout_radius_km == pytest.approx(0.1)
    assert training_cfg.goal_range_km is None
    assert training_cfg.max_goal_speed_km_s is None
    assert training_cfg.max_time_s == pytest.approx(6000.0)
    assert training_cfg.max_target_delta_v_m_s == pytest.approx(1.0)
    assert training_cfg.max_delta_v_m_s == pytest.approx(25.0)
    assert config.scenario.simulator.duration_s == pytest.approx(6000.0)
    assert training_cfg.fail_on_delta_v_budget is False
    assert _game_coast_chaser_after_delta_v_budget(training_cfg) is True
    assert chaser.orbit_control.module == "sim.control.orbit.hcw_pd"
    assert chaser.orbit_control.class_name == "HCWPDController"
    assert chaser.orbit_control.module == source_control["module"]
    assert chaser.orbit_control.class_name == source_control["class_name"]
    assert chaser.orbit_control.params == source_control["params"]
    assert chaser_initial["frame"] == "curv"
    assert np.allclose(chaser_initial["state"], np.array([0.0, -10.0, 0.0, 0.0, 0.0, 0.001]))


def test_level_seven_player_controls_target_while_chaser_controller_runs() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "game"
        / "configs"
        / "game_training_rpo_07_evasive_target_survival.yaml"
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
    level2 = RPOTrainingConfig(enabled=True, scenario_id="rpo_02_vbar_approach")
    level3 = RPOTrainingConfig(enabled=True, scenario_id="rpo_03_rbar_approach")
    level4 = RPOTrainingConfig(enabled=True, scenario_id="rpo_04_rendezvous")
    level5 = RPOTrainingConfig(enabled=True, scenario_id="rpo_05_passive_cross_track_approach")
    level6 = RPOTrainingConfig(enabled=True, scenario_id="rpo_06_defensive_target_demo")
    arcade = RPOTrainingConfig(enabled=True, scenario_id="rpo_arcade_pursuit")
    unmapped = RPOTrainingConfig(enabled=True, scenario_id="rpo_08_unmapped")

    assert _level_music_path(tutorial) == LEVEL_MUSIC_PATHS["rpo_00_tutorial"]
    assert _level_music_path(tutorial).name == "10_training_grid_sunrise.wav"
    assert _level_music_path(level2) == LEVEL_MUSIC_PATHS["rpo_02_vbar_approach"]
    assert _level_music_path(level2).name == "02_rendezvous_vector.wav"
    assert _level_music_path(level3) == LEVEL_MUSIC_PATHS["rpo_03_rbar_approach"]
    assert _level_music_path(level3).name == "18_keepout_zone_accelerando.wav"
    assert _level_music_path(level4) == LEVEL_MUSIC_PATHS["rpo_04_rendezvous"]
    assert _level_music_path(level4).name == "06_casting_the_orbit_line.wav"
    assert _level_music_path(level5) == LEVEL_MUSIC_PATHS["rpo_05_passive_cross_track_approach"]
    assert _level_music_path(level5).name == "19_cross_track_ghost_orbit.wav"
    assert _level_music_path(level6) == LEVEL_MUSIC_PATHS["rpo_06_defensive_target_demo"]
    assert _level_music_path(level6).name == "17_orbital_boss_metal.wav"
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
    training_cfg = RPOTrainingConfig(enabled=True, max_delta_v_m_s=8.0)
    score = type("Score", (), {"approximate_delta_v_m_s": 2.5})()

    assert _arcade_round_time_bonus_s(config, training_cfg, score) == pytest.approx(1050.0)


def test_arcade_round_briefing_lines_show_transition_summary() -> None:
    lines = _arcade_round_briefing_lines(
        cleared_round_index=2,
        next_round_index=3,
        round_score=4200,
        total_score=7000,
        time_used_s=1234.4,
        bonus_time_s=850.0,
        next_time_budget_s=5000.6,
    )

    assert lines[0] == "Round 2 Cleared"
    assert "Round score: 4,200" in lines[1]
    assert "Total score: 7,000" in lines[1]
    assert "Bonus awarded: 850 s" in lines[2]
    assert "Round 3 starts with 5001 s" in lines[3]
    assert "Fuel resets" in lines[4]


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


def test_mission_metrics_show_delta_v_remaining_with_two_decimals() -> None:
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

    assert "OK Chaser dV  6.77 m/s" in metrics
    assert "OK Target dV  0.88 m/s" in metrics


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
    assert "WARN Speed x" in metrics
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
    assert "OK Speed x" in metrics
    assert checklist[:4] == (
        "OK Radial burn",
        "OK In-track burn",
        "OK Cross-track burn",
        "OK Change speed",
    )


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
    assert rbar.approach_gates == ()
    assert _game_plot_overlays_in_zoom(rbar_sim_cfg) is False


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
        "west RC inspection gate",
        "south RC inspection gate",
        "east RC inspection gate",
        "north RC inspection gate",
    ]
    assert np.allclose(cfg.inspection_gates[0].center_ric_km, np.array([0.0, 2.25, -0.75], dtype=float))
    assert np.allclose(cfg.inspection_gates[1].center_ric_km, np.array([-0.75, 0.75, 0.0], dtype=float))
    assert np.allclose(cfg.inspection_gates[2].center_ric_km, np.array([0.0, -0.75, 0.75], dtype=float))
    assert np.allclose(cfg.inspection_gates[3].center_ric_km, np.array([0.75, -2.25, 0.0], dtype=float))
    assert np.allclose(cfg.inspection_gates[0].half_width_ric_km, np.array([0.25, 0.7, 0.12], dtype=float))
    assert np.allclose(cfg.inspection_gates[1].half_width_ric_km, np.array([0.12, 0.7, 0.25], dtype=float))
    assert {gate.max_total_speed_km_s for gate in cfg.inspection_gates} == {None}
    chaser_state = sim_cfg.scenario.objects["chaser"].initial_state["relative_to_target_ric"]["state"]
    assert np.allclose(chaser_state, np.array([0.0, 5.5, 0.0, 0.0, 0.0, 0.0], dtype=float))
    assert len(cfg.forbidden_regions) == 1
    cylinder = cfg.forbidden_regions[0]
    assert cylinder.kind == "cylinder"
    assert cylinder.axis == "I"
    assert cylinder.radius_km == 0.5
    assert cylinder.height_km == 6.0
    assert cylinder.plot_planes == ("RI", "RC")
    assert bool(cylinder.contains_positions(np.array([[0.25, 0.0, 0.25]], dtype=float))[0]) is True
    assert bool(cylinder.contains_positions(np.array([[0.45, 3.1, 0.0]], dtype=float))[0]) is False
    assert bool(cylinder.contains_positions(np.array([[0.6, 0.0, 0.0]], dtype=float))[0]) is False
    assert bool(cylinder.contains_positions(np.array([[0.0, -1.5, 0.6]], dtype=float))[0]) is False
    assert bool(cylinder.contains_positions(np.array([[0.0, -1.5, 0.5]], dtype=float))[0]) is True


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
        np.array([0.0, 2.25, -0.75, 0.0, 0.0001, 0.0], dtype=float),
        np.array([-0.75, 0.75, 0.0, 0.0, 0.0001, 0.0], dtype=float),
        np.array([0.0, -0.75, 0.75, 0.0, 0.0001, 0.0], dtype=float),
        np.array([0.75, -2.25, 0.0, 0.0, 0.0001, 0.0], dtype=float),
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
            np.array([0.35, 2.25, -0.75, 0.0, 0.0001, 0.0], dtype=float),
            np.array([-0.35, 2.25, -0.75, 0.0, 0.0001, 0.0], dtype=float),
        ),
        (
            np.array([-0.75, 0.75, 0.35, 0.0, 0.0001, 0.0], dtype=float),
            np.array([-0.75, 0.75, -0.35, 0.0, 0.0001, 0.0], dtype=float),
        ),
        (
            np.array([-0.35, -0.75, 0.75, 0.0, 0.0001, 0.0], dtype=float),
            np.array([0.35, -0.75, 0.75, 0.0, 0.0001, 0.0], dtype=float),
        ),
        (
            np.array([0.75, -2.25, -0.35, 0.0, 0.0001, 0.0], dtype=float),
            np.array([0.75, -2.25, 0.35, 0.0, 0.0001, 0.0], dtype=float),
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
        np.array([0.0, 2.25, -0.75, 0.0, 0.0020, 0.0], dtype=float),
        np.array([-0.75, 0.75, 0.0, 0.0, 0.0020, 0.0], dtype=float),
        np.array([0.0, -0.75, 0.75, 0.0, 0.0020, 0.0], dtype=float),
        np.array([0.75, -2.25, 0.0, 0.0, 0.0020, 0.0], dtype=float),
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
        np.array([0.0, 2.25, -0.75, 0.0, 0.0001, 0.0], dtype=float),
        np.array([-0.75, 0.75, 0.0, 0.0, 0.0001, 0.0], dtype=float),
        np.array([0.0, -0.75, 0.75, 0.0, 0.0001, 0.0], dtype=float),
        np.array([0.75, -2.25, 0.0, 0.0, 0.0001, 0.0], dtype=float),
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
        np.array([0.0, -0.75, 0.75, 0.0, 0.0001, 0.0], dtype=float),
        np.array([0.75, -2.25, 0.0, 0.0, 0.0001, 0.0], dtype=float),
        np.array([0.0, 2.25, -0.75, 0.0, 0.0001, 0.0], dtype=float),
        np.array([-0.75, 0.75, 0.0, 0.0, 0.0001, 0.0], dtype=float),
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
    assert cfg.max_goal_speed_km_s == pytest.approx(0.0001)
    assert cfg.hard_speed_limit_radius_km == pytest.approx(0.025)
    assert cfg.hard_speed_limit_km_s == pytest.approx(0.0001)


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
    assert "FAIL FR violated" in metrics


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
    assert "OK Range 0.75 km/0.75 km" in metrics


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

    assert tracker.current_hint() == "Inside hold box: slow below 0.30 m/s to finish."


def test_training_tracker_hint_calls_out_range_goal_speed_blocker() -> None:
    cfg = RPOTrainingConfig(
        enabled=True,
        scenario_id="unit-range-hint",
        goal_range_km=0.75,
        max_goal_speed_km_s=0.0003,
    )
    tracker = RPOTrainingTracker(cfg)
    tracker.rel_ric_hist.append(np.array([-0.45, -0.60, 0.0, 0.0005, 0.0, 0.0], dtype=float))

    assert tracker.current_hint() == "Inside green circle: slow below 0.30 m/s to finish."


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
    assert "WARN Goal 20 m/25 m" in metrics
    assert "OK Speed 0.50 m/s/1.00 m/s" in metrics


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
    assert "FAIL Prox V <= 0.10 m/s" in metrics


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
        K_m = "m"
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
                    FakeEvent(FakePygame.KEYDOWN, FakePygame.K_m),
                ]

    state = KeyboardCommandState()

    _poll_pygame_input(PauseStepPygame, state, control_mode="ric_translation")

    assert state.paused is True
    assert state.step_requested is True
    assert state.speed_multiplier_change == 1
    assert state.restart_requested is True
    assert state.music_toggle_requested is True

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
    assert _adjust_speed_multiple(50.0, 1) == 100.0
    assert _adjust_speed_multiple(100.0, 1) == 100.0
    assert _adjust_speed_multiple(50.0, -2) == 10.0


def test_high_speed_maneuver_input_drops_to_control_speed() -> None:
    ric_state = KeyboardCommandState(pitch=1.0)
    coasting_state = KeyboardCommandState()
    no_throttle_state = KeyboardCommandState(pitch=1.0, throttle=0.0)

    assert _speed_after_maneuver_input(100.0, ric_state, control_mode="ric_translation") == 10.0
    assert _speed_after_maneuver_input(50.0, ric_state, control_mode="ric_translation") == 10.0
    assert _speed_after_maneuver_input(25.0, ric_state, control_mode="ric_translation") == 25.0
    assert _speed_after_maneuver_input(100.0, coasting_state, control_mode="ric_translation") == 100.0
    assert _speed_after_maneuver_input(100.0, no_throttle_state, control_mode="ric_translation") == 100.0


def test_high_speed_attitude_or_thrust_input_drops_to_control_speed() -> None:
    rotate_state = KeyboardCommandState(yaw=1.0)
    firing_state = KeyboardCommandState(firing=True)
    coasting_state = KeyboardCommandState()

    assert _speed_after_maneuver_input(100.0, rotate_state, control_mode="attitude_thrust") == 10.0
    assert _speed_after_maneuver_input(50.0, firing_state, control_mode="attitude_thrust") == 10.0
    assert _speed_after_maneuver_input(25.0, firing_state, control_mode="attitude_thrust") == 25.0
    assert _speed_after_maneuver_input(100.0, coasting_state, control_mode="attitude_thrust") == 100.0


def test_dashboard_fps_drops_at_high_speed_unless_recording() -> None:
    assert _dashboard_fps_for_speed(10.0) == 60.0
    assert _dashboard_fps_for_speed(50.0) == 45.0
    assert _dashboard_fps_for_speed(100.0) == 30.0
    assert _dashboard_fps_for_speed(100.0, recording=True) == 60.0


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


def test_dashboard_samples_long_polylines_for_drawing() -> None:
    rows = np.arange(1000.0).reshape(500, 2)

    sampled = _sample_rows(rows, 120)

    assert sampled.shape[0] <= 120
    assert np.allclose(sampled[0], rows[0])
    assert np.allclose(sampled[-1], rows[-1])


def test_coast_prediction_caps_draw_points() -> None:
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.mean_motion_rad_s = 0.001
    dashboard.coast_prediction_orbit_fraction = 1.0
    dashboard.coast_prediction_horizon_s = 300.0
    dashboard.coast_prediction_dt_s = 1.0

    prediction = dashboard._coast_prediction_from(np.array([0.1, -1.0, 0.2, 0.0, 0.001, -0.001], dtype=float))

    assert 2 <= prediction.shape[0] <= 120


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


def test_target_pair_camera_centers_ri_between_current_satellites() -> None:
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.camera_mode = "target_pair"

    center = dashboard._camera_center_ric(
        chaser_current=np.array([0.2, -1.0, 0.4], dtype=float),
        target_current=np.array([-0.4, 0.5, -0.2], dtype=float),
        x_axis=1,
        y_axis=0,
    )

    assert np.allclose(center, np.array([-0.1, -0.25, 0.0], dtype=float))


def test_target_pair_camera_keeps_reference_centered_for_rc() -> None:
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.camera_mode = "target_pair"

    center = dashboard._camera_center_ric(
        chaser_current=np.array([0.2, -1.0, 0.4], dtype=float),
        target_current=np.array([-0.4, 0.5, -0.2], dtype=float),
        x_axis=2,
        y_axis=0,
    )

    assert np.allclose(center, np.zeros(3, dtype=float))


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

    assert rc_min_span == pytest.approx(0.87 * 1.18)
    assert ri_min_span == pytest.approx(3.0 * 1.18)
    assert close_scale == pytest.approx(dashboard._scale_for_plot(pts=[], min_span_km=rc_min_span))
    assert wide_scale < close_scale


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
    assert ri_ignored_span == pytest.approx(0.05)
    assert rc_fixed_span > 5.0
    assert ri_full_overlay_span > 5.0
