from __future__ import annotations

# These owner-aligned tests share deterministic builders and compatibility
# imports from the adjacent support module.
# ruff: noqa: F403, F405
from sim.tests.game_mode_test_support import *


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

    game_runner._poll_pygame_input(FakePygame, state, control_mode="attitude_thrust")

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

    assert game_runner._opposing_key_axis(FakeKeys({"d"}), positive_key="d", negative_key="a") == 1.0
    assert game_runner._opposing_key_axis(FakeKeys({"a"}), positive_key="d", negative_key="a") == -1.0
    assert game_runner._opposing_key_axis(FakeKeys({"d", "a"}), positive_key="d", negative_key="a") == 0.0


def test_space_force_ric_input_flips_a_and_d_without_changing_attitude_yaw() -> None:
    class FakeKeys:
        def __init__(self, pressed: set[str]) -> None:
            self.pressed = pressed

        def __getitem__(self, key):
            return key in self.pressed

    class FakePygame:
        QUIT = "quit"
        KEYDOWN = "keydown"
        K_w = "w"
        K_s = "s"
        K_d = "d"
        K_a = "a"
        K_RIGHT = "right"
        K_LEFT = "left"
        K_SPACE = "space"
        K_r = "r"
        K_m = "m"

        class event:
            @staticmethod
            def get():
                return []

        class key:
            pressed = {"d"}

            @classmethod
            def get_pressed(cls):
                return FakeKeys(cls.pressed)

    space_force = frame_convention_from_preset(FRAME_CONVENTION_PRESET_SPACE_FORCE)
    state = KeyboardCommandState()

    game_runner._poll_pygame_input(
        FakePygame,
        state,
        control_mode="ric_translation",
        frame_convention=space_force,
    )
    assert state.yaw == -1.0

    FakePygame.key.pressed = {"a"}
    game_runner._poll_pygame_input(
        FakePygame,
        state,
        control_mode="ric_translation",
        frame_convention=space_force,
    )
    assert state.yaw == 1.0

    FakePygame.key.pressed = {"d"}
    game_runner._poll_pygame_input(
        FakePygame,
        state,
        control_mode="attitude_thrust",
        frame_convention=space_force,
    )
    assert state.yaw == 1.0


def test_speed_multiple_converts_sim_dt_to_wall_step() -> None:
    assert game_runner._wall_step_s(10.0, 10.0) == 1.0
    assert game_runner._wall_step_s(0.25, 2.0) == 0.125


def test_shared_game_tick_schedule_clamps_to_level_base_dt(tmp_path: Path) -> None:
    leo_like = deepcopy(_game_config(tmp_path / "leo"))
    leo_like["simulator"]["duration_s"] = 2.0
    leo_like["simulator"]["dt_s"] = 1.0
    leo_cfg = SimulationConfig.from_dict(leo_like)

    assert game_runner._game_speed_dt_schedule(leo_cfg) == (
        (1.0, 0.1),
        (10.0, 0.5),
        (25.0, 1.0),
        (50.0, 5.0),
        (100.0, 10.0),
    )
    assert game_runner._game_tick_dt_s(leo_cfg, 1.0) == pytest.approx(0.1)
    assert game_runner._game_tick_dt_s(leo_cfg, 5.0) == pytest.approx(0.1)
    assert game_runner._game_tick_dt_s(leo_cfg, 10.0) == pytest.approx(0.5)
    assert game_runner._game_tick_dt_s(leo_cfg, 25.0) == pytest.approx(1.0)
    assert game_runner._game_tick_dt_s(leo_cfg, 200.0) == pytest.approx(1.0)

    cislunar_like = deepcopy(_game_config(tmp_path / "cislunar_like"))
    cislunar_like["simulator"]["duration_s"] = 20.0
    cislunar_like["simulator"]["dt_s"] = 10.0
    cislunar_cfg = SimulationConfig.from_dict(cislunar_like)

    assert game_runner._game_tick_dt_s(cislunar_cfg, 1.0) == pytest.approx(0.1)
    assert game_runner._game_tick_dt_s(cislunar_cfg, 5.0) == pytest.approx(0.1)
    assert game_runner._game_tick_dt_s(cislunar_cfg, 10.0) == pytest.approx(0.5)
    assert game_runner._game_tick_dt_s(cislunar_cfg, 25.0) == pytest.approx(1.0)
    assert game_runner._game_tick_dt_s(cislunar_cfg, 50.0) == pytest.approx(5.0)
    assert game_runner._game_tick_dt_s(cislunar_cfg, 200.0) == pytest.approx(10.0)


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
    assert game_runner._coerce_speed_multiple(3.0) == 2.0
    assert game_runner._adjust_speed_multiple(1.0, -1) == 1.0
    assert game_runner._adjust_speed_multiple(1.0, 1) == 2.0
    assert game_runner._adjust_speed_multiple(2.0, 1) == 5.0
    assert game_runner._adjust_speed_multiple(10.0, 1) == 25.0
    assert game_runner._adjust_speed_multiple(25.0, 1) == 50.0
    assert game_runner._adjust_speed_multiple(50.0, 1) == 100.0
    assert game_runner._adjust_speed_multiple(100.0, 1) == 200.0
    assert game_runner._adjust_speed_multiple(200.0, 1) == 200.0
    assert game_runner._adjust_speed_multiple(50.0, -2) == 10.0


def test_operator_burn_cinematic_arms_near_next_burn_and_caps_speed() -> None:
    runtime = OperatorBurnCinematicRuntime()
    provider = type("Provider", (), {"next_burn_time_s": lambda self: 103.0})()

    assert game_runner._operator_burn_cinematic_should_arm(provider, current_sim_time_s=97.0, dt_s=1.0) is False
    game_runner._update_operator_burn_cinematic(runtime, provider, now_wall_s=10.0, current_sim_time_s=99.0, dt_s=1.0)

    assert runtime.active is True
    assert runtime.hold_until_wall_s is None
    assert game_runner._operator_burn_cinematic_speed_multiple(100.0, runtime) == pytest.approx(10.0)
    assert game_runner._operator_burn_cinematic_speed_multiple(5.0, runtime) == pytest.approx(5.0)


def test_operator_burn_cinematic_arms_for_high_speed_frame_horizon() -> None:
    runtime = OperatorBurnCinematicRuntime()
    provider = type("Provider", (), {"next_burn_time_s": lambda self: 125.0})()

    assert game_runner._operator_burn_cinematic_should_arm(provider, current_sim_time_s=100.0, dt_s=10.0) is False
    game_runner._update_operator_burn_cinematic(
        runtime,
        provider,
        now_wall_s=10.0,
        current_sim_time_s=100.0,
        dt_s=10.0,
        frame_horizon_s=30.0,
    )

    assert runtime.active is True
    assert game_runner._operator_burn_cinematic_speed_multiple(1000.0, runtime, options=(1.0, 10.0, 1000.0)) == pytest.approx(
        10.0
    )


def test_operator_burn_visual_duration_scales_with_delta_v() -> None:
    assert game_runner._operator_burn_visual_duration_s(0.0) == pytest.approx(1.0)
    assert game_runner._operator_burn_visual_duration_s(0.5) == pytest.approx(1.1)
    assert game_runner._operator_burn_visual_duration_s(2.0) == pytest.approx(1.4)
    assert game_runner._operator_burn_visual_duration_s(5.0) == pytest.approx(2.0)
    assert game_runner._operator_burn_visual_duration_s(20.0) == pytest.approx(2.0)


def test_operator_burn_cinematic_holds_until_animation_finishes() -> None:
    runtime = OperatorBurnCinematicRuntime(active=True)
    provider = type("Provider", (), {"next_burn_time_s": lambda self: None})()

    game_runner._operator_burn_cinematic_hold_for_animation(runtime, now_wall_s=20.0, duration_s=1.15)
    game_runner._update_operator_burn_cinematic(runtime, provider, now_wall_s=21.0, current_sim_time_s=110.0, dt_s=1.0)

    assert runtime.active is True
    assert game_runner._operator_burn_cinematic_speed_multiple(50.0, runtime) == pytest.approx(10.0)

    game_runner._update_operator_burn_cinematic(runtime, provider, now_wall_s=21.16, current_sim_time_s=110.0, dt_s=1.0)

    assert runtime.active is False
    assert game_runner._operator_burn_cinematic_speed_multiple(50.0, runtime) == pytest.approx(50.0)


def test_operator_terminal_phase_waits_for_burn_animation() -> None:
    score = type("Score", (), {"level_passed": True, "level_failed": False})()
    runtime = OperatorBurnCinematicRuntime(active=True)

    assert (
        game_runner._phase_from_score_with_operator_animation(
            score,
            paused=True,
            game_mode="operator",
            operator_burn_cinematic=runtime,
        )
        == game_runner.GamePhase.PLAYING
    )

    runtime.reset()

    assert (
        game_runner._phase_from_score_with_operator_animation(
            score,
            paused=True,
            game_mode="operator",
            operator_burn_cinematic=runtime,
        )
        == game_runner.GamePhase.PASSED
    )


def test_command_status_uses_capitalized_indicators() -> None:
    ric_status = game_runner._command_status(KeyboardCommandState(paused=True, yaw=1.0), control_mode="ric_translation")
    attitude_status = game_runner._command_status(KeyboardCommandState(firing=False), control_mode="attitude_thrust")

    assert ric_status == "W/S R  A -I / D +I  Left/Right C  C Camera  O/P ECI  M Music"
    assert "M Music" in ric_status
    assert "PAUSED" not in ric_status
    assert "Throttle=" not in ric_status
    assert "W/S Pitch" in attitude_status
    assert "Space Fire" in attitude_status
    assert "Thrust=Coast" in attitude_status

    space_force_status = game_runner._command_status(
        KeyboardCommandState(),
        control_mode="ric_translation",
        frame_convention=frame_convention_from_preset(FRAME_CONVENTION_PRESET_SPACE_FORCE),
    )
    assert "A +I / D -I" in space_force_status


def test_operator_command_status_shows_next_burn_instead_of_keyboard_controls() -> None:
    provider = GameOperatorController(
        parse_operator_burn_plan("T= 50 s, 2.0 m/s R, 1.0 m/s I, 0.2 m/s C"),
        GameOperatorInputAdapter(source_id="test/operator", boot_id="test-boot"),
    )

    status = game_runner._operator_next_burn_status(provider)
    operator_status = game_runner._game_command_status(
        KeyboardCommandState(),
        control_mode="ric_translation",
        game_mode="operator",
        command_provider=provider,
    )
    pilot_status = game_runner._game_command_status(
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
    provider = GameOperatorController(
        parse_operator_burn_plan("T= 0 s, 1.0 m/s R"),
        GameOperatorInputAdapter(source_id="test/operator", boot_id="test-boot"),
    )
    provider.observe_time(0.0)

    assert game_runner._operator_next_burn_status(provider) == "Next Burn: None"


def test_operator_mode_clears_and_skips_pilot_live_prediction_sync() -> None:
    class FakeDashboard:
        def __init__(self) -> None:
            self.calls: list[tuple[np.ndarray, float]] = []

        def set_live_prediction_burn(self, accel_ric_km_s2: np.ndarray, elapsed_s: float) -> None:
            self.calls.append((np.array(accel_ric_km_s2, dtype=float), float(elapsed_s)))

    dashboard = FakeDashboard()
    active_state = KeyboardCommandState(pitch=1.0)

    game_runner._clear_live_prediction_burn(dashboard)
    game_runner._sync_live_prediction_burn_for_mode(
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

    game_runner._sync_live_prediction_burn_for_mode(
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

    assert game_runner._speed_after_maneuver_input(200.0, ric_state, control_mode="ric_translation") == 10.0
    assert game_runner._speed_after_maneuver_input(100.0, ric_state, control_mode="ric_translation") == 10.0
    assert game_runner._speed_after_maneuver_input(50.0, ric_state, control_mode="ric_translation") == 10.0
    assert game_runner._speed_after_maneuver_input(25.0, ric_state, control_mode="ric_translation") == 10.0
    assert game_runner._speed_after_maneuver_input(10.0, ric_state, control_mode="ric_translation") == 10.0
    assert game_runner._speed_after_maneuver_input(5.0, ric_state, control_mode="ric_translation") == 5.0
    assert game_runner._speed_after_maneuver_input(200.0, coasting_state, control_mode="ric_translation") == 200.0
    assert game_runner._speed_after_maneuver_input(200.0, no_throttle_state, control_mode="ric_translation") == 200.0


def test_operator_mode_bypasses_manual_maneuver_speed_bookkeeping(tmp_path: Path) -> None:
    cfg_dict = _game_config(tmp_path)
    cfg_dict["metadata"]["game"]["two_rail_speed_control"] = True
    config = SimulationConfig.from_dict(cfg_dict)
    active_state = KeyboardCommandState(pitch=1.0, use_timing_accumulator=True)
    active_state.accumulate_timed_input(0.5, speed_multiple=100.0, control_mode="ric_translation")

    assert game_runner._manual_maneuver_active_for_mode("operator", active_state, control_mode="ric_translation") is False
    assert game_runner._pending_maneuver_sim_s_for_mode("operator", active_state, control_mode="ric_translation") == pytest.approx(
        0.0
    )
    assert (
        game_runner._effective_speed_multiple_for_mode(
            config,
            100.0,
            active_state,
            game_mode="operator",
            control_mode="ric_translation",
        )
        == pytest.approx(100.0)
    )

    assert game_runner._manual_maneuver_active_for_mode("pilot", active_state, control_mode="ric_translation") is True
    assert game_runner._pending_maneuver_sim_s_for_mode("pilot", active_state, control_mode="ric_translation") > 0.0
    assert (
        game_runner._effective_speed_multiple_for_mode(
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

    capped = game_runner._speed_after_maneuver_input(selected_speed, active_state, control_mode="ric_translation")
    unchanged = game_runner._speed_after_maneuver_input(capped, released_state, control_mode="ric_translation")

    assert capped == pytest.approx(10.0)
    assert unchanged == pytest.approx(10.0)


def test_maneuver_input_can_use_configured_control_speed_cap() -> None:
    ric_state = KeyboardCommandState(pitch=1.0)
    options = (10.0, 25.0, 50.0, 100.0, 200.0, 500.0)

    assert (
        game_runner._speed_after_maneuver_input(
            10.0,
            ric_state,
            control_mode="ric_translation",
            options=options,
            maneuver_control_speed_multiple=100.0,
        )
        == 10.0
    )
    assert (
        game_runner._speed_after_maneuver_input(
            50.0,
            ric_state,
            control_mode="ric_translation",
            options=options,
            maneuver_control_speed_multiple=100.0,
        )
        == 50.0
    )
    assert (
        game_runner._speed_after_maneuver_input(
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

    assert game_runner._speed_after_maneuver_input(200.0, rotate_state, control_mode="attitude_thrust") == 10.0
    assert game_runner._speed_after_maneuver_input(100.0, rotate_state, control_mode="attitude_thrust") == 10.0
    assert game_runner._speed_after_maneuver_input(50.0, firing_state, control_mode="attitude_thrust") == 10.0
    assert game_runner._speed_after_maneuver_input(25.0, firing_state, control_mode="attitude_thrust") == 10.0
    assert game_runner._speed_after_maneuver_input(10.0, firing_state, control_mode="attitude_thrust") == 10.0
    assert game_runner._speed_after_maneuver_input(5.0, firing_state, control_mode="attitude_thrust") == 5.0
    assert game_runner._speed_after_maneuver_input(200.0, coasting_state, control_mode="attitude_thrust") == 200.0


def test_dashboard_fps_drops_at_high_speed_unless_recording() -> None:
    assert game_runner._dashboard_fps_for_speed(10.0) == 60.0
    assert game_runner._dashboard_fps_for_speed(10.0, fps_cap=30.0) == 30.0
    assert game_runner._dashboard_fps_for_speed(10.0, static_screen=True) == game_runner.STATIC_DASHBOARD_FPS
    assert game_runner._dashboard_fps_for_speed(100.0, static_screen=True, high_speed_fps=60.0) == game_runner.STATIC_DASHBOARD_FPS
    assert game_runner._dashboard_fps_for_speed(50.0) == 45.0
    assert game_runner._dashboard_fps_for_speed(100.0) == 30.0
    assert game_runner._dashboard_fps_for_speed(200.0) == 30.0
    assert game_runner._dashboard_fps_for_speed(100.0, high_speed_fps=60.0) == 60.0
    assert game_runner._dashboard_fps_for_speed(100.0, high_speed_fps=60.0, high_speed_fps_max_multiple=100.0) == 60.0
    assert game_runner._dashboard_fps_for_speed(200.0, high_speed_fps=60.0, high_speed_fps_max_multiple=100.0) == 30.0
    assert game_runner._dashboard_fps_for_speed(100.0, fps_cap=45.0, high_speed_fps=60.0) == 45.0
    assert game_runner._dashboard_fps_for_speed(200.0, recording=True) == game_runner.GAME_RECORDING_FPS
    assert game_runner._dashboard_fps_for_speed(10.0, recording=True, fps_cap=20.0) == game_runner.GAME_RECORDING_FPS
    assert game_runner._dashboard_fps_for_speed(200.0, recording=True, recording_fps=24.0) == 24.0


def test_clip_recording_status_shows_elapsed_and_recent_messages(tmp_path: Path) -> None:
    class FakeRecorder:
        saved = False

    controller = game_recording_controller.GameClipRecordingController(
        config=SimulationConfig.from_dict(_game_config(tmp_path)),
        difficulty="easy",
        recorder=FakeRecorder(),
    )

    assert game_runner._clip_recording_status(controller, started_wall_s=10.0, now_wall_s=75.0) == (
        "REC 01:05  G/F9 discard  Enter save"
    )

    controller.recorder = None

    assert (
        game_runner._clip_recording_status(
            controller,
            started_wall_s=None,
            now_wall_s=12.0,
            status_message="Clip saved",
            status_until_wall_s=13.0,
        )
        == "Clip saved"
    )
    assert (
        game_runner._clip_recording_status(
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

    score = game_runner._step_game_attempt(
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

    score = game_runner._step_game_attempt(
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
    assert game_runner._split_game_step_dt(0.75, max_step_dt_s=1.0) == pytest.approx((0.75,))
    assert game_runner._split_game_step_dt(2.5, max_step_dt_s=1.0) == pytest.approx((1.0, 1.0, 0.5))


def test_operator_step_split_starts_short_impulse_at_tag() -> None:
    provider = GameOperatorController(
        OperatorBurnPlan(burns=(OperatorBurn(time_s=10.0, delta_v_ric_m_s=(1.0, 0.0, 0.0)),)),
        GameOperatorInputAdapter(source_id="test/operator", boot_id="test-boot"),
        impulse_duration_s=0.01,
    )

    chunks = game_runner._split_game_step_dt(
        1.0,
        current_time_s=9.5,
        operator_command_provider=provider,
    )

    assert chunks == pytest.approx((0.5, provider.impulse_duration_s, 0.5 - provider.impulse_duration_s))


def test_realtime_steps_due_supports_multi_step_catchup_for_100x() -> None:
    steps, next_wall = game_runner._realtime_steps_due(now_s=10.016, last_step_wall_s=10.0, wall_step_s=0.01)

    assert steps == 1
    assert np.isclose(next_wall, 10.01)

    steps, next_wall = game_runner._realtime_steps_due(now_s=10.033, last_step_wall_s=10.01, wall_step_s=0.01)

    assert steps == 2
    assert np.isclose(next_wall, 10.03)


def test_realtime_steps_due_caps_stall_catchup() -> None:
    steps, next_wall = game_runner._realtime_steps_due(now_s=20.0, last_step_wall_s=10.0, wall_step_s=0.01, max_steps=12)

    assert steps == 12
    assert next_wall == 20.0


def test_cw_coast_state_zero_time_returns_initial_state() -> None:
    x0 = np.array([0.1, -1.0, 0.2, 0.0, 0.001, -0.001], dtype=float)

    out = game_pygame_dashboard._cw_coast_state(x0, 0.0, 0.001)

    assert np.allclose(out, x0)


def test_cw_coast_states_matches_scalar_propagation() -> None:
    x0 = np.array([0.1, -1.0, 0.2, 0.0002, 0.001, -0.001], dtype=float)
    times = np.linspace(0.0, 120.0, 25)
    expected = np.vstack([game_pygame_dashboard._cw_coast_state(x0, float(t), 0.001) for t in times])

    out = game_pygame_dashboard._cw_coast_states(x0, times, 0.001)

    np.testing.assert_allclose(out, expected)


def test_cw_forced_state_advances_short_visual_burn() -> None:
    x0 = np.zeros(6, dtype=float)
    accel = np.array([1.0e-5, 0.0, 0.0], dtype=float)

    out = game_pygame_dashboard._cw_forced_state(x0, accel, 0.5, 0.001)

    assert out[0] > 0.0
    assert out[3] == pytest.approx(5.0e-6, rel=1.0e-3)


def test_live_prediction_accel_ric_matches_manual_translation_scaling() -> None:
    state = KeyboardCommandState(pitch=1.0, yaw=1.0, roll=0.0, throttle=0.5)

    accel = game_runner._live_prediction_accel_ric(
        state,
        control_mode="ric_translation",
        max_accel_km_s2=2.0e-5,
    )

    assert np.linalg.norm(accel) == pytest.approx(1.0e-5)
    assert accel[0] == pytest.approx(accel[1])


def test_live_prediction_accel_ric_clears_when_paused() -> None:
    state = KeyboardCommandState(pitch=1.0, yaw=0.0, roll=0.0, throttle=1.0, paused=True)

    accel = game_runner._live_prediction_accel_ric(
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

    accel, elapsed = game_runner._live_prediction_burn(
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

    accel, elapsed = game_runner._live_prediction_burn(
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

    accel, elapsed = game_runner._live_prediction_burn(
        state,
        control_mode="ric_translation",
        max_accel_km_s2=2.0e-5,
        elapsed_wall_s=0.75,
        speed_multiple=10.0,
        dt_s=1.0,
    )

    np.testing.assert_allclose(accel, np.zeros(3, dtype=float))
    assert elapsed == pytest.approx(0.0)
