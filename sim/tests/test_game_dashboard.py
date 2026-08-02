from __future__ import annotations

# These owner-aligned tests share deterministic builders and compatibility
# imports from the adjacent support module.
# ruff: noqa: F403, F405
from sim.tests.game_mode_test_support import *


def test_live_prediction_burn_matches_manual_duty_cycle_shape() -> None:
    state = KeyboardCommandState(pitch=1.0, yaw=1.0, throttle=1.0)
    state.use_timing_accumulator = True
    state.pitch_sim_s = 2.0
    state.yaw_sim_s = 1.0

    accel, elapsed = game_runner._live_prediction_burn(
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

    accel, elapsed = game_runner._live_prediction_burn(
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
    assert game_pygame_dashboard._coast_prediction_model_key("HCW") == "hcw"
    assert game_pygame_dashboard._coast_prediction_model_key("elliptic") == "elliptic_linear"
    assert game_pygame_dashboard._coast_prediction_model_key("Tschauner-Hempel") == "tschauner_hempel"
    assert game_pygame_dashboard._cr3bp_projection_mode_key("STM") == "linearized"
    assert game_pygame_dashboard._cr3bp_projection_mode_key("linearized") == "linearized"
    assert game_pygame_dashboard._cr3bp_projection_mode_key("nonlinear") == "nonlinear"


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
    dashboard.target_true_anomaly_deg = game_pygame_dashboard._true_anomaly_deg_from_state(
        np.hstack((target_r, target_v))
    )

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
    dashboard.camera_mode = game_runner._game_camera_mode(config)
    dashboard.camera_rule_mode = game_runner._game_camera_rule_mode(config)
    dashboard.relative_frame = game_runner._game_relative_frame(config)
    dashboard.target_centered_plot_planes = game_runner._game_target_centered_plot_planes(config)
    dashboard.target_centered_plot_axes = game_runner._game_target_centered_plot_axes(config)
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
    assert (
        game_pygame_dashboard._should_draw_cislunar_moon_background(relative_frame="cislunar_l1", x_axis=1, y_axis=2)
        is True
    )
    assert (
        game_pygame_dashboard._should_draw_cislunar_moon_background(relative_frame="cislunar_l1", x_axis=1, y_axis=0)
        is False
    )
    assert (
        game_pygame_dashboard._should_draw_cislunar_moon_background(relative_frame="ric", x_axis=1, y_axis=2) is False
    )

    rect = game_pygame_dashboard._scaled_body_rect_tuple(
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

    lines = ("Result        Forbidden region violated: lower radial rail of V-bar U, lower radial floor in RC.",)

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
        guided_tutorial_burns=(GuidedTutorialBurnConfig(name="radial", axis="radial", sign=1, delta_v_m_s=0.01),),
    )
    runtime = game_runner.GuidedTutorialRuntime()

    assert game_runner._pause_teaching_overlay_enabled(game_runner.GamePhase.PAUSED, training_cfg, runtime) is False
    assert game_runner._pause_teaching_overlay_enabled(
        game_runner.GamePhase.PAUSED,
        replace(training_cfg, guided_tutorial_burns=()),
        runtime,
    )
    runtime.awaiting_speed_step = True
    assert (
        game_runner._pause_teaching_overlay_enabled(
            game_runner.GamePhase.PAUSED,
            replace(training_cfg, guided_tutorial_burns=()),
            runtime,
        )
        is False
    )


def test_elliptic_linear_coast_matches_hcw_for_circular_chief() -> None:
    rel0 = np.array([0.1, -1.0, 0.2, 0.0, 0.001, -0.001], dtype=float)
    chief = np.array([7000.0, 0.0, 0.0, 0.0, np.sqrt(398600.4418 / 7000.0), 0.0], dtype=float)
    n = np.sqrt(398600.4418 / 7000.0**3)
    times = np.array([0.0, 30.0, 60.0, 120.0], dtype=float)

    elliptic = game_pygame_dashboard._elliptic_linear_coast_states(rel0, times, chief)
    circular = np.vstack([game_pygame_dashboard._cw_coast_state(rel0, float(t), n) for t in times])

    assert np.allclose(elliptic, circular, atol=2.0e-5)


def test_coast_prediction_difficulty_maps_to_orbit_fraction() -> None:
    assert game_runner._coast_prediction_orbit_fraction("easy") == 1.0
    assert game_runner._coast_prediction_orbit_fraction("medium") == 0.5
    assert game_runner._coast_prediction_orbit_fraction("hard") == 0.25
    assert game_runner._coast_prediction_orbit_fraction("extreme") == 0.0


def test_operator_difficulty_maps_to_actuator_error_and_full_projection() -> None:
    assert game_runner._operator_actuator_error_fraction("easy") == pytest.approx(0.0)
    assert game_runner._operator_actuator_error_fraction("medium") == pytest.approx(0.01)
    assert game_runner._operator_actuator_error_fraction("hard") == pytest.approx(0.025)
    assert game_runner._operator_actuator_error_fraction("extreme") == pytest.approx(0.05)
    assert game_runner._operator_coast_prediction_orbit_fraction("operator", "extreme") == pytest.approx(1.0)
    assert game_runner._operator_coast_prediction_orbit_fraction("pilot", "extreme") == pytest.approx(0.0)


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

    sampled = game_pygame_dashboard._sample_rows(rows, 120)

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
    moon_view_xy = game_pygame_dashboard._project_moon_rotating_yz_to_plane(orbit[:, :3] - moon[:3])

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
    chaser = game_pygame_dashboard._moon_ric_rect_state_to_cr3bp(
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

    assert game_pygame_dashboard._cr3bp_reference_cache_valid(reference, current, elapsed_s=10.0) is True
    assert game_pygame_dashboard._cr3bp_reference_cache_valid(reference, current) is False


def test_dashboard_eci_projection_uses_target_orbit_plane() -> None:
    dashboard = object.__new__(PygameRPODashboard)
    target_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0], dtype=float)

    basis = dashboard._eci_target_plane_basis(target_state)

    assert basis is not None
    i_hat, r_hat, c_hat = basis
    assert i_hat == pytest.approx([0.0, 1.0, 0.0])
    assert r_hat == pytest.approx([1.0, 0.0, 0.0])
    assert c_hat == pytest.approx([0.0, 0.0, 1.0])
    projected = game_pygame_dashboard._project_eci_positions_to_plane(
        np.array([[7000.0, 0.0, 0.0], [7000.0, 1.0, 0.5]], dtype=float),
        x_hat=i_hat,
        y_hat=r_hat,
    )
    np.testing.assert_allclose(projected, np.array([[0.0, 7000.0], [1.0, 7000.0]], dtype=float))


def test_satellite_marker_size_uses_physical_plot_scale() -> None:
    assert game_pygame_dashboard._satellite_marker_size_px(100.0, 100.0) == 1
    assert game_pygame_dashboard._satellite_marker_size_px(1000.0, 1000.0) == 6
    assert game_pygame_dashboard._satellite_marker_size_px(5000.0, 5000.0) == 30
    assert game_pygame_dashboard._satellite_marker_size_px(100000.0, 100000.0) == 600
    assert game_pygame_dashboard._satellite_marker_size_px(100.0, 100.0, diameter_km=0.05) == 5
    assert game_pygame_dashboard._satellite_marker_size_px(100.0, 100.0, diameter_km=0.11) == 11
    assert game_pygame_dashboard._satellite_marker_size_px(100.0, 100.0, diameter_km=0.12) == 12


def test_satellite_marker_reticle_scales_with_sprite_size() -> None:
    assert game_pygame_dashboard._satellite_marker_reticle_radii_px(0) == (0, 0)
    assert game_pygame_dashboard._satellite_marker_reticle_radii_px(5) == (2, 4)
    assert game_pygame_dashboard._satellite_marker_reticle_radii_px(11) == (2, 4)
    assert game_pygame_dashboard._satellite_marker_reticle_radii_px(20) == (2, 4)
    assert game_pygame_dashboard._satellite_marker_reticle_radii_px(60) == (3, 6)


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


def test_dashboard_dashed_polyline_preserves_legacy_pixel_segments() -> None:
    points = [(3, 5), (44, 29), (44, 29), (9, -8)]
    expected: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for start, end in zip(points[:-1], points[1:], strict=False):
        p0 = np.array(start, dtype=float)
        p1 = np.array(end, dtype=float)
        segment = p1 - p0
        length = float(np.linalg.norm(segment))
        if length <= 0.0:
            continue
        direction = segment / length
        pos = 0.0
        while pos < length:
            segment_start = p0 + direction * pos
            segment_end = p0 + direction * min(pos + 8, length)
            expected.append(
                (
                    (int(segment_start[0]), int(segment_start[1])),
                    (int(segment_end[0]), int(segment_end[1])),
                )
            )
            pos += 14

    calls: list[tuple[tuple[int, int], tuple[int, int]]] = []

    class FakeDraw:
        @staticmethod
        def line(
            screen: object,
            color: tuple[int, int, int],
            start: tuple[int, int],
            end: tuple[int, int],
            *,
            width: int,
        ) -> None:
            assert screen == "screen"
            assert color == (1, 2, 3)
            assert width == 2
            calls.append((start, end))

    dashboard = object.__new__(PygameRPODashboard)
    dashboard.pygame = type("FakePygame", (), {"draw": FakeDraw})()
    dashboard.screen = "screen"

    dashboard._draw_polyline_dashed(points, color=(1, 2, 3), dash_px=8, gap_px=6, width=2)

    assert calls == expected


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
        history = game_pygame_dashboard._history_array_tail(history, row, width=6, max_rows=3)
    dashboard._rel_array = history

    rows = game_pygame_dashboard._dashboard_history_array(dashboard, "_rel_array", [], width=6)

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
    expected_seed = game_pygame_dashboard._cw_forced_state(
        rel[0],
        dashboard.live_prediction_accel_ric_km_s2,
        dashboard.live_prediction_elapsed_s,
        dashboard.mean_motion_rad_s,
    )

    assert visual_latest[:3] == pytest.approx([1.15, 1.7, 3.45])
    assert seeds[0][1] == pytest.approx(expected_seed)
    visual_seed = game_pygame_dashboard._cw_forced_state(
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
    times = game_pygame_dashboard._front_loaded_prediction_times(259200.0, 1.0, max_points=120)

    assert times.shape == (120,)
    np.testing.assert_allclose(times[:6], np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float))
    assert times[-1] == pytest.approx(259200.0)
    assert np.all(np.diff(times) > 0.0)
    assert np.max(np.diff(times[:80])) == pytest.approx(1.0)


def test_front_loaded_prediction_times_support_long_cislunar_horizon() -> None:
    times = game_pygame_dashboard._front_loaded_prediction_times(604800.0, 1.0, max_points=120)

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
    assert not np.allclose(prediction[-1], game_pygame_dashboard._cw_coast_state(rel0, 120.0, 0.001))


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
    dashboard.reference_state_eci = game_pygame_dashboard._two_body_coast_state(reference, 10.0)
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
    maneuvered_reference = game_pygame_dashboard._two_body_coast_state(reference, 10.0)
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
    config_path = Path(__file__).resolve().parents[1] / "game" / "configs" / "game_training_rpo_00_tutorial.yaml"
    config = SimulationConfig.from_yaml(config_path)
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.camera_mode = game_runner._game_camera_mode(config)
    dashboard.target_centered_plot_planes = game_runner._game_target_centered_plot_planes(config)
    dashboard.target_centered_plot_axes = game_runner._game_target_centered_plot_axes(config)
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
    dashboard.plot_prediction_zoom_max_span_km = game_runner._game_plot_prediction_zoom_max_span_km(sim_cfg)
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

    assert game_runner._game_plot_prediction_in_zoom(sim_cfg) is True
    assert game_runner._game_plot_prediction_zoom_max_span_km(sim_cfg) == pytest.approx(8.0)
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
    dashboard.plot_overlays_in_zoom = game_runner._game_plot_overlays_in_zoom(sim_cfg)
    dashboard.plot_overlays_in_zoom_by_plane = game_runner._game_plot_overlays_in_zoom_by_plane(sim_cfg)

    ri_min_span = dashboard._minimum_plot_span_km(x_axis=1, y_axis=0, offset=np.zeros(3, dtype=float))
    rc_min_span = dashboard._minimum_plot_span_km(x_axis=2, y_axis=0, offset=np.zeros(3, dtype=float))

    assert game_runner._game_plot_overlays_in_zoom(sim_cfg) is True
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
    assert (
        game_pygame_dashboard._should_draw_nominal_nmt(dashboard._nmt_points(), dashboard._nmt_boundary_points())
        is False
    )
    assert game_pygame_dashboard._should_draw_nominal_nmt(dashboard._nmt_points(), ()) is True


def test_level2_plot_scale_can_ignore_forbidden_region_zoom_extent() -> None:
    root = Path(__file__).resolve().parents[1] / "game" / "configs"
    sim_cfg = SimulationConfig.from_yaml(root / "game_training_rpo_02_vbar_approach.yaml")
    training_cfg = RPOTrainingConfig.from_metadata(dict(sim_cfg.scenario.metadata or {}))
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.forbidden_regions = training_cfg.forbidden_regions
    dashboard.approach_gates = ()
    dashboard.inspection_gates = ()
    dashboard.plot_overlays_in_zoom = game_runner._game_plot_overlays_in_zoom(sim_cfg)
    dashboard.plot_overlays_in_zoom_by_plane = game_runner._game_plot_overlays_in_zoom_by_plane(sim_cfg)
    dashboard.plot_axis_scale = game_runner._game_plot_axis_scale(sim_cfg)
    dashboard.plot_fixed_axis_half_span_km = game_runner._game_plot_fixed_axis_half_span_km(sim_cfg)
    dashboard.plot_equal_axis_scale_planes = game_runner._game_plot_equal_axis_scale_planes(sim_cfg)
    dashboard.proximity_ring_plot_planes = game_runner._game_proximity_ring_plot_planes(sim_cfg)

    ri_ignored_span = dashboard._minimum_plot_span_km(x_axis=1, y_axis=0, offset=np.zeros(3, dtype=float))
    rc_fixed_span = dashboard._minimum_plot_span_km(x_axis=2, y_axis=0, offset=np.zeros(3, dtype=float))
    dashboard.plot_overlays_in_zoom = True
    dashboard.plot_overlays_in_zoom_by_plane = {}
    ri_full_overlay_span = dashboard._minimum_plot_span_km(x_axis=1, y_axis=0, offset=np.zeros(3, dtype=float))

    assert game_runner._game_camera_mode(sim_cfg) == "target_pair"
    assert game_runner._game_plot_overlays_in_zoom(sim_cfg) is False
    assert game_runner._game_plot_overlays_in_zoom_by_plane(sim_cfg) == {"RC": True}
    assert dashboard._axis_scale_for_plane(x_axis=1, y_axis=0) == pytest.approx((1.2, 1.0))
    assert dashboard._axis_scale_for_plane(x_axis=2, y_axis=0) == pytest.approx((1.0, 1.0))
    assert game_runner._game_plot_fixed_axis_half_span_km(sim_cfg) == {"RI": (None, 0.75), "RC": (None, 0.75)}
    assert game_runner._game_target_centered_plot_axes(sim_cfg) == {"RI": ("y",)}
    assert game_runner._game_plot_equal_axis_scale_planes(sim_cfg) == ("RC",)
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
    assert game_runner._game_proximity_ring_plot_planes(sim_cfg) == ("RI",)
    assert dashboard._show_proximity_rings_for_plane(x_axis=1, y_axis=0) is True
    assert dashboard._show_proximity_rings_for_plane(x_axis=2, y_axis=0) is False
    assert ri_ignored_span == pytest.approx(0.005)
    assert rc_fixed_span > 5.0
    assert ri_full_overlay_span > 5.0
