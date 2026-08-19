from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import run_game as game_entrypoint
from sim.api import SimulationConfig
from sim.game import runner as game_runner
from sim.game.launcher import GameLaunchSelection
from sim.game.presentation import (
    PRESENTATION_QUALITIES,
    PresentationFrameController,
    PresentationSettings,
    create_presentation_controller,
    normalize_presentation_mode,
)
from sim.game.pygame_dashboard import PygameRPODashboard


def _minimal_game_config() -> SimulationConfig:
    return SimulationConfig.from_dict(
        {
            "scenario_name": "presentation_settings",
            "metadata": {"game": {}},
            "objects": {
                "sat": {
                    "kind": "satellite",
                    "initial_state": {
                        "position_eci_km": [7000.0, 0.0, 0.0],
                        "velocity_eci_km_s": [0.0, 7.5, 0.0],
                    },
                    "flight_software": {
                        "stack": "fsw.orbit_reference",
                        "hardware_profile": "hardware.ideal_wrench.v1",
                    },
                }
            },
            "simulator": {"duration_s": 1.0, "dt_s": 1.0},
        }
    )


def test_existing_presentation_architecture_remains_default() -> None:
    settings = game_runner._game_presentation_settings(_minimal_game_config())

    assert settings.mode == "compatibility"
    assert settings.enabled is False
    assert game_runner.run_game_mode.__kwdefaults__["presentation_mode"] is None


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["run_game.py"], "auto"),
        (["run_game.py", "--presentation-mode", "standard"], "standard"),
    ],
)
def test_level_selector_graphics_mode_reaches_game_and_allows_cli_override(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    expected: str,
) -> None:
    selection = GameLaunchSelection(path=Path("level.yaml"), difficulty="easy", presentation_mode="auto")
    selections = iter((selection, None))
    captured: dict[str, object] = {}

    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(game_entrypoint, "choose_game_launch", lambda **_kwargs: next(selections))

    def fake_run_game_mode(_path: Path, **kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(level_passed=False, arcade_score=0, mode="pilot")

    monkeypatch.setattr(game_entrypoint, "run_game_mode", fake_run_game_mode)

    game_entrypoint.main()

    assert captured["presentation_mode"] == expected


def test_candidate_auto_vsync_uses_software_pacing_unless_explicitly_enabled() -> None:
    assert PresentationSettings(mode="auto").requests_vsync is False
    assert PresentationSettings(mode="auto", vsync="off").requests_vsync is False
    assert PresentationSettings(mode="auto", vsync="on").requests_vsync is True
    assert PresentationSettings(mode="compatibility", vsync="on").requests_vsync is False


def test_presentation_settings_can_come_from_game_metadata_or_launch_override() -> None:
    root = _minimal_game_config().to_dict()
    root["metadata"]["game"]["presentation"] = {
        "mode": "auto",
        "fps_cap": 90.0,
        "vsync": "off",
        "diagnostics": True,
        "high_refresh_ceiling_fps": 144.0,
        "refresh_rate_hz": 75.0,
    }
    config = SimulationConfig.from_dict(root)

    configured = game_runner._game_presentation_settings(config)
    overridden = game_runner._game_presentation_settings(
        config,
        mode="standard",
        fps_cap=72.0,
        vsync="on",
        diagnostics=False,
    )

    assert configured == PresentationSettings(
        mode="auto",
        fps_cap=90.0,
        vsync="off",
        diagnostics=True,
        high_refresh_ceiling_fps=144.0,
        refresh_rate_hz=75.0,
    )
    assert overridden.mode == "standard"
    assert overridden.fps_cap == pytest.approx(72.0)
    assert overridden.vsync == "on"
    assert overridden.diagnostics is False


@pytest.mark.parametrize("bad", ["", "legacy", "adaptive-ish"])
def test_unknown_presentation_modes_are_rejected(bad: str) -> None:
    if bad == "":
        assert normalize_presentation_mode(bad) == "compatibility"
    else:
        with pytest.raises(ValueError, match="presentation mode"):
            normalize_presentation_mode(bad)


def test_presentation_modes_choose_display_rates_without_changing_compatibility_policy() -> None:
    compatibility = PresentationFrameController(PresentationSettings(mode="compatibility"), display_refresh_hz=144.0)
    standard = PresentationFrameController(PresentationSettings(mode="standard"), display_refresh_hz=144.0)
    high = PresentationFrameController(PresentationSettings(mode="high_refresh"), display_refresh_hz=144.0)
    automatic = PresentationFrameController(PresentationSettings(mode="auto"), display_refresh_hz=75.0)

    assert compatibility.target_fps(
        compatibility_fps=30.0,
        recording=False,
        recording_fps=30.0,
        static_screen=False,
    ) == pytest.approx(30.0)
    assert standard.target_fps(
        compatibility_fps=30.0,
        recording=False,
        recording_fps=30.0,
        static_screen=False,
    ) == pytest.approx(60.0)
    assert high.target_fps(
        compatibility_fps=30.0,
        recording=False,
        recording_fps=30.0,
        static_screen=False,
    ) == pytest.approx(120.0)
    assert automatic.target_fps(
        compatibility_fps=30.0,
        recording=False,
        recording_fps=30.0,
        static_screen=False,
    ) == pytest.approx(75.0)
    assert high.target_fps(
        compatibility_fps=15.0,
        recording=False,
        recording_fps=30.0,
        static_screen=True,
    ) == pytest.approx(15.0)
    assert high.target_fps(
        compatibility_fps=30.0,
        recording=True,
        recording_fps=24.0,
        static_screen=False,
    ) == pytest.approx(24.0)


def test_high_refresh_mode_targets_ceiling_when_pygame_cannot_detect_refresh() -> None:
    screen = type("Screen", (), {"get_size": lambda self: (1280, 720)})()
    dashboard = type(
        "Dashboard",
        (),
        {
            "screen": screen,
            "fullscreen": True,
            "presentation_vsync_active": False,
        },
    )()
    pygame = type("Pygame", (), {"display": object()})()

    controller = create_presentation_controller(
        pygame,
        dashboard,
        PresentationSettings(mode="high_refresh"),
    )

    assert controller is not None
    assert controller.display_refresh_hz == pytest.approx(120.0)


def test_auto_quality_degrades_and_recovers_with_hysteresis() -> None:
    controller = PresentationFrameController(PresentationSettings(mode="auto"), display_refresh_hz=120.0)
    controller.target_fps(
        compatibility_fps=30.0,
        recording=False,
        recording_fps=30.0,
        static_screen=False,
    )
    controller._last_quality_change_s -= 10.0
    controller._frame_work_s.extend([0.020] * controller._MIN_ADAPTIVE_SAMPLES)

    controller._adapt_if_needed()

    assert controller.quality.name == PRESENTATION_QUALITIES[1].name
    controller._frame_work_s.clear()
    controller._frame_work_s.extend([0.001] * controller._MIN_ADAPTIVE_SAMPLES)
    controller._last_quality_change_s -= 10.0
    controller._adapt_if_needed()
    assert controller.quality.name == PRESENTATION_QUALITIES[0].name
    assert [item["reason"] for item in controller.summary()["quality_transitions"]] == [
        "frame_budget_exceeded",
        "sustained_headroom",
    ]


def test_machine_readable_presentation_diagnostics(tmp_path: Path) -> None:
    output = tmp_path / "presentation.json"
    controller = PresentationFrameController(
        PresentationSettings(mode="standard", diagnostics=True, diagnostics_output=output),
        display_refresh_hz=60.0,
        display_size=(1280, 720),
    )
    controller.record_draw(0.004)
    controller.record_simulation_step(0.002)
    controller.record_prediction_recompute(0.003)
    controller.record_projection(horizon_s=0.5, cap_hit=True, computation_s=0.0005)
    controller.record_reconciliation_error(0.001)
    controller.observe_frame(work_s=0.007, authoritative_steps=1, snapshot_age_s=0.01)

    assert controller.write_summary() == output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "oel.game.presentation_diagnostics.v1"
    assert payload["presentation_mode"] == "standard"
    assert payload["display_size"] == [1280, 720]
    assert payload["simulation_step_s"]["samples"] == 1
    assert payload["prediction_recompute_s"]["samples"] == 1
    assert payload["projection_compute_s"]["samples"] == 1
    assert payload["projection_cap_hit_fraction"] == pytest.approx(1.0)
    assert payload["steps_per_frame"]["median"] == pytest.approx(1.0)


def test_backlog_diagnostics_preserve_existing_realtime_step_decision() -> None:
    legacy = game_runner._realtime_steps_due(
        now_s=2.0,
        last_step_wall_s=0.0,
        wall_step_s=0.1,
    )
    instrumented = game_runner._realtime_steps_due_with_backlog(
        now_s=2.0,
        last_step_wall_s=0.0,
        wall_step_s=0.1,
    )

    assert instrumented[:2] == legacy
    assert instrumented[2] == 7


def test_candidate_scheduler_prevents_high_speed_catchup_spiral() -> None:
    controller = PresentationFrameController(
        PresentationSettings(mode="auto", refresh_rate_hz=120.0),
        display_refresh_hz=120.0,
    )
    controller.target_fps(
        compatibility_fps=30.0,
        recording=False,
        recording_fps=30.0,
        static_screen=False,
    )

    assert controller.authoritative_step_limit(wall_step_s=0.01) == 1
    steps, next_wall, discarded = game_runner._realtime_steps_due_with_backlog(
        now_s=1.0,
        last_step_wall_s=0.0,
        wall_step_s=0.01,
        max_steps=controller.authoritative_step_limit(wall_step_s=0.01),
    )

    assert steps == 1
    assert next_wall == pytest.approx(1.0)
    assert discarded == 98


def test_candidate_scheduler_uses_measured_physics_cost_for_active_control() -> None:
    controller = PresentationFrameController(
        PresentationSettings(mode="auto", refresh_rate_hz=120.0),
        display_refresh_hz=120.0,
    )
    controller.target_fps(
        compatibility_fps=30.0,
        recording=False,
        recording_fps=30.0,
        static_screen=False,
    )
    controller.record_draw(0.003)
    for _ in range(45):
        controller.record_simulation_step(0.014)

    assert controller.authoritative_step_limit(wall_step_s=0.001) == 1
    summary = controller.summary()
    assert summary["scheduler_step_limit"] == 1
    assert summary["scheduler_compute_limited_frames"] == 1

    slow = PresentationFrameController(
        PresentationSettings(mode="auto", refresh_rate_hz=120.0),
        display_refresh_hz=120.0,
    )
    slow.target_fps(
        compatibility_fps=30.0,
        recording=False,
        recording_fps=30.0,
        static_screen=False,
    )
    slow.record_draw(0.010)
    slow.record_simulation_step(0.100)
    assert slow.authoritative_step_limit(wall_step_s=0.001) == 1


def _bare_dashboard(*, mode: str) -> PygameRPODashboard:
    dashboard = object.__new__(PygameRPODashboard)
    dashboard.presentation_mode = mode
    dashboard.visual_extrapolation_enabled = True
    dashboard._render_motion_enabled = True
    dashboard.sample_wall_s = [100.0]
    dashboard._render_wall_time_s = 100.05
    dashboard._render_speed_multiple = 10.0
    dashboard.t_s = [0.0, 0.5]
    dashboard.mean_motion_rad_s = 0.001
    dashboard.coast_prediction_model = "hcw"
    dashboard.relative_frame = "ric"
    dashboard.live_prediction_accel_ric_km_s2 = np.zeros(3, dtype=float)
    dashboard.burn_marker_threshold_km_s2 = 1.0e-12
    dashboard._presentation_last_states = {}
    dashboard._presentation_reconciliations = {}
    dashboard.presentation_controller = None
    return dashboard


def test_new_visual_path_uses_trajectory_projection_without_mutating_authoritative_rows() -> None:
    dashboard = _bare_dashboard(mode="standard")
    rows = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, -0.002, 0.0],
            [1.0, -0.001, 0.0, 0.0, -0.002, 0.0],
        ],
        dtype=float,
    )
    original = rows.copy()

    visual = dashboard._visual_state_rows(rows, series_name="rel")

    np.testing.assert_array_equal(rows, original)
    np.testing.assert_array_equal(visual[0], original[0])
    assert np.all(np.isfinite(visual))
    assert not np.array_equal(visual[-1], original[-1])


def test_analytical_projection_retains_dense_curve_at_low_quality() -> None:
    dashboard = _bare_dashboard(mode="standard")
    dashboard.presentation_ghost_draw_points = 36

    assert dashboard._presentation_ghost_draw_points() == 180

    dashboard.coast_prediction_model = "cr3bp"
    assert dashboard._presentation_ghost_draw_points() == 36


def test_candidate_dashed_projection_keeps_dash_phase_across_vertices() -> None:
    calls: list[tuple[tuple[int, int], tuple[int, int]]] = []

    class FakeDraw:
        @staticmethod
        def line(screen, color, start, end, *, width):
            del screen, color, width
            calls.append((start, end))

    dashboard = object.__new__(PygameRPODashboard)
    dashboard.presentation_mode = "standard"
    dashboard.pygame = type("FakePygame", (), {"draw": FakeDraw})()
    dashboard.screen = object()

    dashboard._draw_polyline_dashed(
        [(0, 0), (10, 0), (10, 10)],
        color=(1, 2, 3),
        dash_px=8,
        gap_px=6,
        width=2,
    )

    assert calls == [((0, 0), (8, 0)), ((10, 4), (10, 10))]


def test_compatibility_visual_path_retains_one_second_linear_cap() -> None:
    dashboard = _bare_dashboard(mode="compatibility")
    dashboard._render_wall_time_s = 101.0
    dashboard._render_speed_multiple = 100.0
    dashboard.visual_extrapolation_max_sim_s = 1.0
    rows = np.array([[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]], dtype=float)

    visual = dashboard._visual_state_rows(rows)

    np.testing.assert_allclose(visual[-1, :3], [1.05, 2.1, 3.15], rtol=0.0, atol=1.0e-15)


def test_discontinuity_invalidates_visual_reconciliation() -> None:
    dashboard = _bare_dashboard(mode="standard")
    dashboard._presentation_last_states["rel"] = np.array([1.01, 0.0, 0.0, 0.0, 0.0, 0.0])
    dashboard._presentation_reconciliations["rel"] = {"offset_km": np.ones(3), "started_wall_s": 0.0}

    dashboard._begin_presentation_reconciliation(
        "rel",
        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        discontinuity=True,
    )

    assert "rel" not in dashboard._presentation_reconciliations


def test_reconciliation_measures_model_error_at_new_authoritative_time() -> None:
    dashboard = _bare_dashboard(mode="standard")
    dashboard.t_s = [0.0]
    dashboard.rel_hist = [np.array([1.0, 0.0, 0.0, 0.0, -0.002, 0.0])]
    dashboard.target_rel_hist = []
    dashboard.target_reference_rel_hist = []
    dashboard._presentation_last_states["rel"] = np.array([9.0, 9.0, 0.0, 0.0, 0.0, 0.0])
    predicted = dashboard._presentation_project_state(
        np.asarray(dashboard.rel_hist),
        elapsed_sim_s=10.0,
        series_name="rel",
    )
    authoritative = predicted.copy()
    authoritative[0] += 0.001

    dashboard._begin_presentation_reconciliation(
        "rel",
        authoritative,
        discontinuity=False,
        authoritative_time_s=10.0,
    )

    offset = dashboard._presentation_reconciliations["rel"]["offset_km"]
    np.testing.assert_allclose(offset, [-0.001, 0.0, 0.0], rtol=0.0, atol=1.0e-12)


def test_cislunar_presentation_builds_and_reuses_display_only_trajectory() -> None:
    dashboard = _bare_dashboard(mode="standard")
    dashboard.coast_prediction_model = "cr3bp"
    dashboard.relative_frame = "cislunar_l1"
    dashboard.t_s = [0.0, 10.0]
    dashboard.presentation_ghost_draw_points = 36
    dashboard._presentation_trajectory_cache = {}
    state = np.array([25.0, -10.0, 2.0, 0.001, -0.002, 0.0], dtype=float)

    first = dashboard._presentation_cr3bp_state(state, elapsed_sim_s=2.0, series_name="rel")
    cached = dashboard._presentation_trajectory_cache["rel"]
    second = dashboard._presentation_cr3bp_state(state, elapsed_sim_s=3.0, series_name="rel")

    assert first is not None and second is not None
    assert np.all(np.isfinite(first)) and np.all(np.isfinite(second))
    assert dashboard._presentation_trajectory_cache["rel"] is cached
    assert not np.array_equal(first, second)
