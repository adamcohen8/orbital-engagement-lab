"""Sandbox camera geometry and real Pygame event regression checks."""

import numpy as np
import pytest

from sim.game.dashboard_3d import SandboxCamera


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_sandbox_camera_toggle_changes_rendered_ri_framing(monkeypatch, backend):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    pg = pytest.importorskip("pygame")
    if backend == "rust":
        pytest.importorskip("oel_rust_game")
        pytest.importorskip("oel_rust_orbit")
    from sim.api import SimulationConfig
    from sim.game.backend import configure_game_backend
    from sim.game.dashboard import PygameRPODashboard
    from sim.game.runner_config import (
        _game_camera_mode,
        _game_camera_rule_mode,
        _game_plot_prediction_full_trajectory_only,
        _game_plot_prediction_in_zoom,
        _game_target_centered_plot_planes,
    )

    source = SimulationConfig.from_yaml("sim/game/configs/game_training_rpo_sandbox.yaml")
    config = configure_game_backend(source, backend)
    dashboard = PygameRPODashboard(
        fullscreen=False,
        numeric_backend=backend,
        camera_mode=_game_camera_mode(config),
        camera_rule_mode=_game_camera_rule_mode(config),
        plot_prediction_in_zoom=_game_plot_prediction_in_zoom(config),
        plot_prediction_full_trajectory_only=_game_plot_prediction_full_trajectory_only(config),
        target_centered_plot_planes=_game_target_centered_plot_planes(config),
    )
    try:
        # A distant old trail and coast prediction previously held the same
        # RI zoom in both modes, despite the HUD reporting a successful toggle.
        rel = np.array([[0., -4., 0., 0., 0., 0.], [0., .05, 0., 0., 0., 0.]])
        target = np.zeros_like(rel)
        dashboard.rel_hist = list(rel)
        dashboard.target_rel_hist = list(target)
        dashboard.t_s = [0., 1.]
        dashboard._frame_cache = {
            "rel": rel,
            "target_rel": target,
            "target_reference_rel": target,
            "ghost": rel.copy(),
            "target_ghost": target,
            "nmt": np.empty((0, 3)),
            "nmt_bounds": (),
            "pixel_polyline_cache": {},
        }
        dashboard._frame_cache_dirty = False
        panel = pg.Rect(36, 124, 580, 360)

        def ri_scale():
            dashboard._draw_panel(panel, "RI Plane", 1, 0)
            return dashboard._frame_cache["plot_transforms"][(1, 0)]["scale_x"]

        full_scale = ri_scale()
        assert dashboard.toggle_camera_rule_mode() == "current_pair"
        assert dashboard._prediction_scales_current_camera() is False
        pair_scale = ri_scale()
        assert pair_scale > full_scale * 5
        assert dashboard.toggle_camera_rule_mode() == "full_trajectory"
        assert ri_scale() == pytest.approx(full_scale)
    finally:
        dashboard.close()


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_sandbox_game_loop_c_key_switches_camera_in_both_backends(tmp_path, monkeypatch, backend):
    from pathlib import Path

    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    pg = pytest.importorskip("pygame")
    if backend == "rust":
        pytest.importorskip("oel_rust_game")
        pytest.importorskip("oel_rust_orbit")
    from sim.game import game_loop
    from sim.game.pygame_dashboard import PygameRPODashboard
    from sim.game.tutorial_runtime import _sandbox_setup_from_config

    monkeypatch.setattr(
        game_loop, "_run_sandbox_setup_form", lambda dashboard, config, **kwargs: _sandbox_setup_from_config(config)
    )
    original = PygameRPODashboard.draw
    modes = []

    def draw(self, **kwargs):
        original(self, **kwargs)
        modes.append(self.camera_rule_mode)
        frame = len(modes)
        if frame == 1:
            pg.event.post(pg.event.Event(pg.KEYDOWN, key=pg.K_SPACE))
        elif frame in (2, 4):
            pg.event.post(pg.event.Event(pg.KEYDOWN, key=pg.K_c))
        elif frame >= 6:
            pg.event.post(pg.event.Event(pg.QUIT))

    monkeypatch.setattr(PygameRPODashboard, "draw", draw)
    game_loop.run_game_mode(
        Path(__file__).resolve().parents[1] / "game/configs/game_training_rpo_sandbox.yaml",
        backend=backend,
        music_enabled=False,
        debrief_output_dir=tmp_path,
    )
    assert modes[2] == "current_pair"
    assert modes[4] == "full_trajectory"


def test_orthographic_basis_preserves_equal_world_scale():
    camera = SandboxCamera()
    radial_on_screen = camera.basis() @ np.array([1.0, 0.0, 0.0])
    assert radial_on_screen[0] == 0.0
    assert radial_on_screen[1] > 0.0
    for yaw in (-2, 0, 1):
        for elevation in (-np.pi / 2, 0, np.pi / 2):
            camera.yaw, camera.elevation = yaw, elevation
            np.testing.assert_allclose(camera.basis() @ camera.basis().T, np.eye(2), atol=1e-14)


def test_temporary_speed_cap_restores_latest_selection_and_preserves_pause():
    camera = SandboxCamera(enabled=True, drag_button=1)
    assert camera.effective_speed(100) == 1
    assert camera.effective_speed(0) == 0
    camera.drag_button = 0
    camera.zoom_until = 10
    assert camera.effective_speed(100, 9.9) == 1
    assert camera.effective_speed(20, 10.1) == 20
    camera.cancel()
    assert camera.effective_speed(100) == 100


def test_mouse_navigation_toggle_focus_and_render(tmp_path, monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    pg = pytest.importorskip("pygame")
    from sim.game.dashboard import PygameRPODashboard

    dashboard = PygameRPODashboard(fullscreen=False, sandbox_3d_enabled=True)
    try:
        dashboard._frame_cache = {
            "rel": np.array([[0.5, -1, 0.3, 0, 0, 0], [0.7, -0.8, 0.4, 0, 0, 0]]),
            "target_rel": np.zeros((2, 6)),
            "ghost": np.array([[0.7, -0.8, 0.4], [0.9, 0, 0.6], [0.2, 0.8, -0.2]]),
        }
        rect = pg.Rect(36, 124, 1208, 464)
        dashboard._draw_camera_buttons(rect)

        def click(pos):
            return pg.event.Event(pg.MOUSEBUTTONDOWN, button=1, pos=pos)

        dashboard.handle_camera_event(click(dashboard._camera_buttons["3D"].center))
        camera = dashboard._sandbox_camera()
        assert camera.enabled
        dashboard._draw_sandbox_3d(rect)
        dashboard._draw_camera_buttons(rect)
        pos = dashboard._camera_plot.center
        dashboard.handle_camera_event(click(pos))
        assert dashboard.camera_effective_speed(100) == 1
        yaw = camera.yaw
        dashboard.handle_camera_event(pg.event.Event(pg.MOUSEMOTION, rel=(30, 10), pos=pos))
        assert camera.yaw != yaw
        dashboard.handle_camera_event(pg.event.Event(pg.MOUSEBUTTONUP, button=1, pos=pos))
        assert dashboard.camera_effective_speed(100) == 100
        focus = camera.focus.copy()
        dashboard.handle_camera_event(pg.event.Event(pg.MOUSEBUTTONDOWN, button=2, pos=pos))
        dashboard.handle_camera_event(pg.event.Event(pg.MOUSEMOTION, rel=(20, 10), pos=pos))
        assert not np.array_equal(camera.focus, focus)
        dashboard.handle_camera_event(pg.event.Event(pg.WINDOWFOCUSLOST))
        assert not camera.active()
        monkeypatch.setattr(pg.mouse, "get_pos", lambda: pos)
        span = camera.span
        dashboard.handle_camera_event(pg.event.Event(pg.MOUSEWHEEL, y=2))
        assert camera.span < span
        assert dashboard.camera_effective_speed(100) == 1
        dashboard.handle_camera_event(click(pos), blocked=True)
        assert not camera.active()
        saved = (camera.yaw, camera.span, camera.focus.copy())
        dashboard.handle_camera_event(click(dashboard._camera_buttons["2D"].center))
        dashboard._draw_camera_buttons(rect)
        dashboard.handle_camera_event(click(dashboard._camera_buttons["3D"].center))
        assert camera.yaw == saved[0] and camera.span == saved[1]
        np.testing.assert_array_equal(camera.focus, saved[2])
        dashboard._draw_sandbox_3d(rect)
        dashboard._draw_camera_buttons(rect)
        pg.image.save(dashboard.screen, str(tmp_path / "sandbox-3d.png"))
        for elevation in (0.0, np.pi / 2, -np.pi / 2):
            camera.elevation = elevation
            camera.focus += np.array([1.0, 2.0, 3.0])
            dashboard._draw_sandbox_3d(rect)
        dashboard.sandbox_3d_enabled = False
        assert not dashboard.handle_camera_event(click(pos))
        assert dashboard.camera_effective_speed(100) == 100
    finally:
        dashboard.close()


@pytest.mark.parametrize("scenario", ["game_training_rpo_sandbox.yaml", "game_training_rpo_03_rbar_approach.yaml"])
def test_game_loop_mouse_events_cap_and_restore_speed(tmp_path, monkeypatch, scenario):
    from pathlib import Path

    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    pg = pytest.importorskip("pygame")
    from sim.game import game_loop
    from sim.game.pygame_dashboard import PygameRPODashboard
    from sim.game.tutorial_runtime import _sandbox_setup_from_config

    monkeypatch.setattr(
        game_loop, "_run_sandbox_setup_form", lambda dashboard, config, **kwargs: _sandbox_setup_from_config(config)
    )
    original = PygameRPODashboard.draw
    speeds = []

    def draw(self, **kwargs):
        original(self, **kwargs)
        speeds.append((kwargs.get("speed_multiple"), kwargs.get("selected_speed_multiple")))
        frame = len(speeds)

        def post(kind, **attrs):
            pg.event.post(pg.event.Event(kind, **attrs))

        if frame == 1:
            post(pg.KEYDOWN, key=pg.K_SPACE)
        elif frame == 2:
            post(pg.MOUSEBUTTONDOWN, button=1, pos=self._camera_buttons["3D"].center)
        elif frame == 3:
            post(pg.MOUSEBUTTONDOWN, button=1, pos=self._camera_plot.center)
        elif frame in (4, 5):
            post(pg.MOUSEMOTION, rel=(12, 8), pos=self._camera_plot.center)
        elif frame == 6:
            post(pg.MOUSEBUTTONUP, button=1, pos=self._camera_plot.center)
        elif frame >= 8:
            post(pg.QUIT)

    monkeypatch.setattr(PygameRPODashboard, "draw", draw)
    game_loop.run_game_mode(
        Path(__file__).resolve().parents[1] / "game/configs" / scenario,
        music_enabled=False,
        speed_multiple=10,
        debrief_output_dir=tmp_path,
    )
    assert (1.0, 10.0) in speeds
    assert speeds[-1] == (10.0, 10.0)


@pytest.mark.parametrize("viewport", [(1200, 380), (400, 900), (600, 600)])
@pytest.mark.parametrize("yaw,elevation", [(-0.65, 0.5), (0.0, 0.0), (1.0, -0.7)])
def test_fit_uses_projected_viewport_bounds_with_padding(viewport, yaw, elevation):
    camera = SandboxCamera(yaw=yaw, elevation=elevation)
    points = np.array([[0.0, 0.0, 0.0], [2.0, -3.0, 1.0]])
    camera.fit(points, viewport)
    projected = (points - camera.focus) @ camera.basis().T
    pixels = projected * min(viewport) / camera.span
    occupancy = np.ptp(pixels, axis=0) / viewport
    assert np.max(occupancy) == pytest.approx(0.8)
    assert np.all(np.abs(pixels) <= np.asarray(viewport) * 0.4 + 1e-10)


def test_pan_keeps_both_satellites_inside_margin():
    camera = SandboxCamera()
    points = np.array([[0.0, 0.0, 0.0], [2.0, -3.0, 1.0]])
    viewport = np.array([1200.0, 380.0])
    camera.fit(points, viewport)
    for delta in ((10000, 10000), (-20000, -20000), (10000, -10000)):
        camera.pan(delta, points, viewport)
        pixels = (points - camera.focus) @ camera.basis().T * min(viewport) / camera.span
        assert np.all(np.abs(pixels) <= viewport / 2 - 32 + 1e-9)


def test_pan_allows_recovery_without_snapping_or_moving_farther_away():
    camera = SandboxCamera(yaw=0.0, elevation=0.0, span=10.0)
    points = np.array([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]])
    camera.focus = np.array([0.0, 20.0, 0.0])
    camera.pan((-100, 0), points, (600, 400))
    assert camera.focus[1] == 20.0
    camera.pan((100, 0), points, (600, 400))
    assert 1.0 < camera.focus[1] < 20.0
    camera.pan((10000, 0), points, (600, 400))
    pixels = (points - camera.focus) @ camera.basis().T * 40
    assert np.all(np.abs(pixels) <= np.array([300, 200]) - 32 + 1e-9)


def test_pan_can_recover_either_satellite_when_zoom_cannot_fit_both():
    camera = SandboxCamera(yaw=0.0, elevation=0.0, span=1.0)
    points = np.array([[0.0, -10.0, 0.0], [0.0, 10.0, 0.0]])
    for delta in ((100000, 0), (-200000, 0)):
        camera.pan(delta, points, (600, 400))
        pixels = (points - camera.focus) @ camera.basis().T * 400
        assert np.any(np.all(np.abs(pixels) <= np.array([300, 200]) - 32 + 1e-9, axis=1))


@pytest.mark.parametrize("viewport", [(1200, 380), (400, 900), (600, 600)])
def test_visibility_guard_handles_rotation_zoom_pan_and_spacecraft_motion(viewport):
    camera = SandboxCamera()
    points = np.array([[0.0, 0.0, 0.0], [2.0, -3.0, 1.0]])
    camera.fit(points, viewport)
    for yaw, elevation, focus, span, displacement in (
        (0.0, 0.0, [0.0, 0.0, 0.0], 0.01, [0.0, 0.0, 0.0]),
        (1.5, 1.2, [20.0, -30.0, 10.0], 0.2, [0.0, 0.0, 0.0]),
        (-2.0, -0.8, [0.0, 0.0, 0.0], 1.0, [100.0, -40.0, 50.0]),
    ):
        camera.yaw, camera.elevation = yaw, elevation
        camera.focus = np.array(focus)
        camera.span = span
        moved = points.copy()
        moved[1] += displacement
        camera.keep_visible(moved, viewport)
        pixels = (moved - camera.focus) @ camera.basis().T * min(viewport) / camera.span
        half = np.asarray(viewport) / 2 - np.maximum(32.0, np.asarray(viewport) * 0.1)
        assert np.all(np.abs(pixels) <= half + 1e-9)
        np.testing.assert_array_equal(camera.focus, focus)
        guarded_span = camera.span
        camera.keep_visible(moved, viewport)
        assert camera.span == pytest.approx(guarded_span)


def test_level3_spherical_corridor_geometry_and_crossings():
    from pathlib import Path

    from sim.api import SimulationConfig
    from sim.game import spherical_corridor
    from sim.game.training import RPOTrainingConfig

    config = SimulationConfig.from_yaml(
        Path(__file__).resolve().parents[1] / "game/configs/game_training_rpo_03_rbar_approach.yaml"
    )
    (region,) = RPOTrainingConfig.from_metadata(config.scenario.metadata).forbidden_regions
    assert region.kind == "spherical_corridor"
    # Central cavity and approach axis are safe; the shell is forbidden at all azimuths.
    for angle in np.linspace(0, 2 * np.pi, 12):
        safe = [-3.0, 3 * np.tan(np.deg2rad(24)) * np.cos(angle), 3 * np.tan(np.deg2rad(24)) * np.sin(angle)]
        unsafe = [-3.0, 3 * np.tan(np.deg2rad(26)) * np.cos(angle), 3 * np.tan(np.deg2rad(26)) * np.sin(angle)]
        assert not region.contains_positions(safe)[0]
        assert region.contains_positions(unsafe)[0]
    assert not region.intersects_segment([-7, 0, 0], [-0.75, 0, 0])
    assert region.intersects_segment([7, 0, 0], [0, 0, 0])
    assert region.intersects_segment([0, -7, 0], [0, 7, 0])
    assert not region.intersects_segment([0, -0.2, 0], [0, 0.2, 0])
    assert region.intersects_segment([5.8, -1, 0], [5.8, 1, 0])
    assert not region.intersects_segment([6.0, -1, 0], [6.0, 1, 0])
    faces, lines = spherical_corridor.surface(region)
    assert faces and lines
    for first, second in ((1, 0), (2, 0)):
        section = spherical_corridor.section(region, first, second)
        radii = np.linalg.norm(section - region.center_ric_km, axis=1)
        assert np.all(np.isclose(radii, 0.95) | np.isclose(radii, 5.8))


@pytest.mark.parametrize("inner,outer,angle", [(0, 5, 25), (2, 1, 25), (1, 5, 0), (1, 5, 90), (1, 5, float("nan"))])
def test_spherical_corridor_rejects_invalid_geometry(inner, outer, angle):
    from sim.game.training import RPOTrainingConfig

    with pytest.raises(ValueError):
        RPOTrainingConfig.from_metadata(
            {
                "game": {
                    "training": {
                        "forbidden_regions": [
                            {
                                "kind": "spherical_corridor",
                                "inner_radius_km": inner,
                                "outer_radius_km": outer,
                                "cone_half_angle_deg": angle,
                            }
                        ]
                    }
                }
            }
        )
