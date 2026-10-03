"""Display forecast parity and exact score decisions for the opt-in backend."""

import json
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("oel_rust_game")
pytest.importorskip("oel_rust_orbit")

from sim.api import SimulationConfig, SimulationSnapshot
from sim.dynamics.orbit.cr3bp import EARTH_MOON_MEAN_MOTION_RAD_S, cr3bp_halo_seed_state_km_s
from sim.flight_software.rust_game_backend import extension
from sim.game import prediction, pygame_dashboard
from sim.game.backend import create_game_physics_session
from sim.game.pygame_dashboard import PygameRPODashboard
from sim.game.training import RPOTrainingConfig, RPOTrainingTracker
from sim.game.training_models import InspectionGateConfig


def dashboard(backend, mode="linearized"):
    d = object.__new__(PygameRPODashboard)
    d.numeric_backend = backend
    d.coast_prediction_model = "cr3bp"
    d.cr3bp_projection_mode = mode
    d.relative_frame = "moon_ric"
    d.coast_prediction_orbit_fraction = None
    d.mean_motion_rad_s = EARTH_MOON_MEAN_MOTION_RAD_S
    d.coast_prediction_horizon_s = 21600.0
    d.cr3bp_coast_prediction_horizon_s = 21600.0
    d.cr3bp_coast_prediction_dt_s = 1.0
    d.coast_prediction_dt_s = 10.0
    d.cr3bp_active_prediction_horizon_s = 1800.0
    d.t_s = [0.0]
    d.reference_state_eci = cr3bp_halo_seed_state_km_s(family="l2_nrho_southern")
    d._prediction_cache = {}
    return d


@pytest.mark.parametrize("mode", ("linearized", "nonlinear"))
@pytest.mark.parametrize("active", (False, True))
def test_cislunar_actual_dashboard_forecasts(mode, active):
    left, right = dashboard("python", mode), dashboard("rust", mode)
    rel = np.array([-3.0, 4.0, 0.5, 0.0, 0.0, 2e-6])
    a = left._coast_prediction_from_cached("chaser", rel, active_burn=active)
    b = right._coast_prediction_from_cached("chaser", rel, active_burn=active)
    assert a.shape == b.shape == (60 if active else 120, 6)
    np.testing.assert_allclose(b[:, :3], a[:, :3], rtol=0.0, atol=2e-8)
    np.testing.assert_allclose(b[:, 3:], a[:, 3:], rtol=0.0, atol=1e-11)


@pytest.mark.parametrize("backend", ("python", "rust"))
def test_native_cache_reuse_and_invalidation(backend):
    d = dashboard(backend)
    rel = np.array([-3.0, 4.0, 0.5, 0.0, 0.0, 2e-6])
    original = pygame_dashboard.propagate_cr3bp_reference_stm
    with patch.object(pygame_dashboard, "propagate_cr3bp_reference_stm", wraps=original) as calls:
        first = d._coast_prediction_from_cached("chaser", rel, active_burn=True)
        count = calls.call_count
        assert count > 0
        changed = rel.copy()
        changed[4] += 1e-5
        second = d._coast_prediction_from_cached("chaser", changed, active_burn=True)
        assert calls.call_count == count
        assert not np.array_equal(first, second)
        d.reference_state_eci[0] += 0.01
        d._coast_prediction_from_cached("chaser", changed, active_burn=True)
        assert calls.call_count > count
        table = d._prediction_cache["_linearized_cr3bp_moon_ric_stm_table"]
        d._coast_prediction_from_cached("coast", changed, active_burn=False)
        assert d._prediction_cache["_linearized_cr3bp_moon_ric_stm_table"] is not table


@pytest.mark.parametrize("func", (prediction._elliptic_ya_coast_states, prediction._elliptic_linear_coast_states))
@pytest.mark.parametrize("times", (np.array([]), np.array([0.0]), np.array([300.0, 0.0, 0.5, 40.0, 40.0, 120.0, -1.0])))
def test_elliptic_forecast_schedule(func, times):
    chief = np.array([7000.0, 2000.0, 0.0, -2.0, 7.8, 0.2])
    rel = np.array([1.0, 2.0, 3.0, 0.001, -0.002, 0.0001])
    left = func(rel, times, chief)
    right = func(rel, times, chief, numeric_backend="rust")
    assert left.shape == right.shape == (len(times), 6)
    np.testing.assert_allclose(right, left, rtol=1e-12, atol=1e-11)


@pytest.mark.parametrize("source", sorted(Path("sim/game/configs").glob("*.yaml")), ids=lambda p: p.stem)
def test_level_score_and_history_exact(source):
    config = SimulationConfig.from_yaml(source)
    training = RPOTrainingConfig.from_metadata(dict(config.scenario.metadata))
    session = create_game_physics_session(config, backend="rust")
    trackers = [RPOTrainingTracker(training, numeric_backend=b) for b in ("python", "rust")]
    snapshots = [session.reset(seed=7821)]
    snapshots += [session.step(dt_s=0.5) for _ in range(30)]
    for snapshot in snapshots:
        # Shared truth isolates scoring from propagator arithmetic differences.
        for tracker in trackers:
            tracker.record(snapshot)
        assert json.dumps(asdict(trackers[0].score()), sort_keys=True) == json.dumps(
            asdict(trackers[1].score()), sort_keys=True
        )
    for name in (
        "_range_array",
        "_speed_array",
        "_goal_error_array",
        "_delta_v_interval_km_s_array",
        "_target_delta_v_interval_km_s_array",
    ):
        count = trackers[0]._history_count
        np.testing.assert_array_equal(getattr(trackers[1], name)[:count], getattr(trackers[0], name)[:count])


@pytest.mark.parametrize("goal", ("point", "range", "nmt", "inspection"))
def test_scoring_thresholds_nan_and_goal_invalidation(goal):
    cfg = RPOTrainingConfig(
        enabled=True,
        relative_frame="cislunar",
        goal_radius_km=1.0,
        max_goal_speed_km_s=0.01,
        hard_speed_limit_radius_km=1.0,
        hard_speed_limit_km_s=0.01,
        max_delta_v_m_s=1.0,
    )
    if goal == "range":
        cfg = replace(cfg, goal_range_km=1.0, goal_range_tolerance_km=0.01)
    if goal == "nmt":
        cfg = replace(cfg, goal_nmt_radial_amplitude_km=1.0, goal_nmt_tolerance_km=0.01)
    if goal == "inspection":
        cfg = replace(cfg, inspection_gates=(InspectionGateConfig("test", np.zeros(3), np.ones(3)),))
    trackers = [RPOTrainingTracker(cfg, numeric_backend=b) for b in ("python", "rust")]
    rng = np.random.default_rng(227)
    states = [rng.normal(size=6) for _ in range(80)]
    states += [
        np.array([r, 0.0, 0.0, v, 0.0, 0.0])
        for r in (np.nextafter(1.0, 0.0), 1.0, np.nextafter(1.0, 2.0))
        for v in (np.nextafter(0.01, 0.0), 0.01, np.nextafter(0.01, 1.0))
    ]
    states += [np.full(6, np.nan)]
    for idx, rel in enumerate(states):
        truth = {"target": np.zeros(14), "chaser": np.pad(rel, (0, 8))}
        snap = SimulationSnapshot(
            idx, float(idx), truth, {}, {"chaser": rng.normal(size=3) * 0.001, "target": rng.normal(size=3) * 0.001}, {}
        )
        for t in trackers:
            t.record(snap)
        assert json.dumps(asdict(trackers[0].score()), sort_keys=True) == json.dumps(
            asdict(trackers[1].score()), sort_keys=True
        )
    changed = replace(cfg, goal_relative_ric_km=np.ones(3), goal_nmt_center_ric_km=np.ones(3))
    for t in trackers:
        t.config = changed
        t.record(replace(snap, time_s=100.0, truth={"target": np.zeros(14), "chaser": np.pad(np.ones(6), (0, 8))}))
    np.testing.assert_array_equal(
        trackers[0]._goal_error_array[: trackers[0]._history_count],
        trackers[1]._goal_error_array[: trackers[1]._history_count],
    )


def test_new_wheel_required_and_native_work_bounds(monkeypatch):
    import sim.flight_software.rust_game_backend as adapter

    native = extension()
    with pytest.raises(ValueError, match="bound"):
        native.preview_th_history(np.zeros(6), np.ones(6), [float("inf")], 1.0, 398600.0)
    with pytest.raises(ValueError, match="bound"):
        native.score_sample(np.ones(6), np.ones(3), np.ones(3), b"x")
    with pytest.raises(ValueError, match="numeric_backend"):
        RPOTrainingTracker(RPOTrainingConfig(), numeric_backend="unknown")
    monkeypatch.delattr(native, "score_sample")
    adapter.extension.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="compatible"):
            RPOTrainingTracker(RPOTrainingConfig(), numeric_backend="rust")
    finally:
        adapter.extension.cache_clear()


def test_complete_cislunar_target_display_orbit():
    config = SimulationConfig.from_yaml("sim/game/configs/game_training_rpo_bonus_cislunar_rendezvous.yaml")
    snapshot = create_game_physics_session(config, backend="rust").reset(seed=1)
    game = config.scenario.metadata["game"]
    outputs = []
    for backend in ("python", "rust"):
        d = dashboard(backend)
        d.target_orbit_reference_state_eci = snapshot.truth["target"][:6]
        d.target_coast_prediction_horizon_s = game["target_coast_prediction_horizon_s"]
        d.target_coast_prediction_dt_s = game["target_coast_prediction_dt_s"]
        result = d._cr3bp_target_orbit_prediction()
        outputs.append(result)
        np.testing.assert_array_equal(d._cr3bp_target_orbit_prediction(allow_build=False), result)
    assert outputs[0].shape == outputs[1].shape
    np.testing.assert_allclose(outputs[1][:, :3], outputs[0][:, :3], rtol=0.0, atol=5e-7)
    np.testing.assert_allclose(outputs[1][:, 3:], outputs[0][:, 3:], rtol=0.0, atol=1e-10)


def test_elliptic_dashboard_native_th_fallback():
    native = extension()
    d = dashboard("rust")
    d.coast_prediction_model = "tschauner_hempel"
    d.reference_state_eci = np.array([7000.0, 2000.0, 0.0, -2.0, 7.8, 0.2])
    d.coast_prediction_horizon_s = 120.0
    rel = np.array([1.0, 2.0, 3.0, 0.001, -0.002, 0.0001])
    with patch("sim.game.dashboard_prediction._elliptic_ya_coast_states", side_effect=ValueError("singular")):
        with patch.object(native, "preview_th_history", wraps=native.preview_th_history) as call:
            actual = d._coast_prediction_from(rel)
    assert call.call_count == 1
    d.numeric_backend = "python"
    with patch("sim.game.dashboard_prediction._elliptic_ya_coast_states", side_effect=ValueError("singular")):
        expected = d._coast_prediction_from(rel)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-11)
