from __future__ import annotations

import json
import sqlite3

import numpy as np
import pytest

from sim.api import SimulationConfig, SimulationSession
from sim.config import scenario_config_from_dict


def _scenario(output_dir, *, z_offset_km=0.0, collide=True, numeric_backend=None):
    raw = {
        "scenario_name": "spherical_collision_test",
        "objects": {
            "a": {
                "kind": "satellite", "runtime_profile": "trajectory_only",
                "specs": {"mass_kg": 100.0},
                "initial_state": {
                    "position_eci_km": [7000.0, 0.0, 0.0],
                    "velocity_eci_km_s": [0.01, 7.5, 0.0],
                },
            },
            "b": {
                "kind": "satellite", "runtime_profile": "trajectory_only",
                "specs": {"mass_kg": 200.0},
                "initial_state": {
                    "position_eci_km": [7000.01, 0.0, z_offset_km],
                    "velocity_eci_km_s": [-0.01, 7.5, 0.0],
                },
            },
        },
        "simulator": {
            "duration_s": 1.0, "dt_s": 1.0,
            "dynamics": {"attitude": {"enabled": False}},
        },
        "outputs": {
            "mode": "save", "output_dir": str(output_dir),
            "stats": {"print_summary": False, "save_full_log": True},
            "plots": {"enabled": False},
            "review": {"enabled": True, "detail": "standard"},
        },
    }
    if collide:
        collision_config = {"enabled": True, "radii_m": {"a": 2.0, "b": 2.0}}
        if numeric_backend is not None:
            collision_config["numeric_backend"] = numeric_backend
        raw["simulator"]["collisions"] = collision_config
    return raw


@pytest.mark.parametrize("z_offset_km", [0.0, 0.001])
def test_crossing_between_samples_records_elastic_impact(tmp_path, z_offset_km):
    raw = _scenario(tmp_path, z_offset_km=z_offset_km)
    result = SimulationSession.from_config(SimulationConfig.from_dict(raw)).run()
    payload = json.loads((tmp_path / "master_run_log.json").read_text())
    events = payload["collision_events"]
    assert len(events) == 1
    event = events[0]
    assert 0.25 < event["time_s"] < 0.35
    positions = np.asarray(event["position_eci_km"])
    assert np.linalg.norm(positions[1] - positions[0]) == pytest.approx(0.004, abs=1e-8)
    masses = np.array([100.0, 200.0])
    before = np.asarray(event["pre_velocity_eci_km_s"])
    after = np.asarray(event["post_velocity_eci_km_s"])
    np.testing.assert_allclose(masses @ before, masses @ after, rtol=0, atol=1e-10)
    assert np.sum(masses[:, None] * before**2) == pytest.approx(
        np.sum(masses[:, None] * after**2), abs=1e-9
    )
    normal = np.asarray(event["normal_eci"])
    assert np.dot(after[1] - after[0], normal) == pytest.approx(
        -np.dot(before[1] - before[0], normal), abs=1e-12
    )
    tangential_before = (before[1] - before[0]) - np.dot(before[1] - before[0], normal) * normal
    tangential_after = (after[1] - after[0]) - np.dot(after[1] - after[0], normal) * normal
    np.testing.assert_allclose(tangential_after, tangential_before, rtol=0, atol=1e-12)
    if z_offset_km:
        assert abs(after[0, 2]) > 1e-4
    final_range = np.linalg.norm(result.truth["b"][-1, :3] - result.truth["a"][-1, :3])
    assert final_range > 0.004
    assert result.summary["collisions"]["count"] == 1
    with sqlite3.connect(tmp_path / "review" / "run.sqlite") as conn:
        rows = conn.execute(
            "SELECT time_s, event_type, object_id FROM events WHERE event_type='spherical_collision'"
        ).fetchall()
    assert rows == [(pytest.approx(event["time_s"]), "spherical_collision", "a,b")]


def test_no_contact_matches_existing_passive_path(tmp_path):
    enabled = _scenario(tmp_path / "enabled", z_offset_km=0.02)
    ordinary = _scenario(tmp_path / "ordinary", z_offset_km=0.02, collide=False)
    with_collision_model = SimulationSession.from_config(SimulationConfig.from_dict(enabled)).run()
    baseline = SimulationSession.from_config(SimulationConfig.from_dict(ordinary)).run()
    for object_id in ("a", "b"):
        np.testing.assert_allclose(with_collision_model.truth[object_id], baseline.truth[object_id], rtol=0, atol=1e-11)
    payload = json.loads((tmp_path / "enabled" / "master_run_log.json").read_text())
    assert payload["collision_events"] == []


def test_rust_collision_backend_matches_python_event_and_state(tmp_path):
    python_raw = _scenario(tmp_path / "python", z_offset_km=0.001, numeric_backend="python")
    rust_raw = _scenario(tmp_path / "rust", z_offset_km=0.001, numeric_backend="rust")
    python_result = SimulationSession.from_config(SimulationConfig.from_dict(python_raw)).run()
    rust_result = SimulationSession.from_config(SimulationConfig.from_dict(rust_raw)).run()
    python_event = json.loads((tmp_path / "python" / "master_run_log.json").read_text())["collision_events"][0]
    rust_event = json.loads((tmp_path / "rust" / "master_run_log.json").read_text())["collision_events"][0]
    assert rust_event["object_ids"] == python_event["object_ids"]
    assert rust_event["time_s"] == pytest.approx(python_event["time_s"], rel=0.0, abs=2e-12)
    for key in (
        "position_eci_km",
        "normal_eci",
        "pre_velocity_eci_km_s",
        "post_velocity_eci_km_s",
    ):
        np.testing.assert_allclose(rust_event[key], python_event[key], rtol=0.0, atol=2e-12)
    assert rust_event["closing_speed_m_s"] == pytest.approx(
        python_event["closing_speed_m_s"], rel=0.0, abs=2e-12
    )
    for object_id in ("a", "b"):
        np.testing.assert_allclose(
            rust_result.truth[object_id], python_result.truth[object_id], rtol=0.0, atol=2e-12
        )
    assert rust_result.summary["collisions"]["count"] == 1


def test_fast_pass_detects_contact_despite_disjoint_sample_endpoints(tmp_path):
    raw = _scenario(tmp_path)
    raw["objects"]["a"]["initial_state"]["velocity_eci_km_s"][0] = 5.0
    raw["objects"]["b"]["initial_state"]["velocity_eci_km_s"][0] = -5.0
    result = SimulationSession.from_config(SimulationConfig.from_dict(raw)).run()
    events = json.loads((tmp_path / "master_run_log.json").read_text())["collision_events"]
    assert len(events) == 1
    assert events[0]["time_s"] == pytest.approx(0.0006, abs=1e-7)
    initial_range = np.linalg.norm(result.truth["b"][0, :3] - result.truth["a"][0, :3])
    final_range = np.linalg.norm(result.truth["b"][-1, :3] - result.truth["a"][-1, :3])
    assert initial_range > 0.004 and final_range > 0.004


def test_equal_mass_head_on_contact_exchanges_normal_speeds(tmp_path):
    raw = _scenario(tmp_path)
    raw["objects"]["b"]["specs"]["mass_kg"] = 100.0
    SimulationSession.from_config(SimulationConfig.from_dict(raw)).run()
    event = json.loads((tmp_path / "master_run_log.json").read_text())["collision_events"][0]
    before = np.asarray(event["pre_velocity_eci_km_s"])
    after = np.asarray(event["post_velocity_eci_km_s"])
    assert after[0, 0] == pytest.approx(before[1, 0], abs=1e-9)
    assert after[1, 0] == pytest.approx(before[0, 0], abs=1e-9)


@pytest.mark.parametrize("change, message", [
    (lambda raw: raw["objects"]["b"].update(runtime_profile="flight_software"), "trajectory_only"),
    (lambda raw: raw["simulator"]["dynamics"]["attitude"].update(enabled=True), "attitude.enabled=false"),
    (lambda raw: raw["simulator"]["execution"].update(policy="parallel"), "serial object execution"),
    (lambda raw: raw["simulator"]["collisions"]["radii_m"].update(b=-1.0), "positive radii"),
])
def test_unsupported_collision_envelopes_fail_validation(tmp_path, change, message):
    raw = _scenario(tmp_path)
    raw["simulator"]["execution"] = {"policy": "serial"}
    change(raw)
    with pytest.raises(ValueError, match=message):
        scenario_config_from_dict(raw)
