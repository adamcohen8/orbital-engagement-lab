"""Full runs avoid discarded snapshots without changing step-driven behavior."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from sim.config import scenario_config_from_dict
from sim.performance.suite import physics_payload_hash
from sim.single_run import _SingleRunEngine


def _config(tmp_path):
    return scenario_config_from_dict({
        "scenario_name": "snapshot_behavior",
        "objects": {"satellite": {
            "kind": "satellite", "runtime_profile": "trajectory_only",
            "initial_state": {
                "position_eci_km": [7000, 0, 0], "velocity_eci_km_s": [0, 7.5, 0],
            },
        }},
        "simulator": {"duration_s": 5, "dt_s": 1, "dynamics": {"attitude": {"enabled": False}}},
        "outputs": {
            "output_dir": str(tmp_path), "plots": {"enabled": False}, "review": {"enabled": False},
            "stats": {"save_json": False, "save_full_log": False, "save_csv": False, "print_summary": False},
        },
    })


def test_full_run_retains_callback_observations_and_step_payload(tmp_path):
    cfg = _config(tmp_path)
    full = _SingleRunEngine(deepcopy(cfg))
    stepped = _SingleRunEngine(deepcopy(cfg))
    observed = []
    full.active_step_callback = lambda step, total: observed.append(full.snapshot())
    full_payload = full.run()
    snapshots = []
    while not stepped.done:
        snapshots.append(stepped.step())
    stepped_payload = stepped.run()
    assert physics_payload_hash(full_payload) == physics_payload_hash(stepped_payload)
    assert len(observed) == len(snapshots) == 5
    for actual, expected in zip(observed, snapshots):
        assert actual["step_index"] == expected["step_index"]
        assert actual["time_s"] == expected["time_s"]
        for group in ("truth", "belief", "applied_thrust", "applied_torque"):
            assert actual[group].keys() == expected[group].keys()
            for object_id in actual[group]:
                assert np.array_equal(actual[group][object_id], expected[group][object_id], equal_nan=True)


def test_full_run_skips_discarded_snapshots_and_preserves_step_overrides(tmp_path):
    engine = _SingleRunEngine(_config(tmp_path))
    snapshot_calls = []
    original_snapshot = engine.snapshot
    engine.snapshot = lambda *args, **kwargs: (snapshot_calls.append(True), original_snapshot(*args, **kwargs))[1]
    engine.run()
    assert snapshot_calls == []
    result = engine.step()
    assert result["step_index"] == 5
    assert snapshot_calls == [True]

    overridden = _SingleRunEngine(_config(tmp_path))
    original_step = overridden.step
    override_results = []

    def step():
        result = original_step()
        override_results.append(result)
        return result

    overridden.step = step
    overridden.run()
    assert [result["step_index"] for result in override_results] == [1, 2, 3, 4, 5]
