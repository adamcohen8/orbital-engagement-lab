from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from sim.performance import load_performance_manifest, physics_payload_hash, run_performance_suite
from sim.performance.suite import (
    _deterministic_payload,
    _reset_scenario_measurement_caches,
    _scenario_work_counts,
    _simulated_duration_s,
)


def test_full_path_manifest_covers_distinct_runtime_families() -> None:
    manifest = load_performance_manifest()
    names = {case.name for case in manifest.cases}

    assert {"smoke", "standard", "full"} <= set(manifest.profile_defaults)
    assert {
        "zonal_rk4_accelerated",
        "sensing_relative_ekf",
        "modern_actuator_stack",
        "cr3bp_earth_moon",
        "ogp_sgp4",
        "rocket_ascent",
        "reentry_diagnostics",
        "artifact_output_pipeline",
    } <= names
    if "attitude_reference_oel" in names:
        assert {
            "attitude_reference_oel",
            "attitude_reference_basilisk",
            "full_satellite_serial",
            "full_satellite_parallel",
            "adaptive_high_fidelity",
            "monte_carlo_orchestration",
        } <= names


def test_drag_model_cases_share_one_two_body_fixture() -> None:
    manifest = load_performance_manifest()
    expected_models = {
        "exponential",
        "ussa1976",
        "msis86",
        "nrlmsise00",
        "jacchia70",
        "jb2006",
        "jb2008",
        "harris_priester",
    }
    cases = {case.name: case for case in manifest.cases if case.category == "atmosphere"}

    if manifest.suite_name == "oel_public_core":
        assert cases == {}
        return

    assert set(cases) == {f"drag_{model}" for model in expected_models}
    assert {case.config_path for case in cases.values()} == {
        "configs/performance/drag_model_single_satellite.yaml"
    }
    assert {
        str(case.base_overrides["simulator.environment.atmosphere_model"])
        for case in cases.values()
    } == expected_models
    assert {case.profiles["standard"]["duration_s"] for case in cases.values()} == {1200.0}


def test_drag_measurements_clear_only_trajectory_epoch_caches() -> None:
    manifest = load_performance_manifest()
    cases = {case.name: case for case in manifest.cases}

    if manifest.suite_name == "oel_public_core":
        assert not any(name.startswith("drag_") for name in cases)
        return

    with patch("sim.dynamics.orbit.jacchia70_backend.clear_trajectory_epoch_caches") as clear_jacchia:
        policy = _reset_scenario_measurement_caches(cases["drag_jacchia70"])
    assert policy == "cold_trajectory_epoch_cache"
    clear_jacchia.assert_called_once_with()

    with patch("sim.dynamics.orbit.harris_priester_backend.clear_trajectory_epoch_caches") as clear_harris:
        policy = _reset_scenario_measurement_caches(cases["drag_harris_priester"])
    assert policy == "cold_trajectory_epoch_cache"
    clear_harris.assert_called_once_with()

    assert _reset_scenario_measurement_caches(cases["zonal_rk4_accelerated"]) == "warm_process_state"


def test_physics_hash_ignores_timing_and_output_provenance() -> None:
    first = {
        "truth": {"satellite": [1.0, 2.0, 3.0]},
        "summary": {
            "runtime_profile": {"total_step_wall_s": 1.0},
            "output_dir": "/tmp/run-a",
            "plot_outputs": {"trajectory": "/tmp/run-a/trajectory.png"},
        },
        "reproducibility": {"generated_utc": "first", "config_sha256": "path-sensitive-a"},
    }
    second = {
        "truth": {"satellite": [1.0, 2.0, 3.0]},
        "summary": {
            "runtime_profile": {"total_step_wall_s": 99.0},
            "output_dir": "/tmp/run-b",
            "plot_outputs": {"trajectory": "/tmp/run-b/trajectory.png"},
        },
        "reproducibility": {"generated_utc": "second", "config_sha256": "path-sensitive-b"},
    }

    assert physics_payload_hash(first) == physics_payload_hash(second)
    second["truth"]["satellite"][2] = 4.0
    assert physics_payload_hash(first) != physics_payload_hash(second)


def test_streaming_physics_hash_matches_legacy_canonical_encoding() -> None:
    payload = {
        "z": np.array([[1.0, float("nan")], [float("inf"), -2.5]]),
        "a": (Path("relative/output.txt"), np.int64(4), {"keep": True, "elapsed_ms": 99.0}),
        2: "numeric-key",
        "runtime_profile": {"ignored": "timing"},
    }
    legacy_json = json.dumps(
        _deterministic_payload(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=True,
    ).encode("utf-8")

    assert physics_payload_hash(payload) == hashlib.sha256(legacy_json).hexdigest()


def test_campaign_work_is_aggregated_across_runs() -> None:
    def run_payload(duration_s: float) -> dict:
        return {
            "summary": {
                "duration_s": duration_s,
                "runtime_profile": {
                    "completed_steps": 2,
                    "stage_totals": {"object_step": {"count": 4}},
                    "object_totals": {
                        "satellite": {
                            "stages": {
                                "dynamics_step": {"count": 2},
                                "estimator_update": {"count": 2},
                                "decision": {"count": 2},
                            }
                        }
                    },
                },
            }
        }

    payload = {"runs": [run_payload(10.0), run_payload(20.0)]}

    assert _simulated_duration_s(payload, 1.0) == 30.0
    assert _scenario_work_counts(payload) == {
        "outer_steps": 4,
        "object_steps": 8,
        "dynamics_steps": 4,
        "estimator_updates": 4,
        "decisions": 4,
    }


def test_smoke_case_repeats_with_exact_physics_parity(tmp_path) -> None:
    payload = run_performance_suite(
        profile="smoke",
        case_names={"zonal_rk4_accelerated"},
        warmups=0,
        repeats=2,
        output_dir=tmp_path / "benchmark",
    )

    assert payload["cases_failed"] == 0
    case = payload["cases"][0]
    assert case["status"] == "passed"
    assert case["deterministic_parity"] is True
    assert len(set(case["physics_hashes"])) == 1
    assert case["coverage_passed"] is True
    assert case["measurement_cache_policy"] == "warm_process_state"
    assert (tmp_path / "benchmark" / "benchmark_results.json").is_file()
    assert (tmp_path / "benchmark" / "benchmark_report.md").is_file()
