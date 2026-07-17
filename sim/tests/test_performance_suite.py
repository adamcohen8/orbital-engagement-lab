from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from sim.performance import load_performance_manifest, physics_payload_hash, run_performance_suite
from sim.performance.suite import _deterministic_payload, _scenario_work_counts, _simulated_duration_s


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
    assert (tmp_path / "benchmark" / "benchmark_results.json").is_file()
    assert (tmp_path / "benchmark" / "benchmark_report.md").is_file()
