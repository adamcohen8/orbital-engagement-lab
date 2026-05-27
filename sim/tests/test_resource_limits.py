from __future__ import annotations

from pathlib import Path

import pytest

from sim.config import scenario_config_from_dict
from sim.execution import create_single_run_engine
from sim.resource_limits import (
    ResourceGovernor,
    SimulationMemoryBudgetError,
    _memory_bytes_to_mb,
    _parse_macos_vm_stat_mb,
    apply_resource_profile_to_config_dict,
    current_resource_snapshot,
    estimate_resource_requirements,
)


def _single_target_config(output_dir: Path, *, duration_s: float = 1000.0) -> dict:
    return {
        "scenario_name": "memory_guard_test",
        "rocket": {"enabled": False},
        "target": {
            "enabled": True,
            "specs": {"mass_kg": 100.0},
            "initial_state": {
                "position_eci_km": [7000.0, 0.0, 0.0],
                "velocity_eci_km_s": [0.0, 7.5, 0.0],
            },
        },
        "chaser": {"enabled": False},
        "simulator": {
            "duration_s": duration_s,
            "dt_s": 1.0,
            "termination": {"earth_impact_enabled": False},
            "dynamics": {"attitude": {"enabled": False}},
        },
        "outputs": {
            "output_dir": str(output_dir),
            "mode": "save",
            "stats": {"print_summary": False, "save_json": False, "save_full_log": False},
            "plots": {"enabled": False},
            "animations": {"enabled": False},
        },
    }


def test_history_memory_guard_fails_before_output_directory_creation(tmp_path: Path) -> None:
    outdir = tmp_path / "guarded"
    root = _single_target_config(outdir)
    root["outputs"]["resource_limits"] = {"max_history_memory_mb": 0.01}
    cfg = scenario_config_from_dict(root)

    with pytest.raises(SimulationMemoryBudgetError, match="Estimated simulation history memory"):
        create_single_run_engine(cfg)

    assert not outdir.exists()


def test_history_memory_guard_uses_caller_cap_even_if_config_requests_more(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _single_target_config(tmp_path / "guarded")
    root["outputs"]["resource_limits"] = {"max_history_memory_mb": 1000.0}
    cfg = scenario_config_from_dict(root)
    monkeypatch.setenv("OEL_MAX_HISTORY_MEMORY_MB", "0.01")

    with pytest.raises(SimulationMemoryBudgetError, match="limit=0.01 MB"):
        create_single_run_engine(cfg)


def test_history_memory_guard_allows_run_under_budget(tmp_path: Path) -> None:
    root = _single_target_config(tmp_path / "ok", duration_s=2.0)
    root["outputs"]["resource_limits"] = {"max_history_memory_mb": 10.0}
    cfg = scenario_config_from_dict(root)

    engine = create_single_run_engine(cfg)

    assert engine.history_memory_estimate.estimated_peak_mb < 10.0


def test_laptop_safe_profile_rewrites_batch_config(tmp_path: Path) -> None:
    root = _single_target_config(tmp_path / "mc", duration_s=10.0)
    root["monte_carlo"] = {
        "enabled": True,
        "iterations": 4,
        "parallel_enabled": True,
        "parallel_workers": 8,
        "variations": [],
    }
    root["outputs"]["plots"] = {"enabled": True}

    profiled = apply_resource_profile_to_config_dict(root, "laptop-safe")

    assert profiled["monte_carlo"]["parallel_enabled"] is False
    assert profiled["monte_carlo"]["parallel_workers"] == 1
    assert profiled["outputs"]["plots"]["enabled"] is False
    assert profiled["outputs"]["monte_carlo"]["checkpoint_enabled"] is True
    assert profiled["simulator"]["resource_profile"] == "laptop-safe"


def test_resource_estimate_reports_effective_batch_shape(tmp_path: Path) -> None:
    root = _single_target_config(tmp_path / "mc", duration_s=12.0)
    root["monte_carlo"] = {
        "enabled": True,
        "iterations": 3,
        "parallel_enabled": True,
        "parallel_workers": 2,
        "variations": [],
    }
    cfg = scenario_config_from_dict(root)

    estimate = estimate_resource_requirements(cfg)

    assert estimate.study_type == "monte_carlo"
    assert estimate.runs == 3
    assert estimate.steps_per_run == 12
    assert estimate.effective_workers == 2


def test_resource_estimate_reports_sensitivity_shape(tmp_path: Path) -> None:
    root = _single_target_config(tmp_path / "sensitivity", duration_s=12.0)
    root["analysis"] = {
        "enabled": True,
        "study_type": "sensitivity",
        "execution": {"parallel_enabled": True, "parallel_workers": 3},
        "sensitivity": {
            "method": "two_parameter_grid",
            "parameters": [
                {"parameter_path": "target.specs.mass_kg", "values": [90.0, 100.0]},
                {"parameter_path": "target.specs.drag_area_m2", "values": [0.8, 1.0, 1.2]},
            ],
        },
    }
    cfg = scenario_config_from_dict(root)

    estimate = estimate_resource_requirements(cfg)

    assert estimate.study_type == "sensitivity"
    assert estimate.runs == 6
    assert estimate.requested_workers == 3
    assert estimate.effective_workers == 3


def test_resource_governor_honors_off_profile(tmp_path: Path) -> None:
    root = _single_target_config(tmp_path / "mc", duration_s=2.0)
    root["simulator"]["resource_profile"] = "off"
    cfg = scenario_config_from_dict(root)

    governor = ResourceGovernor(cfg)

    assert governor.enabled is False


def test_simulator_resource_profile_is_canonical_over_legacy_outputs_profile(tmp_path: Path) -> None:
    root = _single_target_config(tmp_path / "mc", duration_s=2.0)
    root["simulator"]["resource_profile"] = "laptop-safe"
    root["outputs"]["resource_limits"] = {"resource_profile": "off"}
    cfg = scenario_config_from_dict(root)

    estimate = estimate_resource_requirements(cfg)
    governor = ResourceGovernor(cfg)

    assert estimate.profile == "laptop-safe"
    assert governor.profile.name == "laptop-safe"


def test_macos_vm_stat_parser_counts_reclaimable_inactive_memory() -> None:
    text = """
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                7100.
Pages active:                            130919.
Pages inactive:                          127528.
Pages speculative:                         2780.
Pages wired down:                        145265.
Pages purgeable:                           4311.
Pages occupied by compressor:             74159.
""".strip()

    total_mb, available_mb = _parse_macos_vm_stat_mb(text)

    free_only_mb = (7100 + 2780) * 16384 / (1024 * 1024)
    assert available_mb is not None
    assert total_mb is not None
    assert available_mb > free_only_mb * 10


def test_windows_memory_bytes_convert_to_mb() -> None:
    total_mb, available_mb = _memory_bytes_to_mb(16 * 1024 * 1024 * 1024, 5 * 1024 * 1024 * 1024)

    assert total_mb == 16384.0
    assert available_mb == 5120.0


def test_resource_snapshot_uses_windows_memory_reader_when_unix_readers_are_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sim.resource_limits._read_proc_mem_available_mb", lambda: (None, None))
    monkeypatch.setattr("sim.resource_limits._read_macos_memory_mb", lambda: (None, None))
    monkeypatch.setattr("sim.resource_limits._read_windows_memory_mb", lambda: (8192.0, 4096.0))

    snapshot = current_resource_snapshot()

    assert snapshot.total_memory_mb == 8192.0
    assert snapshot.available_memory_mb == 4096.0
