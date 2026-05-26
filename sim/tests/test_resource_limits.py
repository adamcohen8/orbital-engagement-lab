from __future__ import annotations

from pathlib import Path

import pytest

from sim.config import scenario_config_from_dict
from sim.execution import create_single_run_engine
from sim.resource_limits import (
    ResourceGovernor,
    SimulationMemoryBudgetError,
    apply_resource_profile_to_config_dict,
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
