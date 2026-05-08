from __future__ import annotations

from pathlib import Path

import pytest

from sim.config import scenario_config_from_dict
from sim.execution import create_single_run_engine
from sim.resource_limits import SimulationMemoryBudgetError


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
