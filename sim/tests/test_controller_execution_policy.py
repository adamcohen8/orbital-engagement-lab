from __future__ import annotations

from pathlib import Path

import pytest

from sim.config import scenario_config_from_dict


def _legacy_config(tmp_path: Path, *, mission_execution: bool = False) -> dict:
    target = {
        "enabled": True,
        "initial_state": {
            "position_eci_km": [7000.0, 0.0, 0.0],
            "velocity_eci_km_s": [0.0, 7.5, 0.0],
        },
        "orbit_control": {
            "module": "sim.control.orbit.zero_controller",
            "class_name": "ZeroController",
        },
    }
    if mission_execution:
        target["mission_execution"] = {
            "module": "sim.mission.modules",
            "class_name": "ControllerPointingExecution",
        }
    return {
        "scenario_name": "controller_execution_policy",
        "objects": {"target": target},
        "simulator": {
            "duration_s": 1.0,
            "dt_s": 1.0,
            "execution": {
                "controller": {
                    "orbit_budget_ms": 1.0,
                    "attitude_budget_ms": 2.0,
                    "deadline_policy": "zero_command",
                }
            },
            "dynamics": {"attitude": {"enabled": False}},
        },
        "outputs": {"output_dir": str(tmp_path)},
    }


@pytest.mark.parametrize("mission_execution", [False, True])
def test_satellite_controller_deadline_path_is_removed(tmp_path: Path, mission_execution: bool) -> None:
    with pytest.raises(ValueError, match="removed GNC v1 satellite field"):
        scenario_config_from_dict(_legacy_config(tmp_path, mission_execution=mission_execution))


def test_v2_stack_uses_its_own_task_timing_contract(tmp_path: Path) -> None:
    config = _legacy_config(tmp_path)
    target = config["objects"]["target"]
    target.pop("orbit_control")
    target["flight_software"] = {
        "stack": "fsw.passive",
        "hardware_profile": "hardware.passive.v1",
        "task_period_s": 0.25,
    }
    cfg = scenario_config_from_dict(config)
    assert cfg.objects["target"].flight_software.task_period_s == 0.25
