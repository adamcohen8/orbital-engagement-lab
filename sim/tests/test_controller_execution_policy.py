from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from sim.config import scenario_config_from_dict
from sim.core.models import Command
from sim.single_run import _run_single_config


class BudgetEchoController:
    def act(self, belief, t_s: float, budget_ms: float) -> Command:
        del belief, t_s
        return Command(
            thrust_eci_km_s2=np.array([1e-5, 0.0, 0.0]),
            mode_flags={"received_budget_ms": float(budget_ms)},
        )


class SlowBudgetEchoController(BudgetEchoController):
    def act(self, belief, t_s: float, budget_ms: float) -> Command:
        time.sleep(0.002)
        return super().act(belief, t_s, budget_ms)


def _config(tmp_path: Path, *, controller_class: str, deadline_policy: str, budget_ms: float) -> dict:
    return {
        "scenario_name": "controller_execution_policy",
        "objects": {
            "target": {
                "enabled": True,
                "initial_state": {
                    "position_eci_km": [7000.0, 0.0, 0.0],
                    "velocity_eci_km_s": [0.0, 7.5, 0.0],
                },
                "orbit_control": {
                    "module": "sim.tests.test_controller_execution_policy",
                    "class_name": controller_class,
                },
            }
        },
        "simulator": {
            "duration_s": 1.0,
            "dt_s": 1.0,
            "execution": {
                "controller": {
                    "orbit_budget_ms": budget_ms,
                    "attitude_budget_ms": 2.0,
                    "deadline_policy": deadline_policy,
                }
            },
            "dynamics": {"attitude": {"enabled": False}},
            "termination": {"earth_impact_enabled": False},
        },
        "outputs": {
            "output_dir": str(tmp_path),
            "mode": "save",
            "stats": {"print_summary": False, "save_json": False, "save_full_log": False, "controller_debug": True},
            "plots": {"enabled": False, "figure_ids": []},
            "animations": {"enabled": False, "types": []},
        },
    }


def test_configured_controller_budget_reaches_controller(tmp_path: Path) -> None:
    payload = _run_single_config(
        scenario_config_from_dict(
            _config(tmp_path, controller_class="BudgetEchoController", deadline_policy="record", budget_ms=7.5)
        )
    )
    row = payload["controller_debug_by_object"]["target"][0]
    assert row["command_orbit"]["mode_flags"]["received_budget_ms"] == 7.5
    assert row["mode_flags"]["orbit_controller_budget_ms"] == 7.5


def test_zero_command_deadline_policy_is_deterministic(tmp_path: Path) -> None:
    payload = _run_single_config(
        scenario_config_from_dict(
            _config(tmp_path, controller_class="SlowBudgetEchoController", deadline_policy="zero_command", budget_ms=1e-6)
        )
    )
    row = payload["controller_debug_by_object"]["target"][0]
    assert row["mode_flags"]["orbit_controller_deadline_missed"] is True
    np.testing.assert_allclose(row["command_applied"]["thrust_eci_km_s2"], np.zeros(3))


def test_integrated_mission_execution_obeys_controller_deadline_policy(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        controller_class="SlowBudgetEchoController",
        deadline_policy="zero_command",
        budget_ms=1e-6,
    )
    config["objects"]["target"]["mission_execution"] = {
        "module": "sim.mission.modules",
        "class_name": "ControllerPointingExecution",
        "params": {"require_attitude_alignment": False, "use_strategy_fallback_thrust": False},
    }

    payload = _run_single_config(scenario_config_from_dict(config))

    row = payload["controller_debug_by_object"]["target"][0]
    assert row["use_integrated_command"] is True
    assert row["mode_flags"]["orbit_controller_deadline_missed"] is True
    np.testing.assert_allclose(row["command_applied"]["thrust_eci_km_s2"], np.zeros(3))
