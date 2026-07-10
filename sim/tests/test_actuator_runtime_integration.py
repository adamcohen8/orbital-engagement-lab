from __future__ import annotations

from pathlib import Path

import numpy as np

from sim.config import scenario_config_from_dict
from sim.single_run import _run_single_config


class ExternalBridgeIntegratedCommandMission:
    def __init__(self, thrust_eci_km_s2: list[float], torque_body_nm: list[float] | None = None):
        self.thrust_eci_km_s2 = np.array(thrust_eci_km_s2, dtype=float)
        self.torque_body_nm = np.zeros(3, dtype=float) if torque_body_nm is None else np.array(torque_body_nm, dtype=float)

    def update(self, **_kwargs):
        return {
            "mission_use_integrated_command": True,
            "thrust_eci_km_s2": self.thrust_eci_km_s2.copy(),
            "torque_body_nm": self.torque_body_nm.copy(),
            "command_mode_flags": {
                "source": "external_bridge",
                "external_command_mode": 2,
            },
        }


def _actuator_config(tmp_path: Path) -> dict:
    return {
        "scenario_name": "actuator_runtime_integration",
        "objects": {
            "target": {
                "enabled": True,
                "specs": {
                    "mass_kg": 500.0,
                    "actuators": {
                        "enabled": True,
                        "orbital": {
                            "electric_propulsion": {
                                "max_thrust_n": 0.5,
                                "isp_s": 1600.0,
                                "max_power_w": 100.0,
                                "power_per_newton_w": 200.0,
                            }
                        },
                    },
                },
                "initial_state": {
                    "position_eci_km": [7000.0, 0.0, 0.0],
                    "velocity_eci_km_s": [0.0, 7.5, 0.0],
                },
                "mission_objectives": [
                    {
                        "module": "sim.tests.test_actuator_runtime_integration",
                        "class_name": "ExternalBridgeIntegratedCommandMission",
                        "params": {"thrust_eci_km_s2": [1.0e-3, 0.0, 0.0]},
                    }
                ],
            }
        },
        "simulator": {
            "duration_s": 2.0,
            "dt_s": 1.0,
            "termination": {"earth_impact_enabled": False},
            "dynamics": {"attitude": {"enabled": False}},
        },
        "outputs": {
            "output_dir": str(tmp_path),
            "mode": "save",
            "stats": {
                "print_summary": False,
                "save_json": False,
                "save_csv": False,
                "save_full_log": False,
                "controller_debug": True,
            },
            "plots": {"enabled": False, "figure_ids": []},
            "animations": {"enabled": False, "types": []},
        },
    }


def test_configured_actuator_stack_processes_external_integrated_command(tmp_path: Path) -> None:
    payload = _run_single_config(scenario_config_from_dict(_actuator_config(tmp_path)))
    debug_rows = payload["controller_debug_by_object"]["target"]
    applied = debug_rows[0]["command_applied"]
    flags = dict(applied["mode_flags"])

    assert flags["source"] == "external_bridge"
    assert flags["actuator_stack_enabled"] is True
    assert flags["electric_propulsion_max_thrust_n"] == 0.5
    assert np.linalg.norm(np.array(applied["thrust_eci_km_s2"], dtype=float)) <= 1.0e-6 + 1e-15
    assert payload["summary"]["actuator_diagnostics_summary"]["target"]["actuator_stack_samples"] > 0
    assert payload["summary"]["actuator_diagnostics_summary"]["target"]["max_electric_propulsion_thrust_n"] > 0.0


def test_actuator_preset_configures_runtime_stack(tmp_path: Path) -> None:
    config = _actuator_config(tmp_path)
    config["objects"]["target"]["specs"] = {
        "mass_kg": 500.0,
        "actuator_preset": "BASIC_ELECTRIC_PROPULSION",
    }

    payload = _run_single_config(scenario_config_from_dict(config))
    applied = payload["controller_debug_by_object"]["target"][0]["command_applied"]
    flags = dict(applied["mode_flags"])

    assert flags["actuator_stack_enabled"] is True
    assert flags["electric_propulsion_max_thrust_n"] == 0.5
    assert np.linalg.norm(np.array(applied["thrust_eci_km_s2"], dtype=float)) <= 1.0e-6 + 1e-15
