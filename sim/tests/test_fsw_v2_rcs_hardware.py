from __future__ import annotations

from sim.config import scenario_config_from_dict
from sim.single_run import _run_single_config


def test_rcs_profile_allocates_thruster_packets_and_realizes_body_force(tmp_path) -> None:
    config = {
        "scenario_name": "gnc_v2_rcs_physical_boundary",
        "objects": {
            "satellite": {
                "kind": "satellite",
                "specs": {"mass_kg": 100.0},
                "initial_state": {
                    "position_eci_km": [7000.0, 0.0, 0.0],
                    "velocity_eci_km_s": [0.0, 7.5, 0.0],
                },
                "flight_software": {
                    "stack": "fsw.orbit_reference",
                    "hardware_profile": "hardware.rcs.v1",
                    "task_period_s": 0.25,
                    "params": {
                        "max_force_n": 1.0,
                        "max_acceleration_m_s2": 0.01,
                        "translation_mode": "scheduled_burn",
                        "scheduled_burns": [
                            {
                                "start_time_s": 0.0,
                                "duration_s": 1.0,
                                "frame": "eci",
                                "delta_v_m_s": [0.01, 0.0, 0.0],
                            }
                        ],
                    },
                },
            }
        },
        "simulator": {
            "duration_s": 1.0,
            "dt_s": 1.0,
            "dynamics": {"attitude": {"enabled": False}},
            "termination": {"earth_impact_enabled": False},
        },
        "outputs": {
            "output_dir": str(tmp_path),
            "mode": "interactive",
            "plots": {"enabled": False},
            "animations": {"enabled": False},
        },
    }

    payload = _run_single_config(scenario_config_from_dict(config))
    evidence = payload["flight_software_evidence_by_object"]["satellite"]

    commands = [command for output in evidence["outputs"] for command in output["commands"]]
    assert any(command["payload"]["schema"] == "thruster_pulse.v1" for command in commands)
    assert any(item["actuator_id"] == "rcs_x_plus" for item in evidence["realizations"])
    assert payload["summary"]["thrust_stats"]["satellite"]["total_dv_m_s"] > 0.0
