from __future__ import annotations

import pytest

from sim import SimulationConfig, SimulationSession
from sim.runtime.state_initialization import _rv_from_initial_state


def test_cr3bp_halo_initialization_rejects_unbounded_phase_work() -> None:
    with pytest.raises(ValueError, match="exceeding the limit"):
        _rv_from_initial_state(
            {
                "cr3bp_halo": {
                    "system": "earth_moon",
                    "family": "l1_northern",
                    "phase_time_s": 1.0e308,
                    "phase_substep_s": 120.0,
                }
            }
        )


def test_cr3bp_run_evidence_labels_rotating_state_and_command_frames() -> None:
    config = SimulationConfig.from_dict(
        {
            "scenario_name": "cr3bp_frame_metadata",
            "simulator": {
                "duration_s": 1.0,
                "dt_s": 1.0,
                "termination": {"earth_impact_enabled": False},
                "dynamics": {
                    "orbit": {"model": "cr3bp", "cr3bp_system": "earth_moon"},
                    "attitude": {"enabled": False},
                },
            },
            "objects": {
                "target": {
                    "enabled": True,
                    "kind": "satellite",
                    "initial_state": {
                        "cr3bp_halo": {
                            "system": "earth_moon",
                            "family": "l1_northern",
                        }
                    },
                }
            },
            "outputs": {
                "mode": "save",
                "stats": {"print_summary": False, "save_json": False, "save_full_log": False},
                "plots": {"enabled": False, "figure_ids": []},
                "animations": {"enabled": False, "types": []},
            },
        }
    )

    result = SimulationSession.from_config(config).run()

    metadata = result.payload["object_propagation"]["target"]
    assert result.payload["object_state_frames"]["target"] == "cr3bp_rotating"
    assert metadata["state_history_frame"] == "cr3bp_rotating"
    assert metadata["command_acceleration_frame"] == "cr3bp_rotating"
