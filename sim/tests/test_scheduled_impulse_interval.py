from __future__ import annotations

import numpy as np

from sim.api import SimulationConfig, SimulationSession
from sim.control.orbit.scheduled_impulse import ScheduledImpulseController
from sim.single_run_support import _segment_sphere_entry_fraction


def test_scheduled_impulse_delivers_off_grid_interval_overlap() -> None:
    controller = ScheduledImpulseController(
        start_time_s=0.5,
        duration_s=0.25,
        delta_v_eci_m_s=np.array([1.0, 0.0, 0.0]),
    )

    controller.set_actuation_interval(0.0, 1.0)
    command = controller.act(None, t_s=0.0, budget_ms=2.0)

    delivered_delta_v_m_s = command.thrust_eci_km_s2 * 1.0 * 1000.0
    np.testing.assert_allclose(delivered_delta_v_m_s, np.array([1.0, 0.0, 0.0]), atol=1e-12)
    assert command.mode_flags["scheduled_impulse_interval_overlap_s"] == 0.25


def test_scheduled_impulse_delivery_is_step_partition_invariant() -> None:
    controller = ScheduledImpulseController(
        start_time_s=0.4,
        duration_s=0.5,
        delta_v_eci_m_s=np.array([0.0, 2.0, 0.0]),
    )
    delivered = np.zeros(3)
    for start, end in ((0.0, 0.3), (0.3, 0.6), (0.6, 1.0)):
        controller.set_actuation_interval(start, end)
        command = controller.act(None, t_s=start, budget_ms=2.0)
        delivered += command.thrust_eci_km_s2 * (end - start) * 1000.0

    np.testing.assert_allclose(delivered, np.array([0.0, 2.0, 0.0]), atol=1e-12)


def test_scheduled_impulse_survives_finer_attitude_substeps(tmp_path) -> None:
    config = SimulationConfig.from_dict(
        {
            "scenario_name": "scheduled_impulse_full_engine",
            "objects": {
                "vehicle": {
                    "kind": "satellite",
                    "enabled": True,
                    "specs": {"mass_kg": 100.0},
                    "initial_state": {
                        "position_eci_km": [7000.0, 0.0, 0.0],
                        "velocity_eci_km_s": [0.0, 7.5, 0.0],
                    },
                    "flight_software": {
                        "stack": "fsw.orbit_reference",
                        "hardware_profile": "hardware.ideal_wrench.v1",
                        "task_period_s": 0.25,
                        "params": {
                            "max_acceleration_m_s2": 10.0,
                            "scheduled_burns": [
                                {
                                    "start_time_s": 0.5,
                                    "duration_s": 0.25,
                                    "frame": "eci",
                                    "delta_v_m_s": [1.0, 0.0, 0.0],
                                }
                            ],
                        },
                    },
                }
            },
            "simulator": {
                "duration_s": 1.0,
                "dt_s": 1.0,
                "dynamics": {
                    "orbit": {"orbit_substep_s": 1.0},
                    "attitude": {"enabled": True, "attitude_substep_s": 0.1},
                },
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
    )

    result = SimulationSession.from_config(config).run()
    delivered = np.array(result.applied_thrust["vehicle"][1], dtype=float) * 1.0 * 1000.0

    np.testing.assert_allclose(delivered, np.array([1.0, 0.0, 0.0]), atol=1e-12)


def test_impact_surface_detects_between_endpoint_crossing() -> None:
    fraction = _segment_sphere_entry_fraction(
        np.array([-7000.0, 0.0, 0.0]),
        np.array([7000.0, 0.0, 0.0]),
        6378.137,
    )
    assert fraction is not None
    assert 0.0 < fraction < 0.5
