from __future__ import annotations

from sim import SimulationConfig, SimulationSession
from sim.flight_software import ClockScale, ClockTag, TaskRelease, canonical_json_bytes, canonical_loads
from sim.flight_software.events import PeriodicSchedule


def test_sensor_tasks_actuators_and_outputs_keep_independent_cadences() -> None:
    schedules = {
        "gyro": PeriodicSchedule("gyro", 100_000_000),
        "orbit_navigation": PeriodicSchedule("orbit_navigation", 1_000_000_000),
        "attitude_control": PeriodicSchedule("attitude_control", 200_000_000),
        "actuator_update": PeriodicSchedule("actuator_update", 50_000_000),
        "output": PeriodicSchedule("output", 500_000_000),
    }
    releases = {name: schedule.releases_through(1_000_000_000) for name, schedule in schedules.items()}
    assert len(releases["gyro"]) == 11
    assert len(releases["orbit_navigation"]) == 2
    assert len(releases["attitude_control"]) == 6
    assert len(releases["actuator_update"]) == 21
    assert len(releases["output"]) == 3


def test_output_schedule_does_not_create_controller_releases() -> None:
    control = PeriodicSchedule("control", 300, first_release_ns=100)
    output = PeriodicSchedule("output", 100)
    assert output.releases_through(600) == (0, 100, 200, 300, 400, 500, 600)
    assert control.releases_through(600) == (100, 400)


def test_task_release_carries_modeled_duration_and_budget_without_host_runtime() -> None:
    release = TaskRelease(
        "attitude_control",
        ClockTag("clock", 10, 1_000_000, ClockScale.ONBOARD),
        modeled_execution_duration_ns=50_000,
        execution_budget_ns=40_000,
    )
    assert canonical_loads(canonical_json_bytes(release)) == release
    assert release.modeled_execution_duration_ns > release.execution_budget_ns  # type: ignore[operator]


def test_runtime_places_fsw_releases_inside_larger_attitude_and_orbit_steps(tmp_path) -> None:
    config = SimulationConfig.from_dict(
        {
            "scenario_name": "fsw_multirate_runtime",
            "objects": {
                "sat": {
                    "kind": "satellite",
                    "initial_state": {"default_circular_earth": True},
                    "flight_software": {
                        "stack": "fsw.passive",
                        "hardware_profile": "hardware.passive.v1",
                        "task_period_s": 0.1,
                    },
                }
            },
            "simulator": {
                "duration_s": 1.0,
                "dt_s": 1.0,
                "termination": {"earth_impact_enabled": False},
                "dynamics": {
                    "orbit": {"orbit_substep_s": 1.0},
                    "attitude": {"enabled": True, "attitude_substep_s": 0.25},
                },
            },
            "outputs": {
                "mode": "save",
                "output_dir": str(tmp_path),
                "stats": {"print_summary": False, "save_json": False, "save_full_log": False},
                "plots": {"enabled": False, "figure_ids": []},
                "animations": {"enabled": False, "types": []},
            },
        }
    )
    result = SimulationSession.from_config(config).run()
    invocations = result.payload["flight_software_evidence_by_object"]["sat"]["invocations"]
    assert [row["invocation_time_ns"] for row in invocations] == [index * 100_000_000 for index in range(11)]
    assert {row["missed_task_releases"] for row in invocations} == {0}
    assert {row["missed_sensor_releases"] for row in invocations} == {0}
