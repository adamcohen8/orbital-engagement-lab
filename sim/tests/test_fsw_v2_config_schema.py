from __future__ import annotations

import pytest

from sim.config import scenario_config_from_dict


def _scenario(object_body: dict) -> dict:
    return {
        "scenario_name": "fsw_v2_config",
        "objects": {
            "sat": {
                "kind": "satellite",
                "initial_state": {"default_circular_earth": True},
                **object_body,
            }
        },
        "simulator": {"duration_s": 1.0, "dt_s": 1.0},
    }


def test_v2_stack_and_hardware_selection_are_typed() -> None:
    config = scenario_config_from_dict(
        _scenario(
            {
                "flight_software": {
                    "stack": "fsw.passive",
                    "hardware_profile": "hardware.passive.v1",
                    "task_period_s": 0.25,
                }
            }
        )
    )
    section = config.objects["sat"].flight_software
    assert section is not None
    assert section.stack == "fsw.passive"
    assert section.task_period_s == 0.25


def test_v2_config_rejects_legacy_satellite_fields_with_migration_guidance() -> None:
    with pytest.raises(ValueError, match="Move goals into flight_software.mission_load"):
        scenario_config_from_dict(
            _scenario(
                {
                    "flight_software": {"stack": "fsw.passive", "hardware_profile": "hardware.passive.v1"},
                    "orbit_control": {"builtin": "orbit.zero"},
                }
            )
        )


def test_v1_only_satellite_fields_are_rejected_instead_of_silently_migrated() -> None:
    with pytest.raises(ValueError, match="removed GNC v1 satellite field"):
        scenario_config_from_dict(_scenario({"orbit_control": {"builtin": "orbit.zero"}}))


def test_satellite_without_onboard_behavior_defaults_to_v2_passive_stack() -> None:
    section = scenario_config_from_dict(_scenario({})).objects["sat"].flight_software
    assert section is not None
    assert section.stack == "fsw.passive"
    assert section.hardware_profile == "hardware.passive.v1"


def test_stack_hardware_compatibility_is_validated() -> None:
    with pytest.raises(ValueError, match="not compatible"):
        scenario_config_from_dict(
            _scenario(
                {
                    "flight_software": {
                        "stack": "fsw.low_thrust_reference",
                        "hardware_profile": "hardware.reaction_wheels.v1",
                    }
                }
            )
        )


def test_custom_stack_pointer_receives_only_explicit_params() -> None:
    config = scenario_config_from_dict(
        _scenario(
            {
                "flight_software": {
                    "module": "example.custom_stack",
                    "class_name": "CustomStack",
                    "params": {"gain": 2.0},
                    "hardware_profile": "hardware.custom.v1",
                }
            }
        )
    )
    section = config.objects["sat"].flight_software
    assert section is not None
    assert section.params == {"gain": 2.0}
    assert not ({"truth", "world_truth", "dynamics", "environment"} & set(section.params))


def test_reaction_wheel_profile_rejects_nonunit_axes() -> None:
    with pytest.raises(ValueError, match="wheel_axes_body must contain unit vectors"):
        scenario_config_from_dict(
            _scenario(
                {
                    "flight_software": {
                        "stack": "fsw.attitude_reference",
                        "hardware_profile": "hardware.reaction_wheels.v1",
                        "params": {
                            "wheel_axes_body": [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                        },
                    }
                }
            )
        )
