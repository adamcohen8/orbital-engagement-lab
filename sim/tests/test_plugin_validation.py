from pathlib import Path
import sys
import tempfile
import unittest

from sim.config import scenario_config_from_dict, validate_scenario_plugins


class TestPluginValidation(unittest.TestCase):
    def test_valid_plugins_pass(self):
        cfg = scenario_config_from_dict(
            {
                "rocket": {
                    "enabled": True,
                    "base_guidance": {
                        "module": "sim.rocket.guidance",
                        "class_name": "OpenLoopPitchProgramGuidance",
                        "params": {},
                    },
                    "guidance_modifiers": [
                        {
                            "module": "sim.rocket.guidance",
                            "class_name": "MaxQThrottleLimiterGuidance",
                            "params": {"max_q_pa": 45000.0, "min_throttle": 0.1},
                        }
                    ],
                    "orbit_control": {"module": "sim.control.orbit.zero_controller", "class_name": "ZeroController", "params": {}},
                    "attitude_control": {
                        "module": "sim.control.attitude.zero_torque",
                        "class_name": "ZeroTorqueController",
                        "params": {},
                    },
                },
                "chaser": {
                    "enabled": True,
                    "mission_strategy": {
                        "module": "sim.mission.modules",
                        "class_name": "PursuitMissionStrategy",
                        "params": {},
                    },
                    "mission_execution": {
                        "module": "sim.mission.modules",
                        "class_name": "ControllerPointingExecution",
                        "params": {},
                    },
                    "orbit_control": {"module": "sim.control.orbit.zero_controller", "class_name": "ZeroController", "params": {}},
                    "attitude_control": {
                        "module": "sim.control.attitude.zero_torque",
                        "class_name": "ZeroTorqueController",
                        "params": {},
                    },
                },
                "target": {"enabled": False},
                "simulator": {"duration_s": 20.0, "dt_s": 1.0},
            }
        )
        errs = validate_scenario_plugins(cfg)
        self.assertEqual(errs, [])

    def test_satellite_guidance_is_rejected_at_parse_time(self):
        with self.assertRaises(ValueError):
            scenario_config_from_dict(
                {
                    "rocket": {"enabled": False},
                    "chaser": {
                        "enabled": True,
                        "guidance": {
                            "module": "sim.control.orbit.zero_controller",
                            "class_name": "ZeroController",
                            "params": {},
                        },
                    },
                    "target": {"enabled": False},
                    "simulator": {"duration_s": 20.0, "dt_s": 1.0},
                }
            )

    def test_invalid_plugins_fail(self):
        cfg = scenario_config_from_dict(
            {
                "rocket": {
                    "enabled": True,
                    "base_guidance": {"module": "sim.control.orbit.zero_controller", "class_name": "ZeroController", "params": {}},
                },
                "chaser": {"enabled": False},
                "target": {"enabled": False},
                "simulator": {"duration_s": 20.0, "dt_s": 1.0},
            }
        )
        errs = validate_scenario_plugins(cfg)
        self.assertTrue(any("rocket.base_guidance" in e for e in errs))

    def test_named_object_plugins_are_validated(self):
        cfg = scenario_config_from_dict(
            {
                "objects": {
                    "blue_one": {
                        "kind": "satellite",
                        "enabled": True,
                        "orbit_control": {
                            "module": "sim.rocket.guidance",
                            "class_name": "OpenLoopPitchProgramGuidance",
                            "params": {},
                        },
                    }
                },
                "simulator": {"duration_s": 20.0, "dt_s": 1.0},
            }
        )

        errs = validate_scenario_plugins(cfg)
        self.assertTrue(any("objects.blue_one.orbit_control" in e for e in errs))

    def test_validation_does_not_construct_plugin_classes(self):
        with tempfile.TemporaryDirectory() as td:
            plugin_path = Path(td) / "constructor_side_effect_plugin.py"
            plugin_path.write_text(
                "\n".join(
                    [
                        "class ConstructorRaisesController:",
                        "    def __init__(self, *args, **kwargs):",
                        "        raise RuntimeError('constructor should not run during validation')",
                        "",
                        "    def act(self, **kwargs):",
                        "        return {}",
                    ]
                ),
                encoding="utf-8",
            )
            sys.path.insert(0, td)
            try:
                cfg = scenario_config_from_dict(
                    {
                        "rocket": {"enabled": False},
                        "chaser": {
                            "enabled": True,
                            "orbit_control": {
                                "module": "constructor_side_effect_plugin",
                                "class_name": "ConstructorRaisesController",
                                "params": {"needs_runtime_context": True},
                            },
                        },
                        "target": {"enabled": False},
                        "simulator": {"duration_s": 20.0, "dt_s": 1.0},
                    }
                )

                self.assertEqual(validate_scenario_plugins(cfg), [])
            finally:
                sys.path.remove(td)
                sys.modules.pop("constructor_side_effect_plugin", None)


if __name__ == "__main__":
    unittest.main()
