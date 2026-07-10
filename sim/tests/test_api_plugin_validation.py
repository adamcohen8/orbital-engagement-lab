from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from sim import HostedSimulationWorkspace, SimulationSession


def _invalid_plugin_config(output_dir: Path) -> dict:
    return {
        "scenario_name": "api_plugin_validation",
        "target": {
            "enabled": True,
            "specs": {"mass_kg": 100.0},
            "initial_state": {
                "position_eci_km": [7000.0, 0.0, 0.0],
                "velocity_eci_km_s": [0.0, 7.5, 0.0],
            },
            "orbit_control": {
                "module": "sim.tests.missing_controller_module",
                "class_name": "MissingController",
            },
        },
        "rocket": {"enabled": False},
        "chaser": {"enabled": False},
        "simulator": {
            "duration_s": 1.0,
            "dt_s": 1.0,
            "termination": {"earth_impact_enabled": False},
            "dynamics": {"attitude": {"enabled": False}},
        },
        "outputs": {
            "output_dir": str(output_dir),
            "mode": "save",
            "stats": {"print_summary": False, "save_json": False, "save_full_log": False},
            "plots": {"enabled": False, "figure_ids": []},
            "animations": {"enabled": False, "types": []},
        },
    }


class TestApiPluginValidation(unittest.TestCase):
    def test_hosted_validation_never_imports_plugin_modules(self) -> None:
        config = _invalid_plugin_config(Path("outputs/hosted_validation"))
        config["target"]["orbit_control"] = {
            "module": "sim.control.orbit.zero_controller",
            "class_name": "ZeroController",
        }
        workspace = HostedSimulationWorkspace()

        with patch("sim.config.plugin_validation.importlib.import_module", side_effect=AssertionError("imported")):
            report = workspace.validate(config)

        self.assertTrue(report["ok"], report["errors"])

    def test_hosted_workspace_can_run_a_valid_single_scenario(self) -> None:
        with TemporaryDirectory() as tmpdir:
            config = _invalid_plugin_config(Path(tmpdir))
            config["target"].pop("orbit_control")

            result = HostedSimulationWorkspace().run(config)

        self.assertEqual(result.num_steps, 2)
        self.assertEqual(result.summary["scenario_name"], "api_plugin_validation")

    def test_hosted_workspace_confines_mapping_outputs_when_root_is_explicit(self) -> None:
        with TemporaryDirectory() as root, TemporaryDirectory() as outside:
            config = _invalid_plugin_config(Path(outside))
            config["target"].pop("orbit_control")

            with self.assertRaisesRegex(ValueError, "cannot write outside allowed config roots"):
                HostedSimulationWorkspace(workspace_root=root).run(config)

    def test_hosted_workspace_rejects_untrusted_nested_plugin(self) -> None:
        config = _invalid_plugin_config(Path("outputs/hosted_nested_plugin"))
        config["target"].pop("orbit_control")
        config["target"]["mission_strategy"] = {
            "module": "sim.mission.modules",
            "class_name": "MissionExecutiveStrategy",
            "params": {
                "nested": {
                    "module": "pathlib",
                    "class_name": "Path",
                    "params": {},
                }
            },
        }

        report = HostedSimulationWorkspace().validate(config)

        self.assertFalse(report["ok"])
        self.assertIn("pathlib", "\n".join(report["errors"]))

    def test_session_reset_validates_plugins_when_strict(self) -> None:
        with TemporaryDirectory() as tmpdir:
            session = SimulationSession.from_config(_invalid_plugin_config(Path(tmpdir)))

            with self.assertRaisesRegex(ValueError, "Plugin validation failed"):
                session.reset()

    def test_requested_plugin_constructor_failure_does_not_silently_coast(self) -> None:
        with TemporaryDirectory() as tmpdir:
            config = _invalid_plugin_config(Path(tmpdir))
            config["target"]["orbit_control"] = {
                "module": "sim.control.orbit.hcw_pd",
                "class_name": "HCWPDController",
                "params": {},
            }
            session = SimulationSession.from_config(config)

            with self.assertRaisesRegex(RuntimeError, "Failed to construct requested plugin"):
                session.run()


if __name__ == "__main__":
    unittest.main()
