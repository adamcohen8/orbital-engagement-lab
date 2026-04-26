from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from sim import SimulationSession


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
    def test_session_reset_validates_plugins_when_strict(self) -> None:
        with TemporaryDirectory() as tmpdir:
            session = SimulationSession.from_config(_invalid_plugin_config(Path(tmpdir)))

            with self.assertRaisesRegex(ValueError, "Plugin validation failed"):
                session.reset()


if __name__ == "__main__":
    unittest.main()
