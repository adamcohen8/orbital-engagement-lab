import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from sim.config import load_simulation_yaml, scenario_config_from_dict


class TestScenarioYamlConfig(unittest.TestCase):
    def test_from_dict_parses_public_single_run_sections(self):
        cfg = scenario_config_from_dict(
            {
                "scenario_name": "unit_test",
                "scenario_description": "Unit test scenario",
                "rocket": {
                    "enabled": True,
                    "base_guidance": {"module": "sim.rocket.guidance", "class_name": "OpenLoopPitchProgramGuidance"},
                    "guidance_modifiers": [{"module": "sim.rocket.guidance", "class_name": "MaxQThrottleLimiterGuidance"}],
                },
                "chaser": {"enabled": False},
                "target": {"enabled": True},
                "simulator": {"duration_s": 120.0, "dt_s": 0.5},
                "outputs": {"output_dir": "outputs/test", "mode": "both", "plots": {"enabled": True}},
            }
        )
        self.assertEqual(cfg.scenario_name, "unit_test")
        self.assertEqual(cfg.scenario_description, "Unit test scenario")
        self.assertTrue(cfg.rocket.enabled)
        self.assertIsNotNone(cfg.rocket.base_guidance)
        self.assertEqual(len(cfg.rocket.guidance_modifiers), 1)
        self.assertFalse(cfg.chaser.enabled)
        self.assertFalse(cfg.monte_carlo.enabled)
        self.assertFalse(cfg.analysis.enabled)
        self.assertEqual(cfg.outputs.mode, "both")
        self.assertEqual(cfg.outputs.output_dir, "outputs/test")
        self.assertEqual(cfg.target.reference_orbit, {})

    def test_invalid_outputs_mode_raises(self):
        with self.assertRaises(ValueError):
            scenario_config_from_dict(
                {
                    "simulator": {"duration_s": 100.0, "dt_s": 1.0},
                    "outputs": {"mode": "bad_mode"},
                }
            )

    def test_string_booleans_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "target.enabled"):
            scenario_config_from_dict(
                {
                    "target": {"enabled": "false"},
                    "simulator": {"duration_s": 10.0, "dt_s": 1.0},
                }
            )

    def test_timing_must_be_finite_positive_and_commensurate(self):
        with self.assertRaisesRegex(ValueError, "simulator.duration_s must be an integer multiple"):
            scenario_config_from_dict(
                {
                    "simulator": {"duration_s": 2.5, "dt_s": 1.0},
                }
            )

    def test_public_template_yaml_loads_without_batch_analysis(self):
        try:
            import yaml  # noqa: F401
        except Exception:
            self.skipTest("PyYAML not installed in this environment.")
        root = Path(__file__).resolve().parents[2]
        cfg = load_simulation_yaml(root / "configs" / "simulation_template.yaml")
        self.assertTrue(cfg.target.enabled)
        self.assertFalse(cfg.monte_carlo.enabled)
        self.assertFalse(cfg.analysis.enabled)
        self.assertGreater(cfg.simulator.dt_s, 0.0)
        self.assertGreater(cfg.simulator.duration_s, 0.0)

    def test_plotting_demo_yaml_loads(self):
        try:
            import yaml  # noqa: F401
        except Exception:
            self.skipTest("PyYAML not installed in this environment.")
        root = Path(__file__).resolve().parents[2]
        cfg = load_simulation_yaml(root / "configs" / "plotting_rendezvous_demo.yaml")
        self.assertEqual(cfg.scenario_name, "plotting_rendezvous_demo")
        self.assertTrue(cfg.outputs.plots.get("enabled"))
        self.assertIn("rendezvous", list(cfg.outputs.plots.get("preset", [])))
        self.assertIn("estimation", list(cfg.outputs.plots.get("preset", [])))
        self.assertTrue(cfg.chaser.knowledge.get("targets"))

    def test_agent_preset_yaml_merges_specs_from_relative_path(self):
        try:
            import yaml  # noqa: F401
        except Exception:
            self.skipTest("PyYAML not installed in this environment.")

        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            preset_dir = base / "presets"
            preset_dir.mkdir()
            (preset_dir / "satellite.yaml").write_text(
                """
name: Test Satellite
preset_type: satellite
specs:
  dry_mass_kg: 260.0
  fuel_mass_kg: 40.0
  max_thrust_n: 35.0
""".lstrip(),
                encoding="utf-8",
            )
            scenario_path = base / "scenario.yaml"
            scenario_path.write_text(
                """
target:
  enabled: true
  preset: "presets/satellite.yaml"
  specs:
    dry_mass_kg: 180.0
    max_thrust_n: 12.0
simulator:
  duration_s: 10.0
  dt_s: 1.0
""".lstrip(),
                encoding="utf-8",
            )

            cfg = load_simulation_yaml(scenario_path)

        self.assertEqual(cfg.target.specs["dry_mass_kg"], 180.0)
        self.assertEqual(cfg.target.specs["fuel_mass_kg"], 40.0)
        self.assertEqual(cfg.target.specs["max_thrust_n"], 12.0)


if __name__ == "__main__":
    unittest.main()
