import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from sim.config import load_simulation_yaml, scenario_config_from_dict


class TestScenarioYamlConfig(unittest.TestCase):
    def test_from_dict_parses_expected_sections(self):
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
                "monte_carlo": {
                    "enabled": True,
                    "iterations": 10,
                    "parallel_enabled": True,
                    "parallel_workers": 3,
                    "variations": [{"parameter_path": "simulator.dt_s", "mode": "choice", "options": [0.5, 1.0]}],
                },
            }
        )
        self.assertEqual(cfg.scenario_name, "unit_test")
        self.assertEqual(cfg.scenario_description, "Unit test scenario")
        self.assertTrue(cfg.rocket.enabled)
        self.assertIsNotNone(cfg.rocket.base_guidance)
        self.assertEqual(len(cfg.rocket.guidance_modifiers), 1)
        self.assertFalse(cfg.chaser.enabled)
        self.assertTrue(cfg.monte_carlo.enabled)
        self.assertEqual(cfg.monte_carlo.iterations, 10)
        self.assertTrue(cfg.monte_carlo.parallel_enabled)
        self.assertEqual(cfg.monte_carlo.parallel_workers, 3)
        self.assertEqual(len(cfg.monte_carlo.variations), 1)
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

        with self.assertRaisesRegex(ValueError, "outputs.plots.enabled"):
            scenario_config_from_dict(
                {
                    "simulator": {"duration_s": 10.0, "dt_s": 1.0},
                    "outputs": {"plots": {"enabled": "false"}},
                }
            )

    def test_timing_must_be_finite_positive_and_commensurate(self):
        with self.assertRaisesRegex(ValueError, "simulator.duration_s must be an integer multiple"):
            scenario_config_from_dict(
                {
                    "simulator": {"duration_s": 2.5, "dt_s": 1.0},
                }
            )

        with self.assertRaisesRegex(ValueError, "orbit_substep_s must be less than or equal"):
            scenario_config_from_dict(
                {
                    "simulator": {
                        "duration_s": 10.0,
                        "dt_s": 1.0,
                        "dynamics": {"orbit": {"orbit_substep_s": 2.0}},
                    },
                }
            )

        with self.assertRaisesRegex(ValueError, "simulator.dt_s must be an integer multiple"):
            scenario_config_from_dict(
                {
                    "simulator": {
                        "duration_s": 10.0,
                        "dt_s": 1.0,
                        "dynamics": {"attitude": {"attitude_substep_s": 0.3}},
                    },
                }
            )

    def test_analysis_section_normalizes_monte_carlo_and_sensitivity(self):
        mc_cfg = scenario_config_from_dict(
            {
                "scenario_name": "analysis_mc",
                "simulator": {"duration_s": 20.0, "dt_s": 1.0},
                "analysis": {
                    "enabled": True,
                    "study_type": "monte_carlo",
                    "execution": {"parallel_enabled": True, "parallel_workers": 4},
                    "monte_carlo": {
                        "iterations": 12,
                        "base_seed": 9,
                        "variations": [{"parameter_path": "simulator.dt_s", "options": [0.5, 1.0]}],
                    },
                },
            }
        )
        self.assertTrue(mc_cfg.analysis.enabled)
        self.assertEqual(mc_cfg.analysis.study_type, "monte_carlo")
        self.assertTrue(mc_cfg.monte_carlo.enabled)
        self.assertEqual(mc_cfg.monte_carlo.iterations, 12)
        self.assertTrue(mc_cfg.monte_carlo.parallel_enabled)
        self.assertEqual(mc_cfg.monte_carlo.parallel_workers, 4)

        sens_cfg = scenario_config_from_dict(
            {
                "scenario_name": "analysis_sensitivity",
                "simulator": {"duration_s": 20.0, "dt_s": 1.0},
                "analysis": {
                    "enabled": True,
                    "study_type": "sensitivity",
                    "execution": {"parallel_enabled": False, "parallel_workers": 0},
                    "metrics": ["summary.duration_s", "derived.closest_approach_km"],
                    "sensitivity": {
                        "method": "one_at_a_time",
                        "parameters": [{"parameter_path": "simulator.dt_s", "values": [0.5, 1.0]}],
                    },
                },
            }
        )
        self.assertTrue(sens_cfg.analysis.enabled)
        self.assertEqual(sens_cfg.analysis.study_type, "sensitivity")
        self.assertFalse(sens_cfg.monte_carlo.enabled)
        self.assertEqual(len(sens_cfg.analysis.sensitivity.parameters), 1)
        self.assertEqual(sens_cfg.analysis.metrics, ["summary.duration_s", "derived.closest_approach_km"])

        lhs_cfg = scenario_config_from_dict(
            {
                "scenario_name": "analysis_lhs",
                "simulator": {"duration_s": 20.0, "dt_s": 1.0},
                "analysis": {
                    "enabled": True,
                    "study_type": "sensitivity",
                    "sensitivity": {
                        "method": "lhs",
                        "samples": 8,
                        "seed": 13,
                        "parameters": [
                            {
                                "parameter_path": "simulator.dt_s",
                                "distribution": "uniform",
                                "low": 0.5,
                                "high": 1.5,
                            }
                        ],
                    },
                },
            }
        )
        self.assertEqual(lhs_cfg.analysis.sensitivity.method, "lhs")
        self.assertEqual(lhs_cfg.analysis.sensitivity.samples, 8)
        self.assertEqual(lhs_cfg.analysis.sensitivity.seed, 13)
        self.assertEqual(lhs_cfg.analysis.sensitivity.parameters[0].distribution, "uniform")
        self.assertAlmostEqual(float(lhs_cfg.analysis.sensitivity.parameters[0].low), 0.5)

    def test_template_yaml_loads(self):
        try:
            import yaml  # noqa: F401
        except Exception:
            self.skipTest("PyYAML not installed in this environment.")
        root = Path(__file__).resolve().parents[2]
        p = root / "configs" / "simulation_template.yaml"
        cfg = load_simulation_yaml(p)
        self.assertTrue(cfg.target.enabled)
        self.assertGreater(cfg.simulator.dt_s, 0.0)
        self.assertGreater(cfg.simulator.duration_s, 0.0)

    def test_plotting_demo_yaml_loads(self):
        try:
            import yaml  # noqa: F401
        except Exception:
            self.skipTest("PyYAML not installed in this environment.")
        root = Path(__file__).resolve().parents[2]
        p = root / "configs" / "plotting_rendezvous_demo.yaml"
        cfg = load_simulation_yaml(p)
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
  mass_properties:
    inertia_kg_m2:
      - [1.0, 0.0, 0.0]
      - [0.0, 2.0, 0.0]
      - [0.0, 0.0, 3.0]
knowledge:
  refresh_rate_s: 5.0
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
        self.assertEqual(cfg.target.specs["mass_properties"]["inertia_kg_m2"][2][2], 3.0)
        self.assertEqual(cfg.target.knowledge["refresh_rate_s"], 5.0)

    def test_agent_preset_yaml_mass_override_uses_mass_kg(self):
        try:
            import yaml  # noqa: F401
        except Exception:
            self.skipTest("PyYAML not installed in this environment.")

        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            preset_path = base / "satellite.yaml"
            preset_path.write_text(
                """
specs:
  dry_mass_kg: 260.0
  fuel_mass_kg: 40.0
  max_thrust_n: 35.0
""".lstrip(),
                encoding="utf-8",
            )
            cfg = scenario_config_from_dict(
                {
                    "target": {
                        "preset": "satellite.yaml",
                        "specs": {"mass_kg": 120.0},
                    },
                    "simulator": {"duration_s": 10.0, "dt_s": 1.0},
                },
                source_path=base / "scenario.yaml",
            )

        self.assertEqual(cfg.target.specs["mass_kg"], 120.0)
        self.assertNotIn("dry_mass_kg", cfg.target.specs)
        self.assertNotIn("fuel_mass_kg", cfg.target.specs)

    def test_agent_preset_yaml_name_resolves_from_builtin_objects(self):
        try:
            import yaml  # noqa: F401
        except Exception:
            self.skipTest("PyYAML not installed in this environment.")

        cfg = scenario_config_from_dict(
            {
                "target": {
                    "preset": "basic_satellite",
                    "specs": {"dry_mass_kg": 175.0, "fuel_mass_kg": 25.0},
                },
                "simulator": {"duration_s": 10.0, "dt_s": 1.0},
            }
        )
        self.assertEqual(cfg.target.specs["preset_satellite"], "BASIC_SATELLITE")
        self.assertEqual(cfg.target.specs["dry_mass_kg"], 175.0)
        self.assertEqual(cfg.target.specs["fuel_mass_kg"], 25.0)

    def test_missing_agent_preset_yaml_raises_clear_error(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            with self.assertRaisesRegex(FileNotFoundError, "Could not resolve target preset YAML"):
                scenario_config_from_dict(
                    {
                        "target": {"preset": "missing.yaml"},
                        "simulator": {"duration_s": 10.0, "dt_s": 1.0},
                    },
                    source_path=base / "scenario.yaml",
                )

    def test_missing_agent_sections_use_role_defaults(self):
        cfg = scenario_config_from_dict(
            {
                "scenario_name": "defaults_test",
                "simulator": {"duration_s": 10.0, "dt_s": 1.0},
                "outputs": {"mode": "save", "output_dir": "outputs/defaults_test"},
                "monte_carlo": {"enabled": False},
            }
        )
        self.assertFalse(cfg.rocket.enabled)
        self.assertFalse(cfg.chaser.enabled)
        self.assertTrue(cfg.target.enabled)

    def test_satellite_guidance_field_is_rejected(self):
        with self.assertRaises(ValueError):
            scenario_config_from_dict(
                {
                    "scenario_name": "stale_guidance",
                    "chaser": {
                        "enabled": True,
                        "guidance": {
                            "module": "sim.control.orbit.zero_controller",
                            "class_name": "ZeroController",
                        },
                    },
                    "simulator": {"duration_s": 10.0, "dt_s": 1.0},
                }
            )

    def test_algorithm_pointer_file_field_is_rejected(self):
        with self.assertRaises(ValueError):
            scenario_config_from_dict(
                {
                    "scenario_name": "file_pointer_rejected",
                    "target": {
                        "enabled": True,
                        "orbit_control": {
                            "module": "sim.control.orbit.zero_controller",
                            "class_name": "ZeroController",
                            "file": "plugins/custom_controller.py",
                        },
                    },
                    "simulator": {"duration_s": 10.0, "dt_s": 1.0},
                }
            )

    def test_simulator_nested_defaults_are_preserved(self):
        cfg = scenario_config_from_dict(
            {
                "scenario_name": "sim_defaults_test",
                "simulator": {"duration_s": 10.0, "dt_s": 1.0},
                "outputs": {"mode": "save", "output_dir": "outputs/sim_defaults_test"},
                "monte_carlo": {"enabled": False},
            }
        )
        self.assertEqual(cfg.simulator.plugin_validation.get("strict"), True)
        self.assertEqual(cfg.simulator.termination.get("earth_impact_enabled"), True)
        self.assertAlmostEqual(float(cfg.simulator.termination.get("earth_radius_km")), 6378.137, places=6)

    def test_target_reference_orbit_config_parses(self):
        cfg = scenario_config_from_dict(
            {
                "scenario_name": "target_reference_orbit",
                "target": {
                    "enabled": True,
                    "reference_orbit": {"enabled": True},
                },
                "simulator": {"duration_s": 10.0, "dt_s": 1.0},
            }
        )
        self.assertEqual(cfg.target.reference_orbit, {"enabled": True})

    def test_target_reference_orbit_requires_target_enabled(self):
        with self.assertRaises(ValueError):
            scenario_config_from_dict(
                {
                    "scenario_name": "target_reference_orbit_invalid",
                    "target": {
                        "enabled": False,
                        "reference_orbit": {"enabled": True},
                    },
                    "simulator": {"duration_s": 10.0, "dt_s": 1.0},
                }
            )


if __name__ == "__main__":
    unittest.main()
