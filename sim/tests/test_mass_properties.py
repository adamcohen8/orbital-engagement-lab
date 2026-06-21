from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from sim.config import scenario_config_from_dict, validate_scenario_plugins
from sim.digital_twin.mass_properties import (
    audit_mass_properties,
    import_mass_properties,
    mass_property_report_markdown,
    resolve_inertia_kg_m2,
    validate_mass_properties,
)
from sim.runtime_support import _create_satellite_runtime


class TestMassProperties(unittest.TestCase):
    def test_valid_mass_properties_audit_principal_moments(self):
        specs = {
            "mass_kg": 200.0,
            "mass_properties": {
                "mass_kg": 200.0,
                "center_of_mass_body_m": [0.1, -0.2, 0.3],
                "inertia_kg_m2": [[12.0, 0.1, 0.0], [0.1, 10.0, 0.0], [0.0, 0.0, 8.0]],
                "inertia_reference_point": "center_of_mass",
                "frame": "body",
                "source": "cad_export",
                "confidence": "high",
            },
        }

        result = validate_mass_properties(specs)
        audit = audit_mass_properties(specs)

        self.assertEqual(result.errors, [])
        self.assertEqual(audit.validation.errors, [])
        self.assertTrue(np.allclose(audit.center_of_mass_body_m, np.array([0.1, -0.2, 0.3])))
        self.assertIsNotNone(audit.principal_moments_kg_m2)
        self.assertEqual(audit.source, "cad_export")

    def test_invalid_mass_properties_fail_strict_validation(self):
        cfg = scenario_config_from_dict(
            {
                "rocket": {"enabled": False},
                "target": {
                    "enabled": True,
                    "specs": {
                        "mass_kg": 200.0,
                        "mass_properties": {
                            "inertia_kg_m2": [[1.0, 4.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                            "source": "cad_export",
                            "confidence": "high",
                        },
                    },
                },
                "simulator": {"duration_s": 20.0, "dt_s": 1.0},
            }
        )

        errs = validate_scenario_plugins(cfg)

        self.assertTrue(any("mass_properties.inertia_kg_m2" in err and "symmetric" in err for err in errs))

    def test_explicit_invalid_inertia_does_not_fall_back_at_runtime(self):
        specs = {
            "mass_properties": {
                "inertia_kg_m2": [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
            }
        }

        with self.assertRaisesRegex(ValueError, "principal moments"):
            resolve_inertia_kg_m2(specs)

    def test_explicit_inertia_with_unknown_reference_is_rejected_at_runtime(self):
        specs = {
            "mass_properties": {
                "inertia_kg_m2": [[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]],
                "inertia_reference_point": "unknown",
                "frame": "body",
            }
        }

        with self.assertRaisesRegex(ValueError, "center_of_mass"):
            resolve_inertia_kg_m2(specs)

    def test_missing_inertia_keeps_default_compatibility(self):
        inertia = resolve_inertia_kg_m2({"mass_kg": 200.0})

        self.assertTrue(np.allclose(inertia, np.diag([120.0, 100.0, 80.0])))

    def test_invalid_mass_property_metadata_fails_without_inertia(self):
        specs = {"mass_properties": {"center_of_mass_body_m": [0.0, 0.0], "confidence": "certain"}}

        with self.assertRaisesRegex(ValueError, "center_of_mass_body_m"):
            resolve_inertia_kg_m2(specs)

    def test_runtime_retains_mass_properties_block(self):
        cfg = scenario_config_from_dict(
            {
                "chaser": {
                    "enabled": True,
                    "kind": "satellite",
                    "specs": {
                        "mass_kg": 200.0,
                        "mass_properties": {
                            "center_of_mass_body_m": [0.1, 0.0, 0.0],
                            "inertia_kg_m2": [[12.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 8.0]],
                            "inertia_reference_point": "center_of_mass",
                            "source": "user_supplied",
                            "confidence": "medium",
                        },
                    },
                },
                "target": {"enabled": False},
                "simulator": {"duration_s": 1.0, "dt_s": 1.0},
            }
        )

        runtime = _create_satellite_runtime("chaser", cfg.chaser, cfg, np.random.default_rng(1))

        self.assertEqual(runtime.mass_properties["center_of_mass_body_m"], [0.1, 0.0, 0.0])
        self.assertTrue(np.allclose(runtime.dynamics.inertia_kg_m2, np.diag([12.0, 10.0, 8.0])))

    def test_import_mass_properties_normalizes_generic_json(self):
        with TemporaryDirectory() as td:
            path = Path(td) / "cad_mass.json"
            path.write_text(
                json.dumps(
                    {
                        "mass": 42.0,
                        "center_of_mass": [0.1, 0.2, 0.3],
                        "inertia": [[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]],
                    }
                ),
                encoding="utf-8",
            )

            result = import_mass_properties(path)

        self.assertEqual(result.snippet["mass_kg"], 42.0)
        self.assertEqual(result.snippet["mass_properties"]["center_of_mass_body_m"], [0.1, 0.2, 0.3])
        self.assertEqual(result.audit.validation.errors, [])

    def test_mass_property_report_includes_warnings_and_principal_moments(self):
        specs = {
            "mass_kg": 10.0,
            "mass_properties": {
                "mass_kg": 11.0,
                "inertia_kg_m2": [[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]],
                "inertia_reference_point": "center_of_mass",
                "source": "cad_export",
                "confidence": "high",
            },
        }

        report = mass_property_report_markdown(specs)

        self.assertIn("WARNING", report)
        self.assertIn("Principal moments", report)
        self.assertIn("cad_export", report)

    def test_import_mass_properties_cli_writes_yaml_and_report(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            source = root / "cad_mass.json"
            output = root / "mass_properties.yaml"
            report = root / "report.md"
            source.write_text(
                json.dumps(
                    {
                        "mass": 42.0,
                        "center_of_mass": [0.0, 0.0, 0.0],
                        "inertia": [[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]],
                    }
                ),
                encoding="utf-8",
            )
            cmd = [
                sys.executable,
                "tools/import_mass_properties.py",
                str(source),
                "--output",
                str(output),
                "--report",
                str(report),
                "--summary",
            ]
            completed = subprocess.run(
                cmd,
                cwd=Path(__file__).resolve().parents[2],
                check=True,
                text=True,
                capture_output=True,
            )

            self.assertTrue(output.is_file())
            self.assertTrue(report.is_file())
            self.assertIn('"mass_kg": 42.0', completed.stdout)
            self.assertIn("Mass Properties Audit", report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
