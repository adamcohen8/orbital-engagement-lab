from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from sim.digital_twin.package import SpacecraftTwinPackage

ROOT = Path(__file__).resolve().parents[2]
DEMO_TWIN = ROOT / "examples" / "twins" / "demo_sat" / "twin.yaml"


class TestSpacecraftTwinPackage(unittest.TestCase):
    def test_demo_twin_validates(self):
        package = SpacecraftTwinPackage.load(DEMO_TWIN)
        validation = package.validate()

        self.assertEqual(validation.errors, [])
        self.assertEqual(validation.missing_inputs, [])
        self.assertIsNotNone(validation.geometry_summary)
        self.assertEqual(validation.geometry_summary.sample_count, 6)

    def test_assembled_object_merges_mass_and_geometry_artifacts(self):
        package = SpacecraftTwinPackage.load(DEMO_TWIN)
        obj = package.assembled_object()
        specs = obj["specs"]

        self.assertEqual(obj["kind"], "satellite")
        self.assertEqual(specs["mass_kg"], 42.0)
        self.assertEqual(specs["mass_properties"]["confidence"], "high")
        self.assertEqual(specs["geometry"]["profile_path"], str((DEMO_TWIN.parent / "geometry_area_profile.json").resolve()))

    def test_missing_referenced_file_is_validation_error(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            twin = root / "twin.yaml"
            twin.write_text(
                "\n".join(
                    [
                        "schema: oel.spacecraft_twin.v0",
                        "object_id: missing_demo",
                        "object:",
                        "  path: missing_object.yaml",
                    ]
                ),
                encoding="utf-8",
            )

            validation = SpacecraftTwinPackage.load(twin).validate()

        self.assertTrue(any("object file does not exist" in err for err in validation.errors))

    def test_report_markdown_contains_inventory_and_geometry(self):
        package = SpacecraftTwinPackage.load(DEMO_TWIN)
        report = package.report_markdown()

        self.assertIn("Artifact Inventory", report)
        self.assertIn("Projected area min/mean/max", report)
        self.assertIn("Mass Properties", report)

    def test_write_object_yaml_emits_scenario_object_block(self):
        package = SpacecraftTwinPackage.load(DEMO_TWIN)
        with TemporaryDirectory() as td:
            output = Path(td) / "demo_sat_object.yaml"
            package.write_object_yaml(output)
            data = yaml.safe_load(output.read_text(encoding="utf-8"))

        self.assertIn("objects", data)
        self.assertIn("demo_sat", data["objects"])
        self.assertEqual(data["objects"]["demo_sat"]["specs"]["mass_kg"], 42.0)

    def test_build_spacecraft_twin_cli_writes_report_and_object_yaml(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            report = root / "report.md"
            object_yaml = root / "object.yaml"
            cmd = [
                sys.executable,
                "tools/build_spacecraft_twin.py",
                str(DEMO_TWIN),
                "--validate",
                "--report",
                "--report-path",
                str(report),
                "--emit-object-yaml",
                str(object_yaml),
                "--print-summary",
            ]
            completed = subprocess.run(cmd, cwd=ROOT, check=True, text=True, capture_output=True)

            self.assertTrue(report.is_file())
            self.assertTrue(object_yaml.is_file())
            self.assertIn('"ok": true', completed.stdout)
            self.assertIn("Spacecraft Twin Validation", report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
