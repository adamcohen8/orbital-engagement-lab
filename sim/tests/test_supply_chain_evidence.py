from __future__ import annotations

import json
from pathlib import Path

from tools.generate_python_sbom import build_sbom, write_sbom


def test_generate_python_sbom_contains_project_metadata() -> None:
    sbom = build_sbom(project_name="orbital-engagement-lab")

    assert sbom["bomFormat"] == "CycloneDX"
    assert sbom["specVersion"] == "1.5"
    assert dict(sbom["metadata"])["component"]["name"] == "orbital-engagement-lab"
    assert any(component["name"].lower() == "numpy" for component in list(sbom["components"]))


def test_write_sbom_creates_json_file(tmp_path: Path) -> None:
    output = write_sbom(tmp_path / "sbom.cdx.json")

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["bomFormat"] == "CycloneDX"
    assert saved["components"]
