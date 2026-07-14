from __future__ import annotations

import json
import re
from pathlib import Path

from tools.generate_python_sbom import build_sbom, write_sbom


def _source_project_version() -> str:
    pyproject = (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)
    assert match is not None
    return match.group(1)


def test_generate_python_sbom_contains_project_metadata() -> None:
    sbom = build_sbom(project_name="orbital-engagement-lab")

    assert sbom["bomFormat"] == "CycloneDX"
    assert sbom["specVersion"] == "1.5"
    assert dict(sbom["metadata"])["component"]["name"] == "orbital-engagement-lab"
    assert dict(sbom["metadata"])["component"]["version"] == _source_project_version()
    assert any(component["name"].lower() == "numpy" for component in list(sbom["components"]))


def test_write_sbom_creates_json_file(tmp_path: Path) -> None:
    output = write_sbom(tmp_path / "sbom.cdx.json")

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["bomFormat"] == "CycloneDX"
    assert saved["components"]


def test_security_procurement_docs_match_project_version() -> None:
    root = Path(__file__).resolve().parents[2]
    release_line = f"v{_source_project_version()}"

    for rel_path in ("SECURITY.md", "docs/security/supply-chain.md", "docs/project/product_maturity_roadmap.md"):
        path = root / rel_path
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        assert release_line in text, rel_path
