from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.generate_dependency_evidence import build_dependency_evidence, write_dependency_evidence


def _pip_report(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "version": "1",
                "pip_version": "26.0",
                "install": [
                    {
                        "download_info": {
                            "url": "https://files.pythonhosted.org/packages/numpy-2.4.6-cp314-cp314-win_amd64.whl",
                            "archive_info": {"hashes": {"sha256": "abc123"}},
                        },
                        "metadata": {"name": "numpy", "version": "2.4.6"},
                        "requested": True,
                    },
                    {
                        "download_info": {"url": "file:///Users/example/Orbital%20Engagement%20Lab"},
                        "metadata": {"name": "orbital-engagement-lab", "version": "0.22.2"},
                        "requested": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_dependency_evidence_records_wheel_tags_and_constraints_digest(tmp_path: Path) -> None:
    constraints = tmp_path / "py314.txt"
    constraints.write_text("numpy==2.4.6\n", encoding="utf-8")
    report = _pip_report(tmp_path / "pip-install-report.json")

    payload = build_dependency_evidence(install_report=report, constraints_file=constraints)

    assert payload["schema_version"] == 1
    assert payload["resolver"]["pip_version"] == "26.0"
    assert payload["constraints"]["sha256"] == hashlib.sha256(constraints.read_bytes()).hexdigest()
    numpy = payload["packages"][0]
    assert numpy["name"] == "numpy"
    assert numpy["artifact_type"] == "wheel"
    assert numpy["wheel_tags"] == {"python": "cp314", "abi": "cp314", "platform": "win_amd64"}
    assert numpy["source_url"].startswith("https://files.pythonhosted.org/")


def test_dependency_evidence_redacts_local_source_url_and_writes_json(tmp_path: Path) -> None:
    constraints = tmp_path / "py311.txt"
    constraints.write_text("numpy==2.4.6\n", encoding="utf-8")
    report = _pip_report(tmp_path / "pip-install-report.json")

    output = write_dependency_evidence(
        tmp_path / "wheel-inventory.json",
        install_report=report,
        constraints_file=constraints,
    )

    saved = json.loads(output.read_text(encoding="utf-8"))
    project = next(item for item in saved["packages"] if item["name"] == "orbital-engagement-lab")
    assert project["source_url"] == "<local-source>"


def test_dependency_evidence_merges_build_dependency_report(tmp_path: Path) -> None:
    constraints = tmp_path / "py311.txt"
    constraints.write_text("numpy==2.4.6\nwheel==0.46.3\n", encoding="utf-8")
    report = _pip_report(tmp_path / "pip-install-report.json")
    build_report = tmp_path / "build-install-report.json"
    build_report.write_text(
        json.dumps(
            {
                "version": "1",
                "pip_version": "26.0",
                "install": [
                    {
                        "download_info": {
                            "url": "https://files.pythonhosted.org/packages/wheel-0.46.3-py3-none-any.whl",
                            "archive_info": {"hashes": {"sha256": "wheel123"}},
                        },
                        "metadata": {"name": "wheel", "version": "0.46.3"},
                        "requested": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = build_dependency_evidence(
        install_report=report,
        constraints_file=constraints,
        additional_install_reports=[build_report],
    )

    wheel = next(item for item in payload["packages"] if item["name"] == "wheel")
    assert wheel["artifact"] == "wheel-0.46.3-py3-none-any.whl"
    assert wheel["sha256"] == "wheel123"
    assert payload["resolver"]["additional_report_paths"] == ["build-install-report.json"]
