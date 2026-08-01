from __future__ import annotations

import csv
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs" / "operations" / "evidence" / "oel_mcp_sdk_v2"
PROFILE = "mcp-v2.0.0-macos-arm64-py311"
PUBLIC_REGISTRY = ROOT / "integrations" / "oel_mcp" / "public_registry.py"


def _normalized_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _registry_at_commit_or_current(commit: str) -> str:
    """Read historical private evidence when available, else the exported registry."""
    if (ROOT / ".git").exists():
        commit_check = subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if commit_check.returncode == 0:
            return subprocess.run(
                ["git", "show", f"{commit}:integrations/oel_mcp/public_registry.py"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
    return PUBLIC_REGISTRY.read_text(encoding="utf-8")


def test_mcp_v2_evidence_records_one_complete_exact_profile() -> None:
    freeze = (EVIDENCE / f"{PROFILE}-freeze.txt").read_text(encoding="utf-8").splitlines()
    hashes = (EVIDENCE / f"{PROFILE}-wheelhouse.sha256").read_text(encoding="utf-8").splitlines()
    with (EVIDENCE / f"{PROFILE}-licenses.csv").open(newline="", encoding="utf-8") as stream:
        licenses = list(csv.DictReader(stream))
    audit = json.loads((EVIDENCE / f"{PROFILE}-pip-audit.json").read_text(encoding="utf-8"))
    sbom = json.loads((EVIDENCE / f"{PROFILE}.cdx.json").read_text(encoding="utf-8"))

    assert len(freeze) == 30
    assert "mcp==2.0.0" in freeze
    assert "mcp-types==2.0.0" in freeze
    assert "pip==26.2" in freeze
    assert "setuptools==83.0.0" in freeze

    assert len(hashes) == 30
    assert all(re.fullmatch(r"[0-9a-f]{64}  \S+\.whl", line) for line in hashes)
    assert any(line.endswith("  mcp-2.0.0-py3-none-any.whl") for line in hashes)

    assert len(licenses) == 30
    assert all(row["license_expression"] for row in licenses)
    assert all(row["license_file_present"] == "true" for row in licenses)

    assert len(audit["dependencies"]) == 30
    assert audit["fixes"] == []
    assert all(dependency["vulns"] == [] for dependency in audit["dependencies"])

    assert sbom["bomFormat"] == "CycloneDX"
    assert sbom["specVersion"] == "1.6"
    assert len(sbom["components"]) == 30
    assert any(
        component["name"].lower() == "mcp" and component["version"] == "2.0.0"
        for component in sbom["components"]
    )

    freeze_names = {_normalized_name(line.partition("==")[0]) for line in freeze}
    assert freeze_names == {_normalized_name(row["name"]) for row in licenses}
    assert freeze_names == {_normalized_name(dependency["name"]) for dependency in audit["dependencies"]}
    assert freeze_names == {_normalized_name(component["name"]) for component in sbom["components"]}


def test_mcp_v2_interoperability_and_inspector_audit_evidence() -> None:
    interop = json.loads((EVIDENCE / "interop-2026-07-31.json").read_text(encoding="utf-8"))
    inspector_audit = json.loads(
        (EVIDENCE / "inspector-2.0.0-audit-2026-07-31.json").read_text(encoding="utf-8")
    )

    assert interop["status"] == "passed"
    assert interop["sdk_version"] == "2.0.0"
    assert interop["protocol_revision"] == "2026-07-28"
    assert set(interop["checks"]) == {"official_sdk_stdio", "inspector", "codex", "claude"}
    assert all(check["status"] == "passed" for check in interop["checks"].values())
    assert interop["checks"]["official_sdk_stdio"]["error_handling"]["code"] == -32602
    assert interop["checks"]["official_sdk_stdio"]["error_handling"]["passed"] is True
    assert interop["checks"]["official_sdk_stdio"]["lifecycle"]["clean_shutdowns"] == 2
    assert interop["checks"]["official_sdk_stdio"]["lifecycle"]["ping"] == (
        "not_supported_by_protocol_revision_2026-07-28"
    )
    assert inspector_audit["package"] == "@modelcontextprotocol/inspector@2.0.0"
    assert inspector_audit["scope"] == "test_only_not_packaged"
    assert inspector_audit["vulnerabilities"]["total"] == 0


def test_m3_package_resource_lifecycle_and_host_evidence() -> None:
    evidence = json.loads((EVIDENCE / "interop-m3-2026-07-31.json").read_text(encoding="utf-8"))

    assert evidence["status"] == "passed"
    assert evidence["milestone"] == "M3_supported_read_validate_mcp"
    assert evidence["checks"]["wheel_install"] == {
        "console_entry_point": "oel-mcp",
        "default_adapter": "sdk",
        "packaged_operator_guide": True,
        "packaged_resource_module": True,
        "rollback_adapter": "legacy",
        "status": "passed",
    }
    assert evidence["checks"]["official_sdk_stdio"]["resources_read"] == 4
    assert evidence["checks"]["official_sdk_stdio"]["lifecycle"]["clean_shutdowns"] == 2
    assert evidence["checks"]["inspector"]["status"] == "passed"
    assert evidence["checks"]["codex"]["status"] == "passed"
    assert evidence["checks"]["claude"]["status"] == "passed"


def test_m4_approval_execution_cancellation_and_host_evidence() -> None:
    evidence = json.loads((EVIDENCE / "interop-m4-2026-07-31.json").read_text(encoding="utf-8"))

    assert evidence["status"] == "passed"
    assert evidence["milestone"] == "M4_public_execution_mcp"
    assert evidence["sdk_version"] == "2.0.0"
    assert evidence["protocol_revision"] == "2026-07-28"
    assert evidence["checks"]["official_sdk_stdio"]["tool_count"] == 9
    assert evidence["checks"]["inspector"]["tool_count"] == 9
    assert evidence["checks"]["codex"]["capability_count"] == 9
    assert evidence["checks"]["claude"]["capability_count"] == 9
    execution = evidence["checks"]["m4_execution"]
    assert execution["operator_approval_default"] == "deny"
    assert execution["validation_identity_bound_to_normalized_config"] is True
    assert execution["execution_authorized_by_validation"] is False
    assert execution["sdk_execution_cancellation_manifest"] == "cancelled"
    assert execution["supported_public_task_evidence_packet"] == "passed"

    registry = _registry_at_commit_or_current(evidence["oel_base_commit"])
    assert "oel.run_agent_task.v1" in registry


def test_m5_1_evidence_commit_contains_the_claimed_report_tools() -> None:
    evidence = json.loads((EVIDENCE / "m5-1-2026-07-31.json").read_text(encoding="utf-8"))

    assert evidence["status"] == "passed"
    assert evidence["release_evidence_status"] == "historical_superseded"
    assert evidence["regenerate_on_final_v0_23_0_commit"] is True
    registry = _registry_at_commit_or_current(evidence["oel_commit"])
    assert "oel.prepare_report_packet.v1" in registry
    assert "oel.audit_report.v1" in registry
