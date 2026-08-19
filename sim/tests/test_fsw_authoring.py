from __future__ import annotations

import json
from pathlib import Path

import yaml

from sim.fsw_authoring import inspect_candidate
from sim.fsw_authoring.services import (
    describe_capabilities,
    init_candidate,
    plan_workflow,
    run_contract_tests,
    run_smoke,
    validate_candidate_service,
    verify_receipt,
)

ROOT = Path(__file__).resolve().parents[2]


def _scaffold(root: Path, *, template: str = "adcs", name: str = "public_candidate") -> Path:
    receipt = init_candidate(name, template=template, workspace_root=root)
    assert receipt["status"] == "ready"
    return Path(receipt["result"]["manifest_path"])


def test_public_authoring_capabilities_stop_before_private_verification() -> None:
    capabilities = describe_capabilities()

    assert capabilities["product"] == "OEL Public FSW Authoring Kit"
    assert {item["id"] for item in capabilities["templates"]} == {"adcs", "rpo"}
    assert capabilities["candidate_kinds"] == ["python_stack"]
    assert "controller_bench" in capabilities["private_operations"]
    assert "qualification" in capabilities["private_operations"]
    assert "cfs" + "_sil" in capabilities["private_operations"]
    assert "qualify" not in capabilities["operations"]


def test_public_scaffold_is_safely_inspectable_and_content_bound(tmp_path: Path) -> None:
    manifest = _scaffold(tmp_path)

    inspected = inspect_candidate(manifest, workspace_root=tmp_path)
    assert inspected["status"] == "ready"
    assert inspected["safe_inspection"] is True
    assert inspected["candidate_code_imported"] is False
    assert inspected["candidate_code_executed"] is False
    assert inspected["private_operations_available"] is False
    scaffold_receipt = manifest.parent / ".oel/fsw_authoring_scaffold_receipt.json"
    assert verify_receipt(scaffold_receipt, workspace_root=tmp_path)["status"] == "passed"

    plan = plan_workflow(manifest, "smoke", workspace_root=tmp_path)
    assert plan["effects"]["executes"] is True
    assert plan["source_trust_required"] is True
    assert plan["work_order"]["constraints"]["workers"] == 1
    assert plan["work_order"]["constraints"]["network"] is False

    original_hash = inspected["candidate"]["candidate_sha256"]
    stack_path = next(manifest.parent.joinpath("stacks").glob("*_stack.py"))
    stack_path.write_text(stack_path.read_text(encoding="utf-8") + "\n# revision\n", encoding="utf-8")
    changed = inspect_candidate(manifest, workspace_root=tmp_path)
    assert changed["candidate"]["candidate_sha256"] != original_hash


def test_public_authoring_rejects_simulator_truth_imports_without_importing_candidate(tmp_path: Path) -> None:
    manifest = _scaffold(tmp_path)
    stack_path = next(manifest.parent.joinpath("stacks").glob("*_stack.py"))
    stack_path.write_text(
        stack_path.read_text(encoding="utf-8") + "\nfrom sim.runtime import models as forbidden_truth\n",
        encoding="utf-8",
    )

    receipt = validate_candidate_service(
        manifest,
        workspace_root=tmp_path,
        trusted_import=False,
        write_receipt=False,
    )

    assert receipt["status"] == "invalid"
    assert receipt["result"]["checks"]["candidate_import"] == "not_run"
    assert receipt["result"]["checks"]["truth_firewall"] == "failed"
    assert any(issue["code"] == "truth_boundary_import" for issue in receipt["issues"])


def test_public_authoring_adcs_lifecycle_tests_smoke_and_receipt_currentness(tmp_path: Path) -> None:
    manifest = _scaffold(tmp_path, template="adcs", name="verified_adcs")
    validation_dir = tmp_path / "validation"
    validation_dir.mkdir()
    validated = validate_candidate_service(
        manifest,
        workspace_root=tmp_path,
        trusted_import=True,
        receipt_dir=validation_dir,
    )
    assert validated["status"] == "ready"
    assert all(value == "passed" for value in validated["result"]["checks"].values())
    validation_id = validated["result"]["validation_id"]

    precreated_test_output = tmp_path / "test-output"
    precreated_test_output.mkdir()
    tested = run_contract_tests(
        manifest,
        workspace_root=tmp_path,
        output_dir=precreated_test_output,
        validation_id=validation_id,
    )
    assert tested["status"] == "passed"
    assert tested["execution"]["returncode"] == 0

    smoked = run_smoke(
        manifest,
        workspace_root=tmp_path,
        output_dir=tmp_path / "smoke-output",
        validation_id=validation_id,
    )
    assert smoked["status"] == "passed"
    assert smoked["summary"]["samples"] == 21
    assert smoked["summary"]["runtime_profile"]["executor"]["object_step_backend"] == "serial"
    assert Path(smoked["summary"]["review_sqlite_path"]).is_file()

    receipt_path = validation_dir / "fsw_validation_receipt.json"
    assert verify_receipt(receipt_path, workspace_root=tmp_path)["status"] == "passed"
    source = next(manifest.parent.joinpath("stacks").glob("*_stack.py"))
    source.write_text(source.read_text(encoding="utf-8") + "\n# stale receipt\n", encoding="utf-8")
    stale = verify_receipt(receipt_path, workspace_root=tmp_path)
    assert stale["status"] == "failed"
    assert stale["candidate_current"] is False


def test_public_rpo_scaffold_passes_trusted_contract_validation(tmp_path: Path) -> None:
    manifest = _scaffold(tmp_path, template="rpo", name="public_rpo")
    receipt = validate_candidate_service(
        manifest,
        workspace_root=tmp_path,
        trusted_import=True,
        write_receipt=False,
    )

    assert receipt["status"] == "ready"
    assert receipt["candidate"]["onboard_contract"] == "oel.fsw.boundary.v1"
    smoke = yaml.safe_load(Path(manifest.parent / "configs/public_rpo_smoke.yaml").read_text(encoding="utf-8"))
    assert smoke["outputs"]["plots"]["enabled"] is False
    assert smoke["outputs"]["animations"]["enabled"] is False
    assert smoke["simulator"]["duration_s"] == 20.0


def test_public_authoring_package_has_no_private_product_dependencies() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "sim/fsw_authoring").rglob("*.py"))
    )

    forbidden = (
        "sim.fswdk",
        "sim.controller_lab",
        "sim.gnc_workbench",
        "sim.licensing",
        "integrations." + "cfs_" + "sil",
        "integrations.oel_mcp",
    )
    assert not any(name in source for name in forbidden)

    schema = json.loads((ROOT / "sim/fsw_authoring/schemas/candidate.schema.json").read_text(encoding="utf-8"))
    assert schema["properties"]["kind"]["const"] == "python_stack"
    assert "qualification" not in json.dumps(schema).lower()
