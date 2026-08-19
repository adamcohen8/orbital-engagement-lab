"""Transport-neutral public FSW authoring services."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml

from sim.api import SimulationWorkspace
from sim.config import load_simulation_yaml
from sim.installation.provenance import execution_provenance
from sim.installation.workspace import WORKSPACE_FILENAME, load_workspace
from sim.resource_limits import estimate_resource_requirements

from .candidate import ROOT, CandidateValidationError, clear_candidate_imports, load_candidate, validate_candidate
from .contracts import (
    AUTHORING_NON_CLAIMS,
    CAPABILITIES_SCHEMA_ID,
    EXECUTION_PLAN_SCHEMA_ID,
    RUN_MANIFEST_SCHEMA_ID,
    SCAFFOLD_RECEIPT_SCHEMA_ID,
    TEST_RESULT_SCHEMA_ID,
    VALIDATION_RECEIPT_SCHEMA_ID,
    WORK_ORDER_SCHEMA_ID,
    AuthoringIssue,
    AuthoringReceipt,
    effects,
    generated_utc,
    sha256_file,
    sha256_value,
)
from .scaffold import scaffold_candidate


def _write_json(path: Path, value: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)
    return path


def _artifact(path: Path, *, artifact_id: str) -> dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "path": str(path),
        "exists": path.is_file(),
        "sha256": sha256_file(path) if path.is_file() else "",
    }


def _execution_identity(candidate: Any) -> dict[str, Any]:
    payload = execution_provenance()
    manifest = candidate.workspace_root / WORKSPACE_FILENAME
    if payload.get("workspace") is None and manifest.is_file():
        workspace = load_workspace(manifest)
        payload["workspace"] = {
            "workspace_id": workspace["workspace_id"],
            "manifest_sha256": workspace["manifest_sha256"],
            "locked_version": workspace["engine"]["locked_version"],
            "contracts": workspace["contracts"],
        }
    return payload


def _default_output(candidate_id: str, operation: str, *, workspace_root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return workspace_root / "outputs" / "fsw_authoring" / candidate_id / f"{timestamp}_{operation}"


def _inside_workspace(path: Path, workspace_root: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(workspace_root)
    except ValueError as exc:
        raise PermissionError(f"{label} must remain inside the authorized workspace.") from exc
    return resolved


def _new_output(path: str | Path | None, *, candidate_id: str, operation: str, workspace_root: Path) -> Path:
    output = Path(path).expanduser() if path is not None else _default_output(
        candidate_id, operation, workspace_root=workspace_root
    )
    if not output.is_absolute():
        output = workspace_root / output
    output = _inside_workspace(output, workspace_root, label="FSW authoring output")
    if output.exists():
        if output.is_symlink() or not output.is_dir() or any(output.iterdir()):
            raise FileExistsError(f"FSW authoring output directory must be new or empty: {output}")
    else:
        output.mkdir(parents=True)
    return output


def describe_capabilities() -> dict[str, Any]:
    return {
        "schema": CAPABILITIES_SCHEMA_ID,
        "status": "ready",
        "product": "OEL Public FSW Authoring Kit",
        "templates": [
            {"id": "adcs", "maturity": "starter", "external_network": False},
            {"id": "rpo", "maturity": "starter", "external_network": False},
        ],
        "candidate_kinds": ["python_stack"],
        "operations": ["describe", "doctor", "init", "inspect", "plan", "validate", "test", "smoke", "verify-receipt"],
        "private_operations": [
            "controller_bench",
            "tuning",
            "qualification",
            "baseline_promotion",
            "evidence_packaging",
            "external_process",
            "cfs_sil",
        ],
        "contracts": {
            "candidate": "oel.fsw_authoring.candidate.v1",
            "work_order": WORK_ORDER_SCHEMA_ID,
            "execution_plan": EXECUTION_PLAN_SCHEMA_ID,
            "validation_receipt": VALIDATION_RECEIPT_SCHEMA_ID,
            "test_result": TEST_RESULT_SCHEMA_ID,
            "run_manifest": RUN_MANIFEST_SCHEMA_ID,
        },
        "non_claims": list(AUTHORING_NON_CLAIMS),
    }


def doctor(*, workspace_root: str | Path = ROOT) -> dict[str, Any]:
    root = Path(workspace_root).expanduser().resolve()
    schema = Path(__file__).with_name("schemas") / "candidate.schema.json"
    checks = {
        "workspace_exists": root.is_dir(),
        "workspace_readable": os.access(root, os.R_OK),
        "workspace_writable": os.access(root, os.W_OK),
        "python_supported": sys.version_info >= (3, 10),
        "candidate_schema": schema.is_file(),
        "public_fsw_boundary": (ROOT / "sim/flight_software/contracts.py").is_file(),
        "deterministic_runtime": (ROOT / "run_simulation.py").is_file(),
    }
    return {
        "schema": "oel.fsw_authoring.doctor.v1",
        "status": "ready" if all(checks.values()) else "failed",
        "workspace_root": str(root),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "network_required": False,
        },
        "checks": checks,
        "non_claims": list(AUTHORING_NON_CLAIMS),
    }


def init_candidate(
    name: str,
    *,
    template: str = "adcs",
    workspace_root: str | Path = ROOT,
    output_dir: str | Path | None = None,
    class_name: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    result = scaffold_candidate(
        name,
        template=template,
        workspace_root=workspace_root,
        output_dir=output_dir,
        class_name=class_name,
        force=force,
    )
    receipt = AuthoringReceipt(
        schema=SCAFFOLD_RECEIPT_SCHEMA_ID,
        status="ready",
        operation="init",
        candidate=dict(result.candidate["candidate"]),
        effects=effects(writes=True),
        artifacts=tuple(
            _artifact(path, artifact_id=path.relative_to(result.root_dir).as_posix())
            for path in result.files_written
            if path.is_file()
        ),
        result={"root_dir": str(result.root_dir), "manifest_path": str(result.manifest_path)},
    ).to_dict()
    _write_json(result.root_dir / ".oel" / "fsw_authoring_scaffold_receipt.json", receipt)
    return receipt


def plan_workflow(
    manifest: str | Path,
    operation: str,
    *,
    workspace_root: str | Path = ROOT,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    candidate = load_candidate(manifest, workspace_root=workspace_root)
    supported = {"validate", "test", "smoke"}
    if operation not in supported:
        raise ValueError(f"Unsupported public FSW operation {operation!r}; choose one of {sorted(supported)}")
    executes = operation in {"test", "smoke"}
    destination_path = Path(output_dir).expanduser() if output_dir is not None else _default_output(
        candidate.candidate_id, operation, workspace_root=candidate.workspace_root
    )
    if not destination_path.is_absolute():
        destination_path = candidate.workspace_root / destination_path
    destination_path = _inside_workspace(
        destination_path, candidate.workspace_root, label="Planned FSW authoring output"
    )
    required_inputs = {
        "validate": [candidate.manifest_path],
        "test": [candidate.manifest_path, candidate.verification.component_suite],
        "smoke": [candidate.manifest_path, candidate.verification.smoke_case],
    }[operation]
    work_order = {
        "schema": WORK_ORDER_SCHEMA_ID,
        "status": "ready",
        "candidate": candidate.identity(),
        "operation": operation,
        "requested_output_dir": str(destination_path),
        "constraints": {
            "deterministic": True,
            "workers": 1,
            "network": False,
            "hosted_ai": False,
            "new_output_required": True,
            "private_orchestration": False,
        },
        "work_order_id": "",
    }
    work_order["work_order_id"] = sha256_value(
        {key: value for key, value in work_order.items() if key != "work_order_id"}
    )
    plan = {
        "schema": EXECUTION_PLAN_SCHEMA_ID,
        "status": "ready",
        "operation": operation,
        "candidate": candidate.identity(),
        "required_inputs": [str(item) for item in required_inputs],
        "work_order": work_order,
        "output_dir": str(destination_path),
        "effects": effects(writes=True, executes=executes),
        "approval_required": True,
        "source_trust_required": True,
        "network": False,
        "resource_plan": {"profile": "laptop-safe", "workers": 1, "bounded": True, "action": "proceed"},
        "plan_id": "",
        "non_claims": list(AUTHORING_NON_CLAIMS),
    }
    plan["plan_id"] = sha256_value({key: value for key, value in plan.items() if key != "plan_id"})
    return plan


def validate_candidate_service(
    manifest: str | Path,
    *,
    workspace_root: str | Path = ROOT,
    trusted_import: bool = False,
    write_receipt: bool = True,
    receipt_dir: str | Path | None = None,
) -> dict[str, Any]:
    try:
        candidate, issues, checks = validate_candidate(
            manifest,
            workspace_root=workspace_root,
            trusted_import=trusted_import,
        )
    except CandidateValidationError as exc:
        receipt = AuthoringReceipt(
            schema=VALIDATION_RECEIPT_SCHEMA_ID,
            status="invalid",
            operation="validate",
            effects=effects(writes=bool(write_receipt and receipt_dir is not None)),
            issues=exc.issues,
            result={"validation_id": "", "trusted_import": trusted_import, "manifest_path": str(manifest)},
        ).to_dict()
        if write_receipt and receipt_dir is not None:
            root = Path(workspace_root).expanduser().resolve()
            path = _inside_workspace(
                Path(receipt_dir).expanduser().resolve() / "fsw_validation_receipt.json",
                root,
                label="Validation receipt",
            )
            _write_json(path, receipt)
            receipt["artifacts"] = [_artifact(path, artifact_id="validation_receipt")]
        return receipt
    status = "ready" if not issues else "invalid"
    validation_id = sha256_value(
        {
            "candidate_sha256": candidate.candidate_sha256,
            "checks": checks,
            "issues": [item.to_dict() for item in issues],
        }
    )
    receipt = AuthoringReceipt(
        schema=VALIDATION_RECEIPT_SCHEMA_ID,
        status=status,
        operation="validate",
        candidate=candidate.identity(),
        effects=effects(writes=write_receipt),
        issues=tuple(issues),
        result={
            "validation_id": validation_id,
            "trusted_import": trusted_import,
            "execution_authorized": False,
            "manifest_path": str(candidate.manifest_path),
            "checks": checks,
        },
    ).to_dict()
    if write_receipt:
        path = (
            Path(receipt_dir).expanduser().resolve() / "fsw_validation_receipt.json"
            if receipt_dir is not None
            else candidate.manifest_path.parent / ".oel" / "fsw_validation_receipt.json"
        )
        path = _inside_workspace(path, candidate.workspace_root, label="Validation receipt")
        _write_json(path, receipt)
        receipt["artifacts"] = [_artifact(path, artifact_id="validation_receipt")]
    return receipt


def _require_validated(
    manifest: str | Path,
    *,
    workspace_root: str | Path,
    validation_id: str | None,
) -> tuple[Any, dict[str, Any]]:
    receipt = validate_candidate_service(
        manifest,
        workspace_root=workspace_root,
        trusted_import=True,
        write_receipt=False,
    )
    if receipt["status"] != "ready":
        raise CandidateValidationError(
            [
                AuthoringIssue(str(item["code"]), str(item["message"]), str(item.get("path", "")))
                for item in receipt["issues"]
            ]
        )
    actual = str(receipt["result"]["validation_id"])
    if validation_id not in (None, "") and validation_id != actual:
        raise PermissionError("The supplied validation_id is stale or does not match the current candidate.")
    return load_candidate(manifest, workspace_root=workspace_root), receipt


def _run_subprocess(
    command: list[str], *, cwd: Path, env: Mapping[str, str] | None = None, timeout_s: float = 300.0
) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=dict(env or os.environ),
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": 124,
            "stdout": str(exc.stdout or "")[-200_000:],
            "stderr": (str(exc.stderr or "") + f"\nComponent test timeout after {timeout_s:.1f} s.")[-200_000:],
            "timed_out": True,
        }
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-200_000:],
        "stderr": completed.stderr[-200_000:],
        "output_truncated": len(completed.stdout) > 200_000 or len(completed.stderr) > 200_000,
    }


def run_contract_tests(
    manifest: str | Path,
    *,
    workspace_root: str | Path = ROOT,
    output_dir: str | Path | None = None,
    validation_id: str | None = None,
) -> dict[str, Any]:
    candidate, validation = _require_validated(
        manifest, workspace_root=workspace_root, validation_id=validation_id
    )
    output = _new_output(
        output_dir, candidate_id=candidate.candidate_id, operation="test", workspace_root=candidate.workspace_root
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(candidate.workspace_root), env.get("PYTHONPATH", "")) if value
    )
    execution = _run_subprocess(
        [sys.executable, "-m", "pytest", str(candidate.verification.component_suite), "-q"],
        cwd=candidate.workspace_root,
        env=env,
    )
    packet = {
        "schema": TEST_RESULT_SCHEMA_ID,
        "status": "passed" if execution["returncode"] == 0 else "failed",
        "generated_utc": generated_utc(),
        "candidate": candidate.identity(),
        "manifest_path": str(candidate.manifest_path),
        "execution_provenance": _execution_identity(candidate),
        "validation_id": validation["result"]["validation_id"],
        "execution": execution,
        "effects": effects(writes=True, executes=True),
        "non_claims": list(AUTHORING_NON_CLAIMS),
    }
    path = _write_json(output / "fsw_test_results.json", packet)
    packet["artifacts"] = [_artifact(path, artifact_id="test_results")]
    return packet


def _materialize_scenario(candidate: Any, output: Path) -> Path:
    raw = yaml.safe_load(candidate.verification.smoke_case.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Candidate smoke scenario must be a YAML object.")
    raw.setdefault("outputs", {})["output_dir"] = str(output / "run")
    raw["outputs"].setdefault("review", {"enabled": True, "detail": "standard"})
    generated = output / "materialized_smoke.yaml"
    generated.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return generated


@contextmanager
def _candidate_import_path(workspace_root: Path) -> Iterator[None]:
    value = str(workspace_root)
    added = value not in sys.path
    if added:
        sys.path.insert(0, value)
    try:
        yield
    finally:
        if added:
            sys.path.remove(value)


def run_smoke(
    manifest: str | Path,
    *,
    workspace_root: str | Path = ROOT,
    output_dir: str | Path | None = None,
    validation_id: str | None = None,
) -> dict[str, Any]:
    candidate, validation = _require_validated(
        manifest, workspace_root=workspace_root, validation_id=validation_id
    )
    resource_estimate = asdict(estimate_resource_requirements(load_simulation_yaml(candidate.verification.smoke_case)))
    resource_estimate["action"] = "refuse" if resource_estimate["risk"] == "unsafe" else "proceed"
    if resource_estimate["action"] == "refuse":
        raise ValueError("Candidate smoke failed the OEL resource-safety preflight.")
    output = _new_output(
        output_dir, candidate_id=candidate.candidate_id, operation="smoke", workspace_root=candidate.workspace_root
    )
    scenario = _materialize_scenario(candidate, output)
    workspace = SimulationWorkspace(
        workspace_root=candidate.workspace_root,
        read_roots=(candidate.workspace_root,),
        write_roots=(candidate.workspace_root,),
    )
    with _candidate_import_path(candidate.workspace_root):
        clear_candidate_imports(candidate.source.entrypoint.module)
        result = workspace.run(scenario)
    run_output = Path(str(result.payload.get("output_dir") or output / "run"))
    packet = {
        "schema": RUN_MANIFEST_SCHEMA_ID,
        "status": "passed",
        "generated_utc": generated_utc(),
        "candidate": candidate.identity(),
        "manifest_path": str(candidate.manifest_path),
        "execution_provenance": _execution_identity(candidate),
        "validation_id": validation["result"]["validation_id"],
        "scenario_path": str(scenario),
        "scenario_sha256": sha256_file(scenario),
        "run_output_dir": str(run_output),
        "summary": dict(result.summary),
        "resource_plan": resource_estimate,
        "effects": effects(writes=True, executes=True),
        "non_claims": list(AUTHORING_NON_CLAIMS),
    }
    path = _write_json(output / "fsw_run_manifest.json", packet)
    packet["artifacts"] = [_artifact(path, artifact_id="run_manifest")]
    return packet


def verify_receipt(receipt_path: str | Path, *, workspace_root: str | Path = ROOT) -> dict[str, Any]:
    root = Path(workspace_root).expanduser().resolve()
    path = _inside_workspace(Path(receipt_path), root, label="Receipt")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("FSW authoring receipt must be a JSON object.")
    allowed = {VALIDATION_RECEIPT_SCHEMA_ID, TEST_RESULT_SCHEMA_ID, RUN_MANIFEST_SCHEMA_ID, SCAFFOLD_RECEIPT_SCHEMA_ID}
    schema = str(raw.get("schema", ""))
    if schema not in allowed:
        raise ValueError("Receipt is not a supported public FSW authoring artifact.")
    result = dict(raw.get("result", {}) or {})
    manifest_text = str(raw.get("manifest_path") or result.get("manifest_path") or "")
    candidate_current = False
    current_identity: dict[str, Any] = {}
    if manifest_text:
        manifest = _inside_workspace(Path(manifest_text), root, label="Receipt candidate manifest")
        current = load_candidate(manifest, workspace_root=root)
        current_identity = current.identity()
        expected = str(dict(raw.get("candidate", {}) or {}).get("candidate_sha256", ""))
        candidate_current = bool(expected and expected == current.candidate_sha256)
    artifact_results: list[dict[str, Any]] = []
    artifacts_current = True
    for item in list(raw.get("artifacts", []) or []):
        artifact = dict(item or {})
        artifact_path = _inside_workspace(Path(str(artifact.get("path", ""))), root, label="Receipt artifact")
        expected_hash = str(artifact.get("sha256", ""))
        current_hash = sha256_file(artifact_path) if artifact_path.is_file() else ""
        matches = bool(expected_hash and current_hash == expected_hash)
        artifact_results.append({"path": str(artifact_path), "exists": artifact_path.is_file(), "sha256_matches": matches})
        artifacts_current = artifacts_current and matches
    passed = bool(candidate_current and artifacts_current)
    return {
        "schema": "oel.fsw_authoring.receipt_verification.v1",
        "status": "passed" if passed else "failed",
        "receipt_path": str(path),
        "receipt_sha256": sha256_file(path),
        "receipt_schema": schema,
        "candidate_current": candidate_current,
        "current_candidate": current_identity,
        "artifacts_current": artifacts_current,
        "artifacts": artifact_results,
        "effects": effects(),
        "non_claims": list(AUTHORING_NON_CLAIMS),
    }


__all__ = [
    "describe_capabilities",
    "doctor",
    "init_candidate",
    "plan_workflow",
    "run_contract_tests",
    "run_smoke",
    "validate_candidate_service",
    "verify_receipt",
]
