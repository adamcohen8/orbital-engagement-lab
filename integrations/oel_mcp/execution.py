from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import Any, Callable

import yaml

from integrations.oel_mcp.policy import MCPPathPolicy
from sim.api import SimulationWorkspace
from sim.config import scenario_config_from_dict
from sim.resource_limits import apply_resource_profile_to_config_dict, estimate_resource_requirements
from sim.security import ConfigPathPolicy
from sim.security.sealed_mode import validate_sealed_mode

MAX_SCENARIO_BYTES = 2_000_000
M4_RESOURCE_PROFILES = ("laptop-safe", "standard")
EXECUTION_MANIFEST_NAME = "mcp_execution_manifest.json"
_SAFE_ARTIFACT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")


class MCPExecutionCancelled(RuntimeError):
    """Raised at a deterministic workflow callback boundary after MCP cancellation."""


@dataclass(frozen=True)
class ExecutionApprovalPolicy:
    write_approval_ids: frozenset[str] = frozenset()
    execution_approval_ids: frozenset[str] = frozenset()
    trust_approval_ids: frozenset[str] = frozenset()

    @classmethod
    def configured(cls) -> ExecutionApprovalPolicy:
        return cls(
            write_approval_ids=_approval_ids("OEL_MCP_WRITE_APPROVAL_IDS"),
            execution_approval_ids=_approval_ids("OEL_MCP_EXECUTION_APPROVAL_IDS"),
            trust_approval_ids=_approval_ids("OEL_MCP_TRUST_APPROVAL_IDS"),
        )

    def require(self, approval: dict[str, Any] | None, *, executes: bool) -> str:
        if not isinstance(approval, dict):
            raise PermissionError("Operator approval metadata is required for this operation.")
        approval_id = str(approval.get("approval_id", "")).strip()
        scope = str(approval.get("scope", "")).strip()
        expected_scope = "execute" if executes else "write"
        allowed = self.execution_approval_ids if executes else self.write_approval_ids
        if scope != expected_scope or not approval_id or approval_id not in allowed:
            raise PermissionError("The operation is not enabled by the server's operator approval policy.")
        return approval_id

    def require_trust(self, approval: dict[str, Any] | None) -> str:
        if not isinstance(approval, dict):
            raise PermissionError("Operator trust approval is required before importing scenario plugins.")
        approval_id = str(approval.get("approval_id", "")).strip()
        if (
            str(approval.get("scope", "")).strip() != "trust"
            or not approval_id
            or approval_id not in self.trust_approval_ids
        ):
            raise PermissionError("Plugin trust is not enabled by the server's operator approval policy.")
        return approval_id


@dataclass(frozen=True)
class PreparedScenario:
    source_path: Path
    output_dir: Path
    resource_profile: str
    raw_sha256: str
    normalized_sha256: str
    validation_id: str
    config: Any
    config_dict: dict[str, Any]


def prepare_scenario(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    resource_profile: str,
    path_policy: MCPPathPolicy,
) -> PreparedScenario:
    source = path_policy.resolve_read(config_path, kind="file")
    raw_bytes = _read_file_nofollow(source, maximum=MAX_SCENARIO_BYTES)
    output = path_policy.resolve_write(output_dir)
    profile = str(resource_profile).strip()
    if profile not in M4_RESOURCE_PROFILES:
        raise ValueError("M4 requires the laptop-safe or standard resource profile.")
    raw = yaml.safe_load(raw_bytes)
    if not isinstance(raw, dict):
        raise ValueError("Scenario config must be a YAML mapping.")
    prepared = apply_resource_profile_to_config_dict(raw, profile)
    outputs = dict(prepared.get("outputs", {}) or {})
    outputs["output_dir"] = str(output)
    review = dict(outputs.get("review", {}) or {})
    review["enabled"] = True
    review.setdefault("detail", "standard")
    outputs["review"] = review
    stats = dict(outputs.get("stats", {}) or {})
    stats["print_summary"] = False
    outputs["stats"] = stats
    prepared["outputs"] = outputs
    config_policy = ConfigPathPolicy.default(
        config_path=source,
        workspace_root=Path(__file__).resolve().parents[2],
        read_roots=path_policy.read_roots,
        write_roots=path_policy.write_roots,
        allow_config_dir_writes=False,
    )
    config = scenario_config_from_dict(prepared, source_path=source, path_policy=config_policy)
    sealed_errors = validate_sealed_mode(config)
    ai_report = dict(config.outputs.ai_report or {})
    if bool(ai_report.get("enabled", False)) and not bool(ai_report.get("dry_run", False)):
        sealed_errors.append(
            "outputs.ai_report.enabled: MCP execution forbids all AI-provider calls; "
            "disable the report or use dry_run: true."
        )
    ai_config = dict(config.outputs.ai_config or {})
    if ai_config and bool(ai_config.get("enabled", True)) and not bool(ai_config.get("dry_run", False)):
        sealed_errors.append(
            "outputs.ai_config.enabled: MCP execution forbids all AI-provider calls; "
            "disable the assistant or use dry_run: true."
        )
    if sealed_errors:
        raise ValueError("MCP sealed execution policy failed:\n- " + "\n- ".join(sealed_errors))
    normalized = config.to_dict()
    normalized_sha256 = _sha256_json(normalized)
    validation_id = f"oel-m4-validation-v1:{normalized_sha256}"
    return PreparedScenario(
        source_path=source,
        output_dir=output,
        resource_profile=profile,
        raw_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        normalized_sha256=normalized_sha256,
        validation_id=validation_id,
        config=config,
        config_dict=prepared,
    )


def validate_prepared_scenario(
    prepared: PreparedScenario,
    *,
    path_policy: MCPPathPolicy,
    trust_plugins: bool,
) -> dict[str, Any]:
    workspace = SimulationWorkspace(
        workspace_root=Path(__file__).resolve().parents[2],
        read_roots=path_policy.read_roots,
        write_roots=path_policy.write_roots,
        allow_config_dir_writes=False,
    )
    safe = workspace.validate(prepared.config, import_plugins=False)
    trusted: dict[str, Any] = {
        "ok": None,
        "status": "not_run",
        "reason": "safe_validation_failed" if not bool(safe.get("ok")) else "trust_not_granted",
    }
    if bool(safe.get("ok")) and trust_plugins:
        trusted = workspace.validate(prepared.config, import_plugins=True)
    trusted_ok = bool(trusted.get("ok")) if trust_plugins else False
    return {
        "status": "validated" if trusted_ok else "safe_only" if bool(safe.get("ok")) else "failed",
        "safe_validation": safe,
        "trusted_validation": trusted,
        "execution_ready": trusted_ok,
        "validation_id": prepared.validation_id if trusted_ok else "",
    }


def resource_estimate(prepared: PreparedScenario) -> dict[str, Any]:
    estimate = asdict(estimate_resource_requirements(prepared.config))
    estimate["action"] = (
        "refuse"
        if estimate["risk"] == "unsafe"
        else ("advisory" if estimate["risk"] in {"moderate", "heavy"} else "proceed")
    )
    return estimate


def require_safe_resource_estimate(estimate: dict[str, Any]) -> None:
    if str(estimate.get("risk")) == "unsafe" or str(estimate.get("action")) == "refuse":
        raise ValueError("The scenario failed the active resource-safety preflight.")


def ensure_new_output_dir(path: Path) -> None:
    _ensure_directory_nofollow(path, require_empty=True)


def write_materialized_config(prepared: PreparedScenario) -> Path:
    target = prepared.output_dir / "mcp_execution_config.yaml"
    _write_child_nofollow(
        prepared.output_dir,
        target.name,
        yaml.safe_dump(prepared.config_dict, sort_keys=False, allow_unicode=False).encode("utf-8"),
    )
    return target


def cancellation_callback(cancel_event: Event | None) -> Callable[..., None]:
    def check(*_args: Any, **_kwargs: Any) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise MCPExecutionCancelled("The MCP request was cancelled at a deterministic workflow boundary.")

    return check


def write_execution_manifest(
    output_dir: Path,
    payload: dict[str, Any],
    *,
    filename: str = EXECUTION_MANIFEST_NAME,
) -> Path:
    _ensure_directory_nofollow(output_dir, require_empty=False)
    target = output_dir / filename
    serialized = (json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n").encode("utf-8")
    _write_child_nofollow(output_dir, filename, serialized, atomic=True)
    return target


def _write_child_nofollow(
    directory: Path,
    filename: str,
    payload: bytes,
    *,
    atomic: bool = False,
) -> None:
    if Path(filename).name != filename:
        raise ValueError("MCP artifact filename must be a single path component.")
    directory_fd = _open_directory_nofollow(directory, create=False)
    temporary_name = filename
    try:
        if atomic:
            temporary_name = f".{filename}.{secrets.token_hex(12)}.tmp"
        file_flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
        if atomic:
            file_flags |= os.O_EXCL
        file_fd = os.open(temporary_name, file_flags, 0o600, dir_fd=directory_fd)
        with os.fdopen(file_fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if atomic:
            os.replace(
                temporary_name,
                filename,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
    finally:
        if atomic:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        os.close(directory_fd)


def _open_directory_nofollow(directory: Path, *, create: bool) -> int:
    """Open an absolute directory one component at a time without symlinks."""

    path = Path(directory)
    if not path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts[1:]):
        raise ValueError("MCP artifact directory must be an absolute normalized path")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    current_fd = os.open(os.path.sep, flags)
    try:
        for part in path.parts[1:]:
            try:
                next_fd = os.open(part, flags, dir_fd=current_fd)
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(part, mode=0o700, dir_fd=current_fd)
                next_fd = os.open(part, flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except Exception:
        os.close(current_fd)
        raise


def _ensure_directory_nofollow(directory: Path, *, require_empty: bool) -> None:
    directory_fd = _open_directory_nofollow(directory, create=True)
    try:
        if require_empty and os.listdir(directory_fd):
            raise FileExistsError("The approved output directory must be new or empty.")
    finally:
        os.close(directory_fd)


def _read_file_nofollow(path: Path, *, maximum: int) -> bytes:
    parent_fd = _open_directory_nofollow(path.parent, create=False)
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        file_fd = os.open(path.name, flags, dir_fd=parent_fd)
        try:
            size = os.fstat(file_fd).st_size
            if size > int(maximum):
                raise ValueError("Authorized input exceeds the MCP file-size budget.")
            chunks: list[bytes] = []
            remaining = int(maximum) + 1
            while remaining > 0:
                chunk = os.read(file_fd, min(1024 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            payload = b"".join(chunks)
            if len(payload) > int(maximum):
                raise ValueError("Authorized input exceeds the MCP file-size budget.")
            return payload
        finally:
            os.close(file_fd)
    finally:
        os.close(parent_fd)


def manifest_base(
    *,
    tool_id: str,
    approval_id: str,
    prepared: PreparedScenario | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "tool_id": tool_id,
        "approval_id": approval_id,
        "status": "running",
        "started_utc": _utc_now(),
        "completed_utc": None,
        "cancelled": False,
        "artifacts_complete": False,
    }
    if prepared is not None:
        payload.update(
            {
                "source_config_sha256": prepared.raw_sha256,
                "normalized_config_sha256": prepared.normalized_sha256,
                "validation_id": prepared.validation_id,
                "resource_profile": prepared.resource_profile,
                "output_dir": str(prepared.output_dir),
            }
        )
    return payload


def complete_manifest(
    manifest: dict[str, Any],
    *,
    status: str,
    cancelled: bool = False,
    artifacts: list[str] | None = None,
    error_type: str = "",
) -> dict[str, Any]:
    manifest.update(
        {
            "status": status,
            "completed_utc": _utc_now(),
            "cancelled": cancelled,
            "artifacts": list(artifacts or []),
            "artifacts_complete": status == "completed" and not cancelled,
        }
    )
    if error_type:
        manifest["error_type"] = error_type
    return manifest


def safe_artifact_id(value: str) -> str:
    artifact_id = str(value or "").strip()
    if not _SAFE_ARTIFACT_ID.fullmatch(artifact_id):
        raise ValueError("Artifact id must contain only letters, numbers, dot, underscore, or hyphen.")
    return artifact_id


def _approval_ids(name: str) -> frozenset[str]:
    return frozenset(item.strip() for item in os.environ.get(name, "").split(",") if item.strip())


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


__all__ = [
    "EXECUTION_MANIFEST_NAME",
    "M4_RESOURCE_PROFILES",
    "MAX_SCENARIO_BYTES",
    "ExecutionApprovalPolicy",
    "MCPExecutionCancelled",
    "PreparedScenario",
    "cancellation_callback",
    "complete_manifest",
    "ensure_new_output_dir",
    "manifest_base",
    "prepare_scenario",
    "require_safe_resource_estimate",
    "resource_estimate",
    "safe_artifact_id",
    "validate_prepared_scenario",
    "write_execution_manifest",
    "write_materialized_config",
]
