from __future__ import annotations

import hashlib
import os
import threading
import traceback
import uuid
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import yaml

from sim.config import SimulationScenarioConfig
from sim.config.scenario.loader import scenario_config_from_dict
from sim.execution.service import SimulationExecutionService
from sim.installation.provenance import execution_provenance
from sim.public_api.workspace import SimulationWorkspace
from sim.resource_limits import (
    ResourceEstimate,
    apply_resource_profile_to_config_dict,
    estimate_resource_requirements,
)
from sim.security import ConfigPathPolicy

from .models import (
    ForegroundResult,
    LifecycleError,
    RunHandle,
    RunIdentity,
    RunPhase,
    RunPolicyError,
    RunState,
    canonical_json_bytes,
    local_host_sha256,
    sha256_json,
)
from .store import EVENTS_NAME, LIFECYCLE_DIR_NAME, MANIFEST_NAME, LifecycleStore

MAX_CONFIG_BYTES = 2 * 1024 * 1024
SUPPORTED_RESOURCE_PROFILE = "laptop-safe"
OWNER_HEARTBEAT_INTERVAL_S = 1.0
OWNER_STALE_AFTER_S = 15.0


class ExecutionService(Protocol):
    def study_type(self, cfg: SimulationScenarioConfig) -> str: ...

    def run_single(
        self,
        cfg: SimulationScenarioConfig,
        *,
        step_callback: Any = None,
    ) -> dict[str, Any]: ...

    def wrap_single_file_payload(
        self,
        *,
        payload: dict[str, Any],
        cfg: SimulationScenarioConfig,
        config_path: str | Path,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class PreparedLifecycleRun:
    source_path: Path
    output_dir: Path
    config: SimulationScenarioConfig
    normalized_config_sha256: str
    validation_id: str
    source_config_sha256: str
    resource_profile: str
    resource_plan: dict[str, Any]
    resource_plan_sha256: str


class _OwnerHeartbeat:
    def __init__(self, store: LifecycleStore, run_id: str) -> None:
        self.store = store
        self.run_id = run_id
        self.owner_token = str(uuid.uuid4())
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False
        self._stopped = False

    def start(self) -> None:
        self.store.create_owner(
            self.run_id,
            owner_token=self.owner_token,
            pid=os.getpid(),
            host_sha256=local_host_sha256(),
            heartbeat_interval_s=OWNER_HEARTBEAT_INTERVAL_S,
            stale_after_s=OWNER_STALE_AFTER_S,
        )
        self._thread = threading.Thread(
            target=self._run,
            name=f"oel-run-heartbeat-{self.run_id[:8]}",
            daemon=True,
        )
        self._started = True
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(OWNER_HEARTBEAT_INTERVAL_S):
            try:
                self.store.heartbeat_owner(self.run_id, owner_token=self.owner_token)
            except (OSError, LifecycleError):
                return

    def stop(self) -> None:
        if not self._started or self._stopped:
            return
        self._stopped = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=OWNER_HEARTBEAT_INTERVAL_S * 2.0)
        try:
            self.store.heartbeat_owner(
                self.run_id,
                owner_token=self.owner_token,
                status="exited",
            )
        except (OSError, LifecycleError):
            pass


def _load_yaml_mapping(path: Path) -> tuple[bytes, dict[str, Any]]:
    content = path.read_bytes()
    if len(content) > MAX_CONFIG_BYTES:
        raise RunPolicyError("Lifecycle scenario configuration exceeds the 2 MiB limit.")
    try:
        root = yaml.safe_load(content) or {}
    except yaml.YAMLError as exc:
        raise RunPolicyError("Lifecycle scenario configuration is not valid YAML.") from exc
    if not isinstance(root, dict):
        raise RunPolicyError("Lifecycle scenario configuration must contain a mapping.")
    return content, dict(root)


def _force_lifecycle_outputs(root: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    result = dict(root)
    outputs = dict(result.get("outputs", {}) or {})
    outputs["output_dir"] = str(output_dir)
    outputs["mode"] = "save"
    outputs["review"] = {**dict(outputs.get("review", {}) or {}), "enabled": True, "detail": "standard"}
    outputs["ai_report"] = {**dict(outputs.get("ai_report", {}) or {}), "enabled": False}
    outputs["ai_config"] = {**dict(outputs.get("ai_config", {}) or {}), "enabled": False}
    stats = dict(outputs.get("stats", {}) or {})
    stats["print_summary"] = False
    outputs["stats"] = stats
    result["outputs"] = outputs
    return result


def prepare_foreground_run(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    output_root: str | Path,
    workspace_root: str | Path | None = None,
    resource_profile: str = SUPPORTED_RESOURCE_PROFILE,
    resource_estimator: Callable[[Any], ResourceEstimate] = estimate_resource_requirements,
) -> PreparedLifecycleRun:
    if str(resource_profile) != SUPPORTED_RESOURCE_PROFILE:
        raise RunPolicyError(
            f"Lifecycle v1 supports only the {SUPPORTED_RESOURCE_PROFILE!r} resource profile."
        )
    source = Path(config_path).expanduser().resolve()
    if not source.is_file():
        raise RunPolicyError("Lifecycle scenario configuration was not found.")
    root_path = Path(output_root).expanduser().resolve()
    target = Path(output_dir).expanduser()
    if not target.is_absolute():
        target = root_path / target
    target = target.resolve()
    if target == root_path or root_path not in target.parents:
        raise RunPolicyError("Lifecycle output must be a child of the authorized output root.")

    source_bytes, raw = _load_yaml_mapping(source)
    profiled = apply_resource_profile_to_config_dict(raw, resource_profile)
    materialized = _force_lifecycle_outputs(profiled, target)
    policy = ConfigPathPolicy.default(
        config_path=source,
        workspace_root=workspace_root or Path(__file__).resolve().parents[3],
        read_roots=(source.parent,),
        write_roots=(root_path,),
        allow_config_dir_writes=False,
    )
    try:
        config = scenario_config_from_dict(materialized, source_path=source, path_policy=policy)
    except (TypeError, ValueError) as exc:
        raise RunPolicyError(f"Lifecycle scenario validation failed: {exc}") from exc

    execution = SimulationExecutionService()
    if execution.study_type(config) != "single_run":
        raise RunPolicyError("Lifecycle v1 accepts one deterministic foreground scenario, not analysis workflows.")
    workspace = SimulationWorkspace(
        workspace_root=workspace_root or Path(__file__).resolve().parents[3],
        read_roots=(source.parent,),
        write_roots=(root_path,),
        allow_config_dir_writes=False,
    )
    validation = workspace.validate_candidate_config(config, trust_plugins=True)
    if not bool(validation.get("ok", False)):
        errors = list(dict(validation.get("trusted_validation", {}) or {}).get("errors", []) or [])
        if not errors:
            errors = list(dict(validation.get("safe_validation", {}) or {}).get("errors", []) or [])
        detail = "; ".join(str(item) for item in errors[:3]) or "validation did not pass"
        raise RunPolicyError(f"Lifecycle scenario validation failed: {detail}")

    normalized = config.to_dict()
    normalized_digest = hashlib.sha256(canonical_json_bytes(normalized)).hexdigest()
    estimate = resource_estimator(config)
    plan = asdict(estimate)
    plan["action"] = estimate.action
    if estimate.action == "refuse":
        raise RunPolicyError("Lifecycle resource preflight refused the scenario as unsafe.")
    return PreparedLifecycleRun(
        source_path=source,
        output_dir=target,
        config=config,
        normalized_config_sha256=normalized_digest,
        validation_id=f"oel-run-validation-v1:{normalized_digest}",
        source_config_sha256=hashlib.sha256(source_bytes).hexdigest(),
        resource_profile=resource_profile,
        resource_plan=plan,
        resource_plan_sha256=sha256_json(plan),
    )


def _identity(prepared: PreparedLifecycleRun, store: LifecycleStore) -> RunIdentity:
    run_id = str(uuid.uuid4())
    provenance = execution_provenance()
    return RunIdentity(
        run_id=run_id,
        manifest_ref=store.manifest_ref(run_id, prepared.output_dir),
        normalized_config_sha256=prepared.normalized_config_sha256,
        validation_id=prepared.validation_id,
        source_config_sha256=prepared.source_config_sha256,
        source_config_ref=prepared.source_path.name,
        output_dir=str(prepared.output_dir),
        resource_profile=prepared.resource_profile,
        resource_plan_sha256=prepared.resource_plan_sha256,
        engine_version=str(provenance.get("engine_version") or "unknown"),
        engine_edition=str(provenance.get("edition") or "public"),
        installation_disposition=str(provenance.get("installation_disposition") or "developer"),
        source_revision=os.environ.get("OEL_SOURCE_REVISION", "unknown").strip() or "unknown",
        authorization_disposition="explicit_local_invocation",
        plugin_trust_disposition="trusted_local_cli",
    )


def _artifact_inventory(output_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    excluded = {MANIFEST_NAME, EVENTS_NAME, ".run.lock"}
    files: list[dict[str, Any]] = []
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        if path.parent == output_dir / LIFECYCLE_DIR_NAME and path.name in excluded:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        files.append(
            {
                "path": path.relative_to(output_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    review_path = output_dir / "review" / "run.sqlite"
    return (
        {"complete": True, "file_count": len(files), "files": files},
        {
            "available": review_path.is_file(),
            "evidence_complete": "available" if review_path.is_file() else "missing",
            "path": "review/run.sqlite" if review_path.is_file() else None,
        },
    )


def _execution_summary(payload: object) -> dict[str, Any]:
    summary: dict[str, Any] = {"completed": True, "status": "completed"}
    if isinstance(payload, dict):
        if isinstance(payload.get("scenario_name"), str):
            summary["scenario_name"] = payload["scenario_name"]
        run = payload.get("run")
        sources = (payload, run) if isinstance(run, dict) else (payload,)
        for key in ("duration_s", "dt_s", "samples"):
            value = next((source[key] for source in sources if key in source), None)
            if isinstance(value, (int, float)):
                summary[key] = value
    return summary


def _execute_single(
    service: ExecutionService,
    prepared: PreparedLifecycleRun,
) -> dict[str, Any]:
    payload = service.run_single(prepared.config)
    return service.wrap_single_file_payload(
        payload=payload,
        cfg=prepared.config,
        config_path=prepared.source_path,
    )


def run_foreground(
    prepared: PreparedLifecycleRun,
    *,
    store: LifecycleStore,
    on_handle: Callable[[RunHandle], None] | None = None,
    execution_service: ExecutionService | None = None,
    capture_execution_output: bool = False,
) -> ForegroundResult:
    store.prepare_output_dir(prepared.output_dir)
    accepted = store.create(_identity(prepared, store))
    handle = RunHandle.from_manifest(accepted)
    run_id = handle.identity.run_id
    service = execution_service or SimulationExecutionService()
    heartbeat = _OwnerHeartbeat(store, run_id)
    try:
        heartbeat.start()
        if on_handle is not None:
            on_handle(handle)
        store.transition(
            run_id,
            RunState.STARTING,
            phase=RunPhase.MATERIALIZING,
            event="starting",
            execution={"completed": False, "status": "starting", "resource_plan": prepared.resource_plan},
        )
        store.transition(
            run_id,
            RunState.RUNNING,
            phase=RunPhase.EXECUTING,
            event="running",
            execution={"completed": False, "status": "running", "resource_plan": prepared.resource_plan},
        )
        if capture_execution_output:
            log_path = prepared.output_dir / LIFECYCLE_DIR_NAME / "execution.log"
            with log_path.open("w", encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log):
                payload = _execute_single(service, prepared)
        else:
            payload = _execute_single(service, prepared)
        store.transition(
            run_id,
            RunState.FINALIZING,
            phase=RunPhase.COLLECTING_ARTIFACTS,
            event="finalizing",
            execution=_execution_summary(payload),
        )
        heartbeat.stop()
        artifacts, review = _artifact_inventory(prepared.output_dir)
        terminal = store.transition(
            run_id,
            RunState.COMPLETED,
            phase=RunPhase.COMMITTING_TERMINAL_STATE,
            event="completed",
            execution=_execution_summary(payload),
            artifacts=artifacts,
            review=review,
        )
    except Exception as exc:
        heartbeat.stop()
        if capture_execution_output:
            log_path = prepared.output_dir / LIFECYCLE_DIR_NAME / "execution.log"
            with log_path.open("a", encoding="utf-8") as log:
                traceback.print_exc(file=log)
        current = store.read_manifest(run_id)
        if current.is_terminal:
            terminal = current
        else:
            terminal = store.transition(
                run_id,
                RunState.FAILED,
                phase=RunPhase.COMMITTING_TERMINAL_STATE,
                event="failed",
                execution={"completed": False, "status": "failed"},
                error={
                    "code": "execution_failed",
                    "type": type(exc).__name__,
                    "message": "The foreground OEL scenario did not complete successfully.",
                },
            )
    return ForegroundResult(handle, terminal)
