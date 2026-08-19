"""Transport-independent handlers for public FSW authoring MCP tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from integrations.oel_mcp.base_handlers import require_file_size
from integrations.oel_mcp.contracts import MAX_MANIFEST_BYTES, ToolContract
from integrations.oel_mcp.execution import ensure_new_output_dir
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

from .fsw_authoring_registry import FSW_AUTHORING_TOOL_IDS


def call_fsw_authoring_tool(
    handler: Any,
    contract: ToolContract,
    arguments: dict[str, Any],
) -> dict[str, Any] | None:
    if contract.tool_id not in FSW_AUTHORING_TOOL_IDS:
        return None

    def operation() -> dict[str, Any]:
        result = _run(handler, contract.tool_id, arguments)
        return {
            **_project_result(result),
            "hidden_truth_visible": False,
            "private_orchestration_used": False,
            "external_communication": False,
        }

    return handler._envelope(
        contract=contract,
        arguments=arguments,
        operation=operation,
        outcome_status=lambda result: "completed" if str(result.get("status")) in {"ready", "passed"} else "partial",
        evidence=lambda result: {
            "complete": str(result.get("status")) in {"ready", "passed", "failed"},
            "empty": False,
            "truncated": False,
        },
    )


def _run(handler: Any, tool_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if tool_id == FSW_AUTHORING_TOOL_IDS[0]:
        return describe_capabilities()
    if tool_id == FSW_AUTHORING_TOOL_IDS[3]:
        output = handler.path_policy.resolve_write(arguments["output_dir"])
        workspace = _authorized_workspace(handler, output, write=True)
        return init_candidate(
            str(arguments["name"]),
            template=str(arguments["template"]),
            workspace_root=workspace,
            output_dir=output,
            class_name=arguments.get("class_name"),
        )
    if tool_id == FSW_AUTHORING_TOOL_IDS[7]:
        receipt = handler.path_policy.resolve_read(arguments["receipt_path"], kind="file")
        workspace = _authorized_workspace(handler, receipt)
        return verify_receipt(receipt, workspace_root=workspace)
    manifest = handler.path_policy.resolve_read(arguments["manifest_path"], kind="file")
    require_file_size(manifest, maximum=MAX_MANIFEST_BYTES)
    workspace = _authorized_workspace(handler, manifest)
    if tool_id == FSW_AUTHORING_TOOL_IDS[1]:
        return inspect_candidate(manifest, workspace_root=workspace)
    if tool_id == FSW_AUTHORING_TOOL_IDS[2]:
        output = arguments.get("output_dir")
        if output:
            output = handler.path_policy.resolve_write(output)
            _require_same_workspace(output, workspace)
        return plan_workflow(
            manifest,
            str(arguments["operation"]),
            workspace_root=workspace,
            output_dir=output,
        )
    output = handler.path_policy.resolve_write(arguments["output_dir"])
    _require_same_workspace(output, workspace)
    ensure_new_output_dir(output)
    if tool_id == FSW_AUTHORING_TOOL_IDS[4]:
        return validate_candidate_service(
            manifest,
            workspace_root=workspace,
            trusted_import=True,
            receipt_dir=output,
        )
    common = {
        "workspace_root": workspace,
        "output_dir": output,
        "validation_id": str(arguments["validation_id"]),
    }
    if tool_id == FSW_AUTHORING_TOOL_IDS[5]:
        return run_contract_tests(manifest, **common)
    if tool_id == FSW_AUTHORING_TOOL_IDS[6]:
        return run_smoke(manifest, **common)
    raise PermissionError("Unsupported public FSW authoring tool contract.")


def _authorized_workspace(handler: Any, path: Path, *, write: bool = False) -> Path:
    roots = handler.path_policy.write_roots if write else handler.path_policy.read_roots
    candidates = [root for root in roots if _is_relative_to(path, root)]
    if not candidates:
        raise PermissionError("Path is not inside an authorized public FSW workspace.")
    return max(candidates, key=lambda item: len(item.parts))


def _require_same_workspace(path: Path, workspace: Path) -> None:
    if not _is_relative_to(path, workspace):
        raise PermissionError("Public FSW inputs and outputs must remain inside one authorized workspace.")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _project_result(result: dict[str, Any]) -> dict[str, Any]:
    projected = dict(result)
    summary = projected.get("summary")
    if projected.get("schema") == "oel.fsw_authoring.run_manifest.v1" and isinstance(summary, dict):
        projected["summary"] = {
            key: summary.get(key)
            for key in (
                "scenario_name",
                "duration_s",
                "dt_s",
                "samples",
                "objects",
                "terminated_early",
                "termination_reason",
                "review_sqlite_path",
            )
            if key in summary
        }
    execution = projected.get("execution")
    if isinstance(execution, dict):
        bounded = dict(execution)
        for key in ("stdout", "stderr"):
            if isinstance(bounded.get(key), str) and len(bounded[key]) > 10_000:
                bounded[key] = bounded[key][-10_000:]
                bounded[f"{key}_truncated"] = True
        projected["execution"] = bounded
    return projected


__all__ = ["call_fsw_authoring_tool"]
