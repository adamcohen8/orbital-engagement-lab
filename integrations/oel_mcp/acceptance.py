from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from integrations.oel_mcp.conformance import SDKStdioConformanceClient
from integrations.oel_mcp.public_registry import M3_PUBLIC_TOOL_IDS

PUBLIC_HANDLING = {"marking": "PUBLIC_ACCEPTANCE", "release_scope": "public"}
TRUST_APPROVAL = {"approval_id": "accept-trust", "scope": "trust"}
WRITE_APPROVAL = {"approval_id": "accept-write", "scope": "write"}
EXECUTE_APPROVAL = {"approval_id": "accept-execute", "scope": "execute"}


def run_public_workflow_acceptance(
    *,
    root: Path,
    python_executable: Path,
    work_root: Path,
) -> dict[str, Any]:
    """Exercise supported public workflows through a real SDK stdio subprocess."""

    started = time.monotonic()
    work_root = work_root.expanduser().resolve()
    if work_root.exists() and any(work_root.iterdir()):
        raise FileExistsError("The MCP acceptance work root must be new or empty.")
    work_root.mkdir(parents=True, exist_ok=True)
    client = SDKStdioConformanceClient(
        command=str(python_executable),
        args=("-m", "integrations.oel_mcp"),
        cwd=root,
        env={
            **os.environ,
            "OEL_MCP_ADAPTER": "sdk",
            "OEL_MCP_READ_ROOTS": os.pathsep.join((str(root), str(work_root))),
            "OEL_MCP_WRITE_ROOTS": str(work_root),
            "OEL_MCP_TRUST_APPROVAL_IDS": TRUST_APPROVAL["approval_id"],
            "OEL_MCP_WRITE_APPROVAL_IDS": WRITE_APPROVAL["approval_id"],
            "OEL_MCP_EXECUTION_APPROVAL_IDS": EXECUTE_APPROVAL["approval_id"],
        },
        mode="auto",
    )
    checks: list[dict[str, Any]] = []
    restricted_client = SDKStdioConformanceClient(
        command=str(python_executable),
        args=("-m", "integrations.oel_mcp", "--profile", "direct_frontier_restricted"),
        cwd=root,
        env={
            **os.environ,
            "OEL_MCP_ADAPTER": "sdk",
            "OEL_MCP_READ_ROOTS": str(root),
        },
        mode="auto",
    )
    restricted_tools = tuple(str(item.get("name", "")) for item in restricted_client.list_tools())
    restricted_capability = _call(restricted_client, "oel.describe_capabilities.v1", {})
    restricted_profile = str(dict(restricted_capability.get("result", {}) or {}).get("deployment_profile", ""))
    restricted_passed = restricted_tools == M3_PUBLIC_TOOL_IDS and restricted_profile == "direct_frontier_restricted"
    checks.append(
        {
            "check_id": "restricted_profile",
            "passed": restricted_passed,
            "tool_id": "oel.describe_capabilities.v1",
            "status": restricted_capability.get("status"),
            "evidence": dict(restricted_capability.get("evidence", {}) or {}),
        }
    )
    if not restricted_passed:
        raise RuntimeError(
            "The direct-frontier profile exposed the wrong public registry: "
            f"tools={restricted_tools!r}, profile={restricted_profile!r}"
        )
    scenario_output = work_root / "scenario"
    scenario_args = {
        "config_path": str(root / "configs" / "automation_smoke.yaml"),
        "output_dir": str(scenario_output),
        "resource_profile": "laptop-safe",
        "handling": PUBLIC_HANDLING,
    }
    plan = _call(client, "oel.plan_run.v1", scenario_args)
    _record(checks, "plan", plan, expected="completed")
    validation = _call(
        client,
        "oel.validate_scenario.v1",
        {**scenario_args, "trust_plugins": True, "trust_approval": TRUST_APPROVAL},
    )
    _record(checks, "validate", validation, expected="completed")
    validation_id = str(dict(dict(validation.get("result", {}) or {}).get("identity", {}) or {}).get("validation_id", ""))
    executed = _call(
        client,
        "oel.run_scenario.v1",
        {**scenario_args, "validation_id": validation_id, "approval": EXECUTE_APPROVAL},
    )
    _record(checks, "execute", executed, expected="completed")
    inspected = _call(
        client,
        "oel.inspect_run.v1",
        {"output_dir": str(scenario_output), "handling": PUBLIC_HANDLING},
    )
    _record(checks, "inspect", inspected, expected="completed")
    queried = _call(
        client,
        "oel.query_review.v1",
        {
            "output_dir": str(scenario_output),
            "sql": "SELECT scenario_name, duration_s, samples FROM run_metadata",
            "max_rows": 10,
            "handling": PUBLIC_HANDLING,
        },
    )
    _record(checks, "query", queried, expected="completed")

    task_output = work_root / "task"
    task = _call(
        client,
        "oel.run_agent_task.v1",
        {
            "recipe_id": "quickstart_review",
            "output_dir": str(task_output),
            "resource_profile": "laptop-safe",
            "make_plots": False,
            "max_rows": 25,
            "approval": EXECUTE_APPROVAL,
            "handling": PUBLIC_HANDLING,
        },
    )
    _record(checks, "agent_task", task, expected="completed")
    compared = _call(
        client,
        "oel.compare_runs.v1",
        {
            "base_output_dir": str(task_output),
            "candidate_output_dir": str(task_output),
            "metric_names": ["final_range_km", "closest_approach_km"],
            "max_rows": 25,
            "handling": PUBLIC_HANDLING,
        },
    )
    _record(checks, "compare", compared, expected="completed")
    plotted = _call(
        client,
        "oel.plot_evidence.v1",
        {
            "output_dir": str(task_output),
            "recipe_id": "relative_range",
            "style": "oel_light",
            "format": "png",
            "artifact_id": "acceptance_relative_range",
            "approval": WRITE_APPROVAL,
            "handling": PUBLIC_HANDLING,
        },
    )
    _record(checks, "plot", plotted, expected="completed")

    packet_output = work_root / "report_packet"
    packet = _call(
        client,
        "oel.prepare_report_packet.v1",
        {
            "source_output_dir": str(task_output),
            "packet_output_dir": str(packet_output),
            "packet_id": "acceptance_report",
            "max_rows": 25,
            "approval": WRITE_APPROVAL,
            "handling": PUBLIC_HANDLING,
        },
    )
    _record(checks, "prepare_report_packet", packet, expected="completed")
    packet_path = Path(str(dict(packet.get("result", {}) or {}).get("packet_path", "")))
    packet_payload = json.loads(packet_path.read_text(encoding="utf-8"))
    artifact_ids = [str(item["artifact_id"]) for item in list(packet_payload.get("artifacts", []) or [])]
    citation = f"[evidence:{artifact_ids[0]}]" if artifact_ids else ""
    draft = work_root / "agent_report.md"
    draft.write_text(
        "\n".join(
            (
                "# Acceptance Report",
                "",
                "## Evidence",
                "",
                f"The completed OEL evidence packet was inspected. {citation}".rstrip(),
                "",
                "## Limitations",
                "",
                "This acceptance report makes no operational or flight-qualification claim.",
                "",
            )
        ),
        encoding="utf-8",
    )
    audit = _call(
        client,
        "oel.audit_report.v1",
        {
            "report_path": str(draft),
            "packet_path": str(packet_path),
            "audit_output_dir": str(work_root / "report_audit"),
            "author": "oel_acceptance_fixture",
            "model": "none",
            "approval": WRITE_APPROVAL,
            "handling": PUBLIC_HANDLING,
        },
    )
    _record(checks, "audit_report", audit, expected="completed")
    audit_result = dict(audit.get("result", {}) or {})
    if audit_result.get("status") != "passed":
        raise RuntimeError(f"Report audit acceptance did not pass: {audit_result!r}")

    report = {
        "schema_version": 1,
        "status": "passed" if all(row["passed"] for row in checks) else "failed",
        "transport": "stdio",
        "provider_calls": 0,
        "checks": checks,
        "duration_ms": round((time.monotonic() - started) * 1000),
    }
    (work_root / "acceptance_result.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _call(client: SDKStdioConformanceClient, tool_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    response = client.call_tool(tool_id, arguments)
    payload = dict(response.get("structuredContent", {}) or {})
    if not payload:
        raise RuntimeError(f"The MCP acceptance call returned no structured result: {tool_id}")
    return payload


def _record(rows: list[dict[str, Any]], check_id: str, payload: dict[str, Any], *, expected: str) -> None:
    passed = payload.get("status") == expected and payload.get("error") is None
    rows.append(
        {
            "check_id": check_id,
            "passed": passed,
            "tool_id": payload.get("tool_id"),
            "status": payload.get("status"),
            "evidence": dict(payload.get("evidence", {}) or {}),
        }
    )
    if not passed:
        raise RuntimeError(f"MCP workflow acceptance failed at {check_id}: {payload!r}")


__all__ = ["run_public_workflow_acceptance"]
