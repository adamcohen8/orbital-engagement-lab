from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

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

    try:
        import anyio
    except ImportError as exc:  # pragma: no cover - covered by no-MCP installation checks
        raise RuntimeError(
            'The optional MCP SDK is not installed. Install the OEL MCP profile with `pip install ".[mcp]"`.'
        ) from exc
    return anyio.run(_run_public_workflow_acceptance_async, root, python_executable, work_root)


async def _run_public_workflow_acceptance_async(
    root: Path,
    python_executable: Path,
    work_root: Path,
) -> dict[str, Any]:
    try:
        from mcp import Client, StdioServerParameters, stdio_client
    except ImportError as exc:  # pragma: no cover - covered by no-MCP installation checks
        raise RuntimeError(
            'The optional MCP SDK is not installed. Install the OEL MCP profile with `pip install ".[mcp]"`.'
        ) from exc

    started = time.monotonic()
    work_root = work_root.expanduser().resolve()
    if work_root.exists() and any(work_root.iterdir()):
        raise FileExistsError("The MCP acceptance work root must be new or empty.")
    work_root.mkdir(parents=True, exist_ok=True)
    parameters = StdioServerParameters(
        command=str(python_executable),
        args=["-m", "integrations.oel_mcp"],
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
    )
    restricted_parameters = StdioServerParameters(
        command=str(python_executable),
        args=["-m", "integrations.oel_mcp", "--profile", "direct_frontier_restricted"],
        cwd=root,
        env={
            **os.environ,
            "OEL_MCP_ADAPTER": "sdk",
            "OEL_MCP_READ_ROOTS": str(root),
        },
    )
    async with (
        Client(stdio_client(parameters), mode="auto", cache=None) as client,
        Client(stdio_client(restricted_parameters), mode="auto", cache=None) as restricted_client,
    ):
        return await _run_public_workflow_acceptance_session(
            root=root,
            work_root=work_root,
            client=client,
            restricted_client=restricted_client,
            started=started,
        )


async def _run_public_workflow_acceptance_session(
    *,
    root: Path,
    work_root: Path,
    client: Any,
    restricted_client: Any,
    started: float,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    listed_restricted = await restricted_client.list_tools(cache_mode="reload")
    restricted_tools = tuple(str(item.name) for item in listed_restricted.tools)
    restricted_capability = await _call(restricted_client, "oel.describe_capabilities.v1", {})
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
    plan = await _call(client, "oel.plan_run.v1", scenario_args)
    _record(checks, "plan", plan, expected="completed")
    validation = await _call(
        client,
        "oel.validate_scenario.v1",
        {**scenario_args, "trust_plugins": True, "trust_approval": TRUST_APPROVAL},
    )
    _record(checks, "validate", validation, expected="completed")
    validation_id = str(dict(dict(validation.get("result", {}) or {}).get("identity", {}) or {}).get("validation_id", ""))
    executed = await _call(
        client,
        "oel.run_scenario.v1",
        {
            **scenario_args,
            "validation_id": validation_id,
            "trust_approval": TRUST_APPROVAL,
            "approval": EXECUTE_APPROVAL,
        },
    )
    _record(checks, "execute", executed, expected="completed")
    inspected = await _call(
        client,
        "oel.inspect_run.v1",
        {"output_dir": str(scenario_output), "handling": PUBLIC_HANDLING},
    )
    _record(checks, "inspect", inspected, expected="completed")
    queried = await _call(
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

    study_root = work_root / "study_lifecycle"
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "examples" / "python" / "study_lifecycle_three_domains.py"),
            "--output-root",
            str(study_root),
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        raise RuntimeError("The public study-lifecycle acceptance fixture failed to build.")
    study_bundle = study_root / "trajectory-targeting-canonical-v1"
    inspected_study = await _call(
        client,
        "oel.inspect_study.v1",
        {"bundle_dir": str(study_bundle), "handling": PUBLIC_HANDLING},
    )
    _record(checks, "inspect_study", inspected_study, expected="completed")
    replayed_study = await _call(
        client,
        "oel.replay_study.v1",
        {"bundle_dir": str(study_bundle), "handling": PUBLIC_HANDLING},
    )
    _record(checks, "replay_study", replayed_study, expected="completed")
    compared_studies = await _call(
        client,
        "oel.compare_studies.v1",
        {
            "left_bundle_dir": str(study_bundle),
            "right_bundle_dir": str(study_bundle),
            "handling": PUBLIC_HANDLING,
        },
    )
    _record(checks, "compare_studies", compared_studies, expected="completed")
    inspected_ccsds = await _call(
        client,
        "oel.inspect_ccsds.v1",
        {
            "path": str(root / "sim" / "interchange" / "examples" / "oel_earth_eme2000_utc_v3.oem"),
            "product_kind": "oem",
            "handling": PUBLIC_HANDLING,
        },
    )
    _record(checks, "inspect_ccsds", inspected_ccsds, expected="completed")
    converted_epoch = await _call(
        client,
        "oel.convert_frame_time.v1",
        {
            "operation": "convert_epoch",
            "epoch": "2024-01-01T00:00:00Z",
            "from_scale": "UTC",
            "to_scale": "TAI",
            "handling": PUBLIC_HANDLING,
        },
    )
    _record(checks, "convert_frame_time", converted_epoch, expected="completed")

    fsw_described = await _call(client, "oel.fsw.describe.v1", {"handling": PUBLIC_HANDLING})
    _record(checks, "fsw_describe", fsw_described, expected="completed")
    fsw_candidate_root = work_root / "fsw_candidate"
    fsw_scaffolded = await _call(
        client,
        "oel.fsw.scaffold_candidate.v1",
        {
            "name": "acceptance_adcs",
            "template": "adcs",
            "output_dir": str(fsw_candidate_root),
            "approval": WRITE_APPROVAL,
            "handling": PUBLIC_HANDLING,
        },
    )
    _record(checks, "fsw_scaffold", fsw_scaffolded, expected="completed")
    fsw_manifest = fsw_candidate_root / "candidate.yaml"
    fsw_inspected = await _call(
        client,
        "oel.fsw.inspect_candidate.v1",
        {"manifest_path": str(fsw_manifest), "handling": PUBLIC_HANDLING},
    )
    _record(checks, "fsw_inspect", fsw_inspected, expected="completed")
    fsw_planned = await _call(
        client,
        "oel.fsw.plan_candidate.v1",
        {
            "manifest_path": str(fsw_manifest),
            "operation": "validate",
            "output_dir": str(work_root / "fsw_validation"),
            "handling": PUBLIC_HANDLING,
        },
    )
    _record(checks, "fsw_plan", fsw_planned, expected="completed")
    fsw_validated = await _call(
        client,
        "oel.fsw.validate_candidate.v1",
        {
            "manifest_path": str(fsw_manifest),
            "output_dir": str(work_root / "fsw_validation"),
            "trusted_import": True,
            "trust_approval": TRUST_APPROVAL,
            "approval": WRITE_APPROVAL,
            "handling": PUBLIC_HANDLING,
        },
    )
    _record(checks, "fsw_validate", fsw_validated, expected="completed")
    fsw_validation_id = str(
        dict(dict(fsw_validated.get("result", {}) or {}).get("result", {}) or {}).get("validation_id", "")
    )
    fsw_tested = await _call(
        client,
        "oel.fsw.run_candidate_tests.v1",
        {
            "manifest_path": str(fsw_manifest),
            "output_dir": str(work_root / "fsw_tests"),
            "validation_id": fsw_validation_id,
            "trust_approval": TRUST_APPROVAL,
            "approval": EXECUTE_APPROVAL,
            "handling": PUBLIC_HANDLING,
        },
    )
    _record(checks, "fsw_tests", fsw_tested, expected="completed")
    fsw_smoked = await _call(
        client,
        "oel.fsw.run_candidate_smoke.v1",
        {
            "manifest_path": str(fsw_manifest),
            "output_dir": str(work_root / "fsw_smoke"),
            "validation_id": fsw_validation_id,
            "trust_approval": TRUST_APPROVAL,
            "approval": EXECUTE_APPROVAL,
            "handling": PUBLIC_HANDLING,
        },
    )
    _record(checks, "fsw_smoke", fsw_smoked, expected="completed")
    fsw_verified = await _call(
        client,
        "oel.fsw.verify_receipt.v1",
        {
            "receipt_path": str(work_root / "fsw_validation" / "fsw_validation_receipt.json"),
            "handling": PUBLIC_HANDLING,
        },
    )
    _record(checks, "fsw_verify_receipt", fsw_verified, expected="completed")

    handoff_path = work_root / "handoff" / "completed_state.json"
    exported = await _call(
        client,
        "oel.export_run_product.v1",
        {
            "completed_run": str(scenario_output),
            "product_kind": "completed_run_state",
            "object_id": "target",
            "selector": "final",
            "epoch_jd_utc": 2461254.5,
            "output_path": str(handoff_path),
            "approval": WRITE_APPROVAL,
            "handling": PUBLIC_HANDLING,
        },
    )
    _record(checks, "export_run_product", exported, expected="completed")
    handoff = await _call(
        client,
        "oel.inspect_handoff.v1",
        {"path": str(handoff_path), "handling": PUBLIC_HANDLING},
    )
    _record(checks, "inspect_handoff", handoff, expected="completed")

    materialized_state = await _call(
        client,
        "oel.materialize_onp_handoff.v1",
        {
            "product_path": str(handoff_path),
            "scenario_name": "mcp_acceptance_continuation",
            "scenario_path": str(work_root / "handoff" / "continued.yaml"),
            "run_output_dir": str(work_root / "handoff" / "continued_run"),
            "duration_s": 30.0,
            "dt_s": 10.0,
            "trust_plugins": True,
            "trust_approval": TRUST_APPROVAL,
            "approval": WRITE_APPROVAL,
            "handling": PUBLIC_HANDLING,
        },
    )
    _record(checks, "materialize_onp_handoff", materialized_state, expected="completed")
    materialized_result = dict(materialized_state.get("result", {}) or {})

    overlay_path = work_root / "handoff" / "overlay.yaml"
    overlay_path.write_text(
        "simulator:\n  termination:\n    earth_impact_enabled: false\n"
        "outputs:\n  review:\n    enabled: true\n    detail: standard\n",
        encoding="utf-8",
    )
    overlay_product = work_root / "handoff" / "overlay_product.json"
    emitted_overlay = await _call(
        client,
        "oel.emit_scenario_overlay.v1",
        {
            "source_scenario": str(root / "configs" / "automation_smoke.yaml"),
            "overlay_path": str(overlay_path),
            "overlay_id": "mcp_acceptance_overlay",
            "rationale": "Exercise typed overlay emission through real stdio.",
            "output_path": str(overlay_product),
            "approval": WRITE_APPROVAL,
            "handling": PUBLIC_HANDLING,
        },
    )
    _record(checks, "emit_scenario_overlay", emitted_overlay, expected="completed")
    materialized_patch = await _call(
        client,
        "oel.materialize_scenario_patch.v1",
        {
            "patch_product": str(overlay_product),
            "source_scenario": str(root / "configs" / "automation_smoke.yaml"),
            "scenario_name": "mcp_acceptance_overlay_materialized",
            "scenario_path": str(work_root / "handoff" / "overlay_materialized.yaml"),
            "run_output_dir": str(work_root / "handoff" / "overlay_run"),
            "trust_plugins": True,
            "trust_approval": TRUST_APPROVAL,
            "approval": WRITE_APPROVAL,
            "handling": PUBLIC_HANDLING,
        },
    )
    _record(checks, "materialize_scenario_patch", materialized_patch, expected="completed")
    compared_handoff = await _call(
        client,
        "oel.compare_handoff.v1",
        {
            "product_path": str(handoff_path),
            "scenario_path": str(materialized_result.get("scenario_path", "")),
            "manifest_path": str(materialized_result.get("manifest_path", "")),
            "output_path": str(work_root / "handoff" / "comparison.json"),
            "approval": WRITE_APPROVAL,
            "handling": PUBLIC_HANDLING,
        },
    )
    _record(checks, "compare_handoff", compared_handoff, expected="completed")

    task_output = work_root / "task"
    task = await _call(
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
    readiness = await _call(
        client,
        "oel.assess_maneuver_readiness.v1",
        {
            "completed_run": str(task_output),
            "object_id": "chaser",
            "chief_id": "target",
            "thresholds": {"max_final_range_km": 1000.0},
            "output_path": str(work_root / "maneuver_readiness.json"),
            "approval": WRITE_APPROVAL,
            "handling": PUBLIC_HANDLING,
        },
    )
    _record(checks, "assess_maneuver_readiness", readiness, expected="completed")
    compared = await _call(
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
    plotted = await _call(
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
    typed_plot_arguments = {
        "output_dir": str(task_output),
        "sql": "SELECT time_s, range_km FROM relative_state ORDER BY time_s",
        "x_column": "time_s",
        "y_columns": ["range_km"],
        "plot_type": "line",
        "style": "oel_light",
        "format": "png",
        "artifact_id": "acceptance_typed_relative_range",
        "handling": PUBLIC_HANDLING,
    }
    planned_plot = await _call(client, "oel.plan_review_plot.v1", typed_plot_arguments)
    _record(checks, "plan_review_plot", planned_plot, expected="completed")
    rendered_plot = await _call(
        client,
        "oel.render_review_plot.v2",
        {
            **typed_plot_arguments,
            "plot_plan_id": str(dict(planned_plot.get("result", {}) or {}).get("plot_plan_id", "")),
            "approval": WRITE_APPROVAL,
        },
    )
    _record(checks, "render_review_plot", rendered_plot, expected="completed")
    animation_arguments = {
        "output_dir": str(task_output),
        "recipe_id": "relative_position_ric_2d",
        "style": "oel_light",
        "format": "gif",
        "fps": 10,
        "frame_stride": 2,
        "camera_policy": "fit_history",
        "artifact_id": "acceptance_relative_position_ric_2d",
        "handling": PUBLIC_HANDLING,
    }
    planned_animation = await _call(client, "oel.plan_review_animation.v1", animation_arguments)
    _record(checks, "plan_review_animation", planned_animation, expected="completed")
    rendered_animation = await _call(
        client,
        "oel.render_review_animation.v1",
        {
            **animation_arguments,
            "animation_plan_id": str(
                dict(planned_animation.get("result", {}) or {}).get("animation_plan_id", "")
            ),
            "approval": WRITE_APPROVAL,
        },
    )
    _record(checks, "render_review_animation", rendered_animation, expected="completed")

    packet_output = work_root / "report_packet"
    packet = await _call(
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
    audit = await _call(
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


async def _call(client: Any, tool_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    response = await client.call_tool(tool_id, arguments)
    payload = dict(response.structured_content or {})
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
