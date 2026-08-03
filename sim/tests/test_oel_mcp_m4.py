from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from threading import Event
from unittest.mock import patch

import pytest
import yaml

from integrations.oel_mcp.execution import ExecutionApprovalPolicy
from integrations.oel_mcp.policy import MCPPathPolicy
from integrations.oel_mcp.public_handlers import PublicOELMCPHandlers
from integrations.oel_mcp.public_registry import PUBLIC_TOOL_CONTRACTS
from integrations.oel_mcp.sdk_protocol import build_sdk_server

ROOT = Path(__file__).resolve().parents[2]
HANDLING = {"marking": "PUBLIC", "release_scope": "public"}
EXECUTION_APPROVAL = {"approval_id": "test-execution", "scope": "execute"}
WRITE_APPROVAL = {"approval_id": "test-write", "scope": "write"}
TRUST_APPROVAL = {"approval_id": "test-trust", "scope": "trust"}
M4_IDS = (
    "oel.plan_run.v1",
    "oel.validate_scenario.v1",
    "oel.run_scenario.v1",
    "oel.compare_runs.v1",
    "oel.plot_evidence.v1",
    "oel.run_agent_task.v1",
)
MCP_SDK_AVAILABLE = importlib.util.find_spec("mcp") is not None
PRO_HANDLERS_AVAILABLE = importlib.util.find_spec("integrations.oel_mcp.pro_handlers") is not None
if PRO_HANDLERS_AVAILABLE:
    from integrations.oel_mcp.pro_handlers import ProOELMCPHandlers


def _handlers(tmp_path: Path) -> PublicOELMCPHandlers:
    return PublicOELMCPHandlers(
        read_roots=(ROOT, tmp_path),
        write_roots=(tmp_path,),
        approval_policy=ExecutionApprovalPolicy(
            write_approval_ids=frozenset({"test-write"}),
            execution_approval_ids=frozenset({"test-execution"}),
            trust_approval_ids=frozenset({"test-trust"}),
        ),
    )


def _scenario_arguments(tmp_path: Path, name: str) -> dict[str, object]:
    return {
        "config_path": str(ROOT / "configs" / "automation_smoke.yaml"),
        "output_dir": str(tmp_path / name),
        "resource_profile": "laptop-safe",
        "handling": HANDLING,
    }


def test_m4_registry_effects_and_deployment_views(tmp_path: Path) -> None:
    contracts = {contract.tool_id: contract for contract in PUBLIC_TOOL_CONTRACTS}
    public = _handlers(tmp_path)
    frontier = PublicOELMCPHandlers(
        profile="direct_frontier_restricted",
        read_roots=(ROOT, tmp_path),
        write_roots=(tmp_path,),
    )

    assert tuple(tool_id for tool_id in public.contracts if tool_id in M4_IDS) == M4_IDS
    assert not set(M4_IDS) & set(frontier.contracts)
    if PRO_HANDLERS_AVAILABLE:
        pro = ProOELMCPHandlers(
            profile="pro_local",
            read_roots=(ROOT, tmp_path),
            write_roots=(tmp_path,),
        )
        assert set(M4_IDS) <= set(pro.contracts)
    assert contracts["oel.run_scenario.v1"].capability()["effects"] == {
        "reads": True,
        "writes": True,
        "executes": True,
        "external_communication": False,
    }
    assert contracts["oel.plot_evidence.v1"].mcp_definition()["annotations"] == {
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": False,
        "openWorldHint": False,
    }


def test_plan_validate_and_run_require_external_approval_and_bound_identity(tmp_path: Path) -> None:
    handlers = _handlers(tmp_path)
    arguments = _scenario_arguments(tmp_path, "run")

    plan = handlers.plan_run(**arguments)
    safe_only = handlers.validate_scenario(**arguments, trust_plugins=False)
    with pytest.raises(PermissionError, match="trust approval"):
        handlers.validate_scenario(**arguments, trust_plugins=True)
    trusted = handlers.validate_scenario(**arguments, trust_plugins=True, trust_approval=TRUST_APPROVAL)

    assert plan["status"] == "completed"
    assert plan["result"]["execution_authorized"] is False
    assert safe_only["status"] == "partial"
    assert safe_only["result"]["identity"]["validation_id"] == ""
    assert "trust approval" in safe_only["result"]["next_step"]
    assert "resource envelope" not in safe_only["result"]["next_step"]
    assert trusted["status"] == "completed"
    validation_id = trusted["result"]["identity"]["validation_id"]
    assert validation_id.startswith("oel-m4-validation-v1:")
    assert trusted["result"]["execution_authorized"] is False


    disabled = PublicOELMCPHandlers(read_roots=(ROOT, tmp_path), write_roots=(tmp_path,))
    try:
        disabled.run_scenario(
            **arguments,
            validation_id=validation_id,
            approval=EXECUTION_APPROVAL,
        )
    except PermissionError as exc:
        assert "operator approval policy" in str(exc)
    else:  # pragma: no cover - explicit safety assertion
        raise AssertionError("Execution unexpectedly bypassed the server approval policy.")

    with pytest.raises(PermissionError, match="trust approval"):
        handlers.run_scenario(
            **arguments,
            validation_id=validation_id,
            approval=EXECUTION_APPROVAL,
        )
    assert not Path(str(arguments["output_dir"])).exists()

    mismatch = handlers.run_scenario(
        **arguments,
        validation_id="oel-m4-validation-v1:wrong",
        trust_approval=TRUST_APPROVAL,
        approval=EXECUTION_APPROVAL,
    )
    assert mismatch["status"] == "failed"
    assert not Path(str(arguments["output_dir"])).exists()

    completed = handlers.run_scenario(
        **arguments,
        validation_id=validation_id,
        trust_approval=TRUST_APPROVAL,
        approval=EXECUTION_APPROVAL,
    )
    manifest = json.loads((Path(str(arguments["output_dir"])) / "mcp_execution_manifest.json").read_text())
    assert completed["status"] == "completed"
    assert completed["effects"]["executes"] is True
    assert manifest["status"] == "completed"
    assert manifest["validation_id"] == validation_id
    assert manifest["artifacts_complete"] is True
    assert (Path(str(arguments["output_dir"])) / "review" / "run.sqlite").is_file()

    inspected = handlers.inspect_run(output_dir=arguments["output_dir"], handling=HANDLING)
    provenance = inspected["result"]["execution_provenance"]
    assert provenance["available"] is True
    assert provenance["status"] == "completed"
    assert provenance["validation_id"] == validation_id
    assert provenance["artifact_count"] == len(manifest["artifacts"])
    assert provenance["artifacts_complete"] is True
    assert inspected["result"]["evidence_summary"]["mcp_execution_complete"] is True
    assert inspected["result"]["freshness"]["content_bound_execution_recorded"] is True


def test_m5_2_completed_run_product_export_and_inspection(tmp_path: Path) -> None:
    handlers = _handlers(tmp_path)
    arguments = _scenario_arguments(tmp_path, "glue-source")
    validated = handlers.validate_scenario(
        **arguments, trust_plugins=True, trust_approval=TRUST_APPROVAL
    )
    run = handlers.run_scenario(
        **arguments,
        validation_id=validated["result"]["identity"]["validation_id"],
        trust_approval=TRUST_APPROVAL,
        approval=EXECUTION_APPROVAL,
    )
    assert run["status"] == "completed"

    product_path = tmp_path / "products" / "final-state.json"
    exported = handlers.export_run_product(
        completed_run=tmp_path / "glue-source",
        product_kind="completed_run_state",
        object_id="target",
        selector="final",
        epoch_jd_utc=2461254.5,
        output_path=product_path,
        handling=HANDLING,
        approval=WRITE_APPROVAL,
    )
    assert exported["status"] == "completed"
    assert exported["result"]["execution_occurred"] is False
    assert Path(exported["result"]["operation_manifest_path"]).is_file()

    inspected = handlers.inspect_handoff(path=product_path, handling=HANDLING)
    assert inspected["status"] == "completed"
    assert inspected["result"]["product_kind"] == "oel.completed_run_state"
    assert "materialize_onp_handoff" in inspected["result"]["supported_next_actions"]


def test_unset_write_roots_do_not_inherit_read_authority(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OEL_MCP_READ_ROOTS", str(tmp_path))
    monkeypatch.delenv("OEL_MCP_WRITE_ROOTS", raising=False)
    monkeypatch.delenv("OEL_MCP_ALLOWED_ROOTS", raising=False)

    policy = MCPPathPolicy.configured()

    assert policy.read_roots == (tmp_path.resolve(),)
    assert policy.write_roots == ()
    with pytest.raises(PermissionError, match="not authorized"):
        policy.resolve_write(tmp_path / "output")

    monkeypatch.setenv("OEL_MCP_ALLOWED_ROOTS", str(tmp_path))
    legacy_policy = MCPPathPolicy.configured()
    assert legacy_policy.read_roots == (tmp_path.resolve(),)
    assert legacy_policy.write_roots == (tmp_path.resolve(),)


def test_cancelled_run_is_partial_and_writes_unambiguous_manifest(tmp_path: Path) -> None:
    handlers = _handlers(tmp_path)
    arguments = _scenario_arguments(tmp_path, "cancelled")
    validation = handlers.validate_scenario(**arguments, trust_plugins=True, trust_approval=TRUST_APPROVAL)
    cancel_event = Event()
    cancel_event.set()

    result = handlers.call(
        "oel.run_scenario.v1",
        {
            **arguments,
            "validation_id": validation["result"]["identity"]["validation_id"],
            "trust_approval": TRUST_APPROVAL,
            "approval": EXECUTION_APPROVAL,
        },
        cancel_event=cancel_event,
    )
    manifest = json.loads((Path(str(arguments["output_dir"])) / "mcp_execution_manifest.json").read_text())

    assert result["status"] == "partial"
    assert result["result"]["status"] == "cancelled"
    assert result["evidence"]["complete"] is False
    assert manifest["status"] == "cancelled"
    assert manifest["cancelled"] is True
    assert manifest["artifacts_complete"] is False


def test_supported_task_comparison_and_plot_flow(tmp_path: Path) -> None:
    handlers = _handlers(tmp_path)
    output = tmp_path / "task"
    task = handlers.run_agent_task(
        recipe_id="quickstart_review",
        output_dir=output,
        resource_profile="laptop-safe",
        make_plots=False,
        max_rows=25,
        approval=EXECUTION_APPROVAL,
        handling=HANDLING,
    )
    comparison = handlers.compare_runs(
        base_output_dir=output,
        candidate_output_dir=output,
        metric_names=["final_range_km", "closest_approach_km"],
        max_rows=25,
        handling=HANDLING,
    )
    plot = handlers.plot_evidence(
        output_dir=output,
        recipe_id="relative_range",
        artifact_id="m4_relative_range",
        style="oel_light",
        format="png",
        approval=WRITE_APPROVAL,
        handling=HANDLING,
    )

    assert task["status"] == "completed"
    assert task["result"]["evidence_summary"]["ready_to_cite"] is True
    assert comparison["status"] == "completed"
    assert comparison["result"]["deltas"] == {"closest_approach_km": 0.0, "final_range_km": 0.0}
    assert plot["status"] == "completed"
    assert Path(plot["result"]["artifact"]["path"]).is_file()
    assert Path(plot["result"]["manifest_path"]).is_file()


def test_partial_plot_writes_incomplete_manifest(tmp_path: Path) -> None:
    output = tmp_path / "run"
    review = output / "review"
    review.mkdir(parents=True)
    (review / "run.sqlite").write_bytes(b"")
    handlers = _handlers(tmp_path)
    failed_artifact = {
        "status": "error",
        "path_exists": False,
        "truncated": False,
    }

    with patch("integrations.oel_mcp.public_handlers.create_plot", return_value=failed_artifact):
        plot = handlers.plot_evidence(
            output_dir=output,
            recipe_id="relative_range",
            artifact_id="failed_plot",
            approval=WRITE_APPROVAL,
            handling=HANDLING,
        )

    manifest = json.loads((review / "mcp_plots" / "failed_plot.manifest.json").read_text(encoding="utf-8"))
    assert plot["status"] == "partial"
    assert manifest["status"] == "partial"
    assert manifest["artifacts"] == []
    assert manifest["artifacts_complete"] is False


def test_unapproved_or_private_task_recipe_is_rejected(tmp_path: Path) -> None:
    handlers = _handlers(tmp_path)
    result = handlers.run_agent_task(
        recipe_id="dynamics_od_smoke",
        output_dir=tmp_path / "private-task",
        resource_profile="laptop-safe",
        approval=EXECUTION_APPROVAL,
        handling=HANDLING,
    )

    assert result["status"] == "failed"
    assert result["result"] is None
    assert "Only supported public scenario-run" in result["error"]["message"]


def test_failed_agent_task_packet_remains_failed_with_diagnostics(tmp_path: Path) -> None:
    handlers = _handlers(tmp_path)
    output = tmp_path / "failed-task"
    failed_packet = {
        "status": "failed",
        "evidence_summary": {"ready_to_cite": False},
        "packet_path": "",
        "artifacts": [],
        "failure_hints": [{"code": "simulation_failed", "next_step": "Inspect the deterministic error."}],
    }

    with patch("integrations.oel_mcp.public_handlers.run_recipe", return_value=failed_packet):
        result = handlers.run_agent_task(
            recipe_id="quickstart_review",
            output_dir=output,
            resource_profile="laptop-safe",
            approval=EXECUTION_APPROVAL,
            handling=HANDLING,
        )

    assert result["status"] == "failed"
    assert result["error"] is None
    assert result["result"]["status"] == "failed"
    assert result["result"]["failure_hints"] == failed_packet["failure_hints"]
    manifest = json.loads((output / "mcp_execution_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["artifacts_complete"] is False


@pytest.mark.skipif(not MCP_SDK_AVAILABLE, reason="optional MCP SDK profile is not installed")
def test_sdk_cancellation_reaches_execution_callback_and_finalizes_manifest(tmp_path: Path) -> None:
    import anyio
    from mcp import Client

    source = yaml.safe_load((ROOT / "configs" / "automation_smoke.yaml").read_text())
    source["simulator"]["duration_s"] = 100_000.0
    source["simulator"]["dt_s"] = 1.0
    config_path = tmp_path / "long.yaml"
    config_path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    handlers = _handlers(tmp_path)
    output = tmp_path / "sdk-cancelled"
    arguments = {
        "config_path": str(config_path),
        "output_dir": str(output),
        "resource_profile": "laptop-safe",
        "handling": HANDLING,
    }
    validation = handlers.validate_scenario(**arguments, trust_plugins=True, trust_approval=TRUST_APPROVAL)
    call_arguments = {
        **arguments,
        "validation_id": validation["result"]["identity"]["validation_id"],
        "trust_approval": TRUST_APPROVAL,
        "approval": EXECUTION_APPROVAL,
    }

    async def exercise() -> None:
        async with Client(build_sdk_server(handlers), mode="auto", cache=None) as client:
            scope = anyio.CancelScope()
            done = anyio.Event()

            async def invoke() -> None:
                try:
                    with scope:
                        await client.call_tool("oel.run_scenario.v1", call_arguments)
                finally:
                    done.set()

            async with anyio.create_task_group() as task_group:
                task_group.start_soon(invoke)
                with anyio.fail_after(5):
                    while not (output / "mcp_execution_manifest.json").is_file():
                        await anyio.sleep(0.01)
                scope.cancel()
                await done.wait()
                with anyio.fail_after(5):
                    while True:
                        manifest = json.loads((output / "mcp_execution_manifest.json").read_text())
                        if manifest["status"] != "running":
                            break
                        await anyio.sleep(0.01)

    anyio.run(exercise)
    manifest = json.loads((output / "mcp_execution_manifest.json").read_text())
    assert manifest["status"] == "cancelled"
    assert manifest["cancelled"] is True
    assert manifest["artifacts_complete"] is False
