from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
import yaml

from integrations.oel_mcp.agent_guidance import BOOTSTRAP_URI, WORKFLOWS_URI
from integrations.oel_mcp.execution import ExecutionApprovalPolicy
from integrations.oel_mcp.public_handlers import PublicOELMCPHandlers
from integrations.oel_mcp.resources import build_public_resource_catalog
from sim.installation.contracts import sha256_file
from sim.installation.resources import agent_bootstrap_text
from sim.installation.workspace import init_workspace

ROOT = Path(__file__).resolve().parents[2]
HANDLING = {"marking": "PUBLIC", "release_scope": "public"}


def handlers(tmp_path: Path, approvals: ExecutionApprovalPolicy | None = None) -> PublicOELMCPHandlers:
    return PublicOELMCPHandlers(read_roots=(ROOT, tmp_path), write_roots=(tmp_path,),
                                approval_policy=approvals or ExecutionApprovalPolicy())


def scenario_args(tmp_path: Path) -> dict:
    return {"config_path": str(ROOT / "configs/acceptance_relative_coast.yaml"),
            "output_dir": str(tmp_path / "new-run"), "resource_profile": "laptop-safe", "handling": HANDLING}


def test_workspace_and_mcp_publish_identical_hash_bound_bootstrap(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    init_workspace(workspace, engine_version="0.28.0")
    resources = {r.contract.uri: r.text for r in build_public_resource_catalog(
        profile="public_local", tool_contracts=handlers(tmp_path).contracts.values())}
    assert (workspace / "AGENTS.md").read_text() == resources[BOOTSTRAP_URI] == agent_bootstrap_text()
    manifest = json.loads((workspace / ".oel/template-manifest.json").read_text())
    assert next(r for r in manifest["files"] if r["path"] == "AGENTS.md")["sha256"] == sha256_file(workspace / "AGENTS.md")
    (workspace / "AGENTS.md").write_text("User's own instructions")
    with pytest.raises(FileExistsError):
        init_workspace(workspace, engine_version="0.28.0")
    assert (workspace / "AGENTS.md").read_text() == "User's own instructions"


def test_routes_distinguish_main_and_separate_planner(tmp_path: Path) -> None:
    h = handlers(tmp_path)
    resources = {r.contract.uri: r.text for r in build_public_resource_catalog(
        profile="public_local", tool_contracts=h.contracts.values())}
    routes = json.loads(resources[WORKFLOWS_URI])
    assert routes["study_planner"]["registered_here"] is False
    assert routes["study_planner"]["maturity"] == "prototype"
    scenario = next(r for r in routes["routes"] if r["route_id"] == "scenario")
    assert scenario["all_tools_registered_here"] is True
    assert not any(name.startswith("oel.study.") for name in h.contracts)


def test_discovery_never_confuses_configured_approvals_with_authorization(tmp_path: Path) -> None:
    disabled = handlers(tmp_path).describe_capabilities()["result"]
    assert disabled["agent_guide_uri"] == BOOTSTRAP_URI
    rows = {r["tool_id"]: r for r in disabled["readiness"]["tools"]}
    assert set(rows["oel.run_scenario.v1"]["blockers"]) == {
        "execute_approval_not_configured", "trust_approval_not_configured"}
    assert rows["oel.inspect_run.v1"]["configuration_ready"] is True
    policy = ExecutionApprovalPolicy(execution_approval_ids=frozenset({"secret-not-to-disclose"}),
                                     trust_approval_ids=frozenset({"trust-not-to-disclose"}))
    h = handlers(tmp_path, policy)
    result = h.describe_capabilities()["result"]
    run = next(r for r in result["readiness"]["tools"] if r["tool_id"] == "oel.run_scenario.v1")
    assert run["configuration_ready"] is True
    assert run["execution_authorized"] is False
    assert "not-to-disclose" not in json.dumps(result)
    with pytest.raises(PermissionError) as caught:
        h.call("oel.run_scenario.v1", {**scenario_args(tmp_path), "validation_id": "invalid",
               "approval": {"scope": "execute", "approval_id": "wrong"},
               "trust_approval": {"scope": "trust", "approval_id": "wrong"}})
    assert caught.value.oel_recovery_code == "approval.required"
    assert not (tmp_path / "new-run").exists()


def test_acceptance_config_plans_without_import_or_execution(tmp_path: Path) -> None:
    h = handlers(tmp_path)
    result = h.call("oel.plan_run.v1", scenario_args(tmp_path))
    assert result["status"] == "completed", result["error"]
    assert result["result"]["execution_authorized"] is False
    assert not (tmp_path / "new-run").exists()
    validated = h.call("oel.validate_scenario.v1", {**scenario_args(tmp_path), "trust_plugins": False})
    assert validated["result"]["status"] == "safe_only"
    assert validated["result"]["identity"]["validation_id"] == ""


def test_sealed_policy_recovery_is_mcp_specific_and_preserves_requested_evidence(tmp_path: Path) -> None:
    cfg = yaml.safe_load((ROOT / "configs/acceptance_relative_coast.yaml").read_text())
    cfg["outputs"]["stats"]["save_full_log"] = True
    path = tmp_path / "full-log.yaml"
    path.write_text(yaml.safe_dump(cfg))
    result = handlers(tmp_path).call("oel.plan_run.v1", {**scenario_args(tmp_path), "config_path": str(path)})
    assert result["status"] == "failed"
    assert result["error"]["recovery"]["code"] == "scenario.policy_blocked"
    assert "--allow-high-detail-outputs" not in result["error"]["message"]
    assert yaml.safe_load(path.read_text())["outputs"]["stats"]["save_full_log"] is True
    assert not (tmp_path / "new-run").exists()


@pytest.mark.skipif(importlib.util.find_spec("mcp") is None, reason="optional MCP SDK is not installed")
def test_real_stdio_exposes_bootstrap_and_structured_admission_recovery(tmp_path: Path) -> None:
    import anyio
    from mcp import Client, MCPError, StdioServerParameters, stdio_client

    async def exercise():
        parameters = StdioServerParameters(command=sys.executable,
            args=["-m", "integrations.oel_mcp"], cwd=ROOT,
            env={**os.environ, "OEL_MCP_ADAPTER": "sdk", "OEL_MCP_READ_ROOTS": str(ROOT),
                 "OEL_MCP_WRITE_ROOTS": str(tmp_path), "OEL_MCP_EXECUTION_APPROVAL_IDS": "",
                 "OEL_MCP_WRITE_APPROVAL_IDS": "", "OEL_MCP_TRUST_APPROVAL_IDS": ""})
        async with Client(stdio_client(parameters), mode="auto", cache=None) as client:
            guide = await client.read_resource(BOOTSTRAP_URI)
            assert guide.contents[0].text == agent_bootstrap_text()
            with pytest.raises(MCPError) as caught:
                await client.call_tool("oel.inspect_run.v1", {})
            assert caught.value.code == -32602
            assert caught.value.data["recovery"]["code"] == "handling.review_required"
            assert caught.value.data["recovery"]["actor"] == "operator"
            with pytest.raises(MCPError) as invalid:
                await client.call_tool("oel.validate_scenario.v1", {"handling": HANDLING})
            assert invalid.value.data["recovery"]["code"] == "tool.arguments_invalid"
    anyio.run(exercise)
