from __future__ import annotations

from pathlib import Path

from integrations.oel_mcp.execution import ExecutionApprovalPolicy
from integrations.oel_mcp.fsw_authoring_registry import FSW_AUTHORING_TOOL_IDS
from integrations.oel_mcp.public_handlers import PublicOELMCPHandlers

HANDLING = {"marking": "PUBLIC", "release_scope": "public", "owner": ""}


def _handlers(root: Path, *, profile: str = "public_local") -> PublicOELMCPHandlers:
    return PublicOELMCPHandlers(
        profile=profile,
        read_roots=(root,),
        write_roots=(root,),
        approval_policy=ExecutionApprovalPolicy(
            write_approval_ids=frozenset({"write-ok"}),
            execution_approval_ids=frozenset({"execute-ok"}),
            trust_approval_ids=frozenset({"trust-ok"}),
        ),
    )


def _approval(scope: str) -> dict[str, str]:
    return {"approval_id": f"{scope}-ok", "scope": scope}


def test_public_fsw_authoring_tools_are_local_only_and_explicitly_bounded(tmp_path: Path) -> None:
    public = _handlers(tmp_path)
    restricted = _handlers(tmp_path, profile="direct_frontier_restricted")

    assert all(tool_id in public.contracts for tool_id in FSW_AUTHORING_TOOL_IDS)
    assert all(tool_id not in restricted.contracts for tool_id in FSW_AUTHORING_TOOL_IDS)
    assert all(public.contracts[tool_id].limits["private_orchestration"] is False for tool_id in FSW_AUTHORING_TOOL_IDS)
    assert all(public.contracts[tool_id].limits["hidden_truth_access"] is False for tool_id in FSW_AUTHORING_TOOL_IDS)
    assert public.contracts["oel.fsw.run_candidate_smoke.v1"].limits["workers"] == 1


def test_public_fsw_mcp_scaffold_inspect_validate_and_contract_test(tmp_path: Path) -> None:
    handlers = _handlers(tmp_path)
    scaffold = handlers.call(
        "oel.fsw.scaffold_candidate.v1",
        {
            "name": "Agent ADCS",
            "template": "adcs",
            "output_dir": str(tmp_path / "candidate"),
            "handling": HANDLING,
            "approval": _approval("write"),
        },
    )
    assert scaffold["status"] == "completed"
    manifest = Path(scaffold["result"]["result"]["manifest_path"])

    inspected = handlers.call(
        "oel.fsw.inspect_candidate.v1",
        {"manifest_path": str(manifest), "handling": HANDLING},
    )
    assert inspected["status"] == "completed"
    assert inspected["result"]["candidate_code_imported"] is False
    assert inspected["result"]["private_orchestration_used"] is False

    validated = handlers.call(
        "oel.fsw.validate_candidate.v1",
        {
            "manifest_path": str(manifest),
            "output_dir": str(tmp_path / "validation"),
            "trusted_import": True,
            "trust_approval": _approval("trust"),
            "approval": _approval("write"),
            "handling": HANDLING,
        },
    )
    assert validated["status"] == "completed"
    validation_id = validated["result"]["result"]["validation_id"]

    tested = handlers.call(
        "oel.fsw.run_candidate_tests.v1",
        {
            "manifest_path": str(manifest),
            "output_dir": str(tmp_path / "tests"),
            "validation_id": validation_id,
            "trust_approval": _approval("trust"),
            "approval": _approval("execute"),
            "handling": HANDLING,
        },
    )
    assert tested["status"] == "completed"
    assert tested["result"]["status"] == "passed"
    assert tested["result"]["hidden_truth_visible"] is False


def test_public_fsw_mcp_requires_source_trust_before_candidate_import(tmp_path: Path) -> None:
    handlers = _handlers(tmp_path)
    output = tmp_path / "never-created"

    try:
        handlers.call(
            "oel.fsw.validate_candidate.v1",
            {
                "manifest_path": str(tmp_path / "missing.yaml"),
                "output_dir": str(output),
                "trusted_import": True,
                "trust_approval": {"approval_id": "wrong", "scope": "trust"},
                "approval": _approval("write"),
                "handling": HANDLING,
            },
        )
    except PermissionError as exc:
        assert "trust" in str(exc).lower()
    else:  # pragma: no cover - security assertion
        raise AssertionError("Candidate imports must require explicit source trust.")
    assert not output.exists()
