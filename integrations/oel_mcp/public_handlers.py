from __future__ import annotations

from pathlib import Path
from typing import Any

from integrations.oel_mcp.base_handlers import BaseOELMCPHandlers, require_file_size
from integrations.oel_mcp.contracts import MAX_RESPONSE_BYTES, MAX_REVIEW_STORE_BYTES, ToolContract
from integrations.oel_mcp.public_registry import public_contract_map
from sim.agent_task.runner import inspect_output
from sim.review import ReviewWorkspace


class PublicOELMCPHandlers(BaseOELMCPHandlers):
    def __init__(
        self,
        *,
        profile: str = "public_local",
        read_roots: tuple[str | Path, ...] | None = None,
        write_roots: tuple[str | Path, ...] | None = None,
        max_response_bytes: int = MAX_RESPONSE_BYTES,
    ) -> None:
        super().__init__(
            profile=profile,
            contracts=public_contract_map(profile),
            read_roots=read_roots,
            write_roots=write_roots,
            max_response_bytes=max_response_bytes,
        )

    def inspect_run(self, **arguments: Any) -> dict[str, Any]:
        return self.call("oel.inspect_run.v1", arguments)

    def query_review(self, **arguments: Any) -> dict[str, Any]:
        return self.call("oel.query_review.v1", arguments)

    def _call_contract(self, contract: ToolContract, arguments: dict[str, Any]) -> dict[str, Any]:
        if contract.tool_id == "oel.inspect_run.v1":
            return self._inspect_run(contract, arguments)
        if contract.tool_id == "oel.query_review.v1":
            return self._query_review(contract, arguments)
        raise PermissionError("Tool is not available in this deployment profile.")

    def _inspect_run(self, contract: ToolContract, arguments: dict[str, Any]) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            output = self.path_policy.resolve_read(arguments["output_dir"], kind="directory")
            db_path = output / "review" / "run.sqlite"
            if db_path.is_file():
                require_file_size(db_path, maximum=MAX_REVIEW_STORE_BYTES)
            raw = inspect_output(
                output,
                query_names=arguments.get("query_names"),
                max_rows=int(arguments.get("max_rows", 50)),
                write_packet=False,
            )
            review = dict(raw.get("review", {}) or {})
            return {
                "output_dir": str(output),
                "status": str(raw.get("status", "partial")),
                "evidence_summary": dict(raw.get("evidence_summary", {}) or {}),
                "review": {
                    key: review[key]
                    for key in ("db_path", "tables", "query_summary", "queries", "error")
                    if key in review
                },
                "artifact_summary": dict(raw.get("artifact_summary", {}) or {}),
                "failure_hints": [dict(item) for item in list(raw.get("failure_hints", []) or [])],
                "caveats": [str(item) for item in list(raw.get("caveats", []) or [])],
                "freshness": {
                    "current_config_reproduction_verified": False,
                    "meaning": "Inspection describes the output directory's current contents, not a fresh rerun.",
                },
            }

        return self._envelope(
            contract=contract,
            arguments=arguments,
            operation=operation,
            outcome_status=lambda result: "completed" if result["status"] == "completed" else "partial",
            evidence=lambda result: {
                "complete": result["status"] == "completed"
                and not bool(dict(result.get("review", {}) or {}).get("error")),
                "empty": False,
                "truncated": bool(
                    dict(dict(result.get("review", {}) or {}).get("query_summary", {}) or {}).get("truncated", 0)
                ),
            },
        )

    def _query_review(self, contract: ToolContract, arguments: dict[str, Any]) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            path = self.path_policy.resolve_read(arguments["output_dir"])
            workspace = ReviewWorkspace.open(path)
            require_file_size(workspace.db_path, maximum=MAX_REVIEW_STORE_BYTES)
            result = workspace.query(
                str(arguments["sql"]),
                max_rows=int(arguments.get("max_rows", 100)),
                max_vm_steps=int(arguments.get("max_vm_steps", 250_000)),
            )
            return {
                "output_dir": str(workspace.output_dir),
                "review_store": "review/run.sqlite",
                "sql": str(arguments["sql"]),
                "columns": result.columns,
                "rows": result.rows,
                "row_count": result.row_count,
                "empty_result": result.row_count == 0,
                "empty_result_semantics": "No matching review evidence was recorded for this query.",
                "truncated": result.truncated,
                "units_semantics": "Units are carried by selected columns and saved-query metadata; no units are inferred.",
            }

        return self._envelope(
            contract=contract,
            arguments=arguments,
            operation=operation,
            outcome_status=lambda result: "partial" if result["truncated"] else "completed",
            evidence=lambda result: {
                "complete": not result["truncated"],
                "empty": result["empty_result"],
                "truncated": result["truncated"],
            },
        )
