from __future__ import annotations

import json
from pathlib import Path
from threading import Event
from typing import Any

from integrations.oel_mcp.base_handlers import BaseOELMCPHandlers, require_file_size
from integrations.oel_mcp.contracts import MAX_MANIFEST_BYTES, MAX_RESPONSE_BYTES, MAX_REVIEW_STORE_BYTES, ToolContract
from integrations.oel_mcp.execution import (
    ExecutionApprovalPolicy,
    MCPExecutionCancelled,
    cancellation_callback,
    complete_manifest,
    ensure_new_output_dir,
    manifest_base,
    prepare_scenario,
    require_safe_resource_estimate,
    resource_estimate,
    safe_artifact_id,
    validate_prepared_scenario,
    write_execution_manifest,
    write_materialized_config,
)
from integrations.oel_mcp.public_registry import public_contract_map
from integrations.oel_mcp.reporting import audit_report as audit_report_artifacts
from integrations.oel_mcp.reporting import prepare_report_packet as prepare_report_packet_artifacts
from sim.agent_task.plot_recipes import get_plot_recipe
from sim.agent_task.recipes import get_recipe
from sim.agent_task.runner import AgentTaskCancelled, compare_outputs, create_plot, inspect_output, run_recipe
from sim.execution import run_simulation_config_file
from sim.review import ReviewWorkspace


class PublicOELMCPHandlers(BaseOELMCPHandlers):
    def __init__(
        self,
        *,
        profile: str = "public_local",
        read_roots: tuple[str | Path, ...] | None = None,
        write_roots: tuple[str | Path, ...] | None = None,
        max_response_bytes: int = MAX_RESPONSE_BYTES,
        approval_policy: ExecutionApprovalPolicy | None = None,
    ) -> None:
        super().__init__(
            profile=profile,
            contracts=public_contract_map(profile),
            read_roots=read_roots,
            write_roots=write_roots,
            max_response_bytes=max_response_bytes,
            approval_policy=approval_policy,
        )

    def inspect_run(self, **arguments: Any) -> dict[str, Any]:
        return self.call("oel.inspect_run.v1", arguments)

    def query_review(self, **arguments: Any) -> dict[str, Any]:
        return self.call("oel.query_review.v1", arguments)

    def plan_run(self, **arguments: Any) -> dict[str, Any]:
        return self.call("oel.plan_run.v1", arguments)

    def validate_scenario(self, **arguments: Any) -> dict[str, Any]:
        return self.call("oel.validate_scenario.v1", arguments)

    def run_scenario(self, **arguments: Any) -> dict[str, Any]:
        return self.call("oel.run_scenario.v1", arguments)

    def compare_runs(self, **arguments: Any) -> dict[str, Any]:
        return self.call("oel.compare_runs.v1", arguments)

    def plot_evidence(self, **arguments: Any) -> dict[str, Any]:
        return self.call("oel.plot_evidence.v1", arguments)

    def run_agent_task(self, **arguments: Any) -> dict[str, Any]:
        return self.call("oel.run_agent_task.v1", arguments)

    def prepare_report_packet(self, **arguments: Any) -> dict[str, Any]:
        return self.call("oel.prepare_report_packet.v1", arguments)

    def audit_report(self, **arguments: Any) -> dict[str, Any]:
        return self.call("oel.audit_report.v1", arguments)

    def _call_contract(
        self,
        contract: ToolContract,
        arguments: dict[str, Any],
        *,
        cancel_event: Event | None = None,
        progress: Any | None = None,
    ) -> dict[str, Any]:
        if contract.tool_id == "oel.inspect_run.v1":
            return self._inspect_run(contract, arguments)
        if contract.tool_id == "oel.query_review.v1":
            return self._query_review(contract, arguments)
        if contract.tool_id == "oel.plan_run.v1":
            return self._plan_run(contract, arguments)
        if contract.tool_id == "oel.validate_scenario.v1":
            return self._validate_scenario(contract, arguments)
        if contract.tool_id == "oel.run_scenario.v1":
            return self._run_scenario(contract, arguments, cancel_event=cancel_event, progress=progress)
        if contract.tool_id == "oel.compare_runs.v1":
            return self._compare_runs(contract, arguments)
        if contract.tool_id == "oel.plot_evidence.v1":
            return self._plot_evidence(contract, arguments)
        if contract.tool_id == "oel.run_agent_task.v1":
            return self._run_agent_task(contract, arguments, cancel_event=cancel_event, progress=progress)
        if contract.tool_id == "oel.prepare_report_packet.v1":
            return self._prepare_report_packet(contract, arguments)
        if contract.tool_id == "oel.audit_report.v1":
            return self._audit_report(contract, arguments)
        raise PermissionError("Tool is not available in this deployment profile.")

    def _inspect_run(self, contract: ToolContract, arguments: dict[str, Any]) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            output = self.path_policy.resolve_read(arguments["output_dir"], kind="directory")
            _authorize_review_inputs(self.path_policy, output, require_store=False)
            db_path = self.path_policy.resolve_read_child(
                output, "review", "run.sqlite", kind="file", required=False
            )
            if db_path.is_file():
                require_file_size(db_path, maximum=MAX_REVIEW_STORE_BYTES)
            raw = inspect_output(
                output,
                query_names=arguments.get("query_names"),
                max_rows=int(arguments.get("max_rows", 50)),
                write_packet=False,
            )
            review = dict(raw.get("review", {}) or {})
            execution_provenance = _execution_provenance(output)
            artifact_summary = dict(raw.get("artifact_summary", {}) or {})
            artifact_summary.update(
                {
                    "mcp_manifest_artifact_count": execution_provenance["artifact_count"],
                    "mcp_manifest_artifacts_complete": execution_provenance["artifacts_complete"],
                }
            )
            evidence_summary = dict(raw.get("evidence_summary", {}) or {})
            evidence_summary["mcp_execution_complete"] = bool(
                execution_provenance["available"]
                and execution_provenance["status"] == "completed"
                and execution_provenance["artifacts_complete"]
            )
            return {
                "output_dir": str(output),
                "status": str(raw.get("status", "partial")),
                "evidence_summary": evidence_summary,
                "review": {
                    key: review[key]
                    for key in ("db_path", "tables", "query_summary", "queries", "error")
                    if key in review
                },
                "artifact_summary": artifact_summary,
                "execution_provenance": execution_provenance,
                "failure_hints": [dict(item) for item in list(raw.get("failure_hints", []) or [])],
                "caveats": [str(item) for item in list(raw.get("caveats", []) or [])],
                "freshness": {
                    "current_config_reproduction_verified": False,
                    "content_bound_execution_recorded": bool(execution_provenance["validation_id"]),
                    "meaning": (
                        "Inspection verifies the current output contents and reports their MCP execution manifest; "
                        "it does not rerun the source configuration."
                    ),
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
            path = self.path_policy.resolve_read(arguments["output_dir"], kind="directory")
            _authorize_review_inputs(self.path_policy, path, require_store=True)
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

    def _prepared(self, arguments: dict[str, Any]) -> Any:
        return prepare_scenario(
            config_path=arguments["config_path"],
            output_dir=arguments["output_dir"],
            resource_profile=str(arguments.get("resource_profile", "laptop-safe")),
            path_policy=self.path_policy,
        )

    @staticmethod
    def _scenario_projection(prepared: Any) -> dict[str, Any]:
        cfg = prepared.config
        return {
            "scenario_name": cfg.scenario_name,
            "scenario_description": cfg.scenario_description,
            "duration_s": float(cfg.simulator.duration_s),
            "dt_s": float(cfg.simulator.dt_s),
            "output_dir": str(prepared.output_dir),
            "resource_profile": prepared.resource_profile,
        }

    @staticmethod
    def _identity_projection(prepared: Any, *, validation_id: str = "") -> dict[str, Any]:
        return {
            "source_config_sha256": prepared.raw_sha256,
            "normalized_config_sha256": prepared.normalized_sha256,
            "validation_id": validation_id,
        }

    def _plan_run(self, contract: ToolContract, arguments: dict[str, Any]) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            prepared = self._prepared(arguments)
            validation = validate_prepared_scenario(prepared, path_policy=self.path_policy, trust_plugins=False)
            return {
                "scenario": self._scenario_projection(prepared),
                "identity": self._identity_projection(prepared),
                "safe_validation": validation["safe_validation"],
                "resource_estimate": resource_estimate(prepared),
                "phases": [
                    "safe_validate",
                    "trusted_validate",
                    "resource_preflight",
                    "operator_approval",
                    "deterministic_execute",
                    "inspect_evidence",
                ],
                "approval": {
                    "required_for_execution": True,
                    "configured_outside_model": True,
                    "environment": "OEL_MCP_EXECUTION_APPROVAL_IDS",
                },
                "execution_authorized": False,
            }

        return self._envelope(
            contract=contract,
            arguments=arguments,
            operation=operation,
            outcome_status=lambda result: "completed" if result["safe_validation"].get("ok") else "partial",
            evidence=lambda result: {
                "complete": bool(result["safe_validation"].get("ok")),
                "empty": False,
                "truncated": False,
            },
        )

    def _validate_scenario(self, contract: ToolContract, arguments: dict[str, Any]) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            prepared = self._prepared(arguments)
            validation = validate_prepared_scenario(
                prepared,
                path_policy=self.path_policy,
                trust_plugins=bool(arguments.get("trust_plugins", False)),
            )
            estimate = resource_estimate(prepared)
            status = str(validation["status"])
            return {
                "status": status,
                "scenario": self._scenario_projection(prepared),
                "identity": self._identity_projection(prepared, validation_id=str(validation["validation_id"])),
                "safe_validation": validation["safe_validation"],
                "trusted_validation": validation["trusted_validation"],
                "resource_estimate": estimate,
                "execution_ready": bool(validation["execution_ready"]) and estimate["action"] != "refuse",
                "execution_authorized": False,
                "next_step": _validation_next_step(
                    validation=validation,
                    resource_action=str(estimate["action"]),
                    trust_plugins=bool(arguments.get("trust_plugins", False)),
                ),
            }

        return self._envelope(
            contract=contract,
            arguments=arguments,
            operation=operation,
            outcome_status=lambda result: (
                "completed" if result["status"] == "validated" and result["execution_ready"] else "partial"
            ),
            evidence=lambda result: {"complete": bool(result["execution_ready"]), "empty": False, "truncated": False},
        )

    def _run_scenario(
        self,
        contract: ToolContract,
        arguments: dict[str, Any],
        *,
        cancel_event: Event | None,
        progress: Any | None,
    ) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            prepared = self._prepared(arguments)
            validation = validate_prepared_scenario(prepared, path_policy=self.path_policy, trust_plugins=True)
            if not validation["execution_ready"] or arguments["validation_id"] != prepared.validation_id:
                raise PermissionError("The trusted validation id does not match the exact execution-normalized config.")
            estimate = resource_estimate(prepared)
            require_safe_resource_estimate(estimate)
            ensure_new_output_dir(prepared.output_dir)
            approval_id = str(arguments["approval"]["approval_id"])
            manifest = manifest_base(tool_id=contract.tool_id, approval_id=approval_id, prepared=prepared)
            manifest["resource_estimate"] = estimate
            manifest_path = write_execution_manifest(prepared.output_dir, manifest)
            materialized = write_materialized_config(prepared)
            if progress:
                progress(1, 3, "Validated and materialized the approved scenario.")
            try:
                payload = run_simulation_config_file(
                    materialized,
                    step_callback=cancellation_callback(cancel_event),
                )
                if progress:
                    progress(2, 3, "Deterministic execution completed; collecting artifacts.")
                artifacts = _existing_artifacts(prepared.output_dir, materialized)
                complete_manifest(manifest, status="completed", artifacts=artifacts)
                write_execution_manifest(prepared.output_dir, manifest)
                if progress:
                    progress(3, 3, "Execution evidence manifest completed.")
                return {
                    "status": "completed",
                    "scenario": self._scenario_projection(prepared),
                    "identity": self._identity_projection(prepared, validation_id=prepared.validation_id),
                    "output_dir": str(prepared.output_dir),
                    "run": _run_projection(payload),
                    "artifacts": artifacts,
                    "manifest_path": str(manifest_path),
                }
            except MCPExecutionCancelled:
                artifacts = _existing_artifacts(prepared.output_dir, materialized)
                complete_manifest(
                    manifest,
                    status="cancelled",
                    cancelled=True,
                    artifacts=artifacts,
                    error_type="MCPExecutionCancelled",
                )
                write_execution_manifest(prepared.output_dir, manifest)
                return {
                    "status": "cancelled",
                    "scenario": self._scenario_projection(prepared),
                    "identity": self._identity_projection(prepared, validation_id=prepared.validation_id),
                    "output_dir": str(prepared.output_dir),
                    "run": {"cancelled": True},
                    "artifacts": artifacts,
                    "manifest_path": str(manifest_path),
                }
            except Exception as exc:
                complete_manifest(
                    manifest,
                    status="failed",
                    artifacts=_existing_artifacts(prepared.output_dir, materialized),
                    error_type=type(exc).__name__,
                )
                write_execution_manifest(prepared.output_dir, manifest)
                raise

        return self._envelope(
            contract=contract,
            arguments=arguments,
            operation=operation,
            outcome_status=lambda result: "completed" if result["status"] == "completed" else "partial",
            evidence=lambda result: {"complete": result["status"] == "completed", "empty": False, "truncated": False},
        )

    def _compare_runs(self, contract: ToolContract, arguments: dict[str, Any]) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            base = self.path_policy.resolve_read(arguments["base_output_dir"], kind="directory")
            candidate = self.path_policy.resolve_read(arguments["candidate_output_dir"], kind="directory")
            _authorize_review_inputs(self.path_policy, base, require_store=False)
            _authorize_review_inputs(self.path_policy, candidate, require_store=False)
            raw = compare_outputs(
                base,
                candidate,
                metric_names=arguments["metric_names"],
                max_rows=int(arguments.get("max_rows", 50)),
            )
            return {
                "status": raw["status"],
                "base_output_dir": str(base),
                "candidate_output_dir": str(candidate),
                "metric_names": raw["metric_names"],
                "metrics": raw["metrics"],
                "deltas": raw["deltas"],
                "metric_status": raw["metric_status"],
                "summary": raw["summary"],
            }

        return self._envelope(
            contract=contract,
            arguments=arguments,
            operation=operation,
            outcome_status=lambda result: (
                result["status"] if result["status"] in {"completed", "failed"} else "partial"
            ),
            evidence=lambda result: {
                "complete": result["status"] == "completed",
                "empty": not bool(result["metric_names"]),
                "truncated": False,
            },
        )

    def _plot_evidence(self, contract: ToolContract, arguments: dict[str, Any]) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            output = self.path_policy.resolve_read(arguments["output_dir"], kind="directory")
            _authorize_review_inputs(self.path_policy, output, require_store=True)
            self.path_policy.resolve_write(output)
            recipe_id = str(arguments["recipe_id"])
            recipe = get_plot_recipe(recipe_id)
            if recipe is None or recipe.maturity != "supported":
                raise ValueError("Only supported allowlisted plot recipes are available.")
            artifact_id = safe_artifact_id(str(arguments["artifact_id"]))
            file_format = str(arguments.get("format", "png"))
            target = self.path_policy.resolve_write(output / "review" / "mcp_plots" / f"{artifact_id}.{file_format}")
            artifact = create_plot(
                output,
                recipe_id,
                style_name=str(arguments.get("style", "oel_dark")),
                file_format=file_format,
                artifact_id=artifact_id,
                path=target,
            )
            manifest = complete_manifest(
                manifest_base(tool_id=contract.tool_id, approval_id=str(arguments["approval"]["approval_id"])),
                status="completed",
                artifacts=[str(target)],
            )
            manifest_path = write_execution_manifest(
                output / "review" / "mcp_plots",
                manifest,
                filename=f"{artifact_id}.manifest.json",
            )
            return {
                "status": "completed" if artifact.get("status") == "ok" and artifact.get("path_exists") else "partial",
                "output_dir": str(output),
                "recipe_id": recipe_id,
                "artifact": artifact,
                "manifest_path": str(manifest_path),
            }

        return self._envelope(
            contract=contract,
            arguments=arguments,
            operation=operation,
            outcome_status=lambda result: result["status"],
            evidence=lambda result: {
                "complete": result["status"] == "completed",
                "empty": False,
                "truncated": bool(result["artifact"].get("truncated")),
            },
        )

    def _run_agent_task(
        self,
        contract: ToolContract,
        arguments: dict[str, Any],
        *,
        cancel_event: Event | None,
        progress: Any | None,
    ) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            recipe_id = str(arguments["recipe_id"])
            recipe = get_recipe(recipe_id)
            if (
                recipe is None
                or recipe.maturity != "supported"
                or "public" not in recipe.tags
                or recipe.workflow != "scenario_run"
            ):
                raise ValueError("Only supported public scenario-run task recipes are available.")
            output = self.path_policy.resolve_write(arguments["output_dir"])
            ensure_new_output_dir(output)
            manifest = manifest_base(tool_id=contract.tool_id, approval_id=str(arguments["approval"]["approval_id"]))
            manifest.update(
                {"recipe_id": recipe_id, "resource_profile": arguments["resource_profile"], "output_dir": str(output)}
            )
            manifest_path = write_execution_manifest(output, manifest)
            if progress:
                progress(1, 3, "Prepared the approved public task recipe.")
            try:
                packet = run_recipe(
                    recipe_id,
                    output_dir=output,
                    make_plots=bool(arguments.get("make_plots", False)),
                    style_name=str(arguments.get("style", "oel_dark")),
                    max_rows=int(arguments.get("max_rows", 50)),
                    resource_profile=str(arguments["resource_profile"]),
                    step_callback=_task_cancellation_callback(cancel_event),
                )
                status = str(packet.get("status", "failed"))
                if status not in {"completed", "partial", "failed"}:
                    status = "failed"
                artifacts = [
                    str(item.get("resolved_path") or item.get("path"))
                    for item in packet.get("artifacts", [])
                    if item.get("resolved_path") or item.get("path")
                ]
                if packet.get("packet_path"):
                    artifacts.append(str(packet["packet_path"]))
                complete_manifest(manifest, status=status, artifacts=artifacts)
                write_execution_manifest(output, manifest)
                if progress:
                    progress(3, 3, f"Task evidence packet finished with status {status}.")
                return {
                    "status": status,
                    "recipe_id": recipe_id,
                    "recipe_maturity": recipe.maturity,
                    "output_dir": str(output),
                    "evidence_summary": dict(packet.get("evidence_summary", {}) or {}),
                    "packet_path": str(packet.get("packet_path", "")),
                    "manifest_path": str(manifest_path),
                    "artifacts": [dict(item) for item in packet.get("artifacts", [])],
                    "failure_hints": [dict(item) for item in list(packet.get("failure_hints", []) or [])],
                }
            except AgentTaskCancelled:
                complete_manifest(manifest, status="cancelled", cancelled=True, error_type="MCPExecutionCancelled")
                write_execution_manifest(output, manifest)
                return {
                    "status": "cancelled",
                    "recipe_id": recipe_id,
                    "recipe_maturity": recipe.maturity,
                    "output_dir": str(output),
                    "evidence_summary": {"ready_to_cite": False},
                    "packet_path": "",
                    "manifest_path": str(manifest_path),
                    "artifacts": [],
                    "failure_hints": [],
                }
            except Exception as exc:
                complete_manifest(manifest, status="failed", error_type=type(exc).__name__)
                write_execution_manifest(output, manifest)
                raise

        return self._envelope(
            contract=contract,
            arguments=arguments,
            operation=operation,
            outcome_status=lambda result: (
                result["status"] if result["status"] in {"completed", "failed"} else "partial"
            ),
            evidence=lambda result: {"complete": result["status"] == "completed", "empty": False, "truncated": False},
        )

    def _prepare_report_packet(self, contract: ToolContract, arguments: dict[str, Any]) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            source = self.path_policy.resolve_read(arguments["source_output_dir"], kind="directory")
            _authorize_review_inputs(self.path_policy, source, require_store=False, report_sources=True)
            output = self.path_policy.resolve_write(arguments["packet_output_dir"])
            ensure_new_output_dir(output)
            packet_id = safe_artifact_id(str(arguments["packet_id"]))
            return prepare_report_packet_artifacts(
                source_output_dir=source,
                packet_output_dir=output,
                packet_id=packet_id,
                query_names=[str(item) for item in list(arguments.get("query_names", []) or [])],
                max_rows=int(arguments.get("max_rows", 50)),
                handling=dict(arguments["handling"]),
                approval_id=str(arguments["approval"]["approval_id"]),
            )

        return self._envelope(
            contract=contract,
            arguments=arguments,
            operation=operation,
            outcome_status=lambda result: result["status"],
            evidence=lambda result: {
                "complete": result["status"] == "completed",
                "empty": result["artifact_count"] == 0,
                "truncated": False,
            },
        )

    def _audit_report(self, contract: ToolContract, arguments: dict[str, Any]) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            report = self.path_policy.resolve_read(arguments["report_path"], kind="file")
            packet = self.path_policy.resolve_read(arguments["packet_path"], kind="file")
            packet_payload = json.loads(packet.read_text(encoding="utf-8"))
            if not isinstance(packet_payload, dict):
                raise ValueError("The report packet JSON root must be an object.")
            self.path_policy.resolve_read(str(packet_payload.get("source_output_dir", "")), kind="directory")
            output = self.path_policy.resolve_write(arguments["audit_output_dir"])
            ensure_new_output_dir(output)
            return audit_report_artifacts(
                report_path=report,
                packet_path=packet,
                audit_output_dir=output,
                author=str(arguments["author"]),
                model=str(arguments.get("model", "")),
                approval_id=str(arguments["approval"]["approval_id"]),
            )

        return self._envelope(
            contract=contract,
            arguments=arguments,
            operation=operation,
            outcome_status=lambda result: "completed" if result["status"] == "passed" else "partial",
            evidence=lambda result: {
                "complete": result["status"] == "passed",
                "empty": False,
                "truncated": False,
            },
        )


def _run_projection(payload: dict[str, Any]) -> dict[str, Any]:
    run = dict(payload.get("run", {}) or payload.get("summary", {}) or {})
    keys = (
        "scenario_name",
        "duration_s",
        "dt_s",
        "samples",
        "termination_status",
        "output_dir",
        "closest_approach_km",
        "final_range_km",
        "total_delta_v_m_s",
    )
    return {key: run[key] for key in keys if key in run}


def _existing_artifacts(output_dir: Path, materialized: Path) -> list[str]:
    candidates = (
        materialized,
        output_dir / "index.md",
        output_dir / "master_run_summary.json",
        output_dir / "review" / "run.sqlite",
        output_dir / "agent_evidence_packet.json",
    )
    return [str(path) for path in candidates if path.exists()]


def _authorize_review_inputs(
    path_policy: Any,
    output_dir: Path,
    *,
    require_store: bool,
    report_sources: bool = False,
) -> None:
    children = [
        (("review", "run.sqlite"), "file", require_store),
        (("review", "schema.json"), "file", False),
        (("review", "saved_views.json"), "file", False),
        (("review", "workflow_manifest.json"), "file", False),
        (("mcp_execution_manifest.json",), "file", False),
    ]
    if report_sources:
        children.extend(
            [
                (("master_run_summary.json",), "file", False),
                (("agent_evidence_packet.json",), "file", False),
                (("index.md",), "file", False),
            ]
        )
    for parts, kind, required in children:
        path_policy.resolve_read_child(output_dir, *parts, kind=kind, required=required)


def _execution_provenance(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "mcp_execution_manifest.json"
    empty = {
        "available": False,
        "status": "",
        "tool_id": "",
        "validation_id": "",
        "source_config_sha256": "",
        "normalized_config_sha256": "",
        "resource_profile": "",
        "artifacts_complete": False,
        "artifact_count": 0,
        "cancelled": False,
        "started_utc": None,
        "completed_utc": None,
    }
    if not manifest_path.is_file():
        return empty
    require_file_size(manifest_path, maximum=MAX_MANIFEST_BYTES)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("The MCP execution manifest must contain a JSON object.")
    return {
        "available": True,
        "status": str(payload.get("status", "")),
        "tool_id": str(payload.get("tool_id", "")),
        "validation_id": str(payload.get("validation_id", "")),
        "source_config_sha256": str(payload.get("source_config_sha256", "")),
        "normalized_config_sha256": str(payload.get("normalized_config_sha256", "")),
        "resource_profile": str(payload.get("resource_profile", "")),
        "artifacts_complete": bool(payload.get("artifacts_complete", False)),
        "artifact_count": len(list(payload.get("artifacts", []) or [])),
        "cancelled": bool(payload.get("cancelled", False)),
        "started_utc": payload.get("started_utc"),
        "completed_utc": payload.get("completed_utc"),
    }


def _validation_next_step(
    *,
    validation: dict[str, Any],
    resource_action: str,
    trust_plugins: bool,
) -> str:
    if resource_action == "refuse":
        return "Reduce the resource envelope and validate again before requesting execution approval."
    if bool(validation.get("execution_ready")):
        return "Obtain an operator-configured execution approval and call run_scenario with this validation id."
    if not trust_plugins and bool(dict(validation.get("safe_validation", {}) or {}).get("ok")):
        return "Review the referenced plugins and paths, then repeat validation with an operator-configured trust approval."
    return "Resolve the reported validation issues and validate again before requesting execution approval."


def _task_cancellation_callback(cancel_event: Event | None) -> Any:
    def check(*_args: Any, **_kwargs: Any) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise AgentTaskCancelled("The MCP task was cancelled at a deterministic workflow boundary.")

    return check
