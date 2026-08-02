from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from integrations.oel_mcp.execution import complete_manifest, manifest_base, write_execution_manifest
from sim.agent_task.runner import inspect_output

REPORT_PACKET_SCHEMA = "oel.mcp_report_evidence.v1"
REPORT_AUDIT_SCHEMA = "oel.mcp_report_audit.v1"
MAX_REPORT_SOURCE_BYTES = 2_000_000
MAX_PACKET_ARTIFACTS = 100
MAX_REPORT_ARTIFACT_BYTES = 1_000_000_000
MAX_REPORT_TOTAL_ARTIFACT_BYTES = 2_000_000_000
_SOURCE_EVIDENCE_SUMMARY_KEYS = (
    "status",
    "validation_ok",
    "review_evidence_complete",
    "artifacts_complete",
    "plots_complete",
    "comparison_complete",
    "failure_hint_count",
    "caveat_count",
    "ready_to_cite",
)
_EVIDENCE_REFERENCE = re.compile(r"\[evidence:([A-Za-z0-9_.:-]{1,120})\]")


def prepare_report_packet(
    *,
    source_output_dir: Path,
    packet_output_dir: Path,
    packet_id: str,
    query_names: list[str],
    max_rows: int,
    handling: dict[str, Any],
    approval_id: str,
) -> dict[str, Any]:
    inspection = inspect_output(
        source_output_dir,
        query_names=query_names or None,
        max_rows=max_rows,
        write_packet=False,
    )
    artifacts, artifacts_truncated = _artifact_inventory(inspection, source_output_dir=source_output_dir)
    required_artifacts_missing = any(row["required"] and not row["exists"] for row in artifacts)
    review = _review_projection(dict(inspection.get("review", {}) or {}))
    query_evidence = _query_evidence(review)
    query_summary = dict(review.get("query_summary", {}) or {})
    queries_complete = bool(query_summary.get("evidence_complete", not query_evidence))
    failure_hints = [dict(item) for item in list(inspection.get("failure_hints", []) or [])]
    if not queries_complete:
        failure_hints.append(
            {
                "code": "review_queries_incomplete",
                "unknown_queries": list(query_summary.get("unknown_queries", []) or []),
                "failed_queries": list(query_summary.get("failed_queries", []) or []),
                "next_step": "Use supported saved-query names and resolve failed or unexpectedly empty evidence queries.",
            }
        )
    if required_artifacts_missing:
        failure_hints.append(
            {
                "code": "required_report_artifact_missing",
                "next_step": "Restore or regenerate the required run summary and review store before authoring a report.",
            }
        )
    if artifacts_truncated:
        failure_hints.append(
            {
                "code": "artifact_inventory_truncated",
                "limit": MAX_PACKET_ARTIFACTS,
                "next_step": "Narrow the source artifact set or prepare multiple explicitly scoped report packets.",
            }
        )
    evidence_summary = _evidence_summary(
        source_output_dir=source_output_dir,
        artifacts=artifacts,
        queries_complete=queries_complete,
        failure_hints=failure_hints,
    )
    packet_path = packet_output_dir / "report_evidence_packet.json"
    brief_path = packet_output_dir / "report_brief.md"
    operation_manifest_path = packet_output_dir / "mcp_execution_manifest.json"
    packet = {
        "schema_id": REPORT_PACKET_SCHEMA,
        "packet_id": packet_id,
        "generated_utc": _utc_now(),
        "source_output_dir": str(source_output_dir),
        "handling": {
            "marking": str(handling.get("marking", "")),
            "release_scope": str(handling.get("release_scope", "")),
        },
        "evidence_summary": evidence_summary,
        "review": review,
        "query_evidence": query_evidence,
        "execution_provenance": _report_execution_provenance(source_output_dir),
        "artifact_summary": {**_artifact_summary(artifacts), "truncated": artifacts_truncated},
        "artifacts_truncated": artifacts_truncated,
        "required_artifacts_missing": required_artifacts_missing,
        "artifacts": artifacts,
        "failure_hints": failure_hints,
        "caveats": [str(item) for item in list(inspection.get("caveats", []) or [])],
        "report_contract": {
            "required_sections": ["Evidence", "Limitations"],
            "evidence_reference_syntax": "[evidence:<evidence_id>]",
            "provider_call_made": False,
            "authoring_owner": "connected_agent_or_human",
        },
        "non_claims": [
            "The packet does not author a report or call a model provider.",
            "Artifact integrity does not establish that narrative claims are analytically correct.",
            "Deterministic OEL artifacts remain the evidence authority.",
        ],
    }
    packet["packet_sha256"] = _sha256_json({key: value for key, value in packet.items() if key != "packet_sha256"})
    packet_bytes = _serialized_json(packet)
    if len(packet_bytes) > MAX_REPORT_SOURCE_BYTES:
        raise ValueError("The report packet exceeds the MCP report-packet size budget.")
    _write_serialized_json(packet_path, packet_bytes)
    brief_path.write_text(_report_brief(packet), encoding="utf-8")
    manifest = manifest_base(tool_id="oel.prepare_report_packet.v1", approval_id=approval_id)
    manifest.update(
        {
            "packet_id": packet_id,
            "source_output_dir": str(source_output_dir),
            "packet_sha256": packet["packet_sha256"],
            "provider_call_made": False,
        }
    )
    packet_status = "completed" if not failure_hints and not required_artifacts_missing else "partial"
    complete_manifest(
        manifest,
        status=packet_status,
        artifacts=[str(packet_path), str(brief_path)],
    )
    write_execution_manifest(packet_output_dir, manifest)
    return {
        "status": packet_status,
        "packet_id": packet_id,
        "source_output_dir": str(source_output_dir),
        "packet_output_dir": str(packet_output_dir),
        "packet_path": str(packet_path),
        "brief_path": str(brief_path),
        "manifest_path": str(operation_manifest_path),
        "packet_sha256": packet["packet_sha256"],
        "artifact_count": len(artifacts),
        "evidence_summary": packet["evidence_summary"],
        "provider_call_made": False,
        "non_claims": list(packet["non_claims"]),
    }


def audit_report(
    *,
    report_path: Path,
    packet_path: Path,
    audit_output_dir: Path,
    author: str,
    model: str,
    approval_id: str,
) -> dict[str, Any]:
    if report_path.stat().st_size > MAX_REPORT_SOURCE_BYTES:
        raise ValueError("The report exceeds the MCP report-audit size budget.")
    if packet_path.stat().st_size > MAX_REPORT_SOURCE_BYTES:
        raise ValueError("The report packet exceeds the MCP report-audit size budget.")
    packet = _load_packet(packet_path)
    markdown = report_path.read_text(encoding="utf-8")
    integrity = _verify_packet_artifacts(packet, packet_path=packet_path)
    references = sorted(set(_EVIDENCE_REFERENCE.findall(markdown)))
    artifact_rows = [dict(item) for item in list(packet.get("artifacts", []) or [])]
    artifact_id_list = [str(item.get("artifact_id", "")) for item in artifact_rows]
    artifact_path_list = [str(item.get("relative_path", "")) for item in artifact_rows]
    query_id_list = [str(item.get("evidence_id", "")) for item in list(packet.get("query_evidence", []) or [])]
    evidence_ids = set(artifact_id_list) | set(query_id_list)
    unknown_references = sorted(set(references) - evidence_ids)
    available_evidence_ids = {
        str(item.get("artifact_id", "")) for item in artifact_rows if bool(item.get("exists", False))
    } | {
        str(item.get("evidence_id", ""))
        for item in list(packet.get("query_evidence", []) or [])
        if item.get("status") == "ok" and not item.get("truncated", False)
    }
    unavailable_references = sorted(set(references) - available_evidence_ids - set(unknown_references))
    headings = _markdown_headings(markdown)
    required_sections = [
        str(item) for item in list(dict(packet.get("report_contract", {}) or {}).get("required_sections", []) or [])
    ]
    missing_sections = [section for section in required_sections if section.casefold() not in headings]
    checks = {
        "packet_schema_supported": True,
        "packet_content_hash_valid": _packet_content_hash_valid(packet),
        "artifact_ids_unique": len(artifact_id_list) == len(set(artifact_id_list)),
        "artifact_paths_unique": len(artifact_path_list) == len(set(artifact_path_list)),
        "query_evidence_ids_unique": len(query_id_list) == len(set(query_id_list)),
        "required_core_artifacts_present": {"run_summary", "review_store"}.issubset(set(artifact_id_list)),
        "artifact_integrity_valid": integrity["valid"],
        "required_sections_present": not missing_sections,
        "evidence_references_known": not unknown_references,
        "evidence_references_available": not unavailable_references,
        "evidence_reference_present": bool(references) or not evidence_ids,
    }
    passed = all(checks.values())
    audit = {
        "schema_id": REPORT_AUDIT_SCHEMA,
        "generated_utc": _utc_now(),
        "status": "passed" if passed else "needs_review",
        "author": author,
        "model": model,
        "provider_call_made": False,
        "report_path": str(report_path),
        "packet_path": str(packet_path),
        "packet_id": str(packet.get("packet_id", "")),
        "checks": checks,
        "artifact_integrity": integrity,
        "evidence_references": references,
        "unknown_evidence_references": unknown_references,
        "unavailable_evidence_references": unavailable_references,
        "missing_required_sections": missing_sections,
        "semantic_claim_review_performed": False,
        "non_claims": [
            "This audit verifies packet structure, hashes, report structure, and evidence references.",
            "It does not determine whether every narrative interpretation or operational conclusion is correct.",
            "It does not call a model provider or authorize external release.",
        ],
    }
    audit_json = audit_output_dir / "report_audit.json"
    audit_md = audit_output_dir / "report_audit.md"
    operation_manifest_path = audit_output_dir / "mcp_execution_manifest.json"
    _write_json(audit_json, audit)
    audit_md.write_text(_audit_markdown(audit), encoding="utf-8")
    manifest = manifest_base(tool_id="oel.audit_report.v1", approval_id=approval_id)
    manifest.update(
        {
            "packet_id": audit["packet_id"],
            "audit_status": audit["status"],
            "provider_call_made": False,
            "semantic_claim_review_performed": False,
        }
    )
    complete_manifest(manifest, status="completed" if passed else "partial", artifacts=[str(audit_json), str(audit_md)])
    write_execution_manifest(audit_output_dir, manifest)
    return {
        "status": audit["status"],
        "packet_id": audit["packet_id"],
        "report_path": str(report_path),
        "packet_path": str(packet_path),
        "audit_output_dir": str(audit_output_dir),
        "audit_json_path": str(audit_json),
        "audit_markdown_path": str(audit_md),
        "manifest_path": str(operation_manifest_path),
        "checks": checks,
        "unknown_evidence_references": unknown_references,
        "unavailable_evidence_references": unavailable_references,
        "missing_required_sections": missing_sections,
        "provider_call_made": False,
        "semantic_claim_review_performed": False,
        "non_claims": list(audit["non_claims"]),
    }


def _artifact_inventory(
    inspection: dict[str, Any], *, source_output_dir: Path
) -> tuple[list[dict[str, Any]], bool]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    artifact_ids: set[str] = set()
    total_bytes = 0
    fixed_candidates = (
        ("run_summary", source_output_dir / "master_run_summary.json", "application/json", True),
        ("review_store", source_output_dir / "review" / "run.sqlite", "application/vnd.sqlite3", True),
        ("review_schema", source_output_dir / "review" / "schema.json", "application/json", False),
        ("agent_evidence_packet", source_output_dir / "agent_evidence_packet.json", "application/json", False),
        ("run_index", source_output_dir / "index.md", "text/markdown", False),
        ("execution_manifest", source_output_dir / "mcp_execution_manifest.json", "application/json", False),
    )
    candidates: list[dict[str, Any]] = []
    for artifact_id, path, media_type, required in fixed_candidates:
        candidates.append(
            {
                "artifact_id": artifact_id,
                "resolved_path": str(path),
                "media_type": media_type,
                "required": required,
            }
        )
    candidates.extend(dict(item or {}) for item in list(inspection.get("artifacts", []) or []))
    truncated = False
    for index, raw in enumerate(candidates):
        if len(rows) >= MAX_PACKET_ARTIFACTS:
            truncated = True
            break
        item = dict(raw or {})
        raw_path = str(item.get("resolved_path") or item.get("path") or "")
        if not raw_path:
            continue
        path = Path(raw_path).expanduser().resolve()
        if not _is_within(path, source_output_dir):
            continue
        relative = path.relative_to(source_output_dir).as_posix()
        if relative in seen:
            continue
        required = bool(item.get("required", False))
        exists = path.is_file()
        if not exists and not required:
            continue
        seen.add(relative)
        artifact_id = _unique_identifier(
            _safe_identifier(str(item.get("artifact_id") or item.get("artifact_key") or f"artifact_{index + 1}")),
            artifact_ids,
        )
        artifact_ids.add(artifact_id)
        artifact_bytes = path.stat().st_size if exists else 0
        if artifact_bytes > MAX_REPORT_ARTIFACT_BYTES:
            raise ValueError("An artifact exceeds the MCP report-packet size budget.")
        total_bytes += artifact_bytes
        if total_bytes > MAX_REPORT_TOTAL_ARTIFACT_BYTES:
            raise ValueError("The artifacts exceed the MCP report-packet aggregate size budget.")
        rows.append(
            {
                "artifact_id": artifact_id,
                "relative_path": relative,
                "media_type": str(item.get("media_type") or _media_type(path)),
                "required": required,
                "exists": exists,
                "bytes": artifact_bytes,
                "sha256": _sha256_file(path) if exists else "",
            }
        )
    return rows, truncated


def _review_projection(review: dict[str, Any]) -> dict[str, Any]:
    return {
        key: review[key]
        for key in ("tables", "query_summary", "queries", "error")
        if key in review
    }


def _query_evidence(review: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in list(review.get("queries", []) or []):
        item = dict(raw or {})
        name = str(item.get("name", "query"))
        rows.append(
            {
                "evidence_id": f"query.{_safe_identifier(name)}",
                "name": name,
                "status": str(item.get("status", "")),
                "known": bool(item.get("known", False)),
                "description": str(item.get("description", "")),
                "columns": [str(value) for value in list(item.get("columns", []) or [])],
                "rows": [dict(value) for value in list(item.get("rows", []) or [])],
                "row_count": int(item.get("row_count", 0) or 0),
                "empty_result": bool(item.get("empty_result", False)),
                "truncated": bool(item.get("truncated", False)),
                "reason": str(item.get("reason", "")),
            }
        )
    return rows


def _artifact_summary(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    existing = [row for row in artifacts if row["exists"]]
    missing = [row for row in artifacts if not row["exists"]]
    return {
        "total": len(artifacts),
        "existing": len(existing),
        "missing": len(missing),
        "total_bytes": sum(int(row["bytes"]) for row in existing),
        "missing_artifacts": [str(row["artifact_id"]) for row in missing],
        "required_missing": [str(row["artifact_id"]) for row in missing if row["required"]],
        "artifacts_complete": not any(row["required"] and not row["exists"] for row in artifacts),
    }


def _evidence_summary(
    *,
    source_output_dir: Path,
    artifacts: list[dict[str, Any]],
    queries_complete: bool,
    failure_hints: list[dict[str, Any]],
) -> dict[str, Any]:
    source_summary: dict[str, Any] = {}
    source_packet = source_output_dir / "agent_evidence_packet.json"
    if source_packet.is_file() and source_packet.stat().st_size <= MAX_REPORT_SOURCE_BYTES:
        try:
            payload = json.loads(source_packet.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        if isinstance(payload, dict):
            raw_summary = dict(payload.get("evidence_summary", {}) or {})
            source_summary = {
                key: raw_summary[key]
                for key in _SOURCE_EVIDENCE_SUMMARY_KEYS
                if key in raw_summary and isinstance(raw_summary[key], (str, int, float, bool, type(None)))
            }
    required_complete = not any(row["required"] and not row["exists"] for row in artifacts)
    source_summary.update(
        {
            "ready_to_cite": not failure_hints and queries_complete and required_complete,
            "review_queries_complete": queries_complete,
            "required_artifacts_complete": required_complete,
            "artifact_count": len(artifacts),
        }
    )
    return source_summary


def _report_execution_provenance(source_output_dir: Path) -> dict[str, Any]:
    manifest_path = source_output_dir / "mcp_execution_manifest.json"
    if not manifest_path.is_file() or manifest_path.stat().st_size > MAX_REPORT_SOURCE_BYTES:
        return {"available": False}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"available": False}
    if not isinstance(payload, dict):
        return {"available": False}
    allowed = (
        "tool_id",
        "status",
        "recipe_id",
        "validation_id",
        "source_config_sha256",
        "normalized_config_sha256",
        "resource_profile",
        "artifacts_complete",
        "cancelled",
        "started_utc",
        "completed_utc",
    )
    return {"available": True, **{key: payload.get(key) for key in allowed if key in payload}}


def _load_packet(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_id") != REPORT_PACKET_SCHEMA:
        raise ValueError("The report packet does not use the supported OEL MCP report schema.")
    if not _packet_content_hash_valid(payload):
        raise ValueError("The report packet content hash does not match its content.")
    if not isinstance(payload.get("artifacts"), list) or not isinstance(payload.get("report_contract"), dict):
        raise ValueError("The report packet is missing its artifact or report contract.")
    return payload


def _packet_content_hash_valid(packet: dict[str, Any]) -> bool:
    expected = str(packet.get("packet_sha256", ""))
    actual = _sha256_json({key: value for key, value in packet.items() if key != "packet_sha256"})
    return len(expected) == 64 and expected == actual


def _verify_packet_artifacts(packet: dict[str, Any], *, packet_path: Path) -> dict[str, Any]:
    source_root = Path(str(packet.get("source_output_dir", ""))).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    valid = True
    for raw in list(packet.get("artifacts", []) or []):
        item = dict(raw or {})
        relative = str(item.get("relative_path", ""))
        target = (source_root / relative).resolve()
        within = _is_within(target, source_root)
        exists = within and target.is_file()
        actual_sha = _sha256_file(target) if exists else ""
        matched = exists and actual_sha == str(item.get("sha256", ""))
        required = bool(item.get("required", False))
        valid = valid and matched
        rows.append(
            {
                "artifact_id": str(item.get("artifact_id", "")),
                "relative_path": relative,
                "required": required,
                "exists": exists,
                "sha256_matches": matched,
            }
        )
    return {"valid": valid, "packet_path": str(packet_path), "artifacts": rows}


def _report_brief(packet: dict[str, Any]) -> str:
    artifact_lines = [
        f"- `{row['artifact_id']}`: `{row['relative_path']}` — "
        + (
            f"cite as `[evidence:{row['artifact_id']}]`"
            if row.get("exists")
            else "required artifact is missing; do not cite as available evidence"
        )
        for row in list(packet.get("artifacts", []) or [])
    ]
    query_lines = [
        f"- `{row['evidence_id']}`: saved query `{row['name']}` ({row['status']}) — "
        + (
            f"cite as `[evidence:{row['evidence_id']}]`"
            if row.get("status") == "ok" and not row.get("truncated")
            else "not available as complete report evidence"
        )
        for row in list(packet.get("query_evidence", []) or [])
    ]
    provenance = dict(packet.get("execution_provenance", {}) or {})
    provenance_lines = [
        f"- {key}: `{value}`"
        for key, value in provenance.items()
        if key != "available" and value not in {None, ""}
    ]
    return "\n".join(
        [
            "# OEL MCP Report Brief",
            "",
            "Author the report from the deterministic packet and cited artifacts. OEL has not called a model provider.",
            "",
            "Required report sections: `Evidence` and `Limitations`.",
            "",
            "## Evidence References",
            "",
            *(artifact_lines or ["- No file artifacts were available; explain the incomplete evidence explicitly."]),
            *(query_lines or ["- No saved-query evidence was requested."]),
            "",
            "## Execution Provenance",
            "",
            *(provenance_lines or ["- No MCP execution manifest was available for projection."]),
            "",
            "## Non-Claims",
            "",
            *(f"- {item}" for item in list(packet.get("non_claims", []) or [])),
            "",
        ]
    )


def _audit_markdown(audit: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# OEL MCP Report Audit",
            "",
            f"- Status: `{audit['status']}`",
            f"- Packet: `{audit['packet_id']}`",
            "- Provider call made by OEL: `false`",
            "- Semantic claim review performed: `false`",
            "",
            "## Checks",
            "",
            *(f"- {key}: `{'passed' if value else 'failed'}`" for key, value in audit["checks"].items()),
            "",
            "## Non-Claims",
            "",
            *(f"- {item}" for item in audit["non_claims"]),
            "",
        ]
    )


def _markdown_headings(markdown: str) -> set[str]:
    return {
        match.group(1).strip().casefold()
        for line in markdown.splitlines()
        if (match := re.match(r"^#{1,6}\s+(.+?)\s*$", line))
    }


def _safe_identifier(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.:-]+", "_", value.strip()).strip("_")
    return (normalized or "artifact")[:120]


def _unique_identifier(candidate: str, existing: set[str]) -> str:
    if candidate not in existing:
        return candidate
    suffix = 2
    while f"{candidate[:115]}_{suffix}" in existing:
        suffix += 1
    return f"{candidate[:115]}_{suffix}"


def _media_type(path: Path) -> str:
    return {
        ".json": "application/json",
        ".md": "text/markdown",
        ".csv": "text/csv",
        ".sqlite": "application/vnd.sqlite3",
        ".png": "image/png",
        ".svg": "image/svg+xml",
        ".pdf": "application/pdf",
    }.get(path.suffix.lower(), "application/octet-stream")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _write_serialized_json(path, _serialized_json(payload))


def _serialized_json(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _write_serialized_json(path: Path, payload: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(payload)
    tmp.replace(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


__all__ = [
    "MAX_REPORT_SOURCE_BYTES",
    "MAX_REPORT_ARTIFACT_BYTES",
    "MAX_REPORT_TOTAL_ARTIFACT_BYTES",
    "REPORT_AUDIT_SCHEMA",
    "REPORT_PACKET_SCHEMA",
    "audit_report",
    "prepare_report_packet",
]
