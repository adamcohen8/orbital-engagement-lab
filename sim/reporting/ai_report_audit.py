# ruff: noqa: F401,F403,F405,I001
from .ai_report_models import *
from .ai_report_review import *
from .ai_report_budget import *

def _prepend_agent_report_provenance(
    report_markdown: str,
    *,
    metadata: dict[str, Any],
    quality: dict[str, Any],
) -> str:
    if report_markdown.lstrip().startswith("<!-- OEL agent report provenance -->"):
        return report_markdown
    quality_status = "passed" if bool(quality.get("passed", False)) else "needs review"
    header = "\n".join(
        [
            "<!-- OEL agent report provenance -->",
            "> OEL agent-authored report  ",
            f"> Author: `{_markdown_scalar(metadata.get('author'))}`  ",
            f"> Model: `{_markdown_scalar(metadata.get('model'))}`  ",
            f"> Quality: `{quality_status}`  ",
            f"> Evidence packet: `{_markdown_scalar(metadata.get('packet_json'))}`  ",
            f"> Source draft: `{_markdown_scalar(metadata.get('source_report_md'))}`  ",
            f"> Generated UTC: `{_markdown_scalar(metadata.get('generated_utc'))}`",
            "",
        ]
    )
    return header + report_markdown.lstrip()


def write_agent_report_audit_artifacts(
    *,
    report_path: str | Path,
    packet_path: str | Path,
    outdir: Path,
    author: str = "coding_agent",
    model: str = "",
    fail_on_quality: bool = False,
) -> dict[str, Any]:
    """Render figures and audit agent-authored Markdown against an OEL evidence packet."""

    source_report_path = Path(report_path).expanduser().resolve()
    source_packet_path = Path(packet_path).expanduser().resolve()
    if not source_report_path.is_file():
        raise FileNotFoundError(f"Agent-authored report does not exist: {source_report_path}")
    if not source_packet_path.is_file():
        raise FileNotFoundError(f"Agent report packet does not exist: {source_packet_path}")
    packet_data = json.loads(source_packet_path.read_text(encoding="utf-8"))
    if not isinstance(packet_data, dict):
        raise ValueError("Agent report packet JSON root must be an object.")
    packet = dict(packet_data)
    schema_version = str(packet.get("packet_schema_version", "") or "")
    if schema_version != "oel.agent_report_packet.v1":
        raise ValueError(
            "Agent report audit requires packet_schema_version='oel.agent_report_packet.v1'. "
            "Prepare a new packet with OEL before auditing the report."
        )
    if not isinstance(packet.get("report_rules"), list) or not str(packet.get("payload_kind", "") or ""):
        raise ValueError("Agent report packet is missing report_rules or payload_kind.")

    outdir.mkdir(parents=True, exist_ok=True)
    raw_markdown = source_report_path.read_text(encoding="utf-8")
    rendered_markdown, inserted_figures, unknown_placeholders = _render_figure_placeholders(
        raw_markdown,
        packet,
        base_dir=outdir,
    )
    report_options = dict(packet.get("report_options", {}) or {})
    metadata: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "workflow": "agent_authored_report_audit",
        "authoring_mode": "external_agent",
        "provider_call_made": False,
        "status": "ok",
        "author": str(author or "coding_agent"),
        "model": str(model or ""),
        "provider": "not_applicable",
        "prompt_profile": str(report_options.get("prompt_profile", "agent_authored") or "agent_authored"),
        "dry_run": False,
        "packet_json": str(source_packet_path),
        "source_report_md": str(source_report_path),
        "inserted_figures": inserted_figures,
        "unknown_figure_placeholders": unknown_placeholders,
    }
    quality = _ai_report_quality_checks(
        report_markdown=rendered_markdown,
        raw_report_markdown=raw_markdown,
        packet=packet,
        inserted_figures=inserted_figures,
        unknown_figure_placeholders=unknown_placeholders,
        metadata=metadata,
    )
    metadata["quality_passed"] = bool(quality.get("passed", False))
    metadata["quality_warnings"] = list(quality.get("warnings", []) or [])

    report_md_path = outdir / "master_agent_report.md"
    report_json_path = outdir / "master_agent_report.json"
    quality_path = outdir / "master_agent_report_quality.json"
    metadata_path = outdir / "master_agent_report_metadata.json"
    index_path = outdir / "master_agent_report_index.md"
    final_markdown = _prepend_agent_report_provenance(rendered_markdown, metadata=metadata, quality=quality)
    report_md_path.write_text(final_markdown, encoding="utf-8")
    artifacts = {
        "agent_report_md": str(report_md_path),
        "agent_report_json": str(report_json_path),
        "agent_report_quality_json": str(quality_path),
        "agent_report_metadata_json": str(metadata_path),
        "agent_report_index_md": str(index_path),
        "agent_report_packet_json": str(source_packet_path),
        "agent_report_source_md": str(source_report_path),
    }
    metadata["artifacts"] = artifacts
    write_json(str(quality_path), quality)
    write_json(str(metadata_path), metadata)
    write_json(
        str(report_json_path),
        {
            "report_markdown": final_markdown,
            "raw_report_markdown": raw_markdown,
            "metadata": metadata,
            "quality": quality,
        },
    )
    warnings = list(quality.get("warnings", []) or [])
    index_path.write_text(
        "\n".join(
            [
                "# Agent Report Index",
                "",
                f"- Quality: `{'passed' if quality.get('passed') else 'needs review'}`",
                f"- Author: `{_markdown_scalar(metadata.get('author'))}`",
                f"- Model: `{_markdown_scalar(metadata.get('model'))}`",
                "- Provider call made by OEL: `false`",
                f"- Final report: {_markdown_link_for_path(report_md_path, base_dir=outdir)}",
                f"- Evidence packet: {_markdown_link_for_path(source_packet_path, base_dir=outdir)}",
                f"- Quality checks: {_markdown_link_for_path(quality_path, base_dir=outdir)}",
                "",
                "## Quality Warnings",
                *(f"- {warning}" for warning in warnings),
                *(("- No quality warnings recorded.",) if not warnings else ()),
                "",
            ]
        ),
        encoding="utf-8",
    )
    result = {"metadata": metadata, "quality": quality, "artifacts": artifacts}
    if fail_on_quality and not bool(quality.get("passed", False)):
        raise RuntimeError("Agent-authored report quality checks failed; inspect master_agent_report_quality.json.")
    return result

__all__ = [name for name in globals() if not name.startswith("__")]
