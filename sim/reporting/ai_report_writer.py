# ruff: noqa: F401,F403,F405,I001
from .ai_report_models import *
from .ai_report_prompts import *
from .ai_report_evidence import *
from .ai_report_briefs import *
from .ai_report_adapters import *
from .ai_report_packets import *
from .ai_report_review import *
from .ai_report_budget import *
from .ai_report_audit import *
from .providers import *

def write_ai_report_artifacts(
    *,
    cfg: SimulationScenarioConfig,
    config_path: str | Path,
    outdir: Path,
    payload: dict[str, Any],
    payload_kind: str,
    ai_options: dict[str, Any] | None = None,
    allow_custom_endpoint: bool = False,
) -> dict[str, Any]:
    ai_cfg = _merged_ai_report_config(cfg, ai_options)
    allow_custom_endpoint = bool(
        allow_custom_endpoint or bool(dict(ai_options or {}).get("allow_custom_endpoint", False))
    )
    if not _cfg_enabled(ai_cfg):
        return payload
    if os.environ.get("OEL_SKIP_AI_REPORT", "").strip().lower() in {"1", "true", "yes"}:
        return payload

    outdir.mkdir(parents=True, exist_ok=True)
    request = build_ai_report_request(
        cfg=cfg,
        config_path=config_path,
        payload=payload,
        payload_kind=payload_kind,
        ai_cfg=ai_cfg,
    )
    prompt_profile = str(request["prompt_profile"])
    packet = dict(request["packet"])
    user_prompt = str(request["prompt"])

    input_path = outdir / "master_ai_report_input.json"
    prompt_path = outdir / "master_ai_report_prompt.md"
    metadata_path = outdir / "master_ai_report_metadata.json"
    quality_path = outdir / "master_ai_report_quality.json"
    review_packet_path = outdir / "master_ai_report_review_packet.md"
    usage_path = outdir / "master_ai_report_usage.json"
    index_path = outdir / "master_ai_report_index.md"
    report_md_path = outdir / "master_ai_report.md"
    report_json_path = outdir / "master_ai_report.json"

    write_json(str(input_path), packet)
    prompt_path.write_text(user_prompt, encoding="utf-8")
    cost_estimate = _estimate_ai_report_cost_from_request(request=request, ai_cfg=ai_cfg, payload_kind=payload_kind)
    review_packet_path.write_text(
        _build_ai_report_review_packet_markdown(
            packet=packet,
            request=request,
            ai_cfg=ai_cfg,
            input_path=input_path,
            prompt_path=prompt_path,
            cost_estimate=cost_estimate,
        ),
        encoding="utf-8",
    )

    metadata: dict[str, Any] = {
        **DIRECT_AI_REPORT_POSTURE,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "provider": str(ai_cfg.get("provider", "ollama") or "ollama"),
        "model": str(ai_cfg.get("model", "") or ""),
        "prompt_profile": prompt_profile,
        "report_mode": str(ai_cfg.get("report_mode", "") or ""),
        "data_scope": str(ai_cfg.get("data_scope", "summary_only") or "summary_only"),
        "dry_run": bool(ai_cfg.get("dry_run", False)),
        "fail_on_quality": bool(ai_cfg.get("fail_on_quality", False)),
        "include_json_appendix": bool(ai_cfg.get("include_json_appendix", False)),
        "input_json": str(input_path),
        "prompt_md": str(prompt_path),
        "review_packet_md": str(review_packet_path),
        "usage_json": str(usage_path),
        "index_md": str(index_path),
        "user_questions": list(packet.get("user_questions", []) or []),
    }
    provider_response: dict[str, Any] = {}
    report_markdown = ""
    raw_report_markdown = ""
    inserted_figures: list[str] = []
    unknown_figure_placeholders: list[str] = []

    if bool(ai_cfg.get("dry_run", False)):
        report_markdown = (
            "# AI Report Dry Run\n\n"
            "No model call was made. Review `master_ai_report_input.json` and `master_ai_report_prompt.md` before enabling live AI reports.\n"
        )
        raw_report_markdown = report_markdown
        metadata["status"] = "dry_run"
    else:
        try:
            raw_report_markdown, provider_response = _call_provider(
                ai_cfg,
                user_prompt,
                allow_custom_endpoint=allow_custom_endpoint,
            )
            report_markdown, inserted_figures, unknown_figure_placeholders = _render_figure_placeholders(
                raw_report_markdown,
                packet,
                base_dir=outdir,
            )
            metadata["status"] = "ok"
        except (OSError, urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            metadata["status"] = "error"
            metadata["error"] = f"{type(exc).__name__}: {exc}"
            if bool(ai_cfg.get("fail_on_error", False)):
                write_json(str(metadata_path), metadata)
                raise
            report_markdown = (
                "# AI Report Unavailable\n\n"
                f"The simulation completed, but AI report generation failed: {metadata['error']}\n"
            )
            raw_report_markdown = report_markdown
            payload["ai_report_error"] = metadata["error"]

    metadata["inserted_figures"] = inserted_figures
    metadata["unknown_figure_placeholders"] = unknown_figure_placeholders
    usage = _build_ai_report_usage_reconciliation(
        cost_estimate=cost_estimate,
        provider_response=provider_response,
        metadata=metadata,
    )
    metadata["usage"] = usage["actual"]["usage"]
    metadata["cost_estimate"] = usage["estimate"]["cost_estimate"]
    metadata["actual_cost"] = usage["actual"]["cost"]
    metadata["cost_reconciliation"] = usage["reconciliation"]
    write_json(str(usage_path), usage)
    quality = _ai_report_quality_checks(
        report_markdown=report_markdown,
        raw_report_markdown=raw_report_markdown,
        packet=packet,
        inserted_figures=inserted_figures,
        unknown_figure_placeholders=unknown_figure_placeholders,
        metadata=metadata,
    )
    quality_failed = (
        bool(ai_cfg.get("fail_on_quality", False))
        and metadata.get("status") == "ok"
        and not bool(quality.get("passed", False))
    )
    metadata["quality_passed"] = bool(quality.get("passed", False))
    metadata["quality_warnings"] = list(quality.get("warnings", []) or [])
    if quality_failed:
        metadata["quality_failure"] = "AI report quality checks failed and outputs.ai_report.fail_on_quality is true."
    report_markdown = _prepend_ai_report_provenance(
        report_markdown,
        metadata=metadata,
        quality=quality,
        usage=usage,
    )
    report_md_path.write_text(report_markdown, encoding="utf-8")
    metadata["report_md"] = str(report_md_path)
    metadata["report_json"] = str(report_json_path)
    artifacts = dict(payload.get("artifacts", {}) or {})
    artifacts["ai_report_md"] = str(report_md_path)
    artifacts["ai_report_json"] = str(report_json_path)
    artifacts["ai_report_input_json"] = str(input_path)
    artifacts["ai_report_prompt_md"] = str(prompt_path)
    artifacts["ai_report_metadata_json"] = str(metadata_path)
    artifacts["ai_report_quality_json"] = str(quality_path)
    artifacts["ai_report_review_packet_md"] = str(review_packet_path)
    artifacts["ai_report_usage_json"] = str(usage_path)
    artifacts["ai_report_index_md"] = str(index_path)
    index_path.write_text(
        _build_ai_report_index_markdown(
            metadata=metadata,
            quality=quality,
            usage=usage,
            artifacts=artifacts,
            base_dir=outdir,
        ),
        encoding="utf-8",
    )
    write_json(str(quality_path), quality)
    write_json(
        str(report_json_path),
        {
            "report_markdown": report_markdown,
            "raw_report_markdown": raw_report_markdown,
            "metadata": metadata,
            "quality": quality,
            "usage": usage,
            "provider_response": provider_response,
        },
    )
    write_json(str(metadata_path), metadata)
    payload["artifacts"] = artifacts
    if quality_failed:
        payload["ai_report_quality_error"] = metadata["quality_failure"]
        raise RuntimeError(metadata["quality_failure"])
    return payload
