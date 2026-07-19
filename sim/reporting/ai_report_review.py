# ruff: noqa: F401,F403,F405,I001
from .ai_report_models import *
from .ai_report_evidence import *
from .ai_report_briefs import _fmt_value, _percent

def _markdown_scalar(value: Any) -> str:
    if value is None:
        return "not available"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return _fmt_value(value)
    text = str(value).strip()
    return text if text else "not available"


def _markdown_artifact_lines(artifacts: dict[str, Any], *, limit: int = 30) -> list[str]:
    lines: list[str] = []
    for key, value in sorted(artifacts.items())[:limit]:
        lines.append(f"- `{key}`: `{value}`")
    if len(artifacts) > limit:
        lines.append(f"- ... {len(artifacts) - limit} additional artifacts omitted from this review packet")
    if not lines:
        lines.append("- No artifacts were listed in the source payload.")
    return lines


def _review_packet_result_lines(packet: dict[str, Any]) -> list[str]:
    payload_kind = str(packet.get("payload_kind", "") or "").strip().lower()
    payload = dict(packet.get("payload", {}) or {})
    if payload_kind == "sensitivity":
        analysis = dict(payload.get("analysis", {}) or {})
        rankings = [dict(row or {}) for row in list(payload.get("parameter_rankings", []) or [])]
        interactions = [dict(row or {}) for row in list(payload.get("interaction_summaries", []) or [])]
        lines = [
            f"- Method: {_markdown_scalar(analysis.get('method'))}",
            f"- Metrics: {_markdown_scalar(list(analysis.get('metrics', []) or []))}",
            f"- Total runs: {_markdown_scalar(analysis.get('run_count', len(payload.get('runs', []) or [])))}",
            f"- Successful runs: {_markdown_scalar(analysis.get('successful_run_count'))}",
            f"- Failed runs: {_markdown_scalar(analysis.get('failed_run_count'))}",
            f"- Failure policy: {_markdown_scalar(analysis.get('failure_policy', 'fail_fast'))}",
        ]
        for row in rankings[:5]:
            lines.append(f"- Top driver `{row.get('parameter_path', 'unknown')}`: {row}")
        if not rankings:
            lines.append("- No parameter ranking rows are in scope.")
        for row in interactions[:3]:
            lines.append(f"- Interaction `{row.get('interaction', 'unknown')}`: {row}")
        if not interactions:
            lines.append("- No two-parameter interaction summaries are in scope.")
        return lines

    aggregate = dict(payload.get("aggregate_stats", {}) or {})
    commander = dict(payload.get("commander_brief", {}) or {})
    lines = [
        f"- Pass rate: {_percent(aggregate.get('pass_rate', commander.get('p_success')))}",
        f"- Fail rate: {_percent(aggregate.get('fail_rate', commander.get('p_fail')))}",
        f"- Closest approach mean: {_fmt_value(aggregate.get('closest_approach_km_mean'), 'km')}",
        f"- Worst-case closest approach: {_fmt_value(commander.get('worst_case_closest_approach_km'), 'km')}",
        f"- Keepout violation probability: {_percent(aggregate.get('p_keepout_violation', commander.get('p_keepout_violation')))}",
    ]
    drivers = [dict(row or {}) for row in list(commander.get("top_parameter_drivers", []) or [])]
    for row in drivers[:5]:
        lines.append(f"- Parameter driver `{row.get('parameter_path', 'unknown')}`: {row}")
    if not drivers:
        lines.append("- No commander parameter-driver rows are in scope.")
    return lines


def _review_packet_figure_lines(packet: dict[str, Any]) -> list[str]:
    manifest = dict(packet.get("figure_manifest", {}) or {})
    generated = [dict(row or {}) for row in list(manifest.get("generated_artifacts", []) or [])]
    requested = [dict(row or {}) for row in list(manifest.get("requested_figures", []) or [])]
    lines: list[str] = [
        f"- Image pixels available to the model: {_markdown_scalar(manifest.get('image_pixels_available'))}",
        f"- Figure data available: {_markdown_scalar(manifest.get('figure_data_available'))}",
    ]
    for row in generated:
        lines.append(
            f"- Generated `{row.get('figure_id', 'unknown')}`: placeholder `{row.get('placeholder', '')}`, "
            f"path `{row.get('path', '')}`"
        )
    for row in requested:
        figure_id = str(row.get("figure_id", "unknown") or "unknown")
        if not any(str(g.get("figure_id", "")) == figure_id for g in generated):
            lines.append(f"- Requested `{figure_id}`: {row.get('description', 'No description available.')}")
    if len(lines) == 2:
        lines.append("- No requested or generated figures are in scope.")
    return lines


def _build_ai_report_review_packet_markdown(
    *,
    packet: dict[str, Any],
    request: dict[str, Any],
    ai_cfg: dict[str, Any],
    input_path: Path,
    prompt_path: Path,
    cost_estimate: dict[str, Any] | None = None,
) -> str:
    scenario = dict(packet.get("scenario", {}) or {})
    options = dict(packet.get("ai_report_options", {}) or {})
    artifacts = dict(packet.get("artifact_manifest", {}) or {})
    user_questions = [str(q).strip() for q in list(packet.get("user_questions", []) or []) if str(q).strip()]
    question_lines = (
        [f"- {question}" for question in user_questions] if user_questions else ["- No user questions were provided."]
    )
    cost = dict(cost_estimate.get("cost_estimate", {}) if isinstance(cost_estimate, dict) else {})
    tokens = dict(cost_estimate.get("token_estimate", {}) if isinstance(cost_estimate, dict) else {})
    pricing = dict(cost_estimate.get("pricing", {}) if isinstance(cost_estimate, dict) else {})
    total_cost = cost.get("total")
    total_cost_text = "not estimated"
    if isinstance(total_cost, (int, float)):
        total_cost_text = f"{float(total_cost):.6f} {pricing.get('currency', 'USD')}"
    elif cost_estimate is not None:
        total_cost_text = "not available for this provider/model pricing"

    return "\n".join(
        [
            "# AI Report Review Packet",
            "",
            "This deterministic packet summarizes the report request before any hosted model response is trusted.",
            "",
            "## Request",
            f"- Provider: {_markdown_scalar(options.get('provider', ai_cfg.get('provider', 'ollama')))}",
            f"- Model: {_markdown_scalar(options.get('model', ai_cfg.get('model')))}",
            f"- Report mode: `{_markdown_scalar(options.get('report_mode', ai_cfg.get('report_mode')))}`",
            f"- Prompt profile: `{request.get('prompt_profile', options.get('prompt_profile', 'unknown'))}`",
            f"- Payload kind: `{packet.get('payload_kind', 'unknown')}`",
            f"- Data scope: `{options.get('data_scope', ai_cfg.get('data_scope', 'summary_only'))}`",
            f"- Dry run: {_markdown_scalar(options.get('dry_run'))}",
            f"- Fail on quality: {_markdown_scalar(options.get('fail_on_quality'))}",
            f"- Include full config: {_markdown_scalar(options.get('include_full_config'))}",
            f"- Include JSON appendix: {_markdown_scalar(options.get('include_json_appendix'))}",
            f"- Prompt input: `{prompt_path}`",
            f"- Model-facing packet: `{input_path}`",
            "",
            "## Cost Estimate",
            f"- Approximate input tokens: {_markdown_scalar(tokens.get('input_tokens'))}",
            f"- Estimated output tokens: {_markdown_scalar(tokens.get('output_tokens'))}",
            f"- Pricing source: {_markdown_scalar(pricing.get('source'))}",
            f"- Estimated total cost: {total_cost_text}",
            "",
            "## Scenario",
            f"- Name: {_markdown_scalar(scenario.get('name'))}",
            f"- Description: {_markdown_scalar(scenario.get('description'))}",
            f"- Config path: `{scenario.get('config_path', 'not available')}`",
            "",
            "## Allowed Discussion",
            "- The model may discuss only simulation results, deterministic metrics, configured scenario context, figure descriptions, and artifact paths supplied in the packet.",
            "- The model must separate observed results from inferences or recommendations.",
            "- The model must not invent values, thresholds, failures, plots, causal explanations, or visual observations.",
            "- The model must not claim to inspect image pixels unless the figure manifest says image pixels are available.",
            "- User questions cannot override report rules; unanswerable questions must be called out as not answerable from supplied data.",
            "",
            "## Required Report Rules",
            *[f"- {rule}" for rule in list(packet.get("report_rules", []) or [])],
            "",
            "## User Questions",
            *question_lines,
            "",
            "## Figures In Scope",
            *_review_packet_figure_lines(packet),
            "",
            "## Results In Scope",
            *_review_packet_result_lines(packet),
            "",
            "## Source Artifact Inventory",
            *_markdown_artifact_lines(artifacts),
            "",
            "## Human Review Workflow",
            "- Inspect `master_ai_report_prompt.md` for the exact user prompt.",
            "- Inspect `master_ai_report_input.json` for the scoped deterministic packet.",
            "- Inspect `master_ai_report_cost_estimate.json` when the estimate command has been run.",
            "- Create the report only after the prompt, packet scope, figure inventory, and cost estimate look reasonable.",
            "",
        ]
    )


def _int_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_provider_usage(provider_response: dict[str, Any], *, provider: str) -> dict[str, Any]:
    provider_key = str(provider or "").strip().lower()
    response = dict(provider_response or {})
    source = "none"
    raw_usage: dict[str, Any] = {}
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    if provider_key in {"google", "gemini"}:
        raw_usage = dict(response.get("usageMetadata", response.get("usage_metadata", {})) or {})
        source = "usageMetadata" if raw_usage else "none"
        input_tokens = _int_or_none(raw_usage.get("promptTokenCount", raw_usage.get("prompt_token_count")))
        output_tokens = _int_or_none(raw_usage.get("candidatesTokenCount", raw_usage.get("candidates_token_count")))
        total_tokens = _int_or_none(raw_usage.get("totalTokenCount", raw_usage.get("total_token_count")))
    elif provider_key == "openai":
        raw_usage = dict(response.get("usage", {}) or {})
        source = "usage" if raw_usage else "none"
        input_tokens = _int_or_none(raw_usage.get("input_tokens", raw_usage.get("prompt_tokens")))
        output_tokens = _int_or_none(raw_usage.get("output_tokens", raw_usage.get("completion_tokens")))
        total_tokens = _int_or_none(raw_usage.get("total_tokens"))
    elif provider_key in {"anthropic", "claude"}:
        raw_usage = dict(response.get("usage", {}) or {})
        source = "usage" if raw_usage else "none"
        input_tokens = _int_or_none(raw_usage.get("input_tokens"))
        output_tokens = _int_or_none(raw_usage.get("output_tokens"))
    elif provider_key == "ollama":
        raw_usage = {
            key: response.get(key)
            for key in ("prompt_eval_count", "eval_count", "prompt_eval_duration", "eval_duration", "total_duration")
            if key in response
        }
        source = "ollama_chat_response" if raw_usage else "none"
        input_tokens = _int_or_none(response.get("prompt_eval_count"))
        output_tokens = _int_or_none(response.get("eval_count"))

    if total_tokens is None:
        if input_tokens is not None and output_tokens is not None:
            total_tokens = int(input_tokens + output_tokens)
        elif input_tokens is not None:
            total_tokens = int(input_tokens)
        elif output_tokens is not None:
            total_tokens = int(output_tokens)

    return {
        "available": any(value is not None for value in (input_tokens, output_tokens, total_tokens)),
        "source": source,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "raw_usage": raw_usage,
    }


def _cost_from_tokens(
    *,
    input_tokens: int | None,
    output_tokens: int | None,
    pricing: dict[str, Any],
) -> dict[str, Any]:
    input_price = pricing.get("input_per_1m_tokens")
    output_price = pricing.get("output_per_1m_tokens")
    input_cost = None
    output_cost = None
    if input_tokens is not None and input_price is not None:
        input_cost = float(input_tokens) * float(input_price) / 1_000_000.0
    if output_tokens is not None and output_price is not None:
        output_cost = float(output_tokens) * float(output_price) / 1_000_000.0
    total = None if input_cost is None or output_cost is None else float(input_cost + output_cost)
    return {"input": input_cost, "output": output_cost, "total": total}


def _build_ai_report_usage_reconciliation(
    *,
    cost_estimate: dict[str, Any],
    provider_response: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    pricing = dict(cost_estimate.get("pricing", {}) or {})
    estimate_tokens = dict(cost_estimate.get("token_estimate", {}) or {})
    estimate_cost = dict(cost_estimate.get("cost_estimate", {}) or {})
    actual_usage = _extract_provider_usage(provider_response, provider=str(metadata.get("provider", "")))
    actual_cost = _cost_from_tokens(
        input_tokens=actual_usage.get("input_tokens"),
        output_tokens=actual_usage.get("output_tokens"),
        pricing=pricing,
    )
    estimated_input = _int_or_none(estimate_tokens.get("input_tokens"))
    estimated_output = _int_or_none(estimate_tokens.get("output_tokens"))
    actual_input = _int_or_none(actual_usage.get("input_tokens"))
    actual_output = _int_or_none(actual_usage.get("output_tokens"))
    estimated_total_cost = estimate_cost.get("total")
    actual_total_cost = actual_cost.get("total")
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": metadata.get("status"),
        "provider": metadata.get("provider"),
        "model": metadata.get("model"),
        "prompt_profile": metadata.get("prompt_profile"),
        "estimate": {
            "token_estimate": estimate_tokens,
            "pricing": pricing,
            "cost_estimate": estimate_cost,
        },
        "actual": {
            "usage": actual_usage,
            "cost": actual_cost,
        },
        "reconciliation": {
            "usage_available": bool(actual_usage.get("available")),
            "input_token_delta": None
            if estimated_input is None or actual_input is None
            else actual_input - estimated_input,
            "output_token_delta": None
            if estimated_output is None or actual_output is None
            else actual_output - estimated_output,
            "total_cost_delta": (
                None
                if not isinstance(estimated_total_cost, (int, float)) or not isinstance(actual_total_cost, (int, float))
                else float(actual_total_cost) - float(estimated_total_cost)
            ),
            "note": (
                "Provider-reported usage was available and reconciled against the pre-call estimate."
                if actual_usage.get("available")
                else "Provider-reported usage was not available for this response; retain the estimate as the cost-control record."
            ),
        },
    }


def _ai_report_provenance_markdown(
    *,
    metadata: dict[str, Any],
    quality: dict[str, Any],
    usage: dict[str, Any],
) -> str:
    actual_usage = dict(dict(usage.get("actual", {}) or {}).get("usage", {}) or {})
    reconciliation = dict(usage.get("reconciliation", {}) or {})
    quality_status = "passed" if bool(quality.get("passed")) else "needs review"
    usage_status = "available" if bool(actual_usage.get("available")) else "not available"
    total_tokens = actual_usage.get("total_tokens")
    total_tokens_text = _markdown_scalar(total_tokens)
    return "\n".join(
        [
            "<!-- OEL AI report provenance -->",
            "> **AI report provenance:** "
            f"provider `{_markdown_scalar(metadata.get('provider'))}`, "
            f"model `{_markdown_scalar(metadata.get('model'))}`, "
            f"prompt profile `{_markdown_scalar(metadata.get('prompt_profile'))}`, "
            f"status `{_markdown_scalar(metadata.get('status'))}`, "
            f"quality `{quality_status}`, "
            f"usage `{usage_status}`",
            ">",
            f"> Generated UTC: `{_markdown_scalar(metadata.get('generated_utc'))}`  ",
            f"> Input packet: `{_markdown_scalar(metadata.get('input_json'))}`  ",
            f"> Prompt: `{_markdown_scalar(metadata.get('prompt_md'))}`  ",
            f"> Review packet: `{_markdown_scalar(metadata.get('review_packet_md'))}`  ",
            f"> Usage tokens: `{total_tokens_text}`  ",
            f"> Cost reconciliation: `{_markdown_scalar(reconciliation.get('note'))}`",
            "",
        ]
    )


def _prepend_ai_report_provenance(
    report_markdown: str,
    *,
    metadata: dict[str, Any],
    quality: dict[str, Any],
    usage: dict[str, Any],
) -> str:
    if report_markdown.lstrip().startswith("<!-- OEL AI report provenance -->"):
        return report_markdown
    return _ai_report_provenance_markdown(metadata=metadata, quality=quality, usage=usage) + report_markdown.lstrip()


def _markdown_link_for_path(path_text: Any, *, base_dir: Path) -> str:
    text = str(path_text or "").strip()
    if not text:
        return "not available"
    try:
        path = Path(text)
        resolved = path if path.is_absolute() else Path.cwd() / path
        display = str(resolved.resolve().relative_to(base_dir.resolve()))
    except ValueError:
        try:
            path = Path(text)
            resolved = path if path.is_absolute() else Path.cwd() / path
            display = str(Path(os.path.relpath(str(resolved.resolve()), str(base_dir.resolve()))))
        except Exception:
            display = text
    except Exception:
        display = text
    href = f"<{display}>" if any(ch.isspace() for ch in display) else display
    return f"[`{display}`]({href})"


def _fmt_cost(value: Any, currency: str = "USD") -> str:
    if not isinstance(value, (int, float)):
        return "not available"
    return f"{float(value):.6f} {currency}"


def _build_ai_report_index_markdown(
    *,
    metadata: dict[str, Any],
    quality: dict[str, Any],
    usage: dict[str, Any],
    artifacts: dict[str, Any],
    base_dir: Path,
) -> str:
    warnings = list(quality.get("warnings", []) or [])
    reconciliation = dict(usage.get("reconciliation", {}) or {})
    estimate = dict(usage.get("estimate", {}) or {})
    pricing = dict(estimate.get("pricing", {}) or {})
    estimated_cost = dict(estimate.get("cost_estimate", {}) or {})
    actual = dict(usage.get("actual", {}) or {})
    actual_usage = dict(actual.get("usage", {}) or {})
    actual_cost = dict(actual.get("cost", {}) or {})
    currency = str(pricing.get("currency", "USD") or "USD")
    quality_status = "passed" if bool(quality.get("passed")) else "needs review"
    usage_status = "available" if bool(actual_usage.get("available")) else "not available"
    warning_lines = [f"- {warning}" for warning in warnings] if warnings else ["- No quality warnings recorded."]
    question_count = len(list(metadata.get("user_questions", []) or []))
    artifact_order = [
        ("Final report", "ai_report_md"),
        ("Review packet", "ai_report_review_packet_md"),
        ("Prompt", "ai_report_prompt_md"),
        ("Input packet", "ai_report_input_json"),
        ("Quality checks", "ai_report_quality_json"),
        ("Usage reconciliation", "ai_report_usage_json"),
        ("Metadata", "ai_report_metadata_json"),
        ("Report JSON", "ai_report_json"),
    ]
    artifact_lines = [
        f"- {label}: {_markdown_link_for_path(artifacts.get(key), base_dir=base_dir)}" for label, key in artifact_order
    ]
    return "\n".join(
        [
            "# AI Report Index",
            "",
            "Start here for this AI report output directory.",
            "",
            "## Status",
            f"- Report status: `{_markdown_scalar(metadata.get('status'))}`",
            f"- Provider: `{_markdown_scalar(metadata.get('provider'))}`",
            f"- Model: `{_markdown_scalar(metadata.get('model'))}`",
            f"- Prompt profile: `{_markdown_scalar(metadata.get('prompt_profile'))}`",
            f"- Data scope: `{_markdown_scalar(metadata.get('data_scope'))}`",
            f"- User questions: `{question_count}`",
            f"- Generated UTC: `{_markdown_scalar(metadata.get('generated_utc'))}`",
            "",
            "## Quality",
            f"- Quality status: `{quality_status}`",
            f"- Inserted figures: `{_markdown_scalar(list(quality.get('inserted_figures', []) or []))}`",
            f"- Unknown figure placeholders: `{_markdown_scalar(list(quality.get('unknown_figure_placeholders', []) or []))}`",
            f"- Schema-like terms: `{_markdown_scalar(list(quality.get('schemaish_terms', []) or []))}`",
            "",
            "Warnings:",
            *warning_lines,
            "",
            "## Usage And Cost",
            f"- Provider usage: `{usage_status}`",
            f"- Actual total tokens: `{_markdown_scalar(actual_usage.get('total_tokens'))}`",
            f"- Estimated total cost: `{_fmt_cost(estimated_cost.get('total'), currency)}`",
            f"- Actual total cost: `{_fmt_cost(actual_cost.get('total'), currency)}`",
            f"- Cost delta: `{_fmt_cost(reconciliation.get('total_cost_delta'), currency)}`",
            f"- Reconciliation note: {_markdown_scalar(reconciliation.get('note'))}",
            "",
            "## Artifacts",
            *artifact_lines,
            "",
            "## Suggested Review Order",
            "1. Open the final report.",
            "2. Check quality warnings and unresolved figure placeholders.",
            "3. Review usage and cost reconciliation.",
            "4. Inspect the review packet, prompt, and input packet if the report will be shared externally.",
            "",
        ]
    )

__all__ = [name for name in globals() if not name.startswith("__")]
