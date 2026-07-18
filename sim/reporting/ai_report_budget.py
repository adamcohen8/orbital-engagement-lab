# ruff: noqa: F401,F403,F405,I001
from .ai_report_models import *
from .ai_report_review import *
from .ai_report_packets import *

def _approx_token_count(text: str, *, chars_per_token: float = 4.0) -> int:
    chars_per_token = max(float(chars_per_token or 4.0), 0.1)
    return max(1, int(round(float(len(text)) / chars_per_token)))


def _configured_output_token_estimate(ai_cfg: dict[str, Any]) -> int:
    for key in ("estimated_output_tokens", "output_token_estimate", "max_output_tokens", "max_tokens"):
        value = ai_cfg.get(key)
        if value is not None:
            return max(1, int(value))
    options = _provider_options(ai_cfg)
    for key in ("max_output_tokens", "maxOutputTokens", "max_tokens", "maxTokens"):
        value = options.get(key)
        if value is not None:
            return max(1, int(value))
    generation_config = _gemini_generation_config(ai_cfg)
    value = generation_config.get("maxOutputTokens")
    if value is not None:
        return max(1, int(value))
    return 2048


def _pricing_from_config(ai_cfg: dict[str, Any]) -> tuple[float | None, float | None, str, str]:
    pricing = ai_cfg.get("pricing", {})
    if isinstance(pricing, dict):
        input_price = pricing.get("input_per_1m_tokens", pricing.get("input_per_million_tokens"))
        output_price = pricing.get("output_per_1m_tokens", pricing.get("output_per_million_tokens"))
        if input_price is not None and output_price is not None:
            return float(input_price), float(output_price), str(pricing.get("currency", "USD") or "USD"), "config"

    provider = str(ai_cfg.get("provider", "ollama") or "ollama").strip().lower()
    model = str(ai_cfg.get("model", "") or "").strip().lower()
    key = f"{provider}/{model}"
    for candidate, prices in sorted(
        DEFAULT_AI_PRICE_PER_1M_TOKENS.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if key.startswith(candidate):
            return float(prices["input"]), float(prices["output"]), "USD", "built_in"
    return None, None, "USD", "unknown"


def estimate_ai_report_cost(
    *,
    cfg: SimulationScenarioConfig,
    config_path: str | Path,
    payload: dict[str, Any],
    payload_kind: str,
    ai_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ai_cfg = _merged_ai_report_config(cfg, ai_options)
    request = build_ai_report_request(
        cfg=cfg,
        config_path=config_path,
        payload=payload,
        payload_kind=payload_kind,
        ai_cfg=ai_cfg,
    )
    return _estimate_ai_report_cost_from_request(request=request, ai_cfg=ai_cfg, payload_kind=payload_kind)


def write_ai_report_estimate_artifacts(
    *,
    cfg: SimulationScenarioConfig,
    config_path: str | Path,
    outdir: Path,
    payload: dict[str, Any],
    payload_kind: str,
    ai_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    ai_cfg = _merged_ai_report_config(cfg, ai_options)
    request = build_ai_report_request(
        cfg=cfg,
        config_path=config_path,
        payload=payload,
        payload_kind=payload_kind,
        ai_cfg=ai_cfg,
    )
    estimate = _estimate_ai_report_cost_from_request(request=request, ai_cfg=ai_cfg, payload_kind=payload_kind)
    input_path = outdir / "master_ai_report_input.json"
    prompt_path = outdir / "master_ai_report_prompt.md"
    estimate_path = outdir / "master_ai_report_cost_estimate.json"
    review_packet_path = outdir / "master_ai_report_review_packet.md"
    write_json(str(input_path), request["packet"])
    prompt_path.write_text(str(request["prompt"]), encoding="utf-8")
    write_json(str(estimate_path), estimate)
    review_packet_path.write_text(
        _build_ai_report_review_packet_markdown(
            packet=dict(request["packet"]),
            request=request,
            ai_cfg=ai_cfg,
            input_path=input_path,
            prompt_path=prompt_path,
            cost_estimate=estimate,
        ),
        encoding="utf-8",
    )
    estimate["artifacts"] = {
        "ai_report_input_json": str(input_path),
        "ai_report_prompt_md": str(prompt_path),
        "ai_report_cost_estimate_json": str(estimate_path),
        "ai_report_review_packet_md": str(review_packet_path),
    }
    return estimate


def write_agent_report_packet_artifacts(
    *,
    cfg: SimulationScenarioConfig,
    config_path: str | Path,
    outdir: Path,
    payload: dict[str, Any],
    payload_kind: str,
    report_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write provider-neutral evidence and authoring instructions for a coding agent."""

    outdir.mkdir(parents=True, exist_ok=True)
    options = _merged_ai_report_config(cfg, report_options)
    options["include_json_appendix"] = False
    request = build_ai_report_request(
        cfg=cfg,
        config_path=config_path,
        payload=payload,
        payload_kind=payload_kind,
        ai_cfg=options,
    )
    packet = dict(request["packet"])
    legacy_options = dict(packet.pop("ai_report_options", {}) or {})
    for key in ("provider", "model", "dry_run"):
        legacy_options.pop(key, None)
    config_summary = dict(packet.get("config_summary", {}) or {})
    output_summary = dict(config_summary.get("outputs", {}) or {})
    summarized_report_options = dict(output_summary.get("ai_report", {}) or {})
    for key in (
        "provider",
        "model",
        "endpoint",
        "api_key_env",
        "pricing",
        "options",
        "generation_config",
    ):
        summarized_report_options.pop(key, None)
    output_summary["ai_report"] = summarized_report_options
    config_summary["outputs"] = output_summary
    packet["config_summary"] = config_summary
    if isinstance(packet.get("config"), dict):
        full_config = deepcopy(dict(packet["config"]))
        full_outputs = dict(full_config.get("outputs", {}) or {})
        full_ai_report = dict(full_outputs.get("ai_report", {}) or {})
        for key in (
            "provider",
            "model",
            "endpoint",
            "api_key_env",
            "pricing",
            "options",
            "generation_config",
        ):
            full_ai_report.pop(key, None)
        full_outputs["ai_report"] = full_ai_report
        full_config["outputs"] = full_outputs
        packet["config"] = full_config
    packet.update(
        {
            "packet_schema_version": "oel.agent_report_packet.v1",
            "authoring_mode": "external_agent",
            "report_options": legacy_options,
        }
    )

    packet_path = outdir / "agent_report_packet.json"
    brief_path = outdir / "agent_report_brief.md"
    manifest_path = outdir / "agent_report_manifest.json"
    brief = "\n".join(
        [
            "# OEL Agent Report Brief",
            "",
            "This packet contains deterministic OEL evidence. The coding agent owns synthesis and prose; OEL remains "
            "the authority for simulation outputs, report rules, figure references, and the final audit.",
            "",
            "Save the authored Markdown separately, then run OEL's report audit with this packet. Do not execute a "
            "provider call from OEL for this workflow.",
            "",
            str(request["prompt"]).strip(),
            "",
        ]
    )
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "workflow": "agent_authored_report",
        "authoring_mode": "external_agent",
        "packet_schema_version": packet["packet_schema_version"],
        "payload_kind": payload_kind,
        "prompt_profile": request["prompt_profile"],
        "provider_call_made": False,
        "packet_json": str(packet_path),
        "brief_md": str(brief_path),
        "next_step": "Author a Markdown report from the brief and packet, then run the OEL report audit.",
    }
    write_json(str(packet_path), packet)
    brief_path.write_text(brief, encoding="utf-8")
    write_json(str(manifest_path), manifest)
    artifacts = {
        "agent_report_packet_json": str(packet_path),
        "agent_report_brief_md": str(brief_path),
        "agent_report_manifest_json": str(manifest_path),
    }
    return {"packet": packet, "manifest": manifest, "artifacts": artifacts}

__all__ = [name for name in globals() if not name.startswith("__")]
