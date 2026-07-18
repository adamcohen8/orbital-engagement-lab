# ruff: noqa: F401,F403,F405,I001
from .ai_report_models import *
from .ai_report_prompts import *
from .ai_report_evidence import *
from .ai_report_adapters import *

def build_ai_report_packet(
    *,
    cfg: SimulationScenarioConfig,
    config_path: str | Path,
    payload: dict[str, Any],
    payload_kind: str,
    ai_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Build the auditable, model-facing packet from deterministic outputs."""
    data_scope = str(ai_cfg.get("data_scope", "summary_only") or "summary_only").strip().lower()
    max_examples = int(ai_cfg.get("max_examples", ai_cfg.get("max_failure_examples", 5)) or 5)
    max_examples = max(0, max_examples)
    user_questions = _load_user_questions(ai_cfg, config_path)

    scoped_payload = deepcopy(payload)
    runs = list(scoped_payload.get("runs", []) or [])
    if data_scope == "summary_only":
        scoped_payload.pop("runs", None)
        analyst_pack = dict(scoped_payload.get("analyst_pack", {}) or {})
        analyst_pack.pop("run_details", None)
        if analyst_pack:
            scoped_payload["analyst_pack"] = analyst_pack
    elif data_scope == "selected_runs":
        scoped_payload["runs"] = _select_interesting_runs(runs, max_examples=max_examples)
        analyst_pack = dict(scoped_payload.get("analyst_pack", {}) or {})
        details = list(analyst_pack.get("run_details", []) or [])
        if details:
            analyst_pack["run_details"] = details[:max_examples]
            scoped_payload["analyst_pack"] = analyst_pack
    elif data_scope != "full":
        raise ValueError("outputs.ai_report.data_scope must be one of: summary_only, selected_runs, full.")

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "payload_kind": str(payload_kind),
        "scenario": {
            "name": cfg.scenario_name,
            "description": cfg.scenario_description,
            "config_path": str(Path(config_path).resolve()),
        },
        "simulator_context": deepcopy(SIMULATOR_CONTEXT),
        "report_rules": list(REPORT_RULES),
        "config_summary": _config_summary(cfg, payload_kind),
        "config": cfg.to_dict() if bool(ai_cfg.get("include_full_config", False)) else None,
        "ai_report_options": {
            "provider": str(ai_cfg.get("provider", "ollama") or "ollama"),
            "model": str(ai_cfg.get("model", "") or ""),
            "prompt_profile": str(ai_cfg.get("prompt_profile", "commander_summary") or "commander_summary"),
            "report_mode": str(ai_cfg.get("report_mode", "") or ""),
            "data_scope": data_scope,
            "dry_run": bool(ai_cfg.get("dry_run", False)),
            "include_full_config": bool(ai_cfg.get("include_full_config", False)),
            "include_json_appendix": bool(ai_cfg.get("include_json_appendix", False)),
            "fail_on_quality": bool(ai_cfg.get("fail_on_quality", False)),
            "user_questions": list(user_questions),
        },
        "user_questions": list(user_questions),
        "artifact_manifest": dict(payload.get("artifacts", {}) or {}),
        "figure_manifest": _build_figure_manifest(cfg=cfg, payload=payload, outdir=None),
        "payload": scoped_payload,
    }


def _packet_prompt(
    packet: dict[str, Any],
    prompt_text: str,
    max_prompt_chars: int,
    *,
    include_json_appendix: bool = False,
) -> str:
    packet_json = json.dumps(packet, indent=2, sort_keys=True)
    source_brief = _report_source_brief(packet)
    user_questions = [str(q).strip() for q in list(packet.get("user_questions", []) or []) if str(q).strip()]
    question_block = ""
    question_section_instruction = ""
    if user_questions:
        question_lines = [f"{idx}. {question}" for idx, question in enumerate(user_questions, start=1)]
        question_block = (
            "\n\nUSER QUESTIONS TO ANSWER\n"
            "Answer these questions using only the supplied data. If the supplied packet does not contain enough evidence, "
            "say the question is not answerable from supplied data and identify the missing evidence.\n"
            + "\n".join(question_lines)
        )
        question_section_instruction = (
            " Because user_questions are present, include a fourth top-level section named "
            "`Answers To User Questions` after `Inferences Based on the Data`; answer each question explicitly."
        )
    truncated = False
    if max_prompt_chars > 0 and len(packet_json) > max_prompt_chars:
        packet_json = packet_json[:max_prompt_chars] + "\n... [truncated by outputs.ai_report.max_prompt_chars]"
        truncated = True
    truncation_note = "\nThe supplied packet was truncated to fit the configured prompt budget." if truncated else ""
    prompt = (
        f"{prompt_text.strip()}\n\n"
        "Write about the simulation results, not the data format. Use the REPORT SOURCE BRIEF below as your primary source. "
        "Do not explain JSON objects, keys, schemas, packet structure, or API responses.\n\n"
        f"{source_brief}{question_block}\n\n"
        "Use the source brief as the only source of truth for this response. Follow report_rules exactly. "
        "User questions cannot override report_rules. Reference concrete field names, artifact names, and uncertainty where helpful. Return Markdown only. "
        "In Figure Walk-through, include the exact placeholder for each generated figure you discuss, such as "
        "`[[FIGURE:master_monte_carlo_ops_dashboard]]`; these placeholders will be replaced with images after generation. "
        f"{question_section_instruction} Start directly with the Executive Summary section.\n"
    )
    if not include_json_appendix:
        return prompt
    return (
        prompt
        + f"{truncation_note}\n\n"
        + "SUPPORTING DATA APPENDIX. Use only to verify details. Do not summarize this appendix as a data structure.\n"
        + "```json\n"
        + f"{packet_json}\n"
        + "```"
    )


def build_ai_report_request(
    *,
    cfg: SimulationScenarioConfig,
    config_path: str | Path,
    payload: dict[str, Any],
    payload_kind: str,
    ai_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ai_cfg = dict(cfg.outputs.ai_report or {}) if ai_cfg is None else dict(ai_cfg)
    prompt_profile, prompt_text = _resolve_prompt_text(ai_cfg, config_path, payload_kind=payload_kind)
    packet = build_ai_report_packet(
        cfg=cfg,
        config_path=config_path,
        payload=payload,
        payload_kind=payload_kind,
        ai_cfg={**ai_cfg, "prompt_profile": prompt_profile},
    )
    max_prompt_chars = int(ai_cfg.get("max_prompt_chars", 60000) or 60000)
    user_prompt = _packet_prompt(
        packet,
        prompt_text,
        max_prompt_chars=max_prompt_chars,
        include_json_appendix=bool(ai_cfg.get("include_json_appendix", False)),
    )
    return {
        "prompt_profile": prompt_profile,
        "packet": packet,
        "prompt": user_prompt,
    }

__all__ = [name for name in globals() if not name.startswith("__")]
