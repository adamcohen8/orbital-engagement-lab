# ruff: noqa: F401,F403,F405,I001
from .ai_report_models import *

def _cfg_enabled(ai_cfg: dict[str, Any]) -> bool:
    return bool(ai_cfg.get("enabled", False))


def _default_prompt_profile(payload_kind: str) -> str:
    from .ai_report_adapters import _adapter_for_payload_kind

    return _adapter_for_payload_kind(payload_kind).default_prompt_profile


def _prompt_profile_from_report_mode(ai_cfg: dict[str, Any], *, payload_kind: str) -> str:
    mode = str(ai_cfg.get("report_mode", "") or "").strip().lower()
    if not mode:
        return _default_prompt_profile(payload_kind)
    if mode not in REPORT_MODE_PROMPT_PROFILES:
        valid = ", ".join(sorted(REPORT_MODE_PROMPT_PROFILES.keys()))
        raise ValueError(f"Unknown AI report report_mode '{mode}'. Valid modes: {valid}.")
    by_kind = REPORT_MODE_PROMPT_PROFILES[mode]
    kind = str(payload_kind or "").strip().lower()
    profile = by_kind.get(kind, by_kind.get("default", ""))
    if not profile:
        return _default_prompt_profile(payload_kind)
    return profile


def _resolve_prompt_text(
    ai_cfg: dict[str, Any], config_path: str | Path, *, payload_kind: str = "monte_carlo"
) -> tuple[str, str]:
    prompt_file = str(ai_cfg.get("prompt_file", "") or "").strip()
    if prompt_file:
        path = ConfigPathPolicy.default(
            config_path=config_path,
            allow_external_ai_prompt_files=bool(ai_cfg.get("allow_external_ai_prompt_files", False)),
        ).resolve_ai_prompt_file(prompt_file, purpose="outputs.ai_report.prompt_file")
        return "custom_file", path.read_text(encoding="utf-8")

    default_profile = _prompt_profile_from_report_mode(ai_cfg, payload_kind=payload_kind)
    profile = str(ai_cfg.get("prompt_profile", default_profile) or default_profile).strip()
    if profile not in DEFAULT_PROMPT_PROFILES:
        valid = ", ".join(sorted(DEFAULT_PROMPT_PROFILES.keys()))
        raise ValueError(f"Unknown AI report prompt_profile '{profile}'. Valid profiles: {valid}, or set prompt_file.")
    return profile, DEFAULT_PROMPT_PROFILES[profile]


def _load_user_questions(ai_cfg: dict[str, Any], config_path: str | Path) -> list[str]:
    questions: list[str] = []
    raw_questions = ai_cfg.get("user_questions", [])
    if isinstance(raw_questions, str):
        raw_questions = [raw_questions]
    if isinstance(raw_questions, list):
        for item in raw_questions:
            text = str(item or "").strip()
            if text:
                questions.append(text)

    questions_file = str(ai_cfg.get("user_questions_file", "") or "").strip()
    if questions_file:
        path = ConfigPathPolicy.default(
            config_path=config_path,
            allow_external_ai_prompt_files=bool(ai_cfg.get("allow_external_ai_prompt_files", False)),
        ).resolve_ai_prompt_file(questions_file, purpose="outputs.ai_report.user_questions_file")
        for line in path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            numbered = re.match(r"^\d+[.)]\s*(.+)$", text)
            if numbered:
                text = numbered.group(1).strip()
            elif text.startswith(("-", "*")):
                text = text[1:].strip()
            if text:
                questions.append(text)

    deduped: list[str] = []
    seen: set[str] = set()
    for question in questions:
        key = question.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(question)
    return deduped

__all__ = [name for name in globals() if not name.startswith("__")]
