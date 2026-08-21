# ruff: noqa: F401,F403,F405,I001
from ..ai_report_models import *
from .common import *

def _gemini_generation_config(ai_cfg: dict[str, Any]) -> dict[str, Any]:
    raw = ai_cfg.get("generation_config", ai_cfg.get("generationConfig", ai_cfg.get("options", {})))
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    aliases = {
        "temperature": "temperature",
        "top_p": "topP",
        "topP": "topP",
        "top_k": "topK",
        "topK": "topK",
        "max_output_tokens": "maxOutputTokens",
        "maxOutputTokens": "maxOutputTokens",
        "stop_sequences": "stopSequences",
        "stopSequences": "stopSequences",
        "candidate_count": "candidateCount",
        "candidateCount": "candidateCount",
    }
    for key, value in raw.items():
        mapped = aliases.get(str(key), str(key))
        out[mapped] = value
    return out

def _call_gemini(
    ai_cfg: dict[str, Any],
    user_prompt: str,
    *,
    allow_custom_endpoint: bool = False,
) -> tuple[str, dict[str, Any]]:
    endpoint = resolve_ai_endpoint(
        ai_cfg,
        provider="google",
        default_endpoint="https://generativelanguage.googleapis.com/v1beta",
        allow_custom_endpoint=allow_custom_endpoint,
        **_custom_endpoint_policy(ai_cfg),
        config_path="outputs.ai_report.endpoint",
    )
    _, api_key = _api_key_from_env(ai_cfg, default_env="GEMINI_API_KEY", provider_name="google")
    model = str(ai_cfg.get("model", "") or "").strip()
    if not model:
        raise ValueError("outputs.ai_report.model is required for provider='google'.")
    timeout_s = float(ai_cfg.get("timeout_s", 120.0) or 120.0)
    body: dict[str, Any] = {
        "systemInstruction": {
            "parts": [
                {
                    "text": (
                        "You write careful engineering reports from supplied simulation outputs. "
                        "You never invent data, never claim to inspect unavailable images, and always obey report_rules."
                    )
                }
            ]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_prompt}],
            }
        ],
    }
    generation_config = _gemini_generation_config(ai_cfg)
    if generation_config:
        body["generationConfig"] = generation_config
    request = urllib.request.Request(
        f"{endpoint}/models/{model}:generateContent",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )
    _add_nonredirected_secret_header(request, "x-goog-api-key", api_key)
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        raw = _read_bounded_response(response).decode("utf-8")
    payload = json.loads(raw)
    parts: list[str] = []
    for candidate in list(payload.get("candidates", []) or []):
        content = dict(candidate.get("content", {}) or {})
        for part in list(content.get("parts", []) or []):
            text = dict(part or {}).get("text")
            if text:
                parts.append(str(text))
    if not parts:
        prompt_feedback = payload.get("promptFeedback", payload.get("prompt_feedback"))
        raise ValueError(f"Gemini response did not include text candidates. promptFeedback={prompt_feedback!r}")
    return "\n".join(parts), payload

__all__ = [name for name in globals() if not name.startswith("__")]
