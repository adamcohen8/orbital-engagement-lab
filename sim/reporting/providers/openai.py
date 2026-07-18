# ruff: noqa: F401,F403,F405,I001
from ..ai_report_models import *
from .common import *

def _openai_request_options(ai_cfg: dict[str, Any]) -> dict[str, Any]:
    raw = _provider_options(ai_cfg)
    out: dict[str, Any] = {}
    allowed = {
        "temperature",
        "top_p",
        "max_output_tokens",
        "reasoning",
        "text",
        "truncation",
        "parallel_tool_calls",
        "metadata",
        "user",
    }
    for key, value in raw.items():
        if str(key) in allowed:
            out[str(key)] = value
    if "max_output_tokens" not in out and "max_tokens" in raw:
        out["max_output_tokens"] = raw["max_tokens"]
    return out


def _extract_openai_response_text(payload: dict[str, Any]) -> str:
    direct = payload.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct
    parts: list[str] = []
    for item in list(payload.get("output", []) or []):
        item_dict = dict(item or {})
        for content in list(item_dict.get("content", []) or []):
            content_dict = dict(content or {})
            if content_dict.get("type") in {"output_text", "text"} and content_dict.get("text"):
                parts.append(str(content_dict.get("text")))
    if parts:
        return "\n".join(parts)
    raise ValueError(
        f"OpenAI response did not include text output. status={payload.get('status')!r} error={payload.get('error')!r}"
    )


def _call_openai(
    ai_cfg: dict[str, Any],
    user_prompt: str,
    *,
    allow_custom_endpoint: bool = False,
) -> tuple[str, dict[str, Any]]:
    endpoint = resolve_ai_endpoint(
        ai_cfg,
        provider="openai",
        default_endpoint="https://api.openai.com/v1",
        allow_custom_endpoint=allow_custom_endpoint,
        **_custom_endpoint_policy(ai_cfg),
        config_path="outputs.ai_report.endpoint",
    )
    _, api_key = _api_key_from_env(ai_cfg, default_env="OPENAI_API_KEY", provider_name="openai")
    model = str(ai_cfg.get("model", "") or "").strip()
    if not model:
        raise ValueError("outputs.ai_report.model is required for provider='openai'.")
    timeout_s = float(ai_cfg.get("timeout_s", 120.0) or 120.0)
    body: dict[str, Any] = {
        "model": model,
        "instructions": (
            "You write careful engineering reports from supplied simulation outputs. "
            "You never invent data, never claim to inspect unavailable images, and always obey report_rules."
        ),
        "input": user_prompt,
    }
    body.update(_openai_request_options(ai_cfg))
    request = urllib.request.Request(
        f"{endpoint}/responses",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        raw = response.read().decode("utf-8")
    payload = json.loads(raw)
    return _extract_openai_response_text(payload), payload

__all__ = [name for name in globals() if not name.startswith("__")]
