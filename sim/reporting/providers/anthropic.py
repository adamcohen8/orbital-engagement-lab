# ruff: noqa: F401,F403,F405,I001
from ..ai_report_models import *
from .common import *

def _anthropic_request_options(ai_cfg: dict[str, Any]) -> dict[str, Any]:
    raw = _provider_options(ai_cfg)
    out: dict[str, Any] = {}
    for key in ("temperature", "top_p", "top_k", "stop_sequences", "metadata"):
        if key in raw:
            out[key] = raw[key]
    max_tokens = raw.get("max_tokens", raw.get("max_output_tokens", ai_cfg.get("max_tokens", 2048)))
    out["max_tokens"] = int(max_tokens or 2048)
    return out


def _extract_anthropic_response_text(payload: dict[str, Any]) -> str:
    parts: list[str] = []
    for content in list(payload.get("content", []) or []):
        content_dict = dict(content or {})
        if content_dict.get("type") == "text" and content_dict.get("text"):
            parts.append(str(content_dict.get("text")))
    if parts:
        return "\n".join(parts)
    raise ValueError(f"Anthropic response did not include text content. stop_reason={payload.get('stop_reason')!r}")


def _call_anthropic(
    ai_cfg: dict[str, Any],
    user_prompt: str,
    *,
    allow_custom_endpoint: bool = False,
) -> tuple[str, dict[str, Any]]:
    endpoint = resolve_ai_endpoint(
        ai_cfg,
        provider="anthropic",
        default_endpoint="https://api.anthropic.com/v1",
        allow_custom_endpoint=allow_custom_endpoint,
        **_custom_endpoint_policy(ai_cfg),
        config_path="outputs.ai_report.endpoint",
    )
    _, api_key = _api_key_from_env(ai_cfg, default_env="ANTHROPIC_API_KEY", provider_name="anthropic")
    model = str(ai_cfg.get("model", "") or "").strip()
    if not model:
        raise ValueError("outputs.ai_report.model is required for provider='anthropic'.")
    timeout_s = float(ai_cfg.get("timeout_s", 120.0) or 120.0)
    version = str(ai_cfg.get("anthropic_version", "2023-06-01") or "2023-06-01")
    body: dict[str, Any] = {
        "model": model,
        "system": (
            "You write careful engineering reports from supplied simulation outputs. "
            "You never invent data, never claim to inspect unavailable images, and always obey report_rules."
        ),
        "messages": [
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
    }
    body.update(_anthropic_request_options(ai_cfg))
    request = urllib.request.Request(
        f"{endpoint}/messages",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "anthropic-version": version,
        },
        method="POST",
    )
    _add_nonredirected_secret_header(request, "x-api-key", api_key)
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        raw = _read_bounded_response(response).decode("utf-8")
    payload = json.loads(raw)
    return _extract_anthropic_response_text(payload), payload

__all__ = [name for name in globals() if not name.startswith("__")]
