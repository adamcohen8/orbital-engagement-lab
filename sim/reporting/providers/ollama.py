# ruff: noqa: F401,F403,F405,I001
from ..ai_report_models import *

def _call_ollama(
    ai_cfg: dict[str, Any],
    user_prompt: str,
    *,
    allow_custom_endpoint: bool = False,
) -> tuple[str, dict[str, Any]]:
    endpoint = resolve_ai_endpoint(
        ai_cfg,
        provider="ollama",
        default_endpoint="http://localhost:11434",
        allow_custom_endpoint=allow_custom_endpoint,
        config_path="outputs.ai_report.endpoint",
    )
    model = str(ai_cfg.get("model", "") or "").strip()
    if not model:
        raise ValueError("outputs.ai_report.model is required for provider='ollama'.")
    timeout_s = float(ai_cfg.get("timeout_s", 120.0) or 120.0)
    body = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You write careful engineering reports from supplied simulation outputs. "
                    "You never invent data, never claim to inspect unavailable images, and always obey report_rules."
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
    }
    if isinstance(ai_cfg.get("options"), dict):
        body["options"] = dict(ai_cfg.get("options") or {})
    request = urllib.request.Request(
        f"{endpoint}/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        raw = response.read().decode("utf-8")
    payload = json.loads(raw)
    message = dict(payload.get("message", {}) or {})
    content = str(message.get("content", payload.get("response", "")) or "")
    if not content.strip():
        raise ValueError("Ollama response did not include report text.")
    return content, payload

__all__ = [name for name in globals() if not name.startswith("__")]
