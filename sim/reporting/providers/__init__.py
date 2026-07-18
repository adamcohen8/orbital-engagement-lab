# ruff: noqa: F401,F403,F405,I001
from .common import *
from .ollama import *
from .gemini import *
from .openai import *
from .anthropic import *

def _call_provider(
    ai_cfg: dict[str, Any],
    user_prompt: str,
    *,
    allow_custom_endpoint: bool = False,
) -> tuple[str, dict[str, Any]]:
    provider = str(ai_cfg.get("provider", "ollama") or "ollama").strip().lower()
    if provider == "ollama":
        return _call_ollama(ai_cfg, user_prompt, allow_custom_endpoint=allow_custom_endpoint)
    if provider in {"google", "gemini"}:
        return _call_gemini(ai_cfg, user_prompt, allow_custom_endpoint=allow_custom_endpoint)
    if provider == "openai":
        return _call_openai(ai_cfg, user_prompt, allow_custom_endpoint=allow_custom_endpoint)
    if provider in {"anthropic", "claude"}:
        return _call_anthropic(ai_cfg, user_prompt, allow_custom_endpoint=allow_custom_endpoint)
    raise ValueError(
        f"Unsupported AI report provider '{provider}'. Supported providers: ollama, google, openai, anthropic."
    )

__all__ = [name for name in globals() if not name.startswith("__")]
