# ruff: noqa: F401,F403,F405,I001
from ..ai_report_models import *

MAX_AI_RESPONSE_BYTES = 16 * 1024 * 1024


def _read_bounded_response(response: Any, *, maximum: int = MAX_AI_RESPONSE_BYTES) -> bytes:
    payload = response.read(int(maximum) + 1)
    if len(payload) > int(maximum):
        raise ValueError(f"AI provider response exceeds the {maximum}-byte limit.")
    return payload

def _api_key_from_env(ai_cfg: dict[str, Any], *, default_env: str, provider_name: str) -> tuple[str, str]:
    env_name = str(ai_cfg.get("api_key_env", default_env) or default_env).strip()
    if not env_name:
        raise ValueError(f"outputs.ai_report.api_key_env must be non-empty for provider='{provider_name}'.")
    api_key = os.environ.get(env_name, "").strip()
    if not api_key:
        raise ValueError(
            f"Missing API key for provider='{provider_name}'. Set environment variable {env_name}, "
            "or configure outputs.ai_report.api_key_env to another environment variable name."
        )
    return env_name, api_key

def _provider_options(ai_cfg: dict[str, Any]) -> dict[str, Any]:
    raw = ai_cfg.get("options", {})
    return dict(raw) if isinstance(raw, dict) else {}


def _add_nonredirected_secret_header(request: urllib.request.Request, name: str, value: str) -> None:
    """Send a credential on the initial request without forwarding it on redirects."""
    request.add_unredirected_header(str(name), str(value))


def _custom_endpoint_policy(ai_cfg: dict[str, Any]) -> dict[str, bool]:
    return {
        "allow_custom_endpoint_api_key": bool(ai_cfg.get("allow_custom_endpoint_api_key", False)),
        "allow_insecure_custom_endpoint": bool(ai_cfg.get("allow_insecure_custom_endpoint", False)),
    }

__all__ = [name for name in globals() if not name.startswith("__")]
