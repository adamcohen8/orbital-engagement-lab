from __future__ import annotations


def format_parallel_fallback_reason(exc: BaseException) -> str:
    detail = str(exc).strip()
    if detail:
        return f"{type(exc).__name__}: {detail}"
    return f"{type(exc).__name__}: worker process exited without an error message"
