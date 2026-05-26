from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

try:  # pragma: no cover - covered indirectly when numba is installed.
    from numba import njit as _numba_njit  # type: ignore

    NUMBA_AVAILABLE = True
    NUMBA_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - depends on local optional deps.
    _numba_njit = None
    NUMBA_AVAILABLE = False
    NUMBA_IMPORT_ERROR = str(exc)


def njit_or_identity(*, cache: bool = True, fastmath: bool = False) -> Callable[[F], F]:
    """Return numba.njit when available, otherwise leave the function unchanged."""

    def decorator(func: F) -> F:
        if not NUMBA_AVAILABLE or _numba_njit is None:
            return func
        return _numba_njit(cache=cache, fastmath=fastmath)(func)  # type: ignore[return-value]

    return decorator


def acceleration_backend_name() -> str:
    return "numba" if NUMBA_AVAILABLE else "python"
