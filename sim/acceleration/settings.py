from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
from importlib.util import find_spec
from typing import Any

ACCELERATION_ENV = "OEL_ACCELERATION"
VALID_ACCELERATION_MODES = ("off", "auto", "numba")
_ACCELERATION_CONTEXT: ContextVar[tuple[str, bool] | None] = ContextVar("oel_acceleration_context", default=None)
# Checking the module spec is enough to plan a backend. Importing Numba here
# adds tens of megabytes to every OEL process, including acceleration-off runs.
# The kernel modules perform the real import only when an accelerated path is
# selected.
NUMBA_AVAILABLE = find_spec("numba") is not None


@lru_cache(maxsize=1)
def _runtime_numba_available() -> bool:
    """Verify Numba only when an accelerated backend is actually selected."""
    if not NUMBA_AVAILABLE:
        return False
    from sim.acceleration.optional import NUMBA_AVAILABLE as import_succeeded

    return bool(import_succeeded)


@dataclass(frozen=True)
class AccelerationSettings:
    requested_mode: str
    effective_backend: str
    enabled: bool
    numba_available: bool
    reason: str = ""


def _normalize_mode(value: Any, *, default: str = "off") -> str:
    mode = str(value if value not in (None, "") else default).strip().lower()
    if mode == "config":
        mode = default
    if mode not in VALID_ACCELERATION_MODES:
        available = ", ".join(VALID_ACCELERATION_MODES)
        raise ValueError(f"Unknown acceleration mode {value!r}. Available: {available}")
    return mode


def _resolved_requested_mode(mode: Any = None, *, allow_env_override: bool = True) -> str:
    context = _ACCELERATION_CONTEXT.get()
    if context is not None:
        mode, allow_env_override = context
    env_mode = os.environ.get(ACCELERATION_ENV)
    return _normalize_mode(env_mode if allow_env_override and env_mode not in (None, "") else mode)


def acceleration_cache_key(mode: Any = None, *, allow_env_override: bool = True) -> tuple[str, bool]:
    requested = _resolved_requested_mode(mode, allow_env_override=allow_env_override)
    available = bool(NUMBA_AVAILABLE) if requested == "off" else _runtime_numba_available()
    return requested, available


def acceleration_settings_from_mode(mode: Any = None, *, allow_env_override: bool = True) -> AccelerationSettings:
    requested = _resolved_requested_mode(mode, allow_env_override=allow_env_override)
    if requested == "off":
        return AccelerationSettings(
            requested_mode=requested,
            effective_backend="python",
            enabled=False,
            numba_available=NUMBA_AVAILABLE,
            reason="acceleration disabled",
        )
    runtime_available = _runtime_numba_available()
    if requested == "numba" and not runtime_available:
        return AccelerationSettings(
            requested_mode=requested,
            effective_backend="python",
            enabled=False,
            numba_available=False,
            reason="numba requested but unavailable",
        )
    if requested == "auto" and not runtime_available:
        return AccelerationSettings(
            requested_mode=requested,
            effective_backend="python",
            enabled=False,
            numba_available=False,
            reason="numba unavailable",
        )
    return AccelerationSettings(
        requested_mode=requested,
        effective_backend="numba",
        enabled=True,
        numba_available=True,
        reason="numba available",
    )


def acceleration_enabled_from_mode(mode: Any = None, *, allow_env_override: bool = True) -> bool:
    """Resolve only the hot-path enabled flag without constructing settings evidence."""
    requested = _resolved_requested_mode(mode, allow_env_override=allow_env_override)
    return bool(requested != "off" and _runtime_numba_available())


def acceleration_settings_from_config(cfg: Any) -> AccelerationSettings:
    simulator = getattr(cfg, "simulator", None)
    acceleration = dict(getattr(simulator, "acceleration", {}) or {})
    return acceleration_settings_from_mode(
        acceleration.get("mode", "off"),
        allow_env_override=bool(acceleration.get("env_override", True)),
    )


@contextmanager
def acceleration_context(mode: Any, *, allow_env_override: bool = False) -> Iterator[None]:
    token = _ACCELERATION_CONTEXT.set((_normalize_mode(mode), bool(allow_env_override)))
    try:
        yield
    finally:
        _ACCELERATION_CONTEXT.reset(token)


@contextmanager
def acceleration_context_from_config(cfg: Any) -> Iterator[None]:
    settings = acceleration_settings_from_config(cfg)
    with acceleration_context(settings.requested_mode, allow_env_override=False):
        yield
