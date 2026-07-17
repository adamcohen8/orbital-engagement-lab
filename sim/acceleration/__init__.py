"""Optional acceleration API without importing Numba until it is requested."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "NUMBA_AVAILABLE": ("sim.acceleration.optional", "NUMBA_AVAILABLE"),
    "acceleration_backend_name": ("sim.acceleration.optional", "acceleration_backend_name"),
    "AccelerationSettings": ("sim.acceleration.settings", "AccelerationSettings"),
    "acceleration_settings_from_config": ("sim.acceleration.settings", "acceleration_settings_from_config"),
}

__all__ = [
    "NUMBA_AVAILABLE",
    "AccelerationSettings",
    "acceleration_backend_name",
    "acceleration_settings_from_config",
]


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
