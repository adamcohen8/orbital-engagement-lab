from __future__ import annotations

import importlib
from collections.abc import Iterator, Mapping
from typing import Any

_POINTER_TARGET_KEYS = ("class_name", "function")


def plugin_spec_field(value: Any, name: str, default: Any = None) -> Any:
    """Read a plugin-spec field from either a typed pointer or a raw mapping."""

    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def is_plugin_spec(value: Any) -> bool:
    """Return whether *value* structurally declares an importable plugin target."""

    module = str(plugin_spec_field(value, "module", "") or "").strip()
    if not module:
        return False
    return any(str(plugin_spec_field(value, key, "") or "").strip() for key in _POINTER_TARGET_KEYS)


def instantiate_plugin_spec(value: Any, *, description: str = "plugin") -> Any:
    """Construct one plugin target through the shared import path."""

    module_name = str(plugin_spec_field(value, "module", "") or "").strip()
    class_name = str(plugin_spec_field(value, "class_name", "") or "").strip()
    function_name = str(plugin_spec_field(value, "function", "") or "").strip()
    params = dict(plugin_spec_field(value, "params", {}) or {})
    if not module_name:
        raise ValueError(f"{description} spec must include 'module'.")
    if class_name and function_name:
        raise ValueError(f"{description} spec cannot define both class_name and function.")
    try:
        module = importlib.import_module(module_name)
        if class_name:
            return getattr(module, class_name)(**params)
        if function_name:
            return getattr(module, function_name)
        return module
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Failed to construct requested {description} {value!r}: {exc}") from exc


def iter_nested_plugin_specs(value: Any, path: str) -> Iterator[tuple[str, Any]]:
    """Recursively discover plugin specs embedded in a pointer's parameter tree.

    Composite OEL plugins historically accepted raw nested dictionaries and
    imported them themselves.  Safe and sealed validation must nevertheless see
    every import target before any constructor is allowed to run.
    """

    params = plugin_spec_field(value, "params", None)
    if params is not None:
        yield from _walk(params, f"{path}.params")


def _walk(value: Any, path: str) -> Iterator[tuple[str, Any]]:
    if isinstance(value, Mapping):
        if is_plugin_spec(value):
            yield path, value
        for key, child in value.items():
            yield from _walk(child, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _walk(child, f"{path}[{index}]")
