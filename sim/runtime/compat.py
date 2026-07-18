"""Plugin construction and legacy call-signature compatibility."""

from __future__ import annotations

import inspect
from functools import lru_cache
from typing import Any, Callable

from sim.config.plugin_specs import instantiate_plugin_spec


def _module_obj(pointer: Any, *, extra_kwargs: dict[str, Any] | None = None) -> Any | None:
    if pointer is None or pointer.module is None:
        return None
    if extra_kwargs:
        from dataclasses import replace

        pointer = replace(pointer, params={**dict(pointer.params or {}), **dict(extra_kwargs)})
    return instantiate_plugin_spec(pointer)


@lru_cache(maxsize=256)
def _cached_compatibility_plan(
    target: Callable[..., Any],
    bound_method: bool,
) -> tuple[bool, tuple[str, ...], frozenset[str]] | None:
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return None

    parameters = list(signature.parameters.values())
    if bound_method and parameters:
        parameters = parameters[1:]
    accepts_var_kwargs = False
    accepted_names: list[str] = []
    required_names: set[str] = set()
    for param in parameters:
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            accepts_var_kwargs = True
            continue
        if param.kind == inspect.Parameter.POSITIONAL_ONLY:
            return None
        if param.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            continue
        accepted_names.append(param.name)
        if param.default is inspect.Signature.empty:
            required_names.add(param.name)
    return accepts_var_kwargs, tuple(accepted_names), frozenset(required_names)


def _compatibility_plan(method: Callable[..., Any]) -> tuple[bool, tuple[str, ...], frozenset[str]] | None:
    target = getattr(method, "__func__", method)
    bound_method = target is not method
    try:
        hash(target)
    except TypeError:
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            return None
        accepts_var_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()
        )
        if any(param.kind == inspect.Parameter.POSITIONAL_ONLY for param in signature.parameters.values()):
            return None
        accepted_names = tuple(
            param.name
            for param in signature.parameters.values()
            if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        )
        required_names = frozenset(
            param.name
            for param in signature.parameters.values()
            if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            and param.default is inspect.Signature.empty
        )
        return accepts_var_kwargs, accepted_names, required_names
    return _cached_compatibility_plan(target, bound_method)


def _compatible_keyword_args(method: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any] | None:
    plan = _compatibility_plan(method)
    if plan is None:
        return None
    accepts_var_kwargs, accepted_names, required_names = plan
    filtered = {name: kwargs[name] for name in accepted_names if name in kwargs}
    if not required_names.issubset(filtered):
        return None

    return dict(kwargs) if accepts_var_kwargs else filtered


def _call_with_compat_kwargs(
    method: Callable[..., Any],
    *,
    primary_kwargs: dict[str, Any],
    fallback_kwargs: dict[str, Any] | None = None,
) -> Any:
    compatible = _compatible_keyword_args(method, primary_kwargs)
    if compatible is not None:
        return method(**compatible)
    if fallback_kwargs is not None:
        compatible = _compatible_keyword_args(method, fallback_kwargs)
        if compatible is not None:
            return method(**compatible)
    return method(**primary_kwargs)
