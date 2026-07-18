from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any, Callable

import numpy as np

from sim.core.models import Command, StateBelief
from sim.scenarios import ScenarioBuilder as ScenarioBuilder
from sim.scenarios import ValidationIssue as ValidationIssue


def _compatible_call(fn: Callable[..., Any], kwargs: dict[str, Any], fallback_kwargs: dict[str, Any]) -> Any:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(**kwargs)

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return fn(**kwargs)

    filtered: dict[str, Any] = {}
    for name, param in signature.parameters.items():
        if param.kind == inspect.Parameter.POSITIONAL_ONLY:
            return fn(**fallback_kwargs)
        if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY) and name in kwargs:
            filtered[name] = kwargs[name]

    missing_required = [
        name
        for name, param in signature.parameters.items()
        if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and param.default is inspect.Signature.empty
        and name not in filtered
    ]
    if missing_required:
        return fn(**fallback_kwargs)
    return fn(**filtered)


class _CallableControllerAdapter:
    def __init__(self, fn: Callable[..., Any], *, command_kind: str) -> None:
        self.fn = fn
        self.command_kind = str(command_kind)

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        ret = _compatible_call(
            self.fn,
            {
                "belief": belief,
                "state": belief.state,
                "t_s": t_s,
                "budget_ms": budget_ms,
            },
            {
                "belief": belief,
                "t_s": t_s,
            },
        )
        return _coerce_controller_return(ret, command_kind=self.command_kind)


class _CallableMissionAdapter:
    def __init__(self, fn: Callable[..., Any]) -> None:
        self.fn = fn

    def update(self, **kwargs: Any) -> dict[str, Any]:
        ret = _compatible_call(self.fn, dict(kwargs), {"truth": kwargs.get("truth"), "t_s": kwargs.get("t_s", 0.0)})
        return dict(ret) if isinstance(ret, Mapping) else {}


def _coerce_controller_return(value: Any, *, command_kind: str) -> Command:
    if value is None:
        return Command.zero()
    if isinstance(value, Command):
        return value
    if isinstance(value, Mapping):
        cmd = Command.zero()
        if "thrust_eci_km_s2" in value:
            cmd.thrust_eci_km_s2 = np.array(value["thrust_eci_km_s2"], dtype=float).reshape(3)
        elif "accel_eci_km_s2" in value:
            cmd.thrust_eci_km_s2 = np.array(value["accel_eci_km_s2"], dtype=float).reshape(3)
        if "torque_body_nm" in value:
            cmd.torque_body_nm = np.array(value["torque_body_nm"], dtype=float).reshape(3)
        if isinstance(value.get("mode_flags"), Mapping):
            cmd.mode_flags.update(dict(value["mode_flags"]))
        return cmd
    arr = np.array(value, dtype=float).reshape(-1)
    if arr.size != 3:
        raise TypeError("Controller callables must return Command, mapping, None, or a length-3 vector.")
    if command_kind == "attitude":
        return Command(torque_body_nm=arr.copy(), mode_flags={"mode": "api_attitude_controller"})
    return Command(thrust_eci_km_s2=arr.copy(), mode_flags={"mode": "api_orbit_controller"})


def _controller_object(value: Any, *, command_kind: str) -> Any:
    if value is None:
        return None
    if hasattr(value, "act") and callable(value.act):
        return value
    if callable(value):
        return _CallableControllerAdapter(value, command_kind=command_kind)
    raise TypeError("Controller override must be a controller object with .act(), a callable, or None.")


def _mission_object(value: Any) -> Any:
    if value is None:
        return None
    if any(callable(getattr(value, name, None)) for name in ("update", "plan", "decide", "execute", "act")):
        return value
    if callable(value):
        return _CallableMissionAdapter(value)
    raise TypeError("Mission override must be an object with a mission method, a callable, or None.")
