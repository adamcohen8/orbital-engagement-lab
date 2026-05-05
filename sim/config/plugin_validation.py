from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from typing import Any

from sim.config.object_refs import configured_objects, object_parameter_prefix


@dataclass(frozen=True)
class PluginContract:
    methods_all: tuple[str, ...] = ()
    methods_any: tuple[str, ...] = ()
    allow_function: bool = False


_CONTRACTS = {
    "guidance": PluginContract(methods_all=("command",), allow_function=False),
    "orbit_control": PluginContract(methods_all=("act",), allow_function=False),
    "attitude_control": PluginContract(methods_all=("act",), allow_function=False),
    "mission_strategy": PluginContract(methods_all=(), methods_any=("update", "plan", "decide"), allow_function=True),
    "mission_execution": PluginContract(methods_all=(), methods_any=("update", "execute", "act"), allow_function=True),
    "bridge": PluginContract(
        methods_all=(),
        methods_any=("step", "start", "send_command", "receive_command", "external_intent"),
        allow_function=True,
    ),
    "mission_objective": PluginContract(
        methods_all=(), methods_any=("evaluate", "update", "check", "act"), allow_function=True
    ),
}


def _validate_pointer(pointer: Any, contract: PluginContract, path: str) -> list[str]:
    errs: list[str] = []
    if pointer is None:
        return errs
    if not getattr(pointer, "module", None):
        errs.append(f"{path}: missing 'module'.")
        return errs

    try:
        mod = importlib.import_module(pointer.module)
    except Exception as ex:
        errs.append(f"{path}: failed to import module '{pointer.module}': {ex}")
        return errs

    class_name = getattr(pointer, "class_name", None)
    function = getattr(pointer, "function", None)

    if class_name:
        if not hasattr(mod, class_name):
            errs.append(f"{path}: class '{class_name}' not found in module '{pointer.module}'.")
            return errs
        cls = getattr(mod, class_name)
        if not inspect.isclass(cls):
            errs.append(f"{path}: '{class_name}' in module '{pointer.module}' is not a class.")
            return errs
        for m in contract.methods_all:
            if not _class_has_callable(cls, m):
                errs.append(f"{path}: class '{class_name}' missing required callable method '{m}'.")
        if contract.methods_any:
            if not any(_class_has_callable(cls, m) for m in contract.methods_any):
                errs.append(f"{path}: class '{class_name}' must implement one of {list(contract.methods_any)}.")
        return errs

    if function:
        if not contract.allow_function:
            errs.append(f"{path}: function pointers are not allowed for this plugin type.")
            return errs
        if not hasattr(mod, function):
            errs.append(f"{path}: function '{function}' not found in module '{pointer.module}'.")
            return errs
        fn = getattr(mod, function)
        if not callable(fn):
            errs.append(f"{path}: '{function}' in module '{pointer.module}' is not callable.")
        return errs

    errs.append(f"{path}: must define either 'class_name' or 'function'.")
    return errs


def _class_has_callable(cls: type, method_name: str) -> bool:
    for base in cls.__mro__:
        if method_name in base.__dict__:
            attr = base.__dict__[method_name]
            if isinstance(attr, (staticmethod, classmethod)):
                return callable(attr.__func__)
            return callable(attr)
    return False


def validate_scenario_plugins(cfg: Any) -> list[str]:
    errs: list[str] = []
    for object_id, agent in configured_objects(cfg).items():
        if not getattr(agent, "enabled", False):
            continue
        path = object_parameter_prefix(str(object_id))
        if str(getattr(agent, "kind", "satellite")).strip().lower() == "rocket":
            errs.extend(_validate_pointer(getattr(agent, "guidance", None), _CONTRACTS["guidance"], f"{path}.guidance"))
            errs.extend(
                _validate_pointer(
                    getattr(agent, "base_guidance", None), _CONTRACTS["guidance"], f"{path}.base_guidance"
                )
            )
            for i, p in enumerate(getattr(agent, "guidance_modifiers", []) or []):
                errs.extend(_validate_rocket_guidance_modifier(p, f"{path}.guidance_modifiers[{i}]"))
        errs.extend(
            _validate_pointer(
                getattr(agent, "orbit_control", None), _CONTRACTS["orbit_control"], f"{path}.orbit_control"
            )
        )
        errs.extend(
            _validate_pointer(
                getattr(agent, "attitude_control", None), _CONTRACTS["attitude_control"], f"{path}.attitude_control"
            )
        )
        errs.extend(
            _validate_pointer(
                getattr(agent, "mission_strategy", None), _CONTRACTS["mission_strategy"], f"{path}.mission_strategy"
            )
        )
        errs.extend(
            _validate_pointer(
                getattr(agent, "mission_execution", None), _CONTRACTS["mission_execution"], f"{path}.mission_execution"
            )
        )
        bridge = getattr(agent, "bridge", None)
        if bridge is not None and getattr(bridge, "enabled", False):
            errs.extend(_validate_pointer(bridge, _CONTRACTS["bridge"], f"{path}.bridge"))
        for i, p in enumerate(getattr(agent, "mission_objectives", []) or []):
            errs.extend(_validate_pointer(p, _CONTRACTS["mission_objective"], f"{path}.mission_objectives[{i}]"))
    return errs


def _validate_rocket_guidance_modifier(pointer: Any, path: str) -> list[str]:
    errs: list[str] = []
    if pointer is None:
        return errs
    if getattr(pointer, "kind", "python") != "python":
        return [f"{path}: only kind='python' is supported."]
    if not getattr(pointer, "module", None):
        return [f"{path}: 'module' is required for python pointers."]
    try:
        mod = importlib.import_module(str(pointer.module))
    except Exception as ex:
        return [f"{path}: failed to import module '{pointer.module}': {ex}"]
    class_name = getattr(pointer, "class_name", None)
    if not class_name:
        return [f"{path}: must define 'class_name'."]
    if not hasattr(mod, class_name):
        return [f"{path}: class '{class_name}' not found in module '{pointer.module}'."]
    cls = getattr(mod, class_name)
    if not inspect.isclass(cls):
        return [f"{path}: '{class_name}' in module '{pointer.module}' is not a class."]
    if not _class_has_callable(cls, "command"):
        errs.append(f"{path}: class '{class_name}' missing required callable method 'command'.")
    return errs
