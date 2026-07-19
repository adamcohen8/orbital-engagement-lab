from __future__ import annotations

import builtins
import dis
import inspect
from collections.abc import Iterable
from types import CodeType, ModuleType


def _nested_code_objects(code: CodeType) -> Iterable[CodeType]:
    yield code
    for constant in code.co_consts:
        if isinstance(constant, CodeType):
            yield from _nested_code_objects(constant)


def _owned_functions(module: ModuleType) -> Iterable[tuple[str, object]]:
    namespace = vars(module)
    seen: set[int] = set()
    for name, value in namespace.items():
        candidates: list[tuple[str, object]] = []
        if inspect.isfunction(value):
            candidates.append((name, value))
        elif inspect.isclass(value) and value.__module__ == module.__name__:
            for member_name, member in vars(value).items():
                if isinstance(member, (staticmethod, classmethod)):
                    member = member.__func__
                elif isinstance(member, property):
                    for accessor_name, accessor in (
                        ("getter", member.fget),
                        ("setter", member.fset),
                        ("deleter", member.fdel),
                    ):
                        if accessor is not None:
                            candidates.append((f"{name}.{member_name}.{accessor_name}", accessor))
                    continue
                if inspect.isfunction(member):
                    candidates.append((f"{name}.{member_name}", member))
        for qualified_name, function in candidates:
            if getattr(function, "__globals__", None) is not namespace or id(function) in seen:
                continue
            seen.add(id(function))
            yield qualified_name, function


def unresolved_function_globals(modules: Iterable[ModuleType]) -> dict[str, list[str]]:
    """Return bytecode global references absent from their owner module.

    This catches decomposition mistakes where a moved function still relies on
    a name that used to share its God-file namespace. Imported compatibility
    functions are ignored because their globals belong to their source module.
    """

    unresolved: dict[str, list[str]] = {}
    builtin_names = set(vars(builtins))
    for module in modules:
        module_names = set(vars(module))
        for qualified_name, function in _owned_functions(module):
            missing: set[str] = set()
            for code in _nested_code_objects(function.__code__):
                for instruction in dis.get_instructions(code):
                    if instruction.opname == "LOAD_GLOBAL" and instruction.argval not in module_names | builtin_names:
                        missing.add(str(instruction.argval))
            if missing:
                unresolved[f"{module.__name__}.{qualified_name}"] = sorted(missing)
    return unresolved
