from __future__ import annotations

import importlib
import logging

import sim.runtime_support as runtime_support
from sim.runtime.architecture import RUNTIME_CONSTRUCTION_FAMILIES, SINGLE_RUN_COLLABORATORS


def test_runtime_ownership_map_resolves_implementations_and_facade_exports() -> None:
    names = [family.name for family in RUNTIME_CONSTRUCTION_FAMILIES]
    capabilities = [capability for family in RUNTIME_CONSTRUCTION_FAMILIES for capability in family.capabilities]
    assert len(names) == len(set(names))
    assert len(capabilities) == len(set(capabilities))
    for family in RUNTIME_CONSTRUCTION_FAMILIES:
        implementation = importlib.import_module(family.module)
        facade = importlib.import_module(family.facade)
        for capability in family.capabilities:
            assert callable(getattr(implementation, capability))
            assert callable(getattr(facade, capability))


def test_single_run_collaborator_modules_are_importable() -> None:
    names = [name for name, _module in SINGLE_RUN_COLLABORATORS]
    assert len(names) == len(set(names))
    for _name, module_name in SINGLE_RUN_COLLABORATORS:
        assert importlib.import_module(module_name) is not None


def test_runtime_support_preserves_historical_logger_namespace() -> None:
    assert isinstance(runtime_support.logger, logging.Logger)
    assert runtime_support.logger.name == "sim.runtime_support"
