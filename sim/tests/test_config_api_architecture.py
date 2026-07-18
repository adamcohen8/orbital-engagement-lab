from __future__ import annotations

import importlib

import sim
from sim.config.scenario.architecture import SCENARIO_CONFIG_FAMILIES
from sim.public_api.architecture import PUBLIC_API_FAMILIES


def _assert_unique_ownership(families) -> None:
    names = [family.name for family in families]
    capabilities = [capability for family in families for capability in family.capabilities]
    assert len(names) == len(set(names))
    assert len(capabilities) == len(set(capabilities))


def test_scenario_config_ownership_map_matches_compatibility_facade() -> None:
    _assert_unique_ownership(SCENARIO_CONFIG_FAMILIES)
    for family in SCENARIO_CONFIG_FAMILIES:
        implementation = importlib.import_module(family.module)
        facade = importlib.import_module(family.facade)
        for capability in family.capabilities:
            assert getattr(facade, capability) is getattr(implementation, capability)


def test_public_api_ownership_map_matches_facade() -> None:
    _assert_unique_ownership(PUBLIC_API_FAMILIES)
    for family in PUBLIC_API_FAMILIES:
        implementation = importlib.import_module(family.module)
        facade = importlib.import_module(family.facade)
        for capability in family.capabilities:
            assert getattr(facade, capability) is getattr(implementation, capability)


def test_public_api_class_identity_and_module_names_remain_stable() -> None:
    api = importlib.import_module("sim.api")
    public_names = (
        "SimulationConfig",
        "SimulationSnapshot",
        "SimulationResult",
        "SimulationSession",
        "HostedSimulationSession",
        "SimulationWorkspace",
        "HostedSimulationWorkspace",
    )
    for name in public_names:
        cls = getattr(api, name)
        assert getattr(sim, name) is cls
        assert cls.__module__ == "sim.api"
