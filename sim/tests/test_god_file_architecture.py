from __future__ import annotations

import importlib

from sim.game import launcher, pygame_dashboard, runner, training
from sim.mission import modules
from sim.tests.god_file_architecture_test_support import unresolved_function_globals


def test_remaining_god_file_facades_publish_ownership_maps() -> None:
    registries = (
        modules.MISSION_STRATEGY_FAMILIES,
        modules.MISSION_EXECUTION_FAMILIES,
        training.TRAINING_CAPABILITY_FAMILIES,
        pygame_dashboard.DASHBOARD_CAPABILITY_FAMILIES,
        launcher.LAUNCHER_CAPABILITY_FAMILIES,
        runner.RUNNER_CAPABILITY_FAMILIES,
    )
    assert all(registry for registry in registries)
    assert all(
        module_path.startswith("sim.")
        for registry in registries
        for module_path in registry.values()
    )


def test_remaining_facades_preserve_primary_class_module_paths() -> None:
    assert modules.PursuitMissionStrategy.__module__ == "sim.mission.modules"
    assert training.RPOTrainingTracker.__module__ == "sim.game.training"
    assert pygame_dashboard.PygameRPODashboard.__module__ == "sim.game.pygame_dashboard"
    assert launcher.GameScenarioOption.__module__ == "sim.game.launcher"
    assert runner.GameRunResult.__module__ == "sim.game.runner"


def test_public_decomposition_owners_have_no_unresolved_function_globals() -> None:
    registries = (
        modules.MISSION_STRATEGY_FAMILIES,
        modules.MISSION_EXECUTION_FAMILIES,
        training.TRAINING_CAPABILITY_FAMILIES,
        pygame_dashboard.DASHBOARD_CAPABILITY_FAMILIES,
        launcher.LAUNCHER_CAPABILITY_FAMILIES,
        runner.RUNNER_CAPABILITY_FAMILIES,
    )
    owner_modules = {
        importlib.import_module(module_path)
        for registry in registries
        for module_path in registry.values()
    }

    assert unresolved_function_globals(owner_modules) == {}
