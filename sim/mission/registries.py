"""Static ownership registries for mission strategy and execution families."""

MISSION_STRATEGY_FAMILIES = {
    "satellite": "sim.mission.strategies.satellite",
    "rocket": "sim.mission.strategies.rocket",
    "executive": "sim.mission.executive",
}

MISSION_EXECUTION_FAMILIES = {
    "pointing": "sim.mission.execution.pointing",
    "burns": "sim.mission.execution.burns",
    "integrated": "sim.mission.execution.integrated",
    "safe_hold": "sim.mission.execution.safe_hold",
    "legacy": "sim.mission.legacy_modules",
}
