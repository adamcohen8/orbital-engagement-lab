from __future__ import annotations

from sim.game.manual import KeyboardCommandState, ManualGameCommandProvider
from sim.game.training import RPOTrainingConfig, RPOTrainingScore, RPOTrainingTracker

__all__ = [
    "KeyboardCommandState",
    "ManualGameCommandProvider",
    "PygameRPODashboard",
    "RPOTrainingConfig",
    "RPOTrainingScore",
    "RPOTrainingTracker",
    "run_game_mode",
]


def __getattr__(name: str):
    if name == "PygameRPODashboard":
        from sim.game.pygame_dashboard import PygameRPODashboard

        return PygameRPODashboard
    if name == "run_game_mode":
        from sim.game.runner import run_game_mode

        return run_game_mode
    raise AttributeError(name)
