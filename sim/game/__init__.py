from __future__ import annotations

from sim.game.manual import KeyboardCommandState, ManualGameCommandProvider

__all__ = [
    "KeyboardCommandState",
    "LiveBattlespaceDashboard",
    "ManualGameCommandProvider",
    "run_game_mode",
]


def __getattr__(name: str):
    if name == "LiveBattlespaceDashboard":
        from sim.game.dashboard import LiveBattlespaceDashboard

        return LiveBattlespaceDashboard
    if name == "run_game_mode":
        from sim.game.runner import run_game_mode

        return run_game_mode
    raise AttributeError(name)
