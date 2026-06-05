from __future__ import annotations

from enum import Enum
from typing import Any


class GamePhase(str, Enum):
    BRIEFING = "briefing"
    PRIMER = "primer"
    PLAYING = "playing"
    PAUSED = "paused"
    PASSED = "passed"
    FAILED = "failed"
    ARCADE_TRANSITION = "arcade_transition"


def phase_from_score(score: Any, *, briefing_open: bool = False, paused: bool = False) -> GamePhase:
    if bool(getattr(score, "level_passed", False)):
        return GamePhase.PASSED
    if bool(getattr(score, "level_failed", False)):
        return GamePhase.FAILED
    if bool(briefing_open):
        return GamePhase.BRIEFING
    if bool(paused):
        return GamePhase.PAUSED
    return GamePhase.PLAYING


def phase_shows_briefing(phase: GamePhase) -> bool:
    return phase in {GamePhase.BRIEFING, GamePhase.ARCADE_TRANSITION}


def phase_is_terminal(phase: GamePhase) -> bool:
    return phase in {GamePhase.PASSED, GamePhase.FAILED}


def mission_state_for_dashboard(phase: GamePhase) -> str:
    if phase == GamePhase.PASSED:
        return "passed"
    if phase == GamePhase.FAILED:
        return "failed"
    return "active"
