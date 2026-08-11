from __future__ import annotations

import numpy as np

from sim.core.models import StateBelief, StateTruth
from sim.game.observer import GameObserverPolicy, GameObserverPolicyKind


def _truth() -> StateTruth:
    return StateTruth(np.ones(3), np.ones(3), np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3), 10.0, 0.0)


def test_observer_policies_keep_truth_assistance_outside_the_stack_plane() -> None:
    belief = StateBelief(np.arange(6.0), np.eye(6), 0.0)
    truth_only = GameObserverPolicy("hud-truth", GameObserverPolicyKind.TRUTH_ASSISTED).sample(
        truth={"sat": _truth()}, beliefs={"sat": belief}
    )
    onboard_only = GameObserverPolicy("hud-onboard", GameObserverPolicyKind.ONBOARD_ONLY).sample(
        truth={"sat": _truth()}, beliefs={"sat": belief}
    )
    hybrid = GameObserverPolicy("hud-hybrid", GameObserverPolicyKind.HYBRID).sample(
        truth={"sat": _truth()}, beliefs={"sat": belief}
    )
    assert "truth" in truth_only and "onboard" not in truth_only
    assert "onboard" in onboard_only and "truth" not in onboard_only
    assert {"truth", "onboard"}.issubset(hybrid)
