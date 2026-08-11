"""One-way game observer policies; observer data is never an FSW input."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

import numpy as np

from sim.core.models import StateBelief, StateTruth


class GameObserverPolicyKind(str, Enum):
    TRUTH_ASSISTED = "truth_assisted"
    ONBOARD_ONLY = "onboard_only"
    HYBRID = "hybrid"


@dataclass(frozen=True, slots=True)
class GameObserverPolicy:
    policy_id: str
    kind: GameObserverPolicyKind
    truth_fields: tuple[str, ...] = ("position", "velocity")
    onboard_fields: tuple[str, ...] = ("state",)

    def __post_init__(self) -> None:
        if not self.policy_id.strip() or not isinstance(self.kind, GameObserverPolicyKind):
            raise ValueError("observer policy requires an identity and kind")

    def sample(
        self,
        *,
        truth: Mapping[str, StateTruth],
        beliefs: Mapping[str, StateBelief],
    ) -> dict[str, object]:
        result: dict[str, object] = {"observer_policy_id": self.policy_id, "observer_policy_kind": self.kind.value}
        if self.kind in (GameObserverPolicyKind.TRUTH_ASSISTED, GameObserverPolicyKind.HYBRID):
            result["truth"] = {
                object_id: {
                    "position_eci_km": np.asarray(state.position_eci_km, dtype=float).tolist(),
                    "velocity_eci_km_s": np.asarray(state.velocity_eci_km_s, dtype=float).tolist(),
                }
                for object_id, state in sorted(truth.items())
            }
        if self.kind in (GameObserverPolicyKind.ONBOARD_ONLY, GameObserverPolicyKind.HYBRID):
            result["onboard"] = {
                object_id: np.asarray(belief.state, dtype=float).tolist()
                for object_id, belief in sorted(beliefs.items())
            }
        return result
