from __future__ import annotations

import numpy as np
import pytest

from sim.core.models import StateTruth
from sim.flight_software import (
    InputEvent,
    InputKind,
    PacketId,
    Quality,
)
from sim.tests.fsw_v2_helpers import BOOT_ID, clock


def test_game_stack_dynamically_rejects_truth_hidden_in_ground_input() -> None:
    truth = StateTruth(np.zeros(3), np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3), 100.0, 0.0)
    with pytest.raises(TypeError, match="ground_command event payload must be GroundCommandPayload"):
        InputEvent(PacketId("ai", BOOT_ID, 0), InputKind.GROUND_COMMAND, clock(1), clock(1), Quality(), truth)
