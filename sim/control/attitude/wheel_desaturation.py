from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.core.interfaces import Controller
from sim.core.models import Command, StateBelief


@dataclass
class WheelDesaturationController(Controller):
    wheel_momentum_body_nms: np.ndarray = field(default_factory=lambda: np.zeros(3))
    momentum_state_slice: tuple[int, int] | None = None
    momentum_threshold_nms: float = 0.1
    unload_gain_s_inv: float = 0.02
    max_unload_torque_nm: float = 0.01

    def set_wheel_momentum(self, wheel_momentum_body_nms: np.ndarray) -> None:
        self.wheel_momentum_body_nms = np.array(wheel_momentum_body_nms, dtype=float).reshape(3)

    def _momentum(self, belief: StateBelief) -> np.ndarray:
        if self.momentum_state_slice is not None:
            i0, i1 = self.momentum_state_slice
            if belief.state.size >= i1 and i1 - i0 == 3:
                return np.array(belief.state[i0:i1], dtype=float).reshape(3)
        return np.array(self.wheel_momentum_body_nms, dtype=float).reshape(3)

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        h_body = self._momentum(belief)
        h_norm = float(np.linalg.norm(h_body))
        threshold = float(max(self.momentum_threshold_nms, 0.0))
        active = h_norm > threshold > 0.0
        torque = np.zeros(3, dtype=float)
        if active:
            torque = -float(max(self.unload_gain_s_inv, 0.0)) * h_body
            n = float(np.linalg.norm(torque))
            max_torque = float(max(self.max_unload_torque_nm, 0.0))
            if max_torque <= 0.0:
                torque = np.zeros(3, dtype=float)
            if n > max_torque > 0.0:
                torque *= max_torque / n
        return Command(
            thrust_eci_km_s2=np.zeros(3, dtype=float),
            torque_body_nm=torque,
            mode_flags={
                "mode": "wheel_desaturation",
                "wheel_desaturation_requested": bool(active),
                "wheel_momentum_body_nms": h_body.tolist(),
                "wheel_momentum_norm_nms": h_norm,
            },
        )
