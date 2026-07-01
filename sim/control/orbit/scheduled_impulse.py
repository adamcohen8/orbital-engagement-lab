from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.core.interfaces import Controller
from sim.core.models import Command, StateBelief


@dataclass
class ScheduledImpulseController(Controller):
    """Coast, apply a deterministic finite-duration impulse, then coast."""

    start_time_s: float
    duration_s: float
    delta_v_eci_m_s: np.ndarray | None = None
    acceleration_eci_km_s2: np.ndarray | None = None
    label: str = "scheduled_impulse"
    _delta_v_km_s: np.ndarray = field(default_factory=lambda: np.zeros(3), init=False, repr=False)
    _accel_km_s2: np.ndarray = field(default_factory=lambda: np.zeros(3), init=False, repr=False)

    def __post_init__(self) -> None:
        if float(self.start_time_s) < 0.0:
            raise ValueError("start_time_s must be non-negative.")
        if float(self.duration_s) <= 0.0:
            raise ValueError("duration_s must be positive.")
        if self.delta_v_eci_m_s is None and self.acceleration_eci_km_s2 is None:
            raise ValueError("Provide delta_v_eci_m_s or acceleration_eci_km_s2.")
        if self.delta_v_eci_m_s is not None and self.acceleration_eci_km_s2 is not None:
            raise ValueError("Provide only one of delta_v_eci_m_s or acceleration_eci_km_s2.")
        if self.delta_v_eci_m_s is not None:
            dv_m_s = np.asarray(self.delta_v_eci_m_s, dtype=float).reshape(3)
            if not np.all(np.isfinite(dv_m_s)):
                raise ValueError("delta_v_eci_m_s entries must be finite.")
            self._delta_v_km_s = dv_m_s / 1000.0
            self._accel_km_s2 = self._delta_v_km_s / float(self.duration_s)
        else:
            accel = np.asarray(self.acceleration_eci_km_s2, dtype=float).reshape(3)
            if not np.all(np.isfinite(accel)):
                raise ValueError("acceleration_eci_km_s2 entries must be finite.")
            self._accel_km_s2 = accel
            self._delta_v_km_s = accel * float(self.duration_s)

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        del belief, budget_ms
        t = float(t_s)
        start = float(self.start_time_s)
        stop = start + float(self.duration_s)
        active = start <= t < stop
        accel = self._accel_km_s2.copy() if active else np.zeros(3, dtype=float)
        return Command(
            thrust_eci_km_s2=accel,
            torque_body_nm=np.zeros(3, dtype=float),
            mode_flags={
                "mode": str(self.label if active else "coast"),
                "scheduled_impulse_active": bool(active),
                "scheduled_impulse_start_time_s": start,
                "scheduled_impulse_duration_s": float(self.duration_s),
                "scheduled_impulse_delta_v_m_s": (self._delta_v_km_s * 1000.0).tolist(),
            },
        )
