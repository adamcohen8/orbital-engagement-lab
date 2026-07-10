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
    _interval_end_t_s: float | None = field(default=None, init=False, repr=False)

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

    def set_actuation_interval(self, start_t_s: float, end_t_s: float) -> None:
        """Provide the integration interval over which the next command applies."""

        start = float(start_t_s)
        end = float(end_t_s)
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            raise ValueError("Scheduled impulse actuation interval must be finite and increasing.")
        self._interval_end_t_s = end

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        del belief, budget_ms
        t = float(t_s)
        start = float(self.start_time_s)
        stop = start + float(self.duration_s)
        interval_end = self._interval_end_t_s
        self._interval_end_t_s = None
        if interval_end is None:
            interval_end = t
        interval_duration = max(float(interval_end) - t, 0.0)
        overlap_s = max(0.0, min(float(interval_end), stop) - max(t, start))
        active = overlap_s > 0.0 or (interval_duration <= 0.0 and start <= t < stop)
        if interval_duration > 0.0:
            accel = self._accel_km_s2 * (overlap_s / interval_duration)
        else:
            accel = self._accel_km_s2.copy() if active else np.zeros(3, dtype=float)
        return Command(
            thrust_eci_km_s2=accel,
            torque_body_nm=np.zeros(3, dtype=float),
            mode_flags={
                "mode": str(self.label if active else "coast"),
                "scheduled_impulse_active": bool(active),
                "scheduled_impulse_start_time_s": start,
                "scheduled_impulse_duration_s": float(self.duration_s),
                "scheduled_impulse_interval_overlap_s": float(overlap_s),
                "scheduled_impulse_delta_v_m_s": (self._delta_v_km_s * 1000.0).tolist(),
            },
        )
