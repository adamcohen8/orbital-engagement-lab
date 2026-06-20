from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.core.interfaces import Controller
from sim.core.models import Command, StateBelief
from sim.utils.quaternion import normalize_quaternion


def _quat_slerp(q_from: np.ndarray, q_to: np.ndarray, alpha: float) -> np.ndarray:
    qa = normalize_quaternion(np.array(q_from, dtype=float).reshape(4))
    qb = normalize_quaternion(np.array(q_to, dtype=float).reshape(4))
    a = float(np.clip(alpha, 0.0, 1.0))
    dot = float(np.dot(qa, qb))
    if dot < 0.0:
        qb = -qb
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return normalize_quaternion((1.0 - a) * qa + a * qb)
    theta = float(np.arccos(dot))
    denom = float(np.sin(theta))
    if abs(denom) <= 1.0e-15:
        return qa
    w0 = float(np.sin((1.0 - a) * theta) / denom)
    w1 = float(np.sin(a * theta) / denom)
    return normalize_quaternion(w0 * qa + w1 * qb)


@dataclass
class AttitudeReplayController(Controller):
    """Replay a time-tagged attitude history through OEL's attitude override hook."""

    times_s: list[float] | np.ndarray
    attitude_quat_bn: list[list[float]] | np.ndarray
    angular_rate_body_rad_s: list[list[float]] | np.ndarray | None = None
    hold_outside_range: bool = True
    _times: np.ndarray = field(init=False, repr=False)
    _q: np.ndarray = field(init=False, repr=False)
    _w: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        times = np.array(self.times_s, dtype=float).reshape(-1)
        q = np.array(self.attitude_quat_bn, dtype=float)
        if times.size == 0:
            raise ValueError("AttitudeReplayController requires at least one sample.")
        if q.shape != (times.size, 4):
            raise ValueError("attitude_quat_bn must be an Nx4 array matching times_s.")
        if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
            raise ValueError("times_s must be finite and strictly increasing.")
        q = np.vstack([normalize_quaternion(row) for row in q])
        if self.angular_rate_body_rad_s is None:
            w = np.zeros((times.size, 3), dtype=float)
        else:
            w = np.array(self.angular_rate_body_rad_s, dtype=float)
            if w.shape != (times.size, 3):
                raise ValueError("angular_rate_body_rad_s must be an Nx3 array matching times_s.")
            if not np.all(np.isfinite(w)):
                raise ValueError("angular_rate_body_rad_s must be finite.")
        self._times = times
        self._q = q
        self._w = w

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        q, w = self._sample(float(t_s))
        return Command(
            thrust_eci_km_s2=np.zeros(3),
            torque_body_nm=np.zeros(3),
            mode_flags={
                "mode": "attitude_replay",
                "attitude_state_override": {
                    "q_next_bn": q.tolist(),
                    "w_next_body_rad_s": w.tolist(),
                    "phase": "replay",
                },
            },
        )

    def _sample(self, t_s: float) -> tuple[np.ndarray, np.ndarray]:
        times = self._times
        if t_s <= float(times[0]):
            if not self.hold_outside_range and t_s < float(times[0]):
                raise ValueError("requested attitude replay time before first sample.")
            return self._q[0].copy(), self._w[0].copy()
        if t_s >= float(times[-1]):
            if not self.hold_outside_range and t_s > float(times[-1]):
                raise ValueError("requested attitude replay time after last sample.")
            return self._q[-1].copy(), self._w[-1].copy()
        idx = int(np.searchsorted(times, t_s, side="right") - 1)
        idx = max(0, min(idx, times.size - 2))
        t0 = float(times[idx])
        t1 = float(times[idx + 1])
        alpha = 0.0 if t1 <= t0 else (float(t_s) - t0) / (t1 - t0)
        q = _quat_slerp(self._q[idx], self._q[idx + 1], alpha)
        w = (1.0 - alpha) * self._w[idx] + alpha * self._w[idx + 1]
        return q, w
