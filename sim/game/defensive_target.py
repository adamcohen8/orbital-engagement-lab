from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sim.core.models import StateTruth
from sim.utils.frames import eci_relative_to_ric_rect, ric_dcm_ir_from_rv


@dataclass
class DefensiveTargetIntentProvider:
    chaser_object_id: str = "chaser"
    trigger_range_km: float = 1.2
    trigger_closing_speed_km_s: float = 0.00025
    keepout_radius_km: float = 0.25
    max_accel_km_s2: float = 7.5e-6
    max_delta_v_m_s: float | None = None
    cross_track_bias: float = 0.65
    pulse_period_s: float = 120.0
    _used_delta_v_m_s: float = 0.0
    _last_t_s: float | None = None

    def __call__(
        self,
        *,
        truth: StateTruth,
        t_s: float,
        world_truth: dict[str, StateTruth] | None = None,
        own_knowledge: dict[str, Any] | None = None,
        dt_s: float | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        dt_s = self._dt_since_last_call(t_s, fallback_dt_s=dt_s)
        chaser_state = dict(own_knowledge or {}).get(str(self.chaser_object_id))
        if chaser_state is None or getattr(chaser_state, "state", np.array([])).size < 6:
            return self._inactive_command()
        rel = eci_relative_to_ric_rect(
            np.array(chaser_state.state[:6], dtype=float),
            np.hstack((truth.position_eci_km, truth.velocity_eci_km_s)),
        )
        r = rel[:3]
        v = rel[3:]
        rng = float(np.linalg.norm(r))
        if not np.isfinite(rng) or rng <= 0.0:
            return self._inactive_command()
        closing_speed = max(-float(np.dot(r, v)) / max(rng, 1.0e-9), 0.0)
        active = rng < float(self.trigger_range_km) or closing_speed > float(self.trigger_closing_speed_km_s)
        if not active:
            return self._inactive_command()

        away_from_chaser = -r / rng
        cross_sign = 1.0 if int(float(t_s) // max(float(self.pulse_period_s), 1.0e-9)) % 2 == 0 else -1.0
        cross = np.array([0.0, 0.0, cross_sign], dtype=float)
        if rng < max(float(self.keepout_radius_km) * 2.0, 1.0e-9):
            direction_ric = away_from_chaser
        else:
            direction_ric = away_from_chaser + float(self.cross_track_bias) * cross
        nrm = float(np.linalg.norm(direction_ric))
        if nrm <= 0.0 or not np.isfinite(nrm):
            return self._inactive_command()
        accel_mag_km_s2 = self._budgeted_accel_km_s2(dt_s)
        if accel_mag_km_s2 <= 0.0:
            return self._inactive_command(budget_exhausted=True)
        accel_ric = direction_ric / nrm * accel_mag_km_s2
        c_ir = ric_dcm_ir_from_rv(truth.position_eci_km, truth.velocity_eci_km_s)
        accel_eci = c_ir @ accel_ric
        self._used_delta_v_m_s += float(np.linalg.norm(accel_ric)) * max(dt_s, 0.0) * 1000.0
        return _target_command(
            accel_eci,
            active=True,
            accel_ric=accel_ric,
            closing_speed_km_s=closing_speed,
            used_delta_v_m_s=self._used_delta_v_m_s,
            max_delta_v_m_s=self.max_delta_v_m_s,
        )

    @property
    def used_delta_v_m_s(self) -> float:
        return float(self._used_delta_v_m_s)

    def _dt_since_last_call(self, t_s: float, *, fallback_dt_s: float | None = None) -> float:
        t = float(t_s)
        if self._last_t_s is None:
            self._last_t_s = t
            return max(float(fallback_dt_s), 0.0) if fallback_dt_s is not None else 0.0
        dt = max(t - float(self._last_t_s), 0.0)
        self._last_t_s = t
        return dt

    def _budgeted_accel_km_s2(self, dt_s: float) -> float:
        accel = max(float(self.max_accel_km_s2), 0.0)
        if self.max_delta_v_m_s is None:
            return accel
        remaining_m_s = max(float(self.max_delta_v_m_s) - float(self._used_delta_v_m_s), 0.0)
        if remaining_m_s <= 0.0:
            return 0.0
        if dt_s <= 0.0:
            return accel
        return min(accel, remaining_m_s / max(dt_s, 1.0e-9) / 1000.0)

    def _inactive_command(self, *, budget_exhausted: bool = False) -> dict[str, Any]:
        return _target_command(
            np.zeros(3, dtype=float),
            active=False,
            used_delta_v_m_s=self._used_delta_v_m_s,
            max_delta_v_m_s=self.max_delta_v_m_s,
            budget_exhausted=budget_exhausted,
        )


def _target_command(
    accel_eci_km_s2: np.ndarray,
    *,
    active: bool,
    accel_ric: np.ndarray | None = None,
    closing_speed_km_s: float = 0.0,
    used_delta_v_m_s: float = 0.0,
    max_delta_v_m_s: float | None = None,
    budget_exhausted: bool = False,
) -> dict[str, Any]:
    accel = np.array(accel_eci_km_s2, dtype=float).reshape(3)
    accel_ric_arr = np.zeros(3, dtype=float) if accel_ric is None else np.array(accel_ric, dtype=float).reshape(3)
    return {
        "thrust_eci_km_s2": accel,
        "mission_mode": {
            "strategy": "defensive_target",
            "active": bool(active),
            "closing_speed_km_s": float(closing_speed_km_s),
            "used_delta_v_m_s": float(used_delta_v_m_s),
            "max_delta_v_m_s": None if max_delta_v_m_s is None else float(max_delta_v_m_s),
            "budget_exhausted": bool(budget_exhausted),
        },
        "command_mode_flags": {
            "target_defensive": bool(active),
            "target_defensive_accel_ric_km_s2": accel_ric_arr.tolist(),
            "target_defensive_used_delta_v_m_s": float(used_delta_v_m_s),
            "target_defensive_max_delta_v_m_s": None if max_delta_v_m_s is None else float(max_delta_v_m_s),
            "target_defensive_budget_exhausted": bool(budget_exhausted),
        },
    }
