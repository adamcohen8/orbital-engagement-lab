from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sim.core.interfaces import Controller
from sim.core.models import Command, StateBelief
from sim.utils.frames import ric_curv_to_rect, ric_dcm_ir_from_rv

EARTH_MU_KM3_S2 = 398600.4418
_RADIAL_CENTER_TO_DRIFT_FACTOR = 3.0


def _finite_float(value: Any, name: str) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite.")
    return out


def _nonnegative_float(value: Any, name: str) -> float:
    out = _finite_float(value, name)
    if out < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return out


def _distance_km(config: dict[str, Any], key: str, default_km: float) -> float:
    if f"{key}_km" in config:
        return _finite_float(config[f"{key}_km"], f"{key}_km")
    if f"{key}_m" in config:
        return _finite_float(config[f"{key}_m"], f"{key}_m") / 1000.0
    if key in config:
        return _finite_float(config[key], key)
    return float(default_km)


def _rate_km_s(config: dict[str, Any], key: str, default_km_s: float) -> float:
    if f"{key}_km_s" in config:
        return _finite_float(config[f"{key}_km_s"], f"{key}_km_s")
    if f"{key}_m_s" in config:
        return _finite_float(config[f"{key}_m_s"], f"{key}_m_s") / 1000.0
    if key in config:
        return _finite_float(config[key], key)
    return float(default_km_s)


def _slice_pair(value: tuple[int, int] | list[int], name: str) -> tuple[int, int]:
    raw = tuple(int(x) for x in value)
    if len(raw) != 2 or raw[1] - raw[0] != 6:
        raise ValueError(f"{name} must select exactly 6 elements.")
    return raw


def _sign_or_zero(value: float, eps: float = 1.0e-12) -> float:
    if value > eps:
        return 1.0
    if value < -eps:
        return -1.0
    return 0.0


def estimate_rmoes_from_rect_ric(relative_ric_rect: np.ndarray, mean_motion_rad_s: float) -> dict[str, float]:
    """Estimate controller RMOEs from a rectangular RIC relative state.

    The definitions match the game's NMT convention where bounded R-I motion
    satisfies ``I_dot + 2 n R = 0`` for zero radial center. The radial center is
    the HCW constant that drives secular in-track drift.
    """

    rel = np.array(relative_ric_rect, dtype=float).reshape(6)
    n = _finite_float(mean_motion_rad_s, "mean_motion_rad_s")
    if abs(n) <= 1.0e-12:
        raise ValueError("mean_motion_rad_s must be non-zero.")

    radial, in_track, cross_track, radial_rate, in_track_rate, cross_track_rate = rel
    radial_center = -(in_track_rate + 2.0 * n * radial) / n
    in_track_drift_rate = -_RADIAL_CENTER_TO_DRIFT_FACTOR * n * radial_center
    radial_osc = radial - radial_center
    radial_amplitude = float(np.sqrt(radial_osc * radial_osc + (radial_rate / n) ** 2))
    cross_track_amplitude = float(np.sqrt(cross_track * cross_track + (cross_track_rate / n) ** 2))
    in_track_center = in_track - 2.0 * radial_rate / n
    return {
        "radial_center_km": float(radial_center),
        "in_track_center_km": float(in_track_center),
        "in_track_drift_rate_km_s": float(in_track_drift_rate),
        "radial_amplitude_km": radial_amplitude,
        "cross_track_amplitude_km": cross_track_amplitude,
        "cross_track_phase_rad": float(np.arctan2(-cross_track_rate / n, cross_track)),
        "ri_phase_rad": float(np.arctan2(-radial_rate / n, radial_osc)),
    }


@dataclass
class RMOEIfThenController(Controller):
    """Rule-based continuous orbit controller for target RMOEs."""

    max_accel_km_s2: float
    mean_motion_rad_s: float = 0.0
    target: dict[str, Any] = field(default_factory=dict)
    tolerances: dict[str, Any] = field(default_factory=dict)
    gains: dict[str, Any] = field(default_factory=dict)
    max_drift_rate_km_s: float | None = None
    max_drift_rate_m_s: float | None = None
    close_zone_km: float | None = None
    close_zone_m: float | None = None
    cross_track_burn_gate_km: float | None = None
    cross_track_burn_gate_m: float | None = None
    ric_curv_state_slice: tuple[int, int] = (0, 6)
    chief_eci_state_slice: tuple[int, int] = (6, 12)

    def __post_init__(self) -> None:
        self.max_accel_km_s2 = _nonnegative_float(self.max_accel_km_s2, "max_accel_km_s2")
        self.mean_motion_rad_s = _nonnegative_float(self.mean_motion_rad_s, "mean_motion_rad_s")
        self.ric_curv_state_slice = _slice_pair(self.ric_curv_state_slice, "ric_curv_state_slice")
        self.chief_eci_state_slice = _slice_pair(self.chief_eci_state_slice, "chief_eci_state_slice")
        self.target = dict(self.target or {})
        self.tolerances = dict(self.tolerances or {})
        self.gains = dict(self.gains or {})

        if self.max_drift_rate_km_s is None:
            self.max_drift_rate_km_s = (
                _finite_float(self.max_drift_rate_m_s, "max_drift_rate_m_s") / 1000.0
                if self.max_drift_rate_m_s is not None
                else 2.0e-5
            )
        self.max_drift_rate_km_s = _nonnegative_float(self.max_drift_rate_km_s, "max_drift_rate_km_s")

        if self.close_zone_km is None:
            self.close_zone_km = (
                _finite_float(self.close_zone_m, "close_zone_m") / 1000.0 if self.close_zone_m is not None else 0.05
            )
        self.close_zone_km = _nonnegative_float(self.close_zone_km, "close_zone_km")

        if self.cross_track_burn_gate_km is None:
            self.cross_track_burn_gate_km = (
                _finite_float(self.cross_track_burn_gate_m, "cross_track_burn_gate_m") / 1000.0
                if self.cross_track_burn_gate_m is not None
                else 0.05
            )
        self.cross_track_burn_gate_km = _nonnegative_float(self.cross_track_burn_gate_km, "cross_track_burn_gate_km")

    def linear_system_summary(self) -> dict[str, object]:
        return {
            "system_type": "rmoe_if_then_feedback",
            "law_label": "priority if-then RMOE targeting",
            "control_axes": ["R", "I", "C"],
            "state_labels": [
                "radial_center",
                "in_track_center",
                "in_track_drift_rate",
                "cross_track_amplitude",
            ],
            "target": self._target_values(),
        }

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        i0, i1 = self.ric_curv_state_slice
        j0, j1 = self.chief_eci_state_slice
        if belief.state.size < max(i1, j1):
            return Command.zero()

        x_curv = np.array(belief.state[i0:i1], dtype=float)
        chief_eci = np.array(belief.state[j0:j1], dtype=float)
        r_chief = chief_eci[:3]
        v_chief = chief_eci[3:]
        r0 = float(np.linalg.norm(r_chief))
        if r0 <= 0.0:
            return Command.zero()

        n = self._mean_motion(r0)
        if n <= 1.0e-12:
            return Command.zero()

        x_rect = ric_curv_to_rect(x_curv, r0_km=r0)
        rmoes = estimate_rmoes_from_rect_ric(x_rect, n)
        target = self._target_values()
        tol = self._tolerance_values()
        gain = self._gain_values()

        accel_ric_pre_limit = np.zeros(3, dtype=float)
        mode = "coast"
        reason = "within_tolerance"
        active_modes: list[str] = []

        drift = rmoes["in_track_drift_rate_km_s"]
        drift_target = self._desired_drift_rate(x_rect, target)
        drift_error = drift_target - drift
        radial_center_error = target["radial_center_km"] - rmoes["radial_center_km"]
        cross_amp_error = target["cross_track_amplitude_km"] - rmoes["cross_track_amplitude_km"]
        in_track_center_error = target["in_track_center_km"] - rmoes["in_track_center_km"]

        if abs(drift) > float(self.max_drift_rate_km_s) + tol["drift_rate_km_s"]:
            desired = float(np.clip(drift, -float(self.max_drift_rate_km_s), float(self.max_drift_rate_km_s)))
            accel_ric_pre_limit[1] = self._in_track_accel_for_drift(desired, rmoes, gain, n)
            mode = "limit_drift"
            reason = "max_drift_rate_exceeded"
            active_modes.append(mode)
        elif abs(drift_error) > tol["drift_rate_km_s"]:
            accel_ric_pre_limit[1] = self._in_track_accel_for_drift(drift_target, rmoes, gain, n)
            mode = "shape_drift"
            reason = "drift_rate_error"
            active_modes.append(mode)
        elif abs(radial_center_error) > tol["radial_center_km"]:
            accel_ric_pre_limit[1] = -gain["radial_center"] * radial_center_error
            mode = "trim_radial_center"
            reason = "radial_center_error"
            active_modes.append(mode)

        if (
            mode not in {"limit_drift", "shape_drift"}
            and self._at_cross_track_burn_gate(x_rect)
            and abs(cross_amp_error) > tol["cross_track_amplitude_km"]
        ):
            cross_rate_sign = _sign_or_zero(float(x_rect[5]))
            if cross_rate_sign == 0.0:
                cross_rate_sign = -_sign_or_zero(float(x_rect[2]))
            accel_ric_pre_limit[2] = gain["cross_track_amplitude"] * cross_amp_error * cross_rate_sign
            active_modes.append("trim_cross_track_amplitude")
            if mode == "coast":
                mode = "trim_cross_track_amplitude"
                reason = "cross_track_amplitude_error_at_c_zero"

        if abs(in_track_center_error) > tol["in_track_center_km"]:
            accel_ric_pre_limit[0] = -gain["in_track_center"] * in_track_center_error
            active_modes.append("trim_in_track_center")
            if mode == "coast":
                mode = "trim_in_track_center"
                reason = "in_track_center_error"

        accel_ric, limit_scale = self._limited(accel_ric_pre_limit)
        accel_eci = ric_dcm_ir_from_rv(r_chief, v_chief) @ accel_ric

        return Command(
            thrust_eci_km_s2=accel_eci,
            torque_body_nm=np.zeros(3),
            mode_flags={
                "mode": "rmoe_if_then",
                "rmoe_mode": mode,
                "rmoe_active_modes": active_modes,
                "rmoe_reason": reason,
                "rmoes": rmoes,
                "target_rmoes": target,
                "tolerances": tol,
                "desired_drift_rate_km_s": drift_target,
                "relative_ric_rect_km": x_rect.tolist(),
                "accel_ric_pre_limit_km_s2": accel_ric_pre_limit.tolist(),
                "accel_ric_km_s2": accel_ric.tolist(),
                "accel_eci_km_s2": accel_eci.tolist(),
                "limit_scale": limit_scale,
                "ric_curv_state_slice": [i0, i1],
                "chief_eci_state_slice": [j0, j1],
            },
        )

    def _mean_motion(self, chief_radius_km: float) -> float:
        if self.mean_motion_rad_s > 0.0:
            return float(self.mean_motion_rad_s)
        return float(np.sqrt(EARTH_MU_KM3_S2 / max(float(chief_radius_km), 1.0e-9) ** 3))

    def _target_values(self) -> dict[str, float]:
        return {
            "radial_center_km": _distance_km(self.target, "radial_center", 0.0),
            "in_track_center_km": _distance_km(self.target, "in_track_center", 0.0),
            "in_track_drift_rate_km_s": _rate_km_s(self.target, "in_track_drift_rate", 0.0),
            "cross_track_amplitude_km": abs(_distance_km(self.target, "cross_track_amplitude", 0.0)),
            "cross_track_phase_deg": _finite_float(self.target.get("cross_track_phase_deg", 0.0), "cross_track_phase_deg"),
            "ri_phase_deg": _finite_float(self.target.get("ri_phase_deg", 0.0), "ri_phase_deg"),
        }

    def _tolerance_values(self) -> dict[str, float]:
        return {
            "radial_center_km": _nonnegative_float(
                _distance_km(self.tolerances, "radial_center", 0.01), "tolerances.radial_center_km"
            ),
            "in_track_center_km": _nonnegative_float(
                _distance_km(self.tolerances, "in_track_center", 0.05), "tolerances.in_track_center_km"
            ),
            "drift_rate_km_s": _nonnegative_float(
                _rate_km_s(self.tolerances, "in_track_drift_rate", 1.0e-6),
                "tolerances.in_track_drift_rate_km_s",
            ),
            "cross_track_amplitude_km": _nonnegative_float(
                _distance_km(self.tolerances, "cross_track_amplitude", 0.02),
                "tolerances.cross_track_amplitude_km",
            ),
        }

    def _gain_values(self) -> dict[str, float]:
        return {
            "drift": _nonnegative_float(self.gains.get("drift", 2.0e-3), "gains.drift"),
            "radial_center": _nonnegative_float(self.gains.get("radial_center", 1.0e-6), "gains.radial_center"),
            "in_track_center": _nonnegative_float(self.gains.get("in_track_center", 1.0e-6), "gains.in_track_center"),
            "cross_track_amplitude": _nonnegative_float(
                self.gains.get("cross_track_amplitude", 1.0e-6), "gains.cross_track_amplitude"
            ),
        }

    def _desired_drift_rate(self, x_rect: np.ndarray, target: dict[str, float]) -> float:
        in_track_error = float(x_rect[1]) - target["in_track_center_km"]
        close_zone = float(self.close_zone_km)
        if close_zone > 0.0 and abs(in_track_error) <= close_zone:
            scale = float(np.clip(in_track_error / close_zone, -1.0, 1.0))
            return target["in_track_drift_rate_km_s"] - float(self.max_drift_rate_km_s) * scale
        if in_track_error > 0.0:
            return target["in_track_drift_rate_km_s"] - float(self.max_drift_rate_km_s)
        return target["in_track_drift_rate_km_s"] + float(self.max_drift_rate_km_s)

    def _in_track_accel_for_drift(
        self,
        desired_drift_km_s: float,
        rmoes: dict[str, float],
        gain: dict[str, float],
        mean_motion_rad_s: float,
    ) -> float:
        desired_radial_center = -float(desired_drift_km_s) / (
            _RADIAL_CENTER_TO_DRIFT_FACTOR * max(float(mean_motion_rad_s), 1.0e-12)
        )
        radial_center_error = desired_radial_center - float(rmoes["radial_center_km"])
        return -gain["drift"] * radial_center_error

    def _at_cross_track_burn_gate(self, x_rect: np.ndarray) -> bool:
        return abs(float(x_rect[2])) <= float(self.cross_track_burn_gate_km)

    def _limited(self, accel_ric_pre_limit: np.ndarray) -> tuple[np.ndarray, float]:
        accel_ric = np.array(accel_ric_pre_limit, dtype=float)
        nrm = float(np.linalg.norm(accel_ric))
        if self.max_accel_km_s2 == 0.0:
            accel_ric[:] = 0.0
            return accel_ric, 0.0
        if nrm > self.max_accel_km_s2:
            scale = float(self.max_accel_km_s2 / nrm)
            accel_ric *= scale
            return accel_ric, scale
        return accel_ric, 1.0
