"""Reference command modules for deterministic burn sequencing and safety gates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sim.core.models import Command, StateBelief, StateTruth
from sim.mission.strategies.base import _desired_attitude_for_thrust
from sim.utils.frames import ric_dcm_ir_from_rv
from sim.utils.quaternion import quaternion_to_dcm_bn


def _vec3(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float).reshape(-1)
    if result.size != 3 or not bool(np.all(np.isfinite(result))):
        raise ValueError(f"{name} must be a finite length-3 vector.")
    return result


def _eci_vector(vector: np.ndarray, frame: str, truth: StateTruth) -> np.ndarray:
    token = str(frame).strip().lower()
    if token == "eci":
        return np.asarray(vector, dtype=float)
    if token in {"ric", "rtn", "rsw"}:
        return ric_dcm_ir_from_rv(truth.position_eci_km, truth.velocity_eci_km_s) @ vector
    raise ValueError("command frame must be 'eci' or 'ric'.")


def _output(
    *,
    intent: dict[str, Any],
    execution: str,
    phase: str,
    thrust: np.ndarray | None = None,
    torque: np.ndarray | None = None,
    desired_attitude: np.ndarray | None = None,
    mode_detail: dict[str, Any] | None = None,
    flags: dict[str, Any] | None = None,
) -> dict[str, Any]:
    thrust_vec = np.zeros(3, dtype=float) if thrust is None else np.asarray(thrust, dtype=float).reshape(3)
    torque_vec = np.zeros(3, dtype=float) if torque is None else np.asarray(torque, dtype=float).reshape(3)
    result: dict[str, Any] = {
        "mission_use_integrated_command": True,
        "thrust_eci_km_s2": thrust_vec,
        "torque_body_nm": torque_vec,
        "command_mode_flags": {"execution": execution, **dict(flags or {})},
        "mission_mode": {
            **dict(intent.get("mission_mode", {}) or {}),
            "execution": execution,
            "phase": phase,
            "burn_requested": bool(np.linalg.norm(thrust_vec) > 1.0e-15),
            **dict(mode_detail or {}),
        },
    }
    if desired_attitude is not None:
        result["desired_attitude_quat_bn"] = np.asarray(desired_attitude, dtype=float).reshape(4)
    return result


@dataclass
class TimedFiniteBurnExecution:
    start_time_s: float
    duration_s: float
    acceleration: np.ndarray
    frame: str = "eci"

    def __post_init__(self) -> None:
        if self.start_time_s < 0.0 or self.duration_s < 0.0:
            raise ValueError("start_time_s and duration_s must be non-negative.")
        self.acceleration = _vec3(self.acceleration, "acceleration")
        _eci_vector(self.acceleration, self.frame, _validation_truth())

    def update(self, *, intent: dict[str, Any], truth: StateTruth, t_s: float, **kwargs: Any) -> dict[str, Any]:
        active = self.start_time_s <= float(t_s) < self.start_time_s + self.duration_s
        complete = float(t_s) >= self.start_time_s + self.duration_s
        thrust = _eci_vector(self.acceleration, self.frame, truth) if active else np.zeros(3)
        return _output(
            intent=intent,
            execution="timed_finite_burn",
            phase="burn" if active else ("complete" if complete else "wait"),
            thrust=thrust,
            mode_detail={"start_time_s": self.start_time_s, "end_time_s": self.start_time_s + self.duration_s},
        )


@dataclass
class OneShotImpulseExecution:
    impulse_time_s: float
    delta_v_m_s: np.ndarray
    equivalent_duration_s: float = 1.0
    frame: str = "eci"
    _fired: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.impulse_time_s < 0.0 or self.equivalent_duration_s <= 0.0:
            raise ValueError("impulse_time_s must be non-negative and equivalent_duration_s positive.")
        self.delta_v_m_s = _vec3(self.delta_v_m_s, "delta_v_m_s")

    def update(self, *, intent: dict[str, Any], truth: StateTruth, t_s: float, dt_s: float = 1.0, **kwargs: Any) -> dict[str, Any]:
        interval_end = float(t_s) + max(float(dt_s), 0.0)
        active = (not self._fired) and float(t_s) <= self.impulse_time_s < interval_end + 1.0e-12
        if active:
            self._fired = True
        accel = self.delta_v_m_s / (1000.0 * self.equivalent_duration_s) if active else np.zeros(3)
        return _output(
            intent=intent,
            execution="one_shot_impulse",
            phase="fire" if active else ("complete" if self._fired else "wait"),
            thrust=_eci_vector(accel, self.frame, truth),
            mode_detail={
                "impulse_time_s": self.impulse_time_s,
                "equivalent_duration_s": self.equivalent_duration_s,
                "equivalent_finite_burn": True,
            },
        )


@dataclass
class PulseTrainExecution:
    acceleration: np.ndarray
    period_s: float
    pulse_width_s: float
    start_time_s: float = 0.0
    end_time_s: float | None = None
    phase_offset_s: float = 0.0
    frame: str = "eci"

    def __post_init__(self) -> None:
        self.acceleration = _vec3(self.acceleration, "acceleration")
        if self.period_s <= 0.0 or self.pulse_width_s < 0.0 or self.pulse_width_s > self.period_s:
            raise ValueError("period_s must be positive and pulse_width_s in [0, period_s].")
        if self.start_time_s < 0.0 or (self.end_time_s is not None and self.end_time_s < self.start_time_s):
            raise ValueError("pulse-train time bounds are invalid.")

    def update(self, *, intent: dict[str, Any], truth: StateTruth, t_s: float, **kwargs: Any) -> dict[str, Any]:
        inside = float(t_s) >= self.start_time_s and (self.end_time_s is None or float(t_s) < self.end_time_s)
        cycle = (float(t_s) - self.start_time_s - self.phase_offset_s) % self.period_s
        active = inside and cycle < self.pulse_width_s
        thrust = _eci_vector(self.acceleration, self.frame, truth) if active else np.zeros(3)
        return _output(
            intent=intent,
            execution="pulse_train",
            phase="pulse" if active else ("coast" if inside else "inactive"),
            thrust=thrust,
            mode_detail={"cycle_time_s": cycle, "period_s": self.period_s, "pulse_width_s": self.pulse_width_s},
        )


@dataclass
class SlewThenBurnExecution:
    acceleration: np.ndarray
    frame: str = "eci"
    thruster_direction_body: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    alignment_tolerance_deg: float = 5.0
    attitude_controller_budget_ms: float = 2.0

    def __post_init__(self) -> None:
        self.acceleration = _vec3(self.acceleration, "acceleration")
        self.thruster_direction_body = _vec3(self.thruster_direction_body, "thruster_direction_body")
        if np.linalg.norm(self.thruster_direction_body) <= 0.0 or self.alignment_tolerance_deg < 0.0:
            raise ValueError("thruster direction must be nonzero and tolerance non-negative.")

    def update(
        self,
        *,
        intent: dict[str, Any],
        truth: StateTruth,
        t_s: float,
        attitude_controller: Any | None = None,
        att_belief: StateBelief | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        thrust = _eci_vector(self.acceleration, self.frame, truth)
        q_des = _desired_attitude_for_thrust(
            truth=truth,
            thrust_eci_km_s2=thrust,
            thruster_direction_body=self.thruster_direction_body,
        )
        if attitude_controller is not None and hasattr(attitude_controller, "set_target"):
            attitude_controller.set_target(q_des)
        attitude_command = (
            attitude_controller.act(att_belief, float(t_s), self.attitude_controller_budget_ms)
            if attitude_controller is not None and att_belief is not None
            else Command.zero()
        )
        axis_body = self.thruster_direction_body / np.linalg.norm(self.thruster_direction_body)
        axis_eci = quaternion_to_dcm_bn(truth.attitude_quat_bn).T @ axis_body
        target_eci = -thrust / max(float(np.linalg.norm(thrust)), 1.0e-15)
        angle = float(np.arccos(np.clip(np.dot(axis_eci, target_eci), -1.0, 1.0)))
        aligned = angle <= np.deg2rad(self.alignment_tolerance_deg)
        return _output(
            intent=intent,
            execution="slew_then_burn",
            phase="burn" if aligned else "slew",
            thrust=thrust if aligned else np.zeros(3),
            torque=attitude_command.torque_body_nm,
            desired_attitude=q_des,
            mode_detail={"alignment_ok": aligned, "alignment_angle_rad": angle},
            flags={"alignment_ok": aligned, "alignment_angle_rad": angle},
        )


@dataclass
class BurnUntilConditionExecution:
    acceleration: np.ndarray
    frame: str = "eci"
    stop_time_s: float | None = None
    stop_speed_km_s: float | None = None
    max_duration_s: float | None = None
    _started_t_s: float | None = field(default=None, init=False, repr=False)
    _complete: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.acceleration = _vec3(self.acceleration, "acceleration")
        for name in ("stop_time_s", "stop_speed_km_s", "max_duration_s"):
            value = getattr(self, name)
            if value is not None and value < 0.0:
                raise ValueError(f"{name} must be non-negative when provided.")

    def update(self, *, intent: dict[str, Any], truth: StateTruth, t_s: float, **kwargs: Any) -> dict[str, Any]:
        if self._started_t_s is None:
            self._started_t_s = float(t_s)
        reasons: list[str] = []
        if self.stop_time_s is not None and float(t_s) >= self.stop_time_s:
            reasons.append("stop_time")
        if self.stop_speed_km_s is not None and np.linalg.norm(truth.velocity_eci_km_s) >= self.stop_speed_km_s:
            reasons.append("stop_speed")
        if self.max_duration_s is not None and float(t_s) - self._started_t_s >= self.max_duration_s:
            reasons.append("max_duration")
        self._complete = bool(self._complete or reasons)
        thrust = np.zeros(3) if self._complete else _eci_vector(self.acceleration, self.frame, truth)
        return _output(
            intent=intent,
            execution="burn_until_condition",
            phase="complete" if self._complete else "burn",
            thrust=thrust,
            mode_detail={"stop_reasons": reasons},
        )


@dataclass
class CoastUntilConditionExecution:
    release_time_s: float | None = None
    release_range_km: float | None = None
    target_id: str | None = None

    def update(
        self,
        *,
        intent: dict[str, Any],
        truth: StateTruth,
        t_s: float,
        own_knowledge: dict[str, StateBelief] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        released = self.release_time_s is not None and float(t_s) >= self.release_time_s
        measured_range = None
        knowledge = dict(own_knowledge or {})
        if self.target_id and self.target_id in knowledge and knowledge[self.target_id].state.size >= 3:
            measured_range = float(np.linalg.norm(knowledge[self.target_id].state[:3] - truth.position_eci_km))
            if self.release_range_km is not None and measured_range <= self.release_range_km:
                released = True
        thrust = np.asarray(intent.get("command_thrust_eci_km_s2", np.zeros(3)), dtype=float) if released else np.zeros(3)
        return _output(
            intent=intent,
            execution="coast_until_condition",
            phase="released" if released else "coast",
            thrust=thrust,
            mode_detail={"measured_range_km": measured_range},
        )


@dataclass
class WaypointSequencerExecution:
    """Run a deterministic time-based sequence of command phases."""

    phases: list[dict[str, Any]]
    _index: int = field(default=0, init=False, repr=False)
    _phase_start_t_s: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.phases:
            raise ValueError("phases must contain at least one phase.")
        for phase in self.phases:
            if float(dict(phase).get("duration_s", 0.0)) < 0.0:
                raise ValueError("phase duration_s must be non-negative.")

    def update(self, *, intent: dict[str, Any], truth: StateTruth, t_s: float, **kwargs: Any) -> dict[str, Any]:
        if self._phase_start_t_s is None:
            self._phase_start_t_s = float(t_s)
        advanced = False
        while self._index + 1 < len(self.phases):
            duration = float(dict(self.phases[self._index]).get("duration_s", 0.0))
            if float(t_s) - self._phase_start_t_s < duration:
                break
            self._phase_start_t_s += duration
            self._index += 1
            advanced = True
        phase = dict(self.phases[self._index])
        vector = _vec3(phase.get("acceleration", [0.0, 0.0, 0.0]), "phase acceleration")
        thrust = _eci_vector(vector, str(phase.get("frame", "eci")), truth)
        return _output(
            intent=intent,
            execution="waypoint_sequencer",
            phase=str(phase.get("name", f"phase_{self._index}")),
            thrust=thrust,
            mode_detail={"sequence_index": self._index, "sequence_count": len(self.phases), "phase_advanced": advanced},
        )


@dataclass
class AbortSafeHoldRetreatExecution:
    retreat_acceleration_ric_km_s2: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self) -> None:
        self.retreat_acceleration_ric_km_s2 = _vec3(
            self.retreat_acceleration_ric_km_s2, "retreat_acceleration_ric_km_s2"
        )

    def update(self, *, intent: dict[str, Any], truth: StateTruth, **kwargs: Any) -> dict[str, Any]:
        abort = bool(intent.get("abort_requested", False))
        retreat = abort and bool(intent.get("retreat_requested", False))
        thrust = (
            _eci_vector(self.retreat_acceleration_ric_km_s2, "ric", truth)
            if retreat
            else np.zeros(3, dtype=float)
        )
        return _output(
            intent=intent,
            execution="abort_safe_hold_retreat",
            phase="retreat" if retreat else ("safe_hold" if abort else "armed"),
            thrust=thrust,
            mode_detail={"abort_requested": abort, "retreat_requested": retreat},
        )


@dataclass
class FuelBudgetGateExecution:
    minimum_fuel_margin_kg: float = 0.0

    def update(self, *, intent: dict[str, Any], truth: StateTruth, dry_mass_kg: float | None = None, **kwargs: Any) -> dict[str, Any]:
        fuel_kg = None if dry_mass_kg is None else max(float(truth.mass_kg) - float(dry_mass_kg), 0.0)
        allowed = fuel_kg is None or fuel_kg >= self.minimum_fuel_margin_kg
        thrust = np.asarray(intent.get("command_thrust_eci_km_s2", np.zeros(3)), dtype=float) if allowed else np.zeros(3)
        return _output(
            intent=intent,
            execution="fuel_budget_gate",
            phase="pass" if allowed else "blocked",
            thrust=thrust,
            mode_detail={"fuel_kg": fuel_kg, "fuel_gate_allowed": allowed},
            flags={} if allowed else {"gate_reason": "fuel_budget"},
        )


@dataclass
class KeepOutGateExecution:
    """Override direct intent with an outward command inside a keep-out sphere."""

    target_id: str
    keep_out_radius_km: float
    retreat_accel_km_s2: float

    def __post_init__(self) -> None:
        if self.keep_out_radius_km <= 0.0 or self.retreat_accel_km_s2 < 0.0:
            raise ValueError("keep_out_radius_km must be positive and retreat_accel_km_s2 non-negative.")

    def update(
        self,
        *,
        intent: dict[str, Any],
        truth: StateTruth,
        own_knowledge: dict[str, StateBelief] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        target = dict(own_knowledge or {}).get(self.target_id)
        range_km = None
        active = False
        thrust = np.asarray(intent.get("command_thrust_eci_km_s2", np.zeros(3)), dtype=float).reshape(3)
        if target is not None and target.state.size >= 3:
            away = np.asarray(truth.position_eci_km, dtype=float) - np.asarray(target.state[:3], dtype=float)
            range_km = float(np.linalg.norm(away))
            active = range_km < self.keep_out_radius_km
            if active:
                direction = away / range_km if range_km > 1.0e-12 else np.array([1.0, 0.0, 0.0])
                thrust = direction * self.retreat_accel_km_s2
        return _output(
            intent=intent,
            execution="keep_out_gate",
            phase="override" if active else "pass",
            thrust=thrust,
            mode_detail={"keep_out_active": active, "range_km": range_km, "keep_out_radius_km": self.keep_out_radius_km},
            flags={"gate_reason": "keep_out_override"} if active else {},
        )


@dataclass
class CommandReplayExecution:
    rows: list[dict[str, Any]]

    def __post_init__(self) -> None:
        self.rows = sorted((dict(row) for row in self.rows), key=lambda row: float(row.get("time_s", 0.0)))
        for row in self.rows:
            _vec3(row.get("thrust_eci_km_s2", [0.0, 0.0, 0.0]), "replay thrust")

    def update(self, *, intent: dict[str, Any], t_s: float, **kwargs: Any) -> dict[str, Any]:
        selected = None
        for row in self.rows:
            if float(row.get("time_s", 0.0)) <= float(t_s):
                selected = row
            else:
                break
        thrust = np.zeros(3) if selected is None else _vec3(selected.get("thrust_eci_km_s2", [0, 0, 0]), "replay thrust")
        torque = np.zeros(3) if selected is None else _vec3(selected.get("torque_body_nm", [0, 0, 0]), "replay torque")
        return _output(
            intent=intent,
            execution="command_replay",
            phase="wait" if selected is None else "replay",
            thrust=thrust,
            torque=torque,
            mode_detail={"source_time_s": None if selected is None else float(selected.get("time_s", 0.0))},
        )


def _validation_truth() -> StateTruth:
    return StateTruth(
        position_eci_km=np.array([7000.0, 0.0, 0.0]),
        velocity_eci_km_s=np.array([0.0, 7.5, 0.0]),
        attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_rate_body_rad_s=np.zeros(3),
        mass_kg=1.0,
        t_s=0.0,
    )
