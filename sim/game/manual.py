from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sim.core.models import StateTruth
from sim.utils.frames import ric_dcm_ir_from_rv
from sim.utils.quaternion import normalize_quaternion, quaternion_delta_from_body_rate, quaternion_multiply


@dataclass
class KeyboardCommandState:
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    firing: bool = False
    throttle: float = 1.0
    reset_requested: bool = False
    restart_requested: bool = False
    paused: bool = False
    step_requested: bool = False
    speed_multiplier_change: int = 0
    quit_requested: bool = False

    def reset_axes(self) -> None:
        self.pitch = 0.0
        self.yaw = 0.0
        self.roll = 0.0
        self.firing = False


@dataclass
class ManualGameCommandProvider:
    command_state: KeyboardCommandState
    max_accel_km_s2: float = 2.0e-5
    attitude_rate_deg_s: float = 8.0
    controlled_object_id: str = "chaser"
    control_mode: str = "attitude_thrust"
    reference_object_id: str = "target"
    _desired_attitude_quat_bn: np.ndarray | None = field(default=None, init=False, repr=False)
    _last_update_t_s: float | None = field(default=None, init=False, repr=False)

    @property
    def desired_attitude_quat_bn(self) -> np.ndarray | None:
        if self._desired_attitude_quat_bn is None:
            return None
        return np.array(self._desired_attitude_quat_bn, dtype=float)

    def reset_target_to_current(self, truth: StateTruth | np.ndarray) -> None:
        if isinstance(truth, StateTruth):
            q = np.array(truth.attitude_quat_bn, dtype=float)
        else:
            q = np.array(truth, dtype=float).reshape(-1)[6:10]
        self._desired_attitude_quat_bn = normalize_quaternion(q)
        self.command_state.reset_requested = False

    def _integrate_target(self, truth: StateTruth, t_s: float, dt_s: float) -> np.ndarray:
        if self._desired_attitude_quat_bn is None or self.command_state.reset_requested:
            self.reset_target_to_current(truth)
        assert self._desired_attitude_quat_bn is not None

        if self._last_update_t_s is None:
            dt = float(max(dt_s, 0.0))
        else:
            dt = float(max(float(t_s) - float(self._last_update_t_s), 0.0))
            if dt <= 0.0:
                dt = float(max(dt_s, 0.0))
        self._last_update_t_s = float(t_s)

        rate = np.deg2rad(float(max(self.attitude_rate_deg_s, 0.0)))
        body_rate_cmd = rate * np.array(
            [
                float(np.clip(self.command_state.roll, -1.0, 1.0)),
                float(np.clip(self.command_state.pitch, -1.0, 1.0)),
                float(np.clip(self.command_state.yaw, -1.0, 1.0)),
            ],
            dtype=float,
        )
        if float(np.linalg.norm(body_rate_cmd)) > 0.0 and dt > 0.0:
            dq = quaternion_delta_from_body_rate(body_rate_cmd, dt)
            self._desired_attitude_quat_bn = normalize_quaternion(quaternion_multiply(self._desired_attitude_quat_bn, dq))
        return np.array(self._desired_attitude_quat_bn, dtype=float)

    def __call__(
        self,
        *,
        truth: StateTruth,
        t_s: float,
        dt_s: float,
        object_id: str | None = None,
        world_truth: dict[str, StateTruth] | None = None,
        own_knowledge: dict[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        if object_id is not None and str(object_id) != str(self.controlled_object_id):
            return {}
        mode = str(self.control_mode or "attitude_thrust").strip().lower()
        if mode in {"ric", "ric_translation", "translation"}:
            throttle = float(np.clip(self.command_state.throttle, 0.0, 1.0))
            accel_ric = np.array(
                [
                    float(np.clip(self.command_state.pitch, -1.0, 1.0)),
                    float(np.clip(self.command_state.yaw, -1.0, 1.0)),
                    float(np.clip(self.command_state.roll, -1.0, 1.0)),
                ],
                dtype=float,
            )
            nrm = float(np.linalg.norm(accel_ric))
            if nrm > 1.0:
                accel_ric /= nrm
            accel_ric *= float(max(self.max_accel_km_s2, 0.0)) * throttle
            ref = dict(own_knowledge or {}).get(str(self.reference_object_id))
            if ref is not None and getattr(ref, "state", np.array([])).size >= 6:
                ref_state = np.array(ref.state[:6], dtype=float)
                c_ir = ric_dcm_ir_from_rv(ref_state[:3], ref_state[3:6])
                thrust_eci = c_ir @ accel_ric
            else:
                thrust_eci = accel_ric
            return {
                "thrust_eci_km_s2": thrust_eci,
                "mission_mode": {
                    "strategy": "manual_ric_translation",
                    "throttle": throttle,
                },
                "command_mode_flags": {
                    "player_controlled": True,
                    "player_control_mode": "ric_translation",
                    "player_throttle": throttle,
                    "player_accel_ric_km_s2": accel_ric.tolist(),
                },
            }
        q_cmd = self._integrate_target(truth=truth, t_s=float(t_s), dt_s=float(dt_s))
        throttle = float(np.clip(self.command_state.throttle, 0.0, 1.0))
        accel_mag = float(max(self.max_accel_km_s2, 0.0)) * throttle if self.command_state.firing else 0.0
        return {
            "desired_attitude_quat_bn": q_cmd,
            # The simulator later replaces this placeholder direction with the
            # body-mounted thruster force direction while preserving magnitude.
            "thrust_eci_km_s2": np.array([accel_mag, 0.0, 0.0], dtype=float),
            "mission_mode": {
                "strategy": "manual_game",
                "firing": bool(self.command_state.firing),
                "throttle": throttle,
            },
            "command_mode_flags": {
                "player_controlled": True,
                "player_firing": bool(self.command_state.firing),
                "player_throttle": throttle,
            },
        }
