from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sim.core.models import StateBelief, StateTruth
from sim.dynamics.orbit.cr3bp import cr3bp_moon_state_km_s
from sim.utils.frames import ric_dcm_ir_from_rv
from sim.utils.quaternion import normalize_quaternion, quaternion_delta_from_body_rate, quaternion_multiply

CISLUNAR_TRANSLATION_MODES = {"cislunar", "cislunar_translation", "cr3bp", "cr3bp_translation"}
MOON_RIC_TRANSLATION_MODES = {"moon_ric", "moon_ric_translation", "lunar_ric", "lunar_ric_translation"}
TRANSLATION_CONTROL_MODES = {"ric", "ric_translation", "translation", *CISLUNAR_TRANSLATION_MODES, *MOON_RIC_TRANSLATION_MODES}


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
    speed_multiplier_change: int = 0
    camera_rule_toggle_requested: bool = False
    music_toggle_requested: bool = False
    clip_record_toggle_requested: bool = False
    clip_record_save_requested: bool = False
    open_debrief_requested: bool = False
    briefing_scroll_px: int = 0
    quit_requested: bool = False
    use_timing_accumulator: bool = False
    pitch_sim_s: float = 0.0
    yaw_sim_s: float = 0.0
    roll_sim_s: float = 0.0
    firing_sim_s: float = 0.0

    def reset_axes(self) -> None:
        self.pitch = 0.0
        self.yaw = 0.0
        self.roll = 0.0
        self.firing = False
        self.clear_timed_input()

    def clear_timed_input(self) -> None:
        self.pitch_sim_s = 0.0
        self.yaw_sim_s = 0.0
        self.roll_sim_s = 0.0
        self.firing_sim_s = 0.0

    def accumulate_timed_input(
        self,
        wall_dt_s: float,
        *,
        speed_multiple: float,
        control_mode: str = "attitude_thrust",
    ) -> None:
        if not bool(self.use_timing_accumulator):
            return
        elapsed_sim_s = float(max(wall_dt_s, 0.0)) * float(max(speed_multiple, 0.0))
        if elapsed_sim_s <= 0.0:
            return
        mode = str(control_mode or "").strip().lower()
        if mode in TRANSLATION_CONTROL_MODES:
            if float(self.throttle) <= 0.0:
                return
            self.pitch_sim_s += float(np.clip(self.pitch, -1.0, 1.0)) * elapsed_sim_s
            self.yaw_sim_s += float(np.clip(self.yaw, -1.0, 1.0)) * elapsed_sim_s
            self.roll_sim_s += float(np.clip(self.roll, -1.0, 1.0)) * elapsed_sim_s
        elif bool(self.firing):
            self.firing_sim_s += elapsed_sim_s

    def consume_ric_duty_cycle(self, dt_s: float) -> np.ndarray:
        if not bool(self.use_timing_accumulator):
            return np.array(
                [
                    float(np.clip(self.pitch, -1.0, 1.0)),
                    float(np.clip(self.yaw, -1.0, 1.0)),
                    float(np.clip(self.roll, -1.0, 1.0)),
                ],
                dtype=float,
            )
        dt = float(max(dt_s, 1.0e-9))
        values = [self.pitch_sim_s, self.yaw_sim_s, self.roll_sim_s]
        consumed: list[float] = []
        remaining: list[float] = []
        for value in values:
            amount = np.sign(float(value)) * min(abs(float(value)), dt)
            consumed.append(float(amount) / dt)
            remaining.append(float(value) - float(amount))
        self.pitch_sim_s, self.yaw_sim_s, self.roll_sim_s = remaining
        return np.array(consumed, dtype=float)

    def consume_firing_duty_cycle(self, dt_s: float) -> float:
        if not bool(self.use_timing_accumulator):
            return 1.0 if bool(self.firing) else 0.0
        dt = float(max(dt_s, 1.0e-9))
        amount = min(max(float(self.firing_sim_s), 0.0), dt)
        self.firing_sim_s = max(float(self.firing_sim_s) - amount, 0.0)
        return float(amount) / dt


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
            self._desired_attitude_quat_bn = normalize_quaternion(
                quaternion_multiply(self._desired_attitude_quat_bn, dq)
            )
        return np.array(self._desired_attitude_quat_bn, dtype=float)

    def __call__(
        self,
        *,
        truth: StateTruth,
        t_s: float,
        dt_s: float,
        object_id: str | None = None,
        own_knowledge: dict[str, StateBelief] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        if object_id is not None and str(object_id) != str(self.controlled_object_id):
            return {}
        mode = str(self.control_mode or "attitude_thrust").strip().lower()
        if mode in TRANSLATION_CONTROL_MODES:
            throttle = float(np.clip(self.command_state.throttle, 0.0, 1.0))
            accel_ric = self.command_state.consume_ric_duty_cycle(float(dt_s))
            nrm = float(np.linalg.norm(accel_ric))
            if nrm > 1.0:
                accel_ric /= nrm
            accel_ric *= float(max(self.max_accel_km_s2, 0.0)) * throttle
            if mode in CISLUNAR_TRANSLATION_MODES:
                thrust_eci = accel_ric
                strategy = "manual_cislunar_translation"
                player_control_mode = "cislunar_translation"
            else:
                ref_state = _reference_state6(
                    reference_object_id=self.reference_object_id,
                    own_knowledge=own_knowledge,
                )
                if ref_state is not None:
                    if mode in MOON_RIC_TRANSLATION_MODES:
                        ref_state = ref_state - cr3bp_moon_state_km_s()
                    c_ir = ric_dcm_ir_from_rv(ref_state[:3], ref_state[3:6])
                    thrust_eci = c_ir @ accel_ric
                else:
                    thrust_eci = accel_ric
                strategy = "manual_moon_ric_translation" if mode in MOON_RIC_TRANSLATION_MODES else "manual_ric_translation"
                player_control_mode = "moon_ric_translation" if mode in MOON_RIC_TRANSLATION_MODES else "ric_translation"
            return {
                "thrust_eci_km_s2": thrust_eci,
                "mission_mode": {
                    "strategy": strategy,
                    "throttle": throttle,
                },
                "command_mode_flags": {
                    "player_controlled": True,
                    "player_control_mode": player_control_mode,
                    "player_throttle": throttle,
                    "player_accel_ric_km_s2": accel_ric.tolist(),
                },
            }
        q_cmd = self._integrate_target(truth=truth, t_s=float(t_s), dt_s=float(dt_s))
        throttle = float(np.clip(self.command_state.throttle, 0.0, 1.0))
        firing_duty = self.command_state.consume_firing_duty_cycle(float(dt_s))
        accel_mag = float(max(self.max_accel_km_s2, 0.0)) * throttle * firing_duty
        return {
            "desired_attitude_quat_bn": q_cmd,
            # The simulator later replaces this placeholder direction with the
            # body-mounted thruster force direction while preserving magnitude.
            "thrust_eci_km_s2": np.array([accel_mag, 0.0, 0.0], dtype=float),
            "mission_mode": {
                "strategy": "manual_game",
                "firing": bool(self.command_state.firing),
                "throttle": throttle,
                "firing_duty_cycle": firing_duty,
            },
            "command_mode_flags": {
                "player_controlled": True,
                "player_firing": bool(self.command_state.firing),
                "player_throttle": throttle,
                "player_firing_duty_cycle": firing_duty,
            },
        }


def _reference_state6(
    *,
    reference_object_id: str,
    own_knowledge: dict[str, StateBelief] | None,
) -> np.ndarray | None:
    ref_id = str(reference_object_id)
    ref = dict(own_knowledge or {}).get(ref_id)
    if ref is not None and getattr(ref, "state", np.array([])).size >= 6:
        return np.array(ref.state[:6], dtype=float)
    return None
