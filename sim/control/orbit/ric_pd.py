from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.control.orbit.hcw_pd import _as_gain_matrix, _as_state
from sim.control.orbit.hcw_transfer import solve_linear_position_rendezvous
from sim.control.orbit.lqr import HCWLQRController
from sim.core.interfaces import Controller
from sim.core.models import Command, StateBelief
from sim.dynamics.orbit.environment import EARTH_J2, EARTH_RADIUS_KM
from sim.dynamics.orbit.relative_linear import RelativeLinearDynamics, normalize_relative_linear_model
from sim.utils.frames import ric_curv_to_rect, ric_dcm_ir_from_rv


def _limit_vector(vec: np.ndarray, limit: float) -> tuple[np.ndarray, float]:
    out = np.array(vec, dtype=float).reshape(3)
    nrm = float(np.linalg.norm(out))
    if limit <= 0.0:
        out[:] = 0.0
        return out, 0.0
    if nrm > limit:
        scale = float(limit / nrm)
        out *= scale
        return out, scale
    return out, 1.0


@dataclass(frozen=True, slots=True)
class RICPDTransferGuidanceResult:
    """Controller-independent output of the reusable transfer guidance law."""

    acceleration_eci_km_s2: np.ndarray
    mode_flags: dict[str, object]


@dataclass
class RICPDTransferController(Controller):
    """RIC PD rendezvous controller that acquires a planned coast arc before terminal cleanup."""

    max_accel_km_s2: float
    mean_motion_rad_s: float
    transfer_time_s: float
    dynamics_model: str = "hcw"
    reference_radius_km: float | None = None
    reference_inclination_rad: float | None = None
    j2: float = EARTH_J2
    earth_radius_km: float = EARTH_RADIUS_KM
    burn_time_constant_s: float = 45.0
    correction_interval_s: float = 300.0
    velocity_deadband_m_s: float = 0.015
    final_brake_start_s: float = 180.0
    terminal_start_s: float = 750.0
    terminal_range_km: float = 0.20
    terminal_kp: np.ndarray = field(default_factory=lambda: np.eye(3) * 2.5e-6)
    terminal_kd: np.ndarray = field(default_factory=lambda: np.eye(3) * 3.5e-3)
    desired_state_ric: np.ndarray = field(default_factory=lambda: np.zeros(6))
    ric_curv_state_slice: tuple[int, int] = (0, 6)
    chief_eci_state_slice: tuple[int, int] = (6, 12)
    state_signs: np.ndarray = field(default_factory=lambda: np.ones(6))

    _arrival_t_s: float | None = field(default=None, init=False, repr=False)
    _next_correction_t_s: float = field(default=-np.inf, init=False, repr=False)
    _target_velocity_ric_km_s: np.ndarray | None = field(default=None, init=False, repr=False)
    _last_ideal_delta_v_ric_km_s: np.ndarray = field(default_factory=lambda: np.zeros(3), init=False, repr=False)
    _planned_arrival_velocity_ric_km_s: np.ndarray = field(default_factory=lambda: np.zeros(3), init=False, repr=False)
    _terminal_gain: np.ndarray = field(default_factory=lambda: np.zeros((3, 6)), init=False, repr=False)
    _zero3: np.ndarray = field(default_factory=lambda: np.zeros(3), init=False, repr=False)
    _relative_dynamics: RelativeLinearDynamics = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_accel_km_s2 < 0.0:
            raise ValueError("max_accel_km_s2 must be non-negative.")
        if self.mean_motion_rad_s <= 0.0:
            raise ValueError("mean_motion_rad_s must be positive.")
        self.dynamics_model = normalize_relative_linear_model(self.dynamics_model)
        self._relative_dynamics = RelativeLinearDynamics(
            model=self.dynamics_model,
            mean_motion_rad_s=self.mean_motion_rad_s,
            reference_radius_km=self.reference_radius_km,
            reference_inclination_rad=self.reference_inclination_rad,
            j2=self.j2,
            earth_radius_km=self.earth_radius_km,
        )
        if self.transfer_time_s <= 0.0:
            raise ValueError("transfer_time_s must be positive.")
        if self.burn_time_constant_s <= 0.0:
            raise ValueError("burn_time_constant_s must be positive.")
        if self.correction_interval_s <= 0.0:
            raise ValueError("correction_interval_s must be positive.")
        if self.velocity_deadband_m_s < 0.0:
            raise ValueError("velocity_deadband_m_s must be non-negative.")
        if self.final_brake_start_s < 0.0:
            raise ValueError("final_brake_start_s must be non-negative.")
        if self.terminal_start_s < 0.0:
            raise ValueError("terminal_start_s must be non-negative.")
        if self.terminal_range_km < 0.0:
            raise ValueError("terminal_range_km must be non-negative.")
        if self.ric_curv_state_slice[1] - self.ric_curv_state_slice[0] != 6:
            raise ValueError("ric_curv_state_slice must select exactly 6 elements.")
        if self.chief_eci_state_slice[1] - self.chief_eci_state_slice[0] != 6:
            raise ValueError("chief_eci_state_slice must select exactly 6 elements.")
        self.terminal_kp = _as_gain_matrix(self.terminal_kp, "terminal_kp")
        self.terminal_kd = _as_gain_matrix(self.terminal_kd, "terminal_kd")
        self.desired_state_ric = _as_state(self.desired_state_ric, "desired_state_ric")
        signs = np.array(self.state_signs, dtype=float).reshape(-1)
        if signs.size != 6:
            raise ValueError("state_signs must be length 6.")
        signs[signs == 0.0] = 1.0
        self.state_signs = np.sign(signs)
        self._terminal_gain = np.hstack((self.terminal_kp, self.terminal_kd))

    def linear_system_summary(self) -> dict[str, object]:
        k_gain = self._terminal_gain
        return {
            "system_type": "ric_pd_transfer",
            "law_label": HCWLQRController._control_law_label(self.state_signs),
            "control_axes": ["R", "I", "C"],
            "state_labels": ["R", "I", "C", "dR", "dI", "dC"],
            "terminal_gain_matrix": k_gain.tolist(),
            "transfer_time_s": float(self.transfer_time_s),
            "correction_interval_s": float(self.correction_interval_s),
        }

    def snapshot_state(self) -> dict[str, object]:
        """Return the state needed to resume transfer guidance deterministically."""

        return {
            "arrival_t_s": self._arrival_t_s,
            "next_correction_t_s": (None if not np.isfinite(self._next_correction_t_s) else self._next_correction_t_s),
            "target_velocity_ric_km_s": (
                None if self._target_velocity_ric_km_s is None else self._target_velocity_ric_km_s.tolist()
            ),
            "last_ideal_delta_v_ric_km_s": self._last_ideal_delta_v_ric_km_s.tolist(),
            "planned_arrival_velocity_ric_km_s": self._planned_arrival_velocity_ric_km_s.tolist(),
        }

    def restore_state(self, state: dict[str, object]) -> None:
        """Restore a state produced by :meth:`snapshot_state`."""

        arrival = state.get("arrival_t_s")
        next_correction = state.get("next_correction_t_s")
        self._arrival_t_s = None if arrival is None else float(arrival)
        self._next_correction_t_s = -np.inf if next_correction is None else float(next_correction)
        if self._arrival_t_s is not None and not np.isfinite(self._arrival_t_s):
            raise ValueError("RIC PD transfer arrival time must be finite")
        if not np.isfinite(self._next_correction_t_s) and self._next_correction_t_s != -np.inf:
            raise ValueError("RIC PD transfer correction time is invalid")
        target_velocity = state.get("target_velocity_ric_km_s")
        self._target_velocity_ric_km_s = (
            None if target_velocity is None else _finite_state_vector(target_velocity, "target velocity")
        )
        self._last_ideal_delta_v_ric_km_s = _finite_state_vector(
            state.get("last_ideal_delta_v_ric_km_s", (0.0, 0.0, 0.0)),
            "last ideal delta-v",
        )
        self._planned_arrival_velocity_ric_km_s = _finite_state_vector(
            state.get("planned_arrival_velocity_ric_km_s", (0.0, 0.0, 0.0)),
            "planned arrival velocity",
        )

    def _relative_state(self, belief: StateBelief) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        i0, i1 = self.ric_curv_state_slice
        j0, j1 = self.chief_eci_state_slice
        if belief.state.size < max(i1, j1):
            return None
        x_curv = np.asarray(belief.state[i0:i1], dtype=float)
        chief_eci = np.asarray(belief.state[j0:j1], dtype=float)
        r_chief = chief_eci[:3]
        v_chief = chief_eci[3:]
        r0 = float(np.linalg.norm(r_chief))
        if r0 <= 0.0:
            return None
        return ric_curv_to_rect(x_curv, r0_km=r0), r_chief, v_chief

    def _terminal_accel(self, x_effective: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        accel_pre_limit = -(self._terminal_gain @ x_effective)
        accel_pre_limit -= self._relative_dynamics.system_matrix()[3:, :] @ x_effective
        accel, scale = _limit_vector(accel_pre_limit, float(self.max_accel_km_s2))
        return accel_pre_limit, accel, scale

    def _refresh_guidance(self, x_effective: np.ndarray, t_s: float, remaining_s: float) -> None:
        solution = solve_linear_position_rendezvous(
            x_effective,
            np.zeros(3, dtype=float),
            self._relative_dynamics,
            float(remaining_s),
        )
        self._target_velocity_ric_km_s = np.array(
            solution.required_post_chaser_rel_velocity_ric_km_s,
            dtype=float,
        )
        self._last_ideal_delta_v_ric_km_s = np.array(solution.required_delta_v_ric_km_s, dtype=float)
        self._planned_arrival_velocity_ric_km_s = np.array(solution.rendezvous_state_ric[3:], dtype=float)
        self._next_correction_t_s = float(t_s) + float(self.correction_interval_s)

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        rel = self._relative_state(belief)
        if rel is None:
            return Command.zero()
        x_rect, r_chief, v_chief = rel
        result = self.guide_relative_state(x_rect, r_chief, v_chief, t_s=t_s)
        return Command(
            thrust_eci_km_s2=result.acceleration_eci_km_s2,
            torque_body_nm=self._zero3.copy(),
            mode_flags=result.mode_flags,
        )

    def guide_relative_state(
        self,
        relative_state_ric_rect_km: np.ndarray,
        chief_position_eci_km: np.ndarray,
        chief_velocity_eci_km_s: np.ndarray,
        *,
        t_s: float,
    ) -> RICPDTransferGuidanceResult:
        """Evaluate guidance without using the legacy controller command boundary.

        Complete flight-software stacks use this subordinate interface with an
        onboard relative-navigation solution. The legacy :meth:`act` method is
        retained for component-library compatibility and delegates here.
        """

        x_rect = _finite_six_vector(relative_state_ric_rect_km, "relative state")
        r_chief = _finite_state_vector(chief_position_eci_km, "chief position")
        v_chief = _finite_state_vector(chief_velocity_eci_km_s, "chief velocity")
        if float(np.linalg.norm(r_chief)) <= 0.0:
            raise ValueError("RIC PD transfer chief position must be nonzero")
        if not np.isfinite(t_s):
            raise ValueError("RIC PD transfer time must be finite")
        if self._arrival_t_s is None:
            self._arrival_t_s = float(t_s) + float(self.transfer_time_s)

        err = x_rect - self.desired_state_ric
        x_effective = self.state_signs * err
        remaining_s = max(float(self._arrival_t_s) - float(t_s), 1.0)
        signed_remaining_s = float(self._arrival_t_s) - float(t_s)
        range_km = float(np.linalg.norm(x_effective[:3]))
        final_brake_active = 0.0 < signed_remaining_s <= float(self.final_brake_start_s)
        terminal_active = (
            signed_remaining_s <= 0.0
            or remaining_s <= float(self.terminal_start_s)
            or range_km <= float(self.terminal_range_km)
        ) and not final_brake_active

        phase = "terminal_cleanup" if terminal_active else "coast"
        accel_pre_limit = np.zeros(3, dtype=float)
        limit_scale = 1.0
        target_velocity = (
            np.array(self._target_velocity_ric_km_s, dtype=float)
            if self._target_velocity_ric_km_s is not None
            else np.array(x_effective[3:], dtype=float)
        )
        velocity_error = target_velocity - x_effective[3:]

        if terminal_active:
            accel_pre_limit, accel_ric, limit_scale = self._terminal_accel(x_effective)
        elif final_brake_active:
            phase = "final_brake"
            if self.final_brake_start_s > 0.0:
                ramp = float(np.clip(signed_remaining_s / float(self.final_brake_start_s), 0.0, 1.0))
            else:
                ramp = 0.0
            target_velocity = self._planned_arrival_velocity_ric_km_s * ramp
            velocity_error = target_velocity - x_effective[3:]
            accel_pre_limit = velocity_error / float(self.burn_time_constant_s)
            accel_ric, limit_scale = _limit_vector(accel_pre_limit, float(self.max_accel_km_s2))
        else:
            if self._target_velocity_ric_km_s is None or float(t_s) >= self._next_correction_t_s - 1e-12:
                try:
                    self._refresh_guidance(x_effective=x_effective, t_s=float(t_s), remaining_s=remaining_s)
                    target_velocity = np.array(self._target_velocity_ric_km_s, dtype=float)
                    velocity_error = target_velocity - x_effective[3:]
                except ValueError:
                    accel_pre_limit, accel_ric, limit_scale = self._terminal_accel(x_effective)
                    phase = "terminal_cleanup"
                else:
                    accel_ric = self._zero3.copy()
            else:
                accel_ric = self._zero3.copy()

            if phase != "terminal_cleanup":
                velocity_error_m_s = float(np.linalg.norm(velocity_error) * 1000.0)
                if velocity_error_m_s > float(self.velocity_deadband_m_s):
                    accel_pre_limit = velocity_error / float(self.burn_time_constant_s)
                    accel_ric, limit_scale = _limit_vector(accel_pre_limit, float(self.max_accel_km_s2))
                    phase = "guided_burn"
                else:
                    accel_ric = self._zero3.copy()
                    limit_scale = 1.0
                    phase = "coast"

        c_ir = ric_dcm_ir_from_rv(r_chief, v_chief)
        accel_eci = c_ir @ accel_ric
        i0, i1 = self.ric_curv_state_slice
        j0, j1 = self.chief_eci_state_slice
        return RICPDTransferGuidanceResult(
            accel_eci,
            {
                "mode": "ric_pd_transfer",
                "dynamics_model": self.dynamics_model,
                "dynamics_metadata": self._relative_dynamics.metadata(),
                "phase": phase,
                "ric_curv_state_slice": [i0, i1],
                "chief_eci_state_slice": [j0, j1],
                "desired_state_ric": self.desired_state_ric.tolist(),
                "remaining_s": float(remaining_s),
                "signed_remaining_s": float(signed_remaining_s),
                "range_km": range_km,
                "target_velocity_ric_km_s": target_velocity.tolist(),
                "velocity_error_ric_km_s": velocity_error.tolist(),
                "ideal_delta_v_ric_km_s": self._last_ideal_delta_v_ric_km_s.tolist(),
                "planned_arrival_velocity_ric_km_s": self._planned_arrival_velocity_ric_km_s.tolist(),
                "accel_ric_km_s2": accel_ric.tolist(),
                "limit_scale": float(limit_scale),
                "linear_feedback_debug": HCWLQRController._linear_feedback_debug_payload(
                    control_axes=["R", "I", "C"],
                    k_gain=self._terminal_gain,
                    x_rect=x_rect,
                    x_effective=x_effective,
                    control_pre_limit=accel_pre_limit,
                    control_post_limit=accel_ric,
                    limit_scale=limit_scale,
                    state_signs=self.state_signs,
                ),
            },
        )


def _finite_state_vector(value: object, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float).reshape(-1)
    if vector.size != 3 or not np.all(np.isfinite(vector)):
        raise ValueError(f"RIC PD transfer {name} must contain three finite values")
    return vector


def _finite_six_vector(value: object, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float).reshape(-1)
    if vector.size != 6 or not np.all(np.isfinite(vector)):
        raise ValueError(f"RIC PD transfer {name} must contain six finite values")
    return vector
