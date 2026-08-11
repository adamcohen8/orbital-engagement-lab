"""Reusable deterministic reference controllers for relative-proximity operations."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.control.orbit.hcw_pd import HCWPDController, _as_gain_matrix, _as_state
from sim.control.orbit.ric_pd import RICPDTransferController
from sim.core.interfaces import Controller
from sim.core.models import Command, StateBelief
from sim.utils.frames import ric_curv_to_rect, ric_dcm_ir_from_rv


def _relative_rect_state(
    belief: StateBelief,
    ric_slice: tuple[int, int],
    chief_slice: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    i0, i1 = ric_slice
    j0, j1 = chief_slice
    if i1 - i0 != 6 or j1 - j0 != 6 or belief.state.size < max(i1, j1):
        return None
    curv = np.asarray(belief.state[i0:i1], dtype=float)
    chief = np.asarray(belief.state[j0:j1], dtype=float)
    radius_km = float(np.linalg.norm(chief[:3]))
    if radius_km <= 0.0 or not bool(np.all(np.isfinite(curv))) or not bool(np.all(np.isfinite(chief))):
        return None
    return ric_curv_to_rect(curv, r0_km=radius_km), chief[:3], chief[3:6]


def _limited(vec: np.ndarray, limit: float) -> tuple[np.ndarray, float]:
    result = np.asarray(vec, dtype=float).reshape(3)
    norm = float(np.linalg.norm(result))
    if limit <= 0.0:
        return np.zeros(3, dtype=float), 0.0
    if norm > limit:
        return result * (limit / norm), float(limit / norm)
    return result, 1.0


@dataclass
class RICRelativeHoldController(HCWPDController):
    """Hold a configured rectangular-RIC position and relative velocity."""

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        command = super().act(belief, t_s, budget_ms)
        command.mode_flags["mode"] = "ric_relative_hold"
        command.mode_flags["reference_family"] = "reference_rpo"
        return command


@dataclass
class RICApproachController(Controller):
    """Rate-limited R-, V-, or C-bar approach to a terminal RIC state."""

    max_accel_km_s2: float
    axis: str = "I"
    approach_speed_m_s: float = 0.10
    slowdown_distance_km: float = 0.25
    terminal_state_ric: np.ndarray = field(default_factory=lambda: np.zeros(6))
    kp: np.ndarray = field(default_factory=lambda: np.eye(3) * 2.5e-6)
    kd: np.ndarray = field(default_factory=lambda: np.eye(3) * 3.5e-3)
    ric_curv_state_slice: tuple[int, int] = (0, 6)
    chief_eci_state_slice: tuple[int, int] = (6, 12)

    def __post_init__(self) -> None:
        token = str(self.axis).strip().upper().replace("-BAR", "").replace("BAR", "")
        aliases = {"R": 0, "RADIAL": 0, "V": 1, "I": 1, "INTRACK": 1, "C": 2, "CROSSTRACK": 2}
        if token not in aliases:
            raise ValueError("axis must identify R-bar, V/I-bar, or C-bar.")
        self.axis = ("R", "I", "C")[aliases[token]]
        if self.max_accel_km_s2 < 0.0:
            raise ValueError("max_accel_km_s2 must be non-negative.")
        if self.approach_speed_m_s < 0.0:
            raise ValueError("approach_speed_m_s must be non-negative.")
        if self.slowdown_distance_km <= 0.0:
            raise ValueError("slowdown_distance_km must be positive.")
        self.terminal_state_ric = _as_state(self.terminal_state_ric, "terminal_state_ric")
        self.kp = _as_gain_matrix(self.kp, "kp")
        self.kd = _as_gain_matrix(self.kd, "kd")

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        relative = _relative_rect_state(belief, self.ric_curv_state_slice, self.chief_eci_state_slice)
        if relative is None:
            return Command.zero()
        state, chief_r, chief_v = relative
        axis_index = {"R": 0, "I": 1, "C": 2}[self.axis]
        position_error = state[:3] - self.terminal_state_ric[:3]
        along_remaining = float(self.terminal_state_ric[axis_index] - state[axis_index])
        direction = float(np.sign(along_remaining))
        speed_limit_km_s = float(self.approach_speed_m_s) / 1000.0
        ramp = float(np.clip(abs(along_remaining) / self.slowdown_distance_km, 0.0, 1.0))
        desired_velocity = np.array(self.terminal_state_ric[3:6], dtype=float)
        desired_velocity[axis_index] += direction * speed_limit_km_s * ramp
        velocity_error = state[3:6] - desired_velocity
        accel_pre_limit = -(self.kp @ position_error) - (self.kd @ velocity_error)
        accel_ric, scale = _limited(accel_pre_limit, float(self.max_accel_km_s2))
        command = Command(thrust_eci_km_s2=ric_dcm_ir_from_rv(chief_r, chief_v) @ accel_ric)
        command.mode_flags.update(
            {
                "mode": f"{self.axis.lower()}_bar_approach",
                "reference_family": "reference_rpo",
                "phase": "terminal_brake" if ramp < 1.0 else "approach",
                "axis": self.axis,
                "relative_state_ric": state.tolist(),
                "terminal_state_ric": self.terminal_state_ric.tolist(),
                "desired_velocity_ric_km_s": desired_velocity.tolist(),
                "remaining_along_axis_km": along_remaining,
                "limit_scale": scale,
                "saturated": bool(scale < 1.0),
            }
        )
        return command


@dataclass
class VBarApproachController(RICApproachController):
    axis: str = "I"


@dataclass
class RBarApproachController(RICApproachController):
    axis: str = "R"


@dataclass
class CBarApproachController(RICApproachController):
    axis: str = "C"


@dataclass
class RICWaypointController(Controller):
    """Sequence rectangular-RIC waypoints using position and rate gates."""

    waypoints_ric: list[list[float]]
    max_accel_km_s2: float
    position_tolerance_km: float = 0.02
    velocity_tolerance_m_s: float = 0.02
    kp: np.ndarray = field(default_factory=lambda: np.eye(3) * 3.0e-6)
    kd: np.ndarray = field(default_factory=lambda: np.eye(3) * 4.0e-3)
    loop: bool = False
    ric_curv_state_slice: tuple[int, int] = (0, 6)
    chief_eci_state_slice: tuple[int, int] = (6, 12)
    _index: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.waypoints_ric:
            raise ValueError("waypoints_ric must contain at least one waypoint.")
        self.waypoints_ric = [_as_state(item, "waypoint") for item in self.waypoints_ric]
        if self.max_accel_km_s2 < 0.0:
            raise ValueError("max_accel_km_s2 must be non-negative.")
        if self.position_tolerance_km < 0.0 or self.velocity_tolerance_m_s < 0.0:
            raise ValueError("waypoint tolerances must be non-negative.")
        self.kp = _as_gain_matrix(self.kp, "kp")
        self.kd = _as_gain_matrix(self.kd, "kd")

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        relative = _relative_rect_state(belief, self.ric_curv_state_slice, self.chief_eci_state_slice)
        if relative is None:
            return Command.zero()
        state, chief_r, chief_v = relative
        waypoint = np.asarray(self.waypoints_ric[self._index], dtype=float)
        pos_error = state[:3] - waypoint[:3]
        vel_error = state[3:6] - waypoint[3:6]
        reached = bool(
            np.linalg.norm(pos_error) <= self.position_tolerance_km
            and np.linalg.norm(vel_error) * 1000.0 <= self.velocity_tolerance_m_s
        )
        previous_index = self._index
        if reached:
            if self._index + 1 < len(self.waypoints_ric):
                self._index += 1
            elif self.loop:
                self._index = 0
            waypoint = np.asarray(self.waypoints_ric[self._index], dtype=float)
            pos_error = state[:3] - waypoint[:3]
            vel_error = state[3:6] - waypoint[3:6]
        accel_pre_limit = -(self.kp @ pos_error) - (self.kd @ vel_error)
        accel_ric, scale = _limited(accel_pre_limit, float(self.max_accel_km_s2))
        command = Command(thrust_eci_km_s2=ric_dcm_ir_from_rv(chief_r, chief_v) @ accel_ric)
        command.mode_flags.update(
            {
                "mode": "ric_waypoint",
                "reference_family": "reference_rpo",
                "phase": "complete" if reached and self._index == previous_index and not self.loop else "tracking",
                "waypoint_index": int(self._index),
                "waypoint_count": len(self.waypoints_ric),
                "waypoint_advanced": bool(self._index != previous_index),
                "waypoint_ric": waypoint.tolist(),
                "position_error_km": float(np.linalg.norm(pos_error)),
                "velocity_error_m_s": float(np.linalg.norm(vel_error) * 1000.0),
                "limit_scale": scale,
                "saturated": bool(scale < 1.0),
            }
        )
        return command


@dataclass
class KeepOutStandoffController(Controller):
    """Apply a deterministic outward command inside a protected RIC sphere."""

    keep_out_radius_km: float
    max_accel_km_s2: float
    standoff_margin_km: float = 0.10
    kp: float = 4.0e-6
    kd: float = 4.0e-3
    ric_curv_state_slice: tuple[int, int] = (0, 6)
    chief_eci_state_slice: tuple[int, int] = (6, 12)

    def __post_init__(self) -> None:
        if self.keep_out_radius_km <= 0.0 or self.standoff_margin_km < 0.0:
            raise ValueError("keep_out_radius_km must be positive and standoff_margin_km non-negative.")
        if self.max_accel_km_s2 < 0.0 or self.kp < 0.0 or self.kd < 0.0:
            raise ValueError("authority and gains must be non-negative.")

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        relative = _relative_rect_state(belief, self.ric_curv_state_slice, self.chief_eci_state_slice)
        if relative is None:
            return Command.zero()
        state, chief_r, chief_v = relative
        range_km = float(np.linalg.norm(state[:3]))
        active = range_km < self.keep_out_radius_km
        if not active:
            accel_ric = np.zeros(3, dtype=float)
            scale = 1.0
        else:
            outward = state[:3] / range_km if range_km > 1.0e-12 else np.array([1.0, 0.0, 0.0])
            target_range = self.keep_out_radius_km + self.standoff_margin_km
            outward_speed = float(np.dot(state[3:6], outward))
            pre_limit = self.kp * (target_range - range_km) * outward - self.kd * min(outward_speed, 0.0) * outward
            accel_ric, scale = _limited(pre_limit, self.max_accel_km_s2)
        command = Command(thrust_eci_km_s2=ric_dcm_ir_from_rv(chief_r, chief_v) @ accel_ric)
        command.mode_flags.update(
            {
                "mode": "keep_out_standoff",
                "reference_family": "reference_rpo",
                "phase": "protect" if active else "monitor",
                "keep_out_active": active,
                "range_km": range_km,
                "keep_out_radius_km": self.keep_out_radius_km,
                "limit_scale": scale,
                "saturated": bool(scale < 1.0),
            }
        )
        return command


@dataclass
class PassiveSafeRetreatController(Controller):
    """Acquire a configured outward drift rate, then coast passively."""

    max_accel_km_s2: float
    retreat_speed_m_s: float = 0.20
    coast_range_km: float = 1.0
    velocity_gain_s_inv: float = 0.02
    retreat_axis_ric: np.ndarray | None = None
    ric_curv_state_slice: tuple[int, int] = (0, 6)
    chief_eci_state_slice: tuple[int, int] = (6, 12)

    def __post_init__(self) -> None:
        if min(self.max_accel_km_s2, self.retreat_speed_m_s, self.coast_range_km, self.velocity_gain_s_inv) < 0.0:
            raise ValueError("retreat authority, speed, range, and gain must be non-negative.")
        if self.retreat_axis_ric is not None:
            axis = np.asarray(self.retreat_axis_ric, dtype=float).reshape(-1)
            if axis.size != 3 or not bool(np.all(np.isfinite(axis))) or np.linalg.norm(axis) <= 0.0:
                raise ValueError("retreat_axis_ric must be a finite nonzero length-3 vector.")
            self.retreat_axis_ric = axis / np.linalg.norm(axis)

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        relative = _relative_rect_state(belief, self.ric_curv_state_slice, self.chief_eci_state_slice)
        if relative is None:
            return Command.zero()
        state, chief_r, chief_v = relative
        range_km = float(np.linalg.norm(state[:3]))
        if self.retreat_axis_ric is not None:
            axis = np.asarray(self.retreat_axis_ric, dtype=float)
        elif range_km > 1.0e-12:
            axis = state[:3] / range_km
        else:
            axis = np.array([0.0, -1.0, 0.0], dtype=float)
        desired_speed_km_s = self.retreat_speed_m_s / 1000.0
        outward_speed = float(np.dot(state[3:6], axis))
        coast = bool(range_km >= self.coast_range_km and outward_speed >= desired_speed_km_s * 0.9)
        pre_limit = np.zeros(3, dtype=float) if coast else self.velocity_gain_s_inv * (desired_speed_km_s - outward_speed) * axis
        accel_ric, scale = _limited(pre_limit, self.max_accel_km_s2)
        command = Command(thrust_eci_km_s2=ric_dcm_ir_from_rv(chief_r, chief_v) @ accel_ric)
        command.mode_flags.update(
            {
                "mode": "passive_safe_retreat",
                "reference_family": "reference_rpo",
                "phase": "passive_coast" if coast else "retreat_burn",
                "range_km": range_km,
                "outward_speed_m_s": outward_speed * 1000.0,
                "retreat_axis_ric": axis.tolist(),
                "limit_scale": scale,
                "saturated": bool(scale < 1.0),
            }
        )
        return command


@dataclass
class TerminalBrakingController(Controller):
    """Brake closing motion and settle inside a terminal relative-state box."""

    max_accel_km_s2: float
    terminal_state_ric: np.ndarray = field(default_factory=lambda: np.zeros(6))
    max_closing_speed_m_s: float = 0.05
    terminal_box_km: float = 0.10
    kp: float = 4.0e-6
    kd: float = 5.0e-3
    ric_curv_state_slice: tuple[int, int] = (0, 6)
    chief_eci_state_slice: tuple[int, int] = (6, 12)

    def __post_init__(self) -> None:
        self.terminal_state_ric = _as_state(self.terminal_state_ric, "terminal_state_ric")
        if min(self.max_accel_km_s2, self.max_closing_speed_m_s, self.terminal_box_km, self.kp, self.kd) < 0.0:
            raise ValueError("terminal-braking parameters must be non-negative.")

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        relative = _relative_rect_state(belief, self.ric_curv_state_slice, self.chief_eci_state_slice)
        if relative is None:
            return Command.zero()
        state, chief_r, chief_v = relative
        error = state - self.terminal_state_ric
        range_error = float(np.linalg.norm(error[:3]))
        los = error[:3] / range_error if range_error > 1.0e-12 else np.zeros(3, dtype=float)
        closing_speed_m_s = float(-np.dot(error[3:6], los) * 1000.0)
        desired_velocity = np.array(self.terminal_state_ric[3:6], dtype=float)
        if closing_speed_m_s > self.max_closing_speed_m_s and range_error > 1.0e-12:
            desired_velocity += -los * self.max_closing_speed_m_s / 1000.0
        pre_limit = -self.kp * error[:3] - self.kd * (state[3:6] - desired_velocity)
        accel_ric, scale = _limited(pre_limit, self.max_accel_km_s2)
        command = Command(thrust_eci_km_s2=ric_dcm_ir_from_rv(chief_r, chief_v) @ accel_ric)
        command.mode_flags.update(
            {
                "mode": "terminal_braking",
                "reference_family": "reference_rpo",
                "phase": "terminal_box" if range_error <= self.terminal_box_km else "braking",
                "range_error_km": range_error,
                "closing_speed_m_s": closing_speed_m_s,
                "max_closing_speed_m_s": self.max_closing_speed_m_s,
                "limit_scale": scale,
                "saturated": bool(scale < 1.0),
            }
        )
        return command


@dataclass
class RICFlyaroundController(Controller):
    """Cycle through a polygonal inspection flyaround in a selected RIC plane."""

    radius_km: float
    max_accel_km_s2: float
    plane: str = "RI"
    waypoint_count: int = 8
    position_tolerance_km: float = 0.02
    velocity_tolerance_m_s: float = 0.02
    _delegate: RICWaypointController = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.radius_km <= 0.0 or self.waypoint_count < 4:
            raise ValueError("radius_km must be positive and waypoint_count at least 4.")
        token = str(self.plane).strip().upper()
        axes = {"RI": (0, 1), "IR": (0, 1), "IC": (1, 2), "CI": (1, 2), "RC": (0, 2), "CR": (0, 2)}
        if token not in axes:
            raise ValueError("plane must be one of RI, IC, or RC.")
        first, second = axes[token]
        waypoints: list[list[float]] = []
        for angle in np.linspace(0.0, 2.0 * np.pi, self.waypoint_count, endpoint=False):
            state = np.zeros(6, dtype=float)
            state[first] = self.radius_km * np.cos(angle)
            state[second] = self.radius_km * np.sin(angle)
            waypoints.append(state.tolist())
        self._delegate = RICWaypointController(
            waypoints_ric=waypoints,
            max_accel_km_s2=self.max_accel_km_s2,
            position_tolerance_km=self.position_tolerance_km,
            velocity_tolerance_m_s=self.velocity_tolerance_m_s,
            loop=True,
        )

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        command = self._delegate.act(belief, t_s, budget_ms)
        command.mode_flags.update({"mode": "ric_flyaround", "flyaround_plane": self.plane, "radius_km": self.radius_km})
        return command


@dataclass
class LowThrustPhasingController(VBarApproachController):
    """Low-authority along-track phasing baseline for slow relative acquisition."""

    approach_speed_m_s: float = 0.01

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        command = super().act(belief, t_s, budget_ms)
        command.mode_flags["mode"] = "low_thrust_phasing"
        return command


@dataclass
class PlaneChangeTrimController(CBarApproachController):
    """Cross-track relative-position/rate trim baseline for bounded plane matching."""

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        command = super().act(belief, t_s, budget_ms)
        command.mode_flags["mode"] = "plane_change_trim"
        return command


@dataclass
class HCWRendezvousPlannerController(RICPDTransferController):
    """HCW/SS-J2 closed-form velocity-acquisition planner with finite-burn realization."""

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        command = super().act(belief, t_s, budget_ms)
        command.mode_flags["mode"] = "hcw_rendezvous_planner"
        command.mode_flags["equivalent_finite_burn"] = True
        return command


@dataclass
class ProportionalNavigationController(Controller):
    """Simple target-directed proportional-navigation acceleration in RIC."""

    max_accel_km_s2: float
    navigation_constant: float = 3.0
    minimum_closing_speed_m_s: float = 0.0
    evade: bool = False
    ric_curv_state_slice: tuple[int, int] = (0, 6)
    chief_eci_state_slice: tuple[int, int] = (6, 12)

    def __post_init__(self) -> None:
        if self.max_accel_km_s2 < 0.0 or self.navigation_constant < 0.0 or self.minimum_closing_speed_m_s < 0.0:
            raise ValueError("proportional-navigation parameters must be non-negative.")

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        relative = _relative_rect_state(belief, self.ric_curv_state_slice, self.chief_eci_state_slice)
        if relative is None:
            return Command.zero()
        state, chief_r, chief_v = relative
        position = state[:3]
        velocity = state[3:6]
        range_km = float(np.linalg.norm(position))
        if range_km <= 1.0e-12:
            accel_ric = np.zeros(3, dtype=float)
            closing_speed = 0.0
            los_rate = np.zeros(3, dtype=float)
        else:
            los = position / range_km
            closing_speed = float(-np.dot(velocity, los))
            los_rate = np.cross(position, velocity) / max(range_km * range_km, 1.0e-12)
            lateral = self.navigation_constant * max(
                abs(closing_speed), self.minimum_closing_speed_m_s / 1000.0
            ) * np.cross(los_rate, los)
            direct = -los * max(closing_speed, 0.0) * 0.001
            pre_limit = lateral + direct
            if self.evade:
                pre_limit = -pre_limit
            accel_ric, _ = _limited(pre_limit, self.max_accel_km_s2)
        command = Command(thrust_eci_km_s2=ric_dcm_ir_from_rv(chief_r, chief_v) @ accel_ric)
        command.mode_flags.update(
            {
                "mode": "proportional_navigation_evade" if self.evade else "proportional_navigation_pursuit",
                "reference_family": "reference_rpo",
                "range_km": range_km,
                "closing_speed_m_s": closing_speed * 1000.0,
                "los_rate_rad_s": los_rate.tolist(),
                "navigation_constant": self.navigation_constant,
            }
        )
        return command
