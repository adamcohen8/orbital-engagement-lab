"""Native-SI orbit/RPO control and translation allocation for GNC v2."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from math import atan2, isfinite, sqrt
from typing import Callable

import numpy as np

from sim.control.orbit.curv_pd import curv_accel_to_rect
from sim.control.orbit.ric_pd import RICPDTransferController
from sim.control.orbit.rmoe import estimate_rmoes_from_rect_ric
from sim.dynamics.orbit.elements import (
    coes_target_state_at_current_true_anomaly,
    orbital_element_feedback_accel,
)
from sim.dynamics.orbit.relative_linear import RelativeLinearDynamics, solve_discrete_lqr_gain
from sim.dynamics.orbit.two_body import propagate_two_body_rk4
from sim.flight_software.contracts import (
    ActuatorCommand,
    ClockTag,
    ContinuousEngineCommand,
    FrameId,
    IdealWrenchCommand,
    PacketId,
    TelemetryField,
    ThrusterOnOffCommand,
    ThrusterPulseCommand,
    ValidityInterval,
)
from sim.gnc.contracts import AllocationResult, AllocationStatus, RequestedEffort, RequestedEffortKind
from sim.gnc.navigation_v2 import OrbitNavigationSolution, RelativeStateEstimateSI
from sim.utils.frames import ric_rect_to_curv
from sim.utils.quaternion import quaternion_to_dcm_bn


class TranslationMode(str, Enum):
    SCHEDULED_BURN = "scheduled_burn"
    STATIONKEEPING = "stationkeeping"
    ORBITAL_ELEMENTS = "orbital_elements"
    RIC_HOLD = "ric_hold"
    R_BAR_APPROACH = "r_bar_approach"
    V_BAR_APPROACH = "v_bar_approach"
    C_BAR_APPROACH = "c_bar_approach"
    WAYPOINT = "waypoint"
    RIC_PD_TRANSFER = "ric_pd_transfer"
    TERMINAL_BRAKING = "terminal_braking"
    PASSIVE_RETREAT = "passive_retreat"
    LOW_THRUST_PHASING = "low_thrust_phasing"
    ATMOSPHERIC_PASS = "atmospheric_pass"


class TranslationControlLaw(str, Enum):
    REFERENCE_PD = "reference_pd"
    HCW_LQR = "hcw_lqr"
    CURVILINEAR_RIC_PD = "curvilinear_ric_pd"
    RMOE_IF_THEN = "rmoe_if_then"


@dataclass(frozen=True, slots=True)
class ScheduledBurn:
    start_time_ns: int
    duration_ns: int
    acceleration_m_s2: tuple[float, float, float]
    frame: str = "eci"

    def __post_init__(self) -> None:
        if self.start_time_ns < 0 or self.duration_ns <= 0:
            raise ValueError("scheduled burn times must be nonnegative with positive duration")
        _finite_vector("acceleration_m_s2", self.acceleration_m_s2, 3)
        if self.frame not in {"eci", "ric"}:
            raise ValueError("scheduled burn frame must be 'eci' or 'ric'")


@dataclass(frozen=True, slots=True)
class TranslationControlConfig:
    default_mode: TranslationMode
    assumed_mass_kg: float
    max_acceleration_m_s2: float
    target_state_eci: tuple[float, ...] | None = None
    target_semi_major_axis_m: float | None = None
    target_eccentricity: float = 0.0
    eccentricity_tolerance: float = 1.0e-4
    target_relative_state_ric: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    waypoints_ric: tuple[tuple[float, ...], ...] = ()
    control_axis_mask: tuple[float, float, float] = (1.0, 1.0, 1.0)
    kp_position_s2: float = 4.0e-6
    kd_velocity_s_inv: float = 4.0e-3
    mean_motion_rad_s: float = 0.0
    approach_speed_m_s: float = 0.1
    slowdown_distance_m: float = 250.0
    terminal_box_m: float = 100.0
    terminal_max_closing_speed_m_s: float = 0.05
    retreat_speed_m_s: float = 0.2
    retreat_coast_range_m: float = 1_000.0
    position_tolerance_m: float = 20.0
    velocity_tolerance_m_s: float = 0.02
    target_id: str | None = None
    validity_ticks: int = 1
    scheduled_burns: tuple[ScheduledBurn, ...] = ()
    atmospheric_raise_start_ns: int = 0
    atmospheric_raise_end_ns: int = 0
    atmospheric_prograde_acceleration_m_s2: float = 0.0
    atmospheric_min_raise_altitude_m: float = 0.0
    atmospheric_pass_entry_altitude_m: float | None = None
    atmospheric_pass_exit_altitude_m: float | None = None
    atmospheric_recovery_delta_v_m_s: float = 0.0
    orbital_element_control_law: str = "energy_eccentricity"
    target_coes: tuple[tuple[str, float], ...] = ()
    controlled_elements: tuple[str, ...] = ("a", "ecc", "inc", "raan", "argp")
    energy_gain_per_s: float = 1.0e-3
    eccentricity_gain_per_s: float = 5.0e-4
    plane_gain_per_s: float = 5.0e-4
    control_law: TranslationControlLaw = TranslationControlLaw.REFERENCE_PD
    control_design_dt_s: float = 1.0
    lqr_q_weights: tuple[float, ...] = (8660.0, 8660.0, 8660.0, 1330.0, 1330.0, 1330.0)
    lqr_r_weights: tuple[float, ...] = (1.94e13, 1.94e13, 1.94e13)
    rmoe_target_radial_center_m: float = 0.0
    rmoe_target_in_track_center_m: float = 0.0
    rmoe_target_in_track_drift_rate_m_s: float = 0.0
    rmoe_target_cross_track_amplitude_m: float = 0.0
    rmoe_max_drift_rate_m_s: float = 0.02
    rmoe_close_zone_m: float = 50.0
    rmoe_cross_track_burn_gate_m: float = 50.0
    transfer_time_s: float = 4_800.0
    burn_time_constant_s: float = 45.0
    correction_interval_s: float = 300.0
    velocity_deadband_m_s: float = 0.015
    final_brake_start_s: float = 180.0
    terminal_start_s: float = 750.0
    terminal_range_m: float = 200.0
    thrust_window_period_s: float = 0.0
    thrust_window_duration_s: float = 0.0
    thrust_window_phase_s: float = 0.0
    thrust_command_deadband_m_s2: float = 0.0
    element_averaging_window_s: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.default_mode, TranslationMode):
            raise TypeError("default_mode must be TranslationMode")
        for name, value, positive in (
            ("assumed_mass_kg", self.assumed_mass_kg, True),
            ("max_acceleration_m_s2", self.max_acceleration_m_s2, False),
            ("kp_position_s2", self.kp_position_s2, False),
            ("kd_velocity_s_inv", self.kd_velocity_s_inv, False),
            ("mean_motion_rad_s", self.mean_motion_rad_s, False),
            ("approach_speed_m_s", self.approach_speed_m_s, False),
            ("slowdown_distance_m", self.slowdown_distance_m, True),
            ("terminal_box_m", self.terminal_box_m, False),
            ("terminal_max_closing_speed_m_s", self.terminal_max_closing_speed_m_s, False),
            ("retreat_speed_m_s", self.retreat_speed_m_s, False),
            ("retreat_coast_range_m", self.retreat_coast_range_m, False),
            ("position_tolerance_m", self.position_tolerance_m, False),
            ("velocity_tolerance_m_s", self.velocity_tolerance_m_s, False),
            ("eccentricity_tolerance", self.eccentricity_tolerance, False),
            ("transfer_time_s", self.transfer_time_s, True),
            ("burn_time_constant_s", self.burn_time_constant_s, True),
            ("correction_interval_s", self.correction_interval_s, True),
            ("velocity_deadband_m_s", self.velocity_deadband_m_s, False),
            ("final_brake_start_s", self.final_brake_start_s, False),
            ("terminal_start_s", self.terminal_start_s, False),
            ("terminal_range_m", self.terminal_range_m, False),
            ("thrust_window_period_s", self.thrust_window_period_s, False),
            ("thrust_window_duration_s", self.thrust_window_duration_s, False),
            ("thrust_window_phase_s", self.thrust_window_phase_s, False),
            ("thrust_command_deadband_m_s2", self.thrust_command_deadband_m_s2, False),
            ("element_averaging_window_s", self.element_averaging_window_s, False),
        ):
            if not isfinite(value) or (value <= 0.0 if positive else value < 0.0):
                qualifier = "positive" if positive else "nonnegative"
                raise ValueError(f"{name} must be finite and {qualifier}")
        if self.target_state_eci is not None:
            _finite_vector("target_state_eci", self.target_state_eci, 6)
        _finite_vector("target_relative_state_ric", self.target_relative_state_ric, 6)
        for waypoint in self.waypoints_ric:
            _finite_vector("waypoint", waypoint, 6)
        _finite_vector("control_axis_mask", self.control_axis_mask, 3)
        if any(value < 0.0 or value > 1.0 for value in self.control_axis_mask):
            raise ValueError("control_axis_mask values must lie in [0, 1]")
        ordered_burns = tuple(sorted(self.scheduled_burns, key=lambda item: item.start_time_ns))
        if ordered_burns != self.scheduled_burns:
            raise ValueError("scheduled_burns must be ordered by start_time_ns")
        for previous, following in zip(self.scheduled_burns, self.scheduled_burns[1:]):
            if following.start_time_ns < previous.start_time_ns + previous.duration_ns:
                raise ValueError("scheduled_burns must not overlap")
        if self.target_semi_major_axis_m is not None and self.target_semi_major_axis_m <= 0.0:
            raise ValueError("target_semi_major_axis_m must be positive")
        if not 0.0 <= self.target_eccentricity < 1.0:
            raise ValueError("target_eccentricity must be in [0, 1)")
        if isinstance(self.validity_ticks, bool) or not isinstance(self.validity_ticks, int) or self.validity_ticks < 1:
            raise ValueError("validity_ticks must be a positive integer")
        if self.atmospheric_raise_start_ns < 0 or self.atmospheric_raise_end_ns < self.atmospheric_raise_start_ns:
            raise ValueError("atmospheric raise times must be ordered and nonnegative")
        if self.atmospheric_prograde_acceleration_m_s2 < 0.0 or self.atmospheric_min_raise_altitude_m < 0.0:
            raise ValueError("atmospheric pass authority and altitude must be nonnegative")
        if (self.atmospheric_pass_entry_altitude_m is None) != (self.atmospheric_pass_exit_altitude_m is None):
            raise ValueError("atmospheric pass entry and exit altitudes must be configured together")
        if self.atmospheric_pass_entry_altitude_m is not None:
            assert self.atmospheric_pass_exit_altitude_m is not None
            if self.atmospheric_pass_entry_altitude_m < 0.0 or self.atmospheric_pass_exit_altitude_m < 0.0:
                raise ValueError("atmospheric pass entry and exit altitudes must be nonnegative")
            if self.atmospheric_pass_exit_altitude_m < self.atmospheric_pass_entry_altitude_m:
                raise ValueError("atmospheric pass exit altitude must be at least the entry altitude")
        if not isfinite(self.atmospheric_recovery_delta_v_m_s) or self.atmospheric_recovery_delta_v_m_s < 0.0:
            raise ValueError("atmospheric recovery delta-v must be finite and nonnegative")
        if self.orbital_element_control_law not in {
            "energy_eccentricity",
            "current_anomaly_stationkeep",
            "element_tracking",
        }:
            raise ValueError("orbital_element_control_law is unsupported")
        if not isinstance(self.control_law, TranslationControlLaw):
            raise TypeError("control_law must be TranslationControlLaw")
        if not isfinite(self.control_design_dt_s) or self.control_design_dt_s <= 0.0:
            raise ValueError("control_design_dt_s must be finite and positive")
        _finite_vector("lqr_q_weights", self.lqr_q_weights, 6)
        _finite_vector("lqr_r_weights", self.lqr_r_weights, 3)
        if any(value < 0.0 for value in self.lqr_q_weights) or any(value <= 0.0 for value in self.lqr_r_weights):
            raise ValueError("LQR Q weights must be nonnegative and R weights must be positive")
        for name, value in (
            ("rmoe_target_radial_center_m", self.rmoe_target_radial_center_m),
            ("rmoe_target_in_track_center_m", self.rmoe_target_in_track_center_m),
            ("rmoe_target_in_track_drift_rate_m_s", self.rmoe_target_in_track_drift_rate_m_s),
            ("rmoe_target_cross_track_amplitude_m", self.rmoe_target_cross_track_amplitude_m),
            ("rmoe_max_drift_rate_m_s", self.rmoe_max_drift_rate_m_s),
            ("rmoe_close_zone_m", self.rmoe_close_zone_m),
            ("rmoe_cross_track_burn_gate_m", self.rmoe_cross_track_burn_gate_m),
        ):
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
        if any(
            value < 0.0
            for value in (
                self.rmoe_target_cross_track_amplitude_m,
                self.rmoe_max_drift_rate_m_s,
                self.rmoe_close_zone_m,
                self.rmoe_cross_track_burn_gate_m,
            )
        ):
            raise ValueError("RMOE amplitude, drift limit, close zone, and burn gate must be nonnegative")
        if self.thrust_window_period_s == 0.0:
            if self.thrust_window_duration_s != 0.0 or self.thrust_window_phase_s != 0.0:
                raise ValueError("thrust window duration and phase require a positive period")
        else:
            if not 0.0 < self.thrust_window_duration_s <= self.thrust_window_period_s:
                raise ValueError("thrust window duration must be positive and no greater than its period")
            if not 0.0 <= self.thrust_window_phase_s < self.thrust_window_period_s:
                raise ValueError("thrust window phase must lie in [0, period)")


@dataclass(frozen=True, slots=True)
class TranslationControlResult:
    mode: TranslationMode
    effort: RequestedEffort | None
    goal_satisfied: bool
    saturated: bool
    infeasible_reason: str | None
    position_error_m: float | None
    velocity_error_m_s: float | None
    pointing_direction_eci: tuple[float, float, float] | None
    phase: str


class TranslationController:
    def __init__(self, config: TranslationControlConfig) -> None:
        self.config = config
        self._waypoint_index = 0
        self._stationkeeping_target = (
            None if config.target_state_eci is None else np.asarray(config.target_state_eci, dtype=float).copy()
        )
        self._stationkeeping_target_time_ns: int | None = None
        self._lqr_cache: dict[float, np.ndarray] = {}
        self._ric_pd_transfer: RICPDTransferController | None = None
        self._atmospheric_pass_seen = False
        self._atmospheric_pass_exited = False
        self._atmospheric_recovery_start_ns: int | None = None
        self._last_window_evaluation_ns: int | None = None
        self._observed_thrust_windows: set[int] = set()
        self._missed_thrust_window_count = 0
        self._thrust_window_open = True
        self._element_samples: list[tuple[int, float, float]] = []

    @property
    def thrust_window_open(self) -> bool:
        return self._thrust_window_open

    @property
    def missed_thrust_window_count(self) -> int:
        return self._missed_thrust_window_count

    @property
    def element_averaging_sample_count(self) -> int:
        return len(self._element_samples)

    def control(
        self,
        solution: OrbitNavigationSolution,
        mode: TranslationMode | str | None = None,
    ) -> TranslationControlResult:
        selected = (
            self.config.default_mode
            if mode is None
            else mode
            if isinstance(mode, TranslationMode)
            else TranslationMode(str(mode))
        )
        acceleration: np.ndarray | None
        position_error: float | None = None
        velocity_error: float | None = None
        phase = selected.value
        explicit_goal_satisfied: bool | None = None
        scheduled_override = False
        if selected is not TranslationMode.SCHEDULED_BURN and self.config.scheduled_burns:
            scheduled = self._scheduled_burn(solution)
            if scheduled[4] != "complete":
                acceleration, position_error, velocity_error, explicit_goal_satisfied, phase = scheduled
                scheduled_override = True
            else:
                acceleration = None
        else:
            acceleration = None
        if scheduled_override:
            pass
        elif selected is TranslationMode.SCHEDULED_BURN:
            acceleration, position_error, velocity_error, explicit_goal_satisfied, phase = self._scheduled_burn(
                solution
            )
        elif selected is TranslationMode.STATIONKEEPING:
            acceleration, position_error, velocity_error = self._stationkeeping(solution)
        elif selected is TranslationMode.ORBITAL_ELEMENTS:
            acceleration, position_error, velocity_error = self._orbital_elements(solution)
        elif selected is TranslationMode.ATMOSPHERIC_PASS:
            acceleration, position_error, velocity_error, phase = self._atmospheric_pass(solution)
        else:
            track = solution.relative_track(self.config.target_id)
            acceleration, position_error, velocity_error, phase = self._relative(selected, solution, track)
        if acceleration is None:
            return TranslationControlResult(
                selected,
                None,
                False,
                False,
                "required navigation state is unavailable",
                position_error,
                velocity_error,
                None,
                phase,
            )
        if selected in {TranslationMode.LOW_THRUST_PHASING, TranslationMode.ORBITAL_ELEMENTS}:
            self._thrust_window_open = self._evaluate_thrust_window(_clock_ns(solution.generated_at))
            if not self._thrust_window_open:
                acceleration = np.zeros(3)
                phase = "thrust_window_coast"
            elif (
                self.config.thrust_command_deadband_m_s2 > 0.0
                and float(np.linalg.norm(acceleration)) < self.config.thrust_command_deadband_m_s2
            ):
                acceleration = np.zeros(3)
                phase = "deadband_coast"
        acceleration *= np.asarray(self.config.control_axis_mask, dtype=float)
        pre_limit_norm = float(np.linalg.norm(acceleration))
        saturated = pre_limit_norm > self.config.max_acceleration_m_s2 > 0.0
        if self.config.max_acceleration_m_s2 <= 0.0:
            acceleration[:] = 0.0
            saturated = pre_limit_norm > 0.0
        elif saturated:
            acceleration *= self.config.max_acceleration_m_s2 / pre_limit_norm
        mass = solution.mass_kg if solution.mass_kg is not None else self.config.assumed_mass_kg
        force = acceleration * mass
        validity = ValidityInterval(
            solution.generated_at,
            _add_ticks(solution.generated_at, self.config.validity_ticks),
        )
        if phase == "finite_burn":
            now_ns = _clock_ns(solution.generated_at)
            active_burn = next(
                burn
                for burn in self.config.scheduled_burns
                if burn.start_time_ns <= now_ns < burn.start_time_ns + burn.duration_ns
            )
            burn_end_ticks = (
                active_burn.start_time_ns + active_burn.duration_ns
            ) // solution.generated_at.tick_period_ns
            if burn_end_ticks < validity.expires_at.ticks:
                validity = ValidityInterval(
                    solution.generated_at,
                    replace(solution.generated_at, ticks=burn_end_ticks),
                )
        effort = RequestedEffort(
            f"translation.{selected.value}",
            RequestedEffortKind.FORCE,
            solution.generated_at,
            solution.inertial_frame,
            validity,
            force_n=tuple(float(value) for value in force),
        )
        direction = None if np.linalg.norm(force) <= 1.0e-15 else tuple(float(value) for value in _unit(force))
        secondary_tolerance = (
            self.config.eccentricity_tolerance
            if selected is TranslationMode.ORBITAL_ELEMENTS
            else self.config.velocity_tolerance_m_s
        )
        goal_satisfied = (
            explicit_goal_satisfied
            if explicit_goal_satisfied is not None
            else bool(
                position_error is not None
                and velocity_error is not None
                and position_error <= self.config.position_tolerance_m
                and velocity_error <= secondary_tolerance
            )
        )
        return TranslationControlResult(
            selected,
            effort,
            goal_satisfied,
            saturated,
            None,
            position_error,
            velocity_error,
            direction,
            phase,
        )

    def snapshot_state(self) -> dict[str, object]:
        return {
            "waypoint_index": self._waypoint_index,
            "stationkeeping_target": (
                None if self._stationkeeping_target is None else self._stationkeeping_target.tolist()
            ),
            "stationkeeping_target_time_ns": self._stationkeeping_target_time_ns,
            "atmospheric_pass_seen": self._atmospheric_pass_seen,
            "atmospheric_pass_exited": self._atmospheric_pass_exited,
            "atmospheric_recovery_start_ns": self._atmospheric_recovery_start_ns,
            "last_window_evaluation_ns": self._last_window_evaluation_ns,
            "observed_thrust_windows": sorted(self._observed_thrust_windows),
            "missed_thrust_window_count": self._missed_thrust_window_count,
            "thrust_window_open": self._thrust_window_open,
            "element_samples": [list(sample) for sample in self._element_samples],
            "ric_pd_transfer": (
                None
                if self._ric_pd_transfer is None
                else {
                    "mean_motion_rad_s": self._ric_pd_transfer.mean_motion_rad_s,
                    "guidance": self._ric_pd_transfer.snapshot_state(),
                }
            ),
        }

    def assess_goal(
        self,
        solution: OrbitNavigationSolution,
        mode: TranslationMode | str | None = None,
    ) -> bool:
        selected = (
            self.config.default_mode
            if mode is None
            else mode
            if isinstance(mode, TranslationMode)
            else TranslationMode(str(mode))
        )
        if selected is TranslationMode.SCHEDULED_BURN:
            if not self.config.scheduled_burns:
                return True
            last = self.config.scheduled_burns[-1]
            return _clock_ns(solution.generated_at) >= last.start_time_ns + last.duration_ns
        if selected is TranslationMode.ATMOSPHERIC_PASS:
            if self.config.atmospheric_pass_entry_altitude_m is None:
                return _clock_ns(solution.generated_at) >= self.config.atmospheric_raise_end_ns
            if not self._atmospheric_pass_exited or self._atmospheric_recovery_start_ns is None:
                return False
            acceleration = self.config.atmospheric_prograde_acceleration_m_s2
            recovery_duration_s = (
                0.0 if acceleration <= 0.0 else self.config.atmospheric_recovery_delta_v_m_s / acceleration
            )
            elapsed_s = (_clock_ns(solution.generated_at) - self._atmospheric_recovery_start_ns) / 1.0e9
            return elapsed_s >= recovery_duration_s
        if selected is TranslationMode.STATIONKEEPING:
            if not solution.own_state_valid:
                return False
            state = np.concatenate((solution.position_eci_m, solution.velocity_eci_m_s))
            target = self._stationkeeping_target
            if target is None:
                return False
        elif selected is TranslationMode.ORBITAL_ELEMENTS:
            if not solution.own_state_valid or self.config.target_semi_major_axis_m is None:
                return False
            position = np.asarray(solution.position_eci_m)
            velocity = np.asarray(solution.velocity_eci_m_s)
            radius = float(np.linalg.norm(position))
            mu = 3.986004418e14
            semi_major = 1.0 / (2.0 / radius - float(velocity @ velocity) / mu)
            eccentricity = float(
                np.linalg.norm(np.cross(velocity, np.cross(position, velocity)) / mu - position / radius)
            )
            return (
                abs(semi_major - self.config.target_semi_major_axis_m) <= self.config.position_tolerance_m
                and abs(eccentricity - self.config.target_eccentricity) <= self.config.eccentricity_tolerance
            )
        else:
            track = solution.relative_track(self.config.target_id)
            if track is None:
                return False
            state = np.concatenate((track.position_m, track.velocity_m_s))
            target = (
                np.asarray(self.config.waypoints_ric[-1])
                if selected is TranslationMode.WAYPOINT and self.config.waypoints_ric
                else np.asarray(self.config.target_relative_state_ric)
            )
        return self._within_tolerance(np.asarray(state), target)

    def restore_state(self, state: dict[str, object]) -> None:
        index = int(state.get("waypoint_index", 0))
        if index < 0 or (self.config.waypoints_ric and index >= len(self.config.waypoints_ric)):
            raise ValueError("translation controller waypoint index is invalid")
        self._waypoint_index = index
        target = state.get("stationkeeping_target")
        self._stationkeeping_target = None if target is None else np.asarray(target, dtype=float).reshape(6)
        target_time = state.get("stationkeeping_target_time_ns")
        self._stationkeeping_target_time_ns = None if target_time is None else int(target_time)
        self._atmospheric_pass_seen = bool(state.get("atmospheric_pass_seen", False))
        self._atmospheric_pass_exited = bool(state.get("atmospheric_pass_exited", False))
        recovery_start = state.get("atmospheric_recovery_start_ns")
        self._atmospheric_recovery_start_ns = None if recovery_start is None else int(recovery_start)
        window_time = state.get("last_window_evaluation_ns")
        self._last_window_evaluation_ns = None if window_time is None else int(window_time)
        self._observed_thrust_windows = {int(value) for value in state.get("observed_thrust_windows", ())}
        self._missed_thrust_window_count = int(state.get("missed_thrust_window_count", 0))
        if self._missed_thrust_window_count < 0:
            raise ValueError("translation controller missed thrust-window count is invalid")
        self._thrust_window_open = bool(state.get("thrust_window_open", True))
        raw_element_samples = list(state.get("element_samples", ()))
        if any(not isinstance(sample, (list, tuple)) or len(sample) != 3 for sample in raw_element_samples):
            raise ValueError("translation controller element averaging samples are invalid")
        self._element_samples = [
            (int(sample[0]), float(sample[1]), float(sample[2])) for sample in raw_element_samples
        ]
        transfer = state.get("ric_pd_transfer")
        if transfer is None:
            self._ric_pd_transfer = None
        elif isinstance(transfer, dict) and isinstance(transfer.get("guidance"), dict):
            controller = self._new_ric_pd_transfer(float(transfer["mean_motion_rad_s"]))
            controller.restore_state(transfer["guidance"])
            self._ric_pd_transfer = controller
        else:
            raise ValueError("translation controller RIC PD transfer state is invalid")

    def _stationkeeping(
        self, solution: OrbitNavigationSolution
    ) -> tuple[np.ndarray | None, float | None, float | None]:
        if not solution.own_state_valid:
            return None, None, None
        if self._stationkeeping_target is None:
            self._stationkeeping_target = np.concatenate(
                (np.asarray(solution.position_eci_m), np.asarray(solution.velocity_eci_m_s))
            )
        now_ns = _clock_ns(solution.generated_at)
        if self._stationkeeping_target_time_ns is None:
            self._stationkeeping_target_time_ns = now_ns
        elapsed_s = (now_ns - self._stationkeeping_target_time_ns) / 1.0e9
        if elapsed_s > 0.0:
            target_km = self._stationkeeping_target / 1.0e3
            remaining = elapsed_s
            while remaining > 0.0:
                step_s = min(remaining, 10.0)
                target_km = propagate_two_body_rk4(
                    target_km,
                    step_s,
                    398600.4418,
                    np.zeros(3),
                )
                remaining -= step_s
            self._stationkeeping_target = target_km * 1.0e3
            self._stationkeeping_target_time_ns = now_ns
        target = self._stationkeeping_target
        position_error = target[:3] - np.asarray(solution.position_eci_m)
        velocity_error = target[3:] - np.asarray(solution.velocity_eci_m_s)
        acceleration = self.config.kp_position_s2 * position_error + self.config.kd_velocity_s_inv * velocity_error
        return acceleration, float(np.linalg.norm(position_error)), float(np.linalg.norm(velocity_error))

    def _scheduled_burn(
        self,
        solution: OrbitNavigationSolution,
    ) -> tuple[np.ndarray | None, float | None, float | None, bool, str]:
        if not solution.own_state_valid:
            return None, None, None, False, "waiting_navigation"
        now_ns = _clock_ns(solution.generated_at)
        active = next(
            (
                burn
                for burn in self.config.scheduled_burns
                if burn.start_time_ns <= now_ns < burn.start_time_ns + burn.duration_ns
            ),
            None,
        )
        if active is not None:
            acceleration = np.asarray(active.acceleration_m_s2, dtype=float)
            if active.frame == "ric":
                acceleration = self._ric_to_eci(acceleration, solution)
            return acceleration, 1.0, 0.0, False, "finite_burn"
        future = [burn for burn in self.config.scheduled_burns if burn.start_time_ns > now_ns]
        if future:
            return np.zeros(3), float(future[0].start_time_ns - now_ns), 0.0, False, "coast_to_burn"
        return np.zeros(3), 0.0, 0.0, True, "complete"

    def _orbital_elements(
        self, solution: OrbitNavigationSolution
    ) -> tuple[np.ndarray | None, float | None, float | None]:
        if not solution.own_state_valid or self.config.target_semi_major_axis_m is None:
            return None, None, None
        target_coes = dict(self.config.target_coes)
        if self.config.orbital_element_control_law == "current_anomaly_stationkeep":
            state_km = np.concatenate(
                (np.asarray(solution.position_eci_m) / 1.0e3, np.asarray(solution.velocity_eci_m_s) / 1.0e3)
            )
            desired_km = coes_target_state_at_current_true_anomaly(target_coes, state_km)
            position_error = desired_km[:3] * 1.0e3 - np.asarray(solution.position_eci_m)
            velocity_error = desired_km[3:] * 1.0e3 - np.asarray(solution.velocity_eci_m_s)
            acceleration = self.config.kp_position_s2 * position_error + self.config.kd_velocity_s_inv * velocity_error
            return acceleration, float(np.linalg.norm(position_error)), float(np.linalg.norm(velocity_error))
        if self.config.orbital_element_control_law == "element_tracking":
            state_km = np.concatenate(
                (np.asarray(solution.position_eci_m) / 1.0e3, np.asarray(solution.velocity_eci_m_s) / 1.0e3)
            )
            result = orbital_element_feedback_accel(
                state_km,
                target_coes,
                controlled_elements=self.config.controlled_elements,
                energy_gain_per_s=self.config.energy_gain_per_s,
                eccentricity_gain_per_s=self.config.eccentricity_gain_per_s,
                plane_gain_per_s=self.config.plane_gain_per_s,
                max_accel_km_s2=self.config.max_acceleration_m_s2 / 1.0e3,
            )
            target_a_m = float(target_coes.get("a_km", target_coes.get("semi_major_axis_km", 0.0))) * 1.0e3
            return (
                np.asarray(result.accel_eci_km_s2) * 1.0e3,
                abs(float(result.current_coes.a_km) * 1.0e3 - target_a_m),
                abs(float(result.current_coes.ecc) - float(target_coes.get("ecc", target_coes.get("e", 0.0)))),
            )
        position = np.asarray(solution.position_eci_m)
        velocity = np.asarray(solution.velocity_eci_m_s)
        mu = 3.986004418e14
        radius = float(np.linalg.norm(position))
        speed_squared = float(velocity @ velocity)
        if radius <= 0.0:
            return None, None, None
        semi_major = 1.0 / (2.0 / radius - speed_squared / mu)
        h = np.cross(position, velocity)
        eccentricity_vector = np.cross(velocity, h) / mu - position / radius
        eccentricity = float(np.linalg.norm(eccentricity_vector))
        now_ns = _clock_ns(solution.generated_at)
        self._element_samples.append((now_ns, semi_major, eccentricity))
        window_ns = int(round(self.config.element_averaging_window_s * 1.0e9))
        if window_ns > 0:
            cutoff_ns = now_ns - window_ns
            self._element_samples = [sample for sample in self._element_samples if sample[0] >= cutoff_ns]
        else:
            self._element_samples = self._element_samples[-1:]
        averaged_semi_major = float(np.mean([sample[1] for sample in self._element_samples]))
        averaged_eccentricity = float(np.mean([sample[2] for sample in self._element_samples]))
        a_error = self.config.target_semi_major_axis_m - averaged_semi_major
        e_error = self.config.target_eccentricity - averaged_eccentricity
        tangential = _unit(velocity)
        radial = _unit(position)
        current_energy = -mu / (2.0 * averaged_semi_major)
        target_energy = -mu / (2.0 * self.config.target_semi_major_axis_m)
        speed = sqrt(speed_squared)
        # Both terms are native SI accelerations.  Specific orbital-energy
        # error [m^2/s^2] divided by speed [m/s] gives velocity error [m/s];
        # the configured 1/s gain closes it as m/s^2.  Eccentricity is
        # dimensionless, so multiplying by speed before the same gain yields
        # the same acceleration unit.
        acceleration = (
            self.config.kd_velocity_s_inv * (target_energy - current_energy) / max(speed, 1.0) * tangential
            + self.config.kd_velocity_s_inv * e_error * speed * radial
        )
        return acceleration, abs(a_error), abs(e_error)

    def _evaluate_thrust_window(self, now_ns: int) -> bool:
        period_ns = int(round(self.config.thrust_window_period_s * 1.0e9))
        if period_ns <= 0:
            self._last_window_evaluation_ns = now_ns
            return True
        duration_ns = int(round(self.config.thrust_window_duration_s * 1.0e9))
        phase_ns = int(round(self.config.thrust_window_phase_s * 1.0e9))
        previous_ns = self._last_window_evaluation_ns
        if previous_ns is not None and now_ns > previous_ns:
            first_closed = max(0, (previous_ns - phase_ns - duration_ns) // period_ns + 1)
            last_closed = (now_ns - phase_ns - duration_ns) // period_ns
            for index in range(first_closed, last_closed + 1):
                if index not in self._observed_thrust_windows:
                    self._missed_thrust_window_count += 1
                self._observed_thrust_windows.discard(index)
        relative_ns = now_ns - phase_ns
        if relative_ns < 0:
            open_now = False
        else:
            index, offset_ns = divmod(relative_ns, period_ns)
            open_now = offset_ns < duration_ns
            if open_now:
                self._observed_thrust_windows.add(index)
        self._last_window_evaluation_ns = now_ns
        return open_now

    def _atmospheric_pass(
        self, solution: OrbitNavigationSolution
    ) -> tuple[np.ndarray | None, float | None, float | None, str]:
        if not solution.own_state_valid:
            return None, None, None, "waiting_navigation"
        now_ns = _clock_ns(solution.generated_at)
        radius_m = float(np.linalg.norm(solution.position_eci_m))
        altitude_m = radius_m - 6_378_137.0
        if self.config.atmospheric_pass_entry_altitude_m is not None:
            entry_altitude = self.config.atmospheric_pass_entry_altitude_m
            exit_altitude = self.config.atmospheric_pass_exit_altitude_m
            assert exit_altitude is not None
            if altitude_m <= entry_altitude:
                self._atmospheric_pass_seen = True
                self._atmospheric_pass_exited = False
                self._atmospheric_recovery_start_ns = None
                return np.zeros(3), max(entry_altitude - altitude_m, 0.0), 0.0, "atmospheric_pass"
            if self._atmospheric_pass_seen and not self._atmospheric_pass_exited and altitude_m >= exit_altitude:
                self._atmospheric_pass_exited = True
                self._atmospheric_recovery_start_ns = now_ns
            if not self._atmospheric_pass_exited:
                return np.zeros(3), abs(altitude_m - entry_altitude), 0.0, "awaiting_pass"
            acceleration = self.config.atmospheric_prograde_acceleration_m_s2
            recovery_duration_s = (
                0.0 if acceleration <= 0.0 else self.config.atmospheric_recovery_delta_v_m_s / acceleration
            )
            elapsed_s = (now_ns - int(self._atmospheric_recovery_start_ns or now_ns)) / 1.0e9
            if elapsed_s >= recovery_duration_s:
                return np.zeros(3), 0.0, 0.0, "recovery_complete"
            velocity = np.asarray(solution.velocity_eci_m_s, dtype=float)
            return (
                _unit(velocity) * acceleration,
                max(self.config.atmospheric_recovery_delta_v_m_s - acceleration * elapsed_s, 0.0),
                0.0,
                "post_pass_recovery",
            )
        active = (
            self.config.atmospheric_raise_start_ns <= now_ns < self.config.atmospheric_raise_end_ns
            and altitude_m >= self.config.atmospheric_min_raise_altitude_m
        )
        if not active:
            return np.zeros(3), abs(altitude_m - self.config.atmospheric_min_raise_altitude_m), 0.0, "coast"
        velocity = np.asarray(solution.velocity_eci_m_s, dtype=float)
        return (
            _unit(velocity) * self.config.atmospheric_prograde_acceleration_m_s2,
            abs(altitude_m - self.config.atmospheric_min_raise_altitude_m),
            0.0,
            "raise_burn",
        )

    def _relative(
        self,
        mode: TranslationMode,
        solution: OrbitNavigationSolution,
        track: RelativeStateEstimateSI | None,
    ) -> tuple[np.ndarray | None, float | None, float | None, str]:
        if track is None:
            return None, None, None, mode.value
        if mode is TranslationMode.RIC_PD_TRANSFER:
            return self._ric_pd_transfer_guidance(solution, track)
        state = np.concatenate((track.position_m, track.velocity_m_s))
        target = np.asarray(self.config.target_relative_state_ric)
        phase = mode.value
        if mode is TranslationMode.WAYPOINT:
            if not self.config.waypoints_ric:
                return None, None, None, "waypoint_unconfigured"
            target = np.asarray(self.config.waypoints_ric[self._waypoint_index])
            if self._within_tolerance(state, target) and self._waypoint_index + 1 < len(self.config.waypoints_ric):
                self._waypoint_index += 1
                target = np.asarray(self.config.waypoints_ric[self._waypoint_index])
            phase = f"waypoint_{self._waypoint_index}"
        position_error_vector = state[:3] - target[:3]
        desired_velocity = np.asarray(target[3:])
        if mode in (
            TranslationMode.R_BAR_APPROACH,
            TranslationMode.V_BAR_APPROACH,
            TranslationMode.C_BAR_APPROACH,
            TranslationMode.LOW_THRUST_PHASING,
        ):
            axis = {
                TranslationMode.R_BAR_APPROACH: 0,
                TranslationMode.V_BAR_APPROACH: 1,
                TranslationMode.C_BAR_APPROACH: 2,
                TranslationMode.LOW_THRUST_PHASING: 1,
            }[mode]
            remaining = target[axis] - state[axis]
            ramp = min(abs(remaining) / self.config.slowdown_distance_m, 1.0)
            speed = self.config.approach_speed_m_s * (0.1 if mode is TranslationMode.LOW_THRUST_PHASING else 1.0)
            desired_velocity = desired_velocity.copy()
            desired_velocity[axis] += np.sign(remaining) * speed * ramp
            phase = "terminal_braking" if ramp < 1.0 else "approach"
        if mode is TranslationMode.TERMINAL_BRAKING:
            range_error = float(np.linalg.norm(position_error_vector))
            los = _unit(position_error_vector) if range_error > 1.0e-12 else np.zeros(3)
            closing = float(-np.dot(state[3:] - target[3:], los))
            if closing > self.config.terminal_max_closing_speed_m_s:
                desired_velocity = target[3:] - los * self.config.terminal_max_closing_speed_m_s
            phase = "terminal_box" if range_error <= self.config.terminal_box_m else "braking"
        if mode is TranslationMode.PASSIVE_RETREAT:
            range_m = float(np.linalg.norm(state[:3]))
            outward = _unit(state[:3]) if range_m > 1.0e-12 else np.array([0.0, -1.0, 0.0])
            outward_speed = float(np.dot(state[3:], outward))
            if range_m >= self.config.retreat_coast_range_m and outward_speed >= 0.9 * self.config.retreat_speed_m_s:
                acceleration_ric = np.zeros(3)
                phase = "passive_coast"
            else:
                acceleration_ric = (
                    self.config.kd_velocity_s_inv * (self.config.retreat_speed_m_s - outward_speed) * outward
                )
                phase = "retreat_burn"
            return (
                self._ric_to_eci(acceleration_ric, solution, track),
                max(self.config.retreat_coast_range_m - range_m, 0.0),
                max(self.config.retreat_speed_m_s - outward_speed, 0.0),
                phase,
            )
        if mode is TranslationMode.LOW_THRUST_PHASING:
            # Relative phase is not a translational straight-line error over
            # long arcs.  Prograde thrust raises the deputy orbit and reduces
            # its mean motion; retrograde thrust does the opposite.  The
            # positive feedback sign below is therefore intentional: through
            # the orbit-energy/mean-motion coupling it produces restoring
            # phase acceleration, while the drift-rate term damps the return.
            acceleration_ric = np.zeros(3)
            phase_error_m = float(state[1] - target[1])
            drift_error_m_s = float(state[4] - target[4])
            acceleration_ric[1] = (
                self.config.kp_position_s2 * phase_error_m
                + self.config.kd_velocity_s_inv * drift_error_m_s
            )
            # Keep the non-phasing axes locally bounded without competing
            # with the mean-motion maneuver on the in-track axis.
            acceleration_ric[0] = -0.25 * (
                self.config.kp_position_s2 * float(state[0] - target[0])
                + self.config.kd_velocity_s_inv * float(state[3] - target[3])
            )
            acceleration_ric[2] = -0.25 * (
                self.config.kp_position_s2 * float(state[2] - target[2])
                + self.config.kd_velocity_s_inv * float(state[5] - target[5])
            )
            phase = "raise_to_reduce_phase" if acceleration_ric[1] > 0.0 else "lower_to_increase_phase"
            return (
                self._ric_to_eci(acceleration_ric, solution, track),
                float(np.linalg.norm(position_error_vector)),
                float(np.linalg.norm(state[3:] - target[3:])),
                phase,
            )
        velocity_error_vector = state[3:] - desired_velocity
        acceleration_ric, law_phase = self._relative_feedback_acceleration_ric(
            state,
            target,
            solution,
            track,
        )
        phase = law_phase if law_phase is not None else phase
        return (
            self._ric_to_eci(acceleration_ric, solution, track),
            float(np.linalg.norm(position_error_vector)),
            float(np.linalg.norm(velocity_error_vector)),
            phase,
        )

    def _ric_pd_transfer_guidance(
        self,
        solution: OrbitNavigationSolution,
        track: RelativeStateEstimateSI,
    ) -> tuple[np.ndarray | None, float | None, float | None, str]:
        chief_position = track.chief_position_eci_m or solution.position_eci_m
        chief_velocity = track.chief_velocity_eci_m_s or solution.velocity_eci_m_s
        if chief_position is None or chief_velocity is None:
            return None, None, None, "waiting_chief_state"
        chief_position_m = np.asarray(chief_position, dtype=float)
        chief_velocity_m_s = np.asarray(chief_velocity, dtype=float)
        chief_radius_m = float(np.linalg.norm(chief_position_m))
        if chief_radius_m <= 0.0:
            return None, None, None, "waiting_chief_state"
        if self._ric_pd_transfer is None:
            mean_motion = self.config.mean_motion_rad_s or self._relative_mean_motion(solution, track)
            self._ric_pd_transfer = self._new_ric_pd_transfer(mean_motion)
        state_rect_km = np.concatenate(
            (
                np.asarray(track.position_m, dtype=float) / 1.0e3,
                np.asarray(track.velocity_m_s, dtype=float) / 1.0e3,
            )
        )
        now_s = _clock_ns(solution.generated_at) / 1.0e9
        guidance = self._ric_pd_transfer.guide_relative_state(
            state_rect_km,
            chief_position_m / 1.0e3,
            chief_velocity_m_s / 1.0e3,
            t_s=now_s,
        )
        state_si = np.concatenate((track.position_m, track.velocity_m_s))
        target_si = np.asarray(self.config.target_relative_state_ric, dtype=float)
        phase = str(guidance.mode_flags.get("phase", "ric_pd_transfer"))
        return (
            np.asarray(guidance.acceleration_eci_km_s2, dtype=float) * 1.0e3,
            float(np.linalg.norm(state_si[:3] - target_si[:3])),
            float(np.linalg.norm(state_si[3:] - target_si[3:])),
            phase,
        )

    def _new_ric_pd_transfer(self, mean_motion_rad_s: float) -> RICPDTransferController:
        return RICPDTransferController(
            # TranslationController owns the authoritative SI acceleration
            # limit and saturation evidence for every subordinate law.
            max_accel_km_s2=float("inf"),
            mean_motion_rad_s=float(mean_motion_rad_s),
            transfer_time_s=self.config.transfer_time_s,
            burn_time_constant_s=self.config.burn_time_constant_s,
            correction_interval_s=self.config.correction_interval_s,
            velocity_deadband_m_s=self.config.velocity_deadband_m_s,
            final_brake_start_s=self.config.final_brake_start_s,
            terminal_start_s=self.config.terminal_start_s,
            terminal_range_km=self.config.terminal_range_m / 1.0e3,
            terminal_kp=np.eye(3) * self.config.kp_position_s2,
            terminal_kd=np.eye(3) * self.config.kd_velocity_s_inv,
            desired_state_ric=np.asarray(self.config.target_relative_state_ric, dtype=float) / 1.0e3,
        )

    def _relative_feedback_acceleration_ric(
        self,
        state: np.ndarray,
        target: np.ndarray,
        solution: OrbitNavigationSolution,
        track: RelativeStateEstimateSI,
    ) -> tuple[np.ndarray, str | None]:
        error = np.asarray(state, dtype=float) - np.asarray(target, dtype=float)
        law = self.config.control_law
        if law is TranslationControlLaw.HCW_LQR:
            mean_motion = self._relative_mean_motion(solution, track)
            gain = self._lqr_gain(mean_motion)
            # The gain is designed against the complete HCW plant, including
            # its natural drift terms.  Adding drift cancellation here would
            # double-count those dynamics and can reverse weak subsecond
            # cross-track feedback.
            return -(gain @ error), "hcw_lqr"
        if law is TranslationControlLaw.CURVILINEAR_RIC_PD:
            radius_m = self._chief_radius_m(solution, track)
            current_curv_km = ric_rect_to_curv(np.asarray(state, dtype=float) / 1.0e3, radius_m / 1.0e3)
            target_curv_km = ric_rect_to_curv(np.asarray(target, dtype=float) / 1.0e3, radius_m / 1.0e3)
            error_curv = current_curv_km - target_curv_km
            acceleration_curv_km_s2 = -(
                self.config.kp_position_s2 * error_curv[:3] + self.config.kd_velocity_s_inv * error_curv[3:]
            )
            acceleration_rect_km_s2 = curv_accel_to_rect(
                acceleration_curv_km_s2,
                position_curv_km=current_curv_km[:3],
                r0_km=radius_m / 1.0e3,
            )
            return np.asarray(acceleration_rect_km_s2) * 1.0e3, "curvilinear_ric_pd"
        if law is TranslationControlLaw.RMOE_IF_THEN:
            return self._rmoe_acceleration_ric(state, solution, track)
        acceleration = -(self.config.kp_position_s2 * error[:3] + self.config.kd_velocity_s_inv * error[3:])
        if self.config.mean_motion_rad_s > 0.0:
            n = self.config.mean_motion_rad_s
            acceleration += self._hcw_drift_cancellation(state, n)
        return acceleration, None

    @staticmethod
    def _hcw_drift_cancellation(state: np.ndarray, mean_motion_rad_s: float) -> np.ndarray:
        x, _y, z, xd, yd, _zd = np.asarray(state, dtype=float)
        n = float(mean_motion_rad_s)
        return np.array([-3.0 * n * n * x - 2.0 * n * yd, 2.0 * n * xd, n * n * z])

    def _lqr_gain(self, mean_motion_rad_s: float) -> np.ndarray:
        cache_key = round(float(mean_motion_rad_s), 15)
        cached = self._lqr_cache.get(cache_key)
        if cached is not None:
            return cached
        dynamics = RelativeLinearDynamics(model="hcw", mean_motion_rad_s=mean_motion_rad_s)
        ad, bd = dynamics.discrete_matrices(self.config.control_design_dt_s)
        q = np.diag(np.asarray(self.config.lqr_q_weights, dtype=float))
        r = np.diag(np.asarray(self.config.lqr_r_weights, dtype=float))
        gain = solve_discrete_lqr_gain(ad, bd, q, r)
        self._lqr_cache[cache_key] = gain
        return gain

    def _rmoe_acceleration_ric(
        self,
        state: np.ndarray,
        solution: OrbitNavigationSolution,
        track: RelativeStateEstimateSI,
    ) -> tuple[np.ndarray, str]:
        n = self._relative_mean_motion(solution, track)
        state_km = np.asarray(state, dtype=float) / 1.0e3
        rmoe = estimate_rmoes_from_rect_ric(state_km, n)
        target_radial_km = self.config.rmoe_target_radial_center_m / 1.0e3
        target_intrack_km = self.config.rmoe_target_in_track_center_m / 1.0e3
        target_drift_km_s = self.config.rmoe_target_in_track_drift_rate_m_s / 1.0e3
        target_cross_km = self.config.rmoe_target_cross_track_amplitude_m / 1.0e3
        max_drift_km_s = self.config.rmoe_max_drift_rate_m_s / 1.0e3
        close_zone_km = self.config.rmoe_close_zone_m / 1.0e3
        in_track_error_km = float(state_km[1]) - target_intrack_km
        if close_zone_km > 0.0 and abs(in_track_error_km) <= close_zone_km:
            desired_drift = target_drift_km_s - max_drift_km_s * float(
                np.clip(in_track_error_km / close_zone_km, -1.0, 1.0)
            )
        elif in_track_error_km > 0.0:
            desired_drift = target_drift_km_s - max_drift_km_s
        else:
            desired_drift = target_drift_km_s + max_drift_km_s
        acceleration_km_s2 = np.zeros(3)
        phase = "rmoe_coast"
        drift = float(rmoe["in_track_drift_rate_km_s"])
        if abs(drift) > max_drift_km_s + 1.0e-6:
            desired_drift = float(np.clip(drift, -max_drift_km_s, max_drift_km_s))
            phase = "rmoe_limit_drift"
        if abs(desired_drift - drift) > 1.0e-6:
            desired_radial_center = -desired_drift / (3.0 * max(n, 1.0e-12))
            acceleration_km_s2[1] = -2.0e-3 * (desired_radial_center - float(rmoe["radial_center_km"]))
            if phase == "rmoe_coast":
                phase = "rmoe_shape_drift"
        elif abs(target_radial_km - float(rmoe["radial_center_km"])) > 0.01:
            acceleration_km_s2[1] = -1.0e-6 * (target_radial_km - float(rmoe["radial_center_km"]))
            phase = "rmoe_trim_radial_center"
        if abs(float(state_km[2])) <= self.config.rmoe_cross_track_burn_gate_m / 1.0e3:
            cross_error = target_cross_km - float(rmoe["cross_track_amplitude_km"])
            if abs(cross_error) > 0.02:
                rate_sign = float(np.sign(state_km[5])) or -float(np.sign(state_km[2]))
                acceleration_km_s2[2] = 1.0e-6 * cross_error * rate_sign
                if phase == "rmoe_coast":
                    phase = "rmoe_trim_cross_track_amplitude"
        in_track_center_error = target_intrack_km - float(rmoe["in_track_center_km"])
        if abs(in_track_center_error) > 0.05:
            acceleration_km_s2[0] = -1.0e-6 * in_track_center_error
            if phase == "rmoe_coast":
                phase = "rmoe_trim_in_track_center"
        return acceleration_km_s2 * 1.0e3, phase

    def _chief_radius_m(
        self,
        solution: OrbitNavigationSolution,
        track: RelativeStateEstimateSI,
    ) -> float:
        position = track.chief_position_eci_m if track.chief_position_eci_m is not None else solution.position_eci_m
        return max(float(np.linalg.norm(position)), 1.0)

    def _relative_mean_motion(
        self,
        solution: OrbitNavigationSolution,
        track: RelativeStateEstimateSI,
    ) -> float:
        if self.config.mean_motion_rad_s > 0.0:
            return self.config.mean_motion_rad_s
        return sqrt(3.986004418e14 / self._chief_radius_m(solution, track) ** 3)

    def _within_tolerance(self, state: np.ndarray, target: np.ndarray) -> bool:
        return bool(
            np.linalg.norm(state[:3] - target[:3]) <= self.config.position_tolerance_m
            and np.linalg.norm(state[3:] - target[3:]) <= self.config.velocity_tolerance_m_s
        )

    def _ric_to_eci(
        self,
        vector_ric: np.ndarray,
        solution: OrbitNavigationSolution,
        track: RelativeStateEstimateSI | None = None,
    ) -> np.ndarray | None:
        if not solution.own_state_valid:
            return None
        # RPO references are defined in the tracked chief's RIC frame.  Ideal
        # and state-vector navigation carries the chief basis explicitly.  A
        # range/LOS-only track falls back to the deputy basis and is marked as
        # degraded by navigation; that approximation is appropriate only for
        # close-proximity operations.
        position = np.asarray(
            track.chief_position_eci_m
            if track is not None and track.chief_position_eci_m is not None
            else solution.position_eci_m
        )
        velocity = np.asarray(
            track.chief_velocity_eci_m_s
            if track is not None and track.chief_velocity_eci_m_s is not None
            else solution.velocity_eci_m_s
        )
        radial = _unit(position)
        cross_track = _unit(np.cross(position, velocity))
        in_track = _unit(np.cross(cross_track, radial))
        c_ri = np.vstack((radial, in_track, cross_track))
        return c_ri.T @ np.asarray(vector_ric, dtype=float)


class TranslationAllocatorKind(str, Enum):
    IDEAL_WRENCH = "ideal_wrench"
    CONTINUOUS_ENGINE = "continuous_engine"
    RCS_PULSE = "rcs_pulse"
    RCS_ON_OFF = "rcs_on_off"


@dataclass(frozen=True, slots=True)
class RcsThrusterBelief:
    thruster_id: str
    force_direction_body: tuple[float, float, float]
    max_thrust_n: float

    def __post_init__(self) -> None:
        if not self.thruster_id.strip():
            raise ValueError("thruster_id must be non-empty")
        direction = np.asarray(self.force_direction_body, dtype=float)
        if direction.size != 3 or not np.all(np.isfinite(direction)) or abs(np.linalg.norm(direction) - 1.0) > 1e-10:
            raise ValueError("force_direction_body must be normalized")
        if not isfinite(self.max_thrust_n) or self.max_thrust_n <= 0.0:
            raise ValueError("max_thrust_n must be finite and positive")


@dataclass(frozen=True, slots=True)
class TranslationAllocatorConfig:
    satellite_id: str
    kind: TranslationAllocatorKind
    actuator_id: str
    actuator_frame: FrameId
    max_force_n: float
    rcs_thrusters: tuple[RcsThrusterBelief, ...] = ()
    pulse_window_s: float = 1.0
    gimbal_limit_rad: float = 1.5707963267948966

    def __post_init__(self) -> None:
        if not self.satellite_id.strip() or not self.actuator_id.strip():
            raise ValueError("satellite_id and actuator_id must be non-empty")
        if not isinstance(self.kind, TranslationAllocatorKind):
            raise TypeError("kind must be TranslationAllocatorKind")
        if not isfinite(self.max_force_n) or self.max_force_n <= 0.0:
            raise ValueError("max_force_n must be finite and positive")
        if not isfinite(self.pulse_window_s) or self.pulse_window_s <= 0.0:
            raise ValueError("pulse_window_s must be finite and positive")
        if not isfinite(self.gimbal_limit_rad) or self.gimbal_limit_rad < 0.0:
            raise ValueError("gimbal_limit_rad must be finite and nonnegative")
        if (
            self.kind in (TranslationAllocatorKind.RCS_PULSE, TranslationAllocatorKind.RCS_ON_OFF)
            and not self.rcs_thrusters
        ):
            raise ValueError("RCS allocation requires at least one thruster belief")
        thruster_ids = [thruster.thruster_id for thruster in self.rcs_thrusters]
        if len(thruster_ids) != len(set(thruster_ids)):
            raise ValueError("RCS thruster IDs must be unique")


class TranslationAllocator:
    def __init__(self, config: TranslationAllocatorConfig) -> None:
        self.config = config

    def allocate(
        self,
        effort: RequestedEffort,
        solution: OrbitNavigationSolution,
        *,
        next_command_id: Callable[[], PacketId],
        unavailable_actuators: frozenset[str] = frozenset(),
    ) -> AllocationResult:
        if effort.force_n is None:
            return AllocationResult(effort.effort_id, effort.generated_at, AllocationStatus.INVALID)
        requested_eci = np.asarray(effort.force_n, dtype=float)
        requested_norm = float(np.linalg.norm(requested_eci))
        status = AllocationStatus.EXACT
        commands: list[ActuatorCommand] = []
        achieved_eci = np.zeros(3)
        details: list[TelemetryField] = [TelemetryField("allocator_requested_force_n", requested_norm, "N")]
        if (
            self.config.kind in (TranslationAllocatorKind.IDEAL_WRENCH, TranslationAllocatorKind.CONTINUOUS_ENGINE)
            and self.config.actuator_id in unavailable_actuators
        ):
            return AllocationResult(effort.effort_id, effort.generated_at, AllocationStatus.INFEASIBLE)
        if self.config.kind is TranslationAllocatorKind.IDEAL_WRENCH:
            attitude = solution.attitude.attitude_quat_bn
            if attitude is None:
                return AllocationResult(effort.effort_id, effort.generated_at, AllocationStatus.INFEASIBLE)
            scale = min(1.0, self.config.max_force_n / requested_norm) if requested_norm > 0.0 else 1.0
            achieved_eci = requested_eci * scale
            force_body = quaternion_to_dcm_bn(np.asarray(attitude)) @ achieved_eci
            if scale < 1.0:
                status = AllocationStatus.SATURATED
            commands.append(
                self._command(
                    next_command_id(),
                    effort,
                    IdealWrenchCommand(tuple(float(value) for value in force_body), (0.0, 0.0, 0.0)),
                )
            )
        elif self.config.kind is TranslationAllocatorKind.CONTINUOUS_ENGINE:
            attitude = solution.attitude.attitude_quat_bn
            if attitude is None:
                return AllocationResult(effort.effort_id, effort.generated_at, AllocationStatus.INFEASIBLE)
            c_bn = quaternion_to_dcm_bn(np.asarray(attitude))
            body_force = c_bn @ requested_eci
            force_norm = float(np.linalg.norm(body_force))
            throttle = min(force_norm / self.config.max_force_n, 1.0)
            yaw = atan2(body_force[1], body_force[0]) if force_norm > 0.0 else 0.0
            pitch = atan2(-body_force[2], sqrt(body_force[0] ** 2 + body_force[1] ** 2)) if force_norm > 0.0 else 0.0
            clipped_yaw = float(np.clip(yaw, -self.config.gimbal_limit_rad, self.config.gimbal_limit_rad))
            clipped_pitch = float(np.clip(pitch, -self.config.gimbal_limit_rad, self.config.gimbal_limit_rad))
            gimbal_saturated = clipped_yaw != yaw or clipped_pitch != pitch
            if throttle < force_norm / self.config.max_force_n or gimbal_saturated:
                status = AllocationStatus.SATURATED
            body_direction = np.array(
                [
                    np.cos(clipped_pitch) * np.cos(clipped_yaw),
                    np.cos(clipped_pitch) * np.sin(clipped_yaw),
                    -np.sin(clipped_pitch),
                ]
            )
            achieved_eci = c_bn.T @ (body_direction * throttle * self.config.max_force_n)
            commands.append(
                self._command(
                    next_command_id(),
                    effort,
                    ContinuousEngineCommand(throttle, (clipped_yaw, clipped_pitch)),
                )
            )
            details.extend(
                (
                    TelemetryField("continuous_engine_throttle", throttle),
                    TelemetryField("continuous_engine_gimbal_yaw_rad", clipped_yaw, "rad"),
                    TelemetryField("continuous_engine_gimbal_pitch_rad", clipped_pitch, "rad"),
                    TelemetryField("continuous_engine_gimbal_saturated", gimbal_saturated),
                )
            )
        else:
            attitude = solution.attitude.attitude_quat_bn
            if attitude is None:
                return AllocationResult(effort.effort_id, effort.generated_at, AllocationStatus.INFEASIBLE)
            c_bn = quaternion_to_dcm_bn(np.asarray(attitude))
            force_scale = min(1.0, self.config.max_force_n / requested_norm) if requested_norm > 0.0 else 1.0
            allocation_target_eci = requested_eci * force_scale
            requested_body = c_bn @ allocation_target_eci
            available_thrusters = tuple(
                thruster for thruster in self.config.rcs_thrusters if thruster.thruster_id not in unavailable_actuators
            )
            if not available_thrusters:
                return AllocationResult(effort.effort_id, effort.generated_at, AllocationStatus.INFEASIBLE)
            directions = np.column_stack([thruster.force_direction_body for thruster in available_thrusters])
            maxima = np.asarray([thruster.max_thrust_n for thruster in available_thrusters])
            forces = _bounded_nonnegative_lstsq(directions, requested_body, maxima)
            realized_forces = (
                np.where(forces > 1.0e-12, maxima, 0.0)
                if self.config.kind is TranslationAllocatorKind.RCS_ON_OFF
                else forces
            )
            achieved_body = directions @ realized_forces
            achieved_eci = c_bn.T @ achieved_body
            residual = requested_eci - achieved_eci
            status = AllocationStatus.EXACT if np.linalg.norm(residual) <= 1e-10 else AllocationStatus.RESIDUAL
            if force_scale < 1.0 or (np.any(realized_forces >= maxima - 1e-12) and np.linalg.norm(residual) > 1e-10):
                status = AllocationStatus.SATURATED
            for thruster, force, maximum in zip(available_thrusters, forces, maxima, strict=True):
                if self.config.kind is TranslationAllocatorKind.RCS_PULSE:
                    duration = self.config.pulse_window_s * float(force / maximum)
                    if duration <= 1e-12:
                        continue
                    payload = ThrusterPulseCommand(thruster.thruster_id, effort.generated_at, duration)
                    details.append(
                        TelemetryField(
                            f"requested_impulse.{thruster.thruster_id}_n_s",
                            float(maximum * duration),
                            "N*s",
                        )
                    )
                else:
                    payload = ThrusterOnOffCommand(thruster.thruster_id, bool(force > 1e-12))
                commands.append(self._command(next_command_id(), effort, payload))
        residual = requested_eci - achieved_eci
        details.append(TelemetryField("residual_force_n", float(np.linalg.norm(residual)), "N"))
        return AllocationResult(
            effort.effort_id,
            effort.generated_at,
            status,
            tuple(commands),
            residual_force_n=tuple(float(value) for value in residual),
            status_details=tuple(details),
        )

    def _command(
        self,
        command_id: PacketId,
        effort: RequestedEffort,
        payload: IdealWrenchCommand | ContinuousEngineCommand | ThrusterPulseCommand | ThrusterOnOffCommand,
    ) -> ActuatorCommand:
        actuator_id = (
            payload.thruster_id
            if isinstance(payload, (ThrusterPulseCommand, ThrusterOnOffCommand))
            else self.config.actuator_id
        )
        return ActuatorCommand(
            command_id,
            self.config.satellite_id,
            actuator_id,
            effort.generated_at,
            effort.validity,
            self.config.actuator_frame,
            payload,
        )


def _bounded_nonnegative_lstsq(matrix: np.ndarray, target: np.ndarray, upper: np.ndarray) -> np.ndarray:
    result, *_ = np.linalg.lstsq(matrix, target, rcond=None)
    result = np.clip(result, 0.0, upper)
    for _ in range(matrix.shape[1]):
        free = (result > 0.0) & (result < upper)
        fixed = ~free
        if not np.any(free):
            break
        residual = target - matrix[:, fixed] @ result[fixed]
        candidate, *_ = np.linalg.lstsq(matrix[:, free], residual, rcond=None)
        updated = result.copy()
        updated[free] = np.clip(candidate, 0.0, upper[free])
        if np.allclose(updated, result, rtol=0.0, atol=1e-14):
            break
        result = updated
    return result


def _unit(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(value))
    if not isfinite(norm) or norm <= 0.0:
        raise ValueError("vector must be finite and nonzero")
    return value / norm


def _finite_vector(name: str, vector: tuple[float, ...], size: int) -> None:
    if len(vector) != size or not all(isfinite(float(value)) for value in vector):
        raise ValueError(f"{name} must contain {size} finite values")


def _add_ticks(tag: ClockTag, ticks: int) -> ClockTag:
    return ClockTag(tag.clock_id, tag.ticks + ticks, tag.tick_period_ns, tag.scale, tag.validity, tag.reset_counter)


def _clock_ns(tag: ClockTag) -> int:
    return int(tag.ticks) * int(tag.tick_period_ns)
