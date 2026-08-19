"""Truth-free pilot and operator reference flight software for OEL games."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

import numpy as np

from sim.gnc.attitude_v2 import (
    AttitudeAllocator,
    AttitudeAllocatorConfig,
    AttitudeReferenceConfig,
    AttitudeReferenceGenerator,
    AttitudeReferenceMode,
    QuaternionTorqueController,
)
from sim.gnc.contracts import GuidanceReference, RequestedEffort, RequestedEffortKind
from sim.gnc.navigation_v2 import NavigationInitializationMode, OrbitNavigationSolution, OrbitNavigator
from sim.gnc.orbit_v2 import TranslationAllocator, TranslationAllocatorConfig, TranslationAllocatorKind
from sim.utils.quaternion import (
    normalize_quaternion,
    quaternion_delta_from_body_rate,
    quaternion_multiply,
    quaternion_to_dcm_bn,
)

from .contracts import (
    ActuatorCommand,
    AerodynamicEffectorPositionCommand,
    ClockTag,
    DiagnosticTelemetry,
    FlightSoftwareInputBatch,
    FlightSoftwareOutput,
    FrameId,
    GroundCommandKind,
    GroundCommandPayload,
    IdealTrackedObjectStateMeasurement,
    IdealWrenchCommand,
    InputKind,
    MeasurementEvent,
    PacketId,
    PilotInputPayload,
    TelemetryField,
    ValidityInterval,
)
from .reference_stacks import ReferenceStackBase
from .schemas import from_primitive, to_primitive


class GamePilotMode(str, Enum):
    TRANSLATION = "translation"
    DIRECT_ECI = "direct_eci"
    ATTITUDE_THRUST = "attitude_thrust"
    AERODYNAMIC = "aerodynamic"


@dataclass(frozen=True, slots=True)
class GamePilotInputProfile:
    profile_id: str
    mode: GamePilotMode
    radial_axis: str = "translate_r"
    in_track_axis: str = "translate_i"
    cross_track_axis: str = "translate_c"
    roll_axis: str = "roll"
    pitch_axis: str = "pitch"
    yaw_axis: str = "yaw"
    throttle_axis: str = "throttle"
    firing_action: str = "fire"

    def __post_init__(self) -> None:
        if not self.profile_id.strip():
            raise ValueError("profile_id must be non-empty")
        if not isinstance(self.mode, GamePilotMode):
            raise TypeError("mode must be GamePilotMode")
        values = (
            self.radial_axis,
            self.in_track_axis,
            self.cross_track_axis,
            self.roll_axis,
            self.pitch_axis,
            self.yaw_axis,
            self.throttle_axis,
            self.firing_action,
        )
        if any(not value.strip() for value in values):
            raise ValueError("profile controls must be non-empty")


@dataclass(frozen=True, slots=True)
class GameAerodynamicEffectorBinding:
    control_id: str
    actuator_id: str
    coordinate_id: str
    actuator_frame: FrameId
    unit: str
    minimum: float
    maximum: float
    neutral: float = 0.0

    def __post_init__(self) -> None:
        if not self.control_id.strip() or not self.actuator_id.strip() or not self.coordinate_id.strip():
            raise ValueError("effector control and device identities must be non-empty")
        if self.unit not in ("rad", "m", "fraction"):
            raise ValueError("effector unit must be rad, m, or fraction")
        values = (self.minimum, self.maximum, self.neutral)
        if not all(isfinite(float(value)) for value in values) or not self.minimum <= self.neutral <= self.maximum:
            raise ValueError("effector limits must be finite and contain neutral")

    def position_for_axis(self, value: float) -> float:
        axis = float(value)
        if axis < -1.0:
            axis = -1.0
        elif axis > 1.0:
            axis = 1.0
        return self.neutral + (self.maximum - self.neutral) * axis if axis >= 0.0 else self.neutral + (
            self.neutral - self.minimum
        ) * axis


@dataclass(frozen=True, slots=True)
class GamePilotReferenceStackConfig:
    satellite_id: str
    body_frame: FrameId
    inertial_frame: FrameId
    relative_frame: FrameId
    profile: GamePilotInputProfile
    translation_allocator: TranslationAllocatorConfig
    assumed_mass_kg: float
    max_acceleration_m_s2: float
    attitude_allocator: AttitudeAllocatorConfig | None = None
    attitude_controller: QuaternionTorqueController = QuaternionTorqueController()
    maximum_attitude_rate_rad_s: float = 0.25
    effectors: tuple[GameAerodynamicEffectorBinding, ...] = ()
    navigation_initialization: NavigationInitializationMode = NavigationInitializationMode.IDEAL
    validity_ticks: int = 1
    emit_diagnostics: bool = True
    translation_reference_origin_state_eci_m_m_s: tuple[float, ...] | None = None
    operator_impulse_duration_s: float = 1.0e-3

    def __post_init__(self) -> None:
        if not self.satellite_id.strip():
            raise ValueError("satellite_id must be non-empty")
        if self.translation_allocator.satellite_id != self.satellite_id:
            raise ValueError("translation allocator satellite_id must match stack satellite_id")
        if self.attitude_allocator is not None and self.attitude_allocator.satellite_id != self.satellite_id:
            raise ValueError("attitude allocator satellite_id must match stack satellite_id")
        for name, value, positive in (
            ("assumed_mass_kg", self.assumed_mass_kg, True),
            ("max_acceleration_m_s2", self.max_acceleration_m_s2, False),
            ("maximum_attitude_rate_rad_s", self.maximum_attitude_rate_rad_s, False),
        ):
            if not isfinite(float(value)) or (value <= 0.0 if positive else value < 0.0):
                raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
        if isinstance(self.validity_ticks, bool) or not isinstance(self.validity_ticks, int) or self.validity_ticks < 1:
            raise ValueError("validity_ticks must be a positive integer")
        if self.translation_reference_origin_state_eci_m_m_s is not None:
            origin = np.asarray(self.translation_reference_origin_state_eci_m_m_s, dtype=float)
            if origin.shape != (6,) or not np.all(np.isfinite(origin)):
                raise ValueError("translation reference origin state must contain six finite SI values")
        if not isfinite(float(self.operator_impulse_duration_s)) or self.operator_impulse_duration_s <= 0.0:
            raise ValueError("operator_impulse_duration_s must be finite and positive")
        identities = [(binding.actuator_id, binding.coordinate_id) for binding in self.effectors]
        if len(identities) != len(set(identities)):
            raise ValueError("aerodynamic effector identities must be unique")
        if self.profile.mode is GamePilotMode.AERODYNAMIC and not self.effectors:
            raise ValueError("aerodynamic pilot profiles require effector bindings")


class GamePilotReferenceFlightSoftwareStack(ReferenceStackBase):
    """Responsive complete stack driven only by typed pilot/ground inputs."""

    stack_id = "fsw.game_pilot_reference"

    def __init__(
        self,
        config: GamePilotReferenceStackConfig,
        *,
        _live_navigation_fast_path: bool = True,
        _live_command_fast_path: bool = True,
    ) -> None:
        super().__init__(satellite_id=config.satellite_id, identity_material=config)
        self.config = config
        self._live_navigation_fast_path = bool(_live_navigation_fast_path)
        self._live_command_fast_path = bool(_live_command_fast_path)
        self._navigator = OrbitNavigator(
            initialization=config.navigation_initialization,
            body_frame=config.body_frame,
            inertial_frame=config.inertial_frame,
            relative_frame=config.relative_frame,
            retain_full_provenance=not self._live_navigation_fast_path,
        )
        self._translation_allocator = TranslationAllocator(config.translation_allocator)
        self._attitude_allocator = (
            None if config.attitude_allocator is None else AttitudeAllocator(config.attitude_allocator)
        )
        self._axes: dict[str, float] = {}
        self._held_actions: set[str] = set()
        self._seen_inputs: set[PacketId] = set()
        self._desired_attitude: tuple[float, float, float, float] | None = None
        self._last_step_time: ClockTag | None = None
        self._pending_delta_v_ric_m_s: tuple[float, float, float] | None = None
        self._pending_impulse_duration_s: float | None = None
        self._last_ground_command_id: str | None = None
        self._reference_state_eci_m_m_s: tuple[float, ...] | None = None
        self._cached_allocator_attitude_bytes: bytes | None = None
        self._cached_allocator_dcm_bn: np.ndarray | None = None

    def _step(self, batch: FlightSoftwareInputBatch) -> FlightSoftwareOutput:
        self._navigator.ingest(batch.events)
        self._ingest_inputs(batch)
        solution = (
            self._navigator.control_solution(batch.invocation_time)
            if self._live_navigation_fast_path
            else self._navigator.solution(batch.invocation_time)
        )
        dt_s = _elapsed_seconds(self._last_step_time, batch.invocation_time)
        self._last_step_time = batch.invocation_time
        commands: list[ActuatorCommand] = []
        mode = self.config.profile.mode
        if mode is GamePilotMode.AERODYNAMIC:
            commands.extend(self._aerodynamic_commands(batch.invocation_time))
        else:
            commands.extend(self._attitude_commands(solution, dt_s))
            request = self._requested_force(solution)
            if request is not None:
                force, validity_ticks = request
                effort = RequestedEffort(
                    f"game.{mode.value}",
                    RequestedEffortKind.FORCE,
                    batch.invocation_time,
                    self.config.inertial_frame,
                    ValidityInterval(batch.invocation_time, _add_ticks(batch.invocation_time, validity_ticks)),
                    force_n=tuple(float(value) for value in force),
                )
                commands.extend(self._translation_commands(effort, solution))
        faulted = {component for component, _code in solution.active_faults}
        if faulted:
            commands = [command for command in commands if command.actuator_id not in faulted]
        telemetry = ()
        if self.config.emit_diagnostics:
            fields = (
                TelemetryField("input_profile_id", self.config.profile.profile_id),
                TelemetryField("pilot_mode", mode.value),
                TelemetryField("navigation_valid", solution.own_state_valid),
                TelemetryField("command_count", len(commands)),
                TelemetryField("held_action_count", len(self._held_actions)),
                TelemetryField("last_ground_command_id", self._last_ground_command_id),
                *(TelemetryField(f"axis.{name}", value) for name, value in sorted(self._axes.items())),
            )
            telemetry = (DiagnosticTelemetry(f"{self.stack_id}.status", batch.invocation_time, fields),)
        return FlightSoftwareOutput(batch.satellite_id, batch.invocation_id, tuple(commands), telemetry)

    def _ingest_inputs(self, batch: FlightSoftwareInputBatch) -> None:
        for event in batch.events:
            if event.packet_id in self._seen_inputs:
                continue
            if event.kind is InputKind.MEASUREMENT and isinstance(event.payload, MeasurementEvent):
                tracked = event.payload.payload
                if isinstance(tracked, IdealTrackedObjectStateMeasurement):
                    self._reference_state_eci_m_m_s = (*tracked.position_m, *tracked.velocity_m_s)
                continue
            if event.kind not in (InputKind.PILOT_INPUT, InputKind.GROUND_COMMAND):
                continue
            self._seen_inputs.add(event.packet_id)
            if event.kind is InputKind.PILOT_INPUT and isinstance(event.payload, PilotInputPayload):
                if event.payload.input_profile_id != self.config.profile.profile_id:
                    continue
                self._axes.update((axis.control_id, axis.value) for axis in event.payload.axes)
                self._held_actions.update(event.payload.pressed_actions)
                self._held_actions.difference_update(event.payload.released_actions)
            elif isinstance(event.payload, GroundCommandPayload):
                self._ingest_ground_command(event.payload, batch.invocation_time)

    def _ingest_ground_command(self, command: GroundCommandPayload, now: ClockTag) -> None:
        if command.execute_at is not None and _signed_elapsed_seconds(now, command.execute_at) > 0.0:
            return
        self._last_ground_command_id = command.command_id
        if command.kind is not GroundCommandKind.ACTION_REQUEST:
            return
        values = {field.name: field.value for field in command.parameters}
        if all(name in values for name in ("delta_v_r_m_s", "delta_v_i_m_s", "delta_v_c_m_s")):
            self._pending_delta_v_ric_m_s = tuple(
                float(values[name]) for name in ("delta_v_r_m_s", "delta_v_i_m_s", "delta_v_c_m_s")
            )
            duration_s = float(values.get("impulse_duration_s", self.config.operator_impulse_duration_s))
            self._pending_impulse_duration_s = (
                duration_s
                if isfinite(duration_s) and duration_s > 0.0
                else self.config.operator_impulse_duration_s
            )

    def _requested_force(self, solution: OrbitNavigationSolution) -> tuple[np.ndarray, int] | None:
        if not solution.own_state_valid:
            return None
        mass = solution.mass_kg if solution.mass_kg is not None else self.config.assumed_mass_kg
        if self._pending_delta_v_ric_m_s is not None:
            duration_s = self._pending_impulse_duration_s or self.config.operator_impulse_duration_s
            acceleration_ric = np.asarray(self._pending_delta_v_ric_m_s) / duration_s
            self._pending_delta_v_ric_m_s = None
            self._pending_impulse_duration_s = None
            validity_ticks = max(
                1,
                int(round(duration_s / (solution.generated_at.tick_period_ns * 1.0e-9))),
            )
        elif self.config.profile.mode is GamePilotMode.ATTITUDE_THRUST:
            if self.config.profile.firing_action not in self._held_actions:
                return None
            throttle = _throttle(self._axes.get(self.config.profile.throttle_axis))
            direction_body = np.array([1.0, 0.0, 0.0])
            quaternion = solution.attitude.attitude_quat_bn
            if quaternion is None:
                return None
            direction_eci = quaternion_to_dcm_bn(np.asarray(quaternion)).T @ direction_body
            return (
                direction_eci * self.config.max_acceleration_m_s2 * mass * throttle,
                self.config.validity_ticks,
            )
        elif self.config.profile.mode is GamePilotMode.DIRECT_ECI:
            profile = self.config.profile
            acceleration_eci = np.array(
                [
                    self._axes.get(profile.radial_axis, 0.0),
                    self._axes.get(profile.in_track_axis, 0.0),
                    self._axes.get(profile.cross_track_axis, 0.0),
                ],
                dtype=float,
            )
            norm = float(np.linalg.norm(acceleration_eci))
            if norm > 1.0:
                acceleration_eci /= norm
            acceleration_eci *= self.config.max_acceleration_m_s2 * _throttle(
                self._axes.get(profile.throttle_axis)
            )
            return acceleration_eci * mass, self.config.validity_ticks
        else:
            profile = self.config.profile
            acceleration_ric = np.array(
                [
                    self._axes.get(profile.radial_axis, 0.0),
                    self._axes.get(profile.in_track_axis, 0.0),
                    self._axes.get(profile.cross_track_axis, 0.0),
                ],
                dtype=float,
            )
            norm = float(np.linalg.norm(acceleration_ric))
            if norm > 1.0:
                acceleration_ric /= norm
            acceleration_ric *= self.config.max_acceleration_m_s2 * _throttle(
                self._axes.get(profile.throttle_axis)
            )
            validity_ticks = self.config.validity_ticks
        return (
            _ric_to_eci(
                acceleration_ric,
                solution,
                self._reference_state_eci_m_m_s,
                self.config.translation_reference_origin_state_eci_m_m_s,
            )
            * mass,
            validity_ticks,
        )

    def _attitude_commands(self, solution: OrbitNavigationSolution, dt_s: float) -> tuple[ActuatorCommand, ...]:
        if self.config.profile.mode is not GamePilotMode.ATTITUDE_THRUST or self._attitude_allocator is None:
            return ()
        attitude = solution.attitude
        if attitude.attitude_quat_bn is None:
            return ()
        if self._desired_attitude is None:
            self._desired_attitude = tuple(attitude.attitude_quat_bn)
        profile = self.config.profile
        body_rate = self.config.maximum_attitude_rate_rad_s * np.array(
            [
                self._axes.get(profile.roll_axis, 0.0),
                self._axes.get(profile.pitch_axis, 0.0),
                self._axes.get(profile.yaw_axis, 0.0),
            ]
        )
        if dt_s > 0.0 and np.linalg.norm(body_rate) > 0.0:
            self._desired_attitude = tuple(
                float(value)
                for value in normalize_quaternion(
                    quaternion_multiply(
                        np.asarray(self._desired_attitude),
                        quaternion_delta_from_body_rate(body_rate, dt_s),
                    )
                )
            )
        if self._live_command_fast_path:
            desired = np.asarray(self._desired_attitude, dtype=float)
            if (
                desired.size != 4
                or not np.all(np.isfinite(desired))
                or abs(np.linalg.norm(desired) - 1) > 1e-10
            ):
                raise ValueError("quaternion_bn must be normalized")
            quaternion = normalize_quaternion(desired)
            reference = GuidanceReference(
                "attitude.quaternion",
                "attitude",
                self.config.inertial_frame,
                ValidityInterval(
                    attitude.generated_at,
                    _add_ticks(attitude.generated_at, self.config.validity_ticks),
                ),
                attitude_quat_from_frame=tuple(float(value) for value in quaternion),
            )
        else:
            reference = AttitudeReferenceGenerator(
                AttitudeReferenceConfig(
                    AttitudeReferenceMode.QUATERNION,
                    quaternion_bn=self._desired_attitude,
                    validity_ticks=self.config.validity_ticks,
                ),
                inertial_frame=self.config.inertial_frame,
            ).generate(attitude)
        if reference is None:
            return ()
        effort = self.config.attitude_controller.control(attitude, reference)
        if effort is None:
            return ()
        return self._attitude_allocator.allocate(
            effort,
            attitude,
            command_id=self._next_command_id(),
        ).proposed_commands

    def _translation_commands(
        self,
        effort: RequestedEffort,
        solution: OrbitNavigationSolution,
    ) -> tuple[ActuatorCommand, ...]:
        allocator_config = self.config.translation_allocator
        if not self._live_command_fast_path or allocator_config.kind is not TranslationAllocatorKind.IDEAL_WRENCH:
            return self._translation_allocator.allocate(
                effort,
                solution,
                next_command_id=self._next_command_id,
            ).proposed_commands
        if effort.force_n is None or solution.attitude.attitude_quat_bn is None:
            return ()
        requested_eci = np.asarray(effort.force_n, dtype=float)
        requested_norm = float(np.linalg.norm(requested_eci))
        scale = min(1.0, allocator_config.max_force_n / requested_norm) if requested_norm > 0.0 else 1.0
        achieved_eci = requested_eci * scale
        attitude = np.asarray(solution.attitude.attitude_quat_bn, dtype=float)
        attitude_bytes = attitude.tobytes()
        if attitude_bytes != self._cached_allocator_attitude_bytes:
            self._cached_allocator_attitude_bytes = attitude_bytes
            self._cached_allocator_dcm_bn = quaternion_to_dcm_bn(attitude)
        assert self._cached_allocator_dcm_bn is not None
        force_body = self._cached_allocator_dcm_bn @ achieved_eci
        return (
            ActuatorCommand(
                self._next_command_id(),
                allocator_config.satellite_id,
                allocator_config.actuator_id,
                effort.generated_at,
                effort.validity,
                allocator_config.actuator_frame,
                IdealWrenchCommand(tuple(float(value) for value in force_body), (0.0, 0.0, 0.0)),
            ),
        )

    def _aerodynamic_commands(self, now: ClockTag) -> tuple[ActuatorCommand, ...]:
        commands: list[ActuatorCommand] = []
        validity = ValidityInterval(now, _add_ticks(now, self.config.validity_ticks))
        for binding in self.config.effectors:
            position = binding.position_for_axis(self._axes.get(binding.control_id, 0.0))
            commands.append(
                ActuatorCommand(
                    self._next_command_id(),
                    self.config.satellite_id,
                    binding.actuator_id,
                    now,
                    validity,
                    binding.actuator_frame,
                    AerodynamicEffectorPositionCommand(binding.coordinate_id, position, binding.unit),
                )
            )
        return tuple(commands)

    def _snapshot_stack_state(self) -> dict[str, object]:
        return {
            "navigation": self._navigator.snapshot_state(),
            "axes": dict(sorted(self._axes.items())),
            "held_actions": sorted(self._held_actions),
            "seen_inputs": [to_primitive(packet) for packet in sorted(self._seen_inputs, key=_packet_key)],
            "desired_attitude": self._desired_attitude,
            "last_step_time": None if self._last_step_time is None else to_primitive(self._last_step_time),
            "pending_delta_v_ric_m_s": self._pending_delta_v_ric_m_s,
            "pending_impulse_duration_s": self._pending_impulse_duration_s,
            "last_ground_command_id": self._last_ground_command_id,
            "reference_state_eci_m_m_s": self._reference_state_eci_m_m_s,
        }

    def _prepare_restored_stack_state(self, state: dict[str, object]) -> object:
        navigation = state.get("navigation")
        if not isinstance(navigation, dict):
            raise ValueError("game stack snapshot navigation state is invalid")
        navigator = OrbitNavigator(
            initialization=self.config.navigation_initialization,
            body_frame=self.config.body_frame,
            inertial_frame=self.config.inertial_frame,
            relative_frame=self.config.relative_frame,
            retain_full_provenance=not self._live_navigation_fast_path,
        )
        navigator.restore_state(navigation)
        axes = {str(key): float(value) for key, value in dict(state.get("axes", {})).items()}
        held_actions = {str(value) for value in list(state.get("held_actions", []))}
        seen_inputs = {
            from_primitive(PacketId, value) for value in list(state.get("seen_inputs", []))
        }
        desired = _optional_vector(state.get("desired_attitude"), 4)
        last_step_primitive = state.get("last_step_time")
        last_step = None if last_step_primitive is None else from_primitive(ClockTag, last_step_primitive)
        pending = _optional_vector(state.get("pending_delta_v_ric_m_s"), 3)
        pending_duration_raw = state.get("pending_impulse_duration_s")
        pending_duration = None if pending_duration_raw is None else float(pending_duration_raw)
        if pending_duration is not None and (not isfinite(pending_duration) or pending_duration <= 0.0):
            raise ValueError("game stack snapshot impulse duration is invalid")
        last_ground = state.get("last_ground_command_id")
        if last_ground is not None and not isinstance(last_ground, str):
            raise ValueError("game stack snapshot ground-command state is invalid")
        reference_state = _optional_vector(state.get("reference_state_eci_m_m_s"), 6)
        return (
            navigator,
            axes,
            held_actions,
            seen_inputs,
            desired,
            last_step,
            pending,
            pending_duration,
            last_ground,
            reference_state,
        )

    def _commit_restored_stack_state(self, state: object) -> None:
        if not isinstance(state, tuple) or len(state) != 10:
            raise TypeError("restored game stack state is invalid")
        (
            self._navigator,
            self._axes,
            self._held_actions,
            self._seen_inputs,
            self._desired_attitude,
            self._last_step_time,
            self._pending_delta_v_ric_m_s,
            self._pending_impulse_duration_s,
            self._last_ground_command_id,
            self._reference_state_eci_m_m_s,
        ) = state
        self._cached_allocator_attitude_bytes = None
        self._cached_allocator_dcm_bn = None


def _ric_to_eci(
    vector_ric: np.ndarray,
    solution: OrbitNavigationSolution,
    reference_state: tuple[float, ...] | None = None,
    origin_state: tuple[float, ...] | None = None,
) -> np.ndarray:
    state = np.asarray(
        (*solution.position_eci_m, *solution.velocity_eci_m_s)
        if reference_state is None
        else reference_state,
        dtype=float,
    )
    if origin_state is not None:
        state = state - np.asarray(origin_state, dtype=float)
    position = state[:3]
    velocity = state[3:6]
    radial = _unit(position)
    cross_track = _unit(_cross3(position, velocity))
    in_track = _unit(_cross3(cross_track, radial))
    return np.column_stack((radial, in_track, cross_track)) @ np.asarray(vector_ric)


def _cross3(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    a = np.asarray(first, dtype=float).reshape(3)
    b = np.asarray(second, dtype=float).reshape(3)
    return np.array(
        (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ),
        dtype=float,
    )


def _unit(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(value))
    if not isfinite(norm) or norm <= 0.0:
        raise ValueError("vector must be finite and nonzero")
    return value / norm


def _throttle(value: float | None) -> float:
    if value is None:
        return 1.0
    throttle = 0.5 * (float(value) + 1.0)
    if throttle < 0.0:
        return 0.0
    if throttle > 1.0:
        return 1.0
    return throttle


def _elapsed_seconds(start: ClockTag | None, end: ClockTag) -> float:
    if start is None:
        return 0.0
    if (start.clock_id, start.tick_period_ns, start.scale, start.reset_counter) != (
        end.clock_id,
        end.tick_period_ns,
        end.scale,
        end.reset_counter,
    ):
        raise ValueError("game stack clocks must share a domain")
    delta = (end.ticks - start.ticks) * start.tick_period_ns * 1.0e-9
    if delta < 0.0:
        raise ValueError("game stack invocation time must not move backward")
    return delta


def _signed_elapsed_seconds(start: ClockTag, end: ClockTag) -> float:
    if (start.clock_id, start.tick_period_ns, start.scale, start.reset_counter) != (
        end.clock_id,
        end.tick_period_ns,
        end.scale,
        end.reset_counter,
    ):
        raise ValueError("game stack clocks must share a domain")
    return (end.ticks - start.ticks) * start.tick_period_ns * 1.0e-9


def _add_ticks(tag: ClockTag, ticks: int) -> ClockTag:
    return ClockTag(tag.clock_id, tag.ticks + ticks, tag.tick_period_ns, tag.scale, tag.validity, tag.reset_counter)


def _packet_key(packet: PacketId) -> tuple[str, str, int]:
    return packet.source_id, packet.boot_id, packet.sequence


def _optional_vector(value: object, size: int) -> tuple[float, ...] | None:
    if value is None:
        return None
    result = tuple(float(item) for item in list(value))  # type: ignore[arg-type]
    if len(result) != size or not all(isfinite(item) for item in result):
        raise ValueError(f"snapshot vector must contain {size} finite values")
    return result
