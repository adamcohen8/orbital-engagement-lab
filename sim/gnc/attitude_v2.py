"""Typed attitude navigation, guidance, control, and allocation for v2 stacks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Protocol

import numpy as np

from sim.flight_software.contracts import (
    ActuatorCommand,
    ClockScale,
    ClockTag,
    CmgGimbalRateCommand,
    DataValidity,
    FrameId,
    GyroMeasurement,
    IdealOwnStateMeasurement,
    IdealWrenchCommand,
    InputEvent,
    InputKind,
    MagnetometerMeasurement,
    MagnetorquerDipoleCommand,
    MeasurementEvent,
    ModeledFaultIndicationPayload,
    PacketId,
    ReactionWheelTorqueCommand,
    StarTrackerMeasurement,
    SunVectorMeasurement,
    TelemetryField,
    TimeValidity,
    ValidityInterval,
)
from sim.gnc.contracts import (
    AllocationResult,
    AllocationStatus,
    BeliefState,
    EstimateValidity,
    GuidanceReference,
    RequestedEffort,
    RequestedEffortKind,
    StateEstimate,
)
from sim.utils.quaternion import (
    normalize_quaternion,
    quaternion_delta_from_body_rate,
    quaternion_multiply,
    quaternion_to_dcm_bn,
)


@dataclass(frozen=True, slots=True)
class SensorMounting:
    sensor_id: str
    quat_body_from_sensor: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    sensor_frame: FrameId | None = None

    def __post_init__(self) -> None:
        if not self.sensor_id.strip():
            raise ValueError("sensor_id must be non-empty")
        quaternion = np.asarray(self.quat_body_from_sensor, dtype=float)
        if quaternion.size != 4 or not np.all(np.isfinite(quaternion)) or abs(np.linalg.norm(quaternion) - 1.0) > 1e-10:
            raise ValueError("quat_body_from_sensor must be a normalized quaternion")


@dataclass(frozen=True, slots=True)
class SensorCalibration:
    sensor_id: str
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    bias: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if not self.sensor_id.strip():
            raise ValueError("sensor_id must be non-empty")
        scale = np.asarray(self.scale, dtype=float)
        bias = np.asarray(self.bias, dtype=float)
        if scale.size != 3 or not np.all(np.isfinite(scale)) or np.any(scale == 0.0):
            raise ValueError("calibration scale must contain three finite, nonzero values")
        if bias.size != 3 or not np.all(np.isfinite(bias)):
            raise ValueError("calibration bias must contain three finite values")


@dataclass(frozen=True, slots=True)
class AttitudeSolution:
    generated_at: ClockTag
    frame: FrameId
    attitude_quat_bn: tuple[float, float, float, float] | None
    angular_rate_body_rad_s: tuple[float, float, float] | None
    position_eci_m: tuple[float, float, float] | None
    velocity_eci_m_s: tuple[float, float, float] | None
    sun_vector_body: tuple[float, float, float] | None
    magnetic_field_body_t: tuple[float, float, float] | None
    active_faults: tuple[tuple[str, str], ...]
    belief: BeliefState

    @property
    def valid_for_control(self) -> bool:
        return self.attitude_quat_bn is not None and self.angular_rate_body_rad_s is not None


class AttitudeNavigator:
    def __init__(
        self,
        *,
        body_frame: FrameId,
        inertial_frame: FrameId,
        mountings: tuple[SensorMounting, ...] = (),
        calibrations: tuple[SensorCalibration, ...] = (),
        retain_full_provenance: bool = True,
    ) -> None:
        self.body_frame = body_frame
        self.inertial_frame = inertial_frame
        self._mountings = {mounting.sensor_id: mounting for mounting in mountings}
        self._calibrations = {calibration.sensor_id: calibration for calibration in calibrations}
        self._retain_full_provenance = bool(retain_full_provenance)
        self._attitude: tuple[float, float, float, float] | None = None
        self._attitude_epoch: ClockTag | None = None
        self._rate: tuple[float, float, float] | None = None
        self._position: tuple[float, float, float] | None = None
        self._velocity: tuple[float, float, float] | None = None
        self._sun_body: tuple[float, float, float] | None = None
        self._magnetic_body: tuple[float, float, float] | None = None
        self._faults: dict[str, str] = {}
        self._seen: set[PacketId] = set()
        self._provenance: list[PacketId] = []
        self._degraded = False

    def ingest(self, events: tuple[InputEvent, ...]) -> None:
        for event in events:
            if event.packet_id in self._seen:
                continue
            self._seen.add(event.packet_id)
            if event.kind is InputKind.MODELED_FAULT_INDICATION:
                self._ingest_fault(event.payload)
                continue
            if event.kind is not InputKind.MEASUREMENT or not isinstance(event.payload, MeasurementEvent):
                continue
            measurement = event.payload
            if (
                measurement.sensor_id in self._faults
                or event.quality.validity is DataValidity.INVALID
                or event.source_time.validity is TimeValidity.INVALID
                or event.delivery_time.validity is TimeValidity.INVALID
                or measurement.sample_time.validity is TimeValidity.INVALID
            ):
                continue
            if event.quality.validity is DataValidity.SUSPECT:
                self._degraded = True
            if self._ingest_measurement(measurement):
                if self._retain_full_provenance:
                    self._provenance.append(event.packet_id)
                else:
                    self._provenance[:] = (event.packet_id,)

    def solution(self, generated_at: ClockTag) -> AttitudeSolution:
        self._propagate_attitude(generated_at)
        attitude_validity = (
            EstimateValidity.INVALID
            if self._attitude is None and self._rate is None
            else EstimateValidity.DEGRADED
            if self._degraded or self._faults
            else EstimateValidity.VALID
        )
        values: list[TelemetryField] = []
        if self._attitude is not None:
            values.extend(TelemetryField(f"q_{name}", value) for name, value in zip("wxyz", self._attitude))
        if self._rate is not None:
            values.extend(
                TelemetryField(f"omega_{axis}_rad_s", value, "rad/s") for axis, value in zip("xyz", self._rate)
            )
        estimate = StateEstimate(
            "attitude",
            generated_at,
            self.body_frame,
            tuple(values),
            source_packets=tuple(self._provenance),
            validity=attitude_validity,
        )
        own_values: list[TelemetryField] = []
        if self._position is not None:
            own_values.extend(
                TelemetryField(f"position_{axis}_m", value, "m") for axis, value in zip("xyz", self._position)
            )
        if self._velocity is not None:
            own_values.extend(
                TelemetryField(f"velocity_{axis}_m_s", value, "m/s") for axis, value in zip("xyz", self._velocity)
            )
        own_estimate = (
            StateEstimate(
                "own-state",
                generated_at,
                self.inertial_frame,
                tuple(own_values),
                source_packets=tuple(self._provenance),
                validity=EstimateValidity.DEGRADED if self._degraded or self._faults else EstimateValidity.VALID,
            )
            if own_values
            else None
        )
        belief = BeliefState(
            generated_at,
            own_state=own_estimate,
            attitude_state=estimate,
            environment_estimates=tuple(
                [
                    *(TelemetryField(f"sun_{axis}_body", value) for axis, value in zip("xyz", self._sun_body or ())),
                    *(
                        TelemetryField(f"magnetic_{axis}_body_t", value, "T")
                        for axis, value in zip("xyz", self._magnetic_body or ())
                    ),
                ]
            ),
            health_state=tuple(
                TelemetryField(f"fault.{component}", code) for component, code in sorted(self._faults.items())
            ),
            provenance=tuple(self._provenance),
        )
        return AttitudeSolution(
            generated_at,
            self.body_frame,
            self._attitude,
            self._rate,
            self._position,
            self._velocity,
            self._sun_body,
            self._magnetic_body,
            tuple(sorted(self._faults.items())),
            belief,
        )

    def control_solution(self, generated_at: ClockTag) -> AttitudeSolution:
        """Return control state without constructing unused audit estimates.

        Real-time consumers that do not publish the embedded belief can use
        this path.  Navigation propagation, state, and fault behavior remain
        identical to :meth:`solution`; only the discarded evidence objects are
        omitted.
        """

        self._propagate_attitude(generated_at)
        return AttitudeSolution(
            generated_at,
            self.body_frame,
            self._attitude,
            self._rate,
            self._position,
            self._velocity,
            self._sun_body,
            self._magnetic_body,
            tuple(sorted(self._faults.items())),
            BeliefState(generated_at),
        )

    def snapshot_state(self) -> dict[str, object]:
        return {
            "attitude": self._attitude,
            "attitude_epoch": _clock_to_dict(self._attitude_epoch),
            "rate": self._rate,
            "position": self._position,
            "velocity": self._velocity,
            "sun_body": self._sun_body,
            "magnetic_body": self._magnetic_body,
            "faults": dict(sorted(self._faults.items())),
            "seen": [
                {"source_id": packet.source_id, "boot_id": packet.boot_id, "sequence": packet.sequence}
                for packet in sorted(self._seen, key=lambda item: (item.source_id, item.boot_id, item.sequence))
            ],
            "provenance": [
                {"source_id": packet.source_id, "boot_id": packet.boot_id, "sequence": packet.sequence}
                for packet in self._provenance
            ],
            "degraded": self._degraded,
        }

    def restore_state(self, state: dict[str, object]) -> None:
        self._attitude = _optional_tuple(state.get("attitude"), 4)
        self._attitude_epoch = _clock_from_dict(state.get("attitude_epoch"))
        self._rate = _optional_tuple(state.get("rate"), 3)
        self._position = _optional_tuple(state.get("position"), 3)
        self._velocity = _optional_tuple(state.get("velocity"), 3)
        self._sun_body = _optional_tuple(state.get("sun_body"), 3)
        self._magnetic_body = _optional_tuple(state.get("magnetic_body"), 3)
        self._faults = {str(key): str(value) for key, value in dict(state.get("faults", {})).items()}
        self._seen = {_packet_from_dict(item) for item in list(state.get("seen", []))}
        self._provenance = [_packet_from_dict(item) for item in list(state.get("provenance", []))]
        self._degraded = bool(state.get("degraded", False))

    def _ingest_fault(self, payload: object) -> None:
        if not isinstance(payload, ModeledFaultIndicationPayload):
            return
        if payload.active:
            self._faults[payload.component_id] = payload.fault_code
        else:
            self._faults.pop(payload.component_id, None)

    def _ingest_measurement(self, measurement: MeasurementEvent) -> bool:
        payload = measurement.payload
        mounting = self._mountings.get(measurement.sensor_id, SensorMounting(measurement.sensor_id))
        if mounting.sensor_frame is not None and measurement.frame != mounting.sensor_frame:
            self._degraded = True
            return False
        calibration = self._calibrations.get(measurement.sensor_id, SensorCalibration(measurement.sensor_id))
        q_bs = np.asarray(mounting.quat_body_from_sensor, dtype=float)
        c_bs = quaternion_to_dcm_bn(q_bs)
        if isinstance(payload, IdealOwnStateMeasurement):
            if payload.attitude_quat_body_from_inertial is not None:
                self._attitude = tuple(payload.attitude_quat_body_from_inertial)
                self._attitude_epoch = measurement.sample_time
            if payload.angular_rate_body_rad_s is not None:
                self._rate = tuple(payload.angular_rate_body_rad_s)
            if payload.position_m is not None:
                self._position = tuple(payload.position_m)
            if payload.velocity_m_s is not None:
                self._velocity = tuple(payload.velocity_m_s)
            return True
        if isinstance(payload, GyroMeasurement):
            self._propagate_attitude(measurement.sample_time)
            self._rate = tuple(
                float(value) for value in c_bs @ _calibrated_vector(payload.angular_rate_rad_s, calibration)
            )
            return True
        if isinstance(payload, StarTrackerMeasurement):
            self._attitude = tuple(
                float(value)
                for value in normalize_quaternion(
                    quaternion_multiply(q_bs, np.asarray(payload.quat_sensor_from_inertial))
                )
            )
            self._attitude_epoch = measurement.sample_time
            return True
        if isinstance(payload, SunVectorMeasurement):
            self._sun_body = tuple(
                float(value) for value in _unit(c_bs @ _calibrated_vector(payload.unit_vector, calibration))
            )
            return True
        if isinstance(payload, MagnetometerMeasurement):
            self._magnetic_body = tuple(
                float(value) for value in c_bs @ _calibrated_vector(payload.magnetic_flux_density_t, calibration)
            )
            return True
        return False

    def _propagate_attitude(self, target: ClockTag) -> None:
        if self._attitude is None or self._rate is None or self._attitude_epoch is None:
            return
        elapsed_s = _elapsed_seconds(self._attitude_epoch, target)
        if elapsed_s is None or elapsed_s < 0.0:
            self._degraded = True
            return
        if elapsed_s == 0.0:
            return
        increment = quaternion_delta_from_body_rate(np.asarray(self._rate), elapsed_s)
        self._attitude = tuple(
            float(value) for value in normalize_quaternion(quaternion_multiply(np.asarray(self._attitude), increment))
        )
        self._attitude_epoch = target


class AttitudeReferenceMode(str, Enum):
    QUATERNION = "quaternion"
    NADIR = "nadir"
    VELOCITY = "velocity"
    SUN = "sun"
    TARGET = "target"
    RIC = "ric"
    THRUST = "thrust"
    REPLAY = "replay"


@dataclass(frozen=True, slots=True)
class AttitudeReferenceConfig:
    mode: AttitudeReferenceMode = AttitudeReferenceMode.QUATERNION
    quaternion_bn: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    target_position_eci_m: tuple[float, float, float] | None = None
    thrust_direction_eci: tuple[float, float, float] | None = None
    ric_axis: str = "radial_out"
    boresight_body: tuple[float, float, float] = (1.0, 0.0, 0.0)
    replay_times_ns: tuple[int, ...] = ()
    replay_quaternions_bn: tuple[tuple[float, float, float, float], ...] = ()
    validity_ticks: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.mode, AttitudeReferenceMode):
            raise TypeError("mode must be AttitudeReferenceMode")
        quaternion = np.asarray(self.quaternion_bn, dtype=float)
        if quaternion.size != 4 or not np.all(np.isfinite(quaternion)) or abs(np.linalg.norm(quaternion) - 1) > 1e-10:
            raise ValueError("quaternion_bn must be normalized")
        for name, value in (
            ("target_position_eci_m", self.target_position_eci_m),
            ("thrust_direction_eci", self.thrust_direction_eci),
        ):
            if value is not None:
                vector = np.asarray(value, dtype=float)
                if vector.size != 3 or not np.all(np.isfinite(vector)):
                    raise ValueError(f"{name} must contain three finite values")
        if self.mode is AttitudeReferenceMode.TARGET and self.target_position_eci_m is None:
            raise ValueError("target mode requires target_position_eci_m")
        if self.mode is AttitudeReferenceMode.THRUST:
            if self.thrust_direction_eci is None or np.linalg.norm(self.thrust_direction_eci) <= 0.0:
                raise ValueError("thrust mode requires a nonzero thrust_direction_eci")
        boresight = np.asarray(self.boresight_body, dtype=float)
        if boresight.size != 3 or not np.all(np.isfinite(boresight)) or np.linalg.norm(boresight) <= 0.0:
            raise ValueError("boresight_body must contain three finite values and be nonzero")
        if self.mode is AttitudeReferenceMode.REPLAY:
            if not self.replay_times_ns or len(self.replay_times_ns) != len(self.replay_quaternions_bn):
                raise ValueError("replay mode requires matching, non-empty replay times and quaternions")
            if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in self.replay_times_ns):
                raise ValueError("replay_times_ns must contain nonnegative integers")
            if any(b <= a for a, b in zip(self.replay_times_ns, self.replay_times_ns[1:])):
                raise ValueError("replay_times_ns must be strictly increasing")
            for value in self.replay_quaternions_bn:
                replay_q = np.asarray(value, dtype=float)
                if replay_q.size != 4 or not np.all(np.isfinite(replay_q)) or abs(np.linalg.norm(replay_q) - 1) > 1e-10:
                    raise ValueError("replay_quaternions_bn must contain normalized quaternions")
        allowed_ric_axes = {
            "radial_out",
            "radial_in",
            "in_track",
            "anti_in_track",
            "cross_track",
            "anti_cross_track",
        }
        if self.ric_axis not in allowed_ric_axes:
            raise ValueError("ric_axis is unsupported")
        if isinstance(self.validity_ticks, bool) or not isinstance(self.validity_ticks, int) or self.validity_ticks < 1:
            raise ValueError("validity_ticks must be a positive integer")


class AttitudeReferenceGenerator:
    def __init__(self, config: AttitudeReferenceConfig, *, inertial_frame: FrameId) -> None:
        self.config = config
        self.inertial_frame = inertial_frame

    def generate(self, solution: AttitudeSolution) -> GuidanceReference | None:
        direction: np.ndarray | None = None
        mode = self.config.mode
        if mode is AttitudeReferenceMode.QUATERNION:
            quaternion = normalize_quaternion(np.asarray(self.config.quaternion_bn))
        elif mode is AttitudeReferenceMode.REPLAY:
            quaternion = self._replay_quaternion(solution.generated_at)
        else:
            direction = self._direction(solution)
            if direction is None:
                return None
            quaternion = _quaternion_from_two_vectors(direction, np.asarray(self.config.boresight_body))
        expires = _add_ticks(solution.generated_at, self.config.validity_ticks)
        return GuidanceReference(
            f"attitude.{mode.value}",
            "attitude",
            self.inertial_frame,
            ValidityInterval(solution.generated_at, expires),
            attitude_quat_from_frame=tuple(float(value) for value in quaternion),
        )

    def _replay_quaternion(self, at: ClockTag) -> np.ndarray:
        time_ns = at.ticks * at.tick_period_ns
        times = self.config.replay_times_ns
        quaternions = self.config.replay_quaternions_bn
        if time_ns <= times[0]:
            return np.asarray(quaternions[0], dtype=float)
        if time_ns >= times[-1]:
            return np.asarray(quaternions[-1], dtype=float)
        upper = int(np.searchsorted(np.asarray(times, dtype=np.int64), time_ns, side="right"))
        lower = upper - 1
        fraction = (time_ns - times[lower]) / (times[upper] - times[lower])
        first = np.asarray(quaternions[lower], dtype=float)
        second = np.asarray(quaternions[upper], dtype=float)
        if float(first @ second) < 0.0:
            second = -second
        return normalize_quaternion((1.0 - fraction) * first + fraction * second)

    def _direction(self, solution: AttitudeSolution) -> np.ndarray | None:
        mode = self.config.mode
        if mode is AttitudeReferenceMode.NADIR and solution.position_eci_m is not None:
            return _unit_or_none(-np.asarray(solution.position_eci_m))
        if mode is AttitudeReferenceMode.VELOCITY and solution.velocity_eci_m_s is not None:
            return _unit_or_none(np.asarray(solution.velocity_eci_m_s))
        if mode is AttitudeReferenceMode.SUN and solution.sun_vector_body is not None and solution.attitude_quat_bn:
            return _unit_or_none(
                quaternion_to_dcm_bn(np.asarray(solution.attitude_quat_bn)).T @ solution.sun_vector_body
            )
        if (
            mode is AttitudeReferenceMode.TARGET
            and solution.position_eci_m is not None
            and self.config.target_position_eci_m is not None
        ):
            return _unit_or_none(np.asarray(self.config.target_position_eci_m) - np.asarray(solution.position_eci_m))
        if mode is AttitudeReferenceMode.THRUST and self.config.thrust_direction_eci is not None:
            return _unit_or_none(np.asarray(self.config.thrust_direction_eci))
        if (
            mode is AttitudeReferenceMode.RIC
            and solution.position_eci_m is not None
            and solution.velocity_eci_m_s is not None
        ):
            radial = _unit_or_none(np.asarray(solution.position_eci_m))
            cross = _unit_or_none(np.cross(solution.position_eci_m, solution.velocity_eci_m_s))
            if radial is None or cross is None:
                return None
            intrack = _unit_or_none(np.cross(cross, radial))
            if intrack is None:
                return None
            return {
                "radial_out": radial,
                "radial_in": -radial,
                "in_track": intrack,
                "anti_in_track": -intrack,
                "cross_track": cross,
                "anti_cross_track": -cross,
            }.get(self.config.ric_axis)
        return None


@dataclass(frozen=True, slots=True)
class QuaternionTorqueController:
    kp: tuple[float, float, float] = (0.25, 0.25, 0.25)
    kd: tuple[float, float, float] = (1.0, 1.0, 1.0)
    max_torque_n_m: float = 0.1
    detumble_rate_threshold_rad_s: float = 0.5

    def __post_init__(self) -> None:
        for name, values in (("kp", self.kp), ("kd", self.kd)):
            array = np.asarray(values, dtype=float)
            if array.size != 3 or not np.all(np.isfinite(array)) or np.any(array < 0.0):
                raise ValueError(f"{name} must contain three finite, nonnegative gains")
        if not isfinite(self.max_torque_n_m) or self.max_torque_n_m <= 0.0:
            raise ValueError("max_torque_n_m must be finite and positive")
        if not isfinite(self.detumble_rate_threshold_rad_s) or self.detumble_rate_threshold_rad_s < 0.0:
            raise ValueError("detumble_rate_threshold_rad_s must be finite and nonnegative")

    def control(self, solution: AttitudeSolution, reference: GuidanceReference) -> RequestedEffort | None:
        if not solution.valid_for_control or reference.attitude_quat_from_frame is None:
            return None
        q = np.asarray(solution.attitude_quat_bn)
        q_des = np.asarray(reference.attitude_quat_from_frame)
        omega = np.asarray(solution.angular_rate_body_rad_s)
        if np.linalg.norm(omega) > self.detumble_rate_threshold_rad_s:
            torque = -np.asarray(self.kd) * omega
        else:
            q_error = _quaternion_error(q_des, q)
            torque = -np.asarray(self.kp) * q_error[1:] - np.asarray(self.kd) * omega
        norm = float(np.linalg.norm(torque))
        if norm > self.max_torque_n_m > 0.0:
            torque *= self.max_torque_n_m / norm
        return RequestedEffort(
            "attitude-torque",
            RequestedEffortKind.TORQUE,
            solution.generated_at,
            solution.frame,
            reference.validity,
            torque_n_m=tuple(float(value) for value in torque),
        )


class AttitudeTorqueController(Protocol):
    def control(self, solution: AttitudeSolution, reference: GuidanceReference) -> RequestedEffort | None: ...


@dataclass(frozen=True, slots=True)
class SmallAngleLqrTorqueController:
    """Snapshot-free small-angle LQR law adapted to typed SI inputs."""

    gain: tuple[tuple[float, ...], ...] = (
        (0.25, 0.0, 0.0, 1.0, 0.0, 0.0),
        (0.0, 0.25, 0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.25, 0.0, 0.0, 1.0),
    )
    max_torque_n_m: float = 0.1

    def __post_init__(self) -> None:
        gain = np.asarray(self.gain, dtype=float)
        if gain.shape != (3, 6) or not np.all(np.isfinite(gain)):
            raise ValueError("gain must be a finite 3x6 matrix")
        if not isfinite(self.max_torque_n_m) or self.max_torque_n_m <= 0.0:
            raise ValueError("max_torque_n_m must be finite and positive")

    def control(self, solution: AttitudeSolution, reference: GuidanceReference) -> RequestedEffort | None:
        if not solution.valid_for_control or reference.attitude_quat_from_frame is None:
            return None
        q_error = _quaternion_error(
            np.asarray(reference.attitude_quat_from_frame),
            np.asarray(solution.attitude_quat_bn),
        )
        state_error = np.concatenate((2.0 * q_error[1:], np.asarray(solution.angular_rate_body_rad_s)))
        torque = -(np.asarray(self.gain) @ state_error)
        norm = float(np.linalg.norm(torque))
        if norm > self.max_torque_n_m:
            torque *= self.max_torque_n_m / norm
        return RequestedEffort(
            "attitude-lqr-torque",
            RequestedEffortKind.TORQUE,
            solution.generated_at,
            solution.frame,
            reference.validity,
            torque_n_m=tuple(float(value) for value in torque),
        )


class AttitudeAllocatorKind(str, Enum):
    REACTION_WHEEL = "reaction_wheel"
    MAGNETORQUER = "magnetorquer"
    CMG = "cmg"
    IDEAL_WRENCH = "ideal_wrench"


@dataclass(frozen=True, slots=True)
class AttitudeAllocatorConfig:
    satellite_id: str
    kind: AttitudeAllocatorKind
    actuator_id: str
    actuator_frame: FrameId
    axes_body: tuple[tuple[float, ...], ...] = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    limits: tuple[float, ...] = (0.1, 0.1, 0.1)
    cmg_momentum_n_m_s: tuple[float, ...] = (1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        if not self.satellite_id.strip() or not self.actuator_id.strip():
            raise ValueError("satellite_id and actuator_id must be non-empty")
        if not isinstance(self.kind, AttitudeAllocatorKind):
            raise TypeError("kind must be AttitudeAllocatorKind")
        axes = np.asarray(self.axes_body, dtype=float)
        limits = np.asarray(self.limits, dtype=float)
        if not np.all(np.isfinite(limits)) or np.any(limits <= 0.0):
            raise ValueError("allocator limits must be finite and positive")
        if self.kind is AttitudeAllocatorKind.REACTION_WHEEL:
            if axes.ndim != 2 or axes.shape[1] != 3 or not np.all(np.isfinite(axes)):
                raise ValueError("reaction-wheel axes must be an Nx3 finite matrix")
            if limits.size not in (1, axes.shape[0]):
                raise ValueError("reaction-wheel limits must be scalar or match wheel count")
        elif limits.size not in (1, 3):
            raise ValueError("attitude allocator limits must be scalar or length three")
        momentum = np.asarray(self.cmg_momentum_n_m_s, dtype=float)
        if self.kind is AttitudeAllocatorKind.CMG and (
            momentum.size != 3 or not np.all(np.isfinite(momentum)) or np.any(momentum <= 0.0)
        ):
            raise ValueError("CMG momentum must contain three finite, positive values")


class AttitudeAllocator:
    def __init__(self, config: AttitudeAllocatorConfig) -> None:
        self.config = config

    def allocate(
        self,
        effort: RequestedEffort,
        solution: AttitudeSolution,
        *,
        command_id: PacketId,
    ) -> AllocationResult:
        if effort.frame != solution.frame:
            return AllocationResult(
                effort.effort_id,
                solution.generated_at,
                AllocationStatus.INFEASIBLE,
            )
        torque = np.asarray(effort.torque_n_m, dtype=float)
        payload: ReactionWheelTorqueCommand | MagnetorquerDipoleCommand | CmgGimbalRateCommand | IdealWrenchCommand
        status = AllocationStatus.EXACT
        if self.config.kind is AttitudeAllocatorKind.REACTION_WHEEL:
            axes = np.asarray(self.config.axes_body, dtype=float).T
            wheel_torque = -np.linalg.pinv(axes) @ torque
            wheel_torque, saturated = _clip_components(wheel_torque, self.config.limits)
            payload = ReactionWheelTorqueCommand(tuple(float(value) for value in wheel_torque))
            achieved = -(axes @ wheel_torque)
        elif self.config.kind is AttitudeAllocatorKind.MAGNETORQUER:
            if solution.magnetic_field_body_t is None:
                return AllocationResult(effort.effort_id, solution.generated_at, AllocationStatus.INFEASIBLE)
            magnetic = np.asarray(solution.magnetic_field_body_t)
            magnitude_squared = float(magnetic @ magnetic)
            if magnitude_squared <= 1e-24:
                return AllocationResult(effort.effort_id, solution.generated_at, AllocationStatus.INFEASIBLE)
            dipole = np.cross(magnetic, torque) / magnitude_squared
            dipole, saturated = _clip_components(dipole, self.config.limits)
            payload = MagnetorquerDipoleCommand(tuple(float(value) for value in dipole))
            achieved = np.cross(dipole, magnetic)
        elif self.config.kind is AttitudeAllocatorKind.CMG:
            momentum = np.asarray(self.config.cmg_momentum_n_m_s)
            rates = np.divide(torque, momentum, out=np.zeros_like(torque), where=np.abs(momentum) > 0.0)
            rates, saturated = _clip_components(rates, self.config.limits)
            payload = CmgGimbalRateCommand(tuple(float(value) for value in rates))
            achieved = rates * momentum
        else:
            payload = IdealWrenchCommand((0.0, 0.0, 0.0), tuple(float(value) for value in torque))
            achieved = torque
            saturated = False
        residual = torque - achieved
        if saturated:
            status = AllocationStatus.SATURATED
        elif np.linalg.norm(residual) > 1e-12:
            status = AllocationStatus.RESIDUAL
        command = ActuatorCommand(
            command_id,
            self.config.satellite_id,
            self.config.actuator_id,
            solution.generated_at,
            effort.validity,
            self.config.actuator_frame,
            payload,
        )
        return AllocationResult(
            effort.effort_id,
            solution.generated_at,
            status,
            (command,),
            residual_torque_n_m=tuple(float(value) for value in residual),
        )


def _quaternion_error(desired: np.ndarray, current: np.ndarray) -> np.ndarray:
    conjugate = np.array([desired[0], -desired[1], -desired[2], -desired[3]])
    error = normalize_quaternion(quaternion_multiply(conjugate, current))
    return -error if error[0] < 0.0 else error


def _quaternion_from_two_vectors(reference: np.ndarray, body: np.ndarray) -> np.ndarray:
    first = _unit(reference)
    second = _unit(body)
    dot = float(np.clip(first @ second, -1.0, 1.0))
    if dot < -1.0 + 1e-12:
        candidate = np.array([1.0, 0.0, 0.0]) if abs(first[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = _unit(np.cross(first, candidate))
        return np.array([0.0, *axis])
    # ``quaternion_to_dcm_bn`` is a passive inertial-to-body transform, so the
    # vector part has the opposite sign from the common active-rotation form.
    quaternion = np.array([1.0 + dot, *np.cross(second, first)])
    return normalize_quaternion(quaternion)


def _unit(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=float).reshape(3)
    norm = float(np.linalg.norm(vector))
    if not isfinite(norm) or norm <= 0.0:
        raise ValueError("reference direction must be finite and nonzero")
    return vector / norm


def _unit_or_none(value: np.ndarray) -> np.ndarray | None:
    try:
        return _unit(value)
    except ValueError:
        return None


def _calibrated_vector(values: tuple[float, float, float], calibration: SensorCalibration) -> np.ndarray:
    return (np.asarray(values, dtype=float) - np.asarray(calibration.bias)) * np.asarray(calibration.scale)


def _clip_components(values: np.ndarray, limits: tuple[float, ...]) -> tuple[np.ndarray, bool]:
    limit = np.asarray(limits, dtype=float).reshape(-1)
    if limit.size == 1:
        limit = np.full(values.size, limit[0])
    if limit.size != values.size or np.any(limit <= 0.0):
        raise ValueError("allocator limits must be positive and match device coordinates")
    return np.clip(values, -limit, limit), bool(np.any(np.abs(values) > limit + 1e-15))


def _add_ticks(tag: ClockTag, ticks: int) -> ClockTag:
    return ClockTag(tag.clock_id, tag.ticks + ticks, tag.tick_period_ns, tag.scale, tag.validity, tag.reset_counter)


def _optional_tuple(value: object, size: int):
    if value is None:
        return None
    values = tuple(float(item) for item in list(value))  # type: ignore[arg-type]
    if len(values) != size or not all(isfinite(item) for item in values):
        raise ValueError("invalid navigation snapshot vector")
    return values


def _packet_from_dict(value: object) -> PacketId:
    mapping = dict(value)  # type: ignore[arg-type]
    return PacketId(str(mapping["source_id"]), str(mapping["boot_id"]), int(mapping["sequence"]))


def _clock_to_dict(value: ClockTag | None) -> dict[str, object] | None:
    if value is None:
        return None
    return {
        "clock_id": value.clock_id,
        "ticks": value.ticks,
        "tick_period_ns": value.tick_period_ns,
        "scale": value.scale.value,
        "validity": value.validity.value,
        "reset_counter": value.reset_counter,
    }


def _clock_from_dict(value: object) -> ClockTag | None:
    if value is None:
        return None
    mapping = dict(value)  # type: ignore[arg-type]
    return ClockTag(
        str(mapping["clock_id"]),
        int(mapping["ticks"]),
        int(mapping["tick_period_ns"]),
        ClockScale(str(mapping["scale"])),
        TimeValidity(str(mapping["validity"])),
        int(mapping["reset_counter"]),
    )


def _elapsed_seconds(start: ClockTag, end: ClockTag) -> float | None:
    identity_start = (start.clock_id, start.tick_period_ns, start.scale, start.reset_counter)
    identity_end = (end.clock_id, end.tick_period_ns, end.scale, end.reset_counter)
    if identity_start != identity_end:
        return None
    return (end.ticks - start.ticks) * start.tick_period_ns * 1.0e-9
