"""Typed SI orbit and relative navigation for GNC v2 reference stacks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

import numpy as np

from sim.core.models import Measurement as LegacyMeasurement
from sim.core.models import StateBelief as LegacyStateBelief
from sim.estimation.orbit_ekf import OrbitEKFEstimator
from sim.estimation.relative_hcw_ekf import HCWRelativeEKFEstimator
from sim.flight_software.contracts import (
    ClockScale,
    ClockTag,
    DataValidity,
    FrameId,
    GnssOwnStateMeasurement,
    IdealOwnStateMeasurement,
    IdealTrackedObjectStateMeasurement,
    InputEvent,
    InputKind,
    MeasurementEvent,
    ModeledFaultIndicationPayload,
    PacketId,
    RelativeObservationMeasurement,
    TelemetryField,
    TimeValidity,
)
from sim.gnc.attitude_v2 import AttitudeNavigator, AttitudeSolution, SensorCalibration, SensorMounting
from sim.gnc.contracts import BeliefState, EstimateValidity, StateEstimate
from sim.utils.frames import eci_relative_to_ric_rect
from sim.utils.quaternion import quaternion_to_dcm_bn


class NavigationInitializationMode(str, Enum):
    COLD = "cold"
    LOADED = "loaded"
    IDEAL = "ideal"


class OrbitFilterKind(str, Enum):
    SAMPLE_HOLD = "sample_hold"
    ALPHA_BETA = "alpha_beta"
    EKF = "ekf"


@dataclass(frozen=True, slots=True)
class LoadedOwnState:
    position_eci_m: tuple[float, float, float]
    velocity_eci_m_s: tuple[float, float, float]
    epoch: ClockTag

    def __post_init__(self) -> None:
        _finite_vector("position_eci_m", self.position_eci_m, 3)
        _finite_vector("velocity_eci_m_s", self.velocity_eci_m_s, 3)


@dataclass(frozen=True, slots=True)
class RelativeStateEstimateSI:
    """Deputy (own satellite) state relative to the named chief/target in chief RIC."""

    target_id: str
    frame: FrameId
    epoch: ClockTag
    position_m: tuple[float, float, float]
    velocity_m_s: tuple[float, float, float]
    range_m: float
    range_rate_m_s: float
    source_packets: tuple[PacketId, ...]
    validity: EstimateValidity
    chief_position_eci_m: tuple[float, float, float] | None = None
    chief_velocity_eci_m_s: tuple[float, float, float] | None = None


@dataclass(frozen=True, slots=True)
class OrbitNavigationSolution:
    generated_at: ClockTag
    inertial_frame: FrameId
    relative_frame: FrameId
    position_eci_m: tuple[float, float, float] | None
    velocity_eci_m_s: tuple[float, float, float] | None
    mass_kg: float | None
    attitude: AttitudeSolution
    relative_tracks: tuple[RelativeStateEstimateSI, ...]
    active_faults: tuple[tuple[str, str], ...]
    belief: BeliefState
    own_state_epoch: ClockTag | None = None

    @property
    def own_state_valid(self) -> bool:
        return self.position_eci_m is not None and self.velocity_eci_m_s is not None

    def relative_track(self, target_id: str | None = None) -> RelativeStateEstimateSI | None:
        if target_id is None:
            return self.relative_tracks[0] if self.relative_tracks else None
        return next((track for track in self.relative_tracks if track.target_id == target_id), None)


class OrbitNavigator:
    def __init__(
        self,
        *,
        initialization: NavigationInitializationMode,
        body_frame: FrameId,
        inertial_frame: FrameId,
        relative_frame: FrameId,
        loaded_own_state: LoadedOwnState | None = None,
        sensor_mountings: tuple[SensorMounting, ...] = (),
        sensor_calibrations: tuple[SensorCalibration, ...] = (),
        filter_kind: OrbitFilterKind = OrbitFilterKind.SAMPLE_HOLD,
        alpha: float = 0.85,
        beta: float = 0.05,
        ekf_step_s: float = 1.0,
        ekf_process_noise_diag_si: tuple[float, ...] = (1.0e-4, 1.0e-4, 1.0e-4, 1.0e-8, 1.0e-8, 1.0e-8),
        ekf_measurement_noise_diag_si: tuple[float, ...] = (25.0, 25.0, 25.0, 0.01, 0.01, 0.01),
        ekf_initial_covariance_diag_si: tuple[float, ...] = (1.0e4, 1.0e4, 1.0e4, 100.0, 100.0, 100.0),
        relative_mean_motion_rad_s: float = 0.0011,
        ekf_nis_limit: float = 30.0,
        retain_full_provenance: bool = True,
    ) -> None:
        if not isinstance(initialization, NavigationInitializationMode):
            raise TypeError("initialization must be NavigationInitializationMode")
        if initialization is NavigationInitializationMode.LOADED and loaded_own_state is None:
            raise ValueError("loaded navigation initialization requires loaded_own_state")
        if not isinstance(filter_kind, OrbitFilterKind):
            raise TypeError("filter_kind must be OrbitFilterKind")
        if not isfinite(alpha) or not 0.0 < alpha <= 1.0:
            raise ValueError("navigation alpha must be in (0, 1]")
        if not isfinite(beta) or not 0.0 <= beta <= 1.0:
            raise ValueError("navigation beta must be in [0, 1]")
        if not isfinite(ekf_step_s) or ekf_step_s <= 0.0:
            raise ValueError("ekf_step_s must be finite and positive")
        if not isfinite(relative_mean_motion_rad_s) or relative_mean_motion_rad_s <= 0.0:
            raise ValueError("relative_mean_motion_rad_s must be finite and positive")
        if not isfinite(ekf_nis_limit) or ekf_nis_limit <= 0.0:
            raise ValueError("ekf_nis_limit must be finite and positive")
        for name, values in (
            ("ekf_process_noise_diag_si", ekf_process_noise_diag_si),
            ("ekf_measurement_noise_diag_si", ekf_measurement_noise_diag_si),
            ("ekf_initial_covariance_diag_si", ekf_initial_covariance_diag_si),
        ):
            if len(values) != 6 or any(not isfinite(value) or value < 0.0 for value in values):
                raise ValueError(f"{name} must contain six finite nonnegative values")
        self.initialization = initialization
        self.body_frame = body_frame
        self.inertial_frame = inertial_frame
        self.relative_frame = relative_frame
        self.filter_kind = filter_kind
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.ekf_step_s = float(ekf_step_s)
        self.ekf_nis_limit = float(ekf_nis_limit)
        self.relative_mean_motion_rad_s = float(relative_mean_motion_rad_s)
        scale = np.full(6, 1.0e-6)
        self._ekf_initial_covariance = np.asarray(ekf_initial_covariance_diag_si, dtype=float) * scale
        self._ekf_process_noise = np.asarray(ekf_process_noise_diag_si, dtype=float) * scale
        self._ekf_measurement_noise = np.asarray(ekf_measurement_noise_diag_si, dtype=float) * scale
        self._own_ekf = OrbitEKFEstimator(
            398600.4418,
            self.ekf_step_s,
            self._ekf_process_noise,
            self._ekf_measurement_noise,
        )
        self._relative_ekfs: dict[str, HCWRelativeEKFEstimator] = {}
        self._own_filter_belief: LegacyStateBelief | None = None
        self._relative_filter_beliefs: dict[str, LegacyStateBelief] = {}
        self._ekf_diagnostics: dict[str, float] = {}
        self._sensor_mountings = sensor_mountings
        self._sensor_calibrations = sensor_calibrations
        self._mountings = {mounting.sensor_id: mounting for mounting in sensor_mountings}
        self._calibrations = {calibration.sensor_id: calibration for calibration in sensor_calibrations}
        self._retain_full_provenance = bool(retain_full_provenance)
        self._attitude = AttitudeNavigator(
            body_frame=body_frame,
            inertial_frame=inertial_frame,
            mountings=sensor_mountings,
            calibrations=sensor_calibrations,
            retain_full_provenance=retain_full_provenance,
        )
        self._position = None if loaded_own_state is None else loaded_own_state.position_eci_m
        self._velocity = None if loaded_own_state is None else loaded_own_state.velocity_eci_m_s
        self._mass: float | None = None
        self._own_epoch = None if loaded_own_state is None else loaded_own_state.epoch
        self._own_packets: list[PacketId] = []
        self._tracks: dict[str, RelativeStateEstimateSI] = {}
        self._faults: dict[str, str] = {}
        self._seen: set[PacketId] = set()
        self._last_sample: dict[str, ClockTag] = {}
        self._degraded = False

    def ingest(self, events: tuple[InputEvent, ...]) -> None:
        self._attitude.ingest(events)
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
            if not self._chronological(measurement.sensor_id, measurement.sample_time):
                self._degraded = True
                continue
            if event.quality.validity is DataValidity.SUSPECT:
                self._degraded = True
            payload = measurement.payload
            if isinstance(payload, GnssOwnStateMeasurement):
                if measurement.frame != self.inertial_frame:
                    self._degraded = True
                    continue
                self._set_own(payload.position_m, payload.velocity_m_s, None, measurement.sample_time, event.packet_id)
            elif isinstance(payload, IdealOwnStateMeasurement):
                if self.initialization is not NavigationInitializationMode.IDEAL:
                    continue
                if measurement.frame not in (self.inertial_frame, self.body_frame):
                    self._degraded = True
                    continue
                self._set_own(
                    payload.position_m,
                    payload.velocity_m_s,
                    payload.mass_kg,
                    measurement.sample_time,
                    event.packet_id,
                )
            elif isinstance(payload, IdealTrackedObjectStateMeasurement):
                self._ingest_ideal_tracked(measurement, payload, event.packet_id)
            elif isinstance(payload, RelativeObservationMeasurement):
                self._ingest_relative(measurement, payload, event.packet_id)

    def solution(self, generated_at: ClockTag) -> OrbitNavigationSolution:
        self._propagate_filters(generated_at)
        attitude = self._attitude.solution(generated_at)
        own_validity = (
            EstimateValidity.INVALID
            if self._position is None or self._velocity is None
            else EstimateValidity.DEGRADED
            if self._degraded or self._faults
            else EstimateValidity.VALID
        )
        own_fields: list[TelemetryField] = []
        if self._position is not None:
            own_fields.extend(
                TelemetryField(f"position_{axis}_m", value, "m") for axis, value in zip("xyz", self._position)
            )
        if self._velocity is not None:
            own_fields.extend(
                TelemetryField(f"velocity_{axis}_m_s", value, "m/s") for axis, value in zip("xyz", self._velocity)
            )
        if self._mass is not None:
            own_fields.append(TelemetryField("mass_kg", self._mass, "kg"))
        for name, value in sorted(self._ekf_diagnostics.items()):
            own_fields.append(TelemetryField(name, value))
        own_estimate = (
            StateEstimate(
                "own-orbit",
                generated_at,
                self.inertial_frame,
                tuple(own_fields),
                source_packets=tuple(self._own_packets),
                validity=own_validity,
            )
            if own_fields
            else None
        )
        tracked = tuple(
            StateEstimate(
                f"relative.{track.target_id}",
                track.epoch,
                track.frame,
                tuple(
                    [
                        *(
                            TelemetryField(f"position_{axis}_m", value, "m")
                            for axis, value in zip("ric", track.position_m)
                        ),
                        *(
                            TelemetryField(f"velocity_{axis}_m_s", value, "m/s")
                            for axis, value in zip("ric", track.velocity_m_s)
                        ),
                        TelemetryField("range_m", track.range_m, "m"),
                        TelemetryField("range_rate_m_s", track.range_rate_m_s, "m/s"),
                    ]
                ),
                source_packets=track.source_packets,
                validity=track.validity,
            )
            for track in sorted(self._tracks.values(), key=lambda item: item.target_id)
        )
        provenance = tuple(
            dict.fromkeys((*self._own_packets, *(p for track in self._tracks.values() for p in track.source_packets)))
        )
        belief = BeliefState(
            generated_at,
            own_state=own_estimate,
            attitude_state=attitude.belief.attitude_state,
            tracked_objects=tracked,
            environment_estimates=attitude.belief.environment_estimates,
            health_state=tuple(
                TelemetryField(f"fault.{component}", code) for component, code in sorted(self._faults.items())
            ),
            provenance=provenance,
        )
        return OrbitNavigationSolution(
            generated_at,
            self.inertial_frame,
            self.relative_frame,
            self._position,
            self._velocity,
            self._mass,
            attitude,
            tuple(sorted(self._tracks.values(), key=lambda item: item.target_id)),
            tuple(sorted({**dict(attitude.active_faults), **self._faults}.items())),
            belief,
            own_state_epoch=self._own_epoch,
        )

    def control_solution(self, generated_at: ClockTag) -> OrbitNavigationSolution:
        """Return the same control state without materializing audit beliefs.

        This is intended for real-time stacks whose output does not expose the
        navigation belief.  Filter propagation, navigation state, tracks, and
        active faults are unchanged.
        """

        self._propagate_filters(generated_at)
        attitude = self._attitude.control_solution(generated_at)
        return OrbitNavigationSolution(
            generated_at,
            self.inertial_frame,
            self.relative_frame,
            self._position,
            self._velocity,
            self._mass,
            attitude,
            tuple(sorted(self._tracks.values(), key=lambda item: item.target_id)),
            tuple(sorted({**dict(attitude.active_faults), **self._faults}.items())),
            BeliefState(generated_at),
            own_state_epoch=self._own_epoch,
        )

    def snapshot_state(self) -> dict[str, object]:
        return {
            "position": self._position,
            "velocity": self._velocity,
            "mass": self._mass,
            "own_epoch": _clock_to_dict(self._own_epoch),
            "own_packets": [_packet_to_dict(packet) for packet in self._own_packets],
            "tracks": [
                _track_to_dict(track) for track in sorted(self._tracks.values(), key=lambda item: item.target_id)
            ],
            "faults": dict(sorted(self._faults.items())),
            "seen": [_packet_to_dict(packet) for packet in sorted(self._seen, key=_packet_key)],
            "last_sample": {key: _clock_to_dict(value) for key, value in sorted(self._last_sample.items())},
            "degraded": self._degraded,
            "attitude": self._attitude.snapshot_state(),
            "own_filter_belief": _legacy_belief_to_dict(self._own_filter_belief),
            "relative_filter_beliefs": {
                key: _legacy_belief_to_dict(value) for key, value in sorted(self._relative_filter_beliefs.items())
            },
            "ekf_diagnostics": dict(sorted(self._ekf_diagnostics.items())),
        }

    def restore_state(self, state: dict[str, object]) -> None:
        position = _optional_vector(state.get("position"), 3)
        velocity = _optional_vector(state.get("velocity"), 3)
        mass_value = state.get("mass")
        mass = None if mass_value is None else float(mass_value)
        if mass is not None and (not isfinite(mass) or mass < 0.0):
            raise ValueError("navigation snapshot mass is invalid")
        restored_tracks = [_track_from_dict(item) for item in list(state.get("tracks", []))]
        tracks = {track.target_id: track for track in restored_tracks}
        last_sample = {str(key): _required_clock(value) for key, value in dict(state.get("last_sample", {})).items()}
        attitude_state = state.get("attitude")
        if not isinstance(attitude_state, dict):
            raise ValueError("navigation snapshot attitude state is invalid")
        restored_attitude = AttitudeNavigator(
            body_frame=self.body_frame,
            inertial_frame=self.inertial_frame,
            mountings=self._sensor_mountings,
            calibrations=self._sensor_calibrations,
            retain_full_provenance=self._retain_full_provenance,
        )
        restored_attitude.restore_state(attitude_state)
        self._position = position
        self._velocity = velocity
        self._mass = mass
        self._own_epoch = _clock_from_dict(state.get("own_epoch"))
        self._own_packets = [_packet_from_dict(item) for item in list(state.get("own_packets", []))]
        self._tracks = tracks
        self._faults = {str(key): str(value) for key, value in dict(state.get("faults", {})).items()}
        self._seen = {_packet_from_dict(item) for item in list(state.get("seen", []))}
        self._last_sample = last_sample
        self._degraded = bool(state.get("degraded", False))
        self._attitude = restored_attitude
        self._own_filter_belief = _legacy_belief_from_dict(state.get("own_filter_belief"))
        self._relative_filter_beliefs = {
            str(key): _required_legacy_belief(value)
            for key, value in dict(state.get("relative_filter_beliefs", {})).items()
        }
        self._ekf_diagnostics = {
            str(key): float(value) for key, value in dict(state.get("ekf_diagnostics", {})).items()
        }

    def _set_own(
        self,
        position: tuple[float, float, float] | None,
        velocity: tuple[float, float, float] | None,
        mass: float | None,
        epoch: ClockTag,
        packet: PacketId,
    ) -> None:
        if self.filter_kind is OrbitFilterKind.EKF and position is not None and velocity is not None:
            measured_state = np.concatenate((np.asarray(position), np.asarray(velocity))) / 1000.0
            epoch_s = _clock_seconds(epoch)
            if self._own_filter_belief is None:
                self._own_filter_belief = LegacyStateBelief(
                    measured_state,
                    np.diag(self._ekf_initial_covariance),
                    epoch_s,
                )
            else:
                prior = self._own_filter_belief
                candidate = self._own_ekf.update(prior, LegacyMeasurement(measured_state, epoch_s), epoch_s)
                diagnostics = self._own_ekf.last_update_diagnostics
                if diagnostics is not None and diagnostics.nis > self.ekf_nis_limit:
                    candidate = self._own_ekf.update(prior, None, epoch_s)
                    self._degraded = True
                    self._ekf_diagnostics["own_ekf_measurement_rejected"] = 1.0
                else:
                    self._ekf_diagnostics["own_ekf_measurement_rejected"] = 0.0
                if diagnostics is not None:
                    self._ekf_diagnostics["own_ekf_nis"] = diagnostics.nis
                self._own_filter_belief = candidate
            position = tuple(float(value) for value in self._own_filter_belief.state[:3] * 1000.0)
            velocity = tuple(float(value) for value in self._own_filter_belief.state[3:] * 1000.0)
            self._ekf_diagnostics["own_ekf_covariance_trace_si"] = float(
                np.trace(self._own_filter_belief.covariance) * 1.0e6
            )
        if (
            self.filter_kind is OrbitFilterKind.ALPHA_BETA
            and position is not None
            and velocity is not None
            and self._position is not None
            and self._velocity is not None
            and self._own_epoch is not None
        ):
            dt_s = _elapsed_seconds(self._own_epoch, epoch)
            if dt_s is not None and dt_s > 0.0:
                previous_position = np.asarray(self._position, dtype=float)
                previous_velocity = np.asarray(self._velocity, dtype=float)
                predicted_position = previous_position + previous_velocity * dt_s
                residual = np.asarray(position, dtype=float) - predicted_position
                filtered_position = predicted_position + self.alpha * residual
                residual_velocity = previous_velocity + self.beta * residual / dt_s
                filtered_velocity = residual_velocity + self.alpha * (
                    np.asarray(velocity, dtype=float) - residual_velocity
                )
                position = tuple(float(value) for value in filtered_position)
                velocity = tuple(float(value) for value in filtered_velocity)
        if position is not None:
            self._position = tuple(position)
        if velocity is not None:
            self._velocity = tuple(velocity)
        if mass is not None:
            self._mass = float(mass)
        self._own_epoch = epoch
        if self._retain_full_provenance:
            self._own_packets.append(packet)
        else:
            self._own_packets[:] = (packet,)

    def _ingest_relative(
        self,
        measurement: MeasurementEvent,
        payload: RelativeObservationMeasurement,
        packet: PacketId,
    ) -> None:
        if payload.target_track_id is None:
            self._degraded = True
            return
        mounting = self._mountings.get(measurement.sensor_id)
        if mounting is None:
            if measurement.frame != self.relative_frame:
                self._degraded = True
                return
            transform = np.eye(3)
        else:
            if mounting.sensor_frame is not None and measurement.frame != mounting.sensor_frame:
                self._degraded = True
                return
            transform = self._relative_from_sensor_dcm(mounting, measurement.sample_time)
            if transform is None:
                self._degraded = True
                return
        previous = self._tracks.get(payload.target_track_id)
        range_m = payload.range_m if payload.range_m is not None else (None if previous is None else previous.range_m)
        range_rate = (
            payload.range_rate_m_s
            if payload.range_rate_m_s is not None
            else (None if previous is None else previous.range_rate_m_s)
        )
        los = payload.los_unit
        if los is None and previous is not None and previous.range_m > 0.0:
            los = tuple(value / previous.range_m for value in previous.position_m)
        if range_m is None or range_rate is None or los is None:
            return
        calibration = self._calibrations.get(measurement.sensor_id)
        los_array = _calibrated_vector(los, calibration)
        los_array = transform @ los_array
        los_array /= np.linalg.norm(los_array)
        position = los_array * range_m
        velocity = los_array * range_rate
        if payload.angular_rate_rad_s is not None:
            angular_rate = transform @ _calibrated_vector(payload.angular_rate_rad_s, calibration)
            velocity += range_m * np.cross(angular_rate, los_array)
        if self.filter_kind is OrbitFilterKind.EKF:
            target_id = payload.target_track_id
            state_km = np.concatenate((position, velocity)) / 1000.0
            epoch_s = _clock_seconds(measurement.sample_time)
            belief = self._relative_filter_beliefs.get(target_id)
            if belief is None:
                belief = LegacyStateBelief(state_km, np.diag(self._ekf_initial_covariance), epoch_s)
            else:
                estimator = self._relative_ekf(target_id)
                prior = belief
                candidate = estimator.update(prior, LegacyMeasurement(state_km, epoch_s), epoch_s)
                diagnostics = estimator.last_update_diagnostics
                if diagnostics is not None and diagnostics.nis > self.ekf_nis_limit:
                    candidate = estimator.update(prior, None, epoch_s)
                    self._degraded = True
                    self._ekf_diagnostics[f"relative_{target_id}_measurement_rejected"] = 1.0
                else:
                    self._ekf_diagnostics[f"relative_{target_id}_measurement_rejected"] = 0.0
                if diagnostics is not None:
                    self._ekf_diagnostics[f"relative_{target_id}_nis"] = diagnostics.nis
                belief = candidate
            self._relative_filter_beliefs[target_id] = belief
            position = belief.state[:3] * 1000.0
            velocity = belief.state[3:] * 1000.0
            range_m = float(np.linalg.norm(position))
            range_rate = 0.0 if range_m <= 0.0 else float(position @ velocity) / range_m
            self._ekf_diagnostics[f"relative_{target_id}_covariance_trace_si"] = float(
                np.trace(belief.covariance) * 1.0e6
            )
        if self.filter_kind is OrbitFilterKind.ALPHA_BETA and previous is not None:
            dt_s = _elapsed_seconds(previous.epoch, measurement.sample_time)
            if dt_s is not None and dt_s > 0.0:
                predicted_position = np.asarray(previous.position_m) + np.asarray(previous.velocity_m_s) * dt_s
                residual = position - predicted_position
                position = predicted_position + self.alpha * residual
                predicted_velocity = np.asarray(previous.velocity_m_s) + self.beta * residual / dt_s
                velocity = predicted_velocity + self.alpha * (velocity - predicted_velocity)
        packets = (
            (packet,)
            if previous is None or not self._retain_full_provenance
            else (*previous.source_packets, packet)
        )
        self._tracks[payload.target_track_id] = RelativeStateEstimateSI(
            payload.target_track_id,
            self.relative_frame,
            measurement.sample_time,
            tuple(float(value) for value in position),
            tuple(float(value) for value in velocity),
            float(range_m),
            float(range_rate),
            packets,
            EstimateValidity.DEGRADED if self._degraded else EstimateValidity.VALID,
        )

    def _ingest_ideal_tracked(
        self,
        measurement: MeasurementEvent,
        payload: IdealTrackedObjectStateMeasurement,
        packet: PacketId,
    ) -> None:
        if self.initialization is not NavigationInitializationMode.IDEAL:
            return
        if measurement.frame != self.inertial_frame or self._position is None or self._velocity is None:
            self._degraded = True
            return
        own_position = np.asarray(self._position, dtype=float)
        own_velocity = np.asarray(self._velocity, dtype=float)
        target_position = np.asarray(payload.position_m, dtype=float)
        target_velocity = np.asarray(payload.velocity_m_s, dtype=float)
        relative = eci_relative_to_ric_rect(
            np.concatenate((own_position, own_velocity)),
            np.concatenate((target_position, target_velocity)),
        )
        position = relative[:3]
        velocity = relative[3:]
        previous = self._tracks.get(payload.target_id)
        packets = (
            (packet,)
            if previous is None or not self._retain_full_provenance
            else (*previous.source_packets, packet)
        )
        range_m = float(np.linalg.norm(position))
        range_rate_m_s = 0.0 if range_m <= 0.0 else float(position @ velocity) / range_m
        self._tracks[payload.target_id] = RelativeStateEstimateSI(
            payload.target_id,
            self.relative_frame,
            measurement.sample_time,
            tuple(float(value) for value in position),
            tuple(float(value) for value in velocity),
            range_m,
            range_rate_m_s,
            packets,
            EstimateValidity.DEGRADED if self._degraded else EstimateValidity.VALID,
            tuple(float(value) for value in target_position),
            tuple(float(value) for value in target_velocity),
        )

    def _relative_from_sensor_dcm(
        self,
        mounting: SensorMounting,
        sample_time: ClockTag,
    ) -> np.ndarray | None:
        attitude = self._attitude.solution(sample_time)
        if self._position is None or self._velocity is None or attitude.attitude_quat_bn is None:
            return None
        position = np.asarray(self._position)
        velocity = np.asarray(self._velocity)
        radial = _unit(position)
        cross_track = _unit(np.cross(position, velocity))
        in_track = _unit(np.cross(cross_track, radial))
        inertial_to_relative = np.vstack((radial, in_track, cross_track))
        inertial_to_body = quaternion_to_dcm_bn(np.asarray(attitude.attitude_quat_bn))
        sensor_to_body = quaternion_to_dcm_bn(np.asarray(mounting.quat_body_from_sensor))
        return inertial_to_relative @ inertial_to_body.T @ sensor_to_body

    def _ingest_fault(self, payload: object) -> None:
        if not isinstance(payload, ModeledFaultIndicationPayload):
            return
        if payload.active:
            self._faults[payload.component_id] = payload.fault_code
        else:
            self._faults.pop(payload.component_id, None)

    def _chronological(self, sensor_id: str, sample_time: ClockTag) -> bool:
        previous = self._last_sample.get(sensor_id)
        if previous is not None:
            delta = _elapsed_seconds(previous, sample_time)
            if delta is None or delta < 0.0:
                return False
        self._last_sample[sensor_id] = sample_time
        return True

    def _relative_ekf(self, target_id: str) -> HCWRelativeEKFEstimator:
        estimator = self._relative_ekfs.get(target_id)
        if estimator is None:
            estimator = HCWRelativeEKFEstimator(
                self.relative_mean_motion_rad_s,
                self.ekf_step_s,
                self._ekf_process_noise,
                self._ekf_measurement_noise,
            )
            self._relative_ekfs[target_id] = estimator
        return estimator

    def _propagate_filters(self, generated_at: ClockTag) -> None:
        if self.filter_kind is not OrbitFilterKind.EKF:
            return
        output_s = _clock_seconds(generated_at)
        if self._own_filter_belief is not None and output_s > self._own_filter_belief.last_update_t_s:
            self._own_filter_belief = self._own_ekf.update(self._own_filter_belief, None, output_s)
            self._position = tuple(float(value) for value in self._own_filter_belief.state[:3] * 1000.0)
            self._velocity = tuple(float(value) for value in self._own_filter_belief.state[3:] * 1000.0)
            self._own_epoch = generated_at
        for target_id, belief in tuple(self._relative_filter_beliefs.items()):
            if output_s <= belief.last_update_t_s:
                continue
            updated = self._relative_ekf(target_id).update(belief, None, output_s)
            self._relative_filter_beliefs[target_id] = updated
            previous = self._tracks.get(target_id)
            if previous is None:
                continue
            position = updated.state[:3] * 1000.0
            velocity = updated.state[3:] * 1000.0
            range_m = float(np.linalg.norm(position))
            range_rate = 0.0 if range_m <= 0.0 else float(position @ velocity) / range_m
            self._tracks[target_id] = RelativeStateEstimateSI(
                target_id, previous.frame, generated_at,
                tuple(float(value) for value in position), tuple(float(value) for value in velocity),
                range_m, range_rate, previous.source_packets, previous.validity,
                previous.chief_position_eci_m, previous.chief_velocity_eci_m_s,
            )


def _finite_vector(name: str, values: tuple[float, ...], size: int) -> None:
    if len(values) != size or not all(isfinite(float(value)) for value in values):
        raise ValueError(f"{name} must contain {size} finite values")


def _calibrated_vector(values: tuple[float, float, float], calibration: SensorCalibration | None) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if calibration is None:
        return vector
    return (vector - np.asarray(calibration.bias, dtype=float)) * np.asarray(calibration.scale, dtype=float)


def _unit(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(value))
    if not isfinite(norm) or norm <= 0.0:
        raise ValueError("vector must be finite and nonzero")
    return value / norm


def _optional_vector(value: object, size: int) -> tuple[float, ...] | None:
    if value is None:
        return None
    result = tuple(float(item) for item in list(value))  # type: ignore[arg-type]
    _finite_vector("snapshot vector", result, size)
    return result


def _required_vector(value: object, size: int) -> tuple[float, ...]:
    result = _optional_vector(value, size)
    if result is None:
        raise ValueError("snapshot vector is required")
    return result


def _packet_key(packet: PacketId) -> tuple[str, str, int]:
    return packet.source_id, packet.boot_id, packet.sequence


def _packet_to_dict(packet: PacketId) -> dict[str, object]:
    return {"source_id": packet.source_id, "boot_id": packet.boot_id, "sequence": packet.sequence}


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


def _required_clock(value: object) -> ClockTag:
    result = _clock_from_dict(value)
    if result is None:
        raise ValueError("snapshot clock is required")
    return result


def _elapsed_seconds(start: ClockTag, end: ClockTag) -> float | None:
    if (start.clock_id, start.tick_period_ns, start.scale, start.reset_counter) != (
        end.clock_id,
        end.tick_period_ns,
        end.scale,
        end.reset_counter,
    ):
        return None
    return (end.ticks - start.ticks) * start.tick_period_ns * 1.0e-9


def _clock_seconds(value: ClockTag) -> float:
    return value.ticks * value.tick_period_ns * 1.0e-9


def _legacy_belief_to_dict(value: LegacyStateBelief | None) -> dict[str, object] | None:
    if value is None:
        return None
    return {
        "state": np.asarray(value.state, dtype=float).tolist(),
        "covariance": np.asarray(value.covariance, dtype=float).tolist(),
        "last_update_t_s": float(value.last_update_t_s),
    }


def _legacy_belief_from_dict(value: object) -> LegacyStateBelief | None:
    if value is None:
        return None
    mapping = dict(value)  # type: ignore[arg-type]
    state = np.asarray(mapping["state"], dtype=float).reshape(6)
    covariance = np.asarray(mapping["covariance"], dtype=float).reshape(6, 6)
    epoch = float(mapping["last_update_t_s"])
    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(covariance)) or not isfinite(epoch):
        raise ValueError("navigation EKF snapshot contains non-finite values")
    return LegacyStateBelief(state, covariance, epoch)


def _required_legacy_belief(value: object) -> LegacyStateBelief:
    result = _legacy_belief_from_dict(value)
    if result is None:
        raise ValueError("navigation EKF belief is required")
    return result


def _track_to_dict(track: RelativeStateEstimateSI) -> dict[str, object]:
    return {
        "target_id": track.target_id,
        "frame": {"name": track.frame.name, "registry_version": track.frame.registry_version},
        "epoch": _clock_to_dict(track.epoch),
        "position_m": track.position_m,
        "velocity_m_s": track.velocity_m_s,
        "range_m": track.range_m,
        "range_rate_m_s": track.range_rate_m_s,
        "source_packets": [_packet_to_dict(packet) for packet in track.source_packets],
        "validity": track.validity.value,
        "chief_position_eci_m": track.chief_position_eci_m,
        "chief_velocity_eci_m_s": track.chief_velocity_eci_m_s,
    }


def _track_from_dict(value: object) -> RelativeStateEstimateSI:
    mapping = dict(value)  # type: ignore[arg-type]
    frame = dict(mapping["frame"])  # type: ignore[arg-type]
    return RelativeStateEstimateSI(
        str(mapping["target_id"]),
        FrameId(str(frame["name"]), str(frame["registry_version"])),
        _required_clock(mapping["epoch"]),
        _required_vector(mapping["position_m"], 3),
        _required_vector(mapping["velocity_m_s"], 3),
        float(mapping["range_m"]),
        float(mapping["range_rate_m_s"]),
        tuple(_packet_from_dict(item) for item in list(mapping["source_packets"])),  # type: ignore[arg-type]
        EstimateValidity(str(mapping["validity"])),
        _optional_vector(mapping.get("chief_position_eci_m"), 3),
        _optional_vector(mapping.get("chief_velocity_eci_m_s"), 3),
    )
