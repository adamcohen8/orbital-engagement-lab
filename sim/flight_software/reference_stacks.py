"""Complete in-process reference flight-software stacks for the GNC v2 slices.

Only truth-free boundary records enter or leave this module.  Simulator-side
sensor sampling, actuator realization, and dynamics remain separate owners.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from enum import Enum
from hashlib import sha256
from math import acos, isfinite, sqrt
from typing import Final

import numpy as np

from sim.gnc.attitude_v2 import (
    AttitudeAllocator,
    AttitudeAllocatorConfig,
    AttitudeAllocatorKind,
    AttitudeNavigator,
    AttitudeReferenceConfig,
    AttitudeReferenceGenerator,
    AttitudeReferenceMode,
    QuaternionTorqueController,
    SensorCalibration,
    SensorMounting,
    SmallAngleLqrTorqueController,
)
from sim.gnc.contracts import AllocationResult, AllocationStatus, RequestedEffort, RequestedEffortKind
from sim.gnc.executive_v2 import (
    ExecutiveObservation,
    ReferenceExecutiveConfig,
    ReferenceMissionExecutive,
)
from sim.gnc.navigation_v2 import (
    LoadedOwnState,
    NavigationInitializationMode,
    OrbitFilterKind,
    OrbitNavigationSolution,
    OrbitNavigator,
)
from sim.gnc.operations_v2 import (
    AdcsModeConfig,
    AdcsModeManager,
    AdcsOperationalMode,
    ConjunctionAvoidancePlanner,
    ConjunctionConfig,
    HcwManeuverConfig,
    HcwManeuverPlanner,
    HealthManagerConfig,
    ManeuverPlan,
    MomentumUnloadConfig,
    OnboardCommandService,
    ResourceLimits,
    ResourceMonitor,
    StackHealthManager,
    resource_telemetry,
)
from sim.gnc.orbit_v2 import (
    TranslationAllocator,
    TranslationAllocatorConfig,
    TranslationControlConfig,
    TranslationController,
    TranslationMode,
)
from sim.utils.quaternion import quaternion_to_dcm_bn

from .clocks import clock_tag_elapsed_ns
from .contracts import (
    ActuatorCommand,
    ActuatorCommandReceipt,
    ActuatorTelemetryPayload,
    BootEvent,
    ClockTag,
    CommandDisposition,
    DataValidity,
    DiagnosticTelemetry,
    FlightSoftwareInputBatch,
    FlightSoftwareOutput,
    FlightSoftwareSnapshot,
    FrameId,
    GroundCommandKind,
    InputEvent,
    InputKind,
    MagnetorquerDipoleCommand,
    MeasurementEvent,
    PacketId,
    SatelliteFlightSoftware,
    ShutdownEvent,
    StackIdentity,
    TaskReleaseRequest,
    TelemetryField,
    ValidityInterval,
)
from .loads import MissionLoadManager, MissionLoadResult, OnboardMissionConfigurationLoad
from .schemas import assert_truth_free, canonical_json_bytes, from_primitive, to_primitive

STACK_VERSION: Final = "2.0.0"
CONTRACT_MAJOR: Final = 1


def _elapsed_seconds(start: ClockTag, end: ClockTag) -> float:
    if (start.clock_id, start.tick_period_ns, start.scale, start.reset_counter) != (
        end.clock_id,
        end.tick_period_ns,
        end.scale,
        end.reset_counter,
    ):
        raise ValueError("flight-software clocks must share a domain")
    return (end.ticks - start.ticks) * start.tick_period_ns * 1.0e-9


def _add_ticks(tag: ClockTag, ticks: int) -> ClockTag:
    return ClockTag(
        tag.clock_id,
        tag.ticks + ticks,
        tag.tick_period_ns,
        tag.scale,
        tag.validity,
        tag.reset_counter,
    )


def _packet_key(value: PacketId) -> tuple[str, str, int]:
    return value.source_id, value.boot_id, value.sequence


class StackMaturity(str, Enum):
    EXPERIMENTAL = "experimental"
    SUPPORTED = "supported"
    REFERENCE = "reference"
    OEL_UNRATED = "oel_unrated"


class _Lifecycle(str, Enum):
    COLD = "cold"
    BOOTED = "booted"
    SHUTDOWN = "shutdown"


@dataclass(frozen=True, slots=True)
class PassiveStackConfig:
    satellite_id: str
    emit_diagnostics: bool = True
    ideal_navigation: bool = False
    body_frame: FrameId | None = None
    inertial_frame: FrameId | None = None
    measurement_stale_after_s: float = 30.0
    expected_sensor_frames: tuple[tuple[str, FrameId], ...] = ()

    def __post_init__(self) -> None:
        if not self.satellite_id.strip():
            raise ValueError("satellite_id must be non-empty")
        if self.ideal_navigation and (self.body_frame is None or self.inertial_frame is None):
            raise ValueError("ideal_navigation requires body_frame and inertial_frame")
        if not isfinite(float(self.measurement_stale_after_s)) or self.measurement_stale_after_s < 0.0:
            raise ValueError("measurement_stale_after_s must be finite and nonnegative")
        sensor_ids = [sensor_id for sensor_id, _frame in self.expected_sensor_frames]
        if any(not sensor_id.strip() for sensor_id in sensor_ids):
            raise ValueError("expected sensor IDs must be non-empty")
        if len(sensor_ids) != len(set(sensor_ids)):
            raise ValueError("expected sensor IDs must be unique")


@dataclass(frozen=True, slots=True)
class AttitudeReferenceStackConfig:
    satellite_id: str
    body_frame: FrameId
    inertial_frame: FrameId
    allocator: AttitudeAllocatorConfig
    reference: AttitudeReferenceConfig = AttitudeReferenceConfig()
    controller: QuaternionTorqueController | SmallAngleLqrTorqueController = QuaternionTorqueController()
    sensor_mountings: tuple[SensorMounting, ...] = ()
    sensor_calibrations: tuple[SensorCalibration, ...] = ()
    emit_diagnostics: bool = True
    health: HealthManagerConfig = HealthManagerConfig()
    momentum_unload: MomentumUnloadConfig | None = None
    mode_config: AdcsModeConfig = AdcsModeConfig()
    measurement_stale_after_s: float = 30.0

    def __post_init__(self) -> None:
        if not self.satellite_id.strip():
            raise ValueError("satellite_id must be non-empty")
        if self.allocator.satellite_id != self.satellite_id:
            raise ValueError("allocator satellite_id must match stack satellite_id")
        if not isfinite(float(self.measurement_stale_after_s)) or self.measurement_stale_after_s < 0.0:
            raise ValueError("measurement_stale_after_s must be finite and nonnegative")
        sensor_ids = [mounting.sensor_id for mounting in self.sensor_mountings]
        if len(sensor_ids) != len(set(sensor_ids)):
            raise ValueError("sensor mounting IDs must be unique")
        calibration_ids = [calibration.sensor_id for calibration in self.sensor_calibrations]
        if len(calibration_ids) != len(set(calibration_ids)):
            raise ValueError("sensor calibration IDs must be unique")


@dataclass(frozen=True, slots=True)
class TranslationReferenceStackConfig:
    satellite_id: str
    body_frame: FrameId
    inertial_frame: FrameId
    relative_frame: FrameId
    navigation_initialization: NavigationInitializationMode
    control: TranslationControlConfig
    allocator: TranslationAllocatorConfig
    executive: ReferenceExecutiveConfig
    loaded_own_state: LoadedOwnState | None = None
    attitude_allocator: AttitudeAllocatorConfig | None = None
    attitude_controller: QuaternionTorqueController | SmallAngleLqrTorqueController = QuaternionTorqueController()
    attitude_reference: AttitudeReferenceConfig | None = None
    pointing_tolerance_rad: float = 0.08726646259971647
    require_pointing_for_translation: bool = True
    sensor_mountings: tuple[SensorMounting, ...] = ()
    sensor_calibrations: tuple[SensorCalibration, ...] = ()
    enabled_capabilities: tuple[str, ...] = ()
    emit_diagnostics: bool = True
    dry_mass_kg: float = 0.0
    navigation_filter: OrbitFilterKind = OrbitFilterKind.SAMPLE_HOLD
    navigation_alpha: float = 0.85
    navigation_beta: float = 0.05
    navigation_ekf_step_s: float = 1.0
    navigation_process_noise_diag_si: tuple[float, ...] = (1.0e-4, 1.0e-4, 1.0e-4, 1.0e-8, 1.0e-8, 1.0e-8)
    navigation_measurement_noise_diag_si: tuple[float, ...] = (25.0, 25.0, 25.0, 0.01, 0.01, 0.01)
    navigation_initial_covariance_diag_si: tuple[float, ...] = (1.0e4, 1.0e4, 1.0e4, 100.0, 100.0, 100.0)
    navigation_relative_mean_motion_rad_s: float = 0.0011
    navigation_nis_limit: float = 30.0
    health: HealthManagerConfig = HealthManagerConfig()
    resources: ResourceLimits = ResourceLimits()
    conjunction: ConjunctionConfig = ConjunctionConfig()
    autonomous_maneuver: HcwManeuverConfig = HcwManeuverConfig()
    measurement_stale_after_s: float = 30.0

    def __post_init__(self) -> None:
        if not self.satellite_id.strip():
            raise ValueError("satellite_id must be non-empty")
        if self.allocator.satellite_id != self.satellite_id:
            raise ValueError("translation allocator satellite_id must match stack satellite_id")
        if self.attitude_allocator is not None and self.attitude_allocator.satellite_id != self.satellite_id:
            raise ValueError("attitude allocator satellite_id must match stack satellite_id")
        if not isfinite(self.pointing_tolerance_rad) or not 0.0 <= self.pointing_tolerance_rad <= np.pi:
            raise ValueError("pointing_tolerance_rad must be finite and in [0, pi]")
        if not isfinite(float(self.measurement_stale_after_s)) or self.measurement_stale_after_s < 0.0:
            raise ValueError("measurement_stale_after_s must be finite and nonnegative")
        sensor_ids = [mounting.sensor_id for mounting in self.sensor_mountings]
        calibration_ids = [calibration.sensor_id for calibration in self.sensor_calibrations]
        if len(sensor_ids) != len(set(sensor_ids)) or len(calibration_ids) != len(set(calibration_ids)):
            raise ValueError("sensor mounting and calibration IDs must each be unique")
        if self.navigation_initialization is NavigationInitializationMode.LOADED and self.loaded_own_state is None:
            raise ValueError("loaded navigation initialization requires loaded_own_state")
        if not isfinite(self.dry_mass_kg) or self.dry_mass_kg < 0.0:
            raise ValueError("dry_mass_kg must be finite and nonnegative")
        if not isinstance(self.navigation_filter, OrbitFilterKind):
            raise TypeError("navigation_filter must be OrbitFilterKind")


@dataclass(frozen=True, slots=True)
class OrbitReferenceStackConfig(TranslationReferenceStackConfig):
    pass


@dataclass(frozen=True, slots=True)
class RpoReferenceStackConfig(TranslationReferenceStackConfig):
    pass


@dataclass(frozen=True, slots=True)
class LowThrustReferenceStackConfig(TranslationReferenceStackConfig):
    pass


@dataclass(frozen=True, slots=True)
class BuiltinStackDescriptor:
    stack_id: str
    stack_version: str
    maturity: StackMaturity
    summary: str


BUILTIN_STACKS: Final = (
    BuiltinStackDescriptor(
        "fsw.passive",
        STACK_VERSION,
        StackMaturity.EXPERIMENTAL,
        "Records typed inputs and emits no actuator commands.",
    ),
    BuiltinStackDescriptor(
        "fsw.attitude_reference",
        STACK_VERSION,
        StackMaturity.EXPERIMENTAL,
        "Typed attitude navigation, reference generation, control, and allocation.",
    ),
    BuiltinStackDescriptor(
        "fsw.orbit_reference",
        STACK_VERSION,
        StackMaturity.EXPERIMENTAL,
        "Typed own-state navigation, orbit goals, thrust coordination, and allocation.",
    ),
    BuiltinStackDescriptor(
        "fsw.rpo_reference",
        STACK_VERSION,
        StackMaturity.EXPERIMENTAL,
        "Typed relative navigation, RPO executive, guidance, control, and allocation.",
    ),
    BuiltinStackDescriptor(
        "fsw.low_thrust_reference",
        STACK_VERSION,
        StackMaturity.EXPERIMENTAL,
        "Typed low-thrust phasing, element control, and continuous-engine allocation.",
    ),
    BuiltinStackDescriptor(
        "fsw.game_pilot_reference",
        STACK_VERSION,
        StackMaturity.EXPERIMENTAL,
        "Typed pilot and operator inputs with translation, attitude/thrust, and aerodynamic profiles.",
    ),
)


class ReferenceStackBase:
    stack_id: str

    def __init__(self, *, satellite_id: str, identity_material: object) -> None:
        self._satellite_id = satellite_id
        implementation_hash = sha256(
            canonical_json_bytes(
                {
                    "stack_id": self.stack_id,
                    "stack_version": STACK_VERSION,
                    "contract_major": CONTRACT_MAJOR,
                    "implementation_revision": "reference-stack-v1",
                }
            )
        ).hexdigest()
        self._configuration_hash = sha256(canonical_json_bytes(identity_material)).hexdigest()
        self._identity = StackIdentity(self.stack_id, STACK_VERSION, CONTRACT_MAJOR, implementation_hash, True)
        self._lifecycle = _Lifecycle.COLD
        self._boot_id: str | None = None
        self._last_invocation_id = 0
        self._command_sequence = 0

    @property
    def identity(self) -> StackIdentity:
        return self._identity

    def boot(self, event: BootEvent) -> None:
        if self._lifecycle is not _Lifecycle.COLD:
            raise RuntimeError("flight-software stack may be booted exactly once")
        self._require_satellite(event.satellite_id)
        self._boot_id = event.boot_id
        self._lifecycle = _Lifecycle.BOOTED

    def step(self, batch: FlightSoftwareInputBatch) -> FlightSoftwareOutput:
        """Execute through the standalone, fully checked public boundary."""

        assert_truth_free(batch)
        output = self._step_after_boundary_validation(batch)
        assert_truth_free(output)
        return output

    def _step_after_boundary_validation(
        self,
        batch: FlightSoftwareInputBatch,
    ) -> FlightSoftwareOutput:
        """Execute after the owning runtime has checked the input boundary.

        The simulator runtime calls this private hook only for OEL's built-in
        reference stacks, then validates the returned output itself.  Direct
        callers continue to use :meth:`step` and retain both checks.
        """

        self._require_booted()
        self._require_satellite(batch.satellite_id)
        if batch.invocation_id <= self._last_invocation_id:
            raise ValueError("invocation_id must increase monotonically")
        output = self._step(batch)
        if output.satellite_id != batch.satellite_id or output.invocation_id != batch.invocation_id:
            raise RuntimeError("reference stack returned a mismatched output identity")
        self._last_invocation_id = batch.invocation_id
        return output

    def snapshot(self) -> FlightSoftwareSnapshot:
        if self._lifecycle not in {_Lifecycle.BOOTED, _Lifecycle.SHUTDOWN}:
            raise RuntimeError("flight-software stack has not been booted")
        active_load_id, active_load_revision = self._active_load_identity()
        state_bytes = canonical_json_bytes(
            {
                "implementation_hash": self.identity.implementation_hash,
                "configuration_hash": self._configuration_hash,
                "last_invocation_id": self._last_invocation_id,
                "command_sequence": self._command_sequence,
                "stack_state": self._snapshot_stack_state(),
            }
        )
        return FlightSoftwareSnapshot(
            self.stack_id,
            STACK_VERSION,
            self._required_boot_id(),
            self._last_invocation_id,
            active_load_id,
            active_load_revision,
            state_bytes,
            sha256(state_bytes).hexdigest(),
        )

    def restore(self, snapshot: FlightSoftwareSnapshot) -> None:
        self._require_booted()
        if snapshot.stack_id != self.stack_id or snapshot.stack_version != STACK_VERSION:
            raise ValueError("snapshot stack identity is incompatible")
        if snapshot.boot_id != self._required_boot_id():
            raise ValueError("snapshot boot_id does not match the active boot")
        try:
            state = json.loads(snapshot.state_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("snapshot state is not valid canonical JSON") from exc
        if not isinstance(state, dict):
            raise ValueError("snapshot state must be an object")
        if state.get("implementation_hash") != self.identity.implementation_hash:
            raise ValueError("snapshot implementation hash is incompatible")
        if state.get("configuration_hash") != self._configuration_hash:
            raise ValueError("snapshot configuration hash is incompatible")
        invocation = state.get("last_invocation_id")
        sequence = state.get("command_sequence")
        if (
            isinstance(invocation, bool)
            or not isinstance(invocation, int)
            or invocation < 0
            or invocation != snapshot.invocation_id
        ):
            raise ValueError("snapshot invocation state is invalid")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise ValueError("snapshot command sequence is invalid")
        stack_state = state.get("stack_state")
        if not isinstance(stack_state, dict):
            raise ValueError("snapshot stack state is invalid")
        restored_state = self._prepare_restored_stack_state(stack_state)
        if self._restored_active_load_identity(restored_state) != (
            snapshot.active_load_id,
            snapshot.active_load_revision,
        ):
            raise ValueError("snapshot active load identity is inconsistent with stack state")
        self._commit_restored_stack_state(restored_state)
        self._last_invocation_id = invocation
        self._command_sequence = sequence

    def shutdown(self, event: ShutdownEvent) -> None:
        self._require_booted()
        self._require_satellite(event.satellite_id)
        self._lifecycle = _Lifecycle.SHUTDOWN

    def _next_command_id(self) -> PacketId:
        command_id = PacketId(
            f"{self._satellite_id}/{self.stack_id}",
            self._required_boot_id(),
            self._command_sequence,
        )
        self._command_sequence += 1
        return command_id

    def _require_satellite(self, satellite_id: str) -> None:
        if satellite_id != self._satellite_id:
            raise ValueError(f"stack is configured for satellite {self._satellite_id!r}")

    def _require_booted(self) -> None:
        if self._lifecycle is not _Lifecycle.BOOTED:
            raise RuntimeError("flight-software stack is not booted")

    def _required_boot_id(self) -> str:
        if self._boot_id is None:
            raise RuntimeError("flight-software stack has no active boot ID")
        return self._boot_id

    def _step(self, batch: FlightSoftwareInputBatch) -> FlightSoftwareOutput:
        raise NotImplementedError

    def _snapshot_stack_state(self) -> dict[str, object]:
        raise NotImplementedError

    def _prepare_restored_stack_state(self, state: dict[str, object]) -> object:
        raise NotImplementedError

    def _commit_restored_stack_state(self, state: object) -> None:
        raise NotImplementedError

    def _active_load_identity(self) -> tuple[str | None, int | None]:
        return None, None

    def _restored_active_load_identity(self, _state: object) -> tuple[str | None, int | None]:
        return None, None


class PassiveFlightSoftwareStack(ReferenceStackBase):
    stack_id = "fsw.passive"

    def __init__(self, config: PassiveStackConfig) -> None:
        super().__init__(satellite_id=config.satellite_id, identity_material=config)
        self.config = config
        self._event_count = 0
        self._measurement_count = 0
        self._missing_measurement_batch_count = 0
        self._duplicate_packet_count = 0
        self._out_of_order_packet_count = 0
        self._stale_measurement_count = 0
        self._invalid_measurement_count = 0
        self._suspect_measurement_count = 0
        self._invalid_frame_count = 0
        self._seen_packets: set[PacketId] = set()
        self._last_sequence_by_source: dict[tuple[str, str], int] = {}
        self._expected_sensor_frames = dict(config.expected_sensor_frames)
        self._navigator = (
            AttitudeNavigator(body_frame=config.body_frame, inertial_frame=config.inertial_frame)
            if config.ideal_navigation and config.body_frame is not None and config.inertial_frame is not None
            else None
        )

    def _step(self, batch: FlightSoftwareInputBatch) -> FlightSoftwareOutput:
        self._event_count += len(batch.events)
        navigation_events: list[InputEvent] = []
        batch_measurements = 0
        for event in batch.events:
            is_measurement = event.kind is InputKind.MEASUREMENT and isinstance(event.payload, MeasurementEvent)
            batch_measurements += int(is_measurement)
            duplicate = event.packet_id in self._seen_packets
            if duplicate:
                self._duplicate_packet_count += 1
                continue
            self._seen_packets.add(event.packet_id)
            source_key = (event.packet_id.source_id, event.packet_id.boot_id)
            previous_sequence = self._last_sequence_by_source.get(source_key)
            out_of_order = previous_sequence is not None and event.packet_id.sequence < previous_sequence
            if out_of_order:
                self._out_of_order_packet_count += 1
            self._last_sequence_by_source[source_key] = max(
                event.packet_id.sequence,
                previous_sequence if previous_sequence is not None else event.packet_id.sequence,
            )
            if not is_measurement:
                navigation_events.append(event)
                continue
            self._measurement_count += 1
            measurement = event.payload
            assert isinstance(measurement, MeasurementEvent)
            invalid = event.quality.validity is DataValidity.INVALID
            suspect = event.quality.validity is DataValidity.SUSPECT
            self._invalid_measurement_count += int(invalid)
            self._suspect_measurement_count += int(suspect)
            expected_frame = self._expected_sensor_frames.get(measurement.sensor_id)
            invalid_frame = expected_frame is not None and measurement.frame != expected_frame
            self._invalid_frame_count += int(invalid_frame)
            try:
                age_s = _elapsed_seconds(measurement.sample_time, batch.invocation_time)
                stale = age_s < 0.0 or age_s > self.config.measurement_stale_after_s
            except ValueError:
                stale = True
            self._stale_measurement_count += int(stale)
            if not (out_of_order or invalid or invalid_frame or stale):
                navigation_events.append(event)
        if batch_measurements == 0:
            self._missing_measurement_batch_count += 1
        solution = None
        if self._navigator is not None:
            self._navigator.ingest(tuple(navigation_events))
            solution = self._navigator.solution(batch.invocation_time)
        fields = [
            TelemetryField("events_received_total", self._event_count),
            TelemetryField("events_received_batch", len(batch.events)),
            TelemetryField("measurements_received_total", self._measurement_count),
            TelemetryField("measurement_batches_missing_total", self._missing_measurement_batch_count),
            TelemetryField("duplicate_packets_total", self._duplicate_packet_count),
            TelemetryField("out_of_order_packets_total", self._out_of_order_packet_count),
            TelemetryField("stale_measurements_total", self._stale_measurement_count),
            TelemetryField("invalid_measurements_total", self._invalid_measurement_count),
            TelemetryField("suspect_measurements_total", self._suspect_measurement_count),
            TelemetryField("invalid_frame_measurements_total", self._invalid_frame_count),
        ]
        if solution is not None:
            fields.append(TelemetryField("navigation_valid", solution.valid_for_control))
        telemetry = (
            (
                DiagnosticTelemetry(
                    "fsw.passive.status",
                    batch.invocation_time,
                    tuple(fields),
                ),
            )
            if self.config.emit_diagnostics
            else ()
        )
        return FlightSoftwareOutput(batch.satellite_id, batch.invocation_id, telemetry=telemetry)

    def _snapshot_stack_state(self) -> dict[str, object]:
        return {
            "event_count": self._event_count,
            "measurement_count": self._measurement_count,
            "missing_measurement_batch_count": self._missing_measurement_batch_count,
            "duplicate_packet_count": self._duplicate_packet_count,
            "out_of_order_packet_count": self._out_of_order_packet_count,
            "stale_measurement_count": self._stale_measurement_count,
            "invalid_measurement_count": self._invalid_measurement_count,
            "suspect_measurement_count": self._suspect_measurement_count,
            "invalid_frame_count": self._invalid_frame_count,
            "seen_packets": [
                {"source_id": packet.source_id, "boot_id": packet.boot_id, "sequence": packet.sequence}
                for packet in sorted(self._seen_packets, key=_packet_key)
            ],
            "last_sequence_by_source": [
                {"source_id": source_id, "boot_id": boot_id, "sequence": sequence}
                for (source_id, boot_id), sequence in sorted(self._last_sequence_by_source.items())
            ],
            "navigation": None if self._navigator is None else self._navigator.snapshot_state(),
        }

    def _prepare_restored_stack_state(self, state: dict[str, object]) -> object:
        counter_names = (
            "event_count",
            "measurement_count",
            "missing_measurement_batch_count",
            "duplicate_packet_count",
            "out_of_order_packet_count",
            "stale_measurement_count",
            "invalid_measurement_count",
            "suspect_measurement_count",
            "invalid_frame_count",
        )
        counters: dict[str, int] = {}
        for name in counter_names:
            value = state.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"passive snapshot {name} is invalid")
            counters[name] = value
        seen_packets = {
            PacketId(str(item["source_id"]), str(item["boot_id"]), int(item["sequence"]))
            for item in list(state.get("seen_packets", []))
            if isinstance(item, dict)
        }
        last_sequences = {
            (str(item["source_id"]), str(item["boot_id"])): int(item["sequence"])
            for item in list(state.get("last_sequence_by_source", []))
            if isinstance(item, dict)
        }
        navigation = state.get("navigation")
        if self._navigator is None:
            if navigation is not None:
                raise ValueError("passive snapshot unexpectedly contains navigation state")
            return (counters, seen_packets, last_sequences, None)
        if not isinstance(navigation, dict):
            raise ValueError("passive snapshot navigation state is invalid")
        if self.config.body_frame is None or self.config.inertial_frame is None:
            raise ValueError("passive navigation configuration is incomplete")
        navigator = AttitudeNavigator(body_frame=self.config.body_frame, inertial_frame=self.config.inertial_frame)
        navigator.restore_state(navigation)
        return (counters, seen_packets, last_sequences, navigator)

    def _commit_restored_stack_state(self, state: object) -> None:
        if not isinstance(state, tuple) or len(state) != 4:
            raise TypeError("restored passive state is invalid")
        counters = state[0]
        if not isinstance(counters, dict):
            raise TypeError("restored passive counters are invalid")
        for name, value in counters.items():
            setattr(self, f"_{name}", int(value))
        seen_packets, last_sequences, navigator = state[1:]
        if not isinstance(seen_packets, set) or not isinstance(last_sequences, dict):
            raise TypeError("restored passive packet history is invalid")
        self._seen_packets = seen_packets
        self._last_sequence_by_source = last_sequences
        if navigator is not None and not isinstance(navigator, AttitudeNavigator):
            raise TypeError("restored passive navigation state is invalid")
        self._navigator = navigator


class AttitudeReferenceFlightSoftwareStack(ReferenceStackBase):
    stack_id = "fsw.attitude_reference"

    def __init__(self, config: AttitudeReferenceStackConfig) -> None:
        super().__init__(satellite_id=config.satellite_id, identity_material=config)
        self.config = config
        self._navigator = AttitudeNavigator(
            body_frame=config.body_frame,
            inertial_frame=config.inertial_frame,
            mountings=config.sensor_mountings,
            calibrations=config.sensor_calibrations,
        )
        self._reference = AttitudeReferenceGenerator(config.reference, inertial_frame=config.inertial_frame)
        self._controller = config.controller
        self._allocator = AttitudeAllocator(config.allocator)
        self._health = StackHealthManager(config.health)
        self._adcs = AdcsModeManager(config.momentum_unload, config.mode_config)
        self._commands = OnboardCommandService()
        self._target_position_eci_m = config.reference.target_position_eci_m
        self._target_update_count = 0
        self._target_update_rejection_count = 0

    def _step(self, batch: FlightSoftwareInputBatch) -> FlightSoftwareOutput:
        self._ingest_reference_commands(batch)
        self._navigator.ingest(
            batch.events,
            generated_at=batch.invocation_time,
            stale_after_s=self.config.measurement_stale_after_s,
        )
        solution = self._navigator.solution(batch.invocation_time)
        health = self._health.update(batch.invocation_time, batch.events)
        rate_norm = (
            None
            if solution.angular_rate_body_rad_s is None
            else float(np.linalg.norm(solution.angular_rate_body_rad_s))
        )
        reference_available = self._reference_available(solution, health.isolated_components)
        adcs_mode = self._adcs.update(
            batch.events,
            angular_rate_norm_rad_s=rate_norm,
            navigation_valid=solution.valid_for_control,
            actuator_fault=self.config.allocator.actuator_id in health.isolated_components,
            reference_available=reference_available,
        )
        reference = (
            None
            if adcs_mode is AdcsOperationalMode.COARSE_SUN or not reference_available
            else self._reference.generate(solution)
        )
        allocation = None
        effort = None
        if adcs_mode is AdcsOperationalMode.DETUMBLE:
            effort = self._detumble_effort(solution)
        elif adcs_mode is AdcsOperationalMode.COARSE_SUN:
            effort = self._coarse_sun_effort(solution)
        elif adcs_mode is AdcsOperationalMode.DEGRADED and not reference_available:
            effort = self._detumble_effort(solution)
        elif reference is not None:
            effort = self._controller.control(solution, reference)
        if effort is not None:
            allocation = self._allocator.allocate(
                effort,
                solution,
                command_id=self._next_command_id(),
            )
        commands = list(() if allocation is None else allocation.proposed_commands)
        if self.config.allocator.actuator_id in health.isolated_components:
            commands = []
            fallback_id = health.selected_actuator(self.config.allocator.actuator_id)
            if (
                fallback_id is not None
                and fallback_id != self.config.allocator.actuator_id
                and fallback_id not in health.isolated_components
                and effort is not None
                and solution.magnetic_field_body_t is not None
                and self.config.momentum_unload is not None
            ):
                fallback = AttitudeAllocator(
                    AttitudeAllocatorConfig(
                        batch.satellite_id,
                        AttitudeAllocatorKind.MAGNETORQUER,
                        fallback_id,
                        FrameId(f"OEL/ACTUATOR/{batch.satellite_id}/{fallback_id}", "frames-v1"),
                        limits=(self.config.momentum_unload.max_dipole_a_m2,),
                    )
                )
                allocation = fallback.allocate(effort, solution, command_id=self._next_command_id())
                commands.extend(allocation.proposed_commands)
        dipole = self._adcs.unload_dipole(solution.magnetic_field_body_t)
        if dipole is not None and self.config.momentum_unload is not None:
            validity = ValidityInterval(
                batch.invocation_time,
                _add_ticks(batch.invocation_time, self.config.momentum_unload.command_validity_ticks),
            )
            commands.append(
                ActuatorCommand(
                    self._next_command_id(),
                    batch.satellite_id,
                    self.config.momentum_unload.torquer_actuator_id,
                    batch.invocation_time,
                    validity,
                    FrameId(
                        f"OEL/ACTUATOR/{batch.satellite_id}/{self.config.momentum_unload.torquer_actuator_id}",
                        "frames-v1",
                    ),
                    MagnetorquerDipoleCommand(dipole),
                )
            )
        commands = [command for command in commands if command.actuator_id not in health.isolated_components]
        telemetry = ()
        if self.config.emit_diagnostics:
            fields = [
                TelemetryField("navigation_valid", solution.valid_for_control),
                TelemetryField("active_fault_count", len(solution.active_faults)),
                TelemetryField("command_count", len(commands)),
                TelemetryField("health_state", health.state.value),
                TelemetryField("isolated_component_count", len(health.isolated_components)),
                TelemetryField("adcs_operational_mode", adcs_mode.value),
                TelemetryField("wheel_momentum_fraction", self._adcs.momentum_fraction),
                TelemetryField("reference_available", reference_available),
                TelemetryField("target_update_count", self._target_update_count),
                TelemetryField("target_update_rejection_count", self._target_update_rejection_count),
            ]
            if reference is not None:
                fields.append(TelemetryField("reference_id", reference.reference_id))
                if reference.attitude_quat_from_frame is not None and solution.attitude_quat_bn is not None:
                    for component, value in zip(
                        ("w", "x", "y", "z"),
                        reference.attitude_quat_from_frame,
                        strict=True,
                    ):
                        fields.append(
                            TelemetryField(f"desired_attitude_quat_{component}", float(value))
                        )
                    alignment = abs(
                        float(
                            np.dot(
                                np.asarray(reference.attitude_quat_from_frame, dtype=float),
                                np.asarray(solution.attitude_quat_bn, dtype=float),
                            )
                        )
                    )
                    fields.append(
                        TelemetryField(
                            "attitude_error_rad",
                            2.0 * acos(float(np.clip(alignment, -1.0, 1.0))),
                            "rad",
                        )
                    )
            if solution.angular_rate_body_rad_s is not None:
                fields.append(
                    TelemetryField(
                        "angular_rate_norm_rad_s",
                        float(np.linalg.norm(solution.angular_rate_body_rad_s)),
                        "rad/s",
                    )
                )
            if allocation is not None:
                fields.append(TelemetryField("allocation_status", allocation.status.value))
            telemetry = (DiagnosticTelemetry("fsw.attitude_reference.status", batch.invocation_time, tuple(fields)),)
        return FlightSoftwareOutput(batch.satellite_id, batch.invocation_id, tuple(commands), telemetry)

    def _detumble_effort(self, solution) -> RequestedEffort | None:
        if solution.angular_rate_body_rad_s is None:
            return None
        omega = np.asarray(solution.angular_rate_body_rad_s, dtype=float)
        if isinstance(self._controller, QuaternionTorqueController):
            torque = -np.asarray(self._controller.kd, dtype=float) * omega
            limit = self._controller.max_torque_n_m
        else:
            gain = np.asarray(self._controller.gain, dtype=float)
            torque = -(gain[:, 3:] @ omega)
            limit = self._controller.max_torque_n_m
        norm = float(np.linalg.norm(torque))
        if norm > limit > 0.0:
            torque *= limit / norm
        return RequestedEffort(
            "attitude-detumble",
            RequestedEffortKind.TORQUE,
            solution.generated_at,
            solution.frame,
            ValidityInterval(
                solution.generated_at,
                _add_ticks(solution.generated_at, self.config.reference.validity_ticks),
            ),
            torque_n_m=tuple(float(value) for value in torque),
        )

    def _coarse_sun_effort(self, solution) -> RequestedEffort | None:
        """Acquire the Sun from body-frame vector and optional gyro data.

        Unlike fine inertial pointing, this recovery law deliberately does not
        require an attitude quaternion.  That makes ``coarse_sun`` an actual
        degraded-navigation mode rather than a label around the nominal law.
        """

        if solution.sun_vector_body is None:
            return None
        sun = np.asarray(solution.sun_vector_body, dtype=float)
        sun_norm = float(np.linalg.norm(sun))
        boresight = np.asarray(self.config.reference.boresight_body, dtype=float)
        boresight_norm = float(np.linalg.norm(boresight))
        if sun_norm <= 0.0 or boresight_norm <= 0.0:
            return None
        alignment_error = np.cross(boresight / boresight_norm, sun / sun_norm)
        omega = (
            np.zeros(3)
            if solution.angular_rate_body_rad_s is None
            else np.asarray(solution.angular_rate_body_rad_s, dtype=float)
        )
        if isinstance(self._controller, QuaternionTorqueController):
            torque = np.asarray(self._controller.kp, dtype=float) * alignment_error
            torque -= np.asarray(self._controller.kd, dtype=float) * omega
            limit = self._controller.max_torque_n_m
        else:
            gain = np.asarray(self._controller.gain, dtype=float)
            torque = gain[:, :3] @ alignment_error - gain[:, 3:] @ omega
            limit = self._controller.max_torque_n_m
        norm = float(np.linalg.norm(torque))
        if norm > limit > 0.0:
            torque *= limit / norm
        return RequestedEffort(
            "attitude-coarse-sun",
            RequestedEffortKind.TORQUE,
            solution.generated_at,
            solution.frame,
            ValidityInterval(
                solution.generated_at,
                _add_ticks(solution.generated_at, self.config.reference.validity_ticks),
            ),
            torque_n_m=tuple(float(value) for value in torque),
        )

    def _reference_available(self, solution, isolated_components: tuple[str, ...]) -> bool:
        """Evaluate only onboard knowledge and explicit FDIR indications."""

        if self.config.reference.mode is AttitudeReferenceMode.SUN:
            return solution.sun_vector_body is not None and not any(
                component in isolated_components for component in ("sun", "sun_sensor", "ideal_sun")
            )
        if self.config.reference.mode is AttitudeReferenceMode.TARGET:
            return not any(
                component in isolated_components for component in ("target", "target_sensor", "target_tracker")
            )
        return True

    def _ingest_reference_commands(self, batch: FlightSoftwareInputBatch) -> None:
        for command in self._commands.ingest(batch.invocation_time, batch.events):
            if command.kind is not GroundCommandKind.STACK_COMMAND:
                self._target_update_rejection_count += 1
                continue
            parameters = {field.name: field.value for field in command.parameters}
            if parameters.get("operation") != "set_target_eci" or self.config.reference.mode is not AttitudeReferenceMode.TARGET:
                self._target_update_rejection_count += 1
                continue
            try:
                target = tuple(float(parameters[f"target_{axis}_eci_m"]) for axis in "xyz")
            except (KeyError, TypeError, ValueError):
                self._target_update_rejection_count += 1
                continue
            if not np.all(np.isfinite(target)):
                self._target_update_rejection_count += 1
                continue
            self._target_position_eci_m = target
            self._reference = AttitudeReferenceGenerator(
                replace(self.config.reference, target_position_eci_m=target),
                inertial_frame=self.config.inertial_frame,
            )
            self._target_update_count += 1

    def _snapshot_stack_state(self) -> dict[str, object]:
        return {
            "navigation": self._navigator.snapshot_state(),
            "health": self._health.snapshot_state(),
            "adcs": self._adcs.snapshot_state(),
            "commands": self._commands.snapshot_state(),
            "target_position_eci_m": self._target_position_eci_m,
            "target_update_count": self._target_update_count,
            "target_update_rejection_count": self._target_update_rejection_count,
        }

    def _prepare_restored_stack_state(self, state: dict[str, object]) -> object:
        navigation = state.get("navigation")
        if not isinstance(navigation, dict):
            raise ValueError("attitude snapshot navigation state is invalid")
        navigator = AttitudeNavigator(
            body_frame=self.config.body_frame,
            inertial_frame=self.config.inertial_frame,
            mountings=self.config.sensor_mountings,
            calibrations=self.config.sensor_calibrations,
        )
        navigator.restore_state(navigation)
        health_state = state.get("health", {})
        adcs_state = state.get("adcs", {"mode": "nominal", "wheel_momentum": None})
        command_state = state.get("commands", {})
        if not isinstance(health_state, dict) or not isinstance(adcs_state, dict) or not isinstance(command_state, dict):
            raise ValueError("attitude operational snapshot state is invalid")
        health = StackHealthManager(self.config.health)
        health.restore_state(health_state)
        adcs = AdcsModeManager(self.config.momentum_unload, self.config.mode_config)
        adcs.restore_state(adcs_state)
        commands = OnboardCommandService()
        commands.restore_state(command_state)
        target = state.get("target_position_eci_m", self.config.reference.target_position_eci_m)
        target_position = None if target is None else tuple(float(value) for value in target)
        if target_position is not None and (len(target_position) != 3 or not np.all(np.isfinite(target_position))):
            raise ValueError("attitude snapshot target position is invalid")
        counts = (
            int(state.get("target_update_count", 0)),
            int(state.get("target_update_rejection_count", 0)),
        )
        if any(value < 0 for value in counts):
            raise ValueError("attitude snapshot target-update counters are invalid")
        return navigator, health, adcs, commands, target_position, counts

    def _commit_restored_stack_state(self, state: object) -> None:
        if not isinstance(state, tuple) or len(state) != 6 or not isinstance(state[0], AttitudeNavigator):
            raise TypeError("restored attitude state is invalid")
        self._navigator = state[0]
        self._health = state[1]
        self._adcs = state[2]
        self._commands = state[3]
        self._target_position_eci_m = state[4]
        self._target_update_count, self._target_update_rejection_count = state[5]
        if self._target_position_eci_m is not None:
            self._reference = AttitudeReferenceGenerator(
                replace(self.config.reference, target_position_eci_m=self._target_position_eci_m),
                inertial_frame=self.config.inertial_frame,
            )


@dataclass(slots=True)
class _RestoredTranslationState:
    navigator: OrbitNavigator
    controller: TranslationController
    executive: ReferenceMissionExecutive
    active_load: OnboardMissionConfigurationLoad | None
    health: StackHealthManager
    resources: ResourceMonitor
    commands: OnboardCommandService
    conjunction: ConjunctionAvoidancePlanner
    executed_plans: set[str]
    maneuver_plan: ManeuverPlan | None
    pending_plan_commands: dict[str, set[PacketId]]
    accepted_plan_commands: dict[str, set[PacketId]]
    plan_failures: dict[str, str]
    scheduled_burn_pending: set[PacketId]
    scheduled_burn_had_accepted: bool
    scheduled_burn_failed: bool


class _TranslationReferenceFlightSoftwareStack(ReferenceStackBase):
    config: TranslationReferenceStackConfig
    supported_primary_modes: frozenset[TranslationMode]

    def __init__(
        self,
        config: TranslationReferenceStackConfig,
        *,
        _live_navigation_fast_path: bool = False,
    ) -> None:
        super().__init__(satellite_id=config.satellite_id, identity_material=config)
        self.config = config
        self._live_navigation_fast_path = bool(_live_navigation_fast_path)
        primary_mode = TranslationMode(config.executive.primary_mode)
        if primary_mode not in self.supported_primary_modes:
            raise ValueError(f"{self.stack_id} does not advertise primary mode {primary_mode.value!r}")
        for action in config.executive.actions:
            action_mode = TranslationMode(action.mode)
            if action_mode not in self.supported_primary_modes:
                raise ValueError(f"{self.stack_id} does not advertise action mode {action_mode.value!r}")
        TranslationMode(config.executive.recovery_mode)
        self._navigator = self._new_navigator()
        self._controller = TranslationController(config.control)
        self._allocator = TranslationAllocator(config.allocator)
        self._executive_config = config.executive
        self._executive = ReferenceMissionExecutive(self._executive_config)
        self._attitude_allocator = (
            None if config.attitude_allocator is None else AttitudeAllocator(config.attitude_allocator)
        )
        self._attitude_reference = (
            None
            if config.attitude_reference is None
            else AttitudeReferenceGenerator(config.attitude_reference, inertial_frame=config.inertial_frame)
        )
        self._mission_manager = self._new_mission_manager()
        self._active_load: OnboardMissionConfigurationLoad | None = None
        self._health = StackHealthManager(config.health)
        self._resources = ResourceMonitor(config.resources)
        self._command_service = OnboardCommandService()
        self._conjunction = ConjunctionAvoidancePlanner(config.conjunction)
        self._maneuver_planner = HcwManeuverPlanner(config.autonomous_maneuver)
        self._maneuver_plan = None
        self._executed_plans: set[str] = set()
        self._pending_plan_commands: dict[str, set[PacketId]] = {}
        self._accepted_plan_commands: dict[str, set[PacketId]] = {}
        self._plan_failures: dict[str, str] = {}
        self._scheduled_burn_pending: set[PacketId] = set()
        self._scheduled_burn_had_accepted = False
        self._scheduled_burn_failed = False

    def _step(self, batch: FlightSoftwareInputBatch) -> FlightSoftwareOutput:
        self._update_maneuver_receipts(batch.events)
        load_results = self._ingest_mission_loads(batch)
        navigation_events = tuple(
            event
            for event in batch.events
            if not (event.kind is InputKind.MEASUREMENT and isinstance(event.payload, MeasurementEvent))
            or self._measurement_is_fresh(event.payload, batch.invocation_time)
        )
        self._navigator.ingest(navigation_events)
        solution = (
            self._navigator.control_solution(batch.invocation_time)
            if self._live_navigation_fast_path
            else self._navigator.solution(batch.invocation_time)
        )
        solution = self._fresh_navigation_solution(solution, batch.invocation_time)
        health = self._health.update(batch.invocation_time, batch.events)
        resources = self._resources.update(
            batch.events,
            mass_kg=solution.mass_kg,
            dry_mass_kg=self.config.dry_mass_kg,
        )
        due_commands = self._command_service.ingest(batch.invocation_time, batch.events)
        track = solution.relative_track(self.config.control.target_id)
        primary_mode = TranslationMode(self._executive_config.primary_mode)
        goal_satisfied = self._controller.assess_goal(solution, primary_mode)
        if primary_mode is TranslationMode.SCHEDULED_BURN and self.config.control.scheduled_burns:
            goal_satisfied = bool(
                goal_satisfied
                and self._scheduled_burn_had_accepted
                and not self._scheduled_burn_pending
                and not self._scheduled_burn_failed
            )
        action_conditions: set[str] = set()
        for command in due_commands:
            if command.kind is GroundCommandKind.ACTION_REQUEST:
                action_conditions.add(command.command_id)
                parameters = {field.name: field.value for field in command.parameters}
                for name in ("condition_id", "action_id"):
                    value = parameters.get(name)
                    if isinstance(value, str) and value.strip():
                        action_conditions.add(value)
        if goal_satisfied:
            action_conditions.add("goal_satisfied")
        observation = ExecutiveObservation(
            navigation_ready=self._navigation_ready(solution, primary_mode),
            goal_satisfied=goal_satisfied,
            relative_range_m=None if track is None else track.range_m,
            relative_rate_m_s=None if track is None else track.range_rate_m_s,
            active_faults=tuple(
                sorted(
                    {
                        **dict(solution.active_faults),
                        **dict(health.active_faults),
                        **{f"resource.{name}": name for name in resources.violations},
                    }.items()
                )
            ),
            action_conditions=tuple(sorted(action_conditions)),
            mass_kg=solution.mass_kg,
            dry_mass_kg=self.config.dry_mass_kg,
        )
        faulted_components = set(health.isolated_components) | {
            component for component, _code in solution.active_faults
        }
        executive = self._executive.update(batch.invocation_time, observation)
        control = None
        allocation = None
        pointing_allocation = None
        pointing_error_rad: float | None = None
        pointing_compliant = self._attitude_allocator is None or not self.config.require_pointing_for_translation
        commands = []
        mean_motion = self._relative_mean_motion(solution, track)
        conjunction_plan = self._conjunction.assess(
            batch.invocation_time,
            solution,
            mean_motion,
            completed_plan_ids=frozenset(self._executed_plans),
        )
        if self._maneuver_plan is None and track is not None:
            self._maneuver_plan = self._maneuver_planner.plan(batch.invocation_time, track, mean_motion)
        active_plan = conjunction_plan or self._maneuver_plan
        plan_due = (
            active_plan is not None
            and active_plan.plan_id not in self._executed_plans
            and active_plan.plan_id not in self._pending_plan_commands
            and _elapsed_seconds(active_plan.execute_at, batch.invocation_time) >= 0.0
        )
        if plan_due and resources.command_allowed:
            effort = self._maneuver_effort(active_plan, solution, track)
            if effort is not None:
                allocation = self._allocator.allocate(
                    effort,
                    solution,
                    next_command_id=self._next_command_id,
                    unavailable_actuators=frozenset(faulted_components),
                )
                commands.extend(allocation.proposed_commands)
                if (
                    allocation.status not in {AllocationStatus.INFEASIBLE, AllocationStatus.INVALID}
                    and allocation.proposed_commands
                ):
                    command_ids = {command.command_id for command in allocation.proposed_commands}
                    self._pending_plan_commands[active_plan.plan_id] = command_ids
                    self._accepted_plan_commands[active_plan.plan_id] = set()
                    self._plan_failures.pop(active_plan.plan_id, None)
                else:
                    self._plan_failures[active_plan.plan_id] = allocation.status.value
        elif (
            (active_plan is None or active_plan.plan_id in self._executed_plans)
            and conjunction_plan is None
            and executive.allow_command
            and executive.selected_mode is not None
            and resources.command_allowed
        ):
            try:
                control = self._controller.control(solution, executive.selected_mode)
            except ValueError:
                control = None
            if control is not None and control.effort is not None:
                if (
                    control.pointing_direction_eci is not None
                    and self._attitude_allocator is not None
                    and self.config.require_pointing_for_translation
                ):
                    pointing_error_rad, pointing_allocation = self._point_thrust_axis(
                        solution,
                        control.pointing_direction_eci,
                    )
                    pointing_compliant = (
                        pointing_error_rad is not None and pointing_error_rad <= self.config.pointing_tolerance_rad
                    )
                    if pointing_allocation is not None:
                        commands.extend(pointing_allocation.proposed_commands)
                if pointing_compliant:
                    allocation = self._allocator.allocate(
                        control.effort,
                        solution,
                        next_command_id=self._next_command_id,
                        unavailable_actuators=frozenset(faulted_components),
                    )
                    commands.extend(allocation.proposed_commands)
        if (
            pointing_allocation is None
            and self._attitude_allocator is not None
            and self._attitude_reference is not None
        ):
            reference = self._attitude_reference.generate(solution.attitude)
            if reference is not None:
                effort = self.config.attitude_controller.control(solution.attitude, reference)
                if effort is not None:
                    pointing_allocation = self._attitude_allocator.allocate(
                        effort,
                        solution.attitude,
                        command_id=self._next_command_id(),
                    )
                    commands.extend(pointing_allocation.proposed_commands)
        commands = [command for command in commands if command.actuator_id not in faulted_components]
        if (
            primary_mode is TranslationMode.SCHEDULED_BURN
            and control is not None
            and control.phase == "finite_burn"
            and allocation is not None
        ):
            emitted_ids = {
                command.command_id
                for command in commands
                if command.command_id in {item.command_id for item in allocation.proposed_commands}
            }
            if emitted_ids:
                self._scheduled_burn_pending.update(emitted_ids)
                self._scheduled_burn_failed = False
        telemetry = ()
        if self.config.emit_diagnostics:
            fields = [
                TelemetryField("navigation_ready", observation.navigation_ready),
                TelemetryField("relative_navigation_valid", track is not None),
                TelemetryField("goal_id", executive.progress.goal_id),
                TelemetryField("goal_type", self._executive_config.primary_goal.goal_type),
                TelemetryField("goal_state", executive.progress.state.value),
                TelemetryField("executive_phase", executive.progress.phase.value),
                TelemetryField("selected_mode", executive.selected_mode),
                TelemetryField("control_law", self.config.control.control_law.value),
                TelemetryField("command_count", len(commands)),
                TelemetryField("health_state", health.state.value),
                TelemetryField("isolated_component_count", len(health.isolated_components)),
                TelemetryField("stored_command_count", self._command_service.pending_count),
                TelemetryField("resource_command_allowed", resources.command_allowed),
                TelemetryField(
                    "constraint_violation_count",
                    sum(item.satisfied is False for item in executive.constraints.constraints),
                ),
                TelemetryField(
                    "safety_review_violation_count",
                    sum(
                        item.satisfied is False and item.kind.value == "mission_safety_envelope"
                        for item in executive.constraints.constraints
                    ),
                ),
                TelemetryField("dwell_elapsed_s", executive.progress.dwell_elapsed_s, "s"),
                TelemetryField("maintenance_compliant_s", executive.progress.compliant_elapsed_s, "s"),
                TelemetryField("maintenance_excursion_s", executive.progress.excursion_elapsed_s, "s"),
                TelemetryField("scheduled_burn_pending_receipts", len(self._scheduled_burn_pending)),
                TelemetryField("scheduled_burn_receipt_confirmed", self._scheduled_burn_had_accepted),
                TelemetryField("scheduled_burn_receipt_failed", self._scheduled_burn_failed),
            ]
            if self.stack_id == "fsw.low_thrust_reference":
                fields.extend(
                    (
                        TelemetryField("low_thrust_window_open", self._controller.thrust_window_open),
                        TelemetryField(
                            "low_thrust_missed_window_count",
                            self._controller.missed_thrust_window_count,
                        ),
                        TelemetryField(
                            "element_averaging_sample_count",
                            self._controller.element_averaging_sample_count,
                        ),
                    )
                )
            if control is not None:
                fields.extend(
                    (
                        TelemetryField("control_phase", control.phase),
                        TelemetryField("control_saturated", control.saturated),
                        TelemetryField("position_error_m", control.position_error_m, "m"),
                        TelemetryField("velocity_error_m_s", control.velocity_error_m_s, "m/s"),
                    )
                )
                if control.effort is not None and control.effort.force_n is not None:
                    fields.append(
                        TelemetryField("requested_force_n", float(np.linalg.norm(control.effort.force_n)), "N")
                    )
                if control.phase == "finite_burn":
                    now_ns = clock_tag_elapsed_ns(batch.invocation_time)
                    active_burn = next(
                        (
                            burn
                            for burn in self.config.control.scheduled_burns
                            if burn.start_time_ns <= now_ns < burn.start_time_ns + burn.duration_ns
                        ),
                        None,
                    )
                    if active_burn is not None:
                        original_accel = float(np.linalg.norm(active_burn.acceleration_m_s2))
                        command_mass = (
                            float(solution.mass_kg)
                            if solution.mass_kg is not None
                            else float(self.config.control.assumed_mass_kg)
                        )
                        fields.extend(
                            (
                                TelemetryField(
                                    "scheduled_burn_original_accel_m_s2", original_accel, "m/s^2"
                                ),
                                TelemetryField(
                                    "scheduled_burn_original_force_n", original_accel * command_mass, "N"
                                ),
                                TelemetryField(
                                    "scheduled_burn_original_delta_v_m_s",
                                    original_accel * active_burn.duration_ns * 1.0e-9,
                                    "m/s",
                                ),
                                TelemetryField(
                                    "scheduled_burn_duration_s",
                                    active_burn.duration_ns * 1.0e-9,
                                    "s",
                                ),
                                TelemetryField("scheduled_burn_controller_clipped", control.saturated),
                            )
                        )
            if allocation is not None:
                fields.append(TelemetryField("translation_allocation_status", allocation.status.value))
                fields.extend(allocation.status_details)
            if pointing_allocation is not None:
                fields.append(TelemetryField("attitude_allocation_status", pointing_allocation.status.value))
            if pointing_error_rad is not None:
                fields.extend(
                    (
                        TelemetryField("pointing_error_rad", pointing_error_rad, "rad"),
                        TelemetryField("pointing_compliant", pointing_compliant),
                    )
                )
            for result in load_results:
                fields.extend(
                    (
                        TelemetryField("mission_load_id", result.load_id),
                        TelemetryField("mission_load_revision", result.revision),
                        TelemetryField("mission_load_disposition", result.disposition.value),
                    )
                )
            fields.extend(_actuator_review_fields(batch))
            fields.extend(resource_telemetry(resources))
            if solution.belief.own_state is not None:
                fields.extend(field for field in solution.belief.own_state.values if "ekf_" in field.name)
            if active_plan is not None:
                fields.extend(
                    (
                        TelemetryField("maneuver_plan_id", active_plan.plan_id),
                        TelemetryField("maneuver_plan_reason", active_plan.reason),
                        TelemetryField("maneuver_plan_target_id", active_plan.target_id),
                        TelemetryField(
                            "maneuver_plan_predicted_miss_distance_m", active_plan.predicted_miss_distance_m, "m"
                        ),
                        TelemetryField(
                            "maneuver_plan_time_to_closest_approach_s", active_plan.time_to_closest_approach_s, "s"
                        ),
                        TelemetryField("maneuver_plan_executed", active_plan.plan_id in self._executed_plans),
                        TelemetryField(
                            "maneuver_plan_state",
                            "executed"
                            if active_plan.plan_id in self._executed_plans
                            else "awaiting_receipt"
                            if active_plan.plan_id in self._pending_plan_commands
                            else "retrying"
                            if active_plan.plan_id in self._plan_failures
                            else "planned",
                        ),
                    )
                )
            telemetry = (DiagnosticTelemetry(f"{self.stack_id}.status", batch.invocation_time, tuple(fields)),)
        next_command_release = self._command_service.next_release_at
        requested_releases = (
            ()
            if next_command_release is None
            else (TaskReleaseRequest("stored_ground_command", next_command_release),)
        )
        return FlightSoftwareOutput(
            batch.satellite_id,
            batch.invocation_id,
            tuple(commands),
            telemetry,
            requested_releases,
        )

    def _update_maneuver_receipts(self, events: tuple[InputEvent, ...]) -> None:
        successful = {CommandDisposition.ACCEPTED, CommandDisposition.DUPLICATE}
        for event in events:
            if event.kind is not InputKind.ACTUATOR_RECEIPT:
                continue
            receipt = event.payload
            if not isinstance(receipt, ActuatorCommandReceipt):
                continue
            if receipt.command_id in self._scheduled_burn_pending:
                self._scheduled_burn_pending.remove(receipt.command_id)
                if receipt.disposition in successful:
                    self._scheduled_burn_had_accepted = True
                else:
                    self._scheduled_burn_failed = True
            for plan_id, pending in tuple(self._pending_plan_commands.items()):
                if receipt.command_id not in pending:
                    continue
                if receipt.disposition not in successful:
                    self._plan_failures[plan_id] = receipt.disposition.value
                    self._pending_plan_commands.pop(plan_id, None)
                    self._accepted_plan_commands.pop(plan_id, None)
                    break
                accepted = self._accepted_plan_commands.setdefault(plan_id, set())
                accepted.add(receipt.command_id)
                if accepted >= pending:
                    self._executed_plans.add(plan_id)
                    self._pending_plan_commands.pop(plan_id, None)
                    self._accepted_plan_commands.pop(plan_id, None)
                    self._plan_failures.pop(plan_id, None)
                break

    def _snapshot_stack_state(self) -> dict[str, object]:
        return {
            "navigation": self._navigator.snapshot_state(),
            "controller": self._controller.snapshot_state(),
            "executive": self._executive.snapshot_state(),
            "active_load": None if self._active_load is None else to_primitive(self._active_load),
            "health": self._health.snapshot_state(),
            "resources": self._resources.snapshot_state(),
            "command_service": self._command_service.snapshot_state(),
            "conjunction": self._conjunction.snapshot_state(),
            "maneuver_plan": to_primitive(self._maneuver_plan),
            "executed_plans": sorted(self._executed_plans),
            "pending_plan_commands": {
                plan_id: [to_primitive(command_id) for command_id in sorted(command_ids, key=_packet_key)]
                for plan_id, command_ids in sorted(self._pending_plan_commands.items())
            },
            "accepted_plan_commands": {
                plan_id: [to_primitive(command_id) for command_id in sorted(command_ids, key=_packet_key)]
                for plan_id, command_ids in sorted(self._accepted_plan_commands.items())
            },
            "plan_failures": dict(sorted(self._plan_failures.items())),
            "scheduled_burn_pending": [
                to_primitive(command_id)
                for command_id in sorted(self._scheduled_burn_pending, key=_packet_key)
            ],
            "scheduled_burn_had_accepted": self._scheduled_burn_had_accepted,
            "scheduled_burn_failed": self._scheduled_burn_failed,
        }

    def _prepare_restored_stack_state(self, state: dict[str, object]) -> object:
        navigation_state = state.get("navigation")
        controller_state = state.get("controller")
        executive_state = state.get("executive")
        if not all(isinstance(item, dict) for item in (navigation_state, controller_state, executive_state)):
            raise ValueError("translation stack snapshot component state is invalid")
        active_primitive = state.get("active_load")
        active_load = (
            None if active_primitive is None else from_primitive(OnboardMissionConfigurationLoad, active_primitive)
        )
        executive_config = self._config_for_load(active_load)
        navigator = self._new_navigator()
        navigator.restore_state(navigation_state)  # type: ignore[arg-type]
        controller = TranslationController(self._control_config_for_load(active_load))
        controller.restore_state(controller_state)  # type: ignore[arg-type]
        executive = ReferenceMissionExecutive(executive_config)
        executive.restore_state(executive_state)  # type: ignore[arg-type]
        health = StackHealthManager(self.config.health)
        health.restore_state(dict(state.get("health", {})))
        resources = ResourceMonitor(self.config.resources)
        resources.restore_state(dict(state.get("resources", {})))
        commands = OnboardCommandService()
        commands.restore_state(dict(state.get("command_service", {})))
        conjunction = ConjunctionAvoidancePlanner(self.config.conjunction)
        conjunction.restore_state(dict(state.get("conjunction", {"sequence": 0, "active_plan": None})))
        maneuver_primitive = state.get("maneuver_plan")
        maneuver_plan = None if maneuver_primitive is None else from_primitive(ManeuverPlan, maneuver_primitive)
        return _RestoredTranslationState(
            navigator,
            controller,
            executive,
            active_load,
            health,
            resources,
            commands,
            conjunction,
            {str(value) for value in list(state.get("executed_plans", []))},
            maneuver_plan,
            {
                str(plan_id): {from_primitive(PacketId, value) for value in list(values)}
                for plan_id, values in dict(state.get("pending_plan_commands", {})).items()
            },
            {
                str(plan_id): {from_primitive(PacketId, value) for value in list(values)}
                for plan_id, values in dict(state.get("accepted_plan_commands", {})).items()
            },
            {str(plan_id): str(reason) for plan_id, reason in dict(state.get("plan_failures", {})).items()},
            {
                from_primitive(PacketId, value)
                for value in list(state.get("scheduled_burn_pending", []))
            },
            bool(state.get("scheduled_burn_had_accepted", False)),
            bool(state.get("scheduled_burn_failed", False)),
        )

    def _commit_restored_stack_state(self, state: object) -> None:
        if not isinstance(state, _RestoredTranslationState):
            raise TypeError("restored translation stack state is invalid")
        manager = self._new_mission_manager()
        if state.active_load is not None:
            result = manager.apply(state.active_load, accept=self._accept_load)
            if not result.accepted:
                raise ValueError("restored mission load is no longer compatible")
        self._navigator = state.navigator
        self._controller = state.controller
        self._executive = state.executive
        self._active_load = state.active_load
        self._executive_config = self._config_for_load(state.active_load)
        self._mission_manager = manager
        self._health = state.health
        self._resources = state.resources
        self._command_service = state.commands
        self._conjunction = state.conjunction
        self._executed_plans = state.executed_plans
        self._maneuver_plan = state.maneuver_plan
        self._pending_plan_commands = state.pending_plan_commands
        self._accepted_plan_commands = state.accepted_plan_commands
        self._plan_failures = state.plan_failures
        self._scheduled_burn_pending = state.scheduled_burn_pending
        self._scheduled_burn_had_accepted = state.scheduled_burn_had_accepted
        self._scheduled_burn_failed = state.scheduled_burn_failed

    def _active_load_identity(self) -> tuple[str | None, int | None]:
        if self._active_load is None:
            return None, None
        manifest = self._active_load.manifest
        return manifest.load_id, manifest.revision

    def _restored_active_load_identity(self, state: object) -> tuple[str | None, int | None]:
        if not isinstance(state, _RestoredTranslationState) or state.active_load is None:
            return None, None
        manifest = state.active_load.manifest
        return manifest.load_id, manifest.revision

    def _new_navigator(self) -> OrbitNavigator:
        return OrbitNavigator(
            initialization=self.config.navigation_initialization,
            body_frame=self.config.body_frame,
            inertial_frame=self.config.inertial_frame,
            relative_frame=self.config.relative_frame,
            loaded_own_state=self.config.loaded_own_state,
            sensor_mountings=self.config.sensor_mountings,
            sensor_calibrations=self.config.sensor_calibrations,
            filter_kind=self.config.navigation_filter,
            alpha=self.config.navigation_alpha,
            beta=self.config.navigation_beta,
            ekf_step_s=self.config.navigation_ekf_step_s,
            ekf_process_noise_diag_si=self.config.navigation_process_noise_diag_si,
            ekf_measurement_noise_diag_si=self.config.navigation_measurement_noise_diag_si,
            ekf_initial_covariance_diag_si=self.config.navigation_initial_covariance_diag_si,
            relative_mean_motion_rad_s=self.config.navigation_relative_mean_motion_rad_s,
            ekf_nis_limit=self.config.navigation_nis_limit,
            retain_full_provenance=not self._live_navigation_fast_path,
        )

    def _new_mission_manager(self) -> MissionLoadManager:
        capabilities = tuple(
            sorted(
                {
                    *(mode.value for mode in self.supported_primary_modes),
                    *self.config.enabled_capabilities,
                }
            )
        )
        return MissionLoadManager(
            stack_id=self.stack_id,
            stack_version=STACK_VERSION,
            capabilities=capabilities,
        )

    def _ingest_mission_loads(self, batch: FlightSoftwareInputBatch) -> tuple[MissionLoadResult, ...]:
        results: list[MissionLoadResult] = []
        for event in batch.events:
            if event.kind is not InputKind.MISSION_LOAD or not isinstance(
                event.payload, OnboardMissionConfigurationLoad
            ):
                continue
            prepared: dict[str, object] = {}

            def accept(
                load: OnboardMissionConfigurationLoad,
                prepared_state: dict[str, object] = prepared,
            ) -> tuple[bool, str | None]:
                accepted, reason = self._accept_load(load)
                if not accepted:
                    return accepted, reason
                try:
                    executive_config = self._config_for_load(load)
                    controller = TranslationController(self._control_config_for_load(load))
                    executive = ReferenceMissionExecutive(executive_config)
                except (TypeError, ValueError) as exc:
                    return False, f"mission load parameters are invalid: {exc}"
                prepared_state.update(
                    executive_config=executive_config,
                    controller=controller,
                    executive=executive,
                )
                return True, None

            result = self._mission_manager.apply(event.payload, accept=accept)
            results.append(result)
            if result.accepted:
                self._active_load = event.payload
                self._executive_config = prepared["executive_config"]  # type: ignore[assignment]
                self._executive = prepared["executive"]  # type: ignore[assignment]
                self._controller = prepared["controller"]  # type: ignore[assignment]
        return tuple(results)

    def _measurement_is_fresh(self, measurement: MeasurementEvent, now: ClockTag) -> bool:
        try:
            age_s = _elapsed_seconds(measurement.sample_time, now)
        except ValueError:
            return False
        return 0.0 <= age_s <= self.config.measurement_stale_after_s

    def _fresh_navigation_solution(
        self,
        solution: OrbitNavigationSolution,
        now: ClockTag,
    ) -> OrbitNavigationSolution:
        def fresh(epoch: ClockTag | None) -> bool:
            if epoch is None:
                return False
            try:
                age_s = _elapsed_seconds(epoch, now)
            except ValueError:
                return False
            return 0.0 <= age_s <= self.config.measurement_stale_after_s

        relative_tracks = tuple(track for track in solution.relative_tracks if fresh(track.epoch))
        if fresh(solution.own_state_epoch):
            return replace(solution, relative_tracks=relative_tracks)
        return replace(
            solution,
            position_eci_m=None,
            velocity_eci_m_s=None,
            mass_kg=None,
            relative_tracks=relative_tracks,
        )

    def _accept_load(self, load: OnboardMissionConfigurationLoad) -> tuple[bool, str | None]:
        mode = _mode_for_goal_type(load.primary_goal.goal_type)
        if mode is None or mode not in self.supported_primary_modes:
            return False, f"goal type {load.primary_goal.goal_type!r} is unsupported by {self.stack_id}"
        unsupported_sections = tuple(
            name
            for name, records in (
                ("onboard_geometry", load.onboard_geometry),
                ("calibration", load.calibration),
                ("tuning_tables", load.tuning_tables),
            )
            if records
        )
        if unsupported_sections:
            return (
                False,
                "reference stack cannot atomically activate mission-load sections: " + ", ".join(unsupported_sections),
            )
        parameter_names = {field.name for field in load.primary_goal.parameters}
        if "prediction_acceleration_fractions" in parameter_names:
            return False, "prediction_acceleration_fractions cannot be represented by scalar mission-load parameters"
        return True, None

    def _config_for_load(self, load: OnboardMissionConfigurationLoad | None) -> ReferenceExecutiveConfig:
        if load is None:
            return self.config.executive
        mode = _mode_for_goal_type(load.primary_goal.goal_type)
        if mode is None:
            raise ValueError("active mission load goal type is unsupported")
        return replace(
            self.config.executive,
            primary_goal=load.primary_goal,
            primary_mode=mode.value,
            constraints=load.constraints,
        )

    def _control_config_for_load(
        self,
        load: OnboardMissionConfigurationLoad | None,
    ) -> TranslationControlConfig:
        if load is None:
            return self.config.control
        mode = _mode_for_goal_type(load.primary_goal.goal_type)
        if mode is None:
            raise ValueError("active mission load goal type is unsupported")
        values = {field.name: field.value for field in load.primary_goal.parameters}
        changes: dict[str, object] = {"default_mode": mode}
        relative_names = ("r", "i", "c", "dr", "di", "dc")
        if all(f"target_{name}_{'m' if index < 3 else 'm_s'}" in values for index, name in enumerate(relative_names)):
            changes["target_relative_state_ric"] = tuple(
                float(values[f"target_{name}_{'m' if index < 3 else 'm_s'}"])
                for index, name in enumerate(relative_names)
            )
        eci_names = ("x", "y", "z", "vx", "vy", "vz")
        if all(f"target_{name}_{'m' if index < 3 else 'm_s'}" in values for index, name in enumerate(eci_names)):
            changes["target_state_eci"] = tuple(
                float(values[f"target_{name}_{'m' if index < 3 else 'm_s'}"]) for index, name in enumerate(eci_names)
            )
        scalar_fields = {
            "target_semi_major_axis_m": "target_semi_major_axis_m",
            "target_eccentricity": "target_eccentricity",
            "position_tolerance_m": "position_tolerance_m",
            "velocity_tolerance_m_s": "velocity_tolerance_m_s",
            "approach_speed_m_s": "approach_speed_m_s",
            "maximum_acceleration_m_s2": "max_acceleration_m_s2",
            "prediction_horizon_s": "prediction_horizon_s",
            "prediction_step_s": "prediction_step_s",
            "prediction_decision_interval_s": "prediction_decision_interval_s",
            "prediction_pulse_duration_s": "prediction_pulse_duration_s",
            "capture_radius_m": "capture_radius_m",
            "capture_margin_m": "capture_margin_m",
            "opponent_max_acceleration_m_s2": "opponent_max_acceleration_m_s2",
        }
        for parameter_name, config_name in scalar_fields.items():
            value = values.get(parameter_name)
            if value is not None:
                changes[config_name] = float(value)
        return replace(self.config.control, **changes)

    def _navigation_ready(self, solution: OrbitNavigationSolution, mode: TranslationMode) -> bool:
        if mode in (
            TranslationMode.SCHEDULED_BURN,
            TranslationMode.STATIONKEEPING,
            TranslationMode.ORBITAL_ELEMENTS,
            TranslationMode.ATMOSPHERIC_PASS,
        ):
            return solution.own_state_valid
        return bool(solution.own_state_valid and solution.relative_track(self.config.control.target_id))

    def _relative_mean_motion(
        self,
        solution: OrbitNavigationSolution,
        track,
    ) -> float:
        if self.config.control.mean_motion_rad_s > 0.0:
            return self.config.control.mean_motion_rad_s
        position = (
            track.chief_position_eci_m
            if track is not None and track.chief_position_eci_m is not None
            else solution.position_eci_m
        )
        if position is None:
            return self.config.navigation_relative_mean_motion_rad_s
        radius = max(float(np.linalg.norm(position)), 1.0)
        return sqrt(3.986004418e14 / radius**3)

    def _maneuver_effort(
        self,
        plan: ManeuverPlan,
        solution: OrbitNavigationSolution,
        track,
    ) -> RequestedEffort | None:
        mass_kg = solution.mass_kg or self.config.control.assumed_mass_kg
        if mass_kg <= 0.0 or not solution.own_state_valid:
            return None
        position = np.asarray(
            track.chief_position_eci_m
            if track is not None and track.chief_position_eci_m is not None
            else solution.position_eci_m,
            dtype=float,
        )
        velocity = np.asarray(
            track.chief_velocity_eci_m_s
            if track is not None and track.chief_velocity_eci_m_s is not None
            else solution.velocity_eci_m_s,
            dtype=float,
        )
        radial = position / np.linalg.norm(position)
        cross_track = np.cross(position, velocity)
        cross_track /= np.linalg.norm(cross_track)
        in_track = np.cross(cross_track, radial)
        ric_to_eci = np.column_stack((radial, in_track, cross_track))
        burn_duration_s = max(self.config.navigation_ekf_step_s, 1.0e-3)
        force_eci = ric_to_eci @ np.asarray(plan.delta_v_ric_m_s, dtype=float) * float(mass_kg) / burn_duration_s
        validity_ticks = max(1, int(round(burn_duration_s / (plan.created_at.tick_period_ns * 1.0e-9))))
        return RequestedEffort(
            f"maneuver.{plan.plan_id}",
            RequestedEffortKind.FORCE,
            solution.generated_at,
            solution.inertial_frame,
            ValidityInterval(solution.generated_at, _add_ticks(solution.generated_at, validity_ticks)),
            force_n=tuple(float(value) for value in force_eci),
        )

    def _point_thrust_axis(
        self,
        solution: OrbitNavigationSolution,
        direction_eci: tuple[float, float, float],
    ) -> tuple[float | None, AllocationResult | None]:
        if self._attitude_allocator is None:
            return None, None
        boresight_body = (
            (1.0, 0.0, 0.0) if self._attitude_reference is None else self._attitude_reference.config.boresight_body
        )
        attitude_solution = solution.attitude
        reference = AttitudeReferenceGenerator(
            AttitudeReferenceConfig(
                AttitudeReferenceMode.THRUST,
                thrust_direction_eci=direction_eci,
                boresight_body=boresight_body,
                validity_ticks=self.config.control.validity_ticks,
            ),
            inertial_frame=self.config.inertial_frame,
        ).generate(attitude_solution)
        if reference is None:
            return None, None
        effort = self.config.attitude_controller.control(attitude_solution, reference)
        allocation = (
            None
            if effort is None
            else self._attitude_allocator.allocate(
                effort,
                attitude_solution,
                command_id=self._next_command_id(),
            )
        )
        quaternion = attitude_solution.attitude_quat_bn
        if quaternion is None:
            return None, allocation
        body_direction = quaternion_to_dcm_bn(np.asarray(quaternion)) @ np.asarray(direction_eci)
        body_direction /= max(float(np.linalg.norm(body_direction)), 1.0e-15)
        boresight = np.asarray(boresight_body, dtype=float)
        boresight /= np.linalg.norm(boresight)
        error = acos(float(np.clip(body_direction @ boresight, -1.0, 1.0)))
        return error, allocation


class OrbitReferenceFlightSoftwareStack(_TranslationReferenceFlightSoftwareStack):
    stack_id = "fsw.orbit_reference"
    supported_primary_modes = frozenset(
        {
            TranslationMode.SCHEDULED_BURN,
            TranslationMode.STATIONKEEPING,
            TranslationMode.ORBITAL_ELEMENTS,
            TranslationMode.ATMOSPHERIC_PASS,
        }
    )


class RpoReferenceFlightSoftwareStack(_TranslationReferenceFlightSoftwareStack):
    stack_id = "fsw.rpo_reference"
    supported_primary_modes = frozenset(
        {
            TranslationMode.RIC_HOLD,
            TranslationMode.R_BAR_APPROACH,
            TranslationMode.V_BAR_APPROACH,
            TranslationMode.C_BAR_APPROACH,
            TranslationMode.WAYPOINT,
            TranslationMode.RIC_PD_TRANSFER,
            TranslationMode.TERMINAL_BRAKING,
            TranslationMode.PASSIVE_RETREAT,
            TranslationMode.INTERCEPT_COAST,
            TranslationMode.PREDICTIVE_EVASION,
        }
    )


class LowThrustReferenceFlightSoftwareStack(_TranslationReferenceFlightSoftwareStack):
    stack_id = "fsw.low_thrust_reference"
    supported_primary_modes = frozenset(
        {
            TranslationMode.LOW_THRUST_PHASING,
            TranslationMode.ORBITAL_ELEMENTS,
        }
    )


def build_builtin_stack(
    config: object,
) -> SatelliteFlightSoftware:
    if isinstance(config, PassiveStackConfig):
        return PassiveFlightSoftwareStack(config)
    if isinstance(config, AttitudeReferenceStackConfig):
        return AttitudeReferenceFlightSoftwareStack(config)
    if isinstance(config, OrbitReferenceStackConfig):
        return OrbitReferenceFlightSoftwareStack(config)
    if isinstance(config, RpoReferenceStackConfig):
        return RpoReferenceFlightSoftwareStack(config)
    if isinstance(config, LowThrustReferenceStackConfig):
        return LowThrustReferenceFlightSoftwareStack(config)
    from .game_stacks import GamePilotReferenceFlightSoftwareStack, GamePilotReferenceStackConfig

    if isinstance(config, GamePilotReferenceStackConfig):
        return GamePilotReferenceFlightSoftwareStack(config)
    raise TypeError(f"unsupported built-in stack configuration {type(config).__qualname__}")


def _mode_for_goal_type(goal_type: str) -> TranslationMode | None:
    token = goal_type.strip().lower().replace("-", "_").replace(".", "_")
    aliases = {
        "scheduled_burn": TranslationMode.SCHEDULED_BURN,
        "orbit_scheduled_burn": TranslationMode.SCHEDULED_BURN,
        "stationkeeping": TranslationMode.STATIONKEEPING,
        "orbit_stationkeeping": TranslationMode.STATIONKEEPING,
        "orbital_elements": TranslationMode.ORBITAL_ELEMENTS,
        "orbit_elements": TranslationMode.ORBITAL_ELEMENTS,
        "rpo_hold": TranslationMode.RIC_HOLD,
        "ric_hold": TranslationMode.RIC_HOLD,
        "r_bar_approach": TranslationMode.R_BAR_APPROACH,
        "rpo_r_bar_approach": TranslationMode.R_BAR_APPROACH,
        "v_bar_approach": TranslationMode.V_BAR_APPROACH,
        "rpo_v_bar_approach": TranslationMode.V_BAR_APPROACH,
        "c_bar_approach": TranslationMode.C_BAR_APPROACH,
        "rpo_c_bar_approach": TranslationMode.C_BAR_APPROACH,
        "waypoint": TranslationMode.WAYPOINT,
        "rpo_waypoint": TranslationMode.WAYPOINT,
        "ric_pd_transfer": TranslationMode.RIC_PD_TRANSFER,
        "rpo_ric_pd_transfer": TranslationMode.RIC_PD_TRANSFER,
        "terminal_braking": TranslationMode.TERMINAL_BRAKING,
        "rpo_terminal_braking": TranslationMode.TERMINAL_BRAKING,
        "passive_retreat": TranslationMode.PASSIVE_RETREAT,
        "rpo_passive_retreat": TranslationMode.PASSIVE_RETREAT,
        "intercept_coast": TranslationMode.INTERCEPT_COAST,
        "rpo_intercept_coast": TranslationMode.INTERCEPT_COAST,
        "predictive_evasion": TranslationMode.PREDICTIVE_EVASION,
        "rpo_predictive_evasion": TranslationMode.PREDICTIVE_EVASION,
        "low_thrust_phasing": TranslationMode.LOW_THRUST_PHASING,
        "phasing": TranslationMode.LOW_THRUST_PHASING,
    }
    return aliases.get(token)


def _actuator_review_fields(batch: FlightSoftwareInputBatch) -> tuple[TelemetryField, ...]:
    fields: list[TelemetryField] = []
    for event in batch.events:
        if event.kind is InputKind.ACTUATOR_RECEIPT and isinstance(event.payload, ActuatorCommandReceipt):
            fields.append(
                TelemetryField(
                    f"receipt.{event.payload.command_id.source_id}.{event.payload.command_id.sequence}",
                    event.payload.disposition.value,
                )
            )
        elif event.kind is InputKind.ACTUATOR_TELEMETRY and isinstance(event.payload, ActuatorTelemetryPayload):
            fields.extend(
                TelemetryField(
                    f"realized.{event.payload.actuator_id}.{field.name}",
                    field.value,
                    field.unit,
                )
                for field in event.payload.fields
            )
    return tuple(fields)
