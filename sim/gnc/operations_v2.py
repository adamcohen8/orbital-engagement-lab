"""Operational services shared by the complete GNC v2 reference stacks.

The services in this module consume only flight-software boundary records and
estimated state.  They deliberately do not inspect simulator truth.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from math import isfinite

import numpy as np

from sim.control.orbit.hcw_transfer import propagate_hcw_relative_state, solve_hcw_position_rendezvous
from sim.flight_software.contracts import (
    ActuatorCommandReceipt,
    ActuatorTelemetryPayload,
    ClockTag,
    CommandDisposition,
    GroundCommandPayload,
    InputEvent,
    InputKind,
    MeasurementEvent,
    ModeledFaultIndicationPayload,
    PacketId,
    TelemetryField,
    VehicleResourceMeasurement,
)
from sim.gnc.navigation_v2 import OrbitNavigationSolution, RelativeStateEstimateSI


def elapsed_seconds(start: ClockTag, end: ClockTag) -> float:
    if (start.clock_id, start.tick_period_ns, start.scale, start.reset_counter) != (
        end.clock_id,
        end.tick_period_ns,
        end.scale,
        end.reset_counter,
    ):
        raise ValueError("onboard service clocks must share a domain")
    return (end.ticks - start.ticks) * start.tick_period_ns * 1.0e-9


class HealthState(str, Enum):
    NOMINAL = "nominal"
    DEGRADED = "degraded"
    RECOVERY = "recovery"


@dataclass(frozen=True, slots=True)
class HealthManagerConfig:
    rejection_limit: int = 3
    saturation_limit: int = 5
    clear_dwell_s: float = 0.0
    isolate_on_saturation: bool = True
    actuator_fallbacks: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.rejection_limit < 1 or self.saturation_limit < 1:
            raise ValueError("FDIR count limits must be positive")
        if not isfinite(self.clear_dwell_s) or self.clear_dwell_s < 0.0:
            raise ValueError("FDIR clear_dwell_s must be finite and nonnegative")
        primaries = [item[0] for item in self.actuator_fallbacks]
        if any(not primary.strip() or not backup.strip() for primary, backup in self.actuator_fallbacks):
            raise ValueError("FDIR fallback component IDs must be non-empty")
        if len(primaries) != len(set(primaries)):
            raise ValueError("FDIR primary fallback IDs must be unique")


@dataclass(frozen=True, slots=True)
class HealthAssessment:
    state: HealthState
    active_faults: tuple[tuple[str, str], ...]
    isolated_components: tuple[str, ...]
    selected_actuators: tuple[tuple[str, str], ...]

    def selected_actuator(self, primary: str) -> str | None:
        return dict(self.selected_actuators).get(primary, primary if primary not in self.isolated_components else None)


class StackHealthManager:
    """Deterministic, stack-owned fault detection, isolation, and reconfiguration."""

    _REJECTED = {
        CommandDisposition.REJECTED_SCHEMA,
        CommandDisposition.REJECTED_VERSION,
        CommandDisposition.REJECTED_TARGET,
        CommandDisposition.REJECTED_SEQUENCE,
        CommandDisposition.REJECTED_TIME,
        CommandDisposition.REJECTED_FRAME,
        CommandDisposition.REJECTED_VALUE,
        CommandDisposition.REJECTED_INTERLOCK,
        CommandDisposition.REJECTED_DEVICE_STATE,
    }

    def __init__(self, config: HealthManagerConfig | None = None) -> None:
        self.config = config if config is not None else HealthManagerConfig()
        self._faults: dict[str, str] = {}
        self._rejections: dict[str, int] = {}
        self._saturations: dict[str, int] = {}
        self._clear_started: dict[str, ClockTag] = {}
        self._seen_packet_ids: set[PacketId] = set()
        self._seen_packet_order: deque[PacketId] = deque()

    def update(self, now: ClockTag, events: tuple[InputEvent, ...]) -> HealthAssessment:
        for event in events:
            if event.packet_id in self._seen_packet_ids:
                continue
            self._seen_packet_ids.add(event.packet_id)
            self._seen_packet_order.append(event.packet_id)
            if len(self._seen_packet_order) > 4096:
                self._seen_packet_ids.discard(self._seen_packet_order.popleft())
            payload = event.payload
            if event.kind is InputKind.MODELED_FAULT_INDICATION and isinstance(
                payload, ModeledFaultIndicationPayload
            ):
                if payload.active:
                    self._faults[payload.component_id] = payload.fault_code
                    self._clear_started.pop(payload.component_id, None)
                else:
                    self._clear_started.setdefault(payload.component_id, now)
            elif event.kind is InputKind.ACTUATOR_RECEIPT and isinstance(payload, ActuatorCommandReceipt):
                # v1 receipts preserve the command identity but not the target
                # actuator.  Treat repeated rejection as a command-path fault;
                # modeled indications and telemetry provide device isolation.
                component = "actuator_command_path"
                if payload.disposition in self._REJECTED:
                    self._clear_started.pop(component, None)
                    count = self._rejections.get(component, 0) + 1
                    self._rejections[component] = count
                    if count >= self.config.rejection_limit:
                        self._faults[component] = "repeated_command_rejection"
                elif payload.disposition is CommandDisposition.ACCEPTED:
                    self._rejections[component] = 0
                    if self._faults.get(component) == "repeated_command_rejection":
                        self._clear_started.setdefault(component, now)
            elif event.kind is InputKind.ACTUATOR_TELEMETRY and isinstance(payload, ActuatorTelemetryPayload):
                values = {field.name: field.value for field in payload.fields}
                saturated = bool(values.get("saturated", False))
                if saturated:
                    self._clear_started.pop(payload.actuator_id, None)
                    count = self._saturations.get(payload.actuator_id, 0) + 1
                    self._saturations[payload.actuator_id] = count
                    if self.config.isolate_on_saturation and count >= self.config.saturation_limit:
                        self._faults[payload.actuator_id] = "persistent_saturation"
                else:
                    self._saturations[payload.actuator_id] = 0
                    if self._faults.get(payload.actuator_id) == "persistent_saturation":
                        self._clear_started.setdefault(payload.actuator_id, now)
        for component, started in tuple(self._clear_started.items()):
            if elapsed_seconds(started, now) < self.config.clear_dwell_s:
                continue
            self._faults.pop(component, None)
            self._rejections.pop(component, None)
            self._saturations.pop(component, None)
            del self._clear_started[component]
        isolated = tuple(sorted(self._faults))
        fallback_map = dict(self.config.actuator_fallbacks)
        selected = tuple(
            sorted(
                (primary, backup)
                for primary, backup in fallback_map.items()
                if primary in isolated and backup not in isolated
            )
        )
        state = HealthState.RECOVERY if self._faults else HealthState.NOMINAL
        if not self._faults and (any(self._rejections.values()) or any(self._saturations.values())):
            state = HealthState.DEGRADED
        return HealthAssessment(state, tuple(sorted(self._faults.items())), isolated, selected)

    def snapshot_state(self) -> dict[str, object]:
        return {
            "faults": dict(sorted(self._faults.items())),
            "rejections": dict(sorted(self._rejections.items())),
            "saturations": dict(sorted(self._saturations.items())),
            "clear_started": {
                component: _clock_state(value) for component, value in sorted(self._clear_started.items())
            },
            "seen_packet_ids": [
                {
                    "source_id": packet.source_id,
                    "boot_id": packet.boot_id,
                    "sequence": packet.sequence,
                }
                for packet in self._seen_packet_order
            ],
        }

    def restore_state(self, state: dict[str, object]) -> None:
        self._faults = {str(k): str(v) for k, v in dict(state.get("faults", {})).items()}
        self._rejections = {str(k): int(v) for k, v in dict(state.get("rejections", {})).items()}
        self._saturations = {str(k): int(v) for k, v in dict(state.get("saturations", {})).items()}
        clear_state = state.get("clear_started", {})
        if clear_state is None:  # compatibility with early development snapshots
            clear_state = {}
        self._clear_started = {
            str(component): _required_clock_state(value)
            for component, value in dict(clear_state).items()
        }
        seen_values = list(state.get("seen_packet_ids", []))
        self._seen_packet_order = deque(
            PacketId(str(value["source_id"]), str(value["boot_id"]), int(value["sequence"]))
            for value in seen_values
        )
        if len(self._seen_packet_order) > 4096:
            raise ValueError("health snapshot contains too many replay PacketIds")
        self._seen_packet_ids = set(self._seen_packet_order)


@dataclass(frozen=True, slots=True)
class ResourceLimits:
    minimum_battery_soc: float = 0.15
    minimum_available_power_w: float = 0.0
    maximum_temperature_k: float = 333.15
    maximum_storage_fraction: float = 0.95
    minimum_propellant_kg: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_battery_soc <= 1.0:
            raise ValueError("minimum_battery_soc must be in [0, 1]")
        if self.minimum_available_power_w < 0.0 or self.minimum_propellant_kg < 0.0:
            raise ValueError("minimum power and propellant must be nonnegative")
        if self.maximum_temperature_k <= 0.0 or not 0.0 < self.maximum_storage_fraction <= 1.0:
            raise ValueError("temperature and storage limits are invalid")


@dataclass(frozen=True, slots=True)
class ResourceState:
    battery_soc: float | None = None
    available_power_w: float | None = None
    maximum_temperature_k: float | None = None
    storage_used_bytes: float | None = None
    storage_capacity_bytes: float | None = None
    propellant_kg: float | None = None
    violations: tuple[str, ...] = ()

    @property
    def command_allowed(self) -> bool:
        return not self.violations

    @property
    def storage_fraction(self) -> float | None:
        if self.storage_used_bytes is None or self.storage_capacity_bytes in (None, 0.0):
            return None
        return self.storage_used_bytes / self.storage_capacity_bytes


class ResourceMonitor:
    """Maintains resource belief from typed telemetry; unknown is not fabricated."""

    _FIELD_NAMES = (
        "battery_soc",
        "available_power_w",
        "maximum_temperature_k",
        "storage_used_bytes",
        "storage_capacity_bytes",
        "propellant_kg",
    )

    def __init__(self, limits: ResourceLimits | None = None) -> None:
        self.limits = limits if limits is not None else ResourceLimits()
        self._values: dict[str, float] = {}
        self._propellant_measured = False

    def update(self, events: tuple[InputEvent, ...], *, mass_kg: float | None = None, dry_mass_kg: float = 0.0) -> ResourceState:
        for event in events:
            payload = event.payload
            if event.kind is InputKind.MEASUREMENT and isinstance(payload, MeasurementEvent) and isinstance(
                payload.payload, VehicleResourceMeasurement
            ):
                for name in self._FIELD_NAMES:
                    value = getattr(payload.payload, name)
                    if value is not None:
                        self._values[name] = float(value)
                        if name == "propellant_kg":
                            self._propellant_measured = True
            elif event.kind is InputKind.ACTUATOR_TELEMETRY and isinstance(payload, ActuatorTelemetryPayload):
                for field in payload.fields:
                    if field.name in self._FIELD_NAMES and isinstance(field.value, (int, float)) and isfinite(float(field.value)):
                        self._values[field.name] = float(field.value)
                        if field.name == "propellant_kg":
                            self._propellant_measured = True
        if not self._propellant_measured and mass_kg is not None:
            self._values["propellant_kg"] = max(float(mass_kg) - float(dry_mass_kg), 0.0)
        state = ResourceState(**{name: self._values.get(name) for name in self._FIELD_NAMES})
        violations: list[str] = []
        if state.battery_soc is not None and state.battery_soc < self.limits.minimum_battery_soc:
            violations.append("battery_soc_low")
        if state.available_power_w is not None and state.available_power_w < self.limits.minimum_available_power_w:
            violations.append("available_power_low")
        if state.maximum_temperature_k is not None and state.maximum_temperature_k > self.limits.maximum_temperature_k:
            violations.append("temperature_high")
        if state.storage_fraction is not None and state.storage_fraction > self.limits.maximum_storage_fraction:
            violations.append("storage_high")
        if state.propellant_kg is not None and state.propellant_kg < self.limits.minimum_propellant_kg:
            violations.append("propellant_low")
        return ResourceState(
            state.battery_soc,
            state.available_power_w,
            state.maximum_temperature_k,
            state.storage_used_bytes,
            state.storage_capacity_bytes,
            state.propellant_kg,
            tuple(violations),
        )

    def snapshot_state(self) -> dict[str, object]:
        return {
            "values": dict(sorted(self._values.items())),
            "propellant_measured": self._propellant_measured,
        }

    def restore_state(self, state: dict[str, object]) -> None:
        values = {str(k): float(v) for k, v in dict(state.get("values", {})).items()}
        if any(name not in self._FIELD_NAMES or not isfinite(value) for name, value in values.items()):
            raise ValueError("resource snapshot contains invalid fields")
        self._values = values
        measured = state.get("propellant_measured", False)
        if not isinstance(measured, bool):
            raise ValueError("resource snapshot propellant source is invalid")
        self._propellant_measured = measured


class OnboardCommandService:
    """Stored-command service with deterministic due-time release and deduplication."""

    def __init__(self) -> None:
        self._pending: dict[str, GroundCommandPayload] = {}
        self._completed: set[str] = set()

    def ingest(self, now: ClockTag, events: tuple[InputEvent, ...]) -> tuple[GroundCommandPayload, ...]:
        due: list[GroundCommandPayload] = []
        for event in events:
            if event.kind is not InputKind.GROUND_COMMAND or not isinstance(event.payload, GroundCommandPayload):
                continue
            command = event.payload
            if command.command_id in self._completed or command.command_id in self._pending:
                continue
            if command.execute_at is None or elapsed_seconds(now, command.execute_at) <= 0.0:
                due.append(command)
                self._completed.add(command.command_id)
            else:
                self._pending[command.command_id] = command
        for command_id, command in sorted(tuple(self._pending.items())):
            if command.execute_at is not None and elapsed_seconds(now, command.execute_at) <= 0.0:
                due.append(command)
                self._completed.add(command_id)
                del self._pending[command_id]
        return tuple(due)

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def next_release_at(self) -> ClockTag | None:
        """Return the earliest exact execution boundary for stored commands."""

        pending = [command.execute_at for command in self._pending.values() if command.execute_at is not None]
        if not pending:
            return None
        domains = {
            (tag.clock_id, tag.tick_period_ns, tag.scale, tag.reset_counter)
            for tag in pending
        }
        if len(domains) != 1:
            raise ValueError("stored commands must use one onboard clock domain")
        return min(pending, key=lambda tag: tag.ticks)

    def snapshot_state(self) -> dict[str, object]:
        from sim.flight_software.schemas import to_primitive
        return {
            "pending": [to_primitive(command) for command in self._pending.values()],
            "completed": sorted(self._completed),
        }

    def restore_state(self, state: dict[str, object]) -> None:
        from sim.flight_software.schemas import from_primitive
        pending = [from_primitive(GroundCommandPayload, value) for value in list(state.get("pending", []))]
        self._pending = {command.command_id: command for command in pending}
        self._completed = {str(value) for value in list(state.get("completed", []))}


class AdcsOperationalMode(str, Enum):
    DETUMBLE = "detumble"
    COARSE_SUN = "coarse_sun"
    NOMINAL = "nominal"
    MOMENTUM_UNLOAD = "momentum_unload"
    DEGRADED = "degraded"


@dataclass(frozen=True, slots=True)
class AdcsModeConfig:
    """Declared transition envelope for the reference ADCS executive."""

    detumble_entry_rate_rad_s: float = 0.5
    detumble_exit_rate_rad_s: float = 0.02

    def __post_init__(self) -> None:
        if (
            not isfinite(self.detumble_entry_rate_rad_s)
            or not isfinite(self.detumble_exit_rate_rad_s)
            or self.detumble_entry_rate_rad_s <= self.detumble_exit_rate_rad_s
            or self.detumble_exit_rate_rad_s < 0.0
        ):
            raise ValueError("ADCS detumble rates must satisfy finite entry > exit >= 0")


@dataclass(frozen=True, slots=True)
class MomentumUnloadConfig:
    wheel_actuator_id: str = "attitude"
    torquer_actuator_id: str = "momentum_dump"
    start_fraction: float = 0.8
    stop_fraction: float = 0.55
    wheel_max_momentum_n_m_s: tuple[float, ...] = (1.0, 1.0, 1.0)
    wheel_axes_body: tuple[tuple[float, ...], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    gain: float = 0.25
    max_dipole_a_m2: float = 20.0
    command_validity_ticks: int = 1

    def __post_init__(self) -> None:
        if not 0.0 < self.stop_fraction < self.start_fraction <= 1.0:
            raise ValueError("momentum unload thresholds must satisfy 0 < stop < start <= 1")
        if any(value <= 0.0 for value in self.wheel_max_momentum_n_m_s):
            raise ValueError("wheel momentum limits must be positive")
        axes = np.asarray(self.wheel_axes_body, dtype=float)
        if (
            axes.ndim != 2
            or axes.shape != (len(self.wheel_max_momentum_n_m_s), 3)
            or not np.all(np.isfinite(axes))
            or np.any(np.linalg.norm(axes, axis=1) <= 0.0)
        ):
            raise ValueError("wheel_axes_body must contain one finite nonzero body axis per wheel")
        if self.gain <= 0.0 or self.max_dipole_a_m2 <= 0.0:
            raise ValueError("momentum unload gain and dipole limit must be positive")
        if (
            isinstance(self.command_validity_ticks, bool)
            or not isinstance(self.command_validity_ticks, int)
            or self.command_validity_ticks < 1
        ):
            raise ValueError("momentum unload command validity must be a positive integer tick count")


class AdcsModeManager:
    def __init__(
        self,
        config: MomentumUnloadConfig | None = None,
        mode_config: AdcsModeConfig | None = None,
    ) -> None:
        self.config = config
        self.mode_config = mode_config if mode_config is not None else AdcsModeConfig()
        self.mode = AdcsOperationalMode.NOMINAL
        self._wheel_momentum: np.ndarray | None = None

    def update(
        self,
        events: tuple[InputEvent, ...],
        *,
        angular_rate_norm_rad_s: float | None,
        navigation_valid: bool,
        actuator_fault: bool,
        reference_available: bool = True,
    ) -> AdcsOperationalMode:
        if self.config is not None:
            for event in events:
                if event.kind is not InputKind.ACTUATOR_TELEMETRY or not isinstance(
                    event.payload, ActuatorTelemetryPayload
                ) or event.payload.actuator_id != self.config.wheel_actuator_id:
                    continue
                fields = {field.name: field.value for field in event.payload.fields}
                values = []
                for index in range(len(self.config.wheel_max_momentum_n_m_s)):
                    value = fields.get(f"wheel_{index}_momentum_n_m_s")
                    if not isinstance(value, (int, float)):
                        values = []
                        break
                    values.append(float(value))
                if values:
                    self._wheel_momentum = np.asarray(values)
        if actuator_fault:
            self.mode = AdcsOperationalMode.DEGRADED
        elif angular_rate_norm_rad_s is not None and (
            angular_rate_norm_rad_s > self.mode_config.detumble_entry_rate_rad_s
            or (
                self.mode is AdcsOperationalMode.DETUMBLE
                and angular_rate_norm_rad_s > self.mode_config.detumble_exit_rate_rad_s
            )
        ):
            self.mode = AdcsOperationalMode.DETUMBLE
        elif not reference_available:
            self.mode = AdcsOperationalMode.DEGRADED
        elif not navigation_valid:
            self.mode = AdcsOperationalMode.COARSE_SUN
        elif self.config is not None and self._wheel_momentum is not None:
            fraction = self.momentum_fraction
            if fraction >= self.config.start_fraction or (
                self.mode is AdcsOperationalMode.MOMENTUM_UNLOAD and fraction > self.config.stop_fraction
            ):
                self.mode = AdcsOperationalMode.MOMENTUM_UNLOAD
            else:
                self.mode = AdcsOperationalMode.NOMINAL
        else:
            self.mode = AdcsOperationalMode.NOMINAL
        return self.mode

    @property
    def momentum_fraction(self) -> float:
        if self.config is None or self._wheel_momentum is None:
            return 0.0
        limits = np.asarray(self.config.wheel_max_momentum_n_m_s)
        return float(np.max(np.abs(self._wheel_momentum) / limits))

    def unload_dipole(self, magnetic_field_body_t: tuple[float, float, float] | None) -> tuple[float, float, float] | None:
        if (
            self.mode is not AdcsOperationalMode.MOMENTUM_UNLOAD
            or self.config is None
            or self._wheel_momentum is None
            or magnetic_field_body_t is None
        ):
            return None
        magnetic = np.asarray(magnetic_field_body_t, dtype=float)
        b2 = float(magnetic @ magnetic)
        if b2 <= 1.0e-24:
            return None
        wheel_axes = np.asarray(self.config.wheel_axes_body, dtype=float)
        body_momentum = wheel_axes.T @ self._wheel_momentum
        desired_torque = -self.config.gain * body_momentum
        dipole = np.cross(magnetic, desired_torque) / b2
        norm = float(np.linalg.norm(dipole))
        if norm > self.config.max_dipole_a_m2:
            dipole *= self.config.max_dipole_a_m2 / norm
        return tuple(float(value) for value in dipole)

    def snapshot_state(self) -> dict[str, object]:
        return {
            "mode": self.mode.value,
            "wheel_momentum": None if self._wheel_momentum is None else self._wheel_momentum.tolist(),
        }

    def restore_state(self, state: dict[str, object]) -> None:
        self.mode = AdcsOperationalMode(str(state["mode"]))
        momentum = state.get("wheel_momentum")
        self._wheel_momentum = None if momentum is None else np.asarray(momentum, dtype=float)


@dataclass(frozen=True, slots=True)
class ConjunctionConfig:
    enabled: bool = False
    keep_out_radius_m: float = 100.0
    prediction_horizon_s: float = 600.0
    avoidance_delta_v_m_s: float = 0.1
    maneuver_lead_time_s: float = 30.0

    def __post_init__(self) -> None:
        if self.keep_out_radius_m <= 0.0 or self.prediction_horizon_s <= 0.0:
            raise ValueError("conjunction radius and horizon must be positive")
        if self.avoidance_delta_v_m_s <= 0.0 or self.maneuver_lead_time_s <= 0.0:
            raise ValueError("avoidance delta-v and lead time must be positive")


@dataclass(frozen=True, slots=True)
class ManeuverPlan:
    plan_id: str
    target_id: str
    created_at: ClockTag
    execute_at: ClockTag
    delta_v_ric_m_s: tuple[float, float, float]
    predicted_miss_distance_m: float
    time_to_closest_approach_s: float
    reason: str


def _closest_hcw_approach(
    state_km_km_s: np.ndarray,
    mean_motion_rad_s: float,
    horizon_s: float,
) -> tuple[float, np.ndarray]:
    """Return the bounded-horizon HCW closest approach without sample aliasing.

    A coarse deterministic grid brackets zeros of ``r dot v``.  Those zeros
    contain every smooth local range minimum; bisection then resolves each
    bracket independently of encounter speed.
    """

    sample_count = max(121, int(np.ceil(horizon_s / 0.5)) + 1)
    times = np.linspace(0.0, horizon_s, sample_count)
    states = np.asarray(
        [propagate_hcw_relative_state(state_km_km_s, mean_motion_rad_s, float(value)) for value in times]
    )
    distances_squared = np.einsum("ij,ij->i", states[:, :3], states[:, :3])
    best_index = int(np.argmin(distances_squared))
    best_time = float(times[best_index])
    best_state = states[best_index]
    best_distance_squared = float(distances_squared[best_index])
    radial_rates = np.einsum("ij,ij->i", states[:, :3], states[:, 3:])

    for index in range(len(times) - 1):
        left = float(times[index])
        right = float(times[index + 1])
        left_rate = float(radial_rates[index])
        right_rate = float(radial_rates[index + 1])
        if left_rate > 0.0 or right_rate < 0.0:
            continue
        if left_rate == 0.0:
            candidate_time = left
        elif right_rate == 0.0:
            candidate_time = right
        else:
            # A negative-to-positive radial-rate crossing is a local range
            # minimum.  HCW is smooth, so deterministic bisection is enough.
            for _ in range(60):
                middle = 0.5 * (left + right)
                middle_state = propagate_hcw_relative_state(state_km_km_s, mean_motion_rad_s, middle)
                middle_rate = float(np.dot(middle_state[:3], middle_state[3:]))
                if middle_rate < 0.0:
                    left = middle
                else:
                    right = middle
            candidate_time = 0.5 * (left + right)
        candidate_state = propagate_hcw_relative_state(
            state_km_km_s,
            mean_motion_rad_s,
            candidate_time,
        )
        candidate_distance_squared = float(np.dot(candidate_state[:3], candidate_state[:3]))
        if candidate_distance_squared < best_distance_squared:
            best_time = candidate_time
            best_state = candidate_state
            best_distance_squared = candidate_distance_squared
    return best_time, best_state


class ConjunctionAvoidancePlanner:
    """Deterministic HCW screening and bounded impulsive avoidance planning."""

    def __init__(self, config: ConjunctionConfig | None = None) -> None:
        self.config = config if config is not None else ConjunctionConfig()
        self._sequence = 0
        self.active_plan: ManeuverPlan | None = None
        self._reported_completion_ids: set[str] = set()

    def assess(
        self,
        now: ClockTag,
        solution: OrbitNavigationSolution,
        mean_motion_rad_s: float,
        *,
        completed_plan_ids: frozenset[str] = frozenset(),
    ) -> ManeuverPlan | None:
        if not self.config.enabled:
            self.active_plan = None
            return None
        threat: tuple[RelativeStateEstimateSI, float, float, np.ndarray] | None = None
        for track in solution.relative_tracks:
            state = np.concatenate((np.asarray(track.position_m), np.asarray(track.velocity_m_s))) / 1000.0
            tca, closest_state = _closest_hcw_approach(
                state,
                mean_motion_rad_s,
                self.config.prediction_horizon_s,
            )
            closest = closest_state[:3] * 1000.0
            miss = float(np.linalg.norm(closest))
            if miss < self.config.keep_out_radius_m and (threat is None or miss < threat[1]):
                threat = (track, miss, tca, closest)
        if threat is None:
            self.active_plan = None
            return None
        track, miss, tca, closest = threat
        # Cross-track is preferred because it is decoupled in HCW.  Preserve a
        # deterministic sign away from the predicted closest-approach vector.
        direction = np.array([0.0, 0.0, 1.0 if closest[2] >= 0.0 else -1.0])
        if self.active_plan is not None and self.active_plan.target_id == track.target_id:
            if self.active_plan.plan_id in completed_plan_ids:
                if self.active_plan.plan_id not in self._reported_completion_ids:
                    # Preserve one explicit executed sample in the evidence
                    # stream before allowing a persistent threat to replan.
                    self._reported_completion_ids.add(self.active_plan.plan_id)
                    return self.active_plan
                # The accepted maneuver has already had a physical interval to
                # affect the next navigation solution.  A continuing threat is
                # a new planning problem, not a completed plan to latch forever.
                self.active_plan = None
            else:
                self.active_plan = ManeuverPlan(
                    self.active_plan.plan_id,
                    track.target_id,
                    self.active_plan.created_at,
                    self.active_plan.execute_at,
                    tuple(float(value) for value in direction * self.config.avoidance_delta_v_m_s),
                    miss,
                    tca,
                    self.active_plan.reason,
                )
                return self.active_plan
        ticks = max(1, int(round(self.config.maneuver_lead_time_s / (now.tick_period_ns * 1.0e-9))))
        execute_at = ClockTag(
            now.clock_id, now.ticks + ticks, now.tick_period_ns, now.scale, now.validity, now.reset_counter
        )
        plan = ManeuverPlan(
            f"ca-{self._sequence}",
            track.target_id,
            now,
            execute_at,
            tuple(float(value) for value in direction * self.config.avoidance_delta_v_m_s),
            miss,
            tca,
            "predicted_keep_out_violation",
        )
        self._sequence += 1
        self.active_plan = plan
        return plan

    def snapshot_state(self) -> dict[str, object]:
        from sim.flight_software.schemas import to_primitive
        return {
            "sequence": self._sequence,
            "active_plan": to_primitive(self.active_plan),
            "reported_completion_ids": sorted(self._reported_completion_ids),
        }

    def restore_state(self, state: dict[str, object]) -> None:
        from sim.flight_software.schemas import from_primitive
        self._sequence = int(state.get("sequence", 0))
        value = state.get("active_plan")
        self.active_plan = None if value is None else from_primitive(ManeuverPlan, value)
        self._reported_completion_ids = {
            str(value) for value in list(state.get("reported_completion_ids", []) or [])
        }


@dataclass(frozen=True, slots=True)
class HcwManeuverConfig:
    enabled: bool = False
    transfer_time_s: float = 300.0
    target_position_ric_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    maximum_delta_v_m_s: float = 5.0

    def __post_init__(self) -> None:
        if self.transfer_time_s <= 0.0 or self.maximum_delta_v_m_s <= 0.0:
            raise ValueError("HCW maneuver time and delta-v limit must be positive")
        if len(self.target_position_ric_m) != 3 or not all(isfinite(value) for value in self.target_position_ric_m):
            raise ValueError("HCW target position must contain three finite SI values")


class HcwManeuverPlanner:
    """Plans a deterministic first burn for an HCW waypoint transfer."""

    def __init__(self, config: HcwManeuverConfig | None = None) -> None:
        self.config = config if config is not None else HcwManeuverConfig()
        self._sequence = 0

    def plan(
        self,
        now: ClockTag,
        track: RelativeStateEstimateSI,
        mean_motion_rad_s: float,
    ) -> ManeuverPlan | None:
        if not self.config.enabled:
            return None
        state_km = np.concatenate((np.asarray(track.position_m), np.asarray(track.velocity_m_s))) / 1000.0
        # Shift the origin to the requested terminal waypoint, preserving the
        # target-frame velocity convention.
        state_km[:3] -= np.asarray(self.config.target_position_ric_m) / 1000.0
        solution = solve_hcw_position_rendezvous(
            state_km,
            np.zeros(3),
            mean_motion_rad_s,
            self.config.transfer_time_s,
        )
        delta_v = solution.required_delta_v_ric_km_s * 1000.0
        magnitude = float(np.linalg.norm(delta_v))
        if magnitude > self.config.maximum_delta_v_m_s:
            delta_v *= self.config.maximum_delta_v_m_s / magnitude
        bounded_state = state_km.copy()
        bounded_state[3:] += delta_v / 1000.0
        bounded_terminal = propagate_hcw_relative_state(
            bounded_state,
            mean_motion_rad_s,
            self.config.transfer_time_s,
        )
        plan = ManeuverPlan(
            f"hcw-{self._sequence}",
            track.target_id,
            now,
            now,
            tuple(float(value) for value in delta_v),
            float(np.linalg.norm(bounded_terminal[:3]) * 1000.0),
            self.config.transfer_time_s,
            "autonomous_hcw_waypoint",
        )
        self._sequence += 1
        return plan


def resource_telemetry(state: ResourceState) -> tuple[TelemetryField, ...]:
    fields: list[TelemetryField] = [TelemetryField("resource_violation_count", len(state.violations))]
    for name in ResourceMonitor._FIELD_NAMES:
        value = getattr(state, name)
        if value is not None:
            fields.append(TelemetryField(name, value))
    fields.extend(TelemetryField(f"resource_violation.{name}", True) for name in state.violations)
    return tuple(fields)


def _clock_state(value: ClockTag | None) -> dict[str, object] | None:
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


def _clock_from_state(value: object) -> ClockTag | None:
    if value is None:
        return None
    from sim.flight_software.contracts import ClockScale, TimeValidity
    mapping = dict(value)  # type: ignore[arg-type]
    return ClockTag(
        str(mapping["clock_id"]), int(mapping["ticks"]), int(mapping["tick_period_ns"]),
        ClockScale(str(mapping["scale"])), TimeValidity(str(mapping["validity"])), int(mapping["reset_counter"]),
    )


def _required_clock_state(value: object) -> ClockTag:
    result = _clock_from_state(value)
    if result is None:
        raise ValueError("onboard service snapshot clock is required")
    return result
