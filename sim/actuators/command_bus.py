"""Typed v2 actuator command acceptance, expiry, and demand state."""

from __future__ import annotations

from bisect import bisect_right, insort
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from sim.flight_software.contracts import (
    ACTUATOR_COMMAND_SCHEMA,
    CONTRACT_VERSION,
    ActuatorCommand,
    ActuatorCommandPayload,
    ActuatorCommandReceipt,
    ClockTag,
    CommandDisposition,
    FrameId,
    ThrusterOnOffCommand,
    ThrusterPulseCommand,
)
from sim.flight_software.schemas import (
    _canonical_json_bytes_trusted,
    canonical_json_bytes,
    from_primitive,
    to_primitive,
)


class ExpiryBehavior(str, Enum):
    LATCH = "latch"
    ZERO = "zero"
    IDLE = "idle"


class DemandMode(str, Enum):
    COMMANDED = "commanded"
    LATCHED = "latched"
    ZERO = "zero"
    IDLE = "idle"
    UNCOMMANDED = "uncommanded"


PayloadValidator = Callable[[ActuatorCommandPayload], tuple[bool, str | None]]
Interlock = Callable[[ActuatorCommand], tuple[bool, str | None]]


@dataclass(frozen=True, slots=True)
class ActuatorDeviceDefinition:
    satellite_id: str
    actuator_id: str
    command_frame: FrameId
    payload_types: tuple[type[object], ...]
    expiry_behavior: ExpiryBehavior
    validator: PayloadValidator | None = None
    interlock: Interlock | None = None
    enabled: bool = True
    acknowledge: bool = True
    supports_scheduled_reordering: bool = False

    def __post_init__(self) -> None:
        if not self.satellite_id.strip() or not self.actuator_id.strip():
            raise ValueError("satellite_id and actuator_id must be non-empty")
        if not self.payload_types:
            raise ValueError("payload_types must not be empty")
        if not isinstance(self.expiry_behavior, ExpiryBehavior):
            raise TypeError("expiry_behavior must be ExpiryBehavior")


@dataclass(frozen=True, slots=True)
class ActuatorDemand:
    actuator_id: str
    mode: DemandMode
    source_command: ActuatorCommand | None
    payload: ActuatorCommandPayload | None


@dataclass(frozen=True, slots=True)
class CommandAcceptanceRecord:
    command: ActuatorCommand
    receipt: ActuatorCommandReceipt
    effective_time_ns: int | None
    expiry_time_ns: int | None


@dataclass(frozen=True, slots=True)
class _AcceptedCommand:
    command: ActuatorCommand
    effective_time_ns: int
    expiry_time_ns: int | None
    publication_order: int


class ActuatorCommandBus:
    def __init__(self, devices: tuple[ActuatorDeviceDefinition, ...]) -> None:
        identities = [(device.satellite_id, device.actuator_id) for device in devices]
        if len(identities) != len(set(identities)):
            raise ValueError("actuator device identities must be unique")
        self._devices = {identity: device for identity, device in zip(identities, devices)}
        self._ledger: dict[object, tuple[bytes, CommandDisposition]] = {}
        self._highest_sequence: dict[tuple[str, str, str], int] = {}
        self._accepted: dict[tuple[str, str], list[_AcceptedCommand]] = {identity: [] for identity in identities}
        self._accepted_keys: dict[tuple[str, str], list[tuple[int, int]]] = {
            identity: [] for identity in identities
        }
        self._event_times_ns: list[int] = []
        self._publication_order = 0
        self._records: list[CommandAcceptanceRecord] = []
        self._publications: list[tuple[ActuatorCommand, ClockTag]] = []

    @property
    def records(self) -> tuple[CommandAcceptanceRecord, ...]:
        return tuple(self._records)

    def publish(self, command: ActuatorCommand, *, received_at: ClockTag) -> ActuatorCommandReceipt | None:
        return self._publish(command, received_at=received_at, boundary_validated=False)

    def _publish(
        self,
        command: ActuatorCommand,
        *,
        received_at: ClockTag,
        boundary_validated: bool,
    ) -> ActuatorCommandReceipt | None:
        self._publications.append((command, received_at))
        fingerprint = (
            _canonical_json_bytes_trusted(command)
            if boundary_validated
            else canonical_json_bytes(command)
        )
        previous = self._ledger.get(command.command_id)
        if previous is not None:
            previous_fingerprint, previous_disposition = previous
            disposition = (
                CommandDisposition.DUPLICATE
                if previous_fingerprint == fingerprint and previous_disposition is CommandDisposition.ACCEPTED
                else CommandDisposition.REJECTED_SEQUENCE
            )
            return self._receipt(
                command,
                received_at,
                disposition,
                ("duplicate" if disposition is CommandDisposition.DUPLICATE else "sequence_reuse",),
            )

        disposition, status = self._validate(command, received_at)
        effective_ns: int | None = None
        expiry_ns: int | None = None
        if disposition is CommandDisposition.ACCEPTED:
            effective_ns = max(_effective_ns(command), _tag_ns(received_at))
            expiry_ns = _expiry_ns(command)
            accepted = _AcceptedCommand(command, effective_ns, expiry_ns, self._publication_order)
            self._publication_order += 1
            identity = (command.satellite_id, command.actuator_id)
            key = (accepted.effective_time_ns, accepted.publication_order)
            index = bisect_right(self._accepted_keys[identity], key)
            self._accepted_keys[identity].insert(index, key)
            self._accepted[identity].insert(index, accepted)
            for event_time in (effective_ns, expiry_ns):
                if event_time is not None:
                    event_index = bisect_right(self._event_times_ns, event_time)
                    if event_index == 0 or self._event_times_ns[event_index - 1] != event_time:
                        insort(self._event_times_ns, event_time)
            sequence_key = (command.actuator_id, command.command_id.source_id, command.command_id.boot_id)
            self._highest_sequence[sequence_key] = max(
                command.command_id.sequence,
                self._highest_sequence.get(sequence_key, command.command_id.sequence),
            )
        self._ledger[command.command_id] = (fingerprint, disposition)
        receipt = self._receipt(command, received_at, disposition, status)
        if receipt is not None:
            self._records.append(CommandAcceptanceRecord(command, receipt, effective_ns, expiry_ns))
        return receipt

    def snapshot_state(self) -> dict[str, object]:
        """Return deterministic replay state for a continuation checkpoint."""

        return {
            "publications": [
                {"command": to_primitive(command), "received_at": to_primitive(received_at)}
                for command, received_at in self._publications
            ]
        }

    def restore_state(self, state: object) -> None:
        if self._publications or self._ledger or any(self._accepted.values()):
            raise RuntimeError("actuator command bus restore requires a fresh bus")
        if not isinstance(state, dict) or set(state) != {"publications"}:
            raise ValueError("actuator command bus checkpoint is invalid")
        publications = state.get("publications")
        if not isinstance(publications, (list, tuple)):
            raise ValueError("actuator command bus publications must be a sequence")
        for item in publications:
            if not isinstance(item, dict):
                raise ValueError("actuator command bus publication is invalid")
            command = from_primitive(ActuatorCommand, item.get("command"))
            received_at = from_primitive(ClockTag, item.get("received_at"))
            self.publish(command, received_at=received_at)

    def publish_all(
        self, commands: tuple[ActuatorCommand, ...], *, received_at: ClockTag
    ) -> tuple[ActuatorCommandReceipt, ...]:
        return tuple(
            receipt for command in commands if (receipt := self.publish(command, received_at=received_at)) is not None
        )

    def _publish_all_boundary_validated(
        self, commands: tuple[ActuatorCommand, ...], *, received_at: ClockTag
    ) -> tuple[ActuatorCommandReceipt, ...]:
        """Publish commands contained in an adapter-validated FSW output."""

        return tuple(
            receipt
            for command in commands
            if (
                receipt := self._publish(
                    command,
                    received_at=received_at,
                    boundary_validated=True,
                )
            )
            is not None
        )

    def demand(self, *, satellite_id: str, actuator_id: str, at: ClockTag) -> ActuatorDemand:
        identity = (satellite_id, actuator_id)
        device = self._devices[identity]
        time_ns = _tag_ns(at)
        index = bisect_right(self._accepted_keys[identity], (time_ns, self._publication_order)) - 1
        if index < 0:
            return ActuatorDemand(actuator_id, DemandMode.UNCOMMANDED, None, None)
        active = self._accepted[identity][index]
        if active.expiry_time_ns is None or time_ns < active.expiry_time_ns:
            return ActuatorDemand(actuator_id, DemandMode.COMMANDED, active.command, active.command.payload)
        if device.expiry_behavior is ExpiryBehavior.LATCH:
            return ActuatorDemand(actuator_id, DemandMode.LATCHED, active.command, active.command.payload)
        mode = DemandMode.ZERO if device.expiry_behavior is ExpiryBehavior.ZERO else DemandMode.IDLE
        return ActuatorDemand(actuator_id, mode, active.command, None)

    def hard_event_times_ns(self, *, after_time_ns: int | None = None) -> tuple[int, ...]:
        if after_time_ns is None:
            return tuple(self._event_times_ns)
        return tuple(self._event_times_ns[bisect_right(self._event_times_ns, int(after_time_ns)) :])

    def _validate(self, command: ActuatorCommand, received_at: ClockTag) -> tuple[CommandDisposition, tuple[str, ...]]:
        if command.schema != ACTUATOR_COMMAND_SCHEMA:
            return CommandDisposition.REJECTED_SCHEMA, ("unsupported_command_schema",)
        if command.contract_version != CONTRACT_VERSION:
            return CommandDisposition.REJECTED_VERSION, ("unsupported_contract_version",)
        identity = (command.satellite_id, command.actuator_id)
        device = self._devices.get(identity)
        if device is None:
            return CommandDisposition.REJECTED_TARGET, ("unknown_satellite_or_actuator",)
        if command.frame != device.command_frame:
            return CommandDisposition.REJECTED_FRAME, ("unexpected_command_frame",)
        if not _same_clock_domain(command.issued_at, received_at) or not _same_clock_domain(
            command.validity.not_before, received_at
        ):
            return CommandDisposition.REJECTED_TIME, ("incomparable_clock_domain",)
        if isinstance(command.payload, ThrusterPulseCommand) and not _same_clock_domain(
            command.payload.start_at, received_at
        ):
            return CommandDisposition.REJECTED_TIME, ("incomparable_pulse_clock_domain",)
        received_ns = _tag_ns(received_at)
        if _tag_ns(command.issued_at) > received_ns:
            return CommandDisposition.REJECTED_TIME, ("issued_in_future",)
        expiry_ns = _expiry_ns(command)
        if expiry_ns is not None and received_ns >= expiry_ns:
            return CommandDisposition.REJECTED_TIME, ("command_expired",)
        effective_ns = _effective_ns(command)
        if expiry_ns is not None and effective_ns >= expiry_ns:
            return CommandDisposition.REJECTED_TIME, ("empty_command_validity_interval",)
        sequence_key = (command.actuator_id, command.command_id.source_id, command.command_id.boot_id)
        highest = self._highest_sequence.get(sequence_key)
        if highest is not None and command.command_id.sequence < highest and not device.supports_scheduled_reordering:
            return CommandDisposition.REJECTED_SEQUENCE, ("older_sequence",)
        if not isinstance(command.payload, device.payload_types):
            return CommandDisposition.REJECTED_SCHEMA, ("unsupported_payload_schema",)
        if isinstance(command.payload, (ThrusterPulseCommand, ThrusterOnOffCommand)) and (
            command.payload.thruster_id != command.actuator_id
        ):
            return CommandDisposition.REJECTED_TARGET, ("thruster_payload_target_mismatch",)
        if not device.enabled:
            return CommandDisposition.REJECTED_DEVICE_STATE, ("device_disabled",)
        if device.validator is not None:
            valid, code = device.validator(command.payload)
            if not valid:
                return CommandDisposition.REJECTED_VALUE, ((code or "invalid_payload_value"),)
        if device.interlock is not None:
            permitted, code = device.interlock(command)
            if not permitted:
                return CommandDisposition.REJECTED_INTERLOCK, ((code or "command_interlock"),)
        return CommandDisposition.ACCEPTED, ()

    def _receipt(
        self,
        command: ActuatorCommand,
        received_at: ClockTag,
        disposition: CommandDisposition,
        status: tuple[str, ...],
    ) -> ActuatorCommandReceipt | None:
        device = self._devices.get((command.satellite_id, command.actuator_id))
        if device is not None and not device.acknowledge:
            return None
        return ActuatorCommandReceipt(command.command_id, received_at, disposition, status)


def _tag_ns(tag: ClockTag) -> int:
    return tag.ticks * tag.tick_period_ns


def _same_clock_domain(left: ClockTag, right: ClockTag) -> bool:
    return (
        left.clock_id,
        left.tick_period_ns,
        left.scale,
        left.reset_counter,
    ) == (
        right.clock_id,
        right.tick_period_ns,
        right.scale,
        right.reset_counter,
    )


def _expiry_ns(command: ActuatorCommand) -> int | None:
    expiry = None if command.validity.expires_at is None else _tag_ns(command.validity.expires_at)
    if isinstance(command.payload, ThrusterPulseCommand):
        pulse_end = _tag_ns(command.payload.start_at) + int(round(command.payload.duration_s * 1.0e9))
        expiry = pulse_end if expiry is None else min(expiry, pulse_end)
    return expiry


def _effective_ns(command: ActuatorCommand) -> int:
    effective = _tag_ns(command.validity.not_before)
    if isinstance(command.payload, ThrusterPulseCommand):
        effective = max(effective, _tag_ns(command.payload.start_at))
    return effective
