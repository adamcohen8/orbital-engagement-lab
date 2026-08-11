"""Packetization of independently sampled physical sensor observations."""

from __future__ import annotations

from dataclasses import dataclass

from sim.flight_software.contracts import (
    ClockTag,
    InputEvent,
    InputKind,
    MeasurementEvent,
    PacketId,
    Quality,
)


@dataclass(slots=True)
class PhysicalSensorEventSource:
    """Create boundary packets without exposing how a physical sample was made."""

    sensor_id: str
    boot_id: str
    next_sequence: int = 0

    def __post_init__(self) -> None:
        if not self.sensor_id.strip() or not self.boot_id.strip():
            raise ValueError("sensor_id and boot_id must be non-empty")
        if self.next_sequence < 0:
            raise ValueError("next_sequence must be nonnegative")

    def packetize(
        self,
        measurement: MeasurementEvent | None,
        *,
        source_time: ClockTag,
        delivery_time: ClockTag,
        quality: Quality | None = None,
    ) -> InputEvent | None:
        sequence = self.next_sequence
        self.next_sequence += 1
        if measurement is None:
            return None
        if measurement.sensor_id != self.sensor_id:
            raise ValueError("measurement sensor_id does not match event source")
        if measurement.sample_time != source_time:
            raise ValueError("measurement sample_time must match packet source_time")
        return InputEvent(
            packet_id=PacketId(self.sensor_id, self.boot_id, sequence),
            kind=InputKind.MEASUREMENT,
            source_time=source_time,
            delivery_time=delivery_time,
            quality=Quality() if quality is None else quality,
            payload=measurement,
        )
