from __future__ import annotations

from sim.flight_software import (
    ClockScale,
    ClockTag,
    FlightSoftwareInputBatch,
    FrameId,
    GyroMeasurement,
    InputEvent,
    InputKind,
    MeasurementEvent,
    PacketId,
    PilotInputPayload,
    Quality,
)
from sim.flight_software.delivery import InputDeliveryQueue
from sim.sensors.event_source import PhysicalSensorEventSource


def _event(source: str, sequence: int, delivery_tick: int) -> InputEvent:
    source_time = ClockTag("clock", 0, 100, ClockScale.ONBOARD)
    delivery_time = ClockTag("clock", delivery_tick, 100, ClockScale.ONBOARD)
    return InputEvent(
        PacketId(source, "boot", sequence),
        InputKind.PILOT_INPUT,
        source_time,
        delivery_time,
        Quality(),
        PilotInputPayload("profile"),
    )


def test_delivery_queue_resolves_modeled_order_then_packet_identity() -> None:
    queue = InputDeliveryQueue()
    queue.enqueue(_event("z", 1, 5))
    queue.enqueue(_event("a", 2, 5))
    queue.enqueue(_event("explicit", 8, 5), transport_order=0)
    assert queue.deliver_due(499) == ()
    delivered = queue.deliver_due(500)
    assert [(event.packet_id.source_id, event.packet_id.sequence) for event in delivered] == [
        ("explicit", 8),
        ("a", 2),
        ("z", 1),
    ]


def test_empty_task_release_batch_is_valid_and_does_not_invent_inputs() -> None:
    invocation_time = ClockTag("clock", 10, 100, ClockScale.ONBOARD)
    batch = FlightSoftwareInputBatch("sat", 3, invocation_time, ())
    assert batch.events == ()


def test_missing_sensor_sample_does_not_suppress_another_sensor_packet() -> None:
    time = ClockTag("clock", 1, 100, ClockScale.ONBOARD)
    missing_source = PhysicalSensorEventSource("missing", "boot")
    gyro_source = PhysicalSensorEventSource("gyro", "boot")
    assert missing_source.packetize(None, source_time=time, delivery_time=time) is None
    measurement = MeasurementEvent(
        "gyro",
        "gyro.v1",
        time,
        FrameId("OEL/SENSOR/sat/gyro", "v1"),
        GyroMeasurement((0.0, 0.0, 0.0)),
    )
    gyro_event = gyro_source.packetize(measurement, source_time=time, delivery_time=time)
    assert gyro_event is not None
    queue = InputDeliveryQueue()
    queue.enqueue(gyro_event)
    assert queue.deliver_due(100) == (gyro_event,)
