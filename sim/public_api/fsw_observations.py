"""Read the published spacecraft boundary, without exposing plant state."""

from __future__ import annotations

from sim.flight_software.contracts import ActuatorTelemetryPayload
from sim.flight_software.schemas import to_primitive


def flight_software_observations(engine: object, object_id: str) -> dict:
    runtime = engine.agents[object_id].flight_software_runtime
    if runtime is None:
        raise ValueError(f"{object_id!r} has no flight software")
    topics = {}
    for output in reversed(runtime.evidence.outputs):
        for record in output.telemetry:
            if record.topic not in topics:
                topics[record.topic] = to_primitive(record)
    actuators = {}
    for event in reversed(runtime.evidence.input_events):
        if isinstance(event.payload, ActuatorTelemetryPayload):
            key = event.payload.actuator_id
            if key not in actuators:
                actuators[key] = to_primitive(event)
    return {"telemetry": list(topics.values()), "actuator_telemetry": list(actuators.values())}
