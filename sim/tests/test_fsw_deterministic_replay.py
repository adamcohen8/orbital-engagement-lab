from __future__ import annotations

from sim.flight_software import AttitudeReferenceFlightSoftwareStack, canonical_json_bytes
from sim.tests.fsw_v2_helpers import attitude_config, batch, boot_event, ideal_event


def test_restore_replays_canonical_actuator_command_bytes_exactly() -> None:
    stack = AttitudeReferenceFlightSoftwareStack(attitude_config())
    stack.boot(boot_event())
    stack.step(batch(1, ideal_event(0, 1, rate=(0.6, 0.0, 0.0))))
    start = stack.snapshot()
    stream = (
        batch(2, ideal_event(1, 2, rate=(0.4, 0.1, 0.0))),
        batch(3, ideal_event(2, 3, rate=(0.2, 0.05, 0.0))),
        batch(4, ideal_event(3, 4, rate=(0.1, 0.0, 0.0))),
    )

    first = canonical_json_bytes(tuple(command for item in stream for command in stack.step(item).commands))
    stack.restore(start)
    second = canonical_json_bytes(tuple(command for item in stream for command in stack.step(item).commands))

    assert second == first
