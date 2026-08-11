from __future__ import annotations

import numpy as np

from sim.flight_software import (
    AttitudeReferenceFlightSoftwareStack,
    InputEvent,
    InputKind,
    ModeledFaultIndicationPayload,
    PacketId,
    Quality,
    canonical_json_bytes,
    canonical_loads,
)
from sim.tests.fsw_v2_helpers import BOOT_ID, attitude_config, batch, boot_event, clock, ideal_event


def _fault(sequence: int, tick: int, component: str, *, active: bool) -> InputEvent:
    time = clock(tick)
    return InputEvent(
        PacketId("fault-monitor", BOOT_ID, sequence),
        InputKind.MODELED_FAULT_INDICATION,
        time,
        time,
        Quality(),
        ModeledFaultIndicationPayload(component, "modeled_failure", active, "fault-monitor"),
    )


def test_modeled_fault_indication_has_a_canonical_schema_round_trip() -> None:
    event = _fault(0, 1, "gyro", active=True)
    assert canonical_loads(canonical_json_bytes(event)) == event


def test_sensor_fault_indication_holds_last_accepted_measurement_until_cleared() -> None:
    stack = AttitudeReferenceFlightSoftwareStack(attitude_config())
    stack.boot(boot_event())
    first = stack.step(batch(1, ideal_event(0, 1, rate=(0.2, 0.0, 0.0))))
    held = stack.step(
        batch(
            2,
            _fault(0, 2, "ideal-own-state", active=True),
            ideal_event(1, 2, rate=(0.0, 0.0, 0.0)),
        )
    )
    cleared = stack.step(
        batch(
            3,
            _fault(1, 3, "ideal-own-state", active=False),
            ideal_event(2, 3, rate=(0.0, 0.0, 0.0)),
        )
    )
    first_torque = np.linalg.norm(first.commands[0].payload.torque_n_m)  # type: ignore[union-attr]
    held_torque = np.linalg.norm(held.commands[0].payload.torque_n_m)  # type: ignore[union-attr]
    cleared_torque = np.linalg.norm(cleared.commands[0].payload.torque_n_m)  # type: ignore[union-attr]
    assert held_torque >= first_torque
    assert cleared_torque < held_torque / 10.0


def test_actuator_fault_indication_suppresses_publication_and_clear_resumes() -> None:
    stack = AttitudeReferenceFlightSoftwareStack(attitude_config())
    stack.boot(boot_event())
    failed = stack.step(
        batch(
            1,
            ideal_event(0, 1, rate=(0.2, 0.0, 0.0)),
            _fault(0, 1, "attitude-actuator", active=True),
        )
    )
    recovered = stack.step(
        batch(
            2,
            _fault(1, 2, "attitude-actuator", active=False),
            ideal_event(1, 2, rate=(0.2, 0.0, 0.0)),
        )
    )
    assert failed.commands == ()
    assert len(recovered.commands) == 1
