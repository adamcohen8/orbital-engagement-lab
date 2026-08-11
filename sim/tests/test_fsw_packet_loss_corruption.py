from __future__ import annotations

from dataclasses import replace

from sim.flight_software import (
    ClockScale,
    ClockTag,
    ControlAxisSample,
    DataValidity,
    InputEvent,
    InputKind,
    PacketId,
    PilotInputPayload,
    Quality,
)
from sim.flight_software.delivery import PacketTransport, TransportDisposition


def _event(sequence: int) -> InputEvent:
    time = ClockTag("clock", sequence, 100, ClockScale.ONBOARD)
    return InputEvent(
        PacketId("pilot", "boot", sequence),
        InputKind.PILOT_INPUT,
        time,
        time,
        Quality(),
        PilotInputPayload("profile", (ControlAxisSample("x", 0.25),)),
    )


def _corrupt(payload: object) -> object:
    assert isinstance(payload, PilotInputPayload)
    return replace(payload, axes=(ControlAxisSample("x", -0.75),))


def test_packet_loss_is_local_and_cannot_suppress_unrelated_packets() -> None:
    lost, lost_record = PacketTransport(loss_probability=1.0).transmit(_event(1))
    delivered, delivered_record = PacketTransport().transmit(_event(2))
    assert lost is None and lost_record.disposition is TransportDisposition.LOST
    assert delivered == _event(2) and delivered_record.disposition is TransportDisposition.DELIVERED


def test_detectable_and_undetectable_corruption_have_correct_boundary_visibility() -> None:
    detectable, record = PacketTransport(detectable_corruption_probability=1.0, corrupt_payload=_corrupt).transmit(
        _event(1)
    )
    assert detectable is not None
    assert detectable.quality.validity is DataValidity.SUSPECT
    assert record.disposition is TransportDisposition.DELIVERED_DETECTABLY_CORRUPT

    undetectable, record = PacketTransport(undetectable_corruption_probability=1.0, corrupt_payload=_corrupt).transmit(
        _event(1)
    )
    assert undetectable is not None
    assert undetectable.quality == Quality()
    assert record.disposition is TransportDisposition.DELIVERED_UNDETECTABLY_CORRUPT
    assert not hasattr(undetectable, "corruption_disposition")
