from __future__ import annotations

from dataclasses import replace

from sim.actuators.command_bus import ActuatorCommandBus, ActuatorDeviceDefinition, ExpiryBehavior
from sim.flight_software import (
    ActuatorCommand,
    ClockScale,
    ClockTag,
    CommandDisposition,
    FrameId,
    IdealWrenchCommand,
    PacketId,
    ReactionWheelTorqueCommand,
    ValidityInterval,
)

FRAME = FrameId("OEL/ACTUATOR/sat/wrench", "frames-v1")


def _time(ticks: int) -> ClockTag:
    return ClockTag("clock", ticks, 1_000_000_000, ClockScale.ONBOARD)


def _command(sequence: int, *, force: float = 1.0, actuator_id: str = "wrench") -> ActuatorCommand:
    return ActuatorCommand(
        PacketId("fsw", "boot", sequence),
        "sat",
        actuator_id,
        _time(1),
        ValidityInterval(_time(1), _time(5)),
        FRAME,
        IdealWrenchCommand((force, 0.0, 0.0), (0.0, 0.0, 0.0)),
    )


def _bus(*, enabled: bool = True, interlock: bool = True) -> ActuatorCommandBus:
    return ActuatorCommandBus(
        (
            ActuatorDeviceDefinition(
                "sat",
                "wrench",
                FRAME,
                (IdealWrenchCommand,),
                ExpiryBehavior.ZERO,
                validator=lambda payload: (abs(payload.force_n[0]) <= 2.0, "force_limit"),
                interlock=lambda _command: (interlock, "blocked_mode"),
                enabled=enabled,
            ),
        )
    )


def test_acceptance_duplicate_conflict_and_old_sequence_have_exact_dispositions() -> None:
    bus = _bus()
    command = _command(2)
    assert bus.publish(command, received_at=_time(1)).disposition is CommandDisposition.ACCEPTED  # type: ignore[union-attr]
    assert bus.publish(command, received_at=_time(1)).disposition is CommandDisposition.DUPLICATE  # type: ignore[union-attr]
    conflict = replace(command, payload=IdealWrenchCommand((0.5, 0.0, 0.0), (0.0, 0.0, 0.0)))
    assert bus.publish(conflict, received_at=_time(1)).disposition is CommandDisposition.REJECTED_SEQUENCE  # type: ignore[union-attr]
    assert bus.publish(_command(1), received_at=_time(1)).disposition is CommandDisposition.REJECTED_SEQUENCE  # type: ignore[union-attr]


def test_boundary_validated_publication_preserves_command_bus_results() -> None:
    commands = (
        _command(2),
        _command(2),
        replace(_command(2), payload=IdealWrenchCommand((0.5, 0.0, 0.0), (0.0, 0.0, 0.0))),
        _command(1),
    )
    ordinary = _bus()
    validated = _bus()
    ordinary_receipts = tuple(
        ordinary.publish(command, received_at=_time(1)) for command in commands
    )
    validated_receipts = tuple(
        validated._publish(  # noqa: SLF001 - exercises the runtime-only fast path
            command,
            received_at=_time(1),
            boundary_validated=True,
        )
        for command in commands
    )

    assert validated_receipts == ordinary_receipts
    assert validated.records == ordinary.records
    assert validated.snapshot_state() == ordinary.snapshot_state()


def test_schema_version_target_frame_value_interlock_and_device_rejections_are_distinct() -> None:
    cases = (
        (replace(_command(1), schema="wrong"), _bus(), CommandDisposition.REJECTED_SCHEMA),
        (replace(_command(1), contract_version="9.0"), _bus(), CommandDisposition.REJECTED_VERSION),
        (replace(_command(1), actuator_id="missing"), _bus(), CommandDisposition.REJECTED_TARGET),
        (replace(_command(1), frame=FrameId("wrong", "frames-v1")), _bus(), CommandDisposition.REJECTED_FRAME),
        (_command(1, force=3.0), _bus(), CommandDisposition.REJECTED_VALUE),
        (_command(1), _bus(interlock=False), CommandDisposition.REJECTED_INTERLOCK),
        (_command(1), _bus(enabled=False), CommandDisposition.REJECTED_DEVICE_STATE),
        (
            replace(_command(1), payload=ReactionWheelTorqueCommand((0.1, 0.0, 0.0))),
            _bus(),
            CommandDisposition.REJECTED_SCHEMA,
        ),
    )
    for command, bus, expected in cases:
        receipt = bus.publish(command, received_at=_time(1))
        assert receipt is not None and receipt.disposition is expected


def test_rejection_never_replaces_the_prior_accepted_demand() -> None:
    bus = _bus()
    accepted = _command(1, force=1.0)
    bus.publish(accepted, received_at=_time(1))
    bus.publish(_command(2, force=3.0), received_at=_time(2))
    demand = bus.demand(satellite_id="sat", actuator_id="wrench", at=_time(2))
    assert demand.source_command is accepted
