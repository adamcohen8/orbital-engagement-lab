from __future__ import annotations

import subprocess
import sys

import pytest

from sim.flight_software import (
    BUILTIN_STACKS,
    FlightSoftwareInputBatch,
    PassiveFlightSoftwareStack,
    PassiveStackConfig,
    SatelliteFlightSoftware,
    ShutdownEvent,
    build_builtin_stack,
)
from sim.tests.fsw_v2_helpers import BODY_FRAME, INERTIAL_FRAME, SATELLITE_ID, batch, boot_event, clock, ideal_event


def test_builtin_stack_shelf_exposes_all_implemented_complete_stacks() -> None:
    assert [(entry.stack_id, entry.maturity.value) for entry in BUILTIN_STACKS] == [
        ("fsw.passive", "experimental"),
        ("fsw.attitude_reference", "experimental"),
        ("fsw.orbit_reference", "experimental"),
        ("fsw.rpo_reference", "experimental"),
        ("fsw.low_thrust_reference", "experimental"),
        ("fsw.game_pilot_reference", "experimental"),
    ]


@pytest.mark.parametrize(
    "imports",
    (
        "import sim.gnc; import sim.flight_software; from sim.flight_software import PassiveFlightSoftwareStack",
        "import sim.flight_software; import sim.gnc; from sim.gnc import AttitudeNavigator",
    ),
)
def test_public_facade_import_order_is_stable(imports: str) -> None:
    result = subprocess.run([sys.executable, "-c", imports], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_passive_stack_obeys_complete_lifecycle_and_emits_no_commands() -> None:
    stack = build_builtin_stack(PassiveStackConfig(SATELLITE_ID))
    assert isinstance(stack, PassiveFlightSoftwareStack)
    assert isinstance(stack, SatelliteFlightSoftware)
    assert stack.identity.stack_id == "fsw.passive"
    assert stack.identity.checkpointable

    with pytest.raises(RuntimeError, match="not booted"):
        stack.step(batch(1))
    stack.boot(boot_event())
    with pytest.raises(RuntimeError, match="exactly once"):
        stack.boot(boot_event())

    output = stack.step(batch(1, ideal_event(0, 1)))
    assert output.satellite_id == SATELLITE_ID
    assert output.invocation_id == 1
    assert output.commands == ()
    assert output.telemetry[0].topic == "fsw.passive.status"
    with pytest.raises(ValueError, match="increase monotonically"):
        stack.step(batch(1))

    stack.shutdown(ShutdownEvent(SATELLITE_ID, clock(2), "test complete"))
    with pytest.raises(RuntimeError, match="not booted"):
        stack.step(batch(2))
    with pytest.raises(RuntimeError, match="not booted"):
        stack.shutdown(ShutdownEvent(SATELLITE_ID, clock(3), "twice"))


def test_lifecycle_rejects_cross_satellite_events() -> None:
    stack = PassiveFlightSoftwareStack(PassiveStackConfig(SATELLITE_ID))
    wrong = FlightSoftwareInputBatch("other", 1, clock(1))
    with pytest.raises(ValueError, match="configured for satellite"):
        stack.boot(type(boot_event())("other", "boot", clock(0)))
    stack.boot(boot_event())
    with pytest.raises(ValueError, match="configured for satellite"):
        stack.step(wrong)


def test_passive_stack_can_run_optional_ideal_navigation_without_commanding() -> None:
    stack = PassiveFlightSoftwareStack(
        PassiveStackConfig(
            SATELLITE_ID,
            ideal_navigation=True,
            body_frame=BODY_FRAME,
            inertial_frame=INERTIAL_FRAME,
        )
    )
    stack.boot(boot_event())
    output = stack.step(batch(1, ideal_event(0, 1)))
    assert output.commands == ()
    fields = {field.name: field.value for field in output.telemetry[0].fields}
    assert fields["navigation_valid"] is True
