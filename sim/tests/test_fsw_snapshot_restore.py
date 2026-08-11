from __future__ import annotations

from dataclasses import replace

import pytest

from sim.flight_software import (
    AttitudeReferenceFlightSoftwareStack,
    RpoReferenceFlightSoftwareStack,
    canonical_json_bytes,
)
from sim.gnc.orbit_v2 import TranslationMode
from sim.tests.fsw_v2_helpers import attitude_config, batch, boot_event, ideal_event
from sim.tests.fsw_v2_orbit_helpers import navigation_batch, rpo_config


def test_attitude_stack_snapshot_restores_all_command_relevant_state() -> None:
    stack = AttitudeReferenceFlightSoftwareStack(attitude_config())
    stack.boot(boot_event())
    stack.step(batch(1, ideal_event(0, 1, rate=(0.8, -0.2, 0.1))))
    snapshot = stack.snapshot()

    expected = stack.step(batch(2, ideal_event(1, 2, rate=(0.4, -0.1, 0.05))))
    stack.restore(snapshot)
    replay = stack.step(batch(2, ideal_event(1, 2, rate=(0.4, -0.1, 0.05))))

    assert canonical_json_bytes(replay.commands) == canonical_json_bytes(expected.commands)
    assert replay.commands[0].command_id == expected.commands[0].command_id


def test_rpo_transfer_snapshot_restores_guidance_phase_and_commands() -> None:
    stack = RpoReferenceFlightSoftwareStack(rpo_config(TranslationMode.RIC_PD_TRANSFER))
    stack.boot(boot_event())
    stack.step(navigation_batch(1, range_m=10_000.0))
    snapshot = stack.snapshot()

    expected = stack.step(navigation_batch(2, range_m=9_999.0))
    stack.restore(snapshot)
    replay = stack.step(navigation_batch(2, range_m=9_999.0))

    assert canonical_json_bytes(replay.commands) == canonical_json_bytes(expected.commands)
    assert canonical_json_bytes(replay.telemetry) == canonical_json_bytes(expected.telemetry)


def test_restore_rejects_wrong_stack_version_boot_and_configuration_before_mutation() -> None:
    stack = AttitudeReferenceFlightSoftwareStack(attitude_config())
    stack.boot(boot_event())
    snapshot = stack.snapshot()

    with pytest.raises(ValueError, match="stack identity"):
        stack.restore(replace(snapshot, stack_version="9.0.0"))
    with pytest.raises(ValueError, match="boot_id"):
        stack.restore(replace(snapshot, boot_id="other-boot"))

    different = AttitudeReferenceFlightSoftwareStack(
        attitude_config(controller=replace(attitude_config().controller, kp=(0.5, 0.5, 0.5)))
    )
    different.boot(boot_event())
    with pytest.raises(ValueError, match="configuration hash"):
        different.restore(snapshot)

    # A rejected restore leaves the receiver usable at its original state.
    assert different.step(batch(1, ideal_event(0, 1))).invocation_id == 1
