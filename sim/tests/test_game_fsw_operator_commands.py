from __future__ import annotations

import numpy as np
import pytest

from sim.api import SimulationConfig
from sim.flight_software import FlightSoftwareInputBatch, GamePilotMode, GamePilotReferenceFlightSoftwareStack
from sim.game.attempt_lifecycle import _start_game_attempt
from sim.game.fsw_inputs import GameOperatorInputAdapter
from sim.game.manual import KeyboardCommandState
from sim.game.operator import OperatorBurn, OperatorBurnPlan
from sim.game.training import RPOTrainingConfig
from sim.tests.fsw_v2_helpers import SATELLITE_ID, boot_event, clock, ideal_event
from sim.tests.game_fsw_v2_helpers import game_stack_config


def test_operator_burn_plan_becomes_typed_ground_action_and_device_command() -> None:
    adapter = GameOperatorInputAdapter(source_id="operator", boot_id="boot-wp4")
    event = adapter.scheduled_burn_events(
        (OperatorBurn(0.2, (0.5, 0.0, 0.0)),),
        clock_id="sat/onboard",
        tick_period_ns=100_000_000,
    )[0]
    stack = GamePilotReferenceFlightSoftwareStack(game_stack_config(GamePilotMode.TRANSLATION))
    stack.boot(boot_event())
    stack.step(FlightSoftwareInputBatch(SATELLITE_ID, 1, clock(1), (ideal_event(0, 1),)))
    output = stack.step(FlightSoftwareInputBatch(SATELLITE_ID, 2, clock(2), (ideal_event(1, 2), event)))
    assert output.commands
    fields = {field.name: field.value for field in output.telemetry[0].fields}
    assert fields["last_ground_command_id"] == "operator-burn-0"


@pytest.mark.parametrize(
    ("burn_time_s", "delta_v_m_s"),
    (
        (0.0, 0.1),
        (0.2, 0.001),
        (0.2, 0.5),
    ),
)
def test_operator_burn_realizes_exact_impulse_at_time_zero_and_between_tasks(
    burn_time_s: float,
    delta_v_m_s: float,
) -> None:
    config = SimulationConfig.from_yaml("sim/game/configs/game_training_rpo_00_tutorial.yaml")
    training = RPOTrainingConfig.from_metadata(dict(config.scenario.metadata))
    plan = OperatorBurnPlan((OperatorBurn(burn_time_s, (delta_v_m_s, 0.0, 0.0)),))
    session, _, _ = _start_game_attempt(
        config,
        command_state=KeyboardCommandState(),
        training_cfg=training,
        controlled_object_id="chaser",
        attitude_rate_deg_s=45.0,
        control_mode="ric_translation",
        ric_reference_object_id="target",
        operator_burn_plan=plan,
    )

    snapshot = session.step(dt_s=1.0)
    realized_delta_v_m_s = float(np.linalg.norm(snapshot.applied_thrust["chaser"])) * 1.0e3
    runtime = session._engine.agents["chaser"].flight_software_runtime
    active = [
        item
        for item in runtime.evidence.realizations
        if np.linalg.norm(np.asarray(item.realized_force_n, dtype=float)) > 0.0
    ]

    assert realized_delta_v_m_s == pytest.approx(delta_v_m_s, rel=1.0e-12, abs=1.0e-12)
    assert active
    assert active[0].interval_start_ns == int(round(burn_time_s * 1.0e9))
    assert active[0].interval_end_ns - active[0].interval_start_ns == 1_000_000
    assert active[0].saturated is False


def test_operator_actuator_error_is_applied_to_typed_delta_v_command() -> None:
    adapter = GameOperatorInputAdapter(source_id="operator", boot_id="boot-error")
    event = adapter.scheduled_burn_events(
        (OperatorBurn(1.0, (1.0, -2.0, 0.5)),),
        clock_id="sat/onboard",
        tick_period_ns=1,
        actuator_error_fraction=0.1,
    )[0]
    fields = {field.name: field.value for field in event.payload.parameters}

    assert fields["delta_v_r_m_s"] == pytest.approx(1.1)
    assert fields["delta_v_i_m_s"] == pytest.approx(-2.2)
    assert fields["delta_v_c_m_s"] == pytest.approx(0.55)
    assert fields["planned_delta_v_r_m_s"] == pytest.approx(1.0)
    assert fields["impulse_duration_s"] == pytest.approx(1.0e-3)


def test_operator_adapter_rejects_nonfinite_delta_v_components() -> None:
    adapter = GameOperatorInputAdapter(source_id="operator", boot_id="boot-invalid")

    with pytest.raises(ValueError, match="three finite RIC components"):
        adapter.scheduled_burn_events(
            (OperatorBurn(1.0, (float("nan"), 0.0, 0.0)),),
            clock_id="sat/onboard",
            tick_period_ns=1,
        )
