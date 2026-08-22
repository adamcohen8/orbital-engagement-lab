from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from sim.control.orbit.predictive_engagement import (
    select_evasion_action,
    select_intercept_action,
)
from sim.flight_software import RpoReferenceFlightSoftwareStack
from sim.gnc.orbit_v2 import TranslationControlConfig, TranslationMode
from sim.tests.fsw_v2_helpers import boot_event
from sim.tests.fsw_v2_orbit_helpers import navigation_batch, rpo_config, telemetry_fields

PREDICTION = {
    "mean_motion_rad_s": 0.001078,
    "horizon_s": 600.0,
    "step_s": 10.0,
    "pulse_duration_s": 20.0,
    "capture_radius_m": 100.0,
    "capture_margin_m": 20.0,
    "acceleration_fractions": (0.5, 1.0),
}


def test_interceptor_coasts_when_passive_motion_already_captures() -> None:
    decision = select_intercept_action(
        np.array((200.0, 0.0, 0.0, -0.5, 0.0, 0.0)),
        max_acceleration_m_s2=0.01,
        **PREDICTION,
    )

    assert decision.acceleration_ric_m_s2 == (0.0, 0.0, 0.0)
    assert decision.predicted_capture_time_s is not None
    assert decision.phase in {"passive_intercept_coast", "intercept_coast"}


def test_interceptor_burns_when_coast_does_not_close_the_range() -> None:
    decision = select_intercept_action(
        np.array((0.0, -1_000.0, 0.0, 0.0, 0.0, 0.0)),
        max_acceleration_m_s2=0.01,
        **PREDICTION,
    )

    assert np.linalg.norm(decision.acceleration_ric_m_s2) > 0.0
    assert "burn" in decision.phase


def test_evasion_is_deterministic_and_accounts_for_bounded_pursuit_response() -> None:
    state = np.array((0.0, 1_000.0, 0.0, 0.0, 0.0, 0.0))
    first = select_evasion_action(
        state,
        max_acceleration_m_s2=0.006,
        opponent_max_acceleration_m_s2=0.01,
        **PREDICTION,
    )
    second = select_evasion_action(
        state,
        max_acceleration_m_s2=0.006,
        opponent_max_acceleration_m_s2=0.01,
        **PREDICTION,
    )

    assert first == second
    assert np.linalg.norm(first.acceleration_ric_m_s2) > 0.0
    assert first.phase == "predictive_evasion_burn"


def test_predictive_controller_checkpoint_state_round_trips() -> None:
    config = rpo_config(TranslationMode.INTERCEPT_COAST)
    stack = RpoReferenceFlightSoftwareStack(config)
    stack.boot(boot_event())
    stack.step(navigation_batch(1, range_m=1_000.0))
    snapshot = stack.snapshot()
    controller_state = json.loads(snapshot.state_bytes)["stack_state"]["controller"]
    assert controller_state["predictive_action_until_ns"] is not None
    assert np.linalg.norm(controller_state["predictive_action_ric_m_s2"]) > 0.0

    restored = RpoReferenceFlightSoftwareStack(config)
    restored.boot(boot_event())
    restored.restore(snapshot)

    assert restored.snapshot().state_bytes == snapshot.state_bytes


def test_predictive_pulse_coasts_until_the_configured_replan_time() -> None:
    base = rpo_config(TranslationMode.INTERCEPT_COAST)
    control = replace(
        base.control,
        prediction_horizon_s=10.0,
        prediction_step_s=1.0,
        prediction_decision_interval_s=4.0,
        prediction_pulse_duration_s=2.0,
    )
    stack = RpoReferenceFlightSoftwareStack(replace(base, control=control))
    stack.boot(boot_event())

    first = stack.step(navigation_batch(1, range_m=1_000.0))
    for tick in range(2, 21):
        stack.step(navigation_batch(tick, range_m=1_000.0))
    pulse_end = stack.step(navigation_batch(21, range_m=1_000.0))
    for tick in range(22, 41):
        stack.step(navigation_batch(tick, range_m=1_000.0))
    replanned = stack.step(navigation_batch(41, range_m=1_000.0))

    assert "burn" in str(telemetry_fields(first)["control_phase"])
    assert telemetry_fields(pulse_end)["control_phase"] == "intercept_replan_coast"
    assert "burn" in str(telemetry_fields(replanned)["control_phase"])


def test_prediction_timing_rejects_pulse_longer_than_decision_interval() -> None:
    with pytest.raises(ValueError, match="pulse_duration_s"):
        TranslationControlConfig(
            TranslationMode.INTERCEPT_COAST,
            100.0,
            0.01,
            prediction_decision_interval_s=30.0,
            prediction_pulse_duration_s=60.0,
        )


def test_prediction_work_budget_rejects_unbounded_search() -> None:
    with pytest.raises(ValueError, match="propagation steps"):
        select_intercept_action(
            np.array((1_000.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
            mean_motion_rad_s=0.001,
            max_acceleration_m_s2=0.01,
            horizon_s=1.0e9,
            step_s=1.0,
            pulse_duration_s=10.0,
            capture_radius_m=100.0,
        )
