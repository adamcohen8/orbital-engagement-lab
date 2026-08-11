from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from sim.core.models import Command
from sim.gnc.contracts import build_command_decision, merge_mission_intent_layers


def test_mission_intent_precedence_and_collisions_are_explicit() -> None:
    envelope = merge_mission_intent_layers(
        (
            ("mission_modules", {"phase": "module", "shared": 1}),
            ("mission_strategy", {"phase": "strategy", "shared": 2}),
            ("external_intent", {"shared": 3}),
            ("mission_execution", {"shared": 4, "burn": True}),
        )
    )

    runtime = envelope.to_runtime_dict()
    assert runtime["phase"] == "strategy"
    assert runtime["shared"] == 4
    assert runtime["_mission_field_sources"]["shared"] == "mission_execution"
    assert [item["winning_source"] for item in runtime["_mission_field_collisions"] if item["field"] == "shared"] == [
        "mission_strategy",
        "external_intent",
        "mission_execution",
    ]
    assert runtime["_mission_precedence"] == ["mission_modules", "mission_strategy", "mission_execution"]


def test_contract_metadata_cannot_be_injected_by_plugin_layer() -> None:
    envelope = merge_mission_intent_layers(
        (("mission_strategy", {"value": 2, "_mission_precedence": ["plugin"]}),)
    )
    assert envelope.to_runtime_dict()["_mission_precedence"] != ["plugin"]


def test_compact_command_decision_records_suppressed_burn_reason() -> None:
    raw = Command(thrust_eci_km_s2=np.array([1.0e-5, 0.0, 0.0]))
    applied = Command.zero()
    applied.mode_flags.update({"fuel_depleted": True, "orbit_controller_deadline_missed": False})
    agent = SimpleNamespace(
        orbit_controller=SimpleNamespace(),
        attitude_controller=None,
        mission_strategy=None,
        mission_execution=None,
    )

    row = build_command_decision(
        sample_index=3,
        time_s=4.0,
        interval_end_time_s=5.0,
        dt_s=1.0,
        object_id="vehicle",
        agent=agent,
        mission_intent={"mission_mode": {"phase": "burn"}},
        command_raw=raw,
        command_applied=applied,
    )

    assert row.burn_requested is True
    assert row.burn_applied is False
    assert row.fuel_depleted is True
    assert row.gate_reason == "fuel_depleted"
    assert row.mission_phase == "burn"
