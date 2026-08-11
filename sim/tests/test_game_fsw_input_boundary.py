from __future__ import annotations

import ast
from pathlib import Path

from sim.flight_software import (
    GroundCommandKind,
    GroundCommandPayload,
    TelemetryField,
    canonical_json_bytes,
    canonical_loads,
)


def test_ground_command_payload_has_a_canonical_golden_round_trip() -> None:
    payload = GroundCommandPayload(
        "burn-1",
        GroundCommandKind.ACTION_REQUEST,
        (TelemetryField("delta_v_r_m_s", 0.5, "m/s"),),
    )
    encoded = canonical_json_bytes(payload)
    assert canonical_json_bytes(canonical_loads(encoded)) == encoded


def test_game_fsw_and_input_adapters_have_no_simulator_truth_imports() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative in ("flight_software/game_stacks.py", "game/fsw_inputs.py"):
        source = (root / relative).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        assert "sim.core.models" not in imports
        assert "StateTruth" not in source
        assert "world_truth" not in source
