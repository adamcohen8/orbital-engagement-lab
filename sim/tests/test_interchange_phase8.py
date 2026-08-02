from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import yaml

from sim.execution import run_simulation_config_file
from sim.handoff import compare_handoff
from sim.interchange.cli import main as handoff_main
from sim.interchange.materialization import materialize_onp

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "sim" / "interchange" / "examples"
PRODUCT = EXAMPLES / "state_estimate_accepted_current.json"


def _materialize(tmp_path: Path) -> dict:
    return materialize_onp(
        PRODUCT,
        scenario_name="phase8_parity",
        scenario_path=tmp_path / "continuation.yaml",
        output_dir=tmp_path / "run",
        duration_s=2.0,
        dt_s=1.0,
        trust_plugins=True,
    )


def test_handoff_comparison_packet_proves_materialized_and_executed_state_parity(tmp_path: Path) -> None:
    materialized = _materialize(tmp_path)
    packet_path = tmp_path / "materialized_comparison.json"
    packet = compare_handoff(
        PRODUCT,
        materialized["scenario_path"],
        output_path=packet_path,
    )

    assert packet["status"] == "equivalent"
    assert packet["summary"]["failed_count"] == 0
    assert packet["execution_evidence"] == {"status": "not_supplied", "execution_occurred": False}
    assert packet_path.is_file()
    assert packet["materialization"]["execution_occurred"] is False

    run_simulation_config_file(materialized["scenario_path"])
    executed = compare_handoff(
        PRODUCT,
        materialized["scenario_path"],
        run_output_dir=tmp_path / "run",
        output_path=tmp_path / "executed_comparison.json",
    )
    assert executed["status"] == "equivalent"
    assert executed["execution_evidence"]["status"] == "compared"
    assert executed["execution_evidence"]["execution_occurred"] is True
    assert any(item["check_id"] == "execution.initial_absolute_state" for item in executed["checks"])

    schema = json.loads(
        (ROOT / "sim" / "interchange" / "schemas" / "oel-handoff-comparison-v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.validate(executed, schema)


def test_handoff_comparison_fails_on_semantic_drift(tmp_path: Path) -> None:
    materialized = _materialize(tmp_path)
    scenario_path = Path(materialized["scenario_path"])
    scenario = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    scenario["objects"]["example_satellite"]["initial_state"]["position_eci_km"][0] += 1.0
    scenario_path.write_text(yaml.safe_dump(scenario, sort_keys=False), encoding="utf-8")

    packet = compare_handoff(PRODUCT, scenario_path)

    assert packet["status"] == "failed"
    assert {"manifest.scenario_digest", "state.position_eci_km"}.issubset(
        set(packet["summary"]["failed_check_ids"])
    )


def test_compare_handoff_cli_writes_packet_and_never_executes(tmp_path: Path, capsys) -> None:
    materialized = _materialize(tmp_path)
    packet_path = tmp_path / "cli_comparison.json"
    code = handoff_main(
        [
            "compare-handoff",
            "--product",
            str(PRODUCT),
            "--scenario",
            materialized["scenario_path"],
            "--output",
            str(packet_path),
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["status"] == "equivalent"
    assert payload["execution_evidence"]["execution_occurred"] is False
    assert packet_path.is_file()
    assert (tmp_path / "run").exists() is False


def test_phase8_schema_and_facade_are_public_and_closed() -> None:
    schema = json.loads(
        (ROOT / "sim" / "interchange" / "schemas" / "oel-handoff-comparison-v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    assert schema["properties"]["schema_id"]["const"] == "oel-handoff-comparison-v1"
    assert schema["additionalProperties"] is False
    assert schema["properties"]["checks"]["items"]["additionalProperties"] is False
