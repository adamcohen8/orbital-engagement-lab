from __future__ import annotations

import re
from pathlib import Path

from sim.config import scenario_config_from_dict
from sim.gnc import catalog_entries

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_catalog_ids_metadata_and_evidence_paths_follow_governance_contract() -> None:
    pattern = re.compile(r"^(orbit|attitude|mission|execution|module)\.[a-z0-9_]+$")
    for entry in catalog_entries():
        assert pattern.fullmatch(entry.builtin_id), entry.builtin_id
        assert entry.display_name and entry.summary
        assert entry.packaging in {"public", "pro", "private"}
        for path in (*entry.examples, *entry.tests):
            assert (REPO_ROOT / path).exists(), (entry.builtin_id, path)


def test_all_reference_and_flagship_catalog_surfaces_are_in_product_inventory() -> None:
    inventory = (REPO_ROOT / "docs/product-inventory.md").read_text(encoding="utf-8")
    for entry in catalog_entries(include_internal=False):
        if entry.maturity in {"reference", "flagship"}:
            assert entry.class_name in inventory, entry.builtin_id


def test_builtin_pointer_round_trips_without_mixing_resolved_import_fields() -> None:
    raw = {
        "scenario_name": "builtin_round_trip",
        "objects": {
            "target": {
                "enabled": True,
                "kind": "satellite",
                "specs": {"mass_kg": 100.0},
                "initial_state": {
                    "position_eci_km": [7000.0, 0.0, 0.0],
                    "velocity_eci_km_s": [0.0, 7.5, 0.0],
                },
                "flight_software": {
                    "stack": "fsw.passive",
                    "hardware_profile": "hardware.passive.v1",
                    "params": {},
                },
            }
        },
        "simulator": {"duration_s": 1.0, "dt_s": 1.0, "dynamics": {"attitude": {"enabled": False}}},
    }
    first = scenario_config_from_dict(raw)
    serialized = first.to_dict()
    pointer = serialized["objects"]["target"]["flight_software"]
    assert pointer["stack"] == "fsw.passive"
    assert pointer["hardware_profile"] == "hardware.passive.v1"
    second = scenario_config_from_dict(serialized)
    assert second.objects["target"].flight_software == first.objects["target"].flight_software
