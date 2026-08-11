from __future__ import annotations

import pytest

from sim.config import scenario_config_from_dict
from sim.gnc import catalog_entries, catalog_entry, validate_catalog


def _config(orbit_control: dict) -> dict:
    return {
        "scenario_name": "gnc_builtin_validation",
        "objects": {
            "target": {
                "enabled": True,
                "kind": "satellite",
                "specs": {"mass_kg": 100.0},
                "initial_state": {
                    "position_eci_km": [7000.0, 0.0, 0.0],
                    "velocity_eci_km_s": [0.0, 7.5, 0.0],
                },
                "orbit_control": orbit_control,
            }
        },
        "simulator": {
            "duration_s": 1.0,
            "dt_s": 1.0,
            "dynamics": {"attitude": {"enabled": False}},
        },
    }


def test_catalog_identities_are_unique_and_importable() -> None:
    assert len(catalog_entries()) >= 70
    assert validate_catalog() == []


def test_builtin_pointer_resolves_to_stable_implementation() -> None:
    entry = catalog_entry("orbit.zero")
    assert entry.module == "sim.control.orbit.zero_controller"
    assert entry.class_name == "ZeroController"


def test_builtin_alias_resolves_to_canonical_entry() -> None:
    assert catalog_entry("orbit.hcw_pd") == catalog_entry("orbit.ric_pd_hold")


def test_builtin_constructor_errors_are_validation_errors() -> None:
    with pytest.raises(ValueError, match="removed GNC v1 satellite field"):
        scenario_config_from_dict(
            _config({"builtin": "orbit.hcw_lqr", "params": {"max_accel_km_s2": 1.0e-5}})
        )


def test_builtin_cannot_conflict_with_raw_pointer_fields() -> None:
    data = _config(
        {
            "builtin": "orbit.zero",
            "module": "sim.control.orbit.zero_controller",
            "class_name": "ZeroController",
        }
    )

    try:
        scenario_config_from_dict(data)
    except ValueError as exc:
        assert "removed GNC v1 satellite field" in str(exc)
    else:
        raise AssertionError("conflicting built-in and raw pointer fields were accepted")
