from __future__ import annotations

import pytest

from sim.execution.parameter_paths import set_parameter_path_value


def test_parameter_path_must_exist_in_base_config() -> None:
    root = {"simulator": {"duration_s": 10.0, "dt_s": 1.0}}

    with pytest.raises(KeyError, match="does not exist"):
        set_parameter_path_value(root, "simulator.duration", 20.0)

    assert root == {"simulator": {"duration_s": 10.0, "dt_s": 1.0}}


def test_legacy_object_parameter_path_updates_canonical_object() -> None:
    root = {"objects": {"target": {"specs": {"mass_kg": 100.0}}}}

    set_parameter_path_value(root, "target.specs.mass_kg", 120.0)

    assert root["objects"]["target"]["specs"]["mass_kg"] == 120.0


def test_object_parameter_path_syncs_existing_legacy_alias() -> None:
    root = {
        "objects": {"target": {"specs": {"mass_kg": 100.0}}},
        "target": {"specs": {"mass_kg": 100.0}},
    }

    set_parameter_path_value(root, "objects.target.specs.mass_kg", 120.0)

    assert root["objects"]["target"]["specs"]["mass_kg"] == 120.0
    assert root["target"]["specs"]["mass_kg"] == 120.0
