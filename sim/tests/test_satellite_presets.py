from __future__ import annotations

import numpy as np

from sim.app.services import get_gui_capabilities
from sim.config import scenario_config_from_dict, validate_scenario_plugins
from sim.presets.satellites import (
    ADCS_DEMO_SAT,
    CUBESAT_6U,
    ELECTRIC_PROP_SMALLSAT,
    SMALLSAT_RPO,
    TARGET_BUS_PASSIVE,
)


def test_satellite_preset_constants_have_expected_mass_scale() -> None:
    assert CUBESAT_6U.wet_mass_kg == 12.0
    assert SMALLSAT_RPO.wet_mass_kg == 200.0
    assert TARGET_BUS_PASSIVE.wet_mass_kg == 500.0
    assert ELECTRIC_PROP_SMALLSAT.wet_mass_kg == 205.0
    assert ADCS_DEMO_SAT.wet_mass_kg == 125.0
    assert np.all(np.diag(CUBESAT_6U.inertia_kg_m2) > 0.0)


def test_new_satellite_presets_are_discoverable_by_gui_capabilities() -> None:
    presets = set(get_gui_capabilities().satellite_presets)

    assert {
        "CUBESAT_6U",
        "SMALLSAT_RPO",
        "TARGET_BUS_PASSIVE",
        "ELECTRIC_PROP_SMALLSAT",
        "ADCS_DEMO_SAT",
    }.issubset(presets)


def test_satellite_object_presets_resolve_and_validate() -> None:
    cfg = scenario_config_from_dict(
        {
            "scenario_name": "satellite_preset_catalog_smoke",
            "objects": {
                "cubesat": {"enabled": True, "kind": "satellite", "preset": "cubesat_6u"},
                "rpo": {"enabled": True, "kind": "satellite", "preset": "smallsat_rpo"},
                "target": {"enabled": True, "kind": "satellite", "preset": "target_bus_passive"},
                "electric": {"enabled": True, "kind": "satellite", "preset": "electric_prop_smallsat"},
                "adcs": {"enabled": True, "kind": "satellite", "preset": "adcs_demo_sat"},
            },
            "simulator": {"duration_s": 10.0, "dt_s": 1.0},
        }
    )

    assert validate_scenario_plugins(cfg) == []
    assert cfg.objects["cubesat"].specs["preset_satellite"] == "CUBESAT_6U"
    assert cfg.objects["rpo"].specs["actuator_preset"] == "BASIC_RCS_6DOF"
    assert cfg.objects["target"].specs["attitude_system"] == "PASSIVE"
    assert cfg.objects["electric"].specs["actuator_preset"] == "BASIC_ELECTRIC_PROPULSION"
    assert cfg.objects["adcs"].specs["actuator_preset"] == "BASIC_MAGNETORQUER_TRIAD"
