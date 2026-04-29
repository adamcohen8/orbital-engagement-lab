from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sim import SimulationConfig, SimulationSession
from sim.config import GroundStationSection, scenario_config_from_dict
from sim.ground_stations import evaluate_ground_station_access


def _ground_station_config(output_dir: Path) -> dict:
    return {
        "scenario_name": "ground_station_access_smoke",
        "rocket": {"enabled": False},
        "target": {
            "enabled": True,
            "specs": {"mass_kg": 100.0},
            "initial_state": {
                "position_eci_km": [7000.0, 0.0, 0.0],
                "velocity_eci_km_s": [0.0, 7.5, 0.0],
            },
        },
        "chaser": {"enabled": False},
        "ground_stations": [
            {
                "id": "equator_prime",
                "lat_deg": 0.0,
                "lon_deg": 0.0,
                "alt_km": 0.0,
                "min_elevation_deg": 10.0,
                "max_range_km": 1000.0,
            }
        ],
        "simulator": {
            "duration_s": 1.0,
            "dt_s": 1.0,
            "termination": {"earth_impact_enabled": False},
            "dynamics": {"attitude": {"enabled": False}},
        },
        "outputs": {
            "output_dir": str(output_dir),
            "mode": "save",
            "stats": {"print_summary": False, "save_json": True, "save_full_log": True},
            "plots": {"enabled": False, "figure_ids": []},
            "animations": {"enabled": False, "types": []},
        },
    }


def test_ground_station_yaml_parses_list_and_mapping_forms(tmp_path: Path) -> None:
    cfg = scenario_config_from_dict(_ground_station_config(tmp_path))
    assert len(cfg.ground_stations) == 1
    assert cfg.ground_stations[0].id == "equator_prime"
    assert cfg.ground_stations[0].max_range_km == 1000.0

    mapped = _ground_station_config(tmp_path)
    mapped["ground_stations"] = {
        "hawaii": {
            "lat_deg": 19.8,
            "lon_deg": -155.5,
            "altitude_km": 4.2,
            "min_elevation_deg": 5.0,
        }
    }
    cfg2 = scenario_config_from_dict(mapped)
    assert cfg2.ground_stations[0].id == "hawaii"
    assert cfg2.ground_stations[0].alt_km == 4.2


def test_ground_station_yaml_rejects_invalid_values(tmp_path: Path) -> None:
    cfg = _ground_station_config(tmp_path)
    cfg["ground_stations"][0]["lat_deg"] = 120.0
    with pytest.raises(ValueError, match="lat_deg must be between"):
        scenario_config_from_dict(cfg)

    cfg = _ground_station_config(tmp_path)
    cfg["ground_stations"][0]["max_range_km"] = -1.0
    with pytest.raises(ValueError, match="max_range_km must be positive"):
        scenario_config_from_dict(cfg)


def test_ground_station_access_geometry_applies_los_elevation_and_range() -> None:
    station = GroundStationSection(
        id="equator_prime",
        lat_deg=0.0,
        lon_deg=0.0,
        alt_km=0.0,
        min_elevation_deg=10.0,
        max_range_km=1000.0,
    )
    t_s = np.array([0.0])

    visible_hist, visible_summary = evaluate_ground_station_access(
        ground_stations=[station],
        t_s=t_s,
        truth_hist={"sat": np.array([[7000.0, 0.0, 0.0, 0.0, 0.0, 0.0]])},
    )
    assert visible_hist["equator_prime"]["targets"]["sat"]["access"] == [True]
    assert visible_hist["equator_prime"]["targets"]["sat"]["reason"] == ["ok"]
    assert visible_summary["equator_prime"]["sat"]["access_samples"] == 1

    blocked_hist, _ = evaluate_ground_station_access(
        ground_stations=[station],
        t_s=t_s,
        truth_hist={"sat": np.array([[-7000.0, 0.0, 0.0, 0.0, 0.0, 0.0]])},
    )
    assert blocked_hist["equator_prime"]["targets"]["sat"]["access"] == [False]
    assert blocked_hist["equator_prime"]["targets"]["sat"]["reason"] == ["line_of_sight"]

    short_range_station = GroundStationSection(
        id="equator_prime",
        lat_deg=0.0,
        lon_deg=0.0,
        alt_km=0.0,
        min_elevation_deg=0.0,
        max_range_km=100.0,
    )
    range_hist, _ = evaluate_ground_station_access(
        ground_stations=[short_range_station],
        t_s=t_s,
        truth_hist={"sat": np.array([[7000.0, 0.0, 0.0, 0.0, 0.0, 0.0]])},
    )
    assert range_hist["equator_prime"]["targets"]["sat"]["access"] == [False]
    assert range_hist["equator_prime"]["targets"]["sat"]["reason"] == ["range"]


def test_single_run_records_ground_station_access_payload(tmp_path: Path) -> None:
    result = SimulationSession.from_config(SimulationConfig.from_dict(_ground_station_config(tmp_path))).run()

    access = result.ground_station_access
    station = access["equator_prime"]
    target = station["targets"]["target"]
    assert target["access"][0] is True
    assert target["reason"][0] == "ok"
    assert target["range_km"][0] == pytest.approx(621.863, rel=1e-4)

    summary = result.summary["ground_station_access_summary"]["equator_prime"]["target"]
    assert summary["access_samples"] >= 1
    assert summary["first_access_time_s"] == 0.0
    assert summary["min_range_km"] == pytest.approx(621.863, rel=1e-4)
