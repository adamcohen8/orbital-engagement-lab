from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from sim.api import SimulationConfig, SimulationSession
from sim.config import scenario_config_from_dict, validate_scenario_plugins
from sim.core.models import Command


def _scenario(tmp_path: Path, *, trajectory_only: bool) -> dict:
    satellite: dict[str, object] = {
        "kind": "satellite",
        "specs": {"mass_kg": 200.0},
        "initial_state": {"default_circular_earth": True},
    }
    if trajectory_only:
        satellite["runtime_profile"] = "trajectory_only"
    else:
        satellite["flight_software"] = {
            "stack": "fsw.passive",
            "hardware_profile": "hardware.passive.v1",
            "task_period_s": 1.0,
        }
    return {
        "scenario_name": "trajectory_only_test",
        "objects": {"sat": satellite},
        "simulator": {
            "duration_s": 20.0,
            "dt_s": 1.0,
            "dynamics": {
                "orbit": {"model": "two_body", "orbit_substep_s": 1.0},
                "attitude": {"enabled": False},
            },
        },
        "outputs": {
            "mode": "save",
            "output_dir": str(tmp_path),
            "stats": {
                "enabled": True,
                "print_summary": False,
                "save_json": False,
                "save_csv": False,
                "save_full_log": False,
            },
            "plots": {"enabled": False, "figure_ids": []},
            "animations": {"enabled": False, "types": []},
            "review": {"enabled": True, "detail": "standard"},
        },
    }


def test_trajectory_only_is_explicit_and_preserves_passive_default() -> None:
    default = scenario_config_from_dict(
        {
            "objects": {"sat": {"kind": "satellite", "initial_state": {"default_circular_earth": True}}},
            "simulator": {"duration_s": 1.0, "dt_s": 1.0},
        }
    ).objects["sat"]
    trajectory = scenario_config_from_dict(
        {
            "objects": {
                "sat": {
                    "kind": "satellite",
                    "runtime_profile": "trajectory_only",
                    "initial_state": {"default_circular_earth": True},
                }
            },
            "simulator": {"duration_s": 1.0, "dt_s": 1.0},
        }
    ).objects["sat"]

    assert default.runtime_profile == "flight_software"
    assert default.flight_software is not None
    assert default.flight_software.stack == "fsw.passive"
    assert trajectory.runtime_profile == "trajectory_only"
    assert trajectory.flight_software is None
    serialized = scenario_config_from_dict(
        {
            "objects": {
                "default": {"kind": "satellite", "initial_state": {"default_circular_earth": True}},
                "trajectory": {
                    "kind": "satellite",
                    "runtime_profile": "trajectory_only",
                    "initial_state": {"default_circular_earth": True},
                },
            },
            "simulator": {"duration_s": 1.0, "dt_s": 1.0},
        }
    ).to_dict()
    assert "runtime_profile" not in serialized["objects"]["default"]
    assert serialized["objects"]["trajectory"]["runtime_profile"] == "trajectory_only"


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (
            {"flight_software": {"stack": "fsw.passive", "hardware_profile": "hardware.passive.v1"}},
            "cannot declare flight_software",
        ),
        ({"knowledge": {"targets": ["other"]}}, "cannot declare knowledge"),
    ],
)
def test_trajectory_only_rejects_onboard_configuration(extra: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        scenario_config_from_dict(
            {
                "objects": {
                    "sat": {
                        "kind": "satellite",
                        "runtime_profile": "trajectory_only",
                        "initial_state": {"default_circular_earth": True},
                        **extra,
                    }
                },
                "simulator": {"duration_s": 1.0, "dt_s": 1.0},
            }
        )


def test_trajectory_only_cannot_be_the_game_controlled_object() -> None:
    config = scenario_config_from_dict(
        {
            "metadata": {"game": {"controlled_object_id": "sat"}},
            "objects": {
                "sat": {
                    "kind": "satellite",
                    "runtime_profile": "trajectory_only",
                    "initial_state": {"default_circular_earth": True},
                }
            },
            "simulator": {"duration_s": 1.0, "dt_s": 1.0},
        }
    )

    errors = validate_scenario_plugins(config, import_plugins=False)

    assert any("cannot be the game controlled_object_id" in error for error in errors)


def test_trajectory_only_matches_passive_two_body_truth_exactly(tmp_path: Path) -> None:
    passive = SimulationSession.from_config(
        SimulationConfig.from_dict(_scenario(tmp_path / "passive", trajectory_only=False))
    ).run()
    trajectory_session = SimulationSession.from_config(
        SimulationConfig.from_dict(_scenario(tmp_path / "trajectory", trajectory_only=True))
    )
    trajectory = trajectory_session.run()

    assert np.array_equal(trajectory.truth["sat"], passive.truth["sat"])
    assert np.count_nonzero(np.nan_to_num(trajectory.applied_thrust["sat"], nan=0.0)) == 0
    assert trajectory_session._engine.agents["sat"].flight_software_runtime is None
    assert trajectory.summary["object_runtime_profiles"] == {"sat": "trajectory_only"}
    assert trajectory.payload["object_runtime_profiles"] == {"sat": "trajectory_only"}


def test_trajectory_only_reuses_zero_command_across_builtin_substeps(tmp_path: Path) -> None:
    raw = _scenario(tmp_path / "substeps", trajectory_only=True)
    raw["simulator"]["duration_s"] = 1.0
    raw["simulator"]["dynamics"]["orbit"]["orbit_substep_s"] = 0.25
    session = SimulationSession.from_config(SimulationConfig.from_dict(raw))
    session._ensure_engine()

    with patch.object(Command, "zero", wraps=Command.zero) as zero_command:
        session.run()

    assert zero_command.call_count == 1


def test_serial_object_step_does_not_clone_result_without_bridge(tmp_path: Path) -> None:
    raw = _scenario(tmp_path / "no_bridge", trajectory_only=True)
    raw["simulator"]["duration_s"] = 1.0
    session = SimulationSession.from_config(SimulationConfig.from_dict(raw))
    session._ensure_engine()

    with patch("sim.single_run.replace", side_effect=AssertionError("unexpected result clone")):
        session.run()


def test_review_store_records_trajectory_only_without_fsw_rows(tmp_path: Path) -> None:
    result = SimulationSession.from_config(
        SimulationConfig.from_dict(_scenario(tmp_path / "review", trajectory_only=True))
    ).run()
    db_path = Path(result.summary["review_outputs"]["sqlite"])

    with sqlite3.connect(db_path) as conn:
        profile = conn.execute(
            "SELECT runtime_profile, flight_software_stack FROM objects WHERE object_id = 'sat'"
        ).fetchone()
        invocation_count = conn.execute("SELECT COUNT(*) FROM fsw_invocations").fetchone()[0]

    assert profile == ("trajectory_only", "")
    assert invocation_count == 0


@pytest.mark.parametrize(
    ("case_id", "general", "line1", "line2"),
    (
        (
            "sgp4",
            {"model": "sgp4", "output_frame": "teme"},
            "1 25544U 98067A   24001.00000000  .00016717  00000+0  10270-3 0  9003",
            "2 25544  51.6416  43.6012 0005423  52.3066  50.1234 15.50000000  1004",
        ),
        (
            "sdp4",
            {"model": "sgp4", "output_frame": "eci", "frame_transform": "teme_to_eci_iau80"},
            "1 91001U 24001A   24001.00000000  .00000000  00000+0  00000+0 0  9993",
            "2 91001   0.0500  80.0000 0001000  10.0000  20.0000  1.00270000    00",
        ),
    ),
)
def test_trajectory_only_matches_passive_ogp_truth_exactly(
    tmp_path: Path,
    case_id: str,
    general: dict[str, str],
    line1: str,
    line2: str,
) -> None:
    base = {
        "scenario_name": f"trajectory_only_ogp_{case_id}",
        "objects": {
            "catalog_object": {
                "kind": "satellite",
                "propagation_method": "general",
                "general": general,
                "specs": {"mass_kg": 420000.0},
                "initial_state": {
                    "tle": {
                        "line1": line1,
                        "line2": line2,
                        "require_checksum": True,
                    }
                },
            }
        },
        "simulator": {
            "duration_s": 240.0,
            "dt_s": 120.0,
            "initial_jd_utc": 2460310.5,
            "dynamics": {"orbit": {"model": "two_body"}, "attitude": {"enabled": False}},
        },
        "outputs": {
            "mode": "save",
            "stats": {"enabled": False},
            "plots": {"enabled": False, "figure_ids": []},
            "animations": {"enabled": False, "types": []},
        },
    }
    passive_raw = {
        **base,
        "objects": {
            "catalog_object": {
                **base["objects"]["catalog_object"],
                "flight_software": {
                    "stack": "fsw.passive",
                    "hardware_profile": "hardware.passive.v1",
                },
            }
        },
        "outputs": {**base["outputs"], "output_dir": str(tmp_path / f"passive_ogp_{case_id}")},
    }
    trajectory_raw = {
        **base,
        "objects": {
            "catalog_object": {
                **base["objects"]["catalog_object"],
                "runtime_profile": "trajectory_only",
            }
        },
        "outputs": {**base["outputs"], "output_dir": str(tmp_path / f"trajectory_ogp_{case_id}")},
    }

    passive = SimulationSession.from_config(SimulationConfig.from_dict(passive_raw)).run()
    trajectory = SimulationSession.from_config(SimulationConfig.from_dict(trajectory_raw)).run()

    assert np.array_equal(trajectory.truth["catalog_object"], passive.truth["catalog_object"])
