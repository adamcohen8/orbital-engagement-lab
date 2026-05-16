from __future__ import annotations

from pathlib import Path

import numpy as np

from sim import SimulationConfig, SimulationSession


def _physical_specs_config(
    tmp_path: Path,
    *,
    scenario_name: str,
    specs: dict[str, float],
    orbit_dynamics: dict[str, object],
    environment: dict[str, object],
    duration_s: float,
    dt_s: float,
) -> SimulationConfig:
    return SimulationConfig.from_dict(
        {
            "scenario_name": scenario_name,
            "objects": {
                "sat": {
                    "kind": "satellite",
                    "enabled": True,
                    "specs": {"mass_kg": 100.0, **dict(specs)},
                    "initial_state": {
                        "position_eci_km": [6578.137, 0.0, 0.0],
                        "velocity_eci_km_s": [0.0, 7.784, 0.0],
                    },
                }
            },
            "simulator": {
                "duration_s": duration_s,
                "dt_s": dt_s,
                "dynamics": {
                    "orbit": {"orbit_substep_s": dt_s, **dict(orbit_dynamics)},
                    "attitude": {"enabled": False},
                },
                "environment": dict(environment),
                "termination": {"earth_impact_enabled": False},
            },
            "outputs": {
                "output_dir": str(tmp_path / scenario_name),
                "mode": "save",
                "stats": {"print_summary": False, "save_json": False, "save_full_log": False},
                "plots": {"enabled": False, "figure_ids": []},
                "animations": {"enabled": False, "types": []},
            },
            "monte_carlo": {"enabled": False},
        }
    )


def _final_velocity(cfg: SimulationConfig) -> np.ndarray:
    result = SimulationSession.from_config(cfg).run()
    return np.array(result.truth["sat"][-1, 3:6], dtype=float)


def test_satellite_drag_uses_configured_area_and_cd(tmp_path: Path) -> None:
    orbit = {"drag": True}
    env = {"density_kg_m3": 1.0e-9}

    no_drag_v = _final_velocity(
        _physical_specs_config(
            tmp_path,
            scenario_name="satellite_drag_area_zero",
            specs={"area_ref_m2": 0.0, "cd": 9.0},
            orbit_dynamics=orbit,
            environment=env,
            duration_s=20.0,
            dt_s=1.0,
        )
    )
    high_drag_v = _final_velocity(
        _physical_specs_config(
            tmp_path,
            scenario_name="satellite_drag_area_high",
            specs={"area_ref_m2": 100.0, "cd": 9.0},
            orbit_dynamics=orbit,
            environment=env,
            duration_s=20.0,
            dt_s=1.0,
        )
    )

    assert float(np.linalg.norm(high_drag_v - no_drag_v)) > 1.0e-4
    assert float(high_drag_v[1]) < float(no_drag_v[1])


def test_satellite_srp_uses_configured_area_and_cr(tmp_path: Path) -> None:
    orbit = {"srp": True}
    env = {
        "srp_shadow_model": "none",
        "sun_pos_eci_km": [149597870.7, 0.0, 0.0],
    }

    no_srp_v = _final_velocity(
        _physical_specs_config(
            tmp_path,
            scenario_name="satellite_srp_cr_zero",
            specs={"area_ref_m2": 1000.0, "cr": 0.0},
            orbit_dynamics=orbit,
            environment=env,
            duration_s=100.0,
            dt_s=10.0,
        )
    )
    high_srp_v = _final_velocity(
        _physical_specs_config(
            tmp_path,
            scenario_name="satellite_srp_cr_high",
            specs={"area_ref_m2": 1000.0, "cr": 4.0},
            orbit_dynamics=orbit,
            environment=env,
            duration_s=100.0,
            dt_s=10.0,
        )
    )

    assert float(np.linalg.norm(high_srp_v - no_srp_v)) > 1.0e-5
    assert float(high_srp_v[0]) < float(no_srp_v[0])
