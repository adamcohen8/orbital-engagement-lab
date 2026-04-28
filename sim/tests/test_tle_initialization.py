from __future__ import annotations

import numpy as np

from sim import SimulationConfig, SimulationSession
from sim.dynamics.orbit.tle import parse_tle_lines, tle_to_rv_eci


ISS_LINE1 = "1 25544U 98067A   24001.00000000  .00016717  00000+0  10270-3 0  9005"
ISS_LINE2 = "2 25544  51.6416  43.6012 0005423  52.3066  50.1234 15.50000000  1000"


def _tle_config(*, initial_jd_utc: float | None = None) -> dict:
    simulator = {
        "duration_s": 1.0,
        "dt_s": 1.0,
        "termination": {"earth_impact_enabled": False},
        "dynamics": {"attitude": {"enabled": False}, "orbit": {"model": "two_body"}},
    }
    if initial_jd_utc is not None:
        simulator["initial_jd_utc"] = initial_jd_utc
    return {
        "scenario_name": "tle_initialization",
        "rocket": {"enabled": False},
        "chaser": {"enabled": False},
        "target": {
            "enabled": True,
            "specs": {"mass_kg": 420.0},
            "initial_state": {
                "tle": {
                    "line1": ISS_LINE1,
                    "line2": ISS_LINE2,
                }
            },
        },
        "simulator": simulator,
        "outputs": {
            "mode": "save",
            "stats": {"print_summary": False, "save_json": False, "save_full_log": False},
            "plots": {"enabled": False, "figure_ids": []},
            "animations": {"enabled": False, "types": []},
        },
    }


def test_tle_parser_converts_mean_elements_to_eci_state() -> None:
    elements = parse_tle_lines(ISS_LINE1, ISS_LINE2)
    pos, vel = tle_to_rv_eci(elements)

    assert elements.epoch_jd_utc == 2460310.5
    assert np.linalg.norm(pos) > 6500.0
    assert 7.0 < np.linalg.norm(vel) < 8.0


def test_satellite_initial_state_accepts_tle_lines() -> None:
    elements = parse_tle_lines(ISS_LINE1, ISS_LINE2)
    expected_pos, expected_vel = tle_to_rv_eci(elements)
    session = SimulationSession.from_config(SimulationConfig.from_dict(_tle_config()))

    result = session.run()
    truth0 = result.truth["target"][0]

    np.testing.assert_allclose(truth0[0:3], expected_pos, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(truth0[3:6], expected_vel, rtol=0.0, atol=1e-12)


def test_tle_initial_state_propagates_to_simulator_initial_jd() -> None:
    elements = parse_tle_lines(ISS_LINE1, ISS_LINE2)
    target_jd = elements.epoch_jd_utc + 0.25
    expected_pos, expected_vel = tle_to_rv_eci(elements, target_jd_utc=target_jd)
    session = SimulationSession.from_config(SimulationConfig.from_dict(_tle_config(initial_jd_utc=target_jd)))

    result = session.run()
    truth0 = result.truth["target"][0]

    np.testing.assert_allclose(truth0[0:3], expected_pos, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(truth0[3:6], expected_vel, rtol=0.0, atol=1e-12)
