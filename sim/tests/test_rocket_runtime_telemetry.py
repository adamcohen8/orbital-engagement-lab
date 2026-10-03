"""Rocket telemetry consumes native averages and retains the legacy fallback."""

import numpy as np
import pytest

import sim.single_run_support as owner
from sim.config import scenario_config_from_dict
from sim.single_run import _SingleRunEngine


@pytest.mark.parametrize("backend", ["python", "rust"])
@pytest.mark.parametrize("drop_telemetry", [False, True])
def test_step_acceleration_field_and_missing_field_fallback(tmp_path, monkeypatch, backend, drop_telemetry):
    if backend == "rust":
        pytest.importorskip("oel_rust_orbit")
    config = scenario_config_from_dict({
        "scenario_name": "rocket_telemetry",
        "objects": {"rocket": {
            "enabled": True, "kind": "rocket", "role": "rocket",
            "specs": {"preset_stack": "BASIC_TWO_STAGE_STACK", "payload_mass_kg": 150.0},
            "initial_state": {
                "launch_lat_deg": 28.5, "launch_lon_deg": -80.6,
                "launch_alt_km": 0.1, "launch_azimuth_deg": 90.0,
            },
        }},
        "simulator": {
            "duration_s": 3.0, "dt_s": 1.0,
            "dynamics": {"rocket": {"numeric_backend": backend, "attitude_mode": "cheater"}},
            "termination": {"earth_impact_enabled": False},
        },
        "outputs": {
            "output_dir": str(tmp_path), "plots": {"enabled": False},
            "stats": {"save_json": False, "save_full_log": False, "print_summary": False},
        },
    })
    engine = _SingleRunEngine(config)
    agent = engine.agents["rocket"]
    step = agent.rocket_sim.step

    def mechanical_step(*args, **kwargs):
        state = step(*args, **kwargs)
        assert hasattr(state, "_last_step_thrust_accel_eci_km_s2")
        if drop_telemetry:
            delattr(state, "_last_step_thrust_accel_eci_km_s2")
        return state

    monkeypatch.setattr(agent.rocket_sim, "step", mechanical_step)
    original_rotation = owner.quaternion_to_dcm_bn
    calls = []

    def fallback_rotation(quaternion):
        calls.append(True)
        return original_rotation(quaternion)

    monkeypatch.setattr(owner, "quaternion_to_dcm_bn", fallback_rotation)
    engine.step()
    state = agent.rocket_state
    if drop_telemetry:
        axis = original_rotation(state.attitude_quat_bn).T @ np.asarray(
            getattr(state, "thrust_vector_body", agent.rocket_sim.vehicle_cfg.thrust_axis_body), dtype=float,
        )
        expected = float(state._last_step_thrust_n) / max(state.mass_kg, 1e-9) * axis / 1e3
    else:
        expected = state._last_step_thrust_accel_eci_km_s2
    np.testing.assert_array_equal(engine.thrust_hist["rocket"][1], expected)
    assert len(calls) == int(drop_telemetry)
