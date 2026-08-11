from __future__ import annotations

import numpy as np

from sim.dynamics.coupled_satellite import (
    CoupledDerivative,
    CoupledIntegratorConfig,
    CoupledSatelliteIntegrator,
    CoupledSatelliteState,
)


def test_each_rk_stage_receives_matching_time_state_and_attitude() -> None:
    stages: list[CoupledSatelliteState] = []

    def derivative(_time: float, state: CoupledSatelliteState, _control: object) -> CoupledDerivative:
        stages.append(state)
        return CoupledDerivative(
            state.velocity_eci_km_s,
            np.array([1.0, 0.0, 0.0]),
            np.zeros(3),
            -0.5,
            np.ones_like(state.actuator_state),
        )

    initial = CoupledSatelliteState(
        np.zeros(3),
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        10.0,
        np.array([0.0]),
        0.0,
    )
    result = CoupledSatelliteIntegrator(CoupledIntegratorConfig(1.0, 1.0), derivative).propagate(
        initial, end_time_s=1.0
    )
    assert [stage.t_s for stage in stages] == [0.0, 0.5, 0.5, 1.0]
    assert stages[1].mass_kg == 9.75 and stages[1].actuator_state[0] == 0.5
    assert np.isclose(stages[1].attitude_quat_bn[0], np.cos(0.25), rtol=0.0, atol=1.0e-15)
    np.testing.assert_allclose(result.final_state.position_eci_km, [1.5, 0.0, 0.0])
    assert result.final_state.mass_kg == 9.5
    assert result.final_state.actuator_state[0] == 1.0


def test_substep_configuration_rejects_attitude_larger_than_orbit() -> None:
    try:
        CoupledIntegratorConfig(0.1, 0.2)
    except ValueError as exc:
        assert "must not exceed" in str(exc)
    else:
        raise AssertionError("invalid multirate configuration was accepted")
