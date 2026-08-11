from __future__ import annotations

import numpy as np

from sim.dynamics.coupled_satellite import (
    CoupledIntegratorConfig,
    CoupledSatelliteDynamics,
    CoupledSatelliteIntegrator,
    CoupledSatelliteState,
    StageEffects,
    constant_mass_properties,
    two_body_gravity,
)


def _propagate(step: float) -> CoupledSatelliteState:
    mu = 398600.4418
    radius = 7000.0
    state = CoupledSatelliteState(
        np.array([radius, 0.0, 0.0]),
        np.array([0.0, np.sqrt(mu / radius), 0.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.01]),
        100.0,
        np.zeros(0),
        0.0,
    )
    dynamics = CoupledSatelliteDynamics(
        effects_model=lambda *_: StageEffects(),
        mass_properties_model=constant_mass_properties(np.diag([2.0, 3.0, 4.0])),
        gravity_model=two_body_gravity(mu),
    )
    return (
        CoupledSatelliteIntegrator(CoupledIntegratorConfig(step, step), dynamics.derivative)
        .propagate(state, end_time_s=120.0)
        .final_state
    )


def test_two_body_and_constant_axis_attitude_converge_under_step_refinement() -> None:
    reference = _propagate(0.125)
    coarse = _propagate(2.0)
    medium = _propagate(1.0)
    coarse_error = np.linalg.norm(coarse.position_eci_km - reference.position_eci_km)
    medium_error = np.linalg.norm(medium.position_eci_km - reference.position_eci_km)
    assert medium_error < coarse_error / 10.0
    assert abs(np.linalg.norm(medium.attitude_quat_bn) - 1.0) < 1.0e-14
