from __future__ import annotations

import numpy as np

from sim.dynamics.coupled_satellite import (
    CoupledIntegratorConfig,
    CoupledSatelliteDynamics,
    CoupledSatelliteIntegrator,
    CoupledSatelliteState,
    MassProperties,
    StageEffects,
)


def test_stage_mass_changes_force_acceleration_and_inertia_without_losing_momentum() -> None:
    initial_mass = 10.0
    initial_rate = 0.2
    state = CoupledSatelliteState(
        np.zeros(3),
        np.zeros(3),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([initial_rate, 0.0, 0.0]),
        initial_mass,
        np.zeros(0),
        0.0,
    )

    def properties(stage: CoupledSatelliteState, mass_rate: float) -> MassProperties:
        return MassProperties(np.eye(3) * stage.mass_kg, np.eye(3) * mass_rate)

    dynamics = CoupledSatelliteDynamics(
        effects_model=lambda *_: StageEffects(force_eci_n=np.array([10.0, 0.0, 0.0]), mass_flow_kg_s=1.0),
        mass_properties_model=properties,
    )
    final = (
        CoupledSatelliteIntegrator(CoupledIntegratorConfig(0.01, 0.01), dynamics.derivative)
        .propagate(state, end_time_s=1.0)
        .final_state
    )
    assert np.isclose(final.mass_kg, 9.0)
    np.testing.assert_allclose(final.velocity_eci_km_s[0], 0.01 * np.log(10.0 / 9.0), rtol=1e-10)
    np.testing.assert_allclose(final.angular_rate_body_rad_s[0], initial_rate * 10.0 / 9.0, rtol=1e-10)
    assert abs(9.0 * final.angular_rate_body_rad_s[0] - 10.0 * initial_rate) < 1.0e-10
