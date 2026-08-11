from __future__ import annotations

import numpy as np

from sim.dynamics.coupled_satellite import (
    CoupledIntegratorConfig,
    CoupledSatelliteDynamics,
    CoupledSatelliteIntegrator,
    CoupledSatelliteState,
    StageEffects,
    constant_mass_properties,
)


def test_no_microstep_crosses_command_sensor_output_or_end_boundary() -> None:
    state = CoupledSatelliteState(
        np.zeros(3), np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3), 10.0, np.zeros(0), 0.0
    )
    dynamics = CoupledSatelliteDynamics(
        effects_model=lambda *_: StageEffects(), mass_properties_model=constant_mass_properties(np.eye(3))
    )
    result = CoupledSatelliteIntegrator(CoupledIntegratorConfig(0.4, 0.25), dynamics.derivative).propagate(
        state,
        end_time_s=1.0,
        hard_event_times_s=(0.3, 0.73),
        output_times_s=(0.5, 0.9),
    )
    boundaries = {0.3, 0.5, 0.73, 0.9, 1.0}
    assert boundaries == set(result.boundary_times_s)
    assert all(not (step.start_time_s < boundary < step.end_time_s) for step in result.steps for boundary in boundaries)
    assert [sample.t_s for sample in result.output_samples] == [0.5, 0.9]
