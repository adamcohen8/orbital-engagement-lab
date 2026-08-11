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


def test_aligned_output_cadence_does_not_change_physical_result_or_invoke_tasks() -> None:
    state = CoupledSatelliteState(
        np.zeros(3),
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.1]),
        10.0,
        np.zeros(0),
        0.0,
    )
    calls = 0

    def effects(*_args: object) -> StageEffects:
        nonlocal calls
        calls += 1
        return StageEffects(force_eci_n=np.array([1.0, 0.0, 0.0]))

    dynamics = CoupledSatelliteDynamics(
        effects_model=effects, mass_properties_model=constant_mass_properties(np.eye(3))
    )
    integrator = CoupledSatelliteIntegrator(CoupledIntegratorConfig(0.1, 0.1), dynamics.derivative)
    sparse = integrator.propagate(state, end_time_s=1.0, output_times_s=(0.5, 1.0))
    sparse_calls = calls
    calls = 0
    dense = integrator.propagate(state, end_time_s=1.0, output_times_s=tuple(index / 10 for index in range(1, 11)))
    assert calls == sparse_calls
    np.testing.assert_array_equal(dense.final_state.position_eci_km, sparse.final_state.position_eci_km)
    np.testing.assert_array_equal(dense.final_state.velocity_eci_km_s, sparse.final_state.velocity_eci_km_s)
    np.testing.assert_allclose(
        dense.final_state.attitude_quat_bn,
        sparse.final_state.attitude_quat_bn,
        rtol=0.0,
        atol=2.0e-16,
    )
    assert [sample.t_s for sample in sparse.output_samples] == [0.5, 1.0]
    assert [sample.t_s for sample in dense.output_samples] == [index / 10 for index in range(1, 11)]
