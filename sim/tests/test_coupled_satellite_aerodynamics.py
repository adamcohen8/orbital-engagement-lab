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
from sim.dynamics.spacecraft_geometry import RectangularPrismGeometry
from sim.utils.quaternion import quaternion_to_dcm_bn


def test_aerodynamic_geometry_is_recomputed_from_each_stage_attitude() -> None:
    geometry = RectangularPrismGeometry(1.0, 2.0, 3.0)
    areas: list[float] = []

    def aero(_time: float, state: CoupledSatelliteState, _control: object) -> StageEffects:
        flow_eci = np.array([-1.0, 0.0, 0.0])
        flow_body = quaternion_to_dcm_bn(state.attitude_quat_bn) @ flow_eci
        area = geometry.projected_area_m2(flow_body)
        areas.append(area)
        return StageEffects(force_eci_n=np.array([-area, 0.0, 0.0]))

    initial = CoupledSatelliteState(
        np.zeros(3), np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), 10.0, np.zeros(0), 0.0
    )
    dynamics = CoupledSatelliteDynamics(effects_model=aero, mass_properties_model=constant_mass_properties(np.eye(3)))
    final = (
        CoupledSatelliteIntegrator(CoupledIntegratorConfig(0.5, 0.5), dynamics.derivative)
        .propagate(initial, end_time_s=0.5)
        .final_state
    )
    assert max(areas) - min(areas) > 0.5
    assert final.velocity_eci_km_s[0] < 0.0
    assert len(areas) == 4
