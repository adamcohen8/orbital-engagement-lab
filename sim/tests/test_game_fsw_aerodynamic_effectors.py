from __future__ import annotations

import numpy as np

from sim.dynamics.aerodynamic_effectors import (
    AerodynamicSurfaceGeometry,
    VariableGeometryAerodynamics,
    VariableGeometryAerodynamicsConfig,
)
from sim.dynamics.coupled_satellite import CoupledSatelliteState


def _state() -> CoupledSatelliteState:
    return CoupledSatelliteState(
        np.array([6_678.0, 0.0, 0.0]),
        np.array([0.0, 7.7, 0.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.zeros(3),
        5_000.0,
        np.zeros(2),
        0.0,
    )


def test_realized_effector_geometry_changes_stage_force_torque_and_mass_properties() -> None:
    model = VariableGeometryAerodynamics(
        VariableGeometryAerodynamicsConfig(
            (
                AerodynamicSurfaceGeometry("flap", 20.0, 2.2, 0.5, (0.0, 1.0, 0.0), surface_mass_kg=5.0),
            ),
            bank_actuator_id="bank",
            base_drag_area_m2=5.0,
        )
    )
    retracted, retracted_review = model.evaluate(
        _state(), density_kg_m3=1.0e-10, device_positions={"flap": 0.0, "bank": 0.0}
    )
    deployed, deployed_review = model.evaluate(
        _state(), density_kg_m3=1.0e-10, device_positions={"flap": 1.0, "bank": 0.4}
    )
    assert np.linalg.norm(deployed.force_eci_n) > np.linalg.norm(retracted.force_eci_n)
    assert np.linalg.norm(deployed.torque_body_n_m) > 0.0
    assert deployed_review.drag_area_m2 > retracted_review.drag_area_m2
    assert deployed_review.bank_angle_rad == 0.4
    base_inertia = np.eye(3) * 100.0
    assert not np.array_equal(
        model.mass_properties(base_inertia, device_positions={"flap": 1.0}).inertia_body_kg_m2,
        base_inertia,
    )
