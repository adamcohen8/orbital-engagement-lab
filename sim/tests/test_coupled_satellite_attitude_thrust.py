from __future__ import annotations

import numpy as np
import pytest

from sim.core.models import Command, StateTruth
from sim.dynamics.coupled_satellite import (
    CoupledIntegratorConfig,
    CoupledSatelliteDynamics,
    CoupledSatelliteIntegrator,
    CoupledSatelliteState,
    StageEffects,
    constant_mass_properties,
)
from sim.dynamics.model import OrbitalAttitudeDynamics


def _run(step: float) -> CoupledSatelliteState:
    initial = CoupledSatelliteState(
        np.zeros(3), np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), 10.0, np.zeros(0), 0.0
    )
    dynamics = CoupledSatelliteDynamics(
        effects_model=lambda *_: StageEffects(force_body_n=np.array([10.0, 0.0, 0.0])),
        mass_properties_model=constant_mass_properties(np.eye(3)),
    )
    return (
        CoupledSatelliteIntegrator(CoupledIntegratorConfig(step, step), dynamics.derivative)
        .propagate(initial, end_time_s=1.0)
        .final_state
    )


def test_rotating_body_fixed_thrust_uses_stage_attitude_and_converges() -> None:
    coarse = _run(0.5)
    fine = _run(0.02)
    expected_velocity = np.array([np.sin(1.0), 1.0 - np.cos(1.0), 0.0]) / 1000.0
    assert np.linalg.norm(fine.velocity_eci_km_s - expected_velocity) < 2.0e-8
    assert np.linalg.norm(fine.velocity_eci_km_s - expected_velocity) < np.linalg.norm(
        coarse.velocity_eci_km_s - expected_velocity
    )
    assert abs(fine.velocity_eci_km_s[1]) > 1.0e-5


def test_production_dynamics_uses_rotating_body_force_and_consumes_propellant() -> None:
    dynamics = OrbitalAttitudeDynamics(
        mu_km3_s2=0.0,
        inertia_kg_m2=np.eye(3),
        orbit_substep_s=1.0,
        attitude_substep_s=0.02,
    )
    initial = StateTruth(
        np.array([7_000.0, 0.0, 0.0]),
        np.zeros(3),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        10.0,
        0.0,
    )
    command = Command(
        torque_body_nm=np.zeros(3),
        mode_flags={
            "physical_force_body_n": (10.0, 0.0, 0.0),
            "physical_force_eci_n": (0.0, 0.0, 0.0),
            "mass_flow_kg_s": 0.5,
            "min_mass_kg": 9.0,
        },
    )

    result = dynamics.step(initial, command, {}, 1.0)

    assert result.velocity_eci_km_s[0] > 0.0
    assert result.velocity_eci_km_s[1] > 1.0e-5
    assert result.mass_kg == pytest.approx(9.5)
