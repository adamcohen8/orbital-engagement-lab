from __future__ import annotations

import math

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
from sim.utils.quaternion import normalize_quaternion, omega_matrix


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


def _changing_axis_reference(step_count: int) -> np.ndarray:
    quaternion = np.array([1.0, 0.0, 0.0, 0.0])
    step = 1.0 / float(step_count)

    def rate(q: np.ndarray, time_s: float) -> np.ndarray:
        return 0.5 * omega_matrix(np.array([time_s, 1.0, 0.0])) @ q

    for index in range(step_count):
        time_s = float(index) * step
        k1 = rate(quaternion, time_s)
        k2 = rate(normalize_quaternion(quaternion + 0.5 * step * k1), time_s + 0.5 * step)
        k3 = rate(normalize_quaternion(quaternion + 0.5 * step * k2), time_s + 0.5 * step)
        k4 = rate(normalize_quaternion(quaternion + step * k3), time_s + step)
        quaternion = normalize_quaternion(quaternion + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))
    return quaternion


def _changing_axis_propagation(step: float) -> np.ndarray:
    initial = CoupledSatelliteState(
        np.zeros(3),
        np.zeros(3),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        10.0,
        np.zeros(0),
        0.0,
    )
    dynamics = CoupledSatelliteDynamics(
        effects_model=lambda *_: StageEffects(torque_body_n_m=np.array([1.0, 0.0, 0.0])),
        mass_properties_model=constant_mass_properties(np.eye(3)),
    )
    return (
        CoupledSatelliteIntegrator(CoupledIntegratorConfig(step, step), dynamics.derivative)
        .propagate(initial, end_time_s=1.0)
        .final_state.attitude_quat_bn
    )


def test_changing_axis_attitude_retains_fourth_order_convergence() -> None:
    reference = _changing_axis_reference(512)
    errors = []
    for step in (1.0, 0.5, 0.25, 0.125):
        actual = _changing_axis_propagation(step)
        dot = float(np.clip(abs(np.dot(actual, reference)), -1.0, 1.0))
        errors.append(2.0 * math.acos(dot))

    assert errors[0] > 0.0
    assert errors[0] / errors[1] > 12.0
    assert errors[1] / errors[2] > 12.0
    assert errors[2] / errors[3] > 12.0
