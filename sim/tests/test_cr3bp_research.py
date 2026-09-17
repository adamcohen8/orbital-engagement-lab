"""Numerical verification of CR3BP research transforms and invariants."""

import itertools

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from sim.dynamics.orbit.cr3bp import (
    EARTH_MOON_CR3BP as SYS,
)
from sim.dynamics.orbit.cr3bp import (
    cr3bp_derivative_physical,
    cr3bp_halo_seed_state_km_s,
    propagate_cr3bp_state,
)
from sim.dynamics.orbit.cr3bp_research import (
    cr3bp_jacobi_constant,
    cr3bp_jacobi_diagnostics,
    cr3bp_libration_points,
    cr3bp_zero_velocity_grid,
    transform_cr3bp_state,
)


@pytest.mark.parametrize(
    "source_axes,target_axes,source_origin,target_origin",
    list(
        itertools.product(
            ("rotating", "inertial"), ("rotating", "inertial"), ("barycenter", "p1", "p2"), ("barycenter", "p1", "p2")
        )
    ),
)
def test_frame_roundtrip(source_axes, target_axes, source_origin, target_origin):
    state = np.array([[1e4, -2e4, 3e4, 0.1, -0.2, 0.3], [2e4, 3e4, -1e4, -0.4, 0.2, 0.1]])
    time = np.array([1234.0, 98765.0])
    kw = dict(reference_angle_rad=0.6, reference_time_s=123.0)
    transformed = transform_cr3bp_state(
        state,
        time,
        source_axes=source_axes,
        target_axes=target_axes,
        source_origin=source_origin,
        target_origin=target_origin,
        **kw,
    )
    restored = transform_cr3bp_state(
        transformed,
        time,
        source_axes=target_axes,
        target_axes=source_axes,
        source_origin=target_origin,
        target_origin=source_origin,
        **kw,
    )
    np.testing.assert_allclose(restored, state, atol=2e-10)


@pytest.mark.parametrize("origin", ["barycenter", "p1", "p2"])
def test_velocity_is_position_derivative(origin):
    state = cr3bp_halo_seed_state_km_s(family="nrho")
    t, dt = 123456.0, 0.05
    before, after = state.copy(), state.copy()
    before[:3] -= dt * state[3:]
    after[:3] += dt * state[3:]
    velocity = (
        transform_cr3bp_state(after, t + dt, target_origin=origin)[:3]
        - transform_cr3bp_state(before, t - dt, target_origin=origin)[:3]
    ) / (2 * dt)
    expected = transform_cr3bp_state(state, t, target_origin=origin)[3:]
    np.testing.assert_allclose(velocity, expected, atol=2e-9)


def test_primary_is_stationary_at_its_own_origin():
    p = np.array([(1 - SYS.mu) * SYS.distance_km, 0, 0, 0, 0, 0])
    assert np.max(np.abs(transform_cr3bp_state(p, [0, 1000], target_origin="p2"))) == 0
    inertial = transform_cr3bp_state(p, 0.0)
    assert inertial[4] == pytest.approx(SYS.mean_motion_rad_s * p[0])


def test_libration_equilibria_and_zero_velocity_identity():
    for point in cr3bp_libration_points().values():
        state = np.r_[point, [0.0, 0.0, 0.0]]
        assert np.linalg.norm(cr3bp_derivative_physical(state)) < 1e-14
    state = cr3bp_halo_seed_state_km_s(family="nrho")
    c = cr3bp_jacobi_constant(state)
    assert np.isfinite(c)
    u, v, allowed = cr3bp_zero_velocity_grid(float(c), resolution=31)
    assert u.shape == v.shape == allowed.shape == (31, 31)
    assert (allowed < 0).any() and (allowed > 0).any()


def test_jacobi_convergence():
    # A deliberately coarse pair makes accumulated RK4 error measurable.
    seed = cr3bp_halo_seed_state_km_s(family="l1_northern_large")
    errors = []
    for dt in (1800.0, 900.0, 450.0):
        state = seed.copy()
        history = [state]
        for t in np.arange(0.0, 172800.0, dt):
            state = propagate_cr3bp_state(state, dt, t)
            history.append(state)
        errors.append(cr3bp_jacobi_diagnostics(history)["max_absolute_change"])
    assert errors[1] < errors[0] / 8
    assert errors[2] < errors[1] / 8


@pytest.mark.parametrize("family", ["l1_northern_large", "l2_nrho_southern"])
def test_corrected_seed_periodic_closure(family):
    seed = cr3bp_halo_seed_state_km_s(family=family)

    # Independent adaptive integration checks the stored corrected seeds.
    def crossing(t, state):
        return state[1]

    crossing.direction = -np.sign(seed[4])
    crossing.terminal = True
    half = solve_ivp(
        lambda t, x: cr3bp_derivative_physical(x),
        (1e-6, 2000000.0),
        seed,
        events=crossing,
        rtol=2e-12,
        atol=1e-12,
        max_step=300.0,
    )
    assert len(half.t_events[0]) == 1
    period = 2 * (half.t_events[0][0] - 1e-6)
    final = solve_ivp(
        lambda t, x: cr3bp_derivative_physical(x), (0.0, period), seed, rtol=2e-12, atol=1e-12, max_step=300.0
    ).y[:, -1]
    assert np.linalg.norm(final[:3] - seed[:3]) < 0.1
    assert np.linalg.norm(final[3:] - seed[3:]) < 1e-6


@pytest.mark.parametrize("kwargs", [{"target_axes": "eci"}, {"target_origin": "moon"}, {"reference_time_s": np.nan}])
def test_bad_frame_contract(kwargs):
    with pytest.raises(ValueError):
        transform_cr3bp_state(np.zeros(6), 0.0, **kwargs)


def test_jacobi_change_matches_commanded_work():
    seed = cr3bp_halo_seed_state_km_s(family="nrho")
    command = np.array([2e-7, -3e-7, 1e-7])
    dt = 0.1
    final = propagate_cr3bp_state(seed, dt, 0.0, command_accel_km_s2=command)
    dc_dt = (cr3bp_jacobi_constant(final) - cr3bp_jacobi_constant(seed)) / dt
    expected = -2 * np.dot(seed[3:], command) / (SYS.distance_km * SYS.mean_motion_rad_s) ** 2
    assert dc_dt == pytest.approx(expected, rel=1e-5)
