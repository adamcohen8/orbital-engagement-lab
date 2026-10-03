"""Native CR3BP parity, variational dynamics, accuracy and runtime selection."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from sim.api import SimulationConfig, SimulationSession
from sim.dynamics.orbit.accelerations import OrbitContext
from sim.dynamics.orbit.cr3bp import (
    EARTH_MOON_CR3BP,
    CR3BPSystem,
    cr3bp_derivative_physical,
    cr3bp_halo_seed_state_km_s,
    cr3bp_jacobian_physical,
    propagate_cr3bp_reference_stm,
    propagate_cr3bp_state,
)
from sim.dynamics.orbit.cr3bp_research import cr3bp_jacobi_constant
from sim.dynamics.orbit.propagator import OrbitPropagator


@pytest.fixture
def native():
    orbit = pytest.importorskip("oel_rust_orbit")
    if not hasattr(orbit, "cr3bp_propagate"):
        pytest.skip("CR3BP-enabled optional wheel is not installed")
    return orbit


@pytest.mark.parametrize("system", [EARTH_MOON_CR3BP, CR3BPSystem(distance_km=10000.0, mu=0.1, mean_motion_rad_s=0.0001)])
def test_native_derivative_and_jacobian_match_physical_contract(native, system):
    rng = np.random.default_rng(712)
    for _ in range(16):
        state = rng.uniform(-0.7, 0.7, 6)
        state[:3] *= system.distance_km
        state[3:] *= system.distance_km * system.mean_motion_rad_s
        command = rng.uniform(-1e-7, 1e-7, 3)
        args = (state.tolist(), system.distance_km, system.mu, system.mean_motion_rad_s)
        rate = native.cr3bp_derivative(*args, command.tolist())
        jacobian = np.array(native.cr3bp_jacobian(*args)).reshape(6, 6)
        np.testing.assert_allclose(rate, cr3bp_derivative_physical(state, system=system, command_accel_km_s2=command), rtol=2e-14, atol=1e-18)
        np.testing.assert_allclose(jacobian, cr3bp_jacobian_physical(state, system=system), rtol=2e-14, atol=1e-24)


@pytest.mark.parametrize("family", ["l1_northern", "l1_northern_large", "nrho"])
def test_rk4_complete_forced_history_parity(native, family):
    state = cr3bp_halo_seed_state_km_s(family=family)
    histories = {}
    for backend in ("python", "rust"):
        x = state.copy()
        rows = [x]
        for k in range(240):
            command = np.array([1e-8, -2e-8, 3e-8]) if 60 <= k < 120 else np.zeros(3)
            x = propagate_cr3bp_state(x, 30.0, k * 30.0, command, numeric_backend=backend)
            rows.append(x)
        histories[backend] = np.array(rows)
    np.testing.assert_allclose(histories["rust"], histories["python"], rtol=0.0, atol=1e-8)


@pytest.mark.parametrize("method", ["rkf78", "adaptive", "dopri5"])
def test_adaptive_state_and_evidence_parity(native, method):
    state = cr3bp_halo_seed_state_km_s(family="nrho")
    command = np.array([1e-8, -2e-8, 3e-8])
    options = dict(integrator=method, adaptive_atol=1e-12, adaptive_rtol=1e-11, h_init=3600.0, return_info=True)
    expected, py_info = propagate_cr3bp_state(state, 3600.0, 100.0, command, **options)
    actual, info = propagate_cr3bp_state(state, 3600.0, 100.0, command, numeric_backend="rust", **options)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-8)
    assert info.method == py_info.method
    assert (info.accepted_steps, info.rejected_steps, info.attempted_steps) == (py_info.accepted_steps, py_info.rejected_steps, py_info.attempted_steps)
    assert info.attempted_steps == info.accepted_steps + info.rejected_steps
    if method == "dopri5":
        assert info.rejected_steps > 0
    assert info.suggested_next_step_s > 0


@pytest.mark.parametrize("method", ["rk4", "rkf78", "adaptive", "dopri5"])
def test_reference_stm_nonidentity_matrix_and_composition(native, method):
    state = cr3bp_halo_seed_state_km_s(family="nrho")
    initial_phi = np.eye(6) + np.arange(36).reshape(6, 6) * 1e-5
    opts = dict(integrator=method, adaptive_atol=1e-12, adaptive_rtol=1e-11, return_info=True)
    expected, phi_expected, py_info = propagate_cr3bp_reference_stm(state, initial_phi, 600.0, 0.0, **opts)
    actual, phi, info = propagate_cr3bp_reference_stm(state, initial_phi, 600.0, 0.0, numeric_backend="rust", **opts)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-8)
    np.testing.assert_allclose(phi, phi_expected, rtol=2e-10, atol=2e-9)
    identity_state, identity_phi = propagate_cr3bp_reference_stm(state, np.eye(6), 600.0, 0.0, integrator=method, adaptive_atol=1e-12, adaptive_rtol=1e-11, numeric_backend="rust")
    np.testing.assert_allclose(actual, identity_state, rtol=0, atol=1e-8)
    np.testing.assert_allclose(phi, identity_phi @ initial_phi, rtol=2e-10, atol=2e-9)
    assert (info is None) == (py_info is None) == (method == "rk4")
    if info is not None:
        assert info.method == py_info.method


@pytest.mark.parametrize("method", ["rkf78", "dopri5"])
def test_native_accuracy_against_dop853_and_variational_finite_difference(native, method):
    state = cr3bp_halo_seed_state_km_s(family="nrho")
    duration = 3600.0
    reference = solve_ivp(lambda t, x: cr3bp_derivative_physical(x), (0, duration), state, method="DOP853", rtol=3e-14, atol=1e-13)
    assert reference.success
    opts = dict(integrator=method, adaptive_atol=1e-13, adaptive_rtol=1e-12, numeric_backend="rust")
    actual, phi = propagate_cr3bp_reference_stm(state, np.eye(6), duration, 0.0, **opts)
    np.testing.assert_allclose(actual, reference.y[:, -1], rtol=0.0, atol=1e-8)
    for axis in range(6):
        delta = np.zeros(6)
        # Resolve small cross terms when subtracting ~100,000 km positions;
        # 1e-7 km/s makes the quotient dominated by floating-point cancellation.
        delta[axis] = 0.001 if axis < 3 else 1e-5
        plus = propagate_cr3bp_state(state + delta, duration, 0.0, **opts)
        minus = propagate_cr3bp_state(state - delta, duration, 0.0, **opts)
        np.testing.assert_allclose(phi[:, axis], (plus - minus) / (2 * delta[axis]), rtol=3e-5, atol=3e-5)


def test_unforced_jacobi_conservation(native):
    state = cr3bp_halo_seed_state_km_s(family="l1_northern_large")
    before = cr3bp_jacobi_constant(state)
    result = propagate_cr3bp_state(state, 86400.0, 0.0, integrator="rkf78", adaptive_atol=1e-12, adaptive_rtol=1e-11, numeric_backend="rust")
    assert abs(cr3bp_jacobi_constant(result) - before) < 1e-10


@pytest.mark.parametrize("method", ["rk4", "rkf78", "dopri5"])
def test_zero_duration_and_barycentric_origin(native, method):
    zero = np.zeros(6)
    actual, info = propagate_cr3bp_state(zero, 0.0, 0.0, integrator=method, numeric_backend="rust", return_info=True)
    np.testing.assert_array_equal(actual, zero)
    reference, phi = propagate_cr3bp_reference_stm(zero, np.eye(6), 0.0, 0.0, integrator=method, numeric_backend="rust")
    np.testing.assert_array_equal(reference, zero)
    np.testing.assert_array_equal(phi, np.eye(6))
    assert (info is None) == (method == "rk4")
    if info is not None:
        assert info.attempted_steps == info.accepted_steps == info.rejected_steps == 0


def test_native_rk4_reverse_step_is_preserved(native):
    state = cr3bp_halo_seed_state_km_s()
    expected = propagate_cr3bp_state(state, -60.0, 100.0)
    np.testing.assert_allclose(propagate_cr3bp_state(state, -60.0, 100.0, numeric_backend="rust"), expected, rtol=0, atol=1e-10)


@pytest.mark.parametrize("method", ["cr3bp_derivative", "cr3bp_jacobian"])
def test_native_primitives_reject_malformed_states(native, method):
    system = EARTH_MOON_CR3BP
    tail = (system.distance_km, system.mu, system.mean_motion_rad_s)
    if method == "cr3bp_derivative":
        tail += ([0.0] * 3,)
    function = getattr(native, method)
    with pytest.raises(ValueError, match="6 values"):
        function([0.0] * 5, *tail)
    for value in (np.nan, np.inf):
        with pytest.raises(ValueError, match="finite"):
            function([value] * 6, *tail)


@pytest.mark.parametrize("function", [propagate_cr3bp_state, propagate_cr3bp_reference_stm])
def test_native_invalid_inputs_and_old_wheel_fail_closed(native, monkeypatch, function):
    from sim.dynamics.orbit import rust_cr3bp
    state = cr3bp_halo_seed_state_km_s()
    args = (state, np.eye(6), 60.0, 0.0) if function is propagate_cr3bp_reference_stm else (state, 60.0, 0.0)
    with pytest.raises(ValueError, match="Unknown CR3BP"):
        function(*args, numeric_backend="typo")
    with pytest.raises(ValueError, match="Unsupported CR3BP integrator"):
        function(*args, numeric_backend="rust", integrator="typo")
    for value in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="tolerances"):
            function(*args, numeric_backend="rust", integrator="rkf78", adaptive_rtol=value)
    bad_args = list(args)
    bad_args[0] = np.full(6, np.nan)
    with pytest.raises(ValueError, match="finite"):
        function(*bad_args, numeric_backend="rust")
    monkeypatch.setattr(rust_cr3bp, "_extension", lambda: SimpleNamespace())
    with pytest.raises(RuntimeError, match="CR3BP-enabled"):
        function(*args, numeric_backend="rust")
    assert np.isfinite(function(*args, numeric_backend="python")[0] if function is propagate_cr3bp_reference_stm else function(*args, numeric_backend="python")).all()


@pytest.mark.parametrize("method", ["rk4", "rkf78", "dopri5"])
def test_runtime_dispatch_and_adaptive_rewind(native, method):
    p = OrbitPropagator(model="cr3bp", numeric_backend="rust", integrator=method, adaptive_atol=1e-12, adaptive_rtol=1e-11)
    state = cr3bp_halo_seed_state_km_s(family="nrho")
    command = np.array([1e-8, 0.0, 0.0])
    ctx = OrbitContext(mu_km3_s2=398600.0, mass_kg=100.0)
    first = p.propagate(state, 600.0, 0.0, command, {}, ctx)
    p.propagate(first, 600.0, 600.0, command, {}, ctx)
    np.testing.assert_array_equal(p.propagate(state, 600.0, 0.0, command, {}, ctx), first)
    assert p.last_numeric_path == "rust_native_cr3bp"
    if method != "rk4":
        assert p.adaptive_step_info.accepted_steps > p.last_adaptive_step_info.accepted_steps
        assert p.last_adaptive_step_info.method == method


def test_validated_scenario_session_uses_rust_cr3bp(native):
    source = Path(__file__).parents[2] / "configs/cr3bp_research.yaml"
    histories = {}
    for backend in ("python", "rust"):
        raw = SimulationConfig.from_yaml(source).to_dict()
        raw["simulator"]["duration_s"] = 600.0
        raw["simulator"]["dynamics"]["orbit"]["numeric_backend"] = backend
        cfg = SimulationConfig.from_dict(raw, source_path=source)
        session = SimulationSession.from_config(cfg, history_mode="dynamic")
        snap = session.reset()
        rows = [snap.truth["vehicle"]]
        while not session.done:
            snap = session.step()
            rows.append(snap.truth["vehicle"])
        histories[backend] = np.array(rows)
        if backend == "rust":
            assert session._engine.agents["vehicle"].dynamics.orbit_propagator.last_numeric_path == "rust_native_cr3bp"
    np.testing.assert_allclose(histories["rust"], histories["python"], rtol=0, atol=1e-8)


def test_scenario_rejects_wheel_without_cr3bp(native, monkeypatch):
    source = Path(__file__).parents[2] / "configs/cr3bp_research.yaml"
    raw = SimulationConfig.from_yaml(source).to_dict()
    raw["simulator"]["dynamics"]["orbit"]["numeric_backend"] = "rust"
    monkeypatch.delattr(native, "cr3bp_propagate")
    with pytest.raises(ValueError, match="CR3BP-enabled"):
        SimulationConfig.from_dict(raw)


def test_native_sampled_history_preserves_irregular_scalar_boundaries(native):
    function = getattr(native, "cr3bp_sampled_history_bytes", None)
    if function is None:
        pytest.skip("current optional wheel predates CR3BP sampled histories")
    initial = cr3bp_halo_seed_state_km_s(family="nrho")
    widths = np.asarray([0.3, 0.3, 0.1, 30.0, 0.05], dtype="<f8")
    system = EARTH_MOON_CR3BP
    tail = (system.distance_km, system.mu, system.mean_motion_rad_s)
    raw = function(initial.tolist(), widths.tobytes(), [3, 5], *tail)
    rows = np.frombuffer(raw, dtype="<f8").reshape(3, 6)
    expected = [initial]
    state = initial.copy()
    time = 0.0
    for index, width in enumerate(widths):
        state = propagate_cr3bp_state(state, width, time, numeric_backend="rust")
        time += width
        if index in {2, 4}:
            expected.append(state)
    np.testing.assert_array_equal(rows, expected)
    with pytest.raises(ValueError, match="strictly increasing"):
        function(initial.tolist(), widths.tobytes(), [2, 2], *tail)
    with pytest.raises(ValueError, match="float64"):
        function(initial.tolist(), b"bad", [1], *tail)
    with pytest.raises(ValueError, match="positive"):
        function(initial.tolist(), np.asarray([-0.1], dtype="<f8").tobytes(), [1], *tail)


@pytest.mark.parametrize("method", ["rk4", "rkf78", "dopri5"])
def test_native_stm_owned_buffer_preserves_results_diagnostics_and_legacy_fallback(native, monkeypatch, method):
    function = getattr(native, "cr3bp_propagate_stm_buffer", None)
    if function is None:
        pytest.skip("current optional wheel predates owned CR3BP STM outputs")
    state = cr3bp_halo_seed_state_km_s(family="nrho")
    phi = np.eye(6) + np.arange(36).reshape(6, 6) * 1e-5
    augmented = np.concatenate((state, phi.ravel()))
    system = EARTH_MOON_CR3BP
    args = (
        augmented.tolist(), 0.0, 600.0, system.distance_km, system.mu,
        system.mean_motion_rad_s, method, 1e-12, 1e-11, None,
    )
    expected, expected_info = native.cr3bp_propagate_stm(*args)
    owned, actual_info = function(*args, True)
    assert isinstance(owned, bytearray)
    view = np.frombuffer(owned, dtype="<f8")
    assert view.flags.writeable
    assert view.tobytes() == np.asarray(expected, dtype="<f8").tobytes()
    assert actual_info == expected_info
    suppressed, suppressed_info = function(*args, False)
    assert suppressed_info is None
    assert suppressed == owned
    del owned
    view[0] = 123.0
    assert view[0] == 123.0
    assert state[0] != 123.0
    options = dict(integrator=method, adaptive_atol=1e-12, adaptive_rtol=1e-11, numeric_backend="rust")
    buffered = propagate_cr3bp_reference_stm(state, phi, 600.0, 0.0, **options)
    monkeypatch.delattr(native, "cr3bp_propagate_stm_buffer")
    legacy = propagate_cr3bp_reference_stm(state, phi, 600.0, 0.0, **options)
    for actual, reference in zip(buffered, legacy, strict=True):
        assert actual.flags.writeable
        assert actual.tobytes() == reference.tobytes()
