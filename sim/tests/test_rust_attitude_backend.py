from __future__ import annotations

import numpy as np
import pytest

from sim.dynamics.attitude.disturbances import DisturbanceTorqueConfig, DisturbanceTorqueModel
from sim.dynamics.attitude.rigid_body import propagate_attitude_exponential_map


def _rust_extension():
    extension = pytest.importorskip("oel_rust_orbit")
    required = {
        "attitude_propagate_exponential_map",
        "attitude_rigid_body_derivatives",
        "attitude_propagate_builtin_disturbances",
    }
    if not required.issubset(set(dir(extension))):
        pytest.skip("installed wheel predates the Rust attitude kernels")
    return extension


def test_rust_exponential_map_matches_python_reference() -> None:
    _rust_extension()
    inertia = np.diag([120.0, 100.0, 80.0])
    q0 = np.array([0.91, 0.12, -0.27, 0.22], dtype=float)
    q0 /= np.linalg.norm(q0)
    omega = np.array([0.03, -0.02, 0.11])
    torque = np.array([0.02, -0.01, 0.004])
    expected = propagate_attitude_exponential_map(
        q0,
        omega,
        inertia,
        torque,
        0.75,
        acceleration_mode="off",
        numeric_backend="python",
    )
    actual = propagate_attitude_exponential_map(
        q0,
        omega,
        inertia,
        torque,
        0.75,
        numeric_backend="rust",
    )
    if np.dot(expected[0], actual[0]) < 0.0:
        actual = (-actual[0], actual[1])
    np.testing.assert_allclose(actual[0], expected[0], rtol=0.0, atol=2.0e-12)
    np.testing.assert_allclose(actual[1], expected[1], rtol=0.0, atol=2.0e-12)


def test_rust_builtin_disturbance_substeps_match_python_reference() -> None:
    _rust_extension()
    inertia = np.diag([120.0, 100.0, 80.0])
    config_kwargs = dict(
        use_gravity_gradient=True,
        use_magnetic=True,
        use_drag=True,
        use_srp=True,
        magnetic_dipole_body_a_m2=np.array([0.04, -0.01, 0.02]),
        drag_area_m2=1.5,
        drag_cd=2.2,
        drag_cp_offset_body_m=np.array([0.05, 0.02, -0.01]),
        srp_area_m2=1.0,
        srp_cr=1.3,
        srp_cp_offset_body_m=np.array([-0.02, 0.03, 0.01]),
    )
    python_model = DisturbanceTorqueModel(
        398600.4418,
        inertia,
        DisturbanceTorqueConfig(**config_kwargs, numeric_backend="python"),
    )
    rust_model = DisturbanceTorqueModel(
        398600.4418,
        inertia,
        DisturbanceTorqueConfig(**config_kwargs, numeric_backend="rust"),
    )
    kwargs = {
        "quat_bn": np.array([0.98, 0.04, -0.08, 0.17]),
        "omega_body_rad_s": np.array([0.02, -0.01, 0.04]),
        "command_torque_body_nm": np.array([0.003, -0.002, 0.001]),
        "position_eci_km": np.array([6800.0, -120.0, 250.0]),
        "t_s": 12.0,
        "env": {
            "density_kg_m3": 1.2e-12,
            "drag_v_rel_eci_m_s": np.array([1200.0, -300.0, 80.0]),
            "drag_v_rel_norm_m_s": float(np.linalg.norm([1200.0, -300.0, 80.0])),
            "magnetic_field_eci_t": np.array([1.5e-5, -2.0e-5, 3.0e-5]),
            "sun_dir_eci_unit": np.array([0.8, 0.2, 0.56]),
            "srp_shadow_factor": 0.82,
            "srp_pressure_n_m2": 4.56e-6,
            "srp_distance_scale": 0.97,
        },
        "substeps_s": np.array([0.2, 0.2, 0.2, 0.2]),
        "acceleration_mode": "auto",
        "acceleration_enabled": True,
    }
    expected = python_model.try_propagate_compiled(**kwargs)
    actual = rust_model.try_propagate_compiled(**kwargs)
    assert expected is not None and actual is not None
    np.testing.assert_allclose(actual[0], expected[0], rtol=0.0, atol=2.0e-11)
    np.testing.assert_allclose(actual[1], expected[1], rtol=0.0, atol=2.0e-12)
