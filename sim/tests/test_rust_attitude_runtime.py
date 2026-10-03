"""Runtime routing and built-in facet coverage for the optional Rust attitude path."""

from __future__ import annotations

import numpy as np
import pytest

from sim.config import scenario_config_from_dict
from sim.core.models import Command, StateTruth
from sim.dynamics.attitude.disturbances import DisturbanceTorqueConfig, DisturbanceTorqueModel
from sim.dynamics.attitude.rigid_body import (
    activate_attitude_guardrail_stats,
    get_attitude_guardrail_stats,
    new_attitude_guardrail_stats,
    propagate_attitude_exponential_map,
)
from sim.dynamics.model import OrbitalAttitudeDynamics
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2
from sim.dynamics.orbit.propagator import OrbitPropagator
from sim.dynamics.spacecraft_geometry import GeometryAreaProfile
from sim.runtime.satellite_factory import _create_satellite_runtime


def _rust_attitude_extension():
    extension = pytest.importorskip("oel_rust_orbit")
    required = {
        "attitude_rigid_body_derivatives",
        "attitude_propagate_exponential_map",
        "attitude_builtin_disturbance_torque",
        "attitude_propagate_builtin_disturbances",
    }
    if not required.issubset(set(dir(extension))):
        pytest.skip("installed wheel predates the Rust attitude kernels")
    return extension


def _factory_config(backend: str):
    return scenario_config_from_dict(
        {
            "scenario_name": "rust_attitude_runtime",
            "objects": {
                "sat": {
                    "enabled": True,
                    "kind": "satellite",
                    "runtime_profile": "trajectory_only",
                    "specs": {"mass_kg": 100.0},
                    "initial_state": {
                        "position_eci_km": [7000.0, 0.0, 0.0],
                        "velocity_eci_km_s": [0.0, 7.5, 0.0],
                        "attitude_quat_bn": [1.0, 0.0, 0.0, 0.0],
                        "angular_rate_body_rad_s": [0.0, 0.0, 0.05],
                    },
                }
            },
            "simulator": {
                "duration_s": 1.0,
                "dt_s": 1.0,
                "dynamics": {
                    "orbit": {"model": "two_body", "integrator": "rk4", "numeric_backend": "python"},
                    "attitude": {
                        "enabled": True,
                        "numeric_backend": backend,
                        "disturbance_torques": {
                            "gravity_gradient": True,
                            "magnetic": False,
                            "drag": False,
                            "srp": False,
                        },
                    },
                },
            },
            "outputs": {"mode": "save", "output_dir": "outputs/rust_attitude_runtime"},
        }
    )


def test_satellite_factory_wires_attitude_backend_to_model_and_disturbances() -> None:
    cfg = _factory_config("rust")
    runtime = _create_satellite_runtime("sat", cfg.objects["sat"], cfg, np.random.default_rng(0))

    assert runtime.dynamics.attitude_numeric_backend == "rust"
    assert runtime.dynamics.disturbance_model is not None
    assert runtime.dynamics.disturbance_model.config.numeric_backend == "rust"
    assert runtime.dynamics.disturbance_model._compiled_plan_supported is True
    assert runtime.dynamics.disturbance_model._compiled_has_disturbances is True


def test_facet_disturbance_plan_is_staged_for_native_kernel() -> None:
    config = DisturbanceTorqueConfig(
        use_gravity_gradient=False,
        use_magnetic=False,
        use_drag=True,
        use_srp=True,
        numeric_backend="rust",
        drag_facets=(
            {
                "area_m2": 1.4,
                "normal_body": [1.0, 0.0, 0.0],
                "cp_offset_body_m": [0.0, 0.2, 0.03],
                "drag_cd": 2.1,
            },
            {
                "area_m2": 0.8,
                "normal_body": [0.0, 1.0, 0.0],
                "cp_offset_body_m": [0.1, 0.0, -0.02],
                "cd": 2.3,
            },
        ),
        srp_facets=(
            {
                "area_m2": 1.1,
                "normal_body": [1.0, 0.0, 0.0],
                "cp_offset_body_m": [0.0, -0.1, 0.04],
            },
            {
                "area_m2": 0.6,
                "normal_body": [0.0, 0.0, 1.0],
                "cp_offset_body_m": [-0.04, 0.0, 0.0],
            },
        ),
    )
    model = DisturbanceTorqueModel(EARTH_MU_KM3_S2, np.diag([120.0, 100.0, 80.0]), config)

    assert model._compiled_plan_supported is True
    assert model._compiled_drag_mode == 2
    assert model._compiled_srp_mode == 2
    np.testing.assert_allclose(model._compiled_drag_facet_areas, [1.4, 0.8])
    np.testing.assert_allclose(model._compiled_srp_facet_areas, [1.1, 0.6])
    np.testing.assert_allclose(model._compiled_drag_facet_cd, [2.1, 2.3])


def test_unsupported_geometry_uses_python_fallback_when_rust_is_selected() -> None:
    profile = GeometryAreaProfile(
        directions_body=np.array([[1.0, 0.0, 0.0]]),
        projected_area_m2=np.array([1.0]),
        center_of_pressure_body_m=np.array([[0.0, 0.2, 0.0]]),
    )
    model = DisturbanceTorqueModel(
        EARTH_MU_KM3_S2,
        np.diag([120.0, 100.0, 80.0]),
        DisturbanceTorqueConfig(
            use_gravity_gradient=False,
            use_magnetic=False,
            use_drag=False,
            use_srp=True,
            geometry_area_profile=profile,
            numeric_backend="rust",
        ),
    )
    assert model._compiled_plan_supported is False
    result = model.try_propagate_compiled(
        quat_bn=np.array([1.0, 0.0, 0.0, 0.0]),
        omega_body_rad_s=np.zeros(3),
        command_torque_body_nm=np.zeros(3),
        position_eci_km=np.array([7000.0, 0.0, 0.0]),
        t_s=0.0,
        env={"sun_dir_eci_unit": np.array([1.0, 0.0, 0.0]), "srp_shadow_factor": 1.0},
        substeps_s=np.array([1.0]),
        acceleration_mode="auto",
        acceleration_enabled=True,
    )
    assert result is None


def test_native_attitude_guardrails_report_and_enforce_policy() -> None:
    _rust_attitude_extension()
    inertia = np.diag([120.0, 100.0, 80.0])
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    bad_rate = np.array([np.inf, 0.0, 0.0])
    bad_torque = np.array([np.nan, 0.0, 0.0])

    sanitize = new_attitude_guardrail_stats(policy="sanitize")
    activate_attitude_guardrail_stats(sanitize)
    q_next, omega_next = propagate_attitude_exponential_map(
        q0,
        bad_rate,
        inertia,
        bad_torque,
        0.1,
        numeric_backend="rust",
    )
    stats = get_attitude_guardrail_stats(sanitize)
    assert np.all(np.isfinite(q_next))
    assert np.all(np.isfinite(omega_next))
    assert stats["non_finite_input_events"] > 0
    assert stats["rate_clamp_events"] > 0
    assert stats["torque_clamp_events"] > 0

    strict = new_attitude_guardrail_stats(policy="error")
    activate_attitude_guardrail_stats(strict)
    with pytest.raises(FloatingPointError, match="non_finite_input_events"):
        propagate_attitude_exponential_map(
            q0,
            bad_rate,
            inertia,
            bad_torque,
            0.1,
            numeric_backend="rust",
        )
    activate_attitude_guardrail_stats(new_attitude_guardrail_stats(policy="sanitize"))


def _coupled_body_force_result(attitude_backend: str) -> StateTruth:
    dynamics = OrbitalAttitudeDynamics(
        mu_km3_s2=EARTH_MU_KM3_S2,
        inertia_kg_m2=np.diag([120.0, 100.0, 80.0]),
        orbit_propagator=OrbitPropagator(model="two_body", integrator="rk4", numeric_backend="python"),
        orbit_substep_s=0.5,
        attitude_substep_s=0.1,
        attitude_numeric_backend=attitude_backend,
        acceleration_mode="off",
    )
    state = StateTruth(
        position_eci_km=np.array([7000.0, 0.0, 0.0]),
        velocity_eci_km_s=np.array([0.0, 7.5, 0.0]),
        attitude_quat_bn=np.array([0.97, 0.08, -0.12, 0.18])
        / np.linalg.norm([0.97, 0.08, -0.12, 0.18]),
        angular_rate_body_rad_s=np.array([0.02, -0.015, 0.04]),
        mass_kg=100.0,
        t_s=0.0,
    )
    command = Command(
        torque_body_nm=np.array([0.03, -0.02, 0.01]),
        mode_flags={
            "physical_force_body_n": [0.8, -0.3, 0.5],
            "mass_flow_kg_s": 0.02,
            "min_mass_kg": 10.0,
        },
    )
    return dynamics.step(state, command, {}, 0.75)


def test_coupled_body_force_dispatch_matches_python_attitude_reference() -> None:
    _rust_attitude_extension()
    expected = _coupled_body_force_result("python")
    actual = _coupled_body_force_result("rust")

    np.testing.assert_allclose(actual.position_eci_km, expected.position_eci_km, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(actual.velocity_eci_km_s, expected.velocity_eci_km_s, rtol=0.0, atol=2.0e-12)
    if float(np.dot(actual.attitude_quat_bn, expected.attitude_quat_bn)) < 0.0:
        actual.attitude_quat_bn *= -1.0
    np.testing.assert_allclose(actual.attitude_quat_bn, expected.attitude_quat_bn, rtol=0.0, atol=2.0e-12)
    np.testing.assert_allclose(
        actual.angular_rate_body_rad_s,
        expected.angular_rate_body_rad_s,
        rtol=0.0,
        atol=2.0e-12,
    )
    assert actual.mass_kg == pytest.approx(expected.mass_kg, rel=0.0, abs=1.0e-12)


def test_facet_disturbance_substeps_match_python_reference() -> None:
    _rust_attitude_extension()
    base = dict(
        use_gravity_gradient=False,
        use_magnetic=False,
        use_drag=True,
        use_srp=True,
        drag_area_m2=1.5,
        drag_cd=2.2,
        srp_area_m2=1.0,
        srp_cr=1.3,
        drag_facets=(
            {"area_m2": 1.2, "normal_body": [1.0, 0.0, 0.0], "cp_offset_body_m": [0.0, 0.15, 0.01]},
            {"area_m2": 0.7, "normal_body": [0.0, 1.0, 0.0], "cp_offset_body_m": [0.08, 0.0, -0.02], "cd": 2.5},
        ),
        srp_facets=(
            {"area_m2": 0.9, "normal_body": [1.0, 0.0, 0.0], "cp_offset_body_m": [0.0, -0.06, 0.02]},
            {"area_m2": 0.5, "normal_body": [0.0, 0.0, 1.0], "cp_offset_body_m": [-0.03, 0.0, 0.0]},
        ),
    )
    models = [
        DisturbanceTorqueModel(
            EARTH_MU_KM3_S2,
            np.diag([120.0, 100.0, 80.0]),
            DisturbanceTorqueConfig(**base, numeric_backend=backend),
        )
        for backend in ("python", "rust")
    ]
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
            "sun_dir_eci_unit": np.array([0.8, 0.2, 0.56]),
            "srp_shadow_factor": 0.82,
            "srp_pressure_n_m2": 4.56e-6,
            "srp_distance_scale": 0.97,
        },
        "substeps_s": np.array([0.2, 0.2, 0.2, 0.2]),
        "acceleration_mode": "auto",
        "acceleration_enabled": True,
    }
    expected = models[0].try_propagate_compiled(**kwargs)
    actual = models[1].try_propagate_compiled(**kwargs)
    assert expected is not None and actual is not None
    if float(np.dot(actual[0], expected[0])) < 0.0:
        actual = (-actual[0], actual[1])
    np.testing.assert_allclose(actual[0], expected[0], rtol=0.0, atol=2.0e-11)
    np.testing.assert_allclose(actual[1], expected[1], rtol=0.0, atol=2.0e-12)
