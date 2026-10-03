from __future__ import annotations

import numpy as np
import pytest

from sim.aero.rocket import RocketAeroConfig, compute_aero_loads, compute_aero_state
from sim.dynamics.reentry import (
    ReentryConfig,
    ReentryObjectProperties,
    reentry_metrics_for_state,
)
from sim.presets.rockets import RocketStackPreset, RocketStagePreset
from sim.rocket import (
    GuidanceCommand,
    HoldAttitudeGuidance,
    RocketAscentSimulator,
    RocketSimConfig,
    RocketVehicleConfig,
)
from sim.rust_vehicle_backend import (
    collision_chord_geometry,
    collision_elastic_impact,
    collision_linear_contact_fraction,
    reentry_metrics,
    rocket_propellant_step,
    rocket_stage_engine_perf,
)


def _native_or_skip():
    native = pytest.importorskip("oel_rust_orbit")
    required = {
        "rocket_aero_state",
        "rocket_aero_loads",
        "rocket_stage_engine_perf",
        "rocket_propellant_step",
        "reentry_metrics",
        "collision_chord_geometry",
        "collision_linear_contact_fraction",
        "collision_elastic_impact",
    }
    missing = sorted(name for name in required if not hasattr(native, name))
    if missing:
        pytest.skip("installed Rust wheel predates vehicle/collision kernels: " + ", ".join(missing))
    return native


def test_aero_state_and_loads_match_python_reference():
    _native_or_skip()
    velocity = np.array([1850.0, 115.0, -240.0])
    cfg = RocketAeroConfig(
        reference_area_m2=4.2,
        reference_length_m=17.0,
        cp_offset_body_m=np.array([-1.2, 0.3, 0.4]),
        cd_base=0.21,
        cd_alpha2=0.12,
        cd_supersonic=0.31,
        transonic_peak_cd=0.19,
        transonic_width=0.18,
        cl_alpha_per_rad=0.17,
        cy_beta_per_rad=0.14,
        cm_alpha_per_rad=-0.03,
        cn_beta_per_rad=-0.025,
        cl_roll_per_rad=-0.012,
    )
    python_state = compute_aero_state(
        rho_kg_m3=0.018,
        pressure_pa=2200.0,
        temperature_k=225.0,
        sound_speed_m_s=300.0,
        v_rel_body_m_s=velocity,
        alpha_limit_deg=20.0,
        beta_limit_deg=20.0,
        numeric_backend="python",
    )
    rust_state = compute_aero_state(
        rho_kg_m3=0.018,
        pressure_pa=2200.0,
        temperature_k=225.0,
        sound_speed_m_s=300.0,
        v_rel_body_m_s=velocity,
        alpha_limit_deg=20.0,
        beta_limit_deg=20.0,
        numeric_backend="rust",
    )
    np.testing.assert_allclose(
        [
            rust_state.rho_kg_m3,
            rust_state.pressure_pa,
            rust_state.temperature_k,
            rust_state.sound_speed_m_s,
            rust_state.dynamic_pressure_pa,
            rust_state.speed_m_s,
            rust_state.mach,
            rust_state.alpha_rad,
            rust_state.beta_rad,
        ],
        [
            python_state.rho_kg_m3,
            python_state.pressure_pa,
            python_state.temperature_k,
            python_state.sound_speed_m_s,
            python_state.dynamic_pressure_pa,
            python_state.speed_m_s,
            python_state.mach,
            python_state.alpha_rad,
            python_state.beta_rad,
        ],
        rtol=2e-14,
        atol=2e-12,
    )
    python_loads = compute_aero_loads(velocity, python_state, cfg, numeric_backend="python")
    rust_loads = compute_aero_loads(velocity, rust_state, cfg, numeric_backend="rust")
    np.testing.assert_allclose(rust_loads.force_body_n, python_loads.force_body_n, rtol=2e-14, atol=2e-10)
    np.testing.assert_allclose(rust_loads.moment_body_nm, python_loads.moment_body_nm, rtol=2e-14, atol=2e-10)
    np.testing.assert_allclose(rust_loads.coeff_force_body, python_loads.coeff_force_body, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(rust_loads.coeff_moment_body, python_loads.coeff_moment_body, rtol=2e-14, atol=2e-14)
    assert rust_loads.drag_coefficient == pytest.approx(python_loads.drag_coefficient, rel=2e-14, abs=2e-14)


def test_stage_propellant_and_reentry_scalars_match_python():
    _native_or_skip()
    stage_perf = rocket_stage_engine_perf(
        pressure_pa=19000.0,
        sea_level_thrust_n=1.2e6,
        vacuum_thrust_n=1.4e6,
        sea_level_isp_s=280.0,
        vacuum_isp_s=320.0,
    )
    pressure_weight = 19000.0 / 101325.0
    expected_perf = (
        pressure_weight * 1.2e6 + (1.0 - pressure_weight) * 1.4e6,
        pressure_weight * 280.0 + (1.0 - pressure_weight) * 320.0,
    )
    np.testing.assert_allclose(stage_perf, expected_perf, rtol=2e-14, atol=1e-10)
    native_step = rocket_propellant_step(
        propellant_left_kg=350.0,
        throttle=0.78,
        pressure_pa=19000.0,
        dt_s=0.5,
        mass_start_kg=1400.0,
        sea_level_thrust_n=1.2e6,
        vacuum_thrust_n=1.4e6,
        sea_level_isp_s=280.0,
        vacuum_isp_s=320.0,
    )
    thrust = 0.78 * expected_perf[0]
    dm_full = thrust / (expected_perf[1] * 9.80665) * 0.5
    expected_step = np.array(
        [
            thrust,
            expected_perf[1],
            dm_full,
            1.0,
            1400.0 - 0.5 * dm_full,
            350.0 - dm_full,
            1400.0 - dm_full,
        ]
    )
    np.testing.assert_allclose(native_step, expected_step, rtol=2e-14, atol=1e-10)
    python_metrics = reentry_metrics_for_state(
        r_eci_km=np.array([6878.137, 0.0, 0.0]),
        v_eci_km_s=np.array([0.0, 4.2, 0.0]),
        t_s=0.0,
        dt_s=0.5,
        cfg=ReentryConfig(numeric_backend="python"),
        props=ReentryObjectProperties(120.0, 2.4, 1.6, 0.7, 1.8, 0.25),
        env={"density_kg_m3": 0.002, "drag_frame_model": "inertial_z"},
        active=True,
        previous_heat_load_j_m2=4.0e6,
        previous_heat_rate_w_m2=1.2e7,
    )
    native_metrics = reentry_metrics(
        density_kg_m3=0.002,
        speed_m_s=python_metrics["relative_speed_m_s"],
        mass_kg=120.0,
        drag_area_m2=2.4,
        cd=1.6,
        lift_area_m2=1.8,
        cl=0.25,
        nose_radius_m=0.7,
        coefficient=1.83e-4,
        dt_s=0.5,
        previous_heat_load_j_m2=4.0e6,
        previous_heat_rate_w_m2=1.2e7,
    )
    expected_metrics = np.array(
        [
            python_metrics["dynamic_pressure_pa"],
            python_metrics["drag_decel_m_s2"],
            python_metrics["lift_accel_m_s2"],
            python_metrics["lift_to_drag"],
            python_metrics["g_load"],
            python_metrics["heat_rate_w_m2"],
            python_metrics["heat_load_j_m2"],
        ]
    )
    np.testing.assert_allclose(native_metrics, expected_metrics, rtol=5e-13, atol=1e-8, equal_nan=True)
    rust_dispatch_metrics = reentry_metrics_for_state(
        r_eci_km=np.array([6878.137, 0.0, 0.0]),
        v_eci_km_s=np.array([0.0, 4.2, 0.0]),
        t_s=0.0,
        dt_s=0.5,
        cfg=ReentryConfig(numeric_backend="rust"),
        props=ReentryObjectProperties(120.0, 2.4, 1.6, 0.7, 1.8, 0.25),
        env={"density_kg_m3": 0.002, "drag_frame_model": "inertial_z"},
        active=True,
        previous_heat_load_j_m2=4.0e6,
        previous_heat_rate_w_m2=1.2e7,
    )
    np.testing.assert_allclose(
        [rust_dispatch_metrics[key] for key in (
            "dynamic_pressure_pa",
            "drag_decel_m_s2",
            "lift_accel_m_s2",
            "lift_to_drag",
            "g_load",
            "heat_rate_w_m2",
            "heat_load_j_m2",
        )],
        expected_metrics,
        rtol=5e-13,
        atol=1e-8,
        equal_nan=True,
    )


def test_collision_geometry_and_impulse_match_reference():
    _native_or_skip()
    start = np.array([0.01, 0.002, 0.0])
    end = np.array([-0.005, -0.001, 0.0])
    fraction, miss, norm2 = collision_chord_geometry(start, end)
    delta = end - start
    expected_fraction = np.clip(-np.dot(start, delta) / np.dot(delta, delta), 0.0, 1.0)
    assert fraction == pytest.approx(expected_fraction, rel=0.0, abs=2e-15)
    assert miss == pytest.approx(np.linalg.norm(start + expected_fraction * delta), rel=0.0, abs=2e-15)
    assert norm2 == pytest.approx(float(np.dot(delta, delta)), rel=0.0, abs=2e-15)
    assert collision_linear_contact_fraction(start, end, 0.004) is not None
    post_a, post_b, normal, closing = collision_elastic_impact(
        position_a=np.array([7000.0, 0.0, 0.0]),
        position_b=np.array([7000.004, 0.0, 0.0]),
        velocity_a=np.array([0.01, 0.002, 0.0]),
        velocity_b=np.array([-0.01, -0.001, 0.0]),
        mass_a_kg=100.0,
        mass_b_kg=200.0,
    )
    expected_normal = np.array([1.0, 0.0, 0.0])
    expected_closing = 0.02
    impulse = 2.0 * expected_closing / (1.0 / 100.0 + 1.0 / 200.0)
    expected_a = np.array([0.01, 0.002, 0.0]) - impulse * expected_normal / 100.0
    expected_b = np.array([-0.01, -0.001, 0.0]) + impulse * expected_normal / 200.0
    np.testing.assert_allclose(post_a, expected_a, rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(post_b, expected_b, rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(normal, expected_normal, rtol=0.0, atol=2e-15)
    assert closing == pytest.approx(expected_closing, rel=0.0, abs=2e-15)


def test_multi_step_rocket_burnout_and_attitude_dispatch_match() -> None:
    native = _native_or_skip()
    if not hasattr(native, "attitude_propagate_exponential_map"):
        pytest.skip("installed Rust wheel predates the Rust attitude kernel")
    first = RocketStagePreset(
        name="burnout-1",
        dry_mass_kg=4.0,
        propellant_mass_kg=2.0,
        max_thrust_n=5000.0,
        isp_s=220.0,
        burn_time_s=1.0,
        diameter_m=1.0,
        length_m=3.0,
    )
    second = RocketStagePreset(
        name="burnout-2",
        dry_mass_kg=2.0,
        propellant_mass_kg=3.0,
        max_thrust_n=2500.0,
        isp_s=240.0,
        burn_time_s=2.0,
        diameter_m=0.8,
        length_m=2.0,
    )
    vehicle = RocketVehicleConfig(
        stack=RocketStackPreset(name="multi-stage-parity", stages=(first, second)),
        payload_mass_kg=1.0,
    )
    common = dict(
        dt_s=0.25,
        max_time_s=5.0,
        enable_drag=False,
        enable_j2=False,
        enable_j3=False,
        enable_j4=False,
        aero=RocketAeroConfig(enabled=False),
        attitude_substep_s=0.05,
        tvc_pivot_offset_body_m=np.array([0.0, 1.0, 0.0]),
    )
    python_sim = RocketAscentSimulator(
        sim_cfg=RocketSimConfig(**common),
        vehicle_cfg=vehicle,
        guidance=HoldAttitudeGuidance(throttle=1.0),
    )
    rust_sim = RocketAscentSimulator(
        sim_cfg=RocketSimConfig(**common, numeric_backend="rust"),
        vehicle_cfg=vehicle,
        guidance=HoldAttitudeGuidance(throttle=1.0),
    )
    assert rust_sim._propagator.numeric_backend == "rust"
    python_state = python_sim.initial_state()
    rust_state = rust_sim.initial_state()
    separation_steps = 0
    for _ in range(20):
        command = GuidanceCommand(throttle=1.0)
        python_state = python_sim.step(python_state, command, dt_s=0.25)
        rust_state = rust_sim.step(rust_state, command, dt_s=0.25)
        np.testing.assert_allclose(rust_state.position_eci_km, python_state.position_eci_km, rtol=0.0, atol=2e-12)
        np.testing.assert_allclose(rust_state.velocity_eci_km_s, python_state.velocity_eci_km_s, rtol=0.0, atol=2e-12)
        np.testing.assert_allclose(rust_state.attitude_quat_bn, python_state.attitude_quat_bn, rtol=0.0, atol=2e-12)
        np.testing.assert_allclose(rust_state.angular_rate_body_rad_s, python_state.angular_rate_body_rad_s, rtol=0.0, atol=2e-12)
        assert rust_state.active_stage_index == python_state.active_stage_index
        assert rust_state._last_step_stage_sep == python_state._last_step_stage_sep
        if rust_state._last_step_stage_sep:
            separation_steps += 1
    assert separation_steps == 2
    assert rust_state.active_stage_index == 2
    assert np.linalg.norm(rust_state.angular_rate_body_rad_s) > 0.0
    assert rust_state.mass_kg == pytest.approx(python_state.mass_kg, rel=0.0, abs=2e-12)


def test_multi_step_reentry_heat_history_dispatch_matches() -> None:
    _native_or_skip()
    common = dict(
        r_eci_km=np.array([6878.137, 0.0, 0.0]),
        v_eci_km_s=np.array([0.0, 7.6, 0.0]),
        t_s=0.0,
        dt_s=0.5,
        props=ReentryObjectProperties(120.0, 2.4, 1.6, 0.7, 1.8, 0.25),
        env={"density_kg_m3": 0.002, "drag_frame_model": "inertial_z"},
        active=True,
    )
    python_heat = 0.0
    rust_heat = 0.0
    python_rate = None
    rust_rate = None
    for _ in range(6):
        python_metrics = reentry_metrics_for_state(
            cfg=ReentryConfig(numeric_backend="python"),
            previous_heat_load_j_m2=python_heat,
            previous_heat_rate_w_m2=python_rate,
            **common,
        )
        rust_metrics = reentry_metrics_for_state(
            cfg=ReentryConfig(numeric_backend="rust"),
            previous_heat_load_j_m2=rust_heat,
            previous_heat_rate_w_m2=rust_rate,
            **common,
        )
        for key in (
            "dynamic_pressure_pa",
            "drag_decel_m_s2",
            "lift_accel_m_s2",
            "lift_to_drag",
            "g_load",
            "heat_rate_w_m2",
            "heat_load_j_m2",
        ):
            assert rust_metrics[key] == pytest.approx(python_metrics[key], rel=2e-13, abs=1e-8, nan_ok=True)
        python_heat = python_metrics["heat_load_j_m2"]
        rust_heat = rust_metrics["heat_load_j_m2"]
        python_rate = python_metrics["heat_rate_w_m2"]
        rust_rate = rust_metrics["heat_rate_w_m2"]
    assert python_heat > 0.0
    assert rust_heat == pytest.approx(python_heat, rel=0.0, abs=1e-8)
