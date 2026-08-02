from __future__ import annotations

import math

import pytest

from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2, EARTH_RADIUS_KM
from sim.orbital_calculator.core import (
    altitude_from_period,
    apogee_perigee_from_elements,
    apogee_perigee_velocities_from_altitudes,
    atmospheric_density_from_altitude,
    ballistic_coefficient,
    circular_eclipse_estimate,
    circular_equatorial_elements_to_rv,
    circular_inclined_elements_to_rv,
    circular_orbit_drag_decay_rate,
    circular_orbit_from_altitude,
    classical_coe_to_rv,
    combined_speed_plane_change_delta_v,
    drag_force_acceleration_from_altitude,
    drag_force_acceleration_from_density_speed,
    drag_lifetime_range_estimate,
    elements_from_apogee_perigee_altitudes,
    entry_interface_estimate,
    equatorial_elliptical_elements_to_rv,
    geosynchronous_orbit,
    ground_track_drift_from_altitude,
    hcw_intrack_drift_estimate,
    hcw_natural_motion_from_altitude,
    hohmann_rendezvous_phase_angle,
    hohmann_rendezvous_wait_time,
    hohmann_transfer_between_circular_orbits,
    inclination_change_from_altitude,
    j2_secular_rates_from_altitude,
    j2_secular_rates_from_elements,
    mission_recovery_from_intrack_impulse,
    phasing_drift_from_altitude_change,
    plane_change_delta_v,
    repeat_ground_track_approximation,
    rocket_equation_delta_v,
    rocket_equation_mass_ratio,
    rv_to_robust_elements,
    sun_synchronous_inclination_from_altitude,
    vis_viva_velocity,
)
from sim.orbital_calculator.interactive import (
    CALCULATORS,
    CATEGORIES,
    calculators_for_category,
    format_result,
    prompt_yes_no,
    run_calculation,
)


def test_circular_orbit_from_altitude_reports_expected_leo_values() -> None:
    result = circular_orbit_from_altitude(400.0)

    assert result.radius_km == pytest.approx(EARTH_RADIUS_KM + 400.0)
    assert result.velocity_km_s == pytest.approx(math.sqrt(EARTH_MU_KM3_S2 / result.radius_km))
    assert result.period_min == pytest.approx(92.56, abs=0.01)
    assert result.escape_velocity_km_s == pytest.approx(math.sqrt(2.0) * result.velocity_km_s)


def test_altitude_from_period_round_trips_circular_period() -> None:
    circular = circular_orbit_from_altitude(700.0)
    result = altitude_from_period(circular.period_s)

    assert result.semi_major_axis_km == pytest.approx(circular.radius_km)
    assert result.altitude_km == pytest.approx(700.0)
    assert result.mean_motion_rad_s == pytest.approx(circular.mean_motion_rad_s)


def test_vis_viva_matches_circular_orbit_when_radius_equals_semimajor_axis() -> None:
    circular = circular_orbit_from_altitude(400.0)
    result = vis_viva_velocity(radius_km=circular.radius_km, semi_major_axis_km=circular.radius_km)

    assert result.velocity_km_s == pytest.approx(circular.velocity_km_s)


def test_geosynchronous_orbit_uses_sidereal_day() -> None:
    result = geosynchronous_orbit()

    assert result.period_hr == pytest.approx(23.9344696)
    assert result.semi_major_axis_km == pytest.approx(42164.17, abs=0.01)
    assert result.altitude_km == pytest.approx(35786.03, abs=0.01)
    assert result.circular_velocity_km_s == pytest.approx(3.07466, abs=1.0e-5)


def test_apogee_perigee_from_elements_and_inverse_altitudes_round_trip() -> None:
    apogee_perigee = apogee_perigee_from_elements(semi_major_axis_km=8000.0, eccentricity=0.1)
    inverse = elements_from_apogee_perigee_altitudes(
        perigee_altitude_km=apogee_perigee.perigee_altitude_km,
        apogee_altitude_km=apogee_perigee.apogee_altitude_km,
    )

    assert apogee_perigee.perigee_radius_km == pytest.approx(7200.0)
    assert apogee_perigee.apogee_radius_km == pytest.approx(8800.0)
    assert inverse.semi_major_axis_km == pytest.approx(8000.0)
    assert inverse.eccentricity == pytest.approx(0.1)


def test_apogee_perigee_velocities_use_vis_viva() -> None:
    result = apogee_perigee_velocities_from_altitudes(perigee_altitude_km=400.0, apogee_altitude_km=800.0)

    assert result.perigee_velocity_km_s == pytest.approx(7.77768, abs=1.0e-5)
    assert result.apogee_velocity_km_s == pytest.approx(7.34427, abs=1.0e-5)
    assert result.perigee_velocity_km_s > result.apogee_velocity_km_s


def test_hohmann_transfer_between_leo_altitudes_has_two_burns_and_transfer_time() -> None:
    result = hohmann_transfer_between_circular_orbits(400.0, 800.0)

    assert result.first_burn_delta_v_m_s == pytest.approx(109.12, abs=0.02)
    assert result.second_burn_delta_v_m_s == pytest.approx(107.56, abs=0.02)
    assert result.total_delta_v_m_s == pytest.approx(216.68, abs=0.02)
    assert result.transfer_time_min == pytest.approx(48.34, abs=0.01)


def test_hohmann_rendezvous_phase_angle_reports_synodic_period() -> None:
    result = hohmann_rendezvous_phase_angle(initial_altitude_km=400.0, target_altitude_km=800.0)

    assert result.transfer_time_min == pytest.approx(48.34, abs=0.01)
    assert result.required_phase_angle_deg == pytest.approx(7.470, abs=0.001)
    assert result.target_travel_during_transfer_deg == pytest.approx(172.530, abs=0.001)
    assert result.relative_phase_rate_deg_s < 0.0
    assert result.synodic_period_min == pytest.approx(1_123.15, abs=0.01)


def test_hohmann_rendezvous_wait_time_uses_relative_phase_direction() -> None:
    phase = hohmann_rendezvous_phase_angle(initial_altitude_km=400.0, target_altitude_km=800.0)
    ready = hohmann_rendezvous_wait_time(
        initial_altitude_km=400.0,
        target_altitude_km=800.0,
        current_phase_angle_deg=phase.required_phase_angle_deg,
    )
    from_conjunction = hohmann_rendezvous_wait_time(
        initial_altitude_km=400.0,
        target_altitude_km=800.0,
        current_phase_angle_deg=0.0,
    )

    assert ready.wait_time_s == pytest.approx(0.0)
    assert from_conjunction.wait_time_hr == pytest.approx(18.33, abs=0.01)


def test_plane_change_delta_v_uses_chord_formula() -> None:
    result = plane_change_delta_v(speed_km_s=7.5, angle_deg=10.0)

    assert result.delta_v_km_s == pytest.approx(2.0 * 7.5 * math.sin(math.radians(5.0)))
    assert result.delta_v_m_s == pytest.approx(1307.34, abs=0.01)


def test_inclination_change_from_altitude_uses_circular_speed() -> None:
    circular = circular_orbit_from_altitude(400.0)
    result = inclination_change_from_altitude(altitude_km=400.0, angle_deg=10.0)

    assert result.speed_km_s == pytest.approx(circular.velocity_km_s)
    assert result.delta_v_m_s == pytest.approx(1336.72, abs=0.01)


def test_combined_speed_plane_change_delta_v_uses_vector_law() -> None:
    result = combined_speed_plane_change_delta_v(initial_speed_km_s=7.5, final_speed_km_s=7.7, angle_deg=10.0)

    expected = math.sqrt(7.5**2 + 7.7**2 - 2.0 * 7.5 * 7.7 * math.cos(math.radians(10.0)))
    assert result.delta_v_km_s == pytest.approx(expected)
    assert result.delta_v_m_s == pytest.approx(1339.67, abs=0.01)


def test_sun_synchronous_inclination_from_altitude_matches_first_order_j2_estimate() -> None:
    result = sun_synchronous_inclination_from_altitude(700.0)

    assert result.inclination_deg == pytest.approx(98.188, abs=0.001)
    assert result.nodal_precession_deg_day == pytest.approx(result.target_precession_deg_day)


def test_j2_secular_rates_from_elements_match_first_order_formulas() -> None:
    result = j2_secular_rates_from_elements(
        semi_major_axis_km=7000.0,
        eccentricity=0.01,
        inclination_deg=45.0,
    )
    from_altitude = j2_secular_rates_from_altitude(
        altitude_km=7000.0 - EARTH_RADIUS_KM,
        eccentricity=0.01,
        inclination_deg=45.0,
    )

    assert result.semi_latus_rectum_km == pytest.approx(6999.3)
    assert result.orbital_period_min == pytest.approx(97.142, abs=0.001)
    assert result.raan_rate_deg_day == pytest.approx(-5.08852, abs=1.0e-5)
    assert result.argument_of_perigee_rate_deg_day == pytest.approx(5.39719, abs=1.0e-5)
    assert result.mean_anomaly_j2_rate_deg_day == pytest.approx(1.79897, abs=1.0e-5)
    assert from_altitude.raan_rate_deg_day == pytest.approx(result.raan_rate_deg_day)


def test_phasing_drift_from_lower_altitude_moves_ahead() -> None:
    result = phasing_drift_from_altitude_change(reference_altitude_km=400.0, phasing_altitude_change_km=-10.0)

    assert result.phasing_altitude_km == pytest.approx(390.0)
    assert result.phasing_mean_motion_rad_s > result.reference_mean_motion_rad_s
    assert result.drift_rate_m_s == pytest.approx(17.002, abs=0.001)
    assert result.drift_per_reference_orbit_km == pytest.approx(94.422, abs=0.001)
    assert result.lap_time_hr == pytest.approx(695.81, abs=0.01)


def test_hcw_natural_motion_and_drift_estimate_use_linearized_circular_motion() -> None:
    natural = hcw_natural_motion_from_altitude(400.0)
    drift = hcw_intrack_drift_estimate(
        altitude_km=400.0,
        radial_offset_m=10.0,
        intrack_velocity_bias_m_s=0.01,
        duration_orbits=1.0,
    )

    assert natural.natural_motion_period_min == pytest.approx(92.56, abs=0.01)
    assert natural.mean_motion_rad_s == pytest.approx(circular_orbit_from_altitude(400.0).mean_motion_rad_s)
    assert drift.radial_offset_drift_m == pytest.approx(-376.991, abs=0.001)
    assert drift.intrack_velocity_drift_m == pytest.approx(-166.609, abs=0.001)
    assert drift.total_intrack_drift_m == pytest.approx(-543.600, abs=0.001)


def test_circular_eclipse_estimate_handles_shadow_and_no_eclipse_cases() -> None:
    eclipse = circular_eclipse_estimate(altitude_km=400.0, beta_angle_deg=0.0)
    no_eclipse = circular_eclipse_estimate(altitude_km=400.0, beta_angle_deg=80.0)

    assert eclipse.beta_critical_deg == pytest.approx(70.218, abs=0.001)
    assert eclipse.eclipse_duration_min == pytest.approx(36.108, abs=0.001)
    assert eclipse.eclipse_fraction == pytest.approx(0.39010, abs=1.0e-5)
    assert no_eclipse.eclipse_half_angle_deg is None
    assert no_eclipse.eclipse_duration_s == pytest.approx(0.0)


def test_ground_track_drift_and_repeat_track_approximation() -> None:
    drift = ground_track_drift_from_altitude(400.0)
    repeat = repeat_ground_track_approximation(altitude_km=400.0, max_days=20.0)

    assert drift.earth_rotation_deg_per_orbit == pytest.approx(23.203, abs=0.001)
    assert drift.westward_drift_deg_per_orbit == pytest.approx(-23.203, abs=0.001)
    assert drift.orbits_per_sidereal_day == pytest.approx(15.51493, abs=1.0e-5)
    assert repeat.repeat_days == 2
    assert repeat.repeat_orbits == 31
    assert repeat.ground_track_error_deg == pytest.approx(10.74943, abs=1.0e-5)
    assert repeat.ground_track_error_km == pytest.approx(1196.621, abs=0.001)
    assert repeat.exact_repeat_altitude_km == pytest.approx(404.352, abs=0.001)


def test_entry_interface_estimate_reports_vacuum_speed_and_flight_path_angle() -> None:
    result = entry_interface_estimate(apogee_altitude_km=400.0, perigee_altitude_km=50.0, interface_altitude_km=120.0)

    assert result.semi_major_axis_km == pytest.approx(EARTH_RADIUS_KM + 225.0)
    assert result.eccentricity == pytest.approx(0.026503, abs=1.0e-6)
    assert result.speed_km_s == pytest.approx(7.89406, abs=1.0e-5)
    assert result.flight_path_angle_deg == pytest.approx(-1.215, abs=0.001)
    assert "not a heating" in result.note


def test_rocket_equation_delta_v_and_mass_ratio_are_inverses() -> None:
    delta_v = rocket_equation_delta_v(isp_s=300.0, mass_ratio=2.5)
    ratio = rocket_equation_mass_ratio(delta_v_m_s=delta_v.delta_v_m_s, isp_s=300.0)

    assert delta_v.delta_v_m_s == pytest.approx(2695.72, abs=0.01)
    assert ratio.mass_ratio == pytest.approx(2.5)
    assert ratio.propellant_fraction == pytest.approx(0.6)


def test_mission_recovery_from_retrograde_intrack_impulse_reports_recovery_cost() -> None:
    result = mission_recovery_from_intrack_impulse(
        reference_altitude_km=400.0,
        disturbance_delta_v_m_s=-5.0,
        spacecraft_mass_kg=100.0,
        isp_s=220.0,
        slot_tolerance_deg=1.0,
        max_phasing_orbits=100,
    )

    assert result.disturbance_apsis == "apogee"
    assert result.disturbed_apogee_altitude_km == pytest.approx(400.0)
    assert result.disturbed_perigee_altitude_km == pytest.approx(382.351, abs=0.001)
    assert result.disturbed_period_min < result.reference_period_min
    assert result.recovery_delta_v_m_s == pytest.approx(5.0)
    assert result.recovery_propellant_kg == pytest.approx(0.231485, abs=1.0e-6)
    assert result.total_event_delta_v_m_s == pytest.approx(10.0)
    assert result.slot_recovery_found is True
    assert result.slot_recovery_orbits == 1
    assert result.slot_recovery_time_hr == pytest.approx(1.53966, abs=1.0e-5)
    assert result.slot_recovery_phase_error_deg == pytest.approx(0.702799, abs=1.0e-6)


def test_mission_recovery_from_prograde_intrack_impulse_reports_higher_phasing_orbit() -> None:
    result = mission_recovery_from_intrack_impulse(
        reference_altitude_km=400.0,
        disturbance_delta_v_m_s=5.0,
        spacecraft_mass_kg=100.0,
        isp_s=220.0,
        slot_tolerance_deg=1.0,
        max_phasing_orbits=100,
    )

    assert result.disturbance_apsis == "perigee"
    assert result.disturbed_perigee_altitude_km == pytest.approx(400.0)
    assert result.disturbed_apogee_altitude_km == pytest.approx(417.707, abs=0.001)
    assert result.disturbed_period_min > result.reference_period_min
    assert result.recovery_delta_v_m_s == pytest.approx(5.0)
    assert result.recovery_propellant_kg == pytest.approx(0.231485, abs=1.0e-6)


def test_ballistic_coefficient_uses_mass_over_cd_area() -> None:
    result = ballistic_coefficient(mass_kg=100.0, drag_coefficient=2.2, drag_area_m2=4.0)

    assert result.ballistic_coefficient_kg_m2 == pytest.approx(100.0 / 8.8)


def test_atmospheric_density_from_altitude_uses_public_ussa1976_table() -> None:
    result = atmospheric_density_from_altitude(400.0)

    assert result.density_kg_m3 == pytest.approx(3.725e-12)
    assert "Meaningful drag" in result.warning


def test_atmospheric_density_above_table_range_warns_and_returns_zero_density() -> None:
    result = atmospheric_density_from_altitude(10000.0)

    assert result.density_kg_m3 == pytest.approx(0.0)
    assert "Outside built-in density-table range" in result.warning


def test_drag_force_acceleration_from_density_speed_uses_drag_equation() -> None:
    result = drag_force_acceleration_from_density_speed(
        density_kg_m3=1.0e-12,
        speed_m_s=7600.0,
        drag_coefficient=2.2,
        drag_area_m2=4.0,
        mass_kg=100.0,
    )

    expected_force = 0.5 * 1.0e-12 * 7600.0**2 * 2.2 * 4.0
    assert result.drag_force_n == pytest.approx(expected_force)
    assert result.drag_accel_m_s2 == pytest.approx(expected_force / 100.0)
    assert result.ballistic_coefficient_kg_m2 == pytest.approx(100.0 / 8.8)


def test_drag_force_acceleration_from_altitude_uses_circular_speed_and_density() -> None:
    result = drag_force_acceleration_from_altitude(
        altitude_km=400.0,
        mass_kg=100.0,
        drag_coefficient=2.2,
        drag_area_m2=4.0,
    )

    circular = circular_orbit_from_altitude(400.0)
    expected_force = 0.5 * 3.725e-12 * (circular.velocity_km_s * 1000.0) ** 2 * 2.2 * 4.0
    assert result.circular_speed_m_s == pytest.approx(circular.velocity_km_s * 1000.0)
    assert result.drag_force_n == pytest.approx(expected_force)
    assert result.drag_accel_m_s2 == pytest.approx(expected_force / 100.0)


def test_circular_orbit_drag_decay_rate_is_negative_and_scales_with_ballistic_coefficient() -> None:
    low_bc = circular_orbit_drag_decay_rate(altitude_km=400.0, ballistic_coefficient_kg_m2=25.0)
    high_bc = circular_orbit_drag_decay_rate(altitude_km=400.0, ballistic_coefficient_kg_m2=50.0)

    assert low_bc.decay_rate_m_day < 0.0
    assert high_bc.decay_rate_m_day == pytest.approx(0.5 * low_bc.decay_rate_m_day)
    assert low_bc.decay_rate_m_day == pytest.approx(-669.151, abs=0.001)


def test_drag_lifetime_range_orders_density_cases_and_scales_with_ballistic_coefficient() -> None:
    low_bc = drag_lifetime_range_estimate(
        initial_altitude_km=400.0,
        ballistic_coefficient_kg_m2=25.0,
        deorbit_altitude_km=120.0,
    )
    high_bc = drag_lifetime_range_estimate(
        initial_altitude_km=400.0,
        ballistic_coefficient_kg_m2=50.0,
        deorbit_altitude_km=120.0,
    )

    assert low_bc.high_drag_lifetime_days < low_bc.nominal_lifetime_days < low_bc.low_drag_lifetime_days
    assert low_bc.nominal_lifetime_days == pytest.approx(77.36, abs=0.01)
    assert high_bc.nominal_lifetime_days == pytest.approx(2.0 * low_bc.nominal_lifetime_days)
    assert low_bc.low_drag_density_scale == pytest.approx(0.3)
    assert low_bc.high_drag_density_scale == pytest.approx(3.0)


def test_classical_coe_to_rv_round_trips_through_robust_elements() -> None:
    state = classical_coe_to_rv(
        semi_major_axis_km=8000.0,
        eccentricity=0.1,
        inclination_deg=45.0,
        raan_deg=30.0,
        argp_deg=40.0,
        true_anomaly_deg=50.0,
    )
    elements = rv_to_robust_elements(*state.position_eci_km, *state.velocity_eci_km_s)

    assert elements.orbit_type == "elliptical"
    assert elements.semi_major_axis_km == pytest.approx(8000.0)
    assert elements.eccentricity == pytest.approx(0.1)
    assert elements.inclination_deg == pytest.approx(45.0)
    assert elements.raan_deg == pytest.approx(30.0)
    assert elements.argp_deg == pytest.approx(40.0)
    assert elements.true_anomaly_deg == pytest.approx(50.0)
    assert state.speed_km_s == pytest.approx(math.sqrt(EARTH_MU_KM3_S2 * ((2.0 / state.radius_km) - (1.0 / 8000.0))))


def test_rv_to_robust_elements_reports_circular_equatorial_alternate_angle() -> None:
    radius = EARTH_RADIUS_KM + 400.0
    speed = math.sqrt(EARTH_MU_KM3_S2 / radius)
    elements = rv_to_robust_elements(radius, 0.0, 0.0, 0.0, speed, 0.0)

    assert elements.orbit_type == "circular"
    assert elements.eccentricity == pytest.approx(0.0, abs=1.0e-12)
    assert elements.raan_deg is None
    assert elements.argp_deg is None
    assert elements.true_anomaly_deg is None
    assert elements.true_longitude_deg == pytest.approx(0.0)
    assert elements.argument_of_latitude_deg is None
    assert "Use true longitude" in " ".join(elements.notes)


def test_circular_inclined_elements_to_rv_reports_argument_of_latitude() -> None:
    state = circular_inclined_elements_to_rv(
        semi_major_axis_km=7000.0,
        inclination_deg=45.0,
        raan_deg=30.0,
        argument_of_latitude_deg=80.0,
    )
    elements = rv_to_robust_elements(*state.position_eci_km, *state.velocity_eci_km_s)

    assert elements.eccentricity == pytest.approx(0.0, abs=1.0e-12)
    assert elements.raan_deg == pytest.approx(30.0)
    assert elements.argp_deg is None
    assert elements.true_anomaly_deg is None
    assert elements.argument_of_latitude_deg == pytest.approx(80.0)


def test_equatorial_elliptical_elements_to_rv_reports_longitude_of_perigee() -> None:
    state = equatorial_elliptical_elements_to_rv(
        semi_major_axis_km=8000.0,
        eccentricity=0.2,
        longitude_of_perigee_deg=60.0,
        true_anomaly_deg=20.0,
    )
    elements = rv_to_robust_elements(*state.position_eci_km, *state.velocity_eci_km_s)

    assert elements.eccentricity == pytest.approx(0.2)
    assert elements.raan_deg is None
    assert elements.argp_deg is None
    assert elements.true_anomaly_deg == pytest.approx(20.0)
    assert elements.longitude_of_perigee_deg == pytest.approx(60.0)


def test_circular_equatorial_elements_to_rv_reports_true_longitude() -> None:
    state = circular_equatorial_elements_to_rv(semi_major_axis_km=7000.0, true_longitude_deg=123.0)
    elements = rv_to_robust_elements(*state.position_eci_km, *state.velocity_eci_km_s)

    assert elements.eccentricity == pytest.approx(0.0, abs=1.0e-12)
    assert elements.raan_deg is None
    assert elements.argp_deg is None
    assert elements.true_anomaly_deg is None
    assert elements.true_longitude_deg == pytest.approx(123.0)


def test_calculator_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="altitude_km"):
        circular_orbit_from_altitude(-1.0)
    with pytest.raises(ValueError, match="must differ"):
        hohmann_transfer_between_circular_orbits(400.0, 400.0)
    with pytest.raises(ValueError, match="must differ"):
        hohmann_rendezvous_phase_angle(400.0, 400.0)
    with pytest.raises(ValueError, match="no greater than 180"):
        plane_change_delta_v(speed_km_s=7.5, angle_deg=181.0)
    with pytest.raises(ValueError, match="apogee_altitude_km"):
        elements_from_apogee_perigee_altitudes(perigee_altitude_km=800.0, apogee_altitude_km=400.0)
    with pytest.raises(ValueError, match="phasing altitude"):
        phasing_drift_from_altitude_change(reference_altitude_km=5.0, phasing_altitude_change_km=-10.0)
    with pytest.raises(ValueError, match="mass_ratio"):
        rocket_equation_delta_v(isp_s=300.0, mass_ratio=0.9)
    with pytest.raises(ValueError, match="escaping trajectory"):
        mission_recovery_from_intrack_impulse(
            reference_altitude_km=400.0,
            disturbance_delta_v_m_s=11000.0,
            spacecraft_mass_kg=100.0,
            isp_s=220.0,
        )
    with pytest.raises(ValueError, match="drag_area_m2"):
        ballistic_coefficient(mass_kg=100.0, drag_coefficient=2.2, drag_area_m2=0.0)
    with pytest.raises(ValueError, match="ballistic_coefficient"):
        circular_orbit_drag_decay_rate(altitude_km=400.0, ballistic_coefficient_kg_m2=0.0)
    with pytest.raises(ValueError, match="1000 km limit"):
        circular_orbit_drag_decay_rate(altitude_km=1000.1, ballistic_coefficient_kg_m2=25.0)
    with pytest.raises(ValueError, match="1000 km limit"):
        drag_lifetime_range_estimate(
            initial_altitude_km=1000.1,
            ballistic_coefficient_kg_m2=25.0,
            deorbit_altitude_km=120.0,
        )
    with pytest.raises(ValueError, match="deorbit_altitude_km"):
        drag_lifetime_range_estimate(
            initial_altitude_km=120.0,
            ballistic_coefficient_kg_m2=25.0,
            deorbit_altitude_km=120.0,
        )
    with pytest.raises(ValueError, match="eccentricity"):
        classical_coe_to_rv(
            semi_major_axis_km=8000.0,
            eccentricity=1.0,
            inclination_deg=45.0,
            raan_deg=30.0,
            argp_deg=40.0,
            true_anomaly_deg=50.0,
        )
    with pytest.raises(ValueError, match="angular momentum"):
        rv_to_robust_elements(7000.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="inclination_deg"):
        j2_secular_rates_from_elements(semi_major_axis_km=7000.0, eccentricity=0.0, inclination_deg=181.0)
    with pytest.raises(ValueError, match="beta_angle_deg"):
        circular_eclipse_estimate(altitude_km=400.0, beta_angle_deg=91.0)
    with pytest.raises(ValueError, match="interface_altitude_km"):
        entry_interface_estimate(apogee_altitude_km=400.0, perigee_altitude_km=200.0, interface_altitude_km=120.0)


def test_interactive_menu_contains_requested_public_calculators() -> None:
    titles = {item.title for item in CALCULATORS}

    assert "Geosynchronous orbit altitude" in titles
    assert "Apogee/perigee from semi-major axis and eccentricity" in titles
    assert "Semi-major axis/eccentricity from apogee/perigee altitudes" in titles
    assert "Velocity at perigee and apogee" in titles
    assert "Inclination change cost from circular altitude" in titles
    assert "Hohmann rendezvous phase angle" in titles
    assert "Hohmann rendezvous wait time" in titles
    assert "Combined speed and plane change delta-v" in titles
    assert "Sun-synchronous inclination from altitude" in titles
    assert "J2 secular rates from altitude" in titles
    assert "J2 secular rates from semi-major axis" in titles
    assert "Phasing drift from altitude change" in titles
    assert "HCW natural motion from altitude" in titles
    assert "HCW in-track drift estimate" in titles
    assert "Circular-orbit eclipse estimate" in titles
    assert "Ground-track drift from altitude" in titles
    assert "Repeat ground-track approximation" in titles
    assert "Entry interface from apogee/perigee" in titles
    assert "Ballistic coefficient" in titles
    assert "Density estimate from altitude" in titles
    assert "Drag force and acceleration from altitude" in titles
    assert "Circular-orbit drag decay rate estimate" in titles
    assert "Deorbit lifetime range estimate" in titles
    assert "RV to robust element report" in titles
    assert "Classical COE to RV" in titles
    assert "Circular inclined elements to RV" in titles
    assert "Equatorial elliptical elements to RV" in titles
    assert "Circular equatorial elements to RV" in titles
    assert "Rocket equation delta-v from mass ratio" in titles
    assert "Rocket equation mass ratio from delta-v" in titles
    assert "Mission recovery from in-track impulse" in titles


def test_interactive_calculators_are_grouped_by_category() -> None:
    grouped_titles = {
        category: {item.title for item in calculators_for_category(category)}
        for category in CATEGORIES
    }

    assert "Geosynchronous orbit altitude" in grouped_titles["Circular Orbits"]
    assert "Velocity at perigee and apogee" in grouped_titles["Elliptical Orbits"]
    assert "RV to robust element report" in grouped_titles["State / Elements Conversion"]
    assert "Circular equatorial elements to RV" in grouped_titles["State / Elements Conversion"]
    assert "Combined speed and plane change delta-v" in grouped_titles["Transfers And Delta-V"]
    assert "Mission recovery from in-track impulse" in grouped_titles["Transfers And Delta-V"]
    assert "Sun-synchronous inclination from altitude" in grouped_titles["Sun-Synchronous"]
    assert "J2 secular rates from altitude" in grouped_titles["Sun-Synchronous"]
    assert "Phasing drift from altitude change" in grouped_titles["Phasing"]
    assert "HCW in-track drift estimate" in grouped_titles["Relative Motion / HCW"]
    assert "Circular-orbit eclipse estimate" in grouped_titles["Eclipse"]
    assert "Ground-track drift from altitude" in grouped_titles["Ground Track"]
    assert "Entry interface from apogee/perigee" in grouped_titles["Entry / Reentry"]
    assert "Drag force and acceleration from altitude" in grouped_titles["Atmospheric Drag"]
    assert "Deorbit lifetime range estimate" in grouped_titles["Atmospheric Drag"]
    assert "Rocket equation mass ratio from delta-v" in grouped_titles["Rocket Equation"]
    assert sum(len(calculators_for_category(category)) for category in CATEGORIES) == len(CALCULATORS)


def test_prompt_yes_no_accepts_defaults_and_explicit_answers() -> None:
    assert prompt_yes_no("Again?", input_func=lambda _prompt: "") is True
    assert prompt_yes_no("Again?", input_func=lambda _prompt: "n") is False
    assert prompt_yes_no("Again?", default=False, input_func=lambda _prompt: "") is False
    assert prompt_yes_no("Again?", default=False, input_func=lambda _prompt: "yes") is True


def test_interactive_period_prompt_converts_minutes_to_seconds() -> None:
    spec = next(item for item in CALCULATORS if item.title == "Circular altitude from period")
    result = run_calculation(spec, {"period_s": 90.0})

    assert result.period_s == pytest.approx(5400.0)
    assert result.period_min == pytest.approx(90.0)


def test_interactive_formatter_includes_public_assumptions() -> None:
    spec = next(item for item in CALCULATORS if item.title == "Circular orbit from altitude")
    result = run_calculation(spec, {"altitude_km": 400.0})
    text = format_result(spec, result)

    assert "Circular velocity" in text
    assert "Assumptions: Earth two-body gravity" in text


def test_interactive_formatter_uses_sun_sync_assumptions_for_j2_estimate() -> None:
    spec = next(item for item in CALCULATORS if item.title == "Sun-synchronous inclination from altitude")
    result = run_calculation(spec, {"altitude_km": 700.0})
    text = format_result(spec, result)

    assert "Required inclination" in text
    assert "first-order J2 nodal precession" in text
