from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable, Protocol

from sim.dynamics.orbit.environment import EARTH_RADIUS_KM
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
    circular_orbit_from_radius,
    classical_coe_to_rv,
    combined_speed_plane_change_delta_v,
    drag_force_acceleration_from_altitude,
    drag_force_acceleration_from_density_speed,
    drag_lifetime_range_estimate,
    elements_from_apogee_perigee_altitudes,
    entry_interface_estimate,
    equatorial_elliptical_elements_to_rv,
    escape_velocity_at_radius,
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
    orbital_period_from_altitude,
    orbital_period_from_semimajor_axis,
    phasing_drift_from_altitude_change,
    plane_change_delta_v,
    repeat_ground_track_approximation,
    rocket_equation_delta_v,
    rocket_equation_mass_ratio,
    rv_to_robust_elements,
    sun_synchronous_inclination_from_altitude,
    vis_viva_velocity,
)


class ResultFormatter(Protocol):
    def __call__(self, result: object) -> list[tuple[str, str]]:
        ...


CATEGORY_CIRCULAR = "Circular Orbits"
CATEGORY_ELLIPTICAL = "Elliptical Orbits"
CATEGORY_STATE_ELEMENTS = "State / Elements Conversion"
CATEGORY_TRANSFERS = "Transfers And Delta-V"
CATEGORY_SUN_SYNC = "Sun-Synchronous"
CATEGORY_PHASING = "Phasing"
CATEGORY_RELATIVE = "Relative Motion / HCW"
CATEGORY_ECLIPSE = "Eclipse"
CATEGORY_GROUND_TRACK = "Ground Track"
CATEGORY_ENTRY = "Entry / Reentry"
CATEGORY_DRAG = "Atmospheric Drag"
CATEGORY_ROCKET = "Rocket Equation"

CATEGORIES: tuple[str, ...] = (
    CATEGORY_CIRCULAR,
    CATEGORY_ELLIPTICAL,
    CATEGORY_STATE_ELEMENTS,
    CATEGORY_TRANSFERS,
    CATEGORY_SUN_SYNC,
    CATEGORY_PHASING,
    CATEGORY_RELATIVE,
    CATEGORY_ECLIPSE,
    CATEGORY_GROUND_TRACK,
    CATEGORY_ENTRY,
    CATEGORY_DRAG,
    CATEGORY_ROCKET,
)


@dataclass(frozen=True)
class FloatPrompt:
    key: str
    label: str
    unit: str
    min_value: float | None = None
    max_value: float | None = None
    multiplier: float = 1.0

    @property
    def display_label(self) -> str:
        return f"{self.label} [{self.unit}]"


@dataclass(frozen=True)
class CalculatorSpec:
    category: str
    title: str
    prompts: tuple[FloatPrompt, ...]
    compute: Callable[..., object]
    formatter: ResultFormatter


def _fmt(value: float, unit: str, precision: int = 3) -> str:
    return f"{value:,.{precision}f} {unit}"


def _fmt_optional(value: float | None, unit: str, precision: int = 3) -> str:
    return "Undefined" if value is None else _fmt(value, unit, precision)


def _fmt_vector(values: tuple[float, float, float], unit: str, precision: int = 6) -> str:
    body = ", ".join(f"{value:,.{precision}f}" for value in values)
    return f"[{body}] {unit}"


def _assumptions(spec: CalculatorSpec) -> str:
    if spec.title == "Sun-synchronous inclination from altitude":
        return "Assumptions: first-order J2 nodal precession, circular orbit, spherical Earth, no drag/SRP."
    if spec.title.startswith("J2 secular rates"):
        return "Assumptions: first-order J2 secular rates around Earth; no drag, SRP, third-body gravity, or resonance effects."
    if spec.title.startswith("Hohmann rendezvous"):
        return "Assumptions: coplanar circular two-body orbits and an impulsive Hohmann transfer; positive phase means target ahead of chaser."
    if spec.title == "Phasing drift from altitude change":
        return "Assumptions: two circular two-body orbits; drift from mean-motion difference; no maneuvers/J2/drag."
    if spec.category == CATEGORY_RELATIVE:
        return "Assumptions: linear HCW/Clohessy-Wiltshire motion about a circular chief orbit; valid only for small relative states."
    if spec.category == CATEGORY_ECLIPSE:
        return "Assumptions: circular orbit, cylindrical Earth shadow, fixed beta angle, spherical Earth; no penumbra or seasonal geometry."
    if spec.category == CATEGORY_GROUND_TRACK:
        return "Assumptions: circular inertial orbit over a spherical rotating Earth; no J2, inclination effects, drag, or nodal regression."
    if spec.category == CATEGORY_ENTRY:
        return "Assumptions: vacuum two-body conic at the interface altitude; not a heating, deceleration, skip, or survivability estimate."
    if spec.title == "Deorbit lifetime range estimate":
        return (
            "Assumptions: circular local-density decay integrated over altitude with 0.3x/1x/3x USSA-1976 "
            "density scaling; not a lifetime prediction."
        )
    if spec.category == CATEGORY_STATE_ELEMENTS:
        return (
            "Assumptions: Earth-centered inertial vectors and simulator Earth mu; singular classical angles are "
            "reported as undefined with alternate angles when available."
        )
    if spec.category == CATEGORY_DRAG:
        return "Assumptions: USSA-1976 density estimate, fixed Cd/area, circular speed when altitude is used; not a lifetime predictor."
    if spec.title.startswith("Rocket equation"):
        return "Assumptions: ideal Tsiolkovsky rocket equation with standard gravity 9.80665 m/s^2."
    return "Assumptions: Earth two-body gravity, spherical reference radius, no drag/J2/SRP."


def _format_circular(result: object) -> list[tuple[str, str]]:
    return [
        ("Altitude", _fmt(result.altitude_km, "km")),
        ("Orbit radius", _fmt(result.radius_km, "km")),
        ("Circular velocity", _fmt(result.velocity_km_s, "km/s")),
        ("Orbital period", _fmt(result.period_min, "min", 2)),
        ("Mean motion", _fmt(result.mean_motion_rad_s, "rad/s", 6)),
        ("Escape velocity at radius", _fmt(result.escape_velocity_km_s, "km/s")),
    ]


def _format_period(result: object) -> list[tuple[str, str]]:
    rows = [
        ("Semi-major axis", _fmt(result.semi_major_axis_km, "km")),
        ("Orbital period", _fmt(result.period_min, "min", 2)),
        ("Mean motion", _fmt(result.mean_motion_rad_s, "rad/s", 6)),
    ]
    if result.altitude_km is not None:
        rows.insert(1, ("Altitude", _fmt(result.altitude_km, "km")))
    return rows


def _format_altitude_from_period(result: object) -> list[tuple[str, str]]:
    return [
        ("Orbital period", _fmt(result.period_min, "min", 2)),
        ("Semi-major axis", _fmt(result.semi_major_axis_km, "km")),
        ("Circular altitude", _fmt(result.altitude_km, "km")),
        ("Mean motion", _fmt(result.mean_motion_rad_s, "rad/s", 6)),
    ]


def _format_vis_viva(result: object) -> list[tuple[str, str]]:
    return [
        ("Current radius", _fmt(result.radius_km, "km")),
        ("Semi-major axis", _fmt(result.semi_major_axis_km, "km")),
        ("Orbital speed", _fmt(result.velocity_km_s, "km/s")),
    ]


def _format_hohmann(result: object) -> list[tuple[str, str]]:
    return [
        ("Initial altitude", _fmt(result.initial_altitude_km, "km")),
        ("Final altitude", _fmt(result.final_altitude_km, "km")),
        ("Transfer semi-major axis", _fmt(result.transfer_semimajor_axis_km, "km")),
        ("First burn", _fmt(result.first_burn_delta_v_m_s, "m/s", 2)),
        ("Second burn", _fmt(result.second_burn_delta_v_m_s, "m/s", 2)),
        ("Total delta-v", _fmt(result.total_delta_v_m_s, "m/s", 2)),
        ("Transfer time", _fmt(result.transfer_time_min, "min", 2)),
    ]


def _format_hohmann_phase(result: object) -> list[tuple[str, str]]:
    synodic = "N/A" if result.synodic_period_min is None else _fmt(result.synodic_period_min, "min", 2)
    return [
        ("Initial altitude", _fmt(result.initial_altitude_km, "km")),
        ("Target altitude", _fmt(result.target_altitude_km, "km")),
        ("Transfer semi-major axis", _fmt(result.transfer_semimajor_axis_km, "km")),
        ("Transfer time", _fmt(result.transfer_time_min, "min", 2)),
        ("Target travel during transfer", _fmt(result.target_travel_during_transfer_deg, "deg", 3)),
        ("Required phase angle", _fmt(result.required_phase_angle_deg, "deg", 3)),
        ("Relative phase rate", _fmt(result.relative_phase_rate_deg_s, "deg/s", 8)),
        ("Synodic period", synodic),
    ]


def _format_hohmann_wait(result: object) -> list[tuple[str, str]]:
    wait = "N/A" if result.wait_time_hr is None else f"{result.wait_time_min:,.2f} min ({result.wait_time_hr:,.2f} hr)"
    synodic = "N/A" if result.synodic_period_min is None else _fmt(result.synodic_period_min, "min", 2)
    return [
        ("Initial altitude", _fmt(result.initial_altitude_km, "km")),
        ("Target altitude", _fmt(result.target_altitude_km, "km")),
        ("Current phase angle", _fmt(result.current_phase_angle_deg, "deg", 3)),
        ("Required phase angle", _fmt(result.required_phase_angle_deg, "deg", 3)),
        ("Relative phase rate", _fmt(result.relative_phase_rate_deg_s, "deg/s", 8)),
        ("Wait time", wait),
        ("Synodic period", synodic),
    ]


def _format_plane_change(result: object) -> list[tuple[str, str]]:
    return [
        ("Speed", _fmt(result.speed_km_s, "km/s")),
        ("Plane-change angle", _fmt(result.angle_deg, "deg", 2)),
        ("Delta-v", _fmt(result.delta_v_m_s, "m/s", 2)),
    ]


def _format_escape(result: object) -> list[tuple[str, str]]:
    rows = [("Radius", _fmt(result.radius_km, "km"))]
    if result.altitude_km is not None:
        rows.append(("Altitude", _fmt(result.altitude_km, "km")))
    rows.append(("Escape velocity", _fmt(result.escape_velocity_km_s, "km/s")))
    return rows


def _format_geosynchronous(result: object) -> list[tuple[str, str]]:
    return [
        ("Period", _fmt(result.period_hr, "hr", 3)),
        ("Semi-major axis", _fmt(result.semi_major_axis_km, "km")),
        ("Altitude", _fmt(result.altitude_km, "km")),
        ("Circular velocity", _fmt(result.circular_velocity_km_s, "km/s")),
        ("Mean motion", _fmt(result.mean_motion_rad_s, "rad/s", 8)),
    ]


def _format_apogee_perigee_from_elements(result: object) -> list[tuple[str, str]]:
    return [
        ("Semi-major axis", _fmt(result.semi_major_axis_km, "km")),
        ("Eccentricity", f"{result.eccentricity:.6f}"),
        ("Perigee radius", _fmt(result.perigee_radius_km, "km")),
        ("Apogee radius", _fmt(result.apogee_radius_km, "km")),
        ("Perigee altitude", _fmt(result.perigee_altitude_km, "km")),
        ("Apogee altitude", _fmt(result.apogee_altitude_km, "km")),
    ]


def _format_elements_from_apogee_perigee(result: object) -> list[tuple[str, str]]:
    return [
        ("Perigee altitude", _fmt(result.perigee_altitude_km, "km")),
        ("Apogee altitude", _fmt(result.apogee_altitude_km, "km")),
        ("Perigee radius", _fmt(result.perigee_radius_km, "km")),
        ("Apogee radius", _fmt(result.apogee_radius_km, "km")),
        ("Semi-major axis", _fmt(result.semi_major_axis_km, "km")),
        ("Eccentricity", f"{result.eccentricity:.6f}"),
    ]


def _format_apogee_perigee_velocities(result: object) -> list[tuple[str, str]]:
    return [
        ("Perigee altitude", _fmt(result.perigee_altitude_km, "km")),
        ("Apogee altitude", _fmt(result.apogee_altitude_km, "km")),
        ("Semi-major axis", _fmt(result.semi_major_axis_km, "km")),
        ("Eccentricity", f"{result.eccentricity:.6f}"),
        ("Perigee velocity", _fmt(result.perigee_velocity_km_s, "km/s")),
        ("Apogee velocity", _fmt(result.apogee_velocity_km_s, "km/s")),
    ]


def _format_combined_plane_change(result: object) -> list[tuple[str, str]]:
    return [
        ("Initial speed", _fmt(result.initial_speed_km_s, "km/s")),
        ("Final speed", _fmt(result.final_speed_km_s, "km/s")),
        ("Turn angle", _fmt(result.angle_deg, "deg", 2)),
        ("Delta-v", _fmt(result.delta_v_m_s, "m/s", 2)),
    ]


def _format_sun_sync(result: object) -> list[tuple[str, str]]:
    return [
        ("Altitude", _fmt(result.altitude_km, "km")),
        ("Semi-major axis", _fmt(result.semi_major_axis_km, "km")),
        ("Required inclination", _fmt(result.inclination_deg, "deg", 3)),
        ("Target nodal precession", _fmt(result.target_precession_deg_day, "deg/day", 6)),
        ("Estimated nodal precession", _fmt(result.nodal_precession_deg_day, "deg/day", 6)),
    ]


def _format_j2_rates(result: object) -> list[tuple[str, str]]:
    return [
        ("Semi-major axis", _fmt(result.semi_major_axis_km, "km")),
        ("Eccentricity", f"{result.eccentricity:.6f}"),
        ("Inclination", _fmt(result.inclination_deg, "deg", 3)),
        ("Semi-latus rectum", _fmt(result.semi_latus_rectum_km, "km")),
        ("Orbital period", _fmt(result.orbital_period_min, "min", 2)),
        ("Mean motion", _fmt(result.mean_motion_rad_s, "rad/s", 8)),
        ("RAAN rate", _fmt(result.raan_rate_deg_day, "deg/day", 6)),
        ("Argument of perigee rate", _fmt(result.argument_of_perigee_rate_deg_day, "deg/day", 6)),
        ("Mean anomaly J2 rate", _fmt(result.mean_anomaly_j2_rate_deg_day, "deg/day", 6)),
    ]


def _format_phasing(result: object) -> list[tuple[str, str]]:
    lap = "N/A" if result.lap_time_hr is None else _fmt(result.lap_time_hr, "hr", 2)
    return [
        ("Reference altitude", _fmt(result.reference_altitude_km, "km")),
        ("Phasing altitude", _fmt(result.phasing_altitude_km, "km")),
        ("Altitude change", _fmt(result.altitude_change_km, "km")),
        ("Drift rate", _fmt(result.drift_rate_m_s, "m/s", 3)),
        ("Drift per reference orbit", _fmt(result.drift_per_reference_orbit_km, "km", 3)),
        ("Time to lap by 360 deg", lap),
    ]


def _format_hcw_natural(result: object) -> list[tuple[str, str]]:
    return [
        ("Altitude", _fmt(result.altitude_km, "km")),
        ("Orbit radius", _fmt(result.orbit_radius_km, "km")),
        ("Mean motion", _fmt(result.mean_motion_rad_s, "rad/s", 8)),
        ("Natural motion period", _fmt(result.natural_motion_period_min, "min", 2)),
        ("Circular velocity", _fmt(result.circular_velocity_km_s, "km/s", 6)),
    ]


def _format_hcw_drift(result: object) -> list[tuple[str, str]]:
    return [
        ("Altitude", _fmt(result.altitude_km, "km")),
        ("Duration", _fmt(result.duration_orbits, "orbits", 3)),
        ("Duration", _fmt(result.duration_min, "min", 2)),
        ("Radial offset", _fmt(result.radial_offset_m, "m", 3)),
        ("In-track velocity bias", _fmt(result.intrack_velocity_bias_m_s, "m/s", 6)),
        ("Drift from radial offset", _fmt(result.radial_offset_drift_m, "m", 3)),
        ("Drift from velocity bias", _fmt(result.intrack_velocity_drift_m, "m", 3)),
        ("Total in-track drift", _fmt(result.total_intrack_drift_m, "m", 3)),
    ]


def _format_eclipse(result: object) -> list[tuple[str, str]]:
    half_angle = "No eclipse" if result.eclipse_half_angle_deg is None else _fmt(result.eclipse_half_angle_deg, "deg", 3)
    return [
        ("Altitude", _fmt(result.altitude_km, "km")),
        ("Beta angle", _fmt(result.beta_angle_deg, "deg", 3)),
        ("Critical beta angle", _fmt(result.beta_critical_deg, "deg", 3)),
        ("Orbital period", _fmt(result.orbital_period_min, "min", 2)),
        ("Eclipse half-angle", half_angle),
        ("Eclipse duration", _fmt(result.eclipse_duration_min, "min", 2)),
        ("Eclipse fraction", f"{100.0 * result.eclipse_fraction:.2f}%"),
        ("Sunlight duration", _fmt(result.sunlight_duration_min, "min", 2)),
    ]


def _format_ground_track_drift(result: object) -> list[tuple[str, str]]:
    return [
        ("Altitude", _fmt(result.altitude_km, "km")),
        ("Orbital period", _fmt(result.orbital_period_min, "min", 2)),
        ("Earth rotation per orbit", _fmt(result.earth_rotation_deg_per_orbit, "deg", 3)),
        ("Ground-track drift", _fmt(result.westward_drift_deg_per_orbit, "deg/orbit", 3)),
        ("Equator drift distance", _fmt(result.equator_drift_km_per_orbit, "km/orbit", 2)),
        ("Orbits per sidereal day", f"{result.orbits_per_sidereal_day:.5f}"),
    ]


def _format_repeat_ground_track(result: object) -> list[tuple[str, str]]:
    return [
        ("Input altitude", _fmt(result.altitude_km, "km")),
        ("Search window", f"{result.max_days:d} days"),
        ("Repeat cycle", f"{result.repeat_orbits:d} orbits / {result.repeat_days:d} days"),
        ("Ground-track error", _fmt(result.ground_track_error_deg, "deg/cycle", 6)),
        ("Ground-track error", _fmt(result.ground_track_error_km, "km/cycle", 3)),
        ("Exact repeat altitude", _fmt(result.exact_repeat_altitude_km, "km")),
        ("Exact repeat period", _fmt(result.exact_repeat_period_min, "min", 3)),
    ]


def _format_entry_interface(result: object) -> list[tuple[str, str]]:
    return [
        ("Apogee altitude", _fmt(result.apogee_altitude_km, "km")),
        ("Perigee altitude", _fmt(result.perigee_altitude_km, "km")),
        ("Interface altitude", _fmt(result.interface_altitude_km, "km")),
        ("Semi-major axis", _fmt(result.semi_major_axis_km, "km")),
        ("Eccentricity", f"{result.eccentricity:.6f}"),
        ("Interface speed", _fmt(result.speed_km_s, "km/s", 6)),
        ("Flight-path angle", _fmt(result.flight_path_angle_deg, "deg", 3)),
        ("True anomaly", _fmt(result.true_anomaly_deg, "deg", 3)),
        ("Note", result.note),
    ]


def _format_rocket_delta_v(result: object) -> list[tuple[str, str]]:
    return [
        ("Specific impulse", _fmt(result.isp_s, "s", 2)),
        ("Mass ratio", f"{result.mass_ratio:.4f}"),
        ("Delta-v", _fmt(result.delta_v_m_s, "m/s", 2)),
        ("Delta-v", _fmt(result.delta_v_km_s, "km/s")),
    ]


def _format_rocket_mass_ratio(result: object) -> list[tuple[str, str]]:
    return [
        ("Specific impulse", _fmt(result.isp_s, "s", 2)),
        ("Delta-v", _fmt(result.delta_v_m_s, "m/s", 2)),
        ("Mass ratio", f"{result.mass_ratio:.4f}"),
        ("Propellant fraction", f"{100.0 * result.propellant_fraction:.2f}%"),
    ]


def _format_ballistic_coefficient(result: object) -> list[tuple[str, str]]:
    return [
        ("Mass", _fmt(result.mass_kg, "kg", 3)),
        ("Drag coefficient", f"{result.drag_coefficient:.3f}"),
        ("Drag area", _fmt(result.drag_area_m2, "m^2", 4)),
        ("Ballistic coefficient", _fmt(result.ballistic_coefficient_kg_m2, "kg/m^2", 3)),
    ]


def _format_density(result: object) -> list[tuple[str, str]]:
    return [
        ("Altitude", _fmt(result.altitude_km, "km")),
        ("Density", f"{result.density_kg_m3:.6e} kg/m^3"),
        ("Warning", result.warning),
    ]


def _format_drag_force(result: object) -> list[tuple[str, str]]:
    return [
        ("Density", f"{result.density_kg_m3:.6e} kg/m^3"),
        ("Speed", _fmt(result.speed_m_s, "m/s", 2)),
        ("Mass", _fmt(result.mass_kg, "kg", 3)),
        ("Cd", f"{result.drag_coefficient:.3f}"),
        ("Drag area", _fmt(result.drag_area_m2, "m^2", 4)),
        ("Ballistic coefficient", _fmt(result.ballistic_coefficient_kg_m2, "kg/m^2", 3)),
        ("Drag force", f"{result.drag_force_n:.6e} N"),
        ("Drag acceleration", f"{result.drag_accel_m_s2:.6e} m/s^2"),
    ]


def _format_drag_at_altitude(result: object) -> list[tuple[str, str]]:
    rows = [
        ("Altitude", _fmt(result.altitude_km, "km")),
        ("Density", f"{result.density_kg_m3:.6e} kg/m^3"),
        ("Circular speed", _fmt(result.circular_speed_m_s, "m/s", 2)),
        ("Mass", _fmt(result.mass_kg, "kg", 3)),
        ("Cd", f"{result.drag_coefficient:.3f}"),
        ("Drag area", _fmt(result.drag_area_m2, "m^2", 4)),
        ("Ballistic coefficient", _fmt(result.ballistic_coefficient_kg_m2, "kg/m^2", 3)),
        ("Drag force", f"{result.drag_force_n:.6e} N"),
        ("Drag acceleration", f"{result.drag_accel_m_s2:.6e} m/s^2"),
        ("Warning", result.warning),
    ]
    return rows


def _format_decay_rate(result: object) -> list[tuple[str, str]]:
    return [
        ("Altitude", _fmt(result.altitude_km, "km")),
        ("Density", f"{result.density_kg_m3:.6e} kg/m^3"),
        ("Ballistic coefficient", _fmt(result.ballistic_coefficient_kg_m2, "kg/m^2", 3)),
        ("da/dt", f"{result.decay_rate_m_s:.6e} m/s"),
        ("da/dt", _fmt(result.decay_rate_m_day, "m/day", 3)),
        ("da/dt", _fmt(result.decay_rate_km_day, "km/day", 6)),
        ("Warning", result.warning),
    ]


def _format_lifetime_range(result: object) -> list[tuple[str, str]]:
    return [
        ("Initial altitude", _fmt(result.initial_altitude_km, "km")),
        ("Deorbit altitude", _fmt(result.deorbit_altitude_km, "km")),
        ("Ballistic coefficient", _fmt(result.ballistic_coefficient_kg_m2, "kg/m^2", 3)),
        ("Low drag density scale", f"{result.low_drag_density_scale:.2f}x"),
        ("Nominal density scale", f"{result.nominal_density_scale:.2f}x"),
        ("High drag density scale", f"{result.high_drag_density_scale:.2f}x"),
        ("Low drag / long", f"{result.low_drag_lifetime_days:,.2f} days ({result.low_drag_lifetime_years:,.2f} yr)"),
        ("Nominal", f"{result.nominal_lifetime_days:,.2f} days ({result.nominal_lifetime_years:,.2f} yr)"),
        ("High drag / short", f"{result.high_drag_lifetime_days:,.2f} days ({result.high_drag_lifetime_years:,.2f} yr)"),
        ("Integration step", _fmt(result.integration_step_km, "km", 3)),
        ("Warning", result.warning),
    ]


def _format_rv_to_elements(result: object) -> list[tuple[str, str]]:
    rows = [
        ("Position ECI", _fmt_vector(result.position_eci_km, "km")),
        ("Velocity ECI", _fmt_vector(result.velocity_eci_km_s, "km/s")),
        ("Orbit type", result.orbit_type),
        ("Radius", _fmt(result.radius_km, "km")),
        ("Speed", _fmt(result.speed_km_s, "km/s", 6)),
        ("Specific energy", _fmt(result.specific_energy_km2_s2, "km^2/s^2", 6)),
        ("Angular momentum", _fmt(result.specific_angular_momentum_km2_s, "km^2/s", 6)),
        ("Semi-major axis", _fmt_optional(result.semi_major_axis_km, "km")),
        ("Semi-latus rectum", _fmt(result.semi_latus_rectum_km, "km")),
        ("Eccentricity", f"{result.eccentricity:.9f}"),
        ("Inclination", _fmt(result.inclination_deg, "deg", 6)),
        ("RAAN", _fmt_optional(result.raan_deg, "deg", 6)),
        ("Argument of perigee", _fmt_optional(result.argp_deg, "deg", 6)),
        ("True anomaly", _fmt_optional(result.true_anomaly_deg, "deg", 6)),
        ("Argument of latitude", _fmt_optional(result.argument_of_latitude_deg, "deg", 6)),
        ("Longitude of perigee", _fmt_optional(result.longitude_of_perigee_deg, "deg", 6)),
        ("True longitude", _fmt_optional(result.true_longitude_deg, "deg", 6)),
        ("Perigee altitude", _fmt(result.perigee_altitude_km, "km")),
        ("Apogee altitude", _fmt_optional(result.apogee_altitude_km, "km")),
        ("Period", "N/A" if result.period_min is None else _fmt(result.period_min, "min", 3)),
        ("V-infinity", _fmt_optional(result.v_infinity_km_s, "km/s", 6)),
        ("Turning angle", _fmt_optional(result.turning_angle_deg, "deg", 6)),
    ]
    rows.extend((f"Note {idx}", note) for idx, note in enumerate(result.notes, start=1))
    return rows


def _format_coe_to_rv(result: object) -> list[tuple[str, str]]:
    rows = [
        ("Input mode", result.input_mode),
        ("Position ECI", _fmt_vector(result.position_eci_km, "km")),
        ("Velocity ECI", _fmt_vector(result.velocity_eci_km_s, "km/s")),
        ("Radius", _fmt(result.radius_km, "km")),
        ("Speed", _fmt(result.speed_km_s, "km/s", 6)),
        ("Semi-major axis", _fmt_optional(result.semi_major_axis_km, "km")),
        ("Eccentricity", f"{result.eccentricity:.9f}"),
        ("Inclination", _fmt(result.inclination_deg, "deg", 6)),
        ("RAAN", _fmt_optional(result.raan_deg, "deg", 6)),
        ("Argument of perigee", _fmt_optional(result.argp_deg, "deg", 6)),
        ("True anomaly", _fmt_optional(result.true_anomaly_deg, "deg", 6)),
        ("Argument of latitude", _fmt_optional(result.argument_of_latitude_deg, "deg", 6)),
        ("Longitude of perigee", _fmt_optional(result.longitude_of_perigee_deg, "deg", 6)),
        ("True longitude", _fmt_optional(result.true_longitude_deg, "deg", 6)),
        ("Perigee altitude", _fmt(result.perigee_altitude_km, "km")),
        ("Apogee altitude", _fmt_optional(result.apogee_altitude_km, "km")),
        ("Period", "N/A" if result.period_min is None else _fmt(result.period_min, "min", 3)),
    ]
    rows.extend((f"Note {idx}", note) for idx, note in enumerate(result.notes, start=1))
    return rows


def _escape_from_altitude(altitude_km: float) -> object:
    altitude = float(altitude_km)
    return escape_velocity_at_radius(EARTH_RADIUS_KM + altitude, altitude_km=altitude)


CALCULATORS: tuple[CalculatorSpec, ...] = (
    CalculatorSpec(
        category=CATEGORY_CIRCULAR,
        title="Circular orbit from altitude",
        prompts=(FloatPrompt("altitude_km", "Altitude above Earth", "km", min_value=0.0),),
        compute=circular_orbit_from_altitude,
        formatter=_format_circular,
    ),
    CalculatorSpec(
        category=CATEGORY_CIRCULAR,
        title="Circular orbit from radius",
        prompts=(FloatPrompt("radius_km", "Distance from Earth's center", "km", min_value=1.0),),
        compute=circular_orbit_from_radius,
        formatter=_format_circular,
    ),
    CalculatorSpec(
        category=CATEGORY_CIRCULAR,
        title="Orbit period from altitude",
        prompts=(FloatPrompt("altitude_km", "Altitude above Earth", "km", min_value=0.0),),
        compute=orbital_period_from_altitude,
        formatter=_format_period,
    ),
    CalculatorSpec(
        category=CATEGORY_CIRCULAR,
        title="Orbit period from semi-major axis",
        prompts=(FloatPrompt("semi_major_axis_km", "Semi-major axis", "km", min_value=1.0),),
        compute=orbital_period_from_semimajor_axis,
        formatter=_format_period,
    ),
    CalculatorSpec(
        category=CATEGORY_CIRCULAR,
        title="Circular altitude from period",
        prompts=(FloatPrompt("period_s", "Orbital period", "min", min_value=1.0e-9, multiplier=60.0),),
        compute=altitude_from_period,
        formatter=_format_altitude_from_period,
    ),
    CalculatorSpec(
        category=CATEGORY_CIRCULAR,
        title="Geosynchronous orbit altitude",
        prompts=(),
        compute=geosynchronous_orbit,
        formatter=_format_geosynchronous,
    ),
    CalculatorSpec(
        category=CATEGORY_ELLIPTICAL,
        title="Vis-viva velocity",
        prompts=(
            FloatPrompt("radius_km", "Current distance from Earth's center", "km", min_value=1.0),
            FloatPrompt("semi_major_axis_km", "Orbit semi-major axis", "km", min_value=1.0),
        ),
        compute=vis_viva_velocity,
        formatter=_format_vis_viva,
    ),
    CalculatorSpec(
        category=CATEGORY_ELLIPTICAL,
        title="Apogee/perigee from semi-major axis and eccentricity",
        prompts=(
            FloatPrompt("semi_major_axis_km", "Semi-major axis", "km", min_value=1.0),
            FloatPrompt("eccentricity", "Eccentricity", "unitless", min_value=0.0, max_value=0.999999),
        ),
        compute=apogee_perigee_from_elements,
        formatter=_format_apogee_perigee_from_elements,
    ),
    CalculatorSpec(
        category=CATEGORY_ELLIPTICAL,
        title="Semi-major axis/eccentricity from apogee/perigee altitudes",
        prompts=(
            FloatPrompt("perigee_altitude_km", "Perigee altitude", "km", min_value=0.0),
            FloatPrompt("apogee_altitude_km", "Apogee altitude", "km", min_value=0.0),
        ),
        compute=elements_from_apogee_perigee_altitudes,
        formatter=_format_elements_from_apogee_perigee,
    ),
    CalculatorSpec(
        category=CATEGORY_ELLIPTICAL,
        title="Velocity at perigee and apogee",
        prompts=(
            FloatPrompt("perigee_altitude_km", "Perigee altitude", "km", min_value=0.0),
            FloatPrompt("apogee_altitude_km", "Apogee altitude", "km", min_value=0.0),
        ),
        compute=apogee_perigee_velocities_from_altitudes,
        formatter=_format_apogee_perigee_velocities,
    ),
    CalculatorSpec(
        category=CATEGORY_STATE_ELEMENTS,
        title="RV to robust element report",
        prompts=(
            FloatPrompt("rx_km", "ECI position x", "km"),
            FloatPrompt("ry_km", "ECI position y", "km"),
            FloatPrompt("rz_km", "ECI position z", "km"),
            FloatPrompt("vx_km_s", "ECI velocity x", "km/s"),
            FloatPrompt("vy_km_s", "ECI velocity y", "km/s"),
            FloatPrompt("vz_km_s", "ECI velocity z", "km/s"),
        ),
        compute=rv_to_robust_elements,
        formatter=_format_rv_to_elements,
    ),
    CalculatorSpec(
        category=CATEGORY_STATE_ELEMENTS,
        title="Classical COE to RV",
        prompts=(
            FloatPrompt("semi_major_axis_km", "Semi-major axis", "km", min_value=1.0e-9),
            FloatPrompt("eccentricity", "Eccentricity", "unitless", min_value=0.0, max_value=0.999999),
            FloatPrompt("inclination_deg", "Inclination", "deg", min_value=0.0, max_value=180.0),
            FloatPrompt("raan_deg", "RAAN", "deg"),
            FloatPrompt("argp_deg", "Argument of perigee", "deg"),
            FloatPrompt("true_anomaly_deg", "True anomaly", "deg"),
        ),
        compute=classical_coe_to_rv,
        formatter=_format_coe_to_rv,
    ),
    CalculatorSpec(
        category=CATEGORY_STATE_ELEMENTS,
        title="Circular inclined elements to RV",
        prompts=(
            FloatPrompt("semi_major_axis_km", "Semi-major axis", "km", min_value=1.0e-9),
            FloatPrompt("inclination_deg", "Inclination", "deg", min_value=0.0, max_value=180.0),
            FloatPrompt("raan_deg", "RAAN", "deg"),
            FloatPrompt("argument_of_latitude_deg", "Argument of latitude", "deg"),
        ),
        compute=circular_inclined_elements_to_rv,
        formatter=_format_coe_to_rv,
    ),
    CalculatorSpec(
        category=CATEGORY_STATE_ELEMENTS,
        title="Equatorial elliptical elements to RV",
        prompts=(
            FloatPrompt("semi_major_axis_km", "Semi-major axis", "km", min_value=1.0e-9),
            FloatPrompt("eccentricity", "Eccentricity", "unitless", min_value=1.0e-9, max_value=0.999999),
            FloatPrompt("longitude_of_perigee_deg", "Longitude of perigee", "deg"),
            FloatPrompt("true_anomaly_deg", "True anomaly", "deg"),
        ),
        compute=equatorial_elliptical_elements_to_rv,
        formatter=_format_coe_to_rv,
    ),
    CalculatorSpec(
        category=CATEGORY_STATE_ELEMENTS,
        title="Circular equatorial elements to RV",
        prompts=(
            FloatPrompt("semi_major_axis_km", "Semi-major axis", "km", min_value=1.0e-9),
            FloatPrompt("true_longitude_deg", "True longitude", "deg"),
        ),
        compute=circular_equatorial_elements_to_rv,
        formatter=_format_coe_to_rv,
    ),
    CalculatorSpec(
        category=CATEGORY_TRANSFERS,
        title="Hohmann transfer between circular orbits",
        prompts=(
            FloatPrompt("initial_altitude_km", "Initial circular altitude", "km", min_value=0.0),
            FloatPrompt("final_altitude_km", "Final circular altitude", "km", min_value=0.0),
        ),
        compute=hohmann_transfer_between_circular_orbits,
        formatter=_format_hohmann,
    ),
    CalculatorSpec(
        category=CATEGORY_TRANSFERS,
        title="Hohmann rendezvous phase angle",
        prompts=(
            FloatPrompt("initial_altitude_km", "Chaser initial circular altitude", "km", min_value=0.0),
            FloatPrompt("target_altitude_km", "Target circular altitude", "km", min_value=0.0),
        ),
        compute=hohmann_rendezvous_phase_angle,
        formatter=_format_hohmann_phase,
    ),
    CalculatorSpec(
        category=CATEGORY_TRANSFERS,
        title="Hohmann rendezvous wait time",
        prompts=(
            FloatPrompt("initial_altitude_km", "Chaser initial circular altitude", "km", min_value=0.0),
            FloatPrompt("target_altitude_km", "Target circular altitude", "km", min_value=0.0),
            FloatPrompt("current_phase_angle_deg", "Current target-ahead phase angle", "deg"),
        ),
        compute=hohmann_rendezvous_wait_time,
        formatter=_format_hohmann_wait,
    ),
    CalculatorSpec(
        category=CATEGORY_TRANSFERS,
        title="Plane change delta-v",
        prompts=(
            FloatPrompt("speed_km_s", "Orbital speed at plane change", "km/s", min_value=0.0),
            FloatPrompt("angle_deg", "Plane-change angle", "deg", min_value=0.0, max_value=180.0),
        ),
        compute=plane_change_delta_v,
        formatter=_format_plane_change,
    ),
    CalculatorSpec(
        category=CATEGORY_TRANSFERS,
        title="Inclination change cost from circular altitude",
        prompts=(
            FloatPrompt("altitude_km", "Circular orbit altitude", "km", min_value=0.0),
            FloatPrompt("angle_deg", "Inclination change angle", "deg", min_value=0.0, max_value=180.0),
        ),
        compute=inclination_change_from_altitude,
        formatter=_format_plane_change,
    ),
    CalculatorSpec(
        category=CATEGORY_TRANSFERS,
        title="Combined speed and plane change delta-v",
        prompts=(
            FloatPrompt("initial_speed_km_s", "Initial speed", "km/s", min_value=0.0),
            FloatPrompt("final_speed_km_s", "Final speed", "km/s", min_value=0.0),
            FloatPrompt("angle_deg", "Turn angle", "deg", min_value=0.0, max_value=180.0),
        ),
        compute=combined_speed_plane_change_delta_v,
        formatter=_format_combined_plane_change,
    ),
    CalculatorSpec(
        category=CATEGORY_SUN_SYNC,
        title="Sun-synchronous inclination from altitude",
        prompts=(FloatPrompt("altitude_km", "Circular orbit altitude", "km", min_value=0.0),),
        compute=sun_synchronous_inclination_from_altitude,
        formatter=_format_sun_sync,
    ),
    CalculatorSpec(
        category=CATEGORY_SUN_SYNC,
        title="J2 secular rates from altitude",
        prompts=(
            FloatPrompt("altitude_km", "Orbit altitude", "km", min_value=0.0),
            FloatPrompt("eccentricity", "Eccentricity", "unitless", min_value=0.0, max_value=0.999999),
            FloatPrompt("inclination_deg", "Inclination", "deg", min_value=0.0, max_value=180.0),
        ),
        compute=j2_secular_rates_from_altitude,
        formatter=_format_j2_rates,
    ),
    CalculatorSpec(
        category=CATEGORY_SUN_SYNC,
        title="J2 secular rates from semi-major axis",
        prompts=(
            FloatPrompt("semi_major_axis_km", "Semi-major axis", "km", min_value=1.0e-9),
            FloatPrompt("eccentricity", "Eccentricity", "unitless", min_value=0.0, max_value=0.999999),
            FloatPrompt("inclination_deg", "Inclination", "deg", min_value=0.0, max_value=180.0),
        ),
        compute=j2_secular_rates_from_elements,
        formatter=_format_j2_rates,
    ),
    CalculatorSpec(
        category=CATEGORY_PHASING,
        title="Phasing drift from altitude change",
        prompts=(
            FloatPrompt("reference_altitude_km", "Reference circular altitude", "km", min_value=0.0),
            FloatPrompt("phasing_altitude_change_km", "Phasing altitude change", "km"),
        ),
        compute=phasing_drift_from_altitude_change,
        formatter=_format_phasing,
    ),
    CalculatorSpec(
        category=CATEGORY_RELATIVE,
        title="HCW natural motion from altitude",
        prompts=(FloatPrompt("altitude_km", "Chief circular orbit altitude", "km", min_value=0.0),),
        compute=hcw_natural_motion_from_altitude,
        formatter=_format_hcw_natural,
    ),
    CalculatorSpec(
        category=CATEGORY_RELATIVE,
        title="HCW in-track drift estimate",
        prompts=(
            FloatPrompt("altitude_km", "Chief circular orbit altitude", "km", min_value=0.0),
            FloatPrompt("radial_offset_m", "Initial radial offset", "m"),
            FloatPrompt("intrack_velocity_bias_m_s", "Initial in-track velocity bias", "m/s"),
            FloatPrompt("duration_orbits", "Duration", "orbits", min_value=0.0),
        ),
        compute=hcw_intrack_drift_estimate,
        formatter=_format_hcw_drift,
    ),
    CalculatorSpec(
        category=CATEGORY_ECLIPSE,
        title="Circular-orbit eclipse estimate",
        prompts=(
            FloatPrompt("altitude_km", "Circular orbit altitude", "km", min_value=0.0),
            FloatPrompt("beta_angle_deg", "Sun beta angle", "deg", min_value=-90.0, max_value=90.0),
        ),
        compute=circular_eclipse_estimate,
        formatter=_format_eclipse,
    ),
    CalculatorSpec(
        category=CATEGORY_GROUND_TRACK,
        title="Ground-track drift from altitude",
        prompts=(FloatPrompt("altitude_km", "Circular orbit altitude", "km", min_value=0.0),),
        compute=ground_track_drift_from_altitude,
        formatter=_format_ground_track_drift,
    ),
    CalculatorSpec(
        category=CATEGORY_GROUND_TRACK,
        title="Repeat ground-track approximation",
        prompts=(
            FloatPrompt("altitude_km", "Circular orbit altitude", "km", min_value=0.0),
            FloatPrompt("max_days", "Maximum repeat-cycle search", "days", min_value=1.0),
        ),
        compute=repeat_ground_track_approximation,
        formatter=_format_repeat_ground_track,
    ),
    CalculatorSpec(
        category=CATEGORY_ENTRY,
        title="Entry interface from apogee/perigee",
        prompts=(
            FloatPrompt("apogee_altitude_km", "Apogee altitude", "km", min_value=0.0),
            FloatPrompt("perigee_altitude_km", "Perigee altitude", "km", min_value=0.0),
            FloatPrompt("interface_altitude_km", "Interface altitude", "km", min_value=0.0),
        ),
        compute=entry_interface_estimate,
        formatter=_format_entry_interface,
    ),
    CalculatorSpec(
        category=CATEGORY_DRAG,
        title="Ballistic coefficient",
        prompts=(
            FloatPrompt("mass_kg", "Spacecraft mass", "kg", min_value=1.0e-9),
            FloatPrompt("drag_coefficient", "Drag coefficient", "unitless", min_value=1.0e-9),
            FloatPrompt("drag_area_m2", "Drag reference area", "m^2", min_value=1.0e-9),
        ),
        compute=ballistic_coefficient,
        formatter=_format_ballistic_coefficient,
    ),
    CalculatorSpec(
        category=CATEGORY_DRAG,
        title="Density estimate from altitude",
        prompts=(FloatPrompt("altitude_km", "Altitude above Earth", "km", min_value=0.0),),
        compute=atmospheric_density_from_altitude,
        formatter=_format_density,
    ),
    CalculatorSpec(
        category=CATEGORY_DRAG,
        title="Drag force and acceleration from altitude",
        prompts=(
            FloatPrompt("altitude_km", "Altitude above Earth", "km", min_value=0.0),
            FloatPrompt("mass_kg", "Spacecraft mass", "kg", min_value=1.0e-9),
            FloatPrompt("drag_coefficient", "Drag coefficient", "unitless", min_value=1.0e-9),
            FloatPrompt("drag_area_m2", "Drag reference area", "m^2", min_value=1.0e-9),
        ),
        compute=drag_force_acceleration_from_altitude,
        formatter=_format_drag_at_altitude,
    ),
    CalculatorSpec(
        category=CATEGORY_DRAG,
        title="Drag force and acceleration from density/speed",
        prompts=(
            FloatPrompt("density_kg_m3", "Atmospheric density", "kg/m^3", min_value=0.0),
            FloatPrompt("speed_m_s", "Relative speed", "m/s", min_value=0.0),
            FloatPrompt("drag_coefficient", "Drag coefficient", "unitless", min_value=1.0e-9),
            FloatPrompt("drag_area_m2", "Drag reference area", "m^2", min_value=1.0e-9),
            FloatPrompt("mass_kg", "Spacecraft mass", "kg", min_value=1.0e-9),
        ),
        compute=drag_force_acceleration_from_density_speed,
        formatter=_format_drag_force,
    ),
    CalculatorSpec(
        category=CATEGORY_DRAG,
        title="Circular-orbit drag decay rate estimate",
        prompts=(
            FloatPrompt("altitude_km", "Circular orbit altitude", "km", min_value=0.0),
            FloatPrompt("ballistic_coefficient_kg_m2", "Ballistic coefficient", "kg/m^2", min_value=1.0e-9),
        ),
        compute=circular_orbit_drag_decay_rate,
        formatter=_format_decay_rate,
    ),
    CalculatorSpec(
        category=CATEGORY_DRAG,
        title="Deorbit lifetime range estimate",
        prompts=(
            FloatPrompt("initial_altitude_km", "Initial circular altitude", "km", min_value=0.0),
            FloatPrompt("ballistic_coefficient_kg_m2", "Ballistic coefficient", "kg/m^2", min_value=1.0e-9),
            FloatPrompt("deorbit_altitude_km", "Deorbit altitude", "km", min_value=0.0),
        ),
        compute=drag_lifetime_range_estimate,
        formatter=_format_lifetime_range,
    ),
    CalculatorSpec(
        category=CATEGORY_ROCKET,
        title="Rocket equation delta-v from mass ratio",
        prompts=(
            FloatPrompt("isp_s", "Specific impulse", "s", min_value=1.0e-9),
            FloatPrompt("mass_ratio", "Initial mass / final mass", "unitless", min_value=1.0),
        ),
        compute=rocket_equation_delta_v,
        formatter=_format_rocket_delta_v,
    ),
    CalculatorSpec(
        category=CATEGORY_ROCKET,
        title="Rocket equation mass ratio from delta-v",
        prompts=(
            FloatPrompt("delta_v_m_s", "Delta-v", "m/s", min_value=0.0),
            FloatPrompt("isp_s", "Specific impulse", "s", min_value=1.0e-9),
        ),
        compute=rocket_equation_mass_ratio,
        formatter=_format_rocket_mass_ratio,
    ),
    CalculatorSpec(
        category=CATEGORY_CIRCULAR,
        title="Escape velocity from altitude",
        prompts=(FloatPrompt("altitude_km", "Altitude above Earth", "km", min_value=0.0),),
        compute=_escape_from_altitude,
        formatter=_format_escape,
    ),
)


def run_calculation(spec: CalculatorSpec, values: dict[str, float]) -> object:
    args = {prompt.key: float(values[prompt.key]) * float(prompt.multiplier) for prompt in spec.prompts}
    return spec.compute(**args)


def format_result(spec: CalculatorSpec, result: object) -> str:
    rows = spec.formatter(result)
    width = max(len(label) for label, _ in rows)
    lines = ["", "Results"]
    lines.extend(f"{label:<{width}}  {value}" for label, value in rows)
    lines.extend(["", _assumptions(spec)])
    return "\n".join(lines)


def prompt_float(prompt: FloatPrompt, *, input_func: Callable[[str], str] = input) -> float:
    while True:
        raw = input_func(f"{prompt.display_label}: ").strip()
        try:
            value = float(raw)
        except ValueError:
            print("Please enter a number.")
            continue
        if prompt.min_value is not None and value < prompt.min_value:
            print(f"Please enter a value greater than or equal to {prompt.min_value:g}.")
            continue
        if prompt.max_value is not None and value > prompt.max_value:
            print(f"Please enter a value less than or equal to {prompt.max_value:g}.")
            continue
        return value


def prompt_yes_no(prompt: str, *, default: bool = True, input_func: Callable[[str], str] = input) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        raw = input_func(f"{prompt} {suffix}: ").strip().lower()
        if not raw:
            return bool(default)
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Please enter y or n.")


def _numbered_menu(
    options: list[str],
    *,
    heading: str,
    question: str,
    cancel_label: str,
    input_func: Callable[[str], str] = input,
) -> int | None:
    print(heading)
    print("")
    print(question)
    print("")
    for idx, option in enumerate(options, start=1):
        print(f"{idx}. {option}")
    while True:
        raw = input_func(f"\nChoose an option number, or q to {cancel_label}: ").strip().lower()
        if raw in {"q", "quit", "exit"}:
            return None
        try:
            selected = int(raw)
        except ValueError:
            print("Please enter a valid option number.")
            continue
        if 1 <= selected <= len(options):
            return selected - 1
        print("Please enter a valid option number.")


def _arrow_menu(options: list[str], *, heading: str, question: str, cancel_label: str, quit_hint: str) -> int | None:
    try:
        import termios
        import tty
    except ImportError:
        return _numbered_menu(options, heading=heading, question=question, cancel_label=cancel_label)

    selected = 0
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    def write_line(text: str = "") -> None:
        sys.stdout.write(f"{text}\r\n")

    def render() -> None:
        sys.stdout.write("\x1b[2J\x1b[H")
        write_line(heading)
        write_line()
        write_line(question)
        write_line()
        for idx, option in enumerate(options):
            prefix = "> " if idx == selected else "  "
            write_line(f"{prefix}{option}")
        write_line()
        write_line(quit_hint)
        sys.stdout.flush()

    try:
        tty.setraw(fd)
        render()
        while True:
            ch = sys.stdin.read(1)
            if ch in {"q", "Q", "\x03"}:
                return None
            if ch in {"\r", "\n"}:
                write_line()
                return selected
            if ch == "\x1b":
                seq = sys.stdin.read(2)
                if seq == "[A":
                    selected = (selected - 1) % len(options)
                    render()
                elif seq == "[B":
                    selected = (selected + 1) % len(options)
                    render()
            elif ch in {"k", "K"}:
                selected = (selected - 1) % len(options)
                render()
            elif ch in {"j", "J"}:
                selected = (selected + 1) % len(options)
                render()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _select_menu(
    options: list[str],
    *,
    heading: str,
    question: str,
    cancel_label: str,
    quit_hint: str,
) -> int | None:
    if sys.stdin.isatty() and sys.stdout.isatty():
        return _arrow_menu(options, heading=heading, question=question, cancel_label=cancel_label, quit_hint=quit_hint)
    return _numbered_menu(options, heading=heading, question=question, cancel_label=cancel_label)


def calculators_for_category(category: str) -> tuple[CalculatorSpec, ...]:
    return tuple(spec for spec in CALCULATORS if spec.category == category)


def select_category() -> str | None:
    options = list(CATEGORIES) + ["Quit"]
    selected = _select_menu(
        options,
        heading="Orbital Calculator",
        question="What kind of calculation?",
        cancel_label="quit",
        quit_hint="Use Up/Down arrows and Enter. Press q to quit.",
    )
    if selected is None or selected >= len(CATEGORIES):
        return None
    return CATEGORIES[selected]


def select_calculator(category: str) -> CalculatorSpec | None:
    calculators = calculators_for_category(category)
    options = [spec.title for spec in calculators] + ["Back"]
    selected = _select_menu(
        options,
        heading=category,
        question="What do you want to calculate?",
        cancel_label="go back",
        quit_hint="Use Up/Down arrows and Enter. Press q to go back.",
    )
    if selected is None or selected >= len(calculators):
        return None
    return calculators[selected]


def run_interactive_calculation(spec: CalculatorSpec) -> bool:
    print("")
    print(spec.title)
    values = {prompt.key: prompt_float(prompt) for prompt in spec.prompts}
    try:
        result = run_calculation(spec, values)
    except ValueError as exc:
        print(f"\nUnable to calculate: {exc}")
        return prompt_yes_no("\nCalculate another?")
    print(format_result(spec, result))
    return prompt_yes_no("\nCalculate another?")


def main() -> int:
    while True:
        category = select_category()
        if category is None:
            print("Goodbye.")
            return 0
        while True:
            spec = select_calculator(category)
            if spec is None:
                break
            if not run_interactive_calculation(spec):
                print("Goodbye.")
                return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
