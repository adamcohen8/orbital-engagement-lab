from __future__ import annotations

import argparse

import numpy as np

from sim.acceleration.kernels.attitude import (
    propagate_attitude_builtin_disturbances_kernel,
    propagate_attitude_exponential_map_kernel,
)
from sim.acceleration.kernels.de440 import (
    de440_light_core_kernel,
    de440_sun_moon_from_utc_kernel,
)
from sim.acceleration.kernels.estimation import (
    attitude_ekf_numerical_jacobian_kernel,
    attitude_ekf_propagate_state_kernel,
    orbit_ekf_numerical_jacobian_kernel,
    propagate_two_body_rk4_kernel,
)
from sim.acceleration.kernels.frames import (
    apparent_sidereal_time_iau76_80_kernel,
    eci_relative_to_ric_rect_kernel,
    eci_to_ecef_iau76_80_kernel,
    ric_angular_rate_eci_from_rv_kernel,
    ric_curv_to_rect_kernel,
    ric_dcm_ir_from_rv_kernel,
    ric_rect_state_to_eci_kernel,
    ric_rect_to_curv_kernel,
)
from sim.acceleration.kernels.geodesy import ecef_to_geodetic_deg_km_kernel
from sim.acceleration.kernels.msis86 import denss_kernel as msis86_denss_kernel
from sim.acceleration.kernels.nrlmsise00 import (
    densu_kernel as nrlmsise00_densu_kernel,
)
from sim.acceleration.kernels.nrlmsise00 import (
    quiet_thermosphere_density_kernel as nrlmsise00_quiet_thermosphere_density_kernel,
)
from sim.acceleration.kernels.orbit import (
    j2_accel_eci,
    j3_accel_eci,
    j4_accel_eci,
    rk4_zonal_step_state,
    two_body_accel_eci,
)
from sim.acceleration.kernels.orbit_force_plan import (
    builtin_force_components_kernel,
    rk4_builtin_force_plan_step_kernel,
)
from sim.acceleration.kernels.reentry import (
    atmosphere_relative_velocity_eci_km_s_kernel,
    radial_altitude_km_kernel,
    reentry_scalar_metrics_kernel,
)
from sim.acceleration.kernels.spherical_harmonics import normalized_spherical_harmonic_accel_eci_kernel
from sim.acceleration.kernels.srp import srp_acceleration_kernel
from sim.acceleration.optional import NUMBA_AVAILABLE, NUMBA_IMPORT_ERROR
from sim.dynamics.orbit.frames import _load_nut80_table
from sim.dynamics.orbit.nrlmsise00_backend import (
    _ALPHA,
    _ZN1,
    PD1,
    PDL1,
    PDM1,
    PMA1,
    PS1,
    PT1,
    PTL1,
    PTM1,
)

EARTH_MU_KM3_S2 = 398600.4415


def warmup_acceleration(profile: str = "core") -> dict[str, object]:
    profile_name = str(profile or "core").strip().lower()
    if profile_name not in {"core", "validation"}:
        raise ValueError("warmup profile must be 'core' or 'validation'.")

    r = np.array([7000.0, 10.0, 20.0], dtype=float)
    v = np.array([0.0, 7.5, 0.01], dtype=float)
    x = np.hstack((r, v)).astype(float)
    u = np.array([0.0, 1.0e-9, 0.0], dtype=float)
    rel = np.array([0.1, -1.0, 0.05, 0.0, 0.0001, -0.00002], dtype=float)
    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    omega = np.array([0.01, -0.02, 0.03], dtype=float)
    inertia = np.diag(np.array([100.0, 90.0, 80.0], dtype=float))
    torque = np.array([0.001, -0.002, 0.003], dtype=float)
    att_state = np.hstack((quat, omega)).astype(float)
    att_base = attitude_ekf_propagate_state_kernel(att_state, 1.0, inertia)
    orbit_base = propagate_two_body_rk4_kernel(x, 1.0, EARTH_MU_KM3_S2)
    harmonic_c_nm = np.zeros((3, 1), dtype=float)
    harmonic_s_nm = np.zeros((3, 1), dtype=float)
    harmonic_c_nm[0, 0] = 1.0
    harmonic_c_nm[2, 0] = -4.841693259705e-4
    harmonic_diag_scale = np.zeros(3, dtype=float)
    harmonic_subdiag_scale = np.zeros(3, dtype=float)
    harmonic_subdiag_scale[1] = np.sqrt(3.0)
    harmonic_subdiag_scale[2] = np.sqrt(5.0)
    harmonic_recur_a = np.zeros((3, 1), dtype=float)
    harmonic_recur_b = np.zeros((3, 1), dtype=float)
    harmonic_recur_c = np.zeros((3, 1), dtype=float)
    harmonic_recur_a[2, 0] = np.sqrt(5.0 / 4.0)
    harmonic_recur_b[2, 0] = np.sqrt(3.0)
    harmonic_recur_c[2, 0] = 1.0
    force_codes = np.arange(1, 11, dtype=np.int64)
    force_parameters = np.array(
        [
            EARTH_MU_KM3_S2,
            6378.1363,
            100.0,
            2.2,
            1.0,
            7.2921150e-5,
            1.0,
            1.2,
            4.54e-6,
            149597870.7,
            6378.137,
            695700.0,
            132712440041.93938,
            4902.800066,
            EARTH_MU_KM3_S2,
        ],
        dtype=float,
    )
    identity = np.eye(3, dtype=float)
    stage_rotations = np.repeat(identity[None, :, :], 3, axis=0)
    stage_sun_positions = np.repeat(
        np.array([[149597870.7, 0.0, 0.0]], dtype=float),
        3,
        axis=0,
    )
    stage_moon_positions = np.repeat(
        np.array([[384400.0, 0.0, 0.0]], dtype=float),
        3,
        axis=0,
    )
    planet_positions = np.zeros((9, 3), dtype=float)
    planet_positions[0] = np.array([5.0e7, 1.0e8, 2.0e7], dtype=float)
    planet_mu = np.zeros(9, dtype=float)
    planet_mu[0] = 324858.592
    ephemeris_starts = np.array([2451545.0], dtype=float)
    ephemeris_ends = np.array([2451577.0], dtype=float)
    ephemeris_coefficients = np.array([[1.0]], dtype=float)
    ephemeris_kernel_args = (
        1,
        1,
        32.0,
        ephemeris_starts,
        ephemeris_ends,
        ephemeris_coefficients,
        ephemeris_coefficients,
        ephemeris_coefficients,
    ) * 3
    nutation_coefficients, nutation_terms = _load_nut80_table()

    calls: list[tuple[str, object]] = []
    calls.append(("two_body_accel_eci", two_body_accel_eci(r, EARTH_MU_KM3_S2)))
    calls.append(("j2_accel_eci", j2_accel_eci(r, EARTH_MU_KM3_S2)))
    calls.append(("j3_accel_eci", j3_accel_eci(r, EARTH_MU_KM3_S2)))
    calls.append(("j4_accel_eci", j4_accel_eci(r, EARTH_MU_KM3_S2)))
    calls.append(("rk4_zonal_step_state", rk4_zonal_step_state(x, 1.0, u, EARTH_MU_KM3_S2, True, True, True)))
    calls.append(
        (
            "msis86_denss_kernel",
            msis86_denss_kernel(
                400.0,
                1.0e8,
                1000.0,
                400.0,
                28.0,
                0.0,
                120.0,
                0.02,
                200.0,
                120.0,
                90.0,
                0.5,
                980.0,
                6367.0,
            ),
        )
    )
    calls.append(
        (
            "nrlmsise00_densu_kernel",
            nrlmsise00_densu_kernel(
                400.0,
                1.0e8,
                1000.0,
                400.0,
                28.0,
                0.0,
                0.0,
                120.0,
                0.02,
                5,
                np.array([0.0, 120.0, 110.0, 100.0, 90.0, 72.5], dtype=float),
                np.array([0.0, 400.0, 350.0, 300.0, 250.0, 200.0], dtype=float),
                np.array([0.0, 0.0, 0.0], dtype=float),
                980.0,
                6367.0,
            ),
        )
    )
    calls.append(
        (
            "nrlmsise00_quiet_thermosphere_density_kernel",
            nrlmsise00_quiet_thermosphere_density_kernel(
                90,
                43200.0,
                400.0,
                10.0,
                20.0,
                12.0,
                150.0,
                150.0,
                PT1,
                PS1,
                PD1,
                PDL1,
                PTM1,
                PDM1,
                PTL1,
                PMA1,
                _ZN1,
                _ALPHA,
            ),
        )
    )
    calls.append(("ecef_to_geodetic_deg_km_kernel", ecef_to_geodetic_deg_km_kernel(r)))
    calls.append(
        (
            "eci_to_ecef_iau76_80_kernel",
            eci_to_ecef_iau76_80_kernel(
                0.0,
                2459669.5,
                0.05,
                -0.03,
                0.1,
                37.0,
                0.0,
                0.0,
                nutation_coefficients,
                nutation_terms,
            ),
        )
    )
    calls.append(
        (
            "apparent_sidereal_time_iau76_80_kernel",
            apparent_sidereal_time_iau76_80_kernel(
                2459669.5,
                0.1,
                37.0,
                0.0,
                0.0,
                nutation_coefficients,
                nutation_terms,
            ),
        )
    )
    calls.append(
        (
            "normalized_spherical_harmonic_accel_eci_kernel",
            normalized_spherical_harmonic_accel_eci_kernel(
                r,
                np.eye(3, dtype=float),
                EARTH_MU_KM3_S2,
                6378.1363,
                harmonic_c_nm,
                harmonic_s_nm,
                harmonic_diag_scale,
                harmonic_subdiag_scale,
                harmonic_recur_a,
                harmonic_recur_b,
                harmonic_recur_c,
                2,
                0,
            ),
        )
    )
    calls.append(
        (
            "de440_light_core_kernel",
            de440_light_core_kernel(
                2451546.0,
                *ephemeris_kernel_args,
            ),
        )
    )
    calls.append(
        (
            "de440_sun_moon_from_utc_kernel",
            de440_sun_moon_from_utc_kernel(
                2451546.0,
                37.0,
                81.3005682214972154,
                *ephemeris_kernel_args,
            ),
        )
    )
    calls.append(
        (
            "srp_acceleration_kernel",
            srp_acceleration_kernel(
                r,
                np.array([149597870.7, 0.0, 0.0], dtype=float),
                100.0,
                1.0,
                1.2,
                4.54e-6,
                149597870.7,
                6378.137,
                695700.0,
                2,
            ),
        )
    )
    calls.append(
        (
            "builtin_force_components_kernel",
            builtin_force_components_kernel(
                x,
                force_codes,
                np.full(force_codes.size, 1.0e-12, dtype=float),
                identity,
                identity,
                stage_sun_positions[0],
                stage_moon_positions[0],
                planet_positions,
                planet_mu,
                1,
                force_parameters,
                2,
                np.array([0.0, 0.0, 1.0], dtype=float),
                0.2,
                1.0,
                harmonic_c_nm,
                harmonic_s_nm,
                harmonic_diag_scale,
                harmonic_subdiag_scale,
                harmonic_recur_a,
                harmonic_recur_b,
                harmonic_recur_c,
                2,
                0,
            ),
        )
    )
    calls.append(
        (
            "rk4_builtin_force_plan_step_kernel",
            rk4_builtin_force_plan_step_kernel(
                x,
                1.0,
                u,
                np.arange(1, 6, dtype=np.int64),
                stage_rotations,
                stage_rotations,
                stage_rotations,
                stage_sun_positions,
                stage_moon_positions,
                np.zeros((3, 6), dtype=float),
                force_parameters,
                1,
                1.0e-12,
                2,
                harmonic_c_nm,
                harmonic_s_nm,
                harmonic_diag_scale,
                harmonic_subdiag_scale,
                harmonic_recur_a,
                harmonic_recur_b,
                harmonic_recur_c,
                2,
                0,
                PT1,
                PS1,
                PD1,
                PDL1,
                PTM1,
                PDM1,
                PTL1,
                PMA1,
                _ZN1,
                _ALPHA,
            ),
        )
    )
    calls.append(("propagate_two_body_rk4_kernel", orbit_base))
    calls.append(
        (
            "orbit_ekf_numerical_jacobian_kernel",
            orbit_ekf_numerical_jacobian_kernel(x, orbit_base, 1.0, EARTH_MU_KM3_S2),
        )
    )
    calls.append(("ric_dcm_ir_from_rv_kernel", ric_dcm_ir_from_rv_kernel(r, v)))
    calls.append(("ric_angular_rate_eci_from_rv_kernel", ric_angular_rate_eci_from_rv_kernel(r, v)))
    calls.append(("ric_rect_state_to_eci_kernel", ric_rect_state_to_eci_kernel(rel, r, v)))
    calls.append(("eci_relative_to_ric_rect_kernel", eci_relative_to_ric_rect_kernel(x, np.hstack((r, v)))))
    calls.append(("ric_curv_to_rect_kernel", ric_curv_to_rect_kernel(rel, float(np.linalg.norm(r)))))
    calls.append(("ric_rect_to_curv_kernel", ric_rect_to_curv_kernel(rel, float(np.linalg.norm(r)))))
    calls.append(
        (
            "propagate_attitude_exponential_map_kernel",
            propagate_attitude_exponential_map_kernel(quat, omega, inertia, torque, 1.0),
        )
    )
    empty_facets = np.empty((0, 3), dtype=float)
    empty_scalars = np.empty(0, dtype=float)
    calls.append(
        (
            "propagate_attitude_builtin_disturbances_kernel",
            propagate_attitude_builtin_disturbances_kernel(
                quat,
                omega,
                inertia,
                torque,
                np.array([1.0], dtype=float),
                r,
                EARTH_MU_KM3_S2,
                np.array([1, 0, 0, 0], dtype=np.int64),
                np.zeros(3, dtype=float),
                np.zeros(3, dtype=float),
                False,
                0.0,
                np.zeros(3, dtype=float),
                0.0,
                0,
                0.0,
                0.0,
                np.zeros(3, dtype=float),
                empty_facets,
                empty_scalars,
                empty_scalars,
                empty_facets,
                np.zeros(3, dtype=float),
                0.0,
                0,
                0.0,
                np.zeros(3, dtype=float),
                empty_facets,
                empty_scalars,
                empty_facets,
            ),
        )
    )
    calls.append(("attitude_ekf_propagate_state_kernel", att_base))
    calls.append(
        (
            "attitude_ekf_numerical_jacobian_kernel",
            attitude_ekf_numerical_jacobian_kernel(att_state, att_base, 1.0, inertia),
        )
    )
    if profile_name == "validation":
        calls.append(("radial_altitude_km_kernel", radial_altitude_km_kernel(r)))
        calls.append(
            ("atmosphere_relative_velocity_eci_km_s_kernel", atmosphere_relative_velocity_eci_km_s_kernel(r, v))
        )
        calls.append(
            (
                "reentry_scalar_metrics_kernel",
                reentry_scalar_metrics_kernel(r, v, 1.0e-8, 100.0, 1.0, 2.2, 0.5, 1.83e-4, 1.0, 0.0),
            )
        )
    return {
        "profile": profile_name,
        "backend": "numba" if NUMBA_AVAILABLE else "python",
        "numba_available": NUMBA_AVAILABLE,
        "kernel_count": len(calls),
        "kernels": [name for name, _value in calls],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Warm optional acceleration kernels.")
    parser.add_argument("--profile", choices=["core", "validation"], default="core")
    parser.add_argument("--list", action="store_true", help="List warmup profiles and exit.")
    args = parser.parse_args()
    if args.list:
        print("core")
        print("validation")
        return 0
    result = warmup_acceleration(profile=args.profile)
    print("Acceleration warmup")
    print(f"Profile : {result['profile']}")
    print(f"Backend : {result['backend']}")
    if not NUMBA_AVAILABLE and NUMBA_IMPORT_ERROR:
        print(f"Numba   : unavailable ({NUMBA_IMPORT_ERROR})")
    print(f"Kernels : {result['kernel_count']}")
    for name in result["kernels"]:
        print(f"- {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
