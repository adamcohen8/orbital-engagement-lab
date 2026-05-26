from __future__ import annotations

import argparse

import numpy as np

from sim.acceleration.kernels.attitude import propagate_attitude_exponential_map_kernel
from sim.acceleration.kernels.estimation import (
    attitude_ekf_numerical_jacobian_kernel,
    attitude_ekf_propagate_state_kernel,
    orbit_ekf_numerical_jacobian_kernel,
    propagate_two_body_rk4_kernel,
)
from sim.acceleration.kernels.frames import (
    eci_relative_to_ric_rect_kernel,
    ric_angular_rate_eci_from_rv_kernel,
    ric_curv_to_rect_kernel,
    ric_dcm_ir_from_rv_kernel,
    ric_rect_state_to_eci_kernel,
    ric_rect_to_curv_kernel,
)
from sim.acceleration.kernels.orbit import (
    j2_accel_eci,
    j3_accel_eci,
    j4_accel_eci,
    rk4_zonal_step_state,
    two_body_accel_eci,
)
from sim.acceleration.kernels.reentry import (
    atmosphere_relative_velocity_eci_km_s_kernel,
    radial_altitude_km_kernel,
    reentry_scalar_metrics_kernel,
)
from sim.acceleration.optional import NUMBA_AVAILABLE, NUMBA_IMPORT_ERROR

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

    calls: list[tuple[str, object]] = []
    calls.append(("two_body_accel_eci", two_body_accel_eci(r, EARTH_MU_KM3_S2)))
    calls.append(("j2_accel_eci", j2_accel_eci(r, EARTH_MU_KM3_S2)))
    calls.append(("j3_accel_eci", j3_accel_eci(r, EARTH_MU_KM3_S2)))
    calls.append(("j4_accel_eci", j4_accel_eci(r, EARTH_MU_KM3_S2)))
    calls.append(("rk4_zonal_step_state", rk4_zonal_step_state(x, 1.0, u, EARTH_MU_KM3_S2, True, True, True)))
    calls.append(("propagate_two_body_rk4_kernel", orbit_base))
    calls.append(("orbit_ekf_numerical_jacobian_kernel", orbit_ekf_numerical_jacobian_kernel(x, orbit_base, 1.0, EARTH_MU_KM3_S2)))
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
    calls.append(("attitude_ekf_propagate_state_kernel", att_base))
    calls.append(("attitude_ekf_numerical_jacobian_kernel", attitude_ekf_numerical_jacobian_kernel(att_state, att_base, 1.0, inertia)))
    if profile_name == "validation":
        calls.append(("radial_altitude_km_kernel", radial_altitude_km_kernel(r)))
        calls.append(("atmosphere_relative_velocity_eci_km_s_kernel", atmosphere_relative_velocity_eci_km_s_kernel(r, v)))
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
