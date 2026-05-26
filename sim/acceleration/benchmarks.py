from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

import numpy as np

from sim.acceleration.optional import acceleration_backend_name
from sim.acceleration.warmup import warmup_acceleration
from sim.core.models import Measurement, StateBelief
from sim.dynamics.orbit.accelerations import OrbitContext
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2
from sim.dynamics.orbit.propagator import OrbitPropagator, j2_plugin
from sim.estimation.attitude_ekf import AttitudeEKFEstimator
from sim.estimation.orbit_ekf import OrbitEKFEstimator
from sim.utils.quaternion import normalize_quaternion


@dataclass(frozen=True)
class OrbitKernelBenchmarkResult:
    iterations: int
    backend: str
    python_propagator_s: float
    accelerated_propagator_s: float
    speedup: float
    state_delta_norm: float


@dataclass(frozen=True)
class AttitudeKernelBenchmarkResult:
    iterations: int
    backend: str
    python_propagator_s: float
    accelerated_propagator_s: float
    speedup: float
    quaternion_delta_norm: float
    rate_delta_norm: float


@dataclass(frozen=True)
class EstimationKernelBenchmarkResult:
    iterations: int
    backend: str
    orbit_jacobian_python_s: float
    orbit_jacobian_accelerated_s: float
    orbit_jacobian_speedup: float
    orbit_jacobian_delta_norm: float
    attitude_jacobian_python_s: float
    attitude_jacobian_accelerated_s: float
    attitude_jacobian_speedup: float
    attitude_jacobian_delta_norm: float
    joint_update_python_s: float
    joint_update_accelerated_s: float
    joint_update_speedup: float
    joint_state_delta_norm: float


def benchmark_orbit_kernel(iterations: int = 10_000, *, warmup: bool = True) -> OrbitKernelBenchmarkResult:
    n = int(max(iterations, 1))
    if warmup:
        warmup_acceleration(profile="core")
    x0 = np.array([7000.0, -20.0, 30.0, 0.0, 7.5, 0.01], dtype=float)
    command = np.array([0.0, 1.0e-9, 0.0], dtype=float)
    ctx = OrbitContext(mu_km3_s2=EARTH_MU_KM3_S2, mass_kg=100.0)
    baseline = OrbitPropagator(integrator="rk4", plugins=[j2_plugin], acceleration_mode="off")
    accelerated = OrbitPropagator(integrator="rk4", plugins=[j2_plugin], acceleration_mode="auto")

    baseline.propagate(x0, 1.0, 0.0, command, {}, ctx)
    accelerated.propagate(x0, 1.0, 0.0, command, {}, ctx)

    y = x0.copy()
    t0 = time.perf_counter()
    for i in range(n):
        y = baseline.propagate(y, 1.0, float(i), command, {}, ctx)
    t1 = time.perf_counter()

    z = x0.copy()
    for i in range(n):
        z = accelerated.propagate(z, 1.0, float(i), command, {}, ctx)
    t2 = time.perf_counter()

    python_s = float(t1 - t0)
    accelerated_s = float(t2 - t1)
    return OrbitKernelBenchmarkResult(
        iterations=n,
        backend=acceleration_backend_name(),
        python_propagator_s=python_s,
        accelerated_propagator_s=accelerated_s,
        speedup=float(python_s / max(accelerated_s, 1e-12)),
        state_delta_norm=float(np.linalg.norm(y - z)),
    )


def benchmark_attitude_kernel(iterations: int = 100_000, *, warmup: bool = True) -> AttitudeKernelBenchmarkResult:
    from sim.dynamics.attitude.rigid_body import (
        propagate_attitude_exponential_map,
        reset_attitude_guardrail_stats,
    )

    n = int(max(iterations, 1))
    if warmup:
        warmup_acceleration(profile="core")
    q0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    w0 = np.array([0.01, -0.02, 0.03], dtype=float)
    inertia = np.diag(np.array([100.0, 90.0, 80.0], dtype=float))
    torque = np.array([0.001, -0.002, 0.003], dtype=float)
    dt_s = 0.1

    propagate_attitude_exponential_map(q0, w0, inertia, torque, dt_s, acceleration_mode="off")
    propagate_attitude_exponential_map(q0, w0, inertia, torque, dt_s, acceleration_mode="auto")

    reset_attitude_guardrail_stats()
    q = q0.copy()
    w = w0.copy()
    t0 = time.perf_counter()
    for _ in range(n):
        q, w = propagate_attitude_exponential_map(q, w, inertia, torque, dt_s, acceleration_mode="off")
    t1 = time.perf_counter()

    reset_attitude_guardrail_stats()
    q_acc = q0.copy()
    w_acc = w0.copy()
    for _ in range(n):
        q_acc, w_acc = propagate_attitude_exponential_map(q_acc, w_acc, inertia, torque, dt_s, acceleration_mode="auto")
    t2 = time.perf_counter()

    python_s = float(t1 - t0)
    accelerated_s = float(t2 - t1)
    return AttitudeKernelBenchmarkResult(
        iterations=n,
        backend=acceleration_backend_name(),
        python_propagator_s=python_s,
        accelerated_propagator_s=accelerated_s,
        speedup=float(python_s / max(accelerated_s, 1e-12)),
        quaternion_delta_norm=float(np.linalg.norm(q - q_acc)),
        rate_delta_norm=float(np.linalg.norm(w - w_acc)),
    )


def benchmark_estimation_kernel(iterations: int = 1_000, *, warmup: bool = True) -> EstimationKernelBenchmarkResult:
    n = int(max(iterations, 1))
    if warmup:
        warmup_acceleration(profile="core")

    orbit_state = np.array([7000.0, 1.0, -0.5, -0.001, 7.5460, 0.002], dtype=float)
    orbit_cov = np.eye(6) * 1e-3
    orbit_meas = Measurement(vector=orbit_state + np.array([1e-3, -2e-3, 1e-3, 1e-6, -1e-6, 2e-6]), t_s=1.0)
    orbit_base = OrbitEKFEstimator(
        mu_km3_s2=EARTH_MU_KM3_S2,
        dt_s=1.0,
        process_noise_diag=np.ones(6) * 1e-10,
        meas_noise_diag=np.ones(6) * 1e-6,
        acceleration_mode="off",
    )
    orbit_acc = OrbitEKFEstimator(
        mu_km3_s2=EARTH_MU_KM3_S2,
        dt_s=1.0,
        process_noise_diag=np.ones(6) * 1e-10,
        meas_noise_diag=np.ones(6) * 1e-6,
        acceleration_mode="auto",
    )

    attitude_state = np.hstack(
        (normalize_quaternion(np.array([1.0, 0.01, -0.02, 0.0])), np.array([0.01, -0.02, 0.03]))
    )
    attitude_cov = np.eye(7) * 1e-3
    attitude_meas = Measurement(
        vector=np.hstack(
            (normalize_quaternion(np.array([1.0, 0.012, -0.018, 0.002])), np.array([0.011, -0.019, 0.029]))
        ),
        t_s=1.0,
    )
    inertia = np.diag([10.0, 12.0, 8.0])
    att_base = AttitudeEKFEstimator(
        dt_s=1.0,
        inertia_kg_m2=inertia,
        process_noise_diag=np.ones(7) * 1e-8,
        meas_noise_diag=np.ones(7) * 1e-6,
        acceleration_mode="off",
    )
    att_acc = AttitudeEKFEstimator(
        dt_s=1.0,
        inertia_kg_m2=inertia,
        process_noise_diag=np.ones(7) * 1e-8,
        meas_noise_diag=np.ones(7) * 1e-6,
        acceleration_mode="auto",
    )

    orbit_base._numerical_jacobian(orbit_state, dt_s=1.0)
    orbit_acc._numerical_jacobian(orbit_state, dt_s=1.0)
    att_base._numerical_jacobian(attitude_state, dt_s=1.0)
    att_acc._numerical_jacobian(attitude_state, dt_s=1.0)

    t0 = time.perf_counter()
    for _ in range(n):
        orbit_j_base = orbit_base._numerical_jacobian(orbit_state, dt_s=1.0)
    t1 = time.perf_counter()
    for _ in range(n):
        orbit_j_acc = orbit_acc._numerical_jacobian(orbit_state, dt_s=1.0)
    t2 = time.perf_counter()
    for _ in range(n):
        att_j_base = att_base._numerical_jacobian(attitude_state, dt_s=1.0)
    t3 = time.perf_counter()
    for _ in range(n):
        att_j_acc = att_acc._numerical_jacobian(attitude_state, dt_s=1.0)
    t4 = time.perf_counter()

    orbit_belief = StateBelief(state=orbit_state, covariance=orbit_cov, last_update_t_s=0.0)
    attitude_belief = StateBelief(state=attitude_state, covariance=attitude_cov, last_update_t_s=0.0)
    t5 = time.perf_counter()
    for _ in range(n):
        orbit_upd_base = orbit_base.update(orbit_belief, orbit_meas, 1.0)
        attitude_upd_base = att_base.update(attitude_belief, attitude_meas, 1.0)
    t6 = time.perf_counter()
    for _ in range(n):
        orbit_upd_acc = orbit_acc.update(orbit_belief, orbit_meas, 1.0)
        attitude_upd_acc = att_acc.update(attitude_belief, attitude_meas, 1.0)
    t7 = time.perf_counter()

    joint_state_delta = np.linalg.norm(orbit_upd_base.state - orbit_upd_acc.state) + np.linalg.norm(
        attitude_upd_base.state - attitude_upd_acc.state
    )
    orbit_python_s = float(t1 - t0)
    orbit_acc_s = float(t2 - t1)
    att_python_s = float(t3 - t2)
    att_acc_s = float(t4 - t3)
    update_python_s = float(t6 - t5)
    update_acc_s = float(t7 - t6)
    return EstimationKernelBenchmarkResult(
        iterations=n,
        backend=acceleration_backend_name(),
        orbit_jacobian_python_s=orbit_python_s,
        orbit_jacobian_accelerated_s=orbit_acc_s,
        orbit_jacobian_speedup=float(orbit_python_s / max(orbit_acc_s, 1e-12)),
        orbit_jacobian_delta_norm=float(np.linalg.norm(orbit_j_base - orbit_j_acc)),
        attitude_jacobian_python_s=att_python_s,
        attitude_jacobian_accelerated_s=att_acc_s,
        attitude_jacobian_speedup=float(att_python_s / max(att_acc_s, 1e-12)),
        attitude_jacobian_delta_norm=float(np.linalg.norm(att_j_base - att_j_acc)),
        joint_update_python_s=update_python_s,
        joint_update_accelerated_s=update_acc_s,
        joint_update_speedup=float(update_python_s / max(update_acc_s, 1e-12)),
        joint_state_delta_norm=float(joint_state_delta),
    )


def _print_orbit_result(result: OrbitKernelBenchmarkResult) -> None:
    print("")
    print("=" * 72)
    print("ACCELERATION ORBIT KERNEL BENCHMARK")
    print("=" * 72)
    print(f"Backend              : {result.backend}")
    print(f"Iterations           : {result.iterations}")
    print(f"Python Propagator    : {result.python_propagator_s:.6f} s")
    print(f"Accelerated Path     : {result.accelerated_propagator_s:.6f} s")
    print(f"Speedup              : {result.speedup:.2f}x")
    print(f"State Delta Norm     : {result.state_delta_norm:.3e}")
    print("=" * 72)


def _print_attitude_result(result: AttitudeKernelBenchmarkResult) -> None:
    print("")
    print("=" * 72)
    print("ACCELERATION ATTITUDE KERNEL BENCHMARK")
    print("=" * 72)
    print(f"Backend              : {result.backend}")
    print(f"Iterations           : {result.iterations}")
    print(f"Python Propagator    : {result.python_propagator_s:.6f} s")
    print(f"Accelerated Path     : {result.accelerated_propagator_s:.6f} s")
    print(f"Speedup              : {result.speedup:.2f}x")
    print(f"Quaternion Delta     : {result.quaternion_delta_norm:.3e}")
    print(f"Rate Delta           : {result.rate_delta_norm:.3e}")
    print("=" * 72)


def _print_estimation_result(result: EstimationKernelBenchmarkResult) -> None:
    print("")
    print("=" * 72)
    print("ACCELERATION ESTIMATION KERNEL BENCHMARK")
    print("=" * 72)
    print(f"Backend              : {result.backend}")
    print(f"Iterations           : {result.iterations}")
    print(f"Orbit Jacobian Py    : {result.orbit_jacobian_python_s:.6f} s")
    print(f"Orbit Jacobian Accel : {result.orbit_jacobian_accelerated_s:.6f} s")
    print(f"Orbit Jacobian Speed : {result.orbit_jacobian_speedup:.2f}x")
    print(f"Orbit Jacobian Delta : {result.orbit_jacobian_delta_norm:.3e}")
    print(f"Att Jacobian Py      : {result.attitude_jacobian_python_s:.6f} s")
    print(f"Att Jacobian Accel   : {result.attitude_jacobian_accelerated_s:.6f} s")
    print(f"Att Jacobian Speed   : {result.attitude_jacobian_speedup:.2f}x")
    print(f"Att Jacobian Delta   : {result.attitude_jacobian_delta_norm:.3e}")
    print(f"EKF Update Py        : {result.joint_update_python_s:.6f} s")
    print(f"EKF Update Accel     : {result.joint_update_accelerated_s:.6f} s")
    print(f"EKF Update Speed     : {result.joint_update_speedup:.2f}x")
    print(f"Joint State Delta    : {result.joint_state_delta_norm:.3e}")
    print("=" * 72)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run optional acceleration benchmarks.")
    parser.add_argument("--kind", choices=["orbit", "attitude", "estimation", "all"], default="orbit")
    parser.add_argument("--iterations", type=int, default=10_000, help="Orbit-kernel iteration count.")
    parser.add_argument("--attitude-iterations", type=int, default=100_000, help="Attitude-kernel iteration count.")
    parser.add_argument("--estimation-iterations", type=int, default=1_000, help="Estimation-kernel iteration count.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument("--no-warmup", action="store_true", help="Skip acceleration warmup before timing.")
    args = parser.parse_args()
    results = {}
    if args.kind in {"orbit", "all"}:
        results["orbit"] = benchmark_orbit_kernel(iterations=int(args.iterations), warmup=not bool(args.no_warmup))
    if args.kind in {"attitude", "all"}:
        results["attitude"] = benchmark_attitude_kernel(
            iterations=int(args.attitude_iterations),
            warmup=not bool(args.no_warmup),
        )
    if args.kind in {"estimation", "all"}:
        results["estimation"] = benchmark_estimation_kernel(
            iterations=int(args.estimation_iterations),
            warmup=not bool(args.no_warmup),
        )
    if args.json:
        print(json.dumps({key: asdict(value) for key, value in results.items()}, indent=2))
    else:
        if "orbit" in results:
            _print_orbit_result(results["orbit"])
        if "attitude" in results:
            _print_attitude_result(results["attitude"])
        if "estimation" in results:
            _print_estimation_result(results["estimation"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
