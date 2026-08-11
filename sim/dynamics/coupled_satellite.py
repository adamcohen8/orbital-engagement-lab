"""Event-aligned coupled orbit, attitude, actuator, and mass propagation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from sim.utils.quaternion import (
    normalize_quaternion,
    quaternion_delta_from_body_rate,
    quaternion_multiply,
    quaternion_to_dcm_bn,
)

Array = np.ndarray


@dataclass(frozen=True, slots=True)
class CoupledIntegratorConfig:
    orbit_max_substep_s: float
    attitude_max_substep_s: float

    def __post_init__(self) -> None:
        orbit = float(self.orbit_max_substep_s)
        attitude = float(self.attitude_max_substep_s)
        if not np.isfinite(orbit) or orbit <= 0.0:
            raise ValueError("orbit_max_substep_s must be finite and positive")
        if not np.isfinite(attitude) or attitude <= 0.0:
            raise ValueError("attitude_max_substep_s must be finite and positive")
        if attitude > orbit:
            raise ValueError("attitude_max_substep_s must not exceed orbit_max_substep_s")


@dataclass(frozen=True, slots=True)
class CoupledSatelliteState:
    position_eci_km: Array
    velocity_eci_km_s: Array
    attitude_quat_bn: Array
    angular_rate_body_rad_s: Array
    mass_kg: float
    actuator_state: Array
    t_s: float

    def __post_init__(self) -> None:
        for name, value, size in (
            ("position_eci_km", self.position_eci_km, 3),
            ("velocity_eci_km_s", self.velocity_eci_km_s, 3),
            ("attitude_quat_bn", self.attitude_quat_bn, 4),
            ("angular_rate_body_rad_s", self.angular_rate_body_rad_s, 3),
        ):
            array = np.asarray(value, dtype=float).reshape(-1)
            if array.size != size or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must contain {size} finite values")
            object.__setattr__(self, name, array.copy())
        actuator = np.asarray(self.actuator_state, dtype=float).reshape(-1)
        if not np.all(np.isfinite(actuator)):
            raise ValueError("actuator_state must contain finite values")
        object.__setattr__(self, "actuator_state", actuator.copy())
        quaternion = normalize_quaternion(self.attitude_quat_bn)
        if abs(float(np.linalg.norm(self.attitude_quat_bn)) - 1.0) > 1.0e-10:
            raise ValueError("attitude_quat_bn must be normalized within 1e-10")
        object.__setattr__(self, "attitude_quat_bn", quaternion)
        if not np.isfinite(float(self.mass_kg)) or self.mass_kg <= 0.0:
            raise ValueError("mass_kg must be finite and positive")
        if not np.isfinite(float(self.t_s)):
            raise ValueError("t_s must be finite")


@dataclass(frozen=True, slots=True)
class CoupledDerivative:
    position_rate_km_s: Array
    velocity_rate_km_s2: Array
    angular_acceleration_body_rad_s2: Array
    mass_rate_kg_s: float
    actuator_state_rate: Array

    def as_vector(self, actuator_size: int) -> Array:
        vectors = (
            np.asarray(self.position_rate_km_s, dtype=float).reshape(-1),
            np.asarray(self.velocity_rate_km_s2, dtype=float).reshape(-1),
            np.asarray(self.angular_acceleration_body_rad_s2, dtype=float).reshape(-1),
            np.asarray(self.actuator_state_rate, dtype=float).reshape(-1),
        )
        if vectors[0].size != 3 or vectors[1].size != 3 or vectors[2].size != 3:
            raise ValueError("coupled derivative translational and angular vectors must have length 3")
        if vectors[3].size != actuator_size:
            raise ValueError("actuator derivative size must match actuator state")
        mass_rate = float(self.mass_rate_kg_s)
        if not np.isfinite(mass_rate) or not all(np.all(np.isfinite(vector)) for vector in vectors):
            raise ValueError("coupled derivative must contain only finite values")
        return np.concatenate((vectors[0], vectors[1], vectors[2], np.array([mass_rate]), vectors[3]))


@dataclass(frozen=True, slots=True)
class MassProperties:
    inertia_body_kg_m2: Array
    inertia_rate_body_kg_m2_s: Array | None = None

    def __post_init__(self) -> None:
        inertia = np.asarray(self.inertia_body_kg_m2, dtype=float).reshape(3, 3)
        rate = (
            np.zeros((3, 3), dtype=float)
            if self.inertia_rate_body_kg_m2_s is None
            else np.asarray(self.inertia_rate_body_kg_m2_s, dtype=float).reshape(3, 3)
        )
        if not np.all(np.isfinite(inertia)) or not np.allclose(inertia, inertia.T, rtol=1e-12, atol=1e-12):
            raise ValueError("inertia must be finite and symmetric")
        if np.min(np.linalg.eigvalsh(inertia)) <= 0.0:
            raise ValueError("inertia must be positive definite")
        if not np.all(np.isfinite(rate)):
            raise ValueError("inertia rate must be finite")
        object.__setattr__(self, "inertia_body_kg_m2", inertia.copy())
        object.__setattr__(self, "inertia_rate_body_kg_m2_s", rate.copy())


@dataclass(frozen=True, slots=True)
class StageEffects:
    force_eci_n: Array | None = None
    force_body_n: Array | None = None
    torque_body_n_m: Array | None = None
    mass_flow_kg_s: float = 0.0
    actuator_state_rate: Array | None = None

    def resolved(self, actuator_size: int) -> tuple[Array, Array, Array, float, Array]:
        force_eci = _vector_or_zero(self.force_eci_n, 3, "force_eci_n")
        force_body = _vector_or_zero(self.force_body_n, 3, "force_body_n")
        torque = _vector_or_zero(self.torque_body_n_m, 3, "torque_body_n_m")
        actuator_rate = _vector_or_zero(self.actuator_state_rate, actuator_size, "actuator_state_rate")
        mass_flow = float(self.mass_flow_kg_s)
        if not np.isfinite(mass_flow) or mass_flow < 0.0:
            raise ValueError("mass_flow_kg_s must be finite and nonnegative")
        return force_eci, force_body, torque, mass_flow, actuator_rate


class CoupledDerivativeModel(Protocol):
    def __call__(self, t_s: float, state: CoupledSatelliteState, control: object) -> CoupledDerivative: ...


EffectsModel = Callable[[float, CoupledSatelliteState, object], StageEffects]
GravityModel = Callable[[float, CoupledSatelliteState], Array]
MassPropertiesModel = Callable[[CoupledSatelliteState, float], MassProperties]


class CoupledSatelliteDynamics:
    """Reference derivative owner evaluated independently at every RK stage."""

    def __init__(
        self,
        *,
        effects_model: EffectsModel,
        mass_properties_model: MassPropertiesModel,
        gravity_model: GravityModel | None = None,
    ) -> None:
        self.effects_model = effects_model
        self.mass_properties_model = mass_properties_model
        self.gravity_model = gravity_model or (lambda _t, _state: np.zeros(3, dtype=float))

    def derivative(self, t_s: float, state: CoupledSatelliteState, control: object) -> CoupledDerivative:
        effects = self.effects_model(t_s, state, control)
        force_eci, force_body, torque, mass_flow, actuator_rate = effects.resolved(state.actuator_state.size)
        c_bn = quaternion_to_dcm_bn(state.attitude_quat_bn)
        total_force_eci_n = force_eci + c_bn.T @ force_body
        gravity_km_s2 = np.asarray(self.gravity_model(t_s, state), dtype=float).reshape(3)
        if not np.all(np.isfinite(gravity_km_s2)):
            raise ValueError("gravity_model returned non-finite acceleration")
        acceleration_km_s2 = gravity_km_s2 + total_force_eci_n / state.mass_kg / 1.0e3
        properties = self.mass_properties_model(state, -mass_flow)
        inertia = properties.inertia_body_kg_m2
        inertia_rate = properties.inertia_rate_body_kg_m2_s
        omega = state.angular_rate_body_rad_s
        angular_acceleration = np.linalg.solve(
            inertia,
            torque - np.cross(omega, inertia @ omega) - inertia_rate @ omega,
        )
        return CoupledDerivative(
            position_rate_km_s=state.velocity_eci_km_s,
            velocity_rate_km_s2=acceleration_km_s2,
            angular_acceleration_body_rad_s2=angular_acceleration,
            mass_rate_kg_s=-mass_flow,
            actuator_state_rate=actuator_rate,
        )


@dataclass(frozen=True, slots=True)
class CoupledStepRecord:
    start_time_s: float
    end_time_s: float


@dataclass(frozen=True, slots=True)
class CoupledPropagationResult:
    final_state: CoupledSatelliteState
    steps: tuple[CoupledStepRecord, ...]
    boundary_times_s: tuple[float, ...]
    output_samples: tuple[CoupledSatelliteState, ...]


class CoupledSatelliteIntegrator:
    def __init__(self, config: CoupledIntegratorConfig, derivative_model: CoupledDerivativeModel) -> None:
        self.config = config
        self.derivative_model = derivative_model

    def propagate(
        self,
        state: CoupledSatelliteState,
        *,
        end_time_s: float,
        control: object = None,
        hard_event_times_s: tuple[float, ...] = (),
        output_times_s: tuple[float, ...] = (),
    ) -> CoupledPropagationResult:
        end = float(end_time_s)
        if not np.isfinite(end) or end < state.t_s:
            raise ValueError("end_time_s must be finite and not precede state time")
        for name, values in (("hard_event_times_s", hard_event_times_s), ("output_times_s", output_times_s)):
            if any(not np.isfinite(float(value)) or value < state.t_s or value > end for value in values):
                raise ValueError(f"{name} values must lie inside the propagation interval")
        hard_events = tuple(sorted(set(float(value) for value in hard_event_times_s)))
        outputs = tuple(sorted(set(float(value) for value in output_times_s)))
        boundaries = tuple(sorted(set((*hard_events, *outputs, end))))
        output_set = set(outputs)
        samples: list[CoupledSatelliteState] = []
        records: list[CoupledStepRecord] = []
        boundary_hits: list[float] = []
        current = state
        maximum = min(self.config.orbit_max_substep_s, self.config.attitude_max_substep_s)
        for boundary in boundaries:
            while current.t_s < boundary - 1.0e-14:
                h = min(maximum, boundary - current.t_s)
                start = current.t_s
                current = self._microstep(current, h, control)
                records.append(CoupledStepRecord(start, current.t_s))
            if current.t_s != boundary and abs(current.t_s - boundary) <= 1.0e-13:
                current = _retime_state(current, boundary)
                if records and abs(records[-1].end_time_s - boundary) <= 1.0e-13:
                    records[-1] = CoupledStepRecord(records[-1].start_time_s, boundary)
            if boundary > state.t_s or boundary == end:
                boundary_hits.append(boundary)
            if boundary in output_set:
                samples.append(current)
        return CoupledPropagationResult(current, tuple(records), tuple(boundary_hits), tuple(samples))

    def _microstep(self, state: CoupledSatelliteState, h: float, control: object) -> CoupledSatelliteState:
        z0 = _pack_state(state)
        q0 = state.attitude_quat_bn
        actuator_size = state.actuator_state.size

        state1 = state
        d1 = self.derivative_model(state.t_s, state1, control).as_vector(actuator_size)
        w1 = state1.angular_rate_body_rad_s

        z2 = z0 + 0.5 * h * d1
        q2 = _advance_quaternion(q0, w1, 0.5 * h)
        state2 = _unpack_state(z2, q2, state.t_s + 0.5 * h, actuator_size)
        d2 = self.derivative_model(state2.t_s, state2, control).as_vector(actuator_size)
        w2 = state2.angular_rate_body_rad_s

        z3 = z0 + 0.5 * h * d2
        q3 = _advance_quaternion(q0, w2, 0.5 * h)
        state3 = _unpack_state(z3, q3, state.t_s + 0.5 * h, actuator_size)
        d3 = self.derivative_model(state3.t_s, state3, control).as_vector(actuator_size)
        w3 = state3.angular_rate_body_rad_s

        z4 = z0 + h * d3
        q4 = _advance_quaternion(q0, w3, h)
        state4 = _unpack_state(z4, q4, state.t_s + h, actuator_size)
        d4 = self.derivative_model(state4.t_s, state4, control).as_vector(actuator_size)
        w4 = state4.angular_rate_body_rad_s

        z_next = z0 + (h / 6.0) * (d1 + 2.0 * d2 + 2.0 * d3 + d4)
        rotation_vector = (h / 6.0) * (w1 + 2.0 * w2 + 2.0 * w3 + w4)
        q_next = _advance_quaternion(q0, rotation_vector, 1.0)
        return _unpack_state(z_next, q_next, state.t_s + h, actuator_size)


def two_body_gravity(mu_km3_s2: float) -> GravityModel:
    mu = float(mu_km3_s2)
    if not np.isfinite(mu) or mu < 0.0:
        raise ValueError("mu_km3_s2 must be finite and nonnegative")

    def gravity(_t_s: float, state: CoupledSatelliteState) -> Array:
        radius = state.position_eci_km
        norm = float(np.linalg.norm(radius))
        if norm <= 0.0:
            raise ValueError("two-body gravity is undefined at zero radius")
        return -mu * radius / norm**3

    return gravity


def constant_mass_properties(inertia_body_kg_m2: Array) -> MassPropertiesModel:
    properties = MassProperties(inertia_body_kg_m2)
    return lambda _state, _mass_rate: properties


def _pack_state(state: CoupledSatelliteState) -> Array:
    return np.concatenate(
        (
            state.position_eci_km,
            state.velocity_eci_km_s,
            state.angular_rate_body_rad_s,
            np.array([state.mass_kg]),
            state.actuator_state,
        )
    )


def _unpack_state(z: Array, quaternion: Array, t_s: float, actuator_size: int) -> CoupledSatelliteState:
    values = np.asarray(z, dtype=float).reshape(10 + actuator_size)
    return CoupledSatelliteState(
        position_eci_km=values[0:3],
        velocity_eci_km_s=values[3:6],
        attitude_quat_bn=quaternion,
        angular_rate_body_rad_s=values[6:9],
        mass_kg=float(values[9]),
        actuator_state=values[10:],
        t_s=float(t_s),
    )


def _advance_quaternion(quaternion: Array, body_rotation_rate: Array, duration_s: float) -> Array:
    increment = quaternion_delta_from_body_rate(np.asarray(body_rotation_rate, dtype=float), float(duration_s))
    return normalize_quaternion(quaternion_multiply(quaternion, increment))


def _retime_state(state: CoupledSatelliteState, t_s: float) -> CoupledSatelliteState:
    return CoupledSatelliteState(
        position_eci_km=state.position_eci_km,
        velocity_eci_km_s=state.velocity_eci_km_s,
        attitude_quat_bn=state.attitude_quat_bn,
        angular_rate_body_rad_s=state.angular_rate_body_rad_s,
        mass_kg=state.mass_kg,
        actuator_state=state.actuator_state,
        t_s=t_s,
    )


def _vector_or_zero(value: Array | None, size: int, name: str) -> Array:
    if value is None:
        return np.zeros(size, dtype=float)
    vector = np.asarray(value, dtype=float).reshape(-1)
    if vector.size != size or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain {size} finite values")
    return vector
