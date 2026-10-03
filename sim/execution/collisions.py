"""Opt-in, frictionless spherical impacts for a passive ONP satellite pair."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from sim.core.models import Command, StateTruth
from sim.dynamics.orbit.epoch import TIME_DEPENDENT_ENV_CACHE_KEY
from sim.execution.object_workers import ObjectStepInput, ObjectStepResult
from sim.numeric_backend import normalize_numeric_backend


class SphericalCollisionStepper:
    """Propagate a passive pair to first contact and resume after an elastic impulse."""

    def __init__(self, engine: Any, radii_m: dict[str, float], numeric_backend: str = "rust") -> None:
        self.engine = engine
        self.radii_m = dict(radii_m)
        self.object_ids = tuple(sorted(radii_m))
        self.radius_km = sum(radii_m.values()) / 1000.0
        self.numeric_backend = normalize_numeric_backend(numeric_backend, error_message="numeric_backend must be python or rust")
        if self.numeric_backend not in {"python", "rust"}:
            raise ValueError("numeric_backend must be python or rust")
        self.last_events: list[dict[str, Any]] = []

    def _propagate(self, states: dict[str, StateTruth], dt_s: float) -> dict[str, StateTruth]:
        if dt_s <= 0.0:
            return {oid: state.copy() for oid, state in states.items()}
        current = {oid: state.copy() for oid, state in states.items()}
        remaining = float(dt_s)
        while remaining > 0.0:
            h = min(float(self.engine.sim_substep_s), remaining)
            if h <= 0.0:
                raise RuntimeError("Collision propagation produced a nonpositive substep.")
            following: dict[str, StateTruth] = {}
            for oid in self.object_ids:
                env = {
                    **self.engine.base_environment,
                    "object_id": oid,
                    "world_truth": current,
                    "attitude_disabled": True,
                    TIME_DEPENDENT_ENV_CACHE_KEY: self.engine._time_dependent_env_cache,
                }
                following[oid] = self.engine.agents[oid].dynamics.step(
                    state=current[oid], command=Command.zero(), env=env, dt_s=h,
                )
            current = following
            remaining -= h
            if remaining < 1.0e-12:
                break
        return current

    def _relative(self, states: dict[str, StateTruth]) -> np.ndarray:
        a, b = self.object_ids
        return states[b].position_eci_km - states[a].position_eci_km

    def _closing_speed(self, states: dict[str, StateTruth]) -> float:
        a, b = self.object_ids
        rel = self._relative(states)
        distance = float(np.linalg.norm(rel))
        if distance == 0.0:
            raise ValueError("Spherical collision has no contact normal at coincident centers.")
        velocity = states[b].velocity_eci_km_s - states[a].velocity_eci_km_s
        return -float(np.dot(velocity, rel / distance))

    def _first_contact_in_segment(
        self,
        states: dict[str, StateTruth],
        dt_s: float,
        *,
        depth: int = 0,
    ) -> tuple[float, dict[str, StateTruth]] | None:
        """Search a swept relative segment, refining curvature near possible contact."""
        start_rel = self._relative(states)
        end_states = self._propagate(states, dt_s)
        end_rel = self._relative(end_states)
        chord = end_rel - start_rel
        if self.numeric_backend == "rust":
            from sim.rust_vehicle_backend import collision_chord_geometry

            fraction, chord_miss, chord_norm2 = collision_chord_geometry(start_rel, end_rel)
        else:
            chord_norm2 = float(np.dot(chord, chord))
            fraction = (
                float(np.clip(-np.dot(start_rel, chord) / chord_norm2, 0.0, 1.0))
                if chord_norm2 > 0.0 else 0.0
            )
            chord_miss = float(np.linalg.norm(start_rel + fraction * chord))
        midpoint_states = self._propagate(states, 0.5 * dt_s)
        deviation = float(np.linalg.norm(self._relative(midpoint_states) - 0.5 * (start_rel + end_rel)))
        if fraction == 0.0 and self._closing_speed(states) <= 0.0 and deviation <= 0.1 * self.radius_km:
            return None
        near_contact = chord_miss <= self.radius_km + 2.0 * deviation
        uncertain = abs(chord_miss - self.radius_km) <= 2.0 * deviation + 1.0e-10
        if depth < 16 and dt_s > 1.0e-7 and (deviation > 0.1 * self.radius_km or uncertain):
            first = self._first_contact_in_segment(states, 0.5 * dt_s, depth=depth + 1)
            if first is not None:
                return first
            second = self._first_contact_in_segment(midpoint_states, 0.5 * dt_s, depth=depth + 1)
            if second is not None:
                return 0.5 * dt_s + second[0], second[1]
            return None
        if not near_contact:
            return None

        candidates = sorted({fraction, 0.5, 1.0})
        for candidate in candidates:
            if candidate <= 0.0:
                continue
            trial = self._propagate(states, candidate * dt_s)
            if float(np.linalg.norm(self._relative(trial))) > self.radius_km:
                continue
            lo, hi = 0.0, candidate * dt_s
            for _ in range(48):
                if hi - lo <= max(1.0e-10, dt_s * 1.0e-10):
                    break
                mid = 0.5 * (lo + hi)
                at_mid = self._propagate(states, mid)
                if float(np.linalg.norm(self._relative(at_mid))) <= self.radius_km:
                    hi = mid
                else:
                    lo = mid
            contact = self._propagate(states, hi)
            if self._closing_speed(contact) > 0.0:
                return hi, contact
        return None

    def _impact(self, states: dict[str, StateTruth], time_s: float) -> dict[str, Any]:
        a, b = self.object_ids
        first, second = states[a], states[b]
        normal = self._relative(states)
        normal /= np.linalg.norm(normal)
        relative_velocity = second.velocity_eci_km_s - first.velocity_eci_km_s
        pre_a = first.velocity_eci_km_s.copy()
        pre_b = second.velocity_eci_km_s.copy()
        if self.numeric_backend == "rust":
            from sim.rust_vehicle_backend import collision_elastic_impact

            post_a, post_b, native_normal, closing = collision_elastic_impact(
                position_a=first.position_eci_km,
                position_b=second.position_eci_km,
                velocity_a=pre_a,
                velocity_b=pre_b,
                mass_a_kg=float(first.mass_kg),
                mass_b_kg=float(second.mass_kg),
                restitution=1.0,
            )
            normal = native_normal
            first.velocity_eci_km_s = post_a
            second.velocity_eci_km_s = post_b
        else:
            closing = -float(np.dot(relative_velocity, normal))
            if closing <= 0.0:
                raise RuntimeError("Collision impulse requires approaching objects.")
            impulse = 2.0 * closing / (1.0 / first.mass_kg + 1.0 / second.mass_kg)
            first.velocity_eci_km_s = pre_a - impulse * normal / first.mass_kg
            second.velocity_eci_km_s = pre_b + impulse * normal / second.mass_kg
        return {
            "time_s": float(time_s),
            "object_ids": [a, b],
            "radii_m": [self.radii_m[a], self.radii_m[b]],
            "position_eci_km": [first.position_eci_km.tolist(), second.position_eci_km.tolist()],
            "normal_eci": normal.tolist(),
            "closing_speed_m_s": closing * 1000.0,
            "pre_velocity_eci_km_s": [pre_a.tolist(), pre_b.tolist()],
            "post_velocity_eci_km_s": [first.velocity_eci_km_s.tolist(), second.velocity_eci_km_s.tolist()],
            "restitution": 1.0,
        }

    def step_objects(self, inputs: list[ObjectStepInput]) -> list[ObjectStepResult]:
        if set(item.object_id for item in inputs) != set(self.object_ids) or len(inputs) != 2:
            raise RuntimeError("Spherical collisions require the configured two active satellites.")
        if any(
            item.agent.kind != "satellite"
            or item.agent.runtime_profile != "trajectory_only"
            or item.agent.flight_software_runtime is not None
            or getattr(item.agent.dynamics, "resource_model", None) is not None
            for item in inputs
        ):
            raise RuntimeError("Spherical collisions require passive satellites without spacecraft resources.")
        started = perf_counter()
        t_s = float(inputs[0].t_s)
        t_next = float(inputs[0].t_next)
        if any(item.t_s != t_s or item.t_next != t_next for item in inputs):
            raise RuntimeError("Collision objects must share one time interval.")
        states = {item.object_id: item.initial_truth.copy() for item in inputs}
        self.last_events = []
        if float(np.linalg.norm(self._relative(states))) < self.radius_km - 1.0e-10:
            raise ValueError("Spherical collision objects begin overlapped.")
        current_s = t_s
        while current_s < t_next - 1.0e-12:
            h = min(float(self.engine.sim_substep_s), t_next - current_s)
            contact = self._first_contact_in_segment(states, h)
            if contact is None:
                states = self._propagate(states, h)
                current_s += h
                continue
            tau, states = contact
            current_s += tau
            self.last_events.append(self._impact(states, current_s))
            if len(self.last_events) > 8:
                raise RuntimeError("More than eight collisions occurred in one simulation step; reduce dt_s.")
            if tau <= 1.0e-12 and current_s < t_next:
                # A separating pair at exact contact should advance before it is tested again.
                epsilon_s = min(1.0e-9, t_next - current_s)
                states = self._propagate(states, epsilon_s)
                current_s += epsilon_s
        elapsed = (perf_counter() - started) / 2.0
        return [
            ObjectStepResult(
                object_id=item.object_id,
                stage="spherical_collision_step",
                elapsed_s=elapsed,
                truth=states[item.object_id],
                thrust_eci_km_s2=np.zeros(3, dtype=float),
                torque_body_nm=np.zeros(3, dtype=float),
            )
            for item in inputs
        ]
