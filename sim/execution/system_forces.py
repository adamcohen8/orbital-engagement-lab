"""Synchronized two-object RK4 path for opt-in system acceleration plugins."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Callable

import numpy as np

from sim.config.plugin_specs import instantiate_plugin_spec, plugin_spec_field
from sim.core.models import StateTruth
from sim.dynamics.orbit.system_force import SystemForceContext, propagate_pair_rk4
from sim.execution.object_workers import ObjectStepInput, ObjectStepResult


class SystemForceStepper:
    """Advance a pair together so each plugin sees both states at every RK stage."""

    def __init__(
        self, *, pointers: list[Any], initial_jd_utc: float, substep_s: float,
        numeric_backend: str = "rust",
    ) -> None:
        self.initial_jd_utc = float(initial_jd_utc)
        self.substep_s = float(substep_s)
        self.numeric_backend = str(numeric_backend)
        self.models: list[tuple[str, Callable[[SystemForceContext], Any]]] = []
        for index, pointer in enumerate(pointers):
            name = f"{plugin_spec_field(pointer, 'module')}.{plugin_spec_field(pointer, 'class_name') or plugin_spec_field(pointer, 'function')}"
            target = instantiate_plugin_spec(pointer, description=f"system force model {index}")
            function = getattr(target, "accelerations", None) if plugin_spec_field(pointer, "class_name") else target
            if not callable(function):
                raise ValueError(f"system force model {name} must provide accelerations(context)")
            self.models.append((name, function))

    def step_objects(self, inputs: list[ObjectStepInput]) -> list[ObjectStepResult]:
        if len(inputs) != 2 or len({item.object_id for item in inputs}) != 2:
            raise RuntimeError("system force stepping requires exactly two active objects")
        if any(
            item.agent.kind != "satellite"
            or item.agent.flight_software_runtime is not None
            or getattr(item.agent.dynamics, "resource_model", None) is not None
            for item in inputs
        ):
            raise RuntimeError("system force stepping requires passive satellites without resources")
        start = perf_counter()
        masses = {item.object_id: float(item.initial_truth.mass_kg) for item in inputs}
        states = {
            item.object_id: np.hstack((item.initial_truth.position_eci_km, item.initial_truth.velocity_eci_km_s))
            for item in inputs
        }
        mu = {item.object_id: float(item.agent.dynamics.mu_km3_s2) for item in inputs}
        t_s = float(inputs[0].t_s)
        t_next = float(inputs[0].t_next)
        if any(float(item.t_s) != t_s or float(item.t_next) != t_next for item in inputs):
            raise RuntimeError("system force objects must share an integration interval")

        states = propagate_pair_rk4(
            states=states,
            masses_kg=masses,
            mu_km3_s2=mu,
            models=self.models,
            initial_jd_utc=self.initial_jd_utc,
            t_s=t_s,
            t_next=t_next,
            substep_s=self.substep_s,
            numeric_backend=self.numeric_backend,
        )
        elapsed = (perf_counter() - start) / len(inputs)
        results = []
        for item in inputs:
            oid = item.object_id
            original: StateTruth = item.initial_truth
            updated = original.copy()
            updated.position_eci_km = states[oid][:3].copy()
            updated.velocity_eci_km_s = states[oid][3:].copy()
            updated.t_s = t_next
            results.append(ObjectStepResult(
                object_id=oid,
                stage="system_force_step",
                elapsed_s=elapsed,
                truth=updated,
                thrust_eci_km_s2=np.zeros(3, dtype=float),
                torque_body_nm=np.zeros(3, dtype=float),
            ))
        return results
