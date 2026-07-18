"""Runtime records shared by satellite, rocket, knowledge, and mission factories."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sim.core.models import Command, StateBelief, StateTruth
from sim.dynamics.model import OrbitalAttitudeDynamics
from sim.knowledge.object_tracking import ObjectKnowledgeBase
from sim.rocket import RocketAscentSimulator, RocketState


@dataclass
class AgentRuntime:
    object_id: str
    kind: str
    enabled: bool
    active: bool
    truth: StateTruth | None
    belief: StateBelief | None
    sensor: Any | None
    estimator: Any | None
    orbit_controller: Any | None
    attitude_controller: Any | None
    dynamics: OrbitalAttitudeDynamics | None
    knowledge_base: ObjectKnowledgeBase | None
    bridge: Any | None
    mission_strategy: Any | None
    mission_execution: Any | None
    rocket_sim: RocketAscentSimulator | None
    rocket_state: RocketState | None
    rocket_guidance: Any | None
    deploy_source: str | None
    deploy_time_s: float | None
    deploy_dv_body_m_s: np.ndarray | None
    initialization_delay_s: float
    control_available_time_s: float | None
    mission_modules: list[Any]
    waiting_for_launch: bool
    orbital_isp_s: float | None = None
    dry_mass_kg: float | None = None
    fuel_capacity_kg: float | None = None
    orbital_max_thrust_n: float | None = None
    thruster_direction_body: np.ndarray | None = None
    thruster_position_body_m: np.ndarray | None = None
    actuator: Any | None = None
    actuator_limits: dict[str, Any] = field(default_factory=dict)
    use_actuator_stack: bool = False
    mass_properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class _RateLimitedController:
    base: Any
    period_s: float
    _last_eval_t_s: float | None = None
    _last_cmd: Command = field(default_factory=Command.zero, init=False)

    def __post_init__(self) -> None:
        self.period_s = float(max(self.period_s, 1e-9))

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        if self._last_eval_t_s is None or float(t_s) - float(self._last_eval_t_s) >= self.period_s - 1e-12:
            if hasattr(self.base, "set_actuation_interval"):
                self.base.set_actuation_interval(float(t_s), float(t_s + self.period_s))
            self._last_cmd = self.base.act(belief, t_s, budget_ms)
            self._last_eval_t_s = float(t_s)
        return self._last_cmd

    def __getstate__(self) -> dict[str, Any]:
        return dict(self.__dict__)

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(dict(state))

    def __getattr__(self, item: str) -> Any:
        base = self.__dict__.get("base")
        if base is None:
            raise AttributeError(item)
        return getattr(base, item)
