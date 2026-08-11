"""Runtime records shared by satellite, rocket, knowledge, and mission factories."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sim.core.models import StateBelief, StateTruth
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
    flight_software_runtime: Any | None = None
    runtime_profile: str = "flight_software"
