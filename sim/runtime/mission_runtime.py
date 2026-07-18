"""Mission-module, strategy, execution, and deployment dispatch."""

from __future__ import annotations

from typing import Any

import numpy as np

from sim.core.models import StateBelief, StateTruth
from sim.runtime.commands import _decision_truth_from_belief
from sim.runtime.compat import _call_with_compat_kwargs
from sim.runtime.models import AgentRuntime
from sim.utils.quaternion import quaternion_to_dcm_bn


def _deploy_from_rocket(agent: AgentRuntime, rocket: AgentRuntime, t_next: float) -> None:
    if (
        agent.kind != "satellite"
        or agent.active
        or agent.deploy_source not in {"rocket_deployment", "rocket_insertion"}
        or rocket.rocket_state is None
    ):
        return
    c_bn = quaternion_to_dcm_bn(rocket.rocket_state.attitude_quat_bn)
    dv_body = np.array(agent.deploy_dv_body_m_s if agent.deploy_dv_body_m_s is not None else np.zeros(3), dtype=float)
    dv_eci_km_s = (c_bn.T @ dv_body) / 1e3
    rs = rocket.rocket_state
    mass_kg = float(agent.truth.mass_kg) if agent.truth is not None else 200.0
    agent.truth = StateTruth(
        position_eci_km=np.array(rs.position_eci_km, dtype=float),
        velocity_eci_km_s=np.array(rs.velocity_eci_km_s, dtype=float) + dv_eci_km_s,
        attitude_quat_bn=np.array(rs.attitude_quat_bn, dtype=float),
        angular_rate_body_rad_s=np.array(rs.angular_rate_body_rad_s, dtype=float),
        mass_kg=mass_kg,
        t_s=t_next,
    )
    if agent.belief is not None and agent.belief.state.size >= 13:
        agent.belief = StateBelief(
            state=np.hstack(
                (
                    agent.truth.position_eci_km,
                    agent.truth.velocity_eci_km_s,
                    agent.truth.attitude_quat_bn,
                    agent.truth.angular_rate_body_rad_s,
                )
            ),
            covariance=np.eye(13) * 1e-4,
            last_update_t_s=t_next,
        )
    else:
        agent.belief = StateBelief(
            state=np.hstack((agent.truth.position_eci_km, agent.truth.velocity_eci_km_s)),
            covariance=np.eye(6) * 1e-4,
            last_update_t_s=t_next,
        )
    agent.control_available_time_s = float(t_next) + float(max(agent.initialization_delay_s, 0.0))
    agent.active = True


def _run_mission_modules(
    *,
    agent: AgentRuntime,
    t_s: float,
    dt_s: float,
    env: dict[str, Any],
    orbit_controller: Any | None = None,
    attitude_controller: Any | None = None,
    orb_belief: StateBelief | None = None,
    att_belief: StateBelief | None = None,
) -> dict[str, Any]:
    if not agent.mission_modules:
        return {}
    truth = _decision_truth_from_belief(agent)
    if truth is None:
        return {}
    own_knowledge = agent.knowledge_base.snapshot() if agent.knowledge_base is not None else {}
    out: dict[str, Any] = {}
    for module in agent.mission_modules:
        if not hasattr(module, "update"):
            continue
        ret = _call_with_compat_kwargs(
            module.update,
            primary_kwargs={
                "object_id": agent.object_id,
                "truth": truth,
                "belief": agent.belief,
                "own_knowledge": own_knowledge,
                "env": env,
                "t_s": t_s,
                "dt_s": dt_s,
                "orbit_controller": orbit_controller,
                "attitude_controller": attitude_controller,
                "orb_belief": orb_belief,
                "att_belief": att_belief,
                "rocket_state": agent.rocket_state,
                "rocket_vehicle_cfg": (agent.rocket_sim.vehicle_cfg if agent.rocket_sim is not None else None),
            },
            fallback_kwargs={"truth": truth, "t_s": t_s},
        )
        if isinstance(ret, dict):
            out.update(ret)
    return out


def _run_mission_strategy(
    *,
    agent: AgentRuntime,
    t_s: float,
    dt_s: float,
    env: dict[str, Any],
    orbit_controller: Any | None = None,
    attitude_controller: Any | None = None,
    orb_belief: StateBelief | None = None,
    att_belief: StateBelief | None = None,
) -> dict[str, Any]:
    strategy = agent.mission_strategy
    if strategy is None:
        return {}
    truth = _decision_truth_from_belief(agent)
    if truth is None:
        return {}
    own_knowledge = agent.knowledge_base.snapshot() if agent.knowledge_base is not None else {}
    for method_name in ("update", "plan", "decide"):
        if not hasattr(strategy, method_name):
            continue
        method = getattr(strategy, method_name)
        ret = _call_with_compat_kwargs(
            method,
            primary_kwargs={
                "object_id": agent.object_id,
                "truth": truth,
                "belief": agent.belief,
                "own_knowledge": own_knowledge,
                "env": env,
                "t_s": t_s,
                "dt_s": dt_s,
                "orbit_controller": orbit_controller,
                "attitude_controller": attitude_controller,
                "orb_belief": orb_belief,
                "att_belief": att_belief,
                "rocket_state": agent.rocket_state,
                "rocket_vehicle_cfg": (agent.rocket_sim.vehicle_cfg if agent.rocket_sim is not None else None),
                "dry_mass_kg": agent.dry_mass_kg,
                "fuel_capacity_kg": agent.fuel_capacity_kg,
            },
            fallback_kwargs={"truth": truth, "t_s": t_s},
        )
        return ret if isinstance(ret, dict) else {}
    return {}


def _run_mission_execution(
    *,
    agent: AgentRuntime,
    intent: dict[str, Any],
    t_s: float,
    dt_s: float,
    env: dict[str, Any],
    orbit_controller: Any | None = None,
    attitude_controller: Any | None = None,
    orb_belief: StateBelief | None = None,
    att_belief: StateBelief | None = None,
) -> dict[str, Any]:
    execution = intent.get("_mission_execution_override", agent.mission_execution)
    if execution is None:
        return {}
    truth = _decision_truth_from_belief(agent)
    if truth is None:
        return {}
    own_knowledge = agent.knowledge_base.snapshot() if agent.knowledge_base is not None else {}
    for method_name in ("update", "execute", "act"):
        if not hasattr(execution, method_name):
            continue
        method = getattr(execution, method_name)
        ret = _call_with_compat_kwargs(
            method,
            primary_kwargs={
                "intent": dict(intent or {}),
                "object_id": agent.object_id,
                "truth": truth,
                "belief": agent.belief,
                "own_knowledge": own_knowledge,
                "env": env,
                "t_s": t_s,
                "dt_s": dt_s,
                "orbit_controller": orbit_controller,
                "attitude_controller": attitude_controller,
                "orb_belief": orb_belief,
                "att_belief": att_belief,
                "rocket_state": agent.rocket_state,
                "rocket_vehicle_cfg": (agent.rocket_sim.vehicle_cfg if agent.rocket_sim is not None else None),
                "dry_mass_kg": agent.dry_mass_kg,
                "fuel_capacity_kg": agent.fuel_capacity_kg,
                "orbital_isp_s": agent.orbital_isp_s,
                "orbit_command_period_s": float(env.get("orbit_command_period_s", dt_s)),
            },
            fallback_kwargs={"intent": dict(intent or {}), "truth": truth, "t_s": t_s},
        )
        return ret if isinstance(ret, dict) else {}
    return {}
