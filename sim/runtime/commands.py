"""Runtime command conversion and state-view helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from sim.core.models import Command, StateBelief, StateTruth
from sim.rocket import RocketState
from sim.runtime.models import AgentRuntime
from sim.utils.frames import eci_relative_to_ric_rect, ric_rect_to_curv


def _to_jsonable_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable_value(v) for v in value]
    return value


def _command_to_dict(cmd: Command) -> dict[str, Any]:
    return {
        "thrust_eci_km_s2": np.array(cmd.thrust_eci_km_s2, dtype=float).tolist(),
        "torque_body_nm": np.array(cmd.torque_body_nm, dtype=float).tolist(),
        "mode_flags": _to_jsonable_value(dict(cmd.mode_flags or {})),
    }


def _deep_set(root: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cur: Any = root
    for i, tok in enumerate(parts):
        last = i == len(parts) - 1
        if "[" in tok and tok.endswith("]"):
            key, idx_txt = tok[:-1].split("[", 1)
            idx = int(idx_txt)
            if key:
                cur = cur[key]
            if not isinstance(cur, list):
                raise TypeError(f"'{tok}' is not a list segment in path '{path}'.")
            if last:
                cur[idx] = value
                return
            cur = cur[idx]
            continue
        if last:
            cur[tok] = value
            return
        cur = cur[tok]


def _sample_variation(v: Any, rng: np.random.Generator) -> Any:
    mode = v.mode.lower()
    if mode == "choice":
        if not v.options:
            raise ValueError(f"Variation '{v.parameter_path}' with mode=choice requires options.")
        return v.options[int(rng.integers(0, len(v.options)))]
    if mode == "uniform":
        if v.low is None or v.high is None:
            raise ValueError(f"Variation '{v.parameter_path}' with mode=uniform requires low/high.")
        return float(rng.uniform(v.low, v.high))
    if mode == "normal":
        if v.mean is None or v.std is None:
            raise ValueError(f"Variation '{v.parameter_path}' with mode=normal requires mean/std.")
        return float(rng.normal(v.mean, v.std))
    raise ValueError(f"Unsupported variation mode '{v.mode}'.")


def _combine_commands(orb: Command, att: Command) -> Command:
    return Command(
        thrust_eci_km_s2=np.array(orb.thrust_eci_km_s2, dtype=float),
        torque_body_nm=np.array(att.torque_body_nm, dtype=float),
        mode_flags={**dict(orb.mode_flags or {}), **dict(att.mode_flags or {})},
    )


def _rocket_state_to_truth(s: RocketState) -> StateTruth:
    return StateTruth(
        position_eci_km=np.array(s.position_eci_km, dtype=float),
        velocity_eci_km_s=np.array(s.velocity_eci_km_s, dtype=float),
        attitude_quat_bn=np.array(s.attitude_quat_bn, dtype=float),
        angular_rate_body_rad_s=np.array(s.angular_rate_body_rad_s, dtype=float),
        mass_kg=float(s.mass_kg),
        t_s=float(s.t_s),
    )


def _truth_state6(truth: StateTruth, out: np.ndarray | None = None) -> np.ndarray:
    state = np.empty(6, dtype=float) if out is None else out
    state[0:3] = truth.position_eci_km
    state[3:6] = truth.velocity_eci_km_s
    return state


def _decision_truth_from_belief(agent: AgentRuntime) -> StateTruth | None:
    belief = agent.belief
    if belief is None or belief.state.size < 6:
        return None
    state = np.array(belief.state, dtype=float).reshape(-1)
    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    w = np.zeros(3, dtype=float)
    resource_truth = getattr(agent, "truth", None)
    rocket_state = getattr(agent, "rocket_state", None)
    if resource_truth is not None:
        mass_kg = float(resource_truth.mass_kg)
    elif rocket_state is not None:
        mass_kg = float(rocket_state.mass_kg)
    else:
        mass_kg = 0.0
    if state.size >= 13:
        q = np.array(state[6:10], dtype=float)
        w = np.array(state[10:13], dtype=float)
    return StateTruth(
        position_eci_km=np.array(state[:3], dtype=float),
        velocity_eci_km_s=np.array(state[3:6], dtype=float),
        attitude_quat_bn=q,
        angular_rate_body_rad_s=w,
        mass_kg=mass_kg,
        t_s=float(belief.last_update_t_s),
    )


def _truth_from_state6(state6: np.ndarray, *, t_s: float, fallback_truth: StateTruth | None = None) -> StateTruth:
    state = np.array(state6, dtype=float).reshape(-1)
    if state.size < 6:
        raise ValueError("state6 must contain at least 6 elements.")
    return StateTruth(
        position_eci_km=np.array(state[:3], dtype=float),
        velocity_eci_km_s=np.array(state[3:6], dtype=float),
        attitude_quat_bn=(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
            if fallback_truth is None
            else np.array(fallback_truth.attitude_quat_bn, dtype=float)
        ),
        angular_rate_body_rad_s=(
            np.zeros(3, dtype=float)
            if fallback_truth is None
            else np.array(fallback_truth.angular_rate_body_rad_s, dtype=float)
        ),
        mass_kg=0.0 if fallback_truth is None else float(fallback_truth.mass_kg),
        t_s=float(t_s),
    )


def _attitude_state13_from_belief(
    belief: StateBelief,
    truth: StateTruth,
    out: np.ndarray | None = None,
) -> np.ndarray:
    state = np.empty(13, dtype=float) if out is None else out
    state[0:6] = belief.state[:6]
    if belief.state.size >= 13:
        state[6:10] = belief.state[6:10]
        state[10:13] = belief.state[10:13]
    else:
        state[6:10] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        state[10:13] = np.zeros(3, dtype=float)
    return state


def _relative_orbit_state12(
    chief_truth: StateTruth,
    deputy_truth: StateTruth,
    out: np.ndarray | None = None,
    deputy_state6: np.ndarray | None = None,
    chief_state6: np.ndarray | None = None,
) -> np.ndarray:
    state = np.empty(12, dtype=float) if out is None else out
    r_c = chief_truth.position_eci_km
    v_c = chief_truth.velocity_eci_km_s
    x_dep_eci = np.empty(6, dtype=float) if deputy_state6 is None else deputy_state6
    x_chief_eci = np.empty(6, dtype=float) if chief_state6 is None else chief_state6
    x_dep_eci[0:3] = deputy_truth.position_eci_km
    x_dep_eci[3:6] = deputy_truth.velocity_eci_km_s
    x_chief_eci[0:3] = r_c
    x_chief_eci[3:6] = v_c
    x_rect = eci_relative_to_ric_rect(x_dep_eci=x_dep_eci, x_chief_eci=x_chief_eci)
    state[0:6] = ric_rect_to_curv(x_rect, r0_km=float(np.linalg.norm(r_c)))
    state[6:9] = r_c
    state[9:12] = v_c
    return state
