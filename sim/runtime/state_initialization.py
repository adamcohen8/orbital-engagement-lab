"""Orbital and relative-state initialization for runtime objects."""

from __future__ import annotations

from typing import Any

import numpy as np

from sim.core.models import StateTruth
from sim.dynamics.orbit.cr3bp import (
    cr3bp_halo_seed_state_km_s,
    cr3bp_moon_state_km_s,
    cr3bp_system,
    propagate_cr3bp_state,
)
from sim.dynamics.orbit.elements import coe_to_rv_eci as _coe_to_rv_eci
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2
from sim.dynamics.orbit.tle import ogp_mean_elements_from_mapping, tle_block_to_rv_eci, tle_to_rv_eci_ogp
from sim.runtime.models import AgentRuntime
from sim.utils.frames import ric_curv_to_rect, ric_rect_state_to_eci

MAX_CR3BP_HALO_PHASE_SUBSTEPS = 100_000


def _rv_from_initial_state(s0: dict[str, Any], *, target_jd_utc: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    if not s0 or bool(s0.get("default_circular_earth", False)):
        pos = np.array([7000.0, 0.0, 0.0], dtype=float)
        spd = float(np.sqrt(EARTH_MU_KM3_S2 / np.linalg.norm(pos)))
        return pos, np.array([0.0, spd, 0.0], dtype=float)

    if any(key in s0 for key in ("relative_to_target_ric", "relative_ric_rect", "source", "launch_lat_deg")):
        # These recognized state forms are resolved by their dedicated runtime
        # initializers after the object graph or launch/deployment state exists.
        pos = np.array([7000.0, 0.0, 0.0], dtype=float)
        spd = float(np.sqrt(EARTH_MU_KM3_S2 / np.linalg.norm(pos)))
        return pos, np.array([0.0, spd, 0.0], dtype=float)

    cr3bp_state = s0.get("cr3bp_rotating")
    if isinstance(cr3bp_state, dict):
        raw_state = cr3bp_state.get("state_km_s", cr3bp_state.get("state"))
        if raw_state is None:
            raise ValueError("initial_state.cr3bp_rotating.state_km_s must be a length-6 list.")
        state = np.array(raw_state, dtype=float).reshape(-1)
        if state.size != 6:
            raise ValueError("initial_state.cr3bp_rotating.state_km_s must be length-6.")
        return state[:3], state[3:]

    halo = s0.get("cr3bp_halo")
    if isinstance(halo, dict):
        system = cr3bp_system(str(halo.get("system", "earth_moon") or "earth_moon"))
        family = str(halo.get("family", "l1_northern") or "l1_northern")
        state = cr3bp_halo_seed_state_km_s(system=system, family=family)
        phase_time_s = float(halo.get("phase_time_s", 0.0) or 0.0)
        if not np.isfinite(phase_time_s) or phase_time_s < 0.0:
            raise ValueError("initial_state.cr3bp_halo.phase_time_s must be a nonnegative finite number.")
        if phase_time_s > 0.0:
            remaining_s = phase_time_s
            current_t_s = 0.0
            substep_s = float(halo.get("phase_substep_s", 120.0) or 120.0)
            if not np.isfinite(substep_s) or substep_s <= 0.0:
                raise ValueError("initial_state.cr3bp_halo.phase_substep_s must be a positive finite number.")
            phase_substeps = int(np.ceil(phase_time_s / substep_s))
            if phase_substeps > MAX_CR3BP_HALO_PHASE_SUBSTEPS:
                raise ValueError(
                    "initial_state.cr3bp_halo phasing requires "
                    f"{phase_substeps} substeps, exceeding the limit of "
                    f"{MAX_CR3BP_HALO_PHASE_SUBSTEPS}; increase phase_substep_s or reduce phase_time_s."
                )
            while remaining_s > 1.0e-9:
                dt_s = min(substep_s, remaining_s)
                state = propagate_cr3bp_state(state, dt_s, current_t_s, system=system)
                current_t_s += dt_s
                remaining_s -= dt_s
        return state[:3], state[3:]

    if "position_eci_km" in s0:
        pos = np.array(s0["position_eci_km"], dtype=float).reshape(3)
        if "velocity_eci_km_s" not in s0:
            raise ValueError("initial_state.position_eci_km requires initial_state.velocity_eci_km_s.")
        vel = np.array(s0["velocity_eci_km_s"], dtype=float).reshape(3)
        if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(vel))):
            raise ValueError("Cartesian initial-state position and velocity entries must be finite.")
        return pos, vel

    tle = s0.get("tle")
    if isinstance(tle, dict):
        return tle_block_to_rv_eci(tle, target_jd_utc=target_jd_utc)

    mean_elements = s0.get("ogp_mean_elements")
    if isinstance(mean_elements, dict):
        return tle_to_rv_eci_ogp(
            ogp_mean_elements_from_mapping(mean_elements),
            target_jd_utc=target_jd_utc,
        )

    coes = s0.get("coes")
    if isinstance(coes, dict):
        d = dict(coes)
        return _coe_to_rv_eci(
            a_km=float(d.get("a_km", d.get("semi_major_axis_km", 7000.0))),
            ecc=float(d.get("ecc", d.get("e", 0.0))),
            inc_deg=float(d.get("inc_deg", d.get("inclination_deg", 0.0))),
            raan_deg=float(d.get("raan_deg", 0.0)),
            argp_deg=float(d.get("argp_deg", d.get("arg_periapsis_deg", 0.0))),
            true_anomaly_deg=float(d.get("ta_deg", d.get("true_anomaly_deg", 0.0))),
        )

    raise ValueError(
        "initial_state does not contain a supported orbital-state form. "
        "Use Cartesian position/velocity, coes, tle, ogp_mean_elements, CR3BP, a relative state, "
        "or explicit default_circular_earth: true."
    )


def _default_truth_from_agent(agent_cfg: Any, t_s: float = 0.0, target_jd_utc: float | None = None) -> StateTruth:
    s0 = dict(agent_cfg.initial_state or {})
    specs = dict(agent_cfg.specs or {})
    if ("dry_mass_kg" in specs) or ("fuel_mass_kg" in specs):
        dry_mass_kg = float(specs.get("dry_mass_kg", 0.0))
        fuel_mass_kg = float(specs.get("fuel_mass_kg", 0.0))
        if dry_mass_kg < 0.0 or fuel_mass_kg < 0.0:
            raise ValueError("dry_mass_kg and fuel_mass_kg must be non-negative.")
        mass_kg = dry_mass_kg + fuel_mass_kg
    else:
        mass_kg = float(specs.get("mass_kg", 300.0))
    if not np.isfinite(mass_kg) or mass_kg <= 0.0:
        raise ValueError("Object mass must be a positive finite value.")
    pos, vel = _rv_from_initial_state(s0, target_jd_utc=target_jd_utc)
    return StateTruth(
        position_eci_km=pos,
        velocity_eci_km_s=vel,
        attitude_quat_bn=np.array(s0.get("attitude_quat_bn", [1.0, 0.0, 0.0, 0.0]), dtype=float),
        angular_rate_body_rad_s=np.array(s0.get("angular_rate_body_rad_s", [0.0, 0.0, 0.0]), dtype=float),
        mass_kg=mass_kg,
        t_s=t_s,
    )


def _resolve_chaser_relative_ric_init(initial_state: dict[str, Any]) -> tuple[np.ndarray, str, str] | None:
    s0 = dict(initial_state or {})
    rel_block = s0.get("relative_to_target_ric")
    if isinstance(rel_block, dict):
        frame = str(rel_block.get("frame", "rect")).strip().lower()
        reference_frame = str(rel_block.get("reference_frame", rel_block.get("origin", "target"))).strip().lower()
        state = np.array(rel_block.get("state", []), dtype=float).reshape(-1)
        if state.size != 6:
            raise ValueError("chaser.initial_state.relative_to_target_ric.state must be length-6.")
        if frame not in ("rect", "curv"):
            raise ValueError("chaser.initial_state.relative_to_target_ric.frame must be 'rect' or 'curv'.")
        return state, frame, reference_frame
    if "relative_ric_rect" in s0:
        state = np.array(s0.get("relative_ric_rect"), dtype=float).reshape(-1)
        if state.size != 6:
            raise ValueError("chaser.initial_state.relative_ric_rect must be length-6.")
        return state, "rect", "target"
    if "relative_ric_curv" in s0:
        state = np.array(s0.get("relative_ric_curv"), dtype=float).reshape(-1)
        if state.size != 6:
            raise ValueError("chaser.initial_state.relative_ric_curv must be length-6.")
        return state, "curv", "target"
    return None


def _resolve_relative_cislunar_init(initial_state: dict[str, Any]) -> np.ndarray | None:
    s0 = dict(initial_state or {})
    rel_block = s0.get("relative_to_target_cislunar")
    if isinstance(rel_block, dict):
        state = np.array(rel_block.get("state", []), dtype=float).reshape(-1)
        if state.size != 6:
            raise ValueError("chaser.initial_state.relative_to_target_cislunar.state must be length-6.")
        return state
    if "relative_cislunar" in s0:
        state = np.array(s0.get("relative_cislunar"), dtype=float).reshape(-1)
        if state.size != 6:
            raise ValueError("chaser.initial_state.relative_cislunar must be length-6.")
        return state
    return None


def _apply_relative_init_from_reference(
    *,
    agent: AgentRuntime,
    reference: AgentRuntime,
    initial_state: dict[str, Any],
) -> None:
    rel = _resolve_chaser_relative_ric_init(initial_state)
    if rel is None or agent.truth is None or reference.truth is None:
        return
    x_rel, frame, reference_frame = rel
    moon_state = cr3bp_moon_state_km_s()
    use_moon_ric = reference_frame.replace("-", "_") in {"moon", "moon_ric", "lunar", "lunar_ric"}
    origin_r = moon_state[:3] if use_moon_ric else np.zeros(3, dtype=float)
    origin_v = moon_state[3:] if use_moon_ric else np.zeros(3, dtype=float)
    r_t_abs = np.array(reference.truth.position_eci_km, dtype=float)
    v_t_abs = np.array(reference.truth.velocity_eci_km_s, dtype=float)
    r_t = r_t_abs - origin_r
    v_t = v_t_abs - origin_v
    r0 = float(np.linalg.norm(r_t))
    if r0 <= 0.0:
        return
    x_rel_rect = ric_curv_to_rect(x_rel, r0_km=r0) if frame == "curv" else np.array(x_rel, dtype=float).reshape(6)
    x_agent_eci = ric_rect_state_to_eci(x_rel_rect, r_t, v_t)
    agent.truth.position_eci_km = x_agent_eci[:3] + origin_r
    agent.truth.velocity_eci_km_s = x_agent_eci[3:] + origin_v
    if agent.belief is not None and agent.belief.state.size >= 6:
        agent.belief.state[:3] = agent.truth.position_eci_km
        agent.belief.state[3:6] = agent.truth.velocity_eci_km_s


def _apply_relative_cislunar_init_from_reference(
    *,
    agent: AgentRuntime,
    reference: AgentRuntime,
    initial_state: dict[str, Any],
) -> None:
    rel = _resolve_relative_cislunar_init(initial_state)
    if rel is None or agent.truth is None or reference.truth is None:
        return
    ref_state = np.hstack((reference.truth.position_eci_km, reference.truth.velocity_eci_km_s))
    state = ref_state + np.array(rel, dtype=float).reshape(6)
    agent.truth.position_eci_km = state[:3]
    agent.truth.velocity_eci_km_s = state[3:]
    if agent.belief is not None and agent.belief.state.size >= 6:
        agent.belief.state[:3] = agent.truth.position_eci_km
        agent.belief.state[3:6] = agent.truth.velocity_eci_km_s


def _apply_chaser_relative_init_from_target(
    *,
    chaser: AgentRuntime,
    target: AgentRuntime,
    initial_state: dict[str, Any],
) -> None:
    _apply_relative_init_from_reference(agent=chaser, reference=target, initial_state=initial_state)
    _apply_relative_cislunar_init_from_reference(agent=chaser, reference=target, initial_state=initial_state)
