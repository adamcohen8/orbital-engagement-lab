from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from sim.config import default_reference_object_id, object_section
from sim.dynamics.orbit.elements import (
    ClassicalOrbitalElements,
    coe_to_rv_eci,
    coes_target_state_at_current_true_anomaly,
    rv_to_coe_eci,
)
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2, EARTH_RADIUS_KM
from sim.dynamics.orbit.lambert import solve_lambert_universal_variable
from sim.dynamics.orbit.two_body import propagate_two_body_rk4
from sim.orbital_calculator import mission_recovery_from_intrack_impulse, rocket_equation_mass_ratio
from sim.presets.thrusters import resolve_thruster_max_thrust_n_from_specs


def build_mission_recovery_summary(
    *,
    cfg: Any,
    t_s: np.ndarray,
    truth_hist: dict[str, np.ndarray],
) -> dict[str, Any]:
    section = getattr(getattr(cfg, "analysis", None), "mission_recovery", None)
    if section is None or not bool(getattr(section, "enabled", False)):
        return {}
    object_id = _resolve_object_id(cfg=cfg, section=section, truth_hist=truth_hist)
    if not object_id:
        return _unavailable_summary(section=section, object_id="", notes=["No object state history is available."])
    hist = np.asarray(truth_hist.get(object_id), dtype=float)
    times = np.asarray(t_s, dtype=float).reshape(-1)
    if hist.ndim != 2 or hist.shape[0] <= 0 or hist.shape[1] < 6 or times.size <= 0:
        return _unavailable_summary(
            section=section,
            object_id=object_id,
            notes=[f"Object '{object_id}' does not have a usable 6D truth history."],
        )

    n = int(min(hist.shape[0], times.size))
    assessment_idx = _assessment_index(times[:n], getattr(section, "assessment_time_s", "final"))
    initial_state = np.asarray(hist[0, :6], dtype=float)
    final_state = np.asarray(hist[assessment_idx, :6], dtype=float)
    initial_elements = rv_to_coe_eci(initial_state[:3], initial_state[3:6])
    final_elements = rv_to_coe_eci(final_state[:3], final_state[3:6])
    target_orbit_configured = bool(getattr(section, "target_orbit", {}) or {})
    target_elements = _resolve_target_orbit_elements(section=section, fallback_elements=initial_elements)
    target_reference_state = _state_from_elements(target_elements)
    mass_kg = _resolve_mass_kg(cfg=cfg, object_id=object_id, section=section, hist=hist, idx=assessment_idx)
    isp_s = _resolve_isp_s(cfg=cfg, object_id=object_id, section=section)
    max_thrust_n = _resolve_max_thrust_n(cfg=cfg, object_id=object_id, section=section)
    element_errors = _element_errors(initial_elements, final_elements, initial_state, final_state)
    target_reference_at_assessment = _propagate_state(target_reference_state, float(times[assessment_idx]))
    target_element_errors = _element_errors(
        target_elements,
        final_elements,
        target_reference_at_assessment,
        final_state,
    )
    goal = str(getattr(section, "goal", "orbit_shape") or "orbit_shape")
    recovery = _estimate_recovery(
        section=section,
        goal=goal,
        assessment_time_s=float(times[assessment_idx]),
        initial_state=initial_state,
        final_state=final_state,
        initial_elements=initial_elements,
        final_elements=final_elements,
        mass_kg=mass_kg,
        isp_s=isp_s,
    )
    tolerances = dict(getattr(section, "element_tolerances", {}) or {})
    summary = {
        "enabled": True,
        "object_id": object_id,
        "goal": goal,
        "assessment_time_s": float(times[assessment_idx]),
        "assessment_sample_index": int(assessment_idx),
        "initial_elements": _elements_dict(initial_elements),
        "target_elements": _elements_dict(target_elements),
        "target_orbit_configured": target_orbit_configured,
        "final_elements": _elements_dict(final_elements),
        "element_errors": element_errors,
        "target_element_errors": target_element_errors,
        "element_tolerances": tolerances,
        "within_element_tolerances": _within_tolerances(element_errors, tolerances),
        "within_target_element_tolerances": _within_tolerances(target_element_errors, tolerances),
        "mass_kg": mass_kg,
        "isp_s": isp_s,
        "max_thrust_n": max_thrust_n,
        "recovery_estimate": recovery,
    }
    recovery["scope"] = "original_orbit_reconstitution"
    recovery["display_name"] = "Original-Orbit Recovery Estimate"
    if target_orbit_configured:
        recovery.setdefault("notes", []).append(
            "This analytical estimate remains referenced to the object's initial orbit; "
            "configured target-orbit recommendations come from Orbit Transfer Planner candidates."
        )
    planner = _build_recovery_planner(
        section=section,
        goal=goal,
        assessment_time_s=float(times[assessment_idx]),
        initial_state=initial_state,
        final_state=final_state,
        initial_elements=initial_elements,
        target_elements=target_elements,
        target_reference_state=target_reference_state,
        final_elements=final_elements,
        mass_kg=mass_kg,
        isp_s=isp_s,
        max_thrust_n=max_thrust_n,
        tolerances=tolerances,
        recovery_estimate=recovery,
    )
    if planner:
        summary["planner"] = planner
    return summary


def write_mission_recovery_trade_space_plot(
    *,
    mission_recovery: dict[str, Any],
    outdir: Path,
    mode: str = "save",
    dpi: int = 150,
) -> str | None:
    planner = dict(mission_recovery.get("planner", {}) or {})
    candidates = [dict(item or {}) for item in list(planner.get("candidates", []) or [])]
    if not candidates:
        return None

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        time_s = _finite_or_none(candidate.get("planned_time_s"))
        delta_v = _finite_or_none(candidate.get("planned_delta_v_m_s"))
        if time_s is None or delta_v is None:
            continue
        rows.append(
            {
                "candidate_id": str(candidate.get("candidate_id") or ""),
                "time_min": time_s / 60.0,
                "delta_v_m_s": delta_v,
                "feasible": bool(candidate.get("feasible", False)),
                "verified": bool(candidate.get("verified", False)),
                "source": str(candidate.get("source") or ""),
                "source_family": str(candidate.get("source_family") or ""),
            }
        )
    if not rows:
        return None

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from sim.plotting.style import show_save_close_oel
    from sim.utils.figure_size import cap_figsize

    recommendations = dict(planner.get("recommended", {}) or {})
    modes_by_candidate: dict[str, list[str]] = {}
    for mode_name, candidate_id in recommendations.items():
        if candidate_id is None:
            continue
        modes_by_candidate.setdefault(str(candidate_id), []).append(str(mode_name))

    fig, ax = plt.subplots(figsize=cap_figsize(10.5, 6.0))
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.16, top=0.86)
    palette = {
        "verified": "#2e8b57",
        "feasible": "#3973b7",
        "infeasible": "#b94b48",
    }
    label_offsets = [
        (7, -16),
        (7, 18),
        (-72, 8),
        (-72, -16),
        (7, 20),
        (-75, 20),
        (7, -24),
        (7, 18),
        (-72, 8),
        (-75, -16),
    ]
    for idx, row in enumerate(rows):
        marker = "D" if row["source_family"] == "analytic_reconstitution" else "o"
        if row["verified"]:
            color = palette["verified"]
            label = "Verified"
        elif row["feasible"]:
            color = palette["feasible"]
            label = "Feasible"
        else:
            color = palette["infeasible"]
            label = "Infeasible"
        scatter_kwargs: dict[str, Any] = {
            "s": 110,
            "marker": marker,
            "color": color,
            "linewidths": 0.9,
            "alpha": 0.92,
            "zorder": 3,
        }
        scatter_kwargs["edgecolors"] = "white"
        ax.scatter(row["time_min"], row["delta_v_m_s"], **scatter_kwargs)
        modes = modes_by_candidate.get(row["candidate_id"], [])
        if modes:
            ax.scatter(
                row["time_min"],
                row["delta_v_m_s"],
                s=260,
                marker="*",
                color="#f2b84b",
                edgecolors="#3d2c00",
                linewidths=0.7,
                zorder=4,
            )
            label = f"{row['candidate_id']} / {', '.join(modes)}"
            label_offset = (7, 18)
        else:
            label = row["candidate_id"]
            label_offset = label_offsets[idx % len(label_offsets)]
        ax.annotate(
            label,
            (row["time_min"], row["delta_v_m_s"]),
            xytext=label_offset,
            textcoords="offset points",
            fontsize=8.5,
            color="#d7deea",
        )

    max_time_s = _finite_or_none(planner.get("max_recovery_time_s"))
    if max_time_s is not None and max_time_s > 0.0:
        ax.axvline(max_time_s / 60.0, color="#6f6f6f", linestyle="--", linewidth=1.0, alpha=0.75)
        ax.text(
            max_time_s / 60.0,
            0.98,
            "max time",
            transform=ax.get_xaxis_transform(),
            ha="right",
            va="top",
            rotation=90,
            fontsize=8,
            color="#6f6f6f",
        )
    max_delta_v = _finite_or_none(planner.get("max_recovery_delta_v_m_s"))
    if max_delta_v is not None and max_delta_v > 0.0:
        ax.axhline(max_delta_v, color="#6f6f6f", linestyle=":", linewidth=1.0, alpha=0.75)
        ax.text(ax.get_xlim()[1], max_delta_v, "max dV", ha="right", va="bottom", fontsize=8, color="#6f6f6f")

    x_vals = [float(row["time_min"]) for row in rows]
    y_vals = [float(row["delta_v_m_s"]) for row in rows]
    _pad_axis(ax, "x", x_vals, min_span=1.0)
    _pad_axis(ax, "y", y_vals, min_span=1.0)
    goal = str(mission_recovery.get("goal") or "")
    object_id = str(mission_recovery.get("object_id") or "")
    ax.set_title("Original-Orbit Recovery / Orbit Transfer Trade Space")
    ax.set_xlabel("Planned total recovery time (min)")
    ax.set_ylabel("Planned total delta-V (m/s)")
    subtitle = " / ".join(part for part in (object_id, goal) if part)
    if subtitle:
        ax.text(0.0, 1.01, subtitle, transform=ax.transAxes, fontsize=9.5, color="#666666", va="bottom")
    ax.grid(True, alpha=0.25)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="w", markerfacecolor=palette["verified"], label="Verified", markersize=8),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=palette["feasible"], label="Feasible", markersize=8),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=palette["infeasible"], label="Infeasible", markersize=8),
            Line2D([0], [0], marker="D", color="w", markerfacecolor="#808080", label="Original-orbit analytical baseline", markersize=7),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#808080", label="Orbit Transfer Planner", markersize=8),
            Line2D([0], [0], marker="*", color="w", markerfacecolor="#f2b84b", label="Recommended", markersize=12),
        ],
        loc="best",
        fontsize=8.5,
    )

    out_path = Path(outdir) / "mission_recovery_trade_space.png"
    mode_norm = str(mode or "save").strip().lower()
    should_save = mode_norm in {"save", "both"}
    saved = show_save_close_oel(
        fig,
        mode=mode_norm,
        out_path=out_path if should_save else None,
        dpi=int(dpi),
        artifact_id="mission_recovery_trade_space",
        plt_module=plt,
        close=should_save,
        show_block=False,
    )
    return str(saved or out_path) if should_save else None


def _resolve_object_id(*, cfg: Any, section: Any, truth_hist: dict[str, np.ndarray]) -> str:
    requested = str(getattr(section, "object_id", "") or "").strip()
    if requested:
        return requested
    return str(default_reference_object_id(cfg, available_ids=truth_hist.keys()) or "")


def _assessment_index(times: np.ndarray, value: Any) -> int:
    if isinstance(value, str) and value.strip().lower() == "final":
        return int(times.size - 1)
    target = float(value)
    return int(np.argmin(np.abs(times - target)))


def _resolve_mass_kg(*, cfg: Any, object_id: str, section: Any, hist: np.ndarray, idx: int) -> float | None:
    if hist.shape[1] > 13:
        mass = _finite_positive(hist[idx, 13])
        if mass is not None:
            return mass
    propulsion = dict(getattr(section, "propulsion", {}) or {})
    mass = _finite_positive(propulsion.get("spacecraft_mass_kg"))
    if mass is not None:
        return mass
    obj = object_section(cfg, object_id)
    specs = dict(getattr(obj, "specs", {}) or {}) if obj is not None else {}
    return _finite_positive(specs.get("mass_kg", specs.get("dry_mass_kg")))


def _resolve_isp_s(*, cfg: Any, object_id: str, section: Any) -> float | None:
    propulsion = dict(getattr(section, "propulsion", {}) or {})
    isp = _finite_positive(propulsion.get("isp_s"))
    if isp is not None:
        return isp
    obj = object_section(cfg, object_id)
    specs = dict(getattr(obj, "specs", {}) or {}) if obj is not None else {}
    return _finite_positive(specs.get("isp_s", specs.get("thruster_isp_s")))


def _resolve_max_thrust_n(*, cfg: Any, object_id: str, section: Any) -> float | None:
    propulsion = dict(getattr(section, "propulsion", {}) or {})
    max_thrust = _finite_positive(propulsion.get("max_thrust_n"))
    if max_thrust is not None:
        return max_thrust
    obj = object_section(cfg, object_id)
    specs = dict(getattr(obj, "specs", {}) or {}) if obj is not None else {}
    return _finite_positive(resolve_thruster_max_thrust_n_from_specs(specs))


def _resolve_target_orbit_elements(*, section: Any, fallback_elements: ClassicalOrbitalElements) -> ClassicalOrbitalElements:
    raw = dict(getattr(section, "target_orbit", {}) or {})
    if not raw:
        return fallback_elements
    coes = dict(raw.get("coes", raw) or {})
    merged = {
        "a_km": float(coes.get("a_km", coes.get("semi_major_axis_km", fallback_elements.a_km))),
        "ecc": float(coes.get("ecc", coes.get("e", fallback_elements.ecc))),
        "inc_deg": float(coes.get("inc_deg", coes.get("inclination_deg", fallback_elements.inc_deg))),
        "raan_deg": float(coes.get("raan_deg", fallback_elements.raan_deg)),
        "argp_deg": float(coes.get("argp_deg", coes.get("arg_periapsis_deg", fallback_elements.argp_deg))),
        "true_anomaly_deg": float(coes.get("true_anomaly_deg", coes.get("ta_deg", fallback_elements.true_anomaly_deg))),
    }
    r_tgt, v_tgt = coe_to_rv_eci(**merged)
    return rv_to_coe_eci(r_tgt, v_tgt)


def _state_from_elements(elements: ClassicalOrbitalElements) -> np.ndarray:
    r_eci, v_eci = coe_to_rv_eci(
        a_km=float(elements.a_km),
        ecc=float(elements.ecc),
        inc_deg=float(elements.inc_deg),
        raan_deg=float(elements.raan_deg),
        argp_deg=float(elements.argp_deg),
        true_anomaly_deg=float(elements.true_anomaly_deg),
    )
    return np.hstack((r_eci, v_eci))


def _estimate_recovery(
    *,
    section: Any,
    goal: str,
    assessment_time_s: float,
    initial_state: np.ndarray,
    final_state: np.ndarray,
    initial_elements: ClassicalOrbitalElements,
    final_elements: ClassicalOrbitalElements,
    mass_kg: float | None,
    isp_s: float | None,
) -> dict[str, Any]:
    notes: list[str] = []
    local_target = coes_target_state_at_current_true_anomaly(_elements_dict(initial_elements), final_state)
    local_delta_v_m_s = float(np.linalg.norm(local_target[3:6] - final_state[3:6]) * 1000.0)
    local_position_error_km = float(np.linalg.norm(local_target[:3] - final_state[:3]))
    in_track = _inferred_intrack_impulse(initial_elements=initial_elements, final_elements=final_elements, final_state=final_state)
    if in_track is not None:
        mass_for_calc = float(mass_kg) if mass_kg is not None else 1.0
        isp_for_calc = float(isp_s) if isp_s is not None else 220.0
        calc = mission_recovery_from_intrack_impulse(
            reference_altitude_km=float(initial_elements.a_km - EARTH_RADIUS_KM),
            disturbance_delta_v_m_s=float(in_track["disturbance_delta_v_m_s"]),
            spacecraft_mass_kg=mass_for_calc,
            isp_s=isp_for_calc,
            slot_tolerance_deg=float(getattr(section, "slot_tolerance_deg", 1.0)),
            max_phasing_orbits=int(getattr(section, "max_phasing_orbits", 5000)),
        )
        notes.extend(calc.notes)
        notes.append("Disturbance was inferred from the simulated final orbit, not from a declared burn.")
        if mass_kg is None or isp_s is None:
            notes.append("Propellant estimate unavailable because spacecraft mass or Isp was not available.")
        if abs(float(calc.disturbance_delta_v_m_s)) <= 1e-9 or str(calc.disturbance_apsis) == "circular":
            shape_wait_s = 0.0
        else:
            shape_wait_s = _time_to_apsis_s(final_elements, str(calc.disturbance_apsis))
        if goal == "orbit_slot":
            slot = _slot_recovery_from_assessment(
                calc=calc,
                assessment_time_s=assessment_time_s,
                slot_tolerance_deg=float(getattr(section, "slot_tolerance_deg", 1.0)),
                max_phasing_orbits=int(getattr(section, "max_phasing_orbits", 5000)),
            )
            recovery_time_s = slot["slot_recovery_wait_time_s"]
            if recovery_time_s is None:
                recovery_time_s = slot["best_slot_wait_time_s"]
                notes.append("Using best slot-recovery time because no candidate met the configured tolerance.")
        else:
            slot = None
            recovery_time_s = shape_wait_s
        row = {
            "available": True,
            "method": "sim_state_inferred_intrack_impulse",
            "recovery_delta_v_m_s": float(calc.recovery_delta_v_m_s),
            "recovery_time_s": None if recovery_time_s is None else float(recovery_time_s),
            "recovery_time_basis": "from_disturbance_apsis" if goal == "orbit_slot" else "from_assessment_state_to_same_apsis",
            "propellant_kg": None if mass_kg is None or isp_s is None else float(calc.recovery_propellant_kg),
            "propellant_fraction": None if mass_kg is None or isp_s is None else float(calc.recovery_propellant_fraction),
            "disturbance_delta_v_m_s": float(calc.disturbance_delta_v_m_s),
            "disturbance_apsis": calc.disturbance_apsis,
            "disturbed_period_s": float(calc.disturbed_period_s),
            "slot_recovery_found": bool(calc.slot_recovery_found),
            "slot_recovery_orbits": calc.slot_recovery_orbits,
            "slot_recovery_time_s": calc.slot_recovery_time_s,
            "slot_recovery_phase_error_deg": calc.slot_recovery_phase_error_deg,
            "best_slot_orbits": int(calc.best_slot_orbits),
            "best_slot_time_s": float(calc.best_slot_time_s),
            "best_slot_phase_error_deg": float(calc.best_slot_phase_error_deg),
            "local_orbit_shape_delta_v_m_s": local_delta_v_m_s,
            "local_orbit_shape_position_error_km": local_position_error_km,
            "notes": notes,
        }
        if slot is not None:
            row.update(slot)
        return row
    if mass_kg is None or isp_s is None:
        notes.append("Propellant estimate unavailable because spacecraft mass or Isp was not available.")
    if in_track is None:
        notes.append("Final orbit was not recognized as a simple in-track impulse from an initially circular orbit.")
    propellant = _propellant_for_delta_v(delta_v_m_s=local_delta_v_m_s, mass_kg=mass_kg, isp_s=isp_s)
    return {
        "available": goal == "orbit_shape",
        "method": "local_orbit_shape_velocity_match",
        "recovery_delta_v_m_s": local_delta_v_m_s,
        "recovery_time_s": 0.0 if goal == "orbit_shape" else None,
        "recovery_time_basis": "instantaneous_local_velocity_match" if goal == "orbit_shape" else "unavailable",
        "propellant_kg": propellant[0],
        "propellant_fraction": propellant[1],
        "local_orbit_shape_delta_v_m_s": local_delta_v_m_s,
        "local_orbit_shape_position_error_km": local_position_error_km,
        "notes": notes,
    }


def _build_recovery_planner(
    *,
    section: Any,
    goal: str,
    assessment_time_s: float,
    initial_state: np.ndarray,
    final_state: np.ndarray,
    target_elements: ClassicalOrbitalElements,
    initial_elements: ClassicalOrbitalElements,
    target_reference_state: np.ndarray,
    final_elements: ClassicalOrbitalElements,
    mass_kg: float | None,
    isp_s: float | None,
    max_thrust_n: float | None,
    tolerances: dict[str, float],
    recovery_estimate: dict[str, Any],
) -> dict[str, Any]:
    planner_cfg = dict(getattr(section, "planner", {}) or {})
    if not bool(planner_cfg.get("enabled", False)):
        return {}
    max_time_s = float(planner_cfg.get("max_recovery_time_s", 86400.0) or 0.0)
    max_delta_v = planner_cfg.get("max_recovery_delta_v_m_s")
    max_delta_v_m_s = None if max_delta_v in (None, "") else float(max_delta_v)
    candidate_count = max(int(planner_cfg.get("candidate_count", 12) or 12), 1)
    sources = [
        str(source)
        for source in list(
            planner_cfg.get("sources", ["analytic_reconstitution"])
            or ["analytic_reconstitution"]
        )
    ]
    modes = [str(mode) for mode in list(planner_cfg.get("modes", []) or [])] or [
        "min_delta_v",
        "min_time",
        "constrained",
    ]
    simulate_candidates = bool(planner_cfg.get("simulate_candidates", True))
    candidates = _generate_recovery_candidates(
        section=section,
        goal=goal,
        assessment_time_s=assessment_time_s,
        initial_state=initial_state,
        final_state=final_state,
        initial_elements=initial_elements,
        target_elements=target_elements,
        target_reference_state=target_reference_state,
        final_elements=final_elements,
        mass_kg=mass_kg,
        isp_s=isp_s,
        max_thrust_n=max_thrust_n,
        tolerances=tolerances,
        recovery_estimate=recovery_estimate,
        max_time_s=max_time_s,
        max_delta_v_m_s=max_delta_v_m_s,
        candidate_count=candidate_count,
        simulate_candidates=simulate_candidates,
        planner_cfg=planner_cfg,
    )
    configured_target_orbit = bool(getattr(section, "target_orbit", {}) or {})
    preferred_source_family = "orbit_transfer" if configured_target_orbit else None
    target_transfer_enabled = configured_target_orbit and "orbit_transfer" in sources
    recommendations = {
        mode: _recommended_candidate_id(
            candidates,
            mode=mode,
            max_delta_v_m_s=max_delta_v_m_s,
            max_time_s=max_time_s,
            preferred_source_family=preferred_source_family,
        )
        for mode in modes
    }
    return {
        "enabled": True,
        "sources": sources,
        "modes": modes,
        "max_recovery_time_s": max_time_s,
        "max_recovery_delta_v_m_s": max_delta_v_m_s,
        "candidate_count": len(candidates),
        "simulate_candidates": simulate_candidates,
        "recommendation_basis": (
            "configured_target_orbit_lambert"
            if target_transfer_enabled
            else "configured_target_orbit_no_transfer_source"
            if configured_target_orbit
            else "original_orbit_reconstitution"
        ),
        "analytical_baseline_candidate_ids": [
            str(candidate.get("candidate_id"))
            for candidate in candidates
            if candidate.get("source_family") == "analytic_reconstitution"
        ],
        "recommended": recommendations,
        "candidates": candidates,
    }


def _generate_recovery_candidates(
    *,
    section: Any,
    goal: str,
    assessment_time_s: float,
    initial_state: np.ndarray,
    final_state: np.ndarray,
    initial_elements: ClassicalOrbitalElements,
    target_elements: ClassicalOrbitalElements,
    target_reference_state: np.ndarray,
    final_elements: ClassicalOrbitalElements,
    mass_kg: float | None,
    isp_s: float | None,
    max_thrust_n: float | None,
    tolerances: dict[str, float],
    recovery_estimate: dict[str, Any],
    max_time_s: float,
    max_delta_v_m_s: float | None,
    candidate_count: int,
    simulate_candidates: bool,
    planner_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    sources = set(
        str(source)
        for source in list(
            planner_cfg.get("sources", ["analytic_reconstitution"])
            or ["analytic_reconstitution"]
        )
    )
    if "analytic_reconstitution" in sources:
        analytic_start = len(candidates)
        initial_guess = _candidate_from_state_match(
            candidate_id="candidate_001",
            source="immediate_local_velocity_match",
            description="Immediate burn that restores the initial orbit shape at the assessment position.",
            recovery_time_s=0.0,
            initial_elements=initial_elements,
            disturbed_state=final_state,
            mass_kg=mass_kg,
            isp_s=isp_s,
            max_thrust_n=max_thrust_n,
            tolerances=tolerances,
            max_delta_v_m_s=max_delta_v_m_s,
            max_time_s=max_time_s,
            simulate_candidates=simulate_candidates,
        )
        if goal == "orbit_shape":
            candidates.append(initial_guess)

        in_track = _inferred_intrack_impulse(
            initial_elements=initial_elements,
            final_elements=final_elements,
            final_state=final_state,
        )
        if in_track is not None:
            candidates.extend(
                _intrack_recovery_candidates(
                    section=section,
                    goal=goal,
                    assessment_time_s=assessment_time_s,
                    initial_state=initial_state,
                    final_state=final_state,
                    initial_elements=initial_elements,
                    final_elements=final_elements,
                    mass_kg=mass_kg,
                    isp_s=isp_s,
                    max_thrust_n=max_thrust_n,
                    in_track=in_track,
                    tolerances=tolerances,
                    max_delta_v_m_s=max_delta_v_m_s,
                    max_time_s=max_time_s,
                    simulate_candidates=simulate_candidates,
                    start_index=len(candidates) + 1,
                )
            )

        sample_count = max(candidate_count - len(candidates), 0)
        if goal == "orbit_shape" and sample_count > 0 and max_time_s > 0.0:
            sample_times = np.linspace(0.0, max_time_s, sample_count + 2, dtype=float)[1:-1]
            for dt_s in sample_times:
                disturbed_state = _propagate_state(final_state, float(dt_s))
                candidates.append(
                    _candidate_from_state_match(
                        candidate_id=f"candidate_{len(candidates) + 1:03d}",
                        source="sampled_coast_local_velocity_match",
                        description="Coast, then burn to restore the initial orbit shape at the recovery position.",
                        recovery_time_s=float(dt_s),
                        initial_elements=initial_elements,
                        disturbed_state=disturbed_state,
                        mass_kg=mass_kg,
                        isp_s=isp_s,
                        max_thrust_n=max_thrust_n,
                        tolerances=tolerances,
                        max_delta_v_m_s=max_delta_v_m_s,
                        max_time_s=max_time_s,
                        simulate_candidates=simulate_candidates,
                    )
                )
        for candidate in candidates[analytic_start:]:
            candidate["source_family"] = "analytic_reconstitution"
            candidate["target_basis"] = "initial_orbit"
            candidate.setdefault("notes", []).append(
                "This candidate is an Original-Orbit Recovery Estimate baseline, not a configured target-orbit transfer."
            )

    orbit_transfer_cfg = dict(planner_cfg.get("orbit_transfer", {}) or {})
    if bool(orbit_transfer_cfg.get("enabled", False)) or "orbit_transfer" in sources:
        candidates.extend(
            _orbit_transfer_lambert_candidates(
                section=section,
                goal=goal,
                assessment_time_s=assessment_time_s,
                initial_state=initial_state,
                final_state=final_state,
                target_elements=target_elements,
                target_reference_state=target_reference_state,
                mass_kg=mass_kg,
                isp_s=isp_s,
                max_thrust_n=max_thrust_n,
                tolerances=tolerances,
                max_delta_v_m_s=max_delta_v_m_s,
                max_time_s=max_time_s,
                simulate_candidates=simulate_candidates,
                planner_cfg=orbit_transfer_cfg,
                target_basis=(
                    "configured_target_orbit"
                    if bool(getattr(section, "target_orbit", {}) or {})
                    else "initial_orbit"
                ),
                start_index=len(candidates) + 1,
            )
        )

    unique: dict[tuple[str, float], dict[str, Any]] = {}
    for candidate in candidates:
        phase = float(candidate.get("target_phase_deg", 0.0) or 0.0)
        branch = 1.0 if bool(candidate.get("lambert_short_way", True)) else -1.0
        key = (
            str(candidate.get("source", "")),
            round(float(candidate.get("departure_wait_s", 0.0) or 0.0), 6),
            round(float(candidate.get("time_of_flight_s", candidate.get("planned_time_s", 0.0)) or 0.0), 6),
            round(phase, 6),
            branch,
        )
        existing = unique.get(key)
        if existing is None or float(candidate.get("planned_delta_v_m_s", math.inf)) < float(
            existing.get("planned_delta_v_m_s", math.inf)
        ):
            unique[key] = candidate
    ranked = sorted(
        unique.values(),
        key=lambda item: (
            not bool(item.get("feasible", False)),
            float(item.get("planned_delta_v_m_s", math.inf)),
            float(item.get("planned_time_s", math.inf)),
        ),
    )
    candidate_families = {str(item.get("source_family") or "") for item in ranked}
    if {"analytic_reconstitution", "orbit_transfer"}.issubset(candidate_families):
        ranked = _retain_transfer_with_analytical_baseline(ranked, candidate_count)
    return [
        {**candidate, "candidate_id": f"candidate_{idx:03d}"}
        for idx, candidate in enumerate(ranked[:candidate_count], start=1)
    ]


def _orbit_transfer_lambert_candidates(
    *,
    section: Any,
    goal: str,
    assessment_time_s: float,
    initial_state: np.ndarray,
    final_state: np.ndarray,
    target_elements: ClassicalOrbitalElements,
    target_reference_state: np.ndarray,
    mass_kg: float | None,
    isp_s: float | None,
    max_thrust_n: float | None,
    tolerances: dict[str, float],
    max_delta_v_m_s: float | None,
    max_time_s: float,
    simulate_candidates: bool,
    planner_cfg: dict[str, Any],
    target_basis: str,
    start_index: int,
) -> list[dict[str, Any]]:
    if max_time_s <= 0.0:
        return []
    departure_samples = max(int(planner_cfg.get("departure_samples", 9) or 9), 1)
    tof_samples = max(int(planner_cfg.get("time_of_flight_samples", 12) or 12), 1)
    anomaly_samples = max(int(planner_cfg.get("target_anomaly_samples", 24) or 24), 1)
    impulse_epsilon_m_s = max(float(planner_cfg.get("impulse_epsilon_m_s", 1.0e-2) or 0.0), 0.0)
    min_tof = max(float(planner_cfg.get("min_time_of_flight_s", 60.0) or 60.0), 1.0e-6)
    max_tof_cfg = planner_cfg.get("max_time_of_flight_s")
    max_tof = max_time_s if max_tof_cfg in (None, "") else min(float(max_tof_cfg), max_time_s)
    if max_tof < min_tof:
        return []
    branches: list[bool] = []
    if bool(planner_cfg.get("short_way", True)):
        branches.append(True)
    if bool(planner_cfg.get("long_way", False)):
        branches.append(False)
    if not branches:
        branches.append(True)

    departure_times = _sample_transfer_axis(0.0, max(0.0, max_time_s - min_tof), departure_samples)
    candidates: list[dict[str, Any]] = []
    failures = 0
    for departure_wait_s in departure_times:
        remaining = max_time_s - float(departure_wait_s)
        if remaining < min_tof:
            continue
        tof_values = _sample_transfer_axis(min_tof, min(max_tof, remaining), tof_samples)
        departure_state = _propagate_state(final_state, float(departure_wait_s))
        for tof_s in tof_values:
            total_time_s = float(departure_wait_s + tof_s)
            target_states = _orbit_transfer_target_states(
                goal=goal,
                target_reference_state=target_reference_state,
                target_elements=target_elements,
                assessment_time_s=assessment_time_s,
                total_time_s=total_time_s,
                anomaly_samples=anomaly_samples,
            )
            for target_phase_deg, target_state in target_states:
                for short_way in branches:
                    try:
                        solution = solve_lambert_universal_variable(
                            departure_state[:3],
                            target_state[:3],
                            float(tof_s),
                            short_way=short_way,
                            revolutions=0,
                        )
                    except ValueError:
                        failures += 1
                        continue
                    burn1_km_s = solution.v1_km_s - departure_state[3:6]
                    burn2_km_s = target_state[3:6] - solution.v2_km_s
                    burn1_m_s = float(np.linalg.norm(burn1_km_s) * 1000.0)
                    burn2_m_s = float(np.linalg.norm(burn2_km_s) * 1000.0)
                    total_delta_v_m_s = burn1_m_s + burn2_m_s
                    candidates.append(
                        _orbit_transfer_candidate_row(
                            candidate_id=f"candidate_{start_index + len(candidates):03d}",
                            goal=goal,
                            total_delta_v_m_s=total_delta_v_m_s,
                            departure_wait_s=float(departure_wait_s),
                            time_of_flight_s=float(tof_s),
                            total_time_s=total_time_s,
                            target_phase_deg=target_phase_deg,
                            departure_state=departure_state,
                            target_state=target_state,
                            burn1_km_s=burn1_km_s,
                            burn2_km_s=burn2_km_s,
                            burn1_m_s=burn1_m_s,
                            burn2_m_s=burn2_m_s,
                            solution=solution,
                            mass_kg=mass_kg,
                            isp_s=isp_s,
                            max_thrust_n=max_thrust_n,
                            tolerances=_slot_tolerances(section=section, tolerances=tolerances)
                            if goal == "orbit_slot"
                            else tolerances,
                            max_delta_v_m_s=max_delta_v_m_s,
                            max_time_s=max_time_s,
                            simulate_candidates=simulate_candidates,
                            target_elements=target_elements,
                            target_basis=target_basis,
                            impulse_epsilon_m_s=impulse_epsilon_m_s,
                        )
                    )

    if bool(planner_cfg.get("keep_per_time_best", True)):
        candidates = _best_orbit_transfer_candidates_per_time(candidates)
    if failures:
        for candidate in candidates:
            notes = list(candidate.get("notes", []) or [])
            notes.append(f"Orbit Transfer Planner skipped {failures} non-converged or singular Lambert grid points.")
            candidate["notes"] = notes
    return candidates


def _orbit_transfer_candidate_row(
    *,
    candidate_id: str,
    goal: str,
    total_delta_v_m_s: float,
    departure_wait_s: float,
    time_of_flight_s: float,
    total_time_s: float,
    target_phase_deg: float | None,
    departure_state: np.ndarray,
    target_state: np.ndarray,
    burn1_km_s: np.ndarray,
    burn2_km_s: np.ndarray,
    burn1_m_s: float,
    burn2_m_s: float,
    solution: Any,
    mass_kg: float | None,
    isp_s: float | None,
    max_thrust_n: float | None,
    tolerances: dict[str, float],
    max_delta_v_m_s: float | None,
    max_time_s: float,
    simulate_candidates: bool,
    target_elements: ClassicalOrbitalElements,
    target_basis: str,
    impulse_epsilon_m_s: float,
) -> dict[str, Any]:
    propellant = _propellant_for_delta_v(delta_v_m_s=total_delta_v_m_s, mass_kg=mass_kg, isp_s=isp_s)
    feasible = bool(solution.converged and _candidate_feasible(total_delta_v_m_s, total_time_s, max_delta_v_m_s, max_time_s))
    expected_final_elements: dict[str, float] = {}
    expected_element_errors: dict[str, float] = {}
    within_tolerances = None
    position_residual_km = None
    velocity_residual_m_s = None
    simulated_delta_v = None
    simulated_recovery_time = None
    if simulate_candidates:
        post_burn1 = np.hstack((departure_state[:3], departure_state[3:6] + burn1_km_s))
        transfer_arrival = _propagate_state(post_burn1, time_of_flight_s)
        recovered_state = np.hstack((transfer_arrival[:3], transfer_arrival[3:6] + burn2_km_s))
        recovered_elements = rv_to_coe_eci(recovered_state[:3], recovered_state[3:6])
        reference_elements = rv_to_coe_eci(target_state[:3], target_state[3:6]) if goal == "orbit_slot" else target_elements
        expected_element_errors = _element_errors(reference_elements, recovered_elements, target_state, recovered_state)
        if goal == "orbit_slot":
            expected_element_errors["slot_phase_deg"] = _position_angle_error_deg(target_state, recovered_state)
        expected_final_elements = _elements_dict(recovered_elements)
        within_tolerances = _within_tolerances(expected_element_errors, tolerances)
        position_residual_km = float(np.linalg.norm(recovered_state[:3] - target_state[:3]))
        velocity_residual_m_s = float(np.linalg.norm(recovered_state[3:6] - target_state[3:6]) * 1000.0)
        simulated_delta_v = float(burn1_m_s + burn2_m_s)
        simulated_recovery_time = float(total_time_s)
    verified = bool(simulate_candidates and feasible and within_tolerances is not False)
    if position_residual_km is not None and position_residual_km > 1.0e-2:
        verified = False
    transfer_type = _orbit_transfer_type(
        burn1_m_s=burn1_m_s,
        burn2_m_s=burn2_m_s,
        impulse_epsilon_m_s=impulse_epsilon_m_s,
    )
    burn_sequence = _orbit_transfer_burn_sequence(
        departure_wait_s=departure_wait_s,
        total_time_s=total_time_s,
        burn1_km_s=burn1_km_s,
        burn2_km_s=burn2_km_s,
        burn1_m_s=burn1_m_s,
        burn2_m_s=burn2_m_s,
        mass_kg=mass_kg,
        max_thrust_n=max_thrust_n,
        impulse_epsilon_m_s=impulse_epsilon_m_s,
    )
    collapsed = 2 - len(burn_sequence)
    notes = [
        "Candidate is an Orbit Transfer Planner two-body Lambert solution.",
        "Minimum delta-v claims apply only to the configured sampled search grid.",
        "Verified field reflects deterministic two-body propagation of the planned impulses.",
    ]
    if collapsed:
        notes.append(
            f"Collapsed {collapsed} Lambert impulse(s) at or below "
            f"{float(impulse_epsilon_m_s):.6g} m/s for reporting."
        )
    return {
        "candidate_id": candidate_id,
        "source": "orbit_transfer_lambert",
        "source_family": "orbit_transfer",
        "target_basis": str(target_basis),
        "description": "Orbit Transfer Planner Lambert transfer over a sampled departure and arrival time.",
        "goal": goal,
        "planned_delta_v_m_s": float(total_delta_v_m_s),
        "simulated_delta_v_m_s": simulated_delta_v,
        "planned_time_s": float(total_time_s),
        "simulated_recovery_time_s": simulated_recovery_time,
        "propellant_kg": propellant[0],
        "propellant_fraction": propellant[1],
        "feasible": feasible,
        "verified": verified,
        "within_tolerances": within_tolerances,
        "score": _candidate_score(total_delta_v_m_s, total_time_s, max_delta_v_m_s, max_time_s),
        "expected_final_elements": expected_final_elements,
        "expected_element_errors": expected_element_errors,
        "transfer_type": transfer_type,
        "departure_wait_s": float(departure_wait_s),
        "time_of_flight_s": float(time_of_flight_s),
        "arrival_time_s": float(total_time_s),
        "target_phase_deg": None if target_phase_deg is None else float(target_phase_deg),
        "lambert_short_way": bool(solution.short_way),
        "lambert_revolutions": int(solution.revolutions),
        "solver_iterations": int(solution.iterations),
        "solver_residual_s": float(solution.residual_s),
        "position_residual_km": position_residual_km,
        "velocity_residual_m_s": velocity_residual_m_s,
        "burn_sequence": burn_sequence,
        "notes": notes,
    }


def _orbit_transfer_type(
    *,
    burn1_m_s: float,
    burn2_m_s: float,
    impulse_epsilon_m_s: float,
) -> str:
    departure_active = float(burn1_m_s) > float(impulse_epsilon_m_s)
    arrival_active = float(burn2_m_s) > float(impulse_epsilon_m_s)
    if departure_active and arrival_active:
        return "two_impulse_lambert"
    if departure_active:
        return "one_impulse_departure"
    if arrival_active:
        return "one_impulse_arrival"
    return "zero_impulse"


def _orbit_transfer_burn_sequence(
    *,
    departure_wait_s: float,
    total_time_s: float,
    burn1_km_s: np.ndarray,
    burn2_km_s: np.ndarray,
    burn1_m_s: float,
    burn2_m_s: float,
    mass_kg: float | None,
    max_thrust_n: float | None,
    impulse_epsilon_m_s: float,
) -> list[dict[str, Any]]:
    sequence: list[dict[str, Any]] = []
    for start_time_s, axis, delta_v_km_s, delta_v_m_s in (
        (float(departure_wait_s), "lambert_departure", burn1_km_s, float(burn1_m_s)),
        (float(total_time_s), "lambert_arrival_match", burn2_km_s, float(burn2_m_s)),
    ):
        if delta_v_m_s <= float(impulse_epsilon_m_s):
            continue
        sequence.append(
            {
                "burn_index": len(sequence),
                "start_time_s": float(start_time_s),
                "duration_s": _burn_duration_s(delta_v_m_s=delta_v_m_s, mass_kg=mass_kg, max_thrust_n=max_thrust_n),
                "frame": "eci",
                "axis": axis,
                "delta_v_m_s": float(delta_v_m_s),
                "delta_v_eci_m_s": [float(x) for x in np.asarray(delta_v_km_s, dtype=float) * 1000.0],
            }
        )
    return sequence


def _orbit_transfer_target_states(
    *,
    goal: str,
    target_reference_state: np.ndarray,
    target_elements: ClassicalOrbitalElements,
    assessment_time_s: float,
    total_time_s: float,
    anomaly_samples: int,
) -> list[tuple[float | None, np.ndarray]]:
    if goal == "orbit_slot":
        state = _propagate_state(target_reference_state, float(assessment_time_s + total_time_s))
        return [(None, state)]
    if int(anomaly_samples) == 1:
        anomalies = [float(target_elements.true_anomaly_deg)]
    else:
        anomalies = [float(x) for x in np.linspace(0.0, 360.0, int(anomaly_samples), endpoint=False)]
    out: list[tuple[float | None, np.ndarray]] = []
    for anomaly_deg in anomalies:
        r_tgt, v_tgt = coe_to_rv_eci(
            a_km=float(target_elements.a_km),
            ecc=float(target_elements.ecc),
            inc_deg=float(target_elements.inc_deg),
            raan_deg=float(target_elements.raan_deg),
            argp_deg=float(target_elements.argp_deg),
            true_anomaly_deg=float(anomaly_deg),
        )
        out.append((float(anomaly_deg), np.hstack((r_tgt, v_tgt))))
    return out


def _sample_transfer_axis(start: float, stop: float, count: int) -> np.ndarray:
    if int(count) <= 1 or abs(float(stop) - float(start)) <= 1.0e-12:
        return np.array([float(start)], dtype=float)
    return np.linspace(float(start), float(stop), int(count), dtype=float)


def _best_orbit_transfer_candidates_per_time(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[float, bool], dict[str, Any]] = {}
    for candidate in candidates:
        key = (
            round(float(candidate.get("planned_time_s", 0.0) or 0.0), 6),
            bool(candidate.get("lambert_short_way", True)),
        )
        existing = best.get(key)
        if existing is None or float(candidate.get("planned_delta_v_m_s", math.inf)) < float(
            existing.get("planned_delta_v_m_s", math.inf)
        ):
            best[key] = candidate
    return list(best.values())


def _retain_transfer_with_analytical_baseline(
    ranked: list[dict[str, Any]],
    candidate_count: int,
) -> list[dict[str, Any]]:
    """Prefer configured-target transfers while retaining one original-orbit baseline."""

    limit = max(int(candidate_count), 1)
    transfers = [item for item in ranked if item.get("source_family") == "orbit_transfer"]
    baselines = [item for item in ranked if item.get("source_family") == "analytic_reconstitution"]
    if not transfers:
        return ranked[:limit]
    selected = list(transfers[:limit])
    if limit > 1 and baselines:
        selected = [*transfers[: limit - 1], baselines[0]]
    selected_ids = {id(item) for item in selected}
    for item in ranked:
        if len(selected) >= limit:
            break
        if id(item) not in selected_ids:
            selected.append(item)
            selected_ids.add(id(item))
    rank_by_id = {id(item): idx for idx, item in enumerate(ranked)}
    return sorted(selected, key=lambda item: rank_by_id[id(item)])


def _candidate_from_state_match(
    *,
    candidate_id: str,
    source: str,
    description: str,
    recovery_time_s: float,
    initial_elements: ClassicalOrbitalElements,
    disturbed_state: np.ndarray,
    mass_kg: float | None,
    isp_s: float | None,
    max_thrust_n: float | None,
    tolerances: dict[str, float],
    max_delta_v_m_s: float | None,
    max_time_s: float,
    simulate_candidates: bool,
) -> dict[str, Any]:
    target_state = coes_target_state_at_current_true_anomaly(_elements_dict(initial_elements), disturbed_state)
    delta_v_eci_km_s = target_state[3:6] - disturbed_state[3:6]
    delta_v_m_s = float(np.linalg.norm(delta_v_eci_km_s) * 1000.0)
    recovered_state = np.hstack((disturbed_state[:3], disturbed_state[3:6] + delta_v_eci_km_s))
    recovered_elements = rv_to_coe_eci(recovered_state[:3], recovered_state[3:6])
    errors = _element_errors(initial_elements, recovered_elements, target_state, recovered_state)
    within_tolerances = _within_tolerances(errors, tolerances)
    propellant = _propellant_for_delta_v(delta_v_m_s=delta_v_m_s, mass_kg=mass_kg, isp_s=isp_s)
    feasible = _candidate_feasible(delta_v_m_s, recovery_time_s, max_delta_v_m_s, max_time_s)
    verified = bool(simulate_candidates and (within_tolerances is not False))
    return {
        "candidate_id": candidate_id,
        "source": source,
        "description": description,
        "goal": "orbit_shape",
        "planned_delta_v_m_s": delta_v_m_s,
        "simulated_delta_v_m_s": delta_v_m_s if simulate_candidates else None,
        "planned_time_s": float(recovery_time_s),
        "simulated_recovery_time_s": float(recovery_time_s) if simulate_candidates else None,
        "propellant_kg": propellant[0],
        "propellant_fraction": propellant[1],
        "feasible": feasible,
        "verified": verified,
        "within_tolerances": within_tolerances,
        "score": _candidate_score(delta_v_m_s, recovery_time_s, max_delta_v_m_s, max_time_s),
        "expected_final_elements": _elements_dict(recovered_elements),
        "expected_element_errors": errors,
        "burn_sequence": [
            {
                "burn_index": 0,
                "start_time_s": float(recovery_time_s),
                "duration_s": _burn_duration_s(delta_v_m_s=delta_v_m_s, mass_kg=mass_kg, max_thrust_n=max_thrust_n),
                "frame": "eci",
                "axis": "velocity_match",
                "delta_v_m_s": delta_v_m_s,
                "delta_v_eci_m_s": [float(x) for x in delta_v_eci_km_s * 1000.0],
            }
        ],
        "notes": [
            "Candidate uses a two-body local velocity match against the initial orbit shape.",
            "Verified field reflects analytic post-burn element comparison, not a separate closed-loop controller run.",
        ],
    }


def _intrack_recovery_candidates(
    *,
    section: Any,
    goal: str,
    assessment_time_s: float,
    initial_state: np.ndarray,
    final_state: np.ndarray,
    initial_elements: ClassicalOrbitalElements,
    final_elements: ClassicalOrbitalElements,
    mass_kg: float | None,
    isp_s: float | None,
    max_thrust_n: float | None,
    in_track: dict[str, float],
    tolerances: dict[str, float],
    max_delta_v_m_s: float | None,
    max_time_s: float,
    simulate_candidates: bool,
    start_index: int,
) -> list[dict[str, Any]]:
    mass_for_calc = float(mass_kg) if mass_kg is not None else 1.0
    isp_for_calc = float(isp_s) if isp_s is not None else 220.0
    calc = mission_recovery_from_intrack_impulse(
        reference_altitude_km=float(initial_elements.a_km - EARTH_RADIUS_KM),
        disturbance_delta_v_m_s=float(in_track["disturbance_delta_v_m_s"]),
        spacecraft_mass_kg=mass_for_calc,
        isp_s=isp_for_calc,
        slot_tolerance_deg=float(getattr(section, "slot_tolerance_deg", 1.0)),
        max_phasing_orbits=int(getattr(section, "max_phasing_orbits", 5000)),
    )
    rows: list[dict[str, Any]] = []
    shape_time = 0.0
    if abs(float(calc.disturbance_delta_v_m_s)) > 1e-9 and str(calc.disturbance_apsis) != "circular":
        shape_time = float(_time_to_apsis_s(final_elements, str(calc.disturbance_apsis)) or 0.0)
    signed_recovery_delta_v_m_s = -float(in_track["disturbance_delta_v_m_s"])
    if goal == "orbit_shape":
        rows.append(
            _simple_impulse_candidate(
                candidate_id=f"candidate_{start_index:03d}",
                source="same_apsis_shape_recovery",
                description="Wait to the disturbed apsis, then apply the equal-and-opposite in-track recovery burn.",
                goal="orbit_shape",
                signed_delta_v_m_s=signed_recovery_delta_v_m_s,
                recovery_time_s=shape_time,
                assessment_time_s=assessment_time_s,
                initial_state=initial_state,
                final_state=final_state,
                initial_elements=initial_elements,
                mass_kg=mass_kg,
                isp_s=isp_s,
                max_thrust_n=max_thrust_n,
                tolerances={},
                max_delta_v_m_s=max_delta_v_m_s,
                max_time_s=max_time_s,
                simulate_candidates=simulate_candidates,
                notes=list(calc.notes) + ["Candidate is valid for simple inferred in-track disturbances."],
            )
        )
    if goal == "orbit_slot":
        slot = _slot_recovery_from_assessment(
            calc=calc,
            assessment_time_s=assessment_time_s,
            slot_tolerance_deg=float(getattr(section, "slot_tolerance_deg", 1.0)),
            max_phasing_orbits=int(getattr(section, "max_phasing_orbits", 5000)),
        )
        slot_time = slot["slot_recovery_wait_time_s"]
        if slot_time is None:
            slot_time = slot["best_slot_wait_time_s"]
        rows.append(
            _simple_impulse_candidate(
                candidate_id=f"candidate_{start_index + 1:03d}",
                source="slot_phasing_recovery",
                description="Use same-apsis phasing opportunity to recover the original orbit slot.",
                goal="orbit_slot",
                signed_delta_v_m_s=signed_recovery_delta_v_m_s,
                recovery_time_s=float(slot_time),
                assessment_time_s=assessment_time_s,
                initial_state=initial_state,
                final_state=final_state,
                initial_elements=initial_elements,
                mass_kg=mass_kg,
                isp_s=isp_s,
                max_thrust_n=max_thrust_n,
                tolerances=_slot_tolerances(section=section, tolerances=tolerances),
                max_delta_v_m_s=max_delta_v_m_s,
                max_time_s=max_time_s,
                simulate_candidates=simulate_candidates,
                notes=list(calc.notes)
                + [
                    "Candidate uses the orbital-calculator slot phasing search.",
                    f"Best slot phase error deg: {float(calc.best_slot_phase_error_deg):.6g}.",
                ],
                extra={
                    "slot_recovery_found": bool(calc.slot_recovery_found),
                    "slot_recovery_orbits": calc.slot_recovery_orbits,
                    "slot_recovery_phase_error_deg": calc.slot_recovery_phase_error_deg,
                    "best_slot_orbits": int(calc.best_slot_orbits),
                    "best_slot_phase_error_deg": float(calc.best_slot_phase_error_deg),
                    **slot,
                },
            )
        )
    return rows


def _simple_impulse_candidate(
    *,
    candidate_id: str,
    source: str,
    description: str,
    goal: str,
    signed_delta_v_m_s: float,
    recovery_time_s: float,
    assessment_time_s: float,
    initial_state: np.ndarray,
    final_state: np.ndarray,
    initial_elements: ClassicalOrbitalElements,
    mass_kg: float | None,
    isp_s: float | None,
    max_thrust_n: float | None,
    tolerances: dict[str, float],
    max_delta_v_m_s: float | None,
    max_time_s: float,
    simulate_candidates: bool,
    notes: list[str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    delta_v_m_s = abs(float(signed_delta_v_m_s))
    propellant = _propellant_for_delta_v(delta_v_m_s=delta_v_m_s, mass_kg=mass_kg, isp_s=isp_s)
    feasible = _candidate_feasible(delta_v_m_s, recovery_time_s, max_delta_v_m_s, max_time_s)
    expected_final_elements: dict[str, float] = {}
    expected_element_errors: dict[str, float] = {}
    within_tolerances = None
    delta_v_eci_m_s: list[float] | None = None
    if simulate_candidates:
        burn_state = _propagate_state(final_state, recovery_time_s)
        intrack = _intrack_unit(burn_state)
        delta_v_eci_km_s = intrack * (float(signed_delta_v_m_s) / 1000.0)
        delta_v_eci_m_s = [float(x) for x in delta_v_eci_km_s * 1000.0]
        recovered_state = np.hstack((burn_state[:3], burn_state[3:6] + delta_v_eci_km_s))
        recovered_elements = rv_to_coe_eci(recovered_state[:3], recovered_state[3:6])
        if goal == "orbit_slot":
            reference_state = _propagate_state(initial_state, assessment_time_s + recovery_time_s)
            reference_elements = rv_to_coe_eci(reference_state[:3], reference_state[3:6])
            expected_element_errors = _element_errors(reference_elements, recovered_elements, reference_state, recovered_state)
            expected_element_errors["slot_phase_deg"] = _position_angle_error_deg(reference_state, recovered_state)
        else:
            target_state = coes_target_state_at_current_true_anomaly(_elements_dict(initial_elements), recovered_state)
            expected_element_errors = _element_errors(initial_elements, recovered_elements, target_state, recovered_state)
        expected_final_elements = _elements_dict(recovered_elements)
        within_tolerances = _within_tolerances(expected_element_errors, tolerances)
    row = {
        "candidate_id": candidate_id,
        "source": source,
        "description": description,
        "goal": goal,
        "planned_delta_v_m_s": float(delta_v_m_s),
        "simulated_delta_v_m_s": float(delta_v_m_s) if simulate_candidates else None,
        "planned_time_s": float(recovery_time_s),
        "simulated_recovery_time_s": float(recovery_time_s) if simulate_candidates else None,
        "propellant_kg": propellant[0],
        "propellant_fraction": propellant[1],
        "feasible": feasible,
        "verified": bool(simulate_candidates and feasible and within_tolerances is not False),
        "within_tolerances": within_tolerances,
        "score": _candidate_score(delta_v_m_s, recovery_time_s, max_delta_v_m_s, max_time_s),
        "expected_final_elements": expected_final_elements,
        "expected_element_errors": expected_element_errors,
        "burn_sequence": [
            {
                "burn_index": 0,
                "start_time_s": float(recovery_time_s),
                "duration_s": _burn_duration_s(delta_v_m_s=delta_v_m_s, mass_kg=mass_kg, max_thrust_n=max_thrust_n),
                "frame": "ric",
                "axis": "+I" if float(signed_delta_v_m_s) >= 0.0 else "-I",
                "delta_v_m_s": delta_v_m_s,
                "delta_v_eci_m_s": delta_v_eci_m_s,
            }
        ],
        "notes": notes,
    }
    if extra:
        row.update(dict(extra))
    return row


def _candidate_feasible(
    delta_v_m_s: float,
    recovery_time_s: float,
    max_delta_v_m_s: float | None,
    max_time_s: float,
) -> bool:
    if max_delta_v_m_s is not None and delta_v_m_s > max_delta_v_m_s:
        return False
    if recovery_time_s > max_time_s:
        return False
    return True


def _candidate_score(delta_v_m_s: float, recovery_time_s: float, max_delta_v_m_s: float | None, max_time_s: float) -> float:
    dv_den = max(float(max_delta_v_m_s) if max_delta_v_m_s is not None else max(delta_v_m_s, 1.0), 1.0)
    t_den = max(float(max_time_s), 1.0)
    return float((delta_v_m_s / dv_den) + (recovery_time_s / t_den))


def _recommended_candidate_id(
    candidates: list[dict[str, Any]],
    *,
    mode: str,
    max_delta_v_m_s: float | None,
    max_time_s: float,
    preferred_source_family: str | None = None,
) -> str | None:
    eligible = candidates
    if preferred_source_family is not None:
        eligible = [
            item for item in candidates if item.get("source_family") == preferred_source_family
        ]
    feasible = [item for item in eligible if bool(item.get("feasible", False))]
    if not feasible:
        return None
    verified = [item for item in feasible if bool(item.get("verified", False))]
    if verified:
        feasible = verified
    if mode == "min_time":
        selected = min(feasible, key=lambda item: (float(item.get("planned_time_s", math.inf)), float(item.get("planned_delta_v_m_s", math.inf))))
    elif mode == "constrained":
        constrained = [
            item
            for item in feasible
            if _candidate_feasible(
                float(item.get("planned_delta_v_m_s", math.inf)),
                float(item.get("planned_time_s", math.inf)),
                max_delta_v_m_s,
                max_time_s,
            )
        ]
        if not constrained:
            return None
        selected = min(constrained, key=lambda item: float(item.get("score", math.inf)))
    else:
        selected = min(feasible, key=lambda item: (float(item.get("planned_delta_v_m_s", math.inf)), float(item.get("planned_time_s", math.inf))))
    return str(selected.get("candidate_id"))


def _slot_recovery_from_assessment(
    *,
    calc: Any,
    assessment_time_s: float,
    slot_tolerance_deg: float,
    max_phasing_orbits: int,
) -> dict[str, Any]:
    disturbed_period = float(calc.disturbed_period_s)
    reference_period = float(calc.reference_period_s)
    if disturbed_period <= 0.0 or reference_period <= 0.0:
        return {
            "slot_recovery_wait_time_s": None,
            "slot_recovery_total_time_s": None,
            "slot_recovery_orbits_from_assessment": None,
            "slot_recovery_phase_error_from_assessment_deg": None,
            "best_slot_wait_time_s": None,
            "best_slot_total_time_s": None,
            "best_slot_orbits_from_assessment": None,
            "best_slot_phase_error_from_assessment_deg": None,
        }
    elapsed = max(float(assessment_time_s), 0.0)
    min_orbit = max(1, int(math.ceil((elapsed - 1e-9) / disturbed_period)))
    max_orbit = max(int(max_phasing_orbits), min_orbit)
    reference_mean_motion = 2.0 * math.pi / reference_period
    best_orbit: int | None = None
    best_error = float("inf")
    best_wait: float | None = None
    found_orbit: int | None = None
    found_error: float | None = None
    found_wait: float | None = None
    for orbit_count in range(min_orbit, max_orbit + 1):
        total_time = float(orbit_count * disturbed_period)
        wait_time = total_time - elapsed
        if wait_time < -1e-6:
            continue
        phase_error = abs(_wrap_signed_deg(math.degrees(reference_mean_motion * total_time)))
        if phase_error < best_error:
            best_error = phase_error
            best_orbit = orbit_count
            best_wait = max(wait_time, 0.0)
        if found_orbit is None and phase_error <= float(slot_tolerance_deg):
            found_orbit = orbit_count
            found_error = phase_error
            found_wait = max(wait_time, 0.0)
            break
    best_total = None if best_orbit is None else float(best_orbit * disturbed_period)
    found_total = None if found_orbit is None else float(found_orbit * disturbed_period)
    return {
        "slot_recovery_wait_time_s": None if found_wait is None else float(found_wait),
        "slot_recovery_total_time_s": found_total,
        "slot_recovery_orbits_from_assessment": found_orbit,
        "slot_recovery_phase_error_from_assessment_deg": found_error,
        "best_slot_wait_time_s": None if best_wait is None else float(best_wait),
        "best_slot_total_time_s": best_total,
        "best_slot_orbits_from_assessment": best_orbit,
        "best_slot_phase_error_from_assessment_deg": None if not math.isfinite(best_error) else float(best_error),
    }


def _slot_tolerances(*, section: Any, tolerances: dict[str, float]) -> dict[str, float]:
    merged = dict(tolerances)
    merged.setdefault("slot_phase_deg", float(getattr(section, "slot_tolerance_deg", 1.0)))
    return merged


def _intrack_unit(state: np.ndarray) -> np.ndarray:
    r = np.asarray(state[:3], dtype=float).reshape(3)
    v = np.asarray(state[3:6], dtype=float).reshape(3)
    h = np.cross(r, v)
    radial_norm = float(np.linalg.norm(r))
    h_norm = float(np.linalg.norm(h))
    if radial_norm <= 0.0 or h_norm <= 0.0:
        v_norm = max(float(np.linalg.norm(v)), 1e-12)
        return v / v_norm
    radial = r / radial_norm
    cross_track = h / h_norm
    intrack = np.cross(cross_track, radial)
    norm = max(float(np.linalg.norm(intrack)), 1e-12)
    return intrack / norm


def _position_angle_error_deg(reference_state: np.ndarray, recovered_state: np.ndarray) -> float:
    ref = np.asarray(reference_state[:3], dtype=float).reshape(3)
    rec = np.asarray(recovered_state[:3], dtype=float).reshape(3)
    denom = max(float(np.linalg.norm(ref) * np.linalg.norm(rec)), 1e-12)
    unsigned = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(ref, rec) / denom)))))
    h = np.cross(ref, np.asarray(reference_state[3:6], dtype=float).reshape(3))
    sign = 1.0 if float(np.dot(h, np.cross(ref, rec))) >= 0.0 else -1.0
    return float(sign * unsigned)


def _burn_duration_s(
    *,
    delta_v_m_s: float,
    mass_kg: float | None,
    max_thrust_n: float | None,
) -> float | None:
    if mass_kg is None or max_thrust_n is None:
        return None
    acceleration_m_s2 = float(max_thrust_n) / float(mass_kg)
    if acceleration_m_s2 <= 0.0:
        return None
    return float(abs(float(delta_v_m_s)) / acceleration_m_s2)


def _propagate_state(state: np.ndarray, dt_s: float) -> np.ndarray:
    if dt_s <= 0.0:
        return np.asarray(state, dtype=float).reshape(6).copy()
    out = np.asarray(state, dtype=float).reshape(6).copy()
    max_step_s = 60.0
    steps = max(int(math.ceil(abs(float(dt_s)) / max_step_s)), 1)
    h = float(dt_s) / float(steps)
    zero = np.zeros(3, dtype=float)
    for _ in range(steps):
        out = propagate_two_body_rk4(out, h, EARTH_MU_KM3_S2, zero)
    return out


def _inferred_intrack_impulse(
    *,
    initial_elements: ClassicalOrbitalElements,
    final_elements: ClassicalOrbitalElements,
    final_state: np.ndarray,
) -> dict[str, float] | None:
    if initial_elements.ecc > 5e-3:
        return None
    h_angle_deg = _plane_angle_deg(initial_elements, final_state)
    if h_angle_deg > 1.0e-3:
        return None
    r0 = float(initial_elements.a_km)
    perigee = final_elements.a_km * (1.0 - final_elements.ecc)
    apogee = final_elements.a_km * (1.0 + final_elements.ecc)
    apsis_errors = {"perigee": abs(perigee - r0), "apogee": abs(apogee - r0)}
    apsis, error = min(apsis_errors.items(), key=lambda item: item[1])
    if error > max(1.0, 5e-3 * r0):
        return None
    disturbed_speed = math.sqrt(EARTH_MU_KM3_S2 * (2.0 / r0 - 1.0 / final_elements.a_km))
    circular_speed = math.sqrt(EARTH_MU_KM3_S2 / r0)
    sign = 1.0 if apsis == "perigee" else -1.0
    return {
        "disturbance_delta_v_m_s": float(sign * abs(disturbed_speed - circular_speed) * 1000.0),
        "apsis_error_km": float(error),
        "plane_angle_deg": float(h_angle_deg),
    }


def _plane_angle_deg(initial_elements: ClassicalOrbitalElements, final_state: np.ndarray) -> float:
    initial_target = coes_target_state_at_current_true_anomaly(_elements_dict(initial_elements), final_state)
    h0 = np.cross(initial_target[:3], initial_target[3:6])
    hf = np.cross(final_state[:3], final_state[3:6])
    denom = max(float(np.linalg.norm(h0) * np.linalg.norm(hf)), 1e-12)
    return float(math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(h0, hf) / denom))))))


def _time_to_apsis_s(elements: ClassicalOrbitalElements, apsis: str) -> float | None:
    if elements.ecc >= 1.0 or elements.a_km <= 0.0:
        return None
    target_nu = math.pi if apsis == "apogee" else 0.0
    current_nu = math.radians(elements.true_anomaly_deg)
    e = float(elements.ecc)
    current_m = _mean_anomaly_from_true(current_nu, e)
    target_m = _mean_anomaly_from_true(target_nu, e)
    delta_m = (target_m - current_m) % (2.0 * math.pi)
    if delta_m > 2.0 * math.pi - 1e-9:
        delta_m = 0.0
    mean_motion = math.sqrt(EARTH_MU_KM3_S2 / (elements.a_km**3))
    return float(delta_m / mean_motion)


def _mean_anomaly_from_true(nu_rad: float, ecc: float) -> float:
    if ecc <= 1e-12:
        return float(nu_rad % (2.0 * math.pi))
    e_anom = math.atan2(math.sqrt(1.0 - ecc * ecc) * math.sin(nu_rad), ecc + math.cos(nu_rad))
    return float((e_anom - ecc * math.sin(e_anom)) % (2.0 * math.pi))


def _propellant_for_delta_v(
    *,
    delta_v_m_s: float,
    mass_kg: float | None,
    isp_s: float | None,
) -> tuple[float | None, float | None]:
    if mass_kg is None or isp_s is None:
        return None, None
    mass_ratio = rocket_equation_mass_ratio(delta_v_m_s=delta_v_m_s, isp_s=isp_s)
    return float(float(mass_kg) * mass_ratio.propellant_fraction), float(mass_ratio.propellant_fraction)


def _element_errors(
    initial: ClassicalOrbitalElements,
    final: ClassicalOrbitalElements,
    initial_state: np.ndarray,
    final_state: np.ndarray,
) -> dict[str, float]:
    return {
        "a_km": float(final.a_km - initial.a_km),
        "ecc": float(final.ecc - initial.ecc),
        "inc_deg": _wrap_signed_deg(final.inc_deg - initial.inc_deg),
        "raan_deg": _wrap_signed_deg(final.raan_deg - initial.raan_deg),
        "argp_deg": _wrap_signed_deg(final.argp_deg - initial.argp_deg),
        "true_anomaly_deg": _wrap_signed_deg(final.true_anomaly_deg - initial.true_anomaly_deg),
        "position_norm_error_km": float(np.linalg.norm(final_state[:3]) - np.linalg.norm(initial_state[:3])),
        "speed_error_m_s": float((np.linalg.norm(final_state[3:6]) - np.linalg.norm(initial_state[3:6])) * 1000.0),
    }


def _within_tolerances(errors: dict[str, float], tolerances: dict[str, float]) -> bool | None:
    if not tolerances:
        return None
    for key, tolerance in tolerances.items():
        if abs(float(errors.get(key, 0.0))) > float(tolerance):
            return False
    return True


def _elements_dict(elements: ClassicalOrbitalElements) -> dict[str, float]:
    return {
        "a_km": float(elements.a_km),
        "ecc": float(elements.ecc),
        "inc_deg": float(elements.inc_deg),
        "raan_deg": float(elements.raan_deg),
        "argp_deg": float(elements.argp_deg),
        "true_anomaly_deg": float(elements.true_anomaly_deg),
    }


def _finite_positive(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) and out > 0.0 else None


def _finite_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _pad_axis(ax: Any, axis: str, values: list[float], *, min_span: float) -> None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return
    lo = min(finite)
    hi = max(finite)
    span = max(hi - lo, float(min_span))
    pad = max(span * 0.18, float(min_span) * 0.2)
    lower = lo - pad
    upper = hi + pad
    if axis == "x":
        ax.set_xlim(max(0.0, lower), upper)
    else:
        ax.set_ylim(max(0.0, lower), upper)


def _wrap_signed_deg(value: float) -> float:
    return float((float(value) + 180.0) % 360.0 - 180.0)


def _unavailable_summary(*, section: Any, object_id: str, notes: list[str]) -> dict[str, Any]:
    return {
        "enabled": True,
        "object_id": object_id,
        "goal": str(getattr(section, "goal", "orbit_shape") or "orbit_shape"),
        "recovery_estimate": {
            "available": False,
            "method": "unavailable",
            "recovery_delta_v_m_s": None,
            "recovery_time_s": None,
            "propellant_kg": None,
            "notes": notes,
        },
    }
