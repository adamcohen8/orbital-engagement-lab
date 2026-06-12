from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from sim.config import default_reference_object_id, object_section
from sim.dynamics.orbit.elements import (
    ClassicalOrbitalElements,
    coes_target_state_at_current_true_anomaly,
    rv_to_coe_eci,
)
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2, EARTH_RADIUS_KM
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
    mass_kg = _resolve_mass_kg(cfg=cfg, object_id=object_id, section=section, hist=hist, idx=assessment_idx)
    isp_s = _resolve_isp_s(cfg=cfg, object_id=object_id, section=section)
    max_thrust_n = _resolve_max_thrust_n(cfg=cfg, object_id=object_id, section=section)
    element_errors = _element_errors(initial_elements, final_elements, initial_state, final_state)
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
        "final_elements": _elements_dict(final_elements),
        "element_errors": element_errors,
        "element_tolerances": tolerances,
        "within_element_tolerances": _within_tolerances(element_errors, tolerances),
        "mass_kg": mass_kg,
        "isp_s": isp_s,
        "max_thrust_n": max_thrust_n,
        "recovery_estimate": recovery,
    }
    planner = _build_recovery_planner(
        section=section,
        goal=goal,
        assessment_time_s=float(times[assessment_idx]),
        initial_state=initial_state,
        final_state=final_state,
        initial_elements=initial_elements,
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
    for row in rows:
        if row["verified"]:
            color = palette["verified"]
            marker = "o"
            label = "Verified"
        elif row["feasible"]:
            color = palette["feasible"]
            marker = "o"
            label = "Feasible"
        else:
            color = palette["infeasible"]
            marker = "x"
            label = "Infeasible"
        scatter_kwargs: dict[str, Any] = {
            "s": 110,
            "marker": marker,
            "color": color,
            "linewidths": 0.9,
            "alpha": 0.92,
            "zorder": 3,
        }
        if marker != "x":
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
        else:
            label = row["candidate_id"]
        ax.annotate(
            label,
            (row["time_min"], row["delta_v_m_s"]),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=8.5,
            color="#d7deea",
        )

    max_time_s = _finite_or_none(planner.get("max_recovery_time_s"))
    if max_time_s is not None and max_time_s > 0.0:
        ax.axvline(max_time_s / 60.0, color="#6f6f6f", linestyle="--", linewidth=1.0, alpha=0.75)
        ax.text(max_time_s / 60.0, ax.get_ylim()[1], "max time", ha="right", va="top", fontsize=8, color="#6f6f6f")
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
    ax.set_title("Mission Recovery Trade Space")
    ax.set_xlabel("Planned recovery wait time (min)")
    ax.set_ylabel("Planned recovery delta-V (m/s)")
    subtitle = " / ".join(part for part in (object_id, goal) if part)
    if subtitle:
        ax.text(0.0, 1.01, subtitle, transform=ax.transAxes, fontsize=9.5, color="#666666", va="bottom")
    ax.grid(True, alpha=0.25)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="w", markerfacecolor=palette["verified"], label="Verified", markersize=8),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=palette["feasible"], label="Feasible", markersize=8),
            Line2D([0], [0], marker="x", color=palette["infeasible"], label="Infeasible", markersize=8),
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
    initial_elements: ClassicalOrbitalElements,
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
    )
    recommendations = {
        mode: _recommended_candidate_id(candidates, mode=mode, max_delta_v_m_s=max_delta_v_m_s, max_time_s=max_time_s)
        for mode in modes
    }
    return {
        "enabled": True,
        "modes": modes,
        "max_recovery_time_s": max_time_s,
        "max_recovery_delta_v_m_s": max_delta_v_m_s,
        "candidate_count": len(candidates),
        "simulate_candidates": simulate_candidates,
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
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    initial_guess = _candidate_from_state_match(
        candidate_id="candidate_001",
        source="immediate_local_velocity_match",
        description="Immediate burn that matches the initial orbit shape at the assessment position.",
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
                    description="Coast, then burn to match the initial orbit shape at the recovery position.",
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

    unique: dict[tuple[str, float], dict[str, Any]] = {}
    for candidate in candidates:
        key = (str(candidate.get("source", "")), round(float(candidate.get("planned_time_s", 0.0)), 6))
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
    return [
        {**candidate, "candidate_id": f"candidate_{idx:03d}"}
        for idx, candidate in enumerate(ranked[:candidate_count], start=1)
    ]


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
) -> str | None:
    feasible = [item for item in candidates if bool(item.get("feasible", False))]
    if not feasible:
        return None
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
