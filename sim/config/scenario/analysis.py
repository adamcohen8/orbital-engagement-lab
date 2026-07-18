from __future__ import annotations

import math
from typing import Any

from sim.config.scenario.models import (
    AnalysisBaselineSection,
    AnalysisExecutionSection,
    AnalysisMonteCarloSection,
    AnalysisSection,
    CovarianceCollisionScreeningSection,
    CovarianceFiniteDifferenceSection,
    CovarianceObjectSection,
    CovariancePairSection,
    CovarianceProcessNoiseSection,
    CovarianceSection,
    MissionRecoverySection,
    MonteCarloSection,
    MonteCarloVariation,
    SensitivityParameter,
    SensitivitySection,
)
from sim.config.scenario.primitives import (
    _as_dict,
    _parse_bool,
    _parse_float,
    _parse_optional_float,
    _reject_unknown_fields,
)

__all__ = [
    '_parse_mc_variation',
    '_parse_analysis_execution_section',
    '_parse_analysis_baseline_section',
    '_parse_analysis_monte_carlo_section',
    '_parse_sensitivity_parameter',
    '_parse_sensitivity_section',
    '_parse_covariance_matrix',
    '_parse_covariance_diagonal',
    '_parse_covariance_object_section',
    '_parse_covariance_collision_screening_section',
    '_parse_covariance_pair_section',
    '_parse_covariance_section',
    '_parse_mission_recovery_section',
    '_parse_mission_recovery_target_orbit_section',
    '_parse_mission_recovery_planner_section',
    '_parse_orbit_transfer_planner_section',
    '_parse_orbital_delivery_section',
    '_parse_analysis_section',
    '_monte_carlo_from_analysis',
]

def _parse_mc_variation(value: Any) -> MonteCarloVariation:
    d = _as_dict(value, "monte_carlo.variation")
    path = d.get("parameter_path")
    if not isinstance(path, str) or not path:
        raise ValueError("monte_carlo.variations[*].parameter_path must be a non-empty string.")
    return MonteCarloVariation(
        parameter_path=path,
        mode=str(d.get("mode", "choice")),
        options=list(d.get("options", []) or []),
        low=float(d["low"]) if d.get("low") is not None else None,
        high=float(d["high"]) if d.get("high") is not None else None,
        mean=float(d["mean"]) if d.get("mean") is not None else None,
        std=float(d["std"]) if d.get("std") is not None else None,
    )


def _parse_analysis_execution_section(value: Any) -> AnalysisExecutionSection:
    d = _as_dict(value, "analysis.execution")
    out = AnalysisExecutionSection(
        parallel_enabled=_parse_bool(
            d.get("parallel_enabled", False),
            "analysis.execution.parallel_enabled",
        ),
        parallel_workers=int(d.get("parallel_workers", 0)),
        failure_policy=str(d.get("failure_policy", "fail_fast") or "fail_fast").strip().lower(),
    )
    if out.parallel_workers < 0:
        raise ValueError("analysis.execution.parallel_workers must be >= 0.")
    if out.failure_policy not in {"fail_fast", "continue"}:
        raise ValueError("analysis.execution.failure_policy must be one of: fail_fast, continue.")
    return out


def _parse_analysis_baseline_section(value: Any) -> AnalysisBaselineSection:
    d = _as_dict(value, "analysis.baseline")
    summary_json = str(d.get("summary_json", "") or "")
    enabled = _parse_bool(d.get("enabled", False), "analysis.baseline.enabled")
    raw_mode = str(d.get("mode", "") or "").strip().lower()
    if not raw_mode:
        raw_mode = "file" if summary_json else ("run" if enabled else "none")
    if raw_mode not in {"none", "run", "file"}:
        raise ValueError("analysis.baseline.mode must be one of: none, run, file.")
    if raw_mode == "file" and not summary_json:
        raise ValueError("analysis.baseline.summary_json is required when mode is 'file'.")
    return AnalysisBaselineSection(
        enabled=bool(enabled or raw_mode in {"run", "file"}),
        mode=raw_mode,
        summary_json=summary_json,
    )


def _parse_analysis_monte_carlo_section(value: Any) -> AnalysisMonteCarloSection:
    d = _as_dict(value, "analysis.monte_carlo")
    vars_raw = d.get("variations")
    if vars_raw is None:
        variations = []
    else:
        if not isinstance(vars_raw, list):
            raise ValueError("analysis.monte_carlo.variations must be a list.")
        variations = [_parse_mc_variation(v) for v in vars_raw]
    out = AnalysisMonteCarloSection(
        iterations=int(d.get("iterations", 1)),
        base_seed=int(d.get("base_seed", 0)),
        variations=variations,
    )
    if out.iterations <= 0:
        raise ValueError("analysis.monte_carlo.iterations must be positive.")
    return out


def _parse_sensitivity_parameter(value: Any) -> SensitivityParameter:
    d = _as_dict(value, "analysis.sensitivity.parameter")
    path = d.get("parameter_path", d.get("path"))
    if not isinstance(path, str) or not path:
        raise ValueError("analysis.sensitivity.parameters[*].parameter_path must be a non-empty string.")
    values = d.get("values", [])
    if not isinstance(values, list):
        raise ValueError("analysis.sensitivity.parameters[*].values must be a list.")
    distribution = str(d.get("distribution", "uniform")).strip().lower()
    if distribution not in {"uniform", "normal"}:
        raise ValueError("analysis.sensitivity.parameters[*].distribution must be one of: uniform, normal.")
    return SensitivityParameter(
        parameter_path=path,
        values=list(values),
        distribution=distribution,
        low=float(d["low"]) if d.get("low") is not None else None,
        high=float(d["high"]) if d.get("high") is not None else None,
        mean=float(d["mean"]) if d.get("mean") is not None else None,
        std=float(d["std"]) if d.get("std") is not None else None,
    )


def _parse_sensitivity_section(value: Any) -> SensitivitySection:
    d = _as_dict(value, "analysis.sensitivity")
    params_raw = d.get("parameters", []) or []
    if not isinstance(params_raw, list):
        raise ValueError("analysis.sensitivity.parameters must be a list.")
    out = SensitivitySection(
        method=str(d.get("method", "one_at_a_time")),
        samples=int(d.get("samples", 0)),
        seed=int(d.get("seed", 0)),
        parameters=[_parse_sensitivity_parameter(v) for v in params_raw],
    )
    if out.method not in {"one_at_a_time", "lhs", "two_parameter_grid"}:
        raise ValueError("analysis.sensitivity.method must be one of: one_at_a_time, lhs, two_parameter_grid.")
    if out.samples < 0:
        raise ValueError("analysis.sensitivity.samples must be >= 0.")
    return out


def _parse_covariance_matrix(value: Any, field_name: str) -> list[list[float]]:
    if value in (None, "", []):
        return []
    if not isinstance(value, list) or len(value) != 6:
        raise ValueError(f"{field_name} must be a 6x6 list.")
    rows: list[list[float]] = []
    for i, row_raw in enumerate(value):
        if not isinstance(row_raw, list) or len(row_raw) != 6:
            raise ValueError(f"{field_name}[{i}] must contain 6 values.")
        rows.append([_parse_float(item, f"{field_name}[{i}]") for item in row_raw])
    return rows


def _parse_covariance_diagonal(value: Any, field_name: str) -> list[float]:
    if value in (None, "", []):
        return []
    if not isinstance(value, list) or len(value) != 6:
        raise ValueError(f"{field_name} must contain 6 values.")
    out = [_parse_float(item, f"{field_name}[{i}]") for i, item in enumerate(value)]
    if any(item < 0.0 for item in out):
        raise ValueError(f"{field_name} values must be >= 0.")
    return out


def _parse_covariance_object_section(value: Any, path: str) -> CovarianceObjectSection:
    d = _as_dict(value, path)
    frame = str(d.get("frame", "eci") or "eci").strip().lower()
    if frame != "eci":
        raise ValueError(f"{path}.frame must be 'eci' for covariance analysis v0.")
    matrix_raw = d.get("covariance", d.get("matrix"))
    diagonal_raw = d.get("diagonal")
    if isinstance(matrix_raw, dict):
        diagonal_raw = matrix_raw.get("diagonal", diagonal_raw)
        matrix_raw = matrix_raw.get("matrix")
    covariance = _parse_covariance_matrix(matrix_raw, f"{path}.covariance")
    diagonal = _parse_covariance_diagonal(diagonal_raw, f"{path}.diagonal")
    position_sigma = _parse_optional_float(d.get("position_sigma_km"), f"{path}.position_sigma_km")
    velocity_sigma = _parse_optional_float(d.get("velocity_sigma_km_s"), f"{path}.velocity_sigma_km_s")
    if position_sigma is not None and position_sigma < 0.0:
        raise ValueError(f"{path}.position_sigma_km must be >= 0.")
    if velocity_sigma is not None and velocity_sigma < 0.0:
        raise ValueError(f"{path}.velocity_sigma_km_s must be >= 0.")
    enabled = _parse_bool(d.get("enabled", True), f"{path}.enabled")
    if enabled and not covariance and not diagonal and position_sigma is None and velocity_sigma is None:
        raise ValueError(f"{path} must define covariance, diagonal, position_sigma_km, or velocity_sigma_km_s.")
    return CovarianceObjectSection(
        enabled=enabled,
        frame=frame,
        covariance=covariance,
        diagonal=diagonal,
        position_sigma_km=position_sigma,
        velocity_sigma_km_s=velocity_sigma,
    )


def _parse_covariance_collision_screening_section(value: Any, path: str) -> CovarianceCollisionScreeningSection:
    if isinstance(value, bool):
        d: dict[str, Any] = {"enabled": value}
    else:
        d = _as_dict(value, path)
    enabled = _parse_bool(d.get("enabled", False), f"{path}.enabled")
    method = str(d.get("method", "small_object") or "small_object").strip().lower()
    if method not in {"small_object"}:
        raise ValueError(f"{path}.method must be 'small_object'.")
    radius_km_raw = d.get("hard_body_radius_km")
    radius_m_raw = d.get("hard_body_radius_m")
    if radius_km_raw is not None and radius_m_raw is not None:
        raise ValueError(f"{path} must define only one of hard_body_radius_km or hard_body_radius_m.")
    if radius_km_raw is not None:
        radius_km = _parse_float(radius_km_raw, f"{path}.hard_body_radius_km")
    elif radius_m_raw is not None:
        radius_km = _parse_float(radius_m_raw, f"{path}.hard_body_radius_m") / 1000.0
    else:
        radius_km = 0.01
    if radius_km <= 0.0:
        raise ValueError(f"{path}.hard_body_radius must be positive.")
    return CovarianceCollisionScreeningSection(
        enabled=enabled,
        hard_body_radius_km=radius_km,
        method=method,
    )


def _parse_covariance_pair_section(value: Any, index: int) -> CovariancePairSection:
    d = _as_dict(value, f"analysis.covariance.pairs[{index}]")
    deputy = str(d.get("deputy_id", d.get("deputy", "")) or "").strip()
    chief = str(d.get("chief_id", d.get("chief", "")) or "").strip()
    if not deputy:
        raise ValueError(f"analysis.covariance.pairs[{index}].deputy_id is required.")
    if not chief:
        raise ValueError(f"analysis.covariance.pairs[{index}].chief_id is required.")
    if deputy == chief:
        raise ValueError(f"analysis.covariance.pairs[{index}] deputy_id and chief_id must differ.")
    collision = _parse_covariance_collision_screening_section(
        d.get("collision_screening", d.get("conjunction_screening")),
        f"analysis.covariance.pairs[{index}].collision_screening",
    )
    return CovariancePairSection(deputy_id=deputy, chief_id=chief, collision_screening=collision)


def _parse_covariance_section(value: Any) -> CovarianceSection:
    d = _as_dict(value, "analysis.covariance")
    objects_raw = d.get("objects", {}) or {}
    if not isinstance(objects_raw, dict):
        raise ValueError("analysis.covariance.objects must be a mapping.")
    pairs_raw = d.get("pairs", []) or []
    if not isinstance(pairs_raw, list):
        raise ValueError("analysis.covariance.pairs must be a list.")
    fd = _as_dict(d.get("finite_difference"), "analysis.covariance.finite_difference")
    pos_step = _parse_float(
        fd.get("position_step_km", 1e-3),
        "analysis.covariance.finite_difference.position_step_km",
    )
    vel_step = _parse_float(
        fd.get("velocity_step_km_s", 1e-6),
        "analysis.covariance.finite_difference.velocity_step_km_s",
    )
    if pos_step <= 0.0:
        raise ValueError("analysis.covariance.finite_difference.position_step_km must be positive.")
    if vel_step <= 0.0:
        raise ValueError("analysis.covariance.finite_difference.velocity_step_km_s must be positive.")
    process_noise = _as_dict(d.get("process_noise"), "analysis.covariance.process_noise")
    accel_sigma = _parse_float(
        process_noise.get("acceleration_sigma_km_s2", 0.0),
        "analysis.covariance.process_noise.acceleration_sigma_km_s2",
    )
    if accel_sigma < 0.0:
        raise ValueError("analysis.covariance.process_noise.acceleration_sigma_km_s2 must be >= 0.")
    objects = {
        str(object_id): _parse_covariance_object_section(obj, f"analysis.covariance.objects.{object_id}")
        for object_id, obj in objects_raw.items()
    }
    pairs = [_parse_covariance_pair_section(pair, idx) for idx, pair in enumerate(pairs_raw)]
    object_ids = set(objects)
    for pair in pairs:
        missing = [object_id for object_id in (pair.deputy_id, pair.chief_id) if object_id not in object_ids]
        if missing:
            raise ValueError(
                "analysis.covariance.pairs "
                f"{pair.deputy_id}:{pair.chief_id} requires covariance objects for both participants; "
                f"missing: {', '.join(missing)}."
            )
    return CovarianceSection(
        enabled=_parse_bool(d.get("enabled", True), "analysis.covariance.enabled"),
        objects=objects,
        pairs=pairs,
        finite_difference=CovarianceFiniteDifferenceSection(
            position_step_km=pos_step,
            velocity_step_km_s=vel_step,
        ),
        process_noise=CovarianceProcessNoiseSection(
            enabled=_parse_bool(process_noise.get("enabled", False), "analysis.covariance.process_noise.enabled"),
            acceleration_sigma_km_s2=accel_sigma,
        ),
        write_review_tables=_parse_bool(d.get("write_review_tables", True), "analysis.covariance.write_review_tables"),
    )


def _parse_mission_recovery_section(value: Any) -> MissionRecoverySection:
    d = _as_dict(value, "analysis.mission_recovery")
    goal = str(d.get("goal", d.get("recovery_goal", "orbit_shape")) or "orbit_shape").strip().lower()
    if goal not in {"orbit_shape", "orbit_slot"}:
        raise ValueError("analysis.mission_recovery.goal must be one of: orbit_shape, orbit_slot.")
    assessment_raw = d.get("assessment_time_s", "final")
    if isinstance(assessment_raw, str):
        assessment: float | str = assessment_raw.strip().lower()
        if assessment != "final":
            assessment = _parse_float(assessment_raw, "analysis.mission_recovery.assessment_time_s")
    else:
        assessment = _parse_float(assessment_raw, "analysis.mission_recovery.assessment_time_s")
    slot_tolerance = _parse_float(
        d.get("slot_tolerance_deg", 1.0),
        "analysis.mission_recovery.slot_tolerance_deg",
    )
    if slot_tolerance < 0.0:
        raise ValueError("analysis.mission_recovery.slot_tolerance_deg must be >= 0.")
    max_phasing_orbits = int(d.get("max_phasing_orbits", 5000))
    if max_phasing_orbits < 1:
        raise ValueError("analysis.mission_recovery.max_phasing_orbits must be at least 1.")
    target_orbit_raw = d.get("target_orbit", d.get("desired_orbit", {})) or {}
    if not isinstance(target_orbit_raw, dict):
        raise ValueError("analysis.mission_recovery.target_orbit must be a mapping.")
    target_orbit = _parse_mission_recovery_target_orbit_section(target_orbit_raw)
    planner = _parse_mission_recovery_planner_section(
        d.get("planner"),
        configured_target_orbit=bool(target_orbit),
    )
    propulsion = dict(d.get("propulsion", {}) or {})
    if propulsion.get("isp_s") is not None and _parse_float(propulsion.get("isp_s"), "analysis.mission_recovery.propulsion.isp_s") <= 0.0:
        raise ValueError("analysis.mission_recovery.propulsion.isp_s must be positive.")
    if propulsion.get("spacecraft_mass_kg") is not None and _parse_float(
        propulsion.get("spacecraft_mass_kg"),
        "analysis.mission_recovery.propulsion.spacecraft_mass_kg",
    ) <= 0.0:
        raise ValueError("analysis.mission_recovery.propulsion.spacecraft_mass_kg must be positive.")
    if propulsion.get("max_thrust_n") is not None and _parse_float(
        propulsion.get("max_thrust_n"),
        "analysis.mission_recovery.propulsion.max_thrust_n",
    ) <= 0.0:
        raise ValueError("analysis.mission_recovery.propulsion.max_thrust_n must be positive.")
    tolerances_raw = d.get("element_tolerances", {}) or {}
    if not isinstance(tolerances_raw, dict):
        raise ValueError("analysis.mission_recovery.element_tolerances must be a mapping.")
    tolerances = {
        str(key): _parse_float(value, f"analysis.mission_recovery.element_tolerances.{key}")
        for key, value in tolerances_raw.items()
    }
    if any(value < 0.0 for value in tolerances.values()):
        raise ValueError("analysis.mission_recovery.element_tolerances values must be >= 0.")
    return MissionRecoverySection(
        enabled=_parse_bool(d.get("enabled", False), "analysis.mission_recovery.enabled"),
        object_id=str(d.get("object_id", "") or "").strip(),
        goal=goal,
        assessment_time_s=assessment,
        slot_tolerance_deg=slot_tolerance,
        max_phasing_orbits=max_phasing_orbits,
        planner=planner,
        propulsion=propulsion,
        element_tolerances=tolerances,
        target_orbit=target_orbit,
    )


def _parse_mission_recovery_target_orbit_section(value: dict[str, Any]) -> dict[str, Any]:
    d = dict(value or {})
    if not d:
        return {}
    coes_raw = d.get("coes", d)
    if not isinstance(coes_raw, dict):
        raise ValueError("analysis.mission_recovery.target_orbit.coes must be a mapping.")
    out: dict[str, float] = {}
    aliases = {
        "a_km": ("a_km", "semi_major_axis_km"),
        "ecc": ("ecc", "e"),
        "inc_deg": ("inc_deg", "inclination_deg"),
        "raan_deg": ("raan_deg",),
        "argp_deg": ("argp_deg", "arg_periapsis_deg"),
        "true_anomaly_deg": ("true_anomaly_deg", "ta_deg"),
    }
    for canonical, keys in aliases.items():
        for key in keys:
            if key in coes_raw:
                out[canonical] = _parse_float(
                    coes_raw[key],
                    f"analysis.mission_recovery.target_orbit.coes.{canonical}",
                )
                break
    if "a_km" in out and out["a_km"] <= 0.0:
        raise ValueError("analysis.mission_recovery.target_orbit.coes.a_km must be positive.")
    if "ecc" in out and not (0.0 <= out["ecc"] < 1.0):
        raise ValueError("analysis.mission_recovery.target_orbit.coes.ecc must satisfy 0 <= ecc < 1.")
    return {"coes": out}


def _parse_mission_recovery_planner_section(
    value: Any,
    *,
    configured_target_orbit: bool = False,
) -> dict[str, Any]:
    d = _as_dict(value, "analysis.mission_recovery.planner")
    enabled = _parse_bool(d.get("enabled", False), "analysis.mission_recovery.planner.enabled")
    default_sources = (
        ["analytic_reconstitution", "orbit_transfer"]
        if configured_target_orbit
        else ["analytic_reconstitution"]
    )
    raw_sources = d.get("sources", d.get("source", default_sources))
    if isinstance(raw_sources, str):
        sources = [raw_sources]
    else:
        sources = list(raw_sources or [])
    sources = [str(source).strip().lower() for source in sources if str(source).strip()]
    if not sources:
        sources = list(default_sources)
    source_aliases = {
        "existing": "analytic_reconstitution",
        "legacy": "analytic_reconstitution",
        "lambert": "orbit_transfer",
        "orbit_transfer_planner": "orbit_transfer",
    }
    sources = [source_aliases.get(source, source) for source in sources]
    sources = list(dict.fromkeys(sources))
    allowed_sources = {"analytic_reconstitution", "orbit_transfer"}
    invalid_sources = [source for source in sources if source not in allowed_sources]
    if invalid_sources:
        raise ValueError(
            "analysis.mission_recovery.planner.sources must contain only: "
            "analytic_reconstitution, orbit_transfer."
        )
    raw_modes = d.get("modes", d.get("mode", ["min_delta_v", "min_time", "constrained"]))
    if isinstance(raw_modes, str):
        modes = [raw_modes]
    else:
        modes = list(raw_modes or [])
    modes = [str(mode).strip().lower() for mode in modes if str(mode).strip()]
    if not modes:
        modes = ["min_delta_v", "min_time", "constrained"]
    allowed_modes = {"min_delta_v", "min_time", "constrained"}
    invalid = [mode for mode in modes if mode not in allowed_modes]
    if invalid:
        raise ValueError(
            "analysis.mission_recovery.planner.modes must contain only: constrained, min_delta_v, min_time."
        )
    max_recovery_time_s = _parse_float(
        d.get("max_recovery_time_s", 86400.0),
        "analysis.mission_recovery.planner.max_recovery_time_s",
    )
    if max_recovery_time_s < 0.0:
        raise ValueError("analysis.mission_recovery.planner.max_recovery_time_s must be >= 0.")
    max_recovery_delta_v_m_s = d.get("max_recovery_delta_v_m_s")
    max_recovery_delta_v = (
        None
        if max_recovery_delta_v_m_s in (None, "")
        else _parse_float(
            max_recovery_delta_v_m_s,
            "analysis.mission_recovery.planner.max_recovery_delta_v_m_s",
        )
    )
    if max_recovery_delta_v is not None and max_recovery_delta_v < 0.0:
        raise ValueError("analysis.mission_recovery.planner.max_recovery_delta_v_m_s must be >= 0.")
    candidate_count = int(d.get("candidate_count", 12))
    if candidate_count < 1:
        raise ValueError("analysis.mission_recovery.planner.candidate_count must be at least 1.")
    simulate_candidates = _parse_bool(
        d.get("simulate_candidates", True),
        "analysis.mission_recovery.planner.simulate_candidates",
    )
    orbit_transfer = _parse_orbit_transfer_planner_section(
        d.get("orbit_transfer", d.get("lambert")),
        parent_enabled=enabled,
        sources=sources,
    )
    return {
        "enabled": enabled,
        "sources": sources,
        "modes": modes,
        "max_recovery_time_s": max_recovery_time_s,
        "max_recovery_delta_v_m_s": max_recovery_delta_v,
        "candidate_count": candidate_count,
        "simulate_candidates": simulate_candidates,
        "orbit_transfer": orbit_transfer,
    }


def _parse_orbit_transfer_planner_section(
    value: Any,
    *,
    parent_enabled: bool,
    sources: list[str],
) -> dict[str, Any]:
    d = _as_dict(value, "analysis.mission_recovery.planner.orbit_transfer")
    default_enabled = bool(parent_enabled and "orbit_transfer" in set(sources))
    enabled = _parse_bool(
        d.get("enabled", default_enabled),
        "analysis.mission_recovery.planner.orbit_transfer.enabled",
    )
    departure_samples = int(d.get("departure_samples", 9))
    if departure_samples < 1:
        raise ValueError("analysis.mission_recovery.planner.orbit_transfer.departure_samples must be at least 1.")
    time_of_flight_samples = int(d.get("time_of_flight_samples", 12))
    if time_of_flight_samples < 1:
        raise ValueError("analysis.mission_recovery.planner.orbit_transfer.time_of_flight_samples must be at least 1.")
    target_anomaly_samples = int(d.get("target_anomaly_samples", 24))
    if target_anomaly_samples < 1:
        raise ValueError("analysis.mission_recovery.planner.orbit_transfer.target_anomaly_samples must be at least 1.")
    min_tof = _parse_float(
        d.get("min_time_of_flight_s", 60.0),
        "analysis.mission_recovery.planner.orbit_transfer.min_time_of_flight_s",
    )
    if min_tof <= 0.0:
        raise ValueError("analysis.mission_recovery.planner.orbit_transfer.min_time_of_flight_s must be positive.")
    max_tof_raw = d.get("max_time_of_flight_s")
    max_tof = None if max_tof_raw in (None, "") else _parse_float(
        max_tof_raw,
        "analysis.mission_recovery.planner.orbit_transfer.max_time_of_flight_s",
    )
    if max_tof is not None and max_tof <= 0.0:
        raise ValueError("analysis.mission_recovery.planner.orbit_transfer.max_time_of_flight_s must be positive.")
    if max_tof is not None and max_tof < min_tof:
        raise ValueError(
            "analysis.mission_recovery.planner.orbit_transfer.max_time_of_flight_s "
            "must be >= min_time_of_flight_s."
        )
    multi_revolution_max = int(d.get("multi_revolution_max", 0))
    if multi_revolution_max < 0:
        raise ValueError("analysis.mission_recovery.planner.orbit_transfer.multi_revolution_max must be >= 0.")
    if multi_revolution_max > 0:
        raise ValueError(
            "analysis.mission_recovery.planner.orbit_transfer.multi_revolution_max "
            "must be 0 until multi-revolution Lambert transfers are supported."
        )
    impulse_epsilon_m_s = _parse_float(
        d.get("impulse_epsilon_m_s", 1.0e-2),
        "analysis.mission_recovery.planner.orbit_transfer.impulse_epsilon_m_s",
    )
    if impulse_epsilon_m_s < 0.0:
        raise ValueError("analysis.mission_recovery.planner.orbit_transfer.impulse_epsilon_m_s must be >= 0.")
    return {
        "enabled": enabled,
        "departure_samples": departure_samples,
        "time_of_flight_samples": time_of_flight_samples,
        "target_anomaly_samples": target_anomaly_samples,
        "min_time_of_flight_s": min_tof,
        "max_time_of_flight_s": max_tof,
        "short_way": _parse_bool(d.get("short_way", True), "analysis.mission_recovery.planner.orbit_transfer.short_way"),
        "long_way": _parse_bool(d.get("long_way", False), "analysis.mission_recovery.planner.orbit_transfer.long_way"),
        "multi_revolution_max": multi_revolution_max,
        "impulse_epsilon_m_s": impulse_epsilon_m_s,
        "keep_per_time_best": _parse_bool(
            d.get("keep_per_time_best", True),
            "analysis.mission_recovery.planner.orbit_transfer.keep_per_time_best",
        ),
    }


def _parse_orbital_delivery_section(value: Any) -> dict[str, Any]:
    d = _as_dict(value, "analysis.orbital_delivery")
    if not d:
        return {}
    _reject_unknown_fields(
        d,
        "analysis.orbital_delivery",
        {"enabled", "deployed_object_id", "reference_object_id", "target", "feasibility"},
    )
    enabled = _parse_bool(d.get("enabled", True), "analysis.orbital_delivery.enabled")
    deployed_object_id = str(d.get("deployed_object_id", "") or "").strip()
    reference_object_id = str(d.get("reference_object_id", "") or "").strip()
    target = _as_dict(d.get("target"), "analysis.orbital_delivery.target")
    _reject_unknown_fields(target, "analysis.orbital_delivery.target", {"frame", "state", "coes", "anomaly_policy"})
    if enabled and not deployed_object_id:
        raise ValueError("analysis.orbital_delivery.deployed_object_id is required when enabled.")
    if enabled and not target:
        raise ValueError("analysis.orbital_delivery.target is required when enabled.")
    if not target:
        return {
            "enabled": enabled,
            "deployed_object_id": deployed_object_id,
            "reference_object_id": reference_object_id,
            "target": {},
            "feasibility": {},
        }
    frame = str(target.get("frame", "eci") or "eci").strip().lower()
    if frame not in {"eci", "coes", "relative_ric"}:
        raise ValueError("analysis.orbital_delivery.target.frame must be one of: eci, coes, relative_ric.")
    if frame == "eci":
        state = target.get("state")
        if not isinstance(state, list) or len(state) != 6 or not all(math.isfinite(float(x)) for x in state):
            raise ValueError("analysis.orbital_delivery.target.state must be a finite length-6 ECI state.")
    elif frame == "coes":
        coes = target.get("coes", {}) or {}
        required = {"a_km", "ecc", "inc_deg", "raan_deg", "argp_deg"}
        if not isinstance(coes, dict) or not required.issubset(coes):
            raise ValueError(
                "analysis.orbital_delivery.target.coes must define a_km, ecc, inc_deg, raan_deg, and argp_deg."
            )
        anomaly_policy = str(target.get("anomaly_policy", "configured") or "configured").strip().lower()
        if anomaly_policy not in {"configured", "match_actual"}:
            raise ValueError(
                "analysis.orbital_delivery.target.anomaly_policy must be 'configured' or 'match_actual'."
            )
        if anomaly_policy == "configured" and not any(key in coes for key in ("true_anomaly_deg", "ta_deg")):
            raise ValueError(
                "analysis.orbital_delivery.target.coes requires true_anomaly_deg when anomaly_policy is configured."
            )
    else:
        state = target.get("state")
        if not isinstance(state, list) or len(state) != 6 or not all(math.isfinite(float(x)) for x in state):
            raise ValueError("analysis.orbital_delivery.target.state must be a finite length-6 RIC state.")
        if not reference_object_id:
            raise ValueError("analysis.orbital_delivery.reference_object_id is required for a relative_ric target.")
    feasibility = _as_dict(d.get("feasibility"), "analysis.orbital_delivery.feasibility")
    allowed_feasibility = {
        "max_position_error_km",
        "max_velocity_error_m_s",
        "max_correction_dv_m_s",
        "max_abs_a_error_km",
        "max_abs_ecc_error",
        "max_abs_inc_error_deg",
        "max_abs_raan_error_deg",
    }
    _reject_unknown_fields(feasibility, "analysis.orbital_delivery.feasibility", allowed_feasibility)
    normalized_feasibility = {
        str(key): _parse_float(raw, f"analysis.orbital_delivery.feasibility.{key}")
        for key, raw in feasibility.items()
    }
    if any(value < 0.0 for value in normalized_feasibility.values()):
        raise ValueError("analysis.orbital_delivery.feasibility thresholds must be non-negative.")
    return {
        "enabled": enabled,
        "deployed_object_id": deployed_object_id,
        "reference_object_id": reference_object_id,
        "target": dict(target),
        "feasibility": normalized_feasibility,
    }


def _parse_analysis_section(value: Any) -> AnalysisSection:
    d = _as_dict(value, "analysis")
    _reject_unknown_fields(
        d,
        "analysis",
        {
            "enabled",
            "study_type",
            "execution",
            "metrics",
            "baseline",
            "monte_carlo",
            "sensitivity",
            "covariance",
            "mission_recovery",
            "orbital_delivery",
        },
    )
    metrics = d.get("metrics", []) or []
    if not isinstance(metrics, list):
        raise ValueError("analysis.metrics must be a list.")
    out = AnalysisSection(
        enabled=_parse_bool(d.get("enabled", False), "analysis.enabled"),
        study_type=str(d.get("study_type", "monte_carlo")).strip().lower(),
        execution=_parse_analysis_execution_section(d.get("execution")),
        metrics=list(metrics),
        baseline=_parse_analysis_baseline_section(d.get("baseline")),
        monte_carlo=_parse_analysis_monte_carlo_section(d.get("monte_carlo")),
        sensitivity=_parse_sensitivity_section(d.get("sensitivity")),
        covariance=_parse_covariance_section(d.get("covariance")),
        mission_recovery=_parse_mission_recovery_section(d.get("mission_recovery")),
        orbital_delivery=_parse_orbital_delivery_section(d.get("orbital_delivery")),
    )
    if out.study_type not in {"monte_carlo", "sensitivity", "covariance"}:
        raise ValueError("analysis.study_type must be one of: monte_carlo, sensitivity, covariance.")
    if out.enabled and out.study_type == "covariance" and not out.covariance.objects:
        raise ValueError("analysis.covariance.objects must define at least one object when study_type is covariance.")
    return out


def _monte_carlo_from_analysis(analysis: AnalysisSection) -> MonteCarloSection:
    if analysis.enabled and analysis.study_type == "monte_carlo":
        return MonteCarloSection(
            enabled=True,
            iterations=int(analysis.monte_carlo.iterations),
            base_seed=int(analysis.monte_carlo.base_seed),
            parallel_enabled=bool(analysis.execution.parallel_enabled),
            parallel_workers=int(analysis.execution.parallel_workers),
            variations=list(analysis.monte_carlo.variations),
        )
    return MonteCarloSection()
