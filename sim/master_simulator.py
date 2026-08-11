from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from sim.config import SimulationScenarioConfig
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2, EARTH_RADIUS_KM
from sim.execution.metrics import (
    closest_approach_from_run_payload as _closest_approach_from_run_payload,
)
from sim.execution.metrics import (
    relative_range_series_from_run_payload as _relative_range_series_from_run_payload,
)
from sim.execution.monte_carlo_support import (
    aggregate_knowledge_consistency_from_runs as _aggregate_knowledge_consistency_from_runs,
)
from sim.execution.monte_carlo_support import (
    aggregate_knowledge_detection_from_runs as _aggregate_knowledge_detection_from_runs,
)
from sim.execution.monte_carlo_support import (
    assess_mc_run as _assess_mc_run,
)
from sim.execution.monte_carlo_support import (
    build_baseline_comparison as _build_baseline_comparison,
)
from sim.execution.monte_carlo_support import (
    build_parameter_sensitivity_rankings as _build_parameter_sensitivity_rankings,
)
from sim.execution.monte_carlo_support import (
    coerce_numeric_map as _coerce_numeric_map,
)
from sim.execution.monte_carlo_support import (
    extract_baseline_metrics as _extract_baseline_metrics,
)
from sim.execution.monte_carlo_support import (
    fmt_float as _fmt_float,
)
from sim.execution.monte_carlo_support import (
    get_git_commit_sha as _get_git_commit_sha,
)
from sim.execution.monte_carlo_support import (
    infer_model_profile as _infer_model_profile,
)
from sim.execution.monte_carlo_support import (
    load_json_file as _load_json_file,
)
from sim.execution.monte_carlo_support import (
    mc_initial_relative_ric_curv_samples as _mc_initial_relative_ric_curv_samples,
)
from sim.execution.monte_carlo_support import (
    quantile_stats as _quantile_stats,
)
from sim.execution.monte_carlo_support import (
    safe_float as _safe_float,
)
from sim.execution.monte_carlo_support import (
    satellite_initial_delta_v_budget_m_s as _satellite_initial_delta_v_budget_m_s,
)
from sim.execution.monte_carlo_support import (
    write_commander_brief_markdown as _write_commander_brief_markdown,
)
from sim.execution.study import analysis_study_type as _analysis_study_type
from sim.execution.workers import (
    restore_env_vars as _restore_env_vars,
)
from sim.execution.workers import (
    run_mc_iteration_from_dict as _run_mc_iteration_from_dict,
)
from sim.execution.workers import (
    set_parallel_worker_thread_limits as _set_parallel_worker_thread_limits,
)
from sim.runtime_support import (
    AgentRuntime,
    _apply_chaser_relative_init_from_target,
    _apply_thruster_mount_defaults,
    _attitude_state13_from_belief,
    _build_knowledge_base,
    _build_orbit_propagator,
    _build_rocket_guidance,
    _call_with_compat_kwargs,
    _coe_to_rv_eci,
    _combine_commands,
    _command_to_dict,
    _compatible_keyword_args,
    _create_rocket_runtime,
    _create_satellite_runtime,
    _deep_set,
    _default_truth_from_agent,
    _deploy_from_rocket,
    _module_obj,
    _orbital_elements_basic,
    _relative_orbit_state12,
    _resolve_chaser_relative_ric_init,
    _resolve_rocket_stack,
    _resolve_satellite_inertia_kg_m2,
    _resolve_satellite_isp_s,
    _rocket_altitude_km,
    _rocket_state_to_truth,
    _run_mission_execution,
    _run_mission_modules,
    _run_mission_strategy,
    _sample_variation,
    _to_jsonable_value,
    _truth_from_state6,
    _truth_state6,
)

__all__ = [
    "AgentRuntime",
    "EARTH_MU_KM3_S2",
    "EARTH_RADIUS_KM",
    "prepare_batch_run_configs",
    "run_master_simulation",
    "validate_generated_batch_configs",
    "_aggregate_knowledge_consistency_from_runs",
    "_aggregate_knowledge_detection_from_runs",
    "_analysis_study_type",
    "_apply_chaser_relative_init_from_target",
    "_apply_thruster_mount_defaults",
    "_assess_mc_run",
    "_attitude_state13_from_belief",
    "_build_baseline_comparison",
    "_build_knowledge_base",
    "_build_orbit_propagator",
    "_build_parameter_sensitivity_rankings",
    "_build_rocket_guidance",
    "_call_with_compat_kwargs",
    "_closest_approach_from_run_payload",
    "_coe_to_rv_eci",
    "_coerce_noninteractive_for_automation",
    "_coerce_numeric_map",
    "_combine_commands",
    "_command_to_dict",
    "_compatible_keyword_args",
    "_create_rocket_runtime",
    "_create_satellite_runtime",
    "_deep_get",
    "_deep_set",
    "_default_truth_from_agent",
    "_deploy_from_rocket",
    "_extract_baseline_metrics",
    "_fmt_float",
    "_get_git_commit_sha",
    "_infer_model_profile",
    "_is_truthy_env",
    "_load_json_file",
    "_mc_initial_relative_ric_curv_samples",
    "_module_obj",
    "_orbital_elements_basic",
    "_quantile_stats",
    "_quat_error_angle_deg",
    "_relative_orbit_state12",
    "_relative_range_series_from_run_payload",
    "_resolve_chaser_relative_ric_init",
    "_resolve_rocket_stack",
    "_resolve_satellite_inertia_kg_m2",
    "_resolve_satellite_isp_s",
    "_restore_env_vars",
    "_rocket_altitude_km",
    "_rocket_state_to_truth",
    "_run_mc_iteration_from_dict",
    "_run_mission_execution",
    "_run_mission_modules",
    "_run_mission_strategy",
    "_run_sensitivity_analysis",
    "_run_single_config",
    "_safe_float",
    "_sample_variation",
    "_satellite_initial_delta_v_budget_m_s",
    "_set_parallel_worker_thread_limits",
    "_to_jsonable_value",
    "_truth_from_state6",
    "_truth_state6",
    "_write_commander_brief_markdown",
]


def _deep_get(root: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = root
    for tok in path.split("."):
        if isinstance(cur, dict) and tok in cur:
            cur = cur[tok]
        else:
            return default
    return cur


def _quat_error_angle_deg(q_des: np.ndarray, q_cur: np.ndarray) -> float:
    qd = np.array(q_des, dtype=float).reshape(4)
    qc = np.array(q_cur, dtype=float).reshape(4)
    qd = qd / max(float(np.linalg.norm(qd)), 1e-12)
    qc = qc / max(float(np.linalg.norm(qc)), 1e-12)
    dot = abs(float(np.dot(qd, qc)))
    dot = min(1.0, max(-1.0, dot))
    return float(np.degrees(2.0 * np.arccos(dot)))


def _run_single_config(
    cfg: SimulationScenarioConfig,
    step_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    import sim.runtime_support as _runtime_support
    from sim.single_run import _run_single_config as _single_run_impl

    _runtime_support.EARTH_MU_KM3_S2 = EARTH_MU_KM3_S2
    return _single_run_impl(cfg, step_callback=step_callback)


def _is_truthy_env(name: str) -> bool:
    from sim.single_run import _is_truthy_env as _truthy_impl

    return _truthy_impl(name)


def _coerce_noninteractive_for_automation(cfg: SimulationScenarioConfig) -> SimulationScenarioConfig:
    from sim.single_run import _coerce_noninteractive_for_automation as _coerce_impl

    return _coerce_impl(cfg)


def _run_sensitivity_analysis(
    *,
    config_path: str | Path,
    cfg: SimulationScenarioConfig,
    step_callback: Callable[[int, int], None] | None = None,
    batch_callback: Callable[[int, int], None] | None = None,
    batch_progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    from sim.execution.sensitivity import run_sensitivity_analysis

    return run_sensitivity_analysis(
        config_path=config_path,
        cfg=cfg,
        step_callback=step_callback,
        batch_callback=batch_callback,
        batch_progress_callback=batch_progress_callback,
    )


def prepare_batch_run_configs(cfg: SimulationScenarioConfig) -> list[dict[str, Any]]:
    from sim.execution.validation import prepare_batch_run_configs as _prepare_impl

    return _prepare_impl(cfg)


def validate_generated_batch_configs(cfg: SimulationScenarioConfig) -> dict[str, Any]:
    from sim.execution.validation import validate_generated_batch_configs as _validate_impl

    return _validate_impl(cfg)


def run_master_simulation(
    config_path: str | Path,
    step_callback: Callable[[int, int], None] | None = None,
    mc_callback: Callable[[int, int], None] | None = None,
    mc_progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    from sim.execution import run_simulation_config_file

    return run_simulation_config_file(
        config_path=config_path,
        step_callback=step_callback,
        batch_callback=mc_callback,
        batch_progress_callback=mc_progress_callback,
    )
