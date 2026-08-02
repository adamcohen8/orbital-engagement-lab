from __future__ import annotations

from typing import Any

from sim.config.scenario.models import (
    OutputsSection,
)
from sim.config.scenario.primitives import (
    _OUTPUT_ANIMATIONS_UNSUPPORTED_ALIASES,
    _OUTPUT_PLOTS_UNSUPPORTED_ALIASES,
    _OUTPUTS_UNSUPPORTED_ALIASES,
    _as_dict,
    _reject_unknown_fields,
    _reject_unsupported_aliases,
)
from sim.security import ConfigPathPolicy

__all__ = [
    '_parse_outputs_section',
]

def _parse_outputs_section(value: Any, path_policy: ConfigPathPolicy | None = None) -> OutputsSection:
    d = _as_dict(value, "outputs")
    _reject_unsupported_aliases(d, "outputs", _OUTPUTS_UNSUPPORTED_ALIASES)
    _reject_unknown_fields(
        d,
        "outputs",
        {
            "output_dir",
            "mode",
            "stats",
            "plots",
            "animations",
            "monte_carlo",
            "ai_report",
            "ai_config",
            "review",
            "resource_limits",
        },
    )
    plots = _as_dict(d.get("plots"), "outputs.plots")
    animations = _as_dict(d.get("animations"), "outputs.animations")
    stats = _as_dict(d.get("stats"), "outputs.stats")
    monte_carlo_outputs = _as_dict(d.get("monte_carlo"), "outputs.monte_carlo")
    ai_report = _as_dict(d.get("ai_report"), "outputs.ai_report")
    ai_config = _as_dict(d.get("ai_config"), "outputs.ai_config")
    review = _as_dict(d.get("review"), "outputs.review")
    resource_limits = _as_dict(d.get("resource_limits"), "outputs.resource_limits")
    _reject_unsupported_aliases(plots, "outputs.plots", _OUTPUT_PLOTS_UNSUPPORTED_ALIASES)
    _reject_unsupported_aliases(animations, "outputs.animations", _OUTPUT_ANIMATIONS_UNSUPPORTED_ALIASES)
    _reject_unknown_fields(
        stats,
        "outputs.stats",
        {"enabled", "print_summary", "save_json", "save_csv", "save_full_log", "save_history_npz", "controller_debug"},
    )
    _reject_unknown_fields(
        plots,
        "outputs.plots",
        {
            "enabled", "figure_ids", "dpi", "style", "preset", "reference_object_id",
            "reference_object_label", "orbital_elements_object_id", "keepout_radius_km",
            "burn_marker_object_ids", "ric_2d_planes", "draw_earth_map", "thrust_direction_body",
        },
    )
    _reject_unknown_fields(
        animations,
        "outputs.animations",
        {
            "enabled", "types", "fps", "style", "frame_stride", "speed_multiple", "draw_earth_map",
            "battlespace_dashboard_attitude_dims_m", "battlespace_dashboard_chaser_object_id",
            "battlespace_dashboard_show_trajectory", "battlespace_dashboard_target_object_id",
            "battlespace_dashboard_thruster_active_threshold_km_s2",
            "attitude_ric_thruster_active_threshold_km_s2", "attitude_ric_thruster_dims_m",
            "attitude_ric_thruster_object_ids", "ric_curv_prism_dims_m", "ric_curv_prism_object_ids",
            "ric_side_by_side_dims_m", "ric_side_by_side_left_object_id", "ric_side_by_side_right_object_id",
            "target_object_id", "target_reference_ric_curv_object_ids",
            "target_reference_ric_curv_3d_show_trajectory", "target_reference_ric_curv_2d_show_trajectory",
            "target_reference_ric_curv_2d_ri_show_trajectory", "target_reference_ric_curv_2d_ic_show_trajectory",
            "target_reference_ric_curv_2d_rc_show_trajectory", "target_reference_ric_curv_2d_planes",
        },
    )
    _reject_unknown_fields(
        monte_carlo_outputs,
        "outputs.monte_carlo",
        {
            "baseline_summary_json", "catastrophic_failure_reasons", "checkpoint_enabled",
            "display_histograms", "display_ops_dashboard", "gates", "require_rocket_insertion",
            "save_aggregate_summary", "save_histograms", "save_iteration_summaries", "save_ops_dashboard",
            "save_raw_runs", "success_termination_reasons",
        },
    )
    _reject_unknown_fields(
        ai_report,
        "outputs.ai_report",
        {
            "enabled", "provider", "model", "endpoint", "api_key_env", "timeout_s", "options", "dry_run",
            "fail_on_error", "data_scope", "include_figure_data", "include_json_appendix", "max_examples",
            "prompt_profile", "prompt_file", "report_mode", "user_questions", "user_questions_file",
            "include_full_config", "fail_on_quality", "pricing", "estimated_output_tokens",
            "output_token_estimate", "max_output_tokens", "max_tokens", "max_prompt_chars", "chars_per_token",
            "max_failure_examples", "generation_config", "generationConfig", "anthropic_version",
        },
    )
    _reject_unknown_fields(
        ai_config,
        "outputs.ai_config",
        {
            "enabled", "provider", "model", "endpoint", "api_key_env", "timeout_s", "validate_timeout_s",
            "options", "dry_run", "repair_attempts",
        },
    )
    _reject_unknown_fields(review, "outputs.review", {"enabled", "detail", "strict"})
    _reject_unknown_fields(
        resource_limits,
        "outputs.resource_limits",
        {
            "max_history_memory_mb", "checkpoint_enabled", "hard_min_available_memory_mb", "max_load_per_cpu",
            "min_available_memory_mb", "resource_max_wait_s", "resource_pause_seconds", "resource_profile",
            "throttle_enabled",
        },
    )
    out = OutputsSection(
        output_dir=str(d.get("output_dir", "outputs")),
        mode=str(d.get("mode", "interactive")),
        stats=stats,
        plots=plots,
        animations=animations,
        monte_carlo=monte_carlo_outputs,
        ai_report=ai_report,
        ai_config=ai_config,
        review=review,
        resource_limits=resource_limits,
    )
    if out.mode not in ("interactive", "save", "both"):
        raise ValueError("outputs.mode must be one of: interactive, save, both.")
    if not out.output_dir.strip():
        raise ValueError("outputs.output_dir must be non-empty.")
    max_history_memory_mb = out.resource_limits.max_history_memory_mb
    if max_history_memory_mb is not None and max_history_memory_mb <= 0:
        raise ValueError("outputs.resource_limits.max_history_memory_mb must be positive when set.")
    if out.review.detail not in {"compact", "standard", "full"}:
        raise ValueError("outputs.review.detail must be one of: compact, standard, full.")
    if path_policy is not None:
        path_policy.resolve_output_dir(out.output_dir, purpose="outputs.output_dir")
    return out
