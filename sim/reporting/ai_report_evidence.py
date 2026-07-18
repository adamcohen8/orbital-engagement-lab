# ruff: noqa: F401,F403,F405,I001
from .ai_report_models import *
from .ai_report_prompts import *

def _select_interesting_runs(runs: list[dict[str, Any]], max_examples: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[int] = set()

    def add(run: dict[str, Any]) -> None:
        if len(selected) >= max_examples:
            return
        idx = int(run.get("iteration", len(seen)))
        if idx in seen:
            return
        seen.add(idx)
        selected.append(run)

    for run in runs:
        assessment = dict(run.get("assessment", {}) or {})
        if assessment and not bool(assessment.get("pass", True)):
            add(run)

    finite_ca = []
    for run in runs:
        try:
            finite_ca.append((float(run.get("closest_approach_km")), run))
        except (TypeError, ValueError):
            continue
    for _, run in sorted(finite_ca, key=lambda item: item[0]):
        add(run)

    for run in runs:
        add(run)

    return selected


def _requested_figure_ids(cfg: SimulationScenarioConfig) -> list[str]:
    plots_cfg = dict(cfg.outputs.plots or {})
    try:
        from sim.master_outputs import _expanded_figure_ids

        return _expanded_figure_ids(plots_cfg)
    except Exception:
        raw = plots_cfg.get("figure_ids", [])
        if isinstance(raw, str):
            return [raw]
        if isinstance(raw, list):
            return [str(x) for x in raw if str(x).strip()]
        return []


def _figure_id_from_artifact_key(key: str, path: str) -> str:
    name = Path(path).stem if path else str(key)
    if name:
        return name
    return str(key).removesuffix("_png").removesuffix("_jpg").removesuffix("_jpeg")


def _iter_image_artifacts(artifact_map: dict[str, Any]) -> list[tuple[str, str]]:
    image_suffixes = {".png", ".jpg", ".jpeg", ".webp"}
    out: list[tuple[str, str]] = []

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, child in sorted(value.items()):
                child_prefix = f"{prefix}.{key}" if prefix else str(key)
                visit(child_prefix, child)
            return
        if isinstance(value, list):
            for idx, child in enumerate(value):
                visit(f"{prefix}[{idx}]", child)
            return
        path_text = str(value)
        if Path(path_text).suffix.lower() in image_suffixes:
            out.append((prefix, path_text))

    visit("", artifact_map)
    return out


def _figure_title(figure_id: str) -> str:
    return str(figure_id).replace("_", " ").strip().title()


def _compact_run_extremes(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        return {}
    closest = []
    duration = []
    total_dv = []
    failed = []
    for run in runs:
        row = dict(run or {})
        assessment = dict(row.get("assessment", {}) or {})
        try:
            closest.append(
                (
                    float(row.get("closest_approach_km", assessment.get("closest_approach_km"))),
                    int(row.get("iteration", 0)),
                )
            )
        except (TypeError, ValueError):
            pass
        try:
            duration.append(
                (
                    float(assessment.get("duration_s", dict(row.get("summary", {}) or {}).get("duration_s"))),
                    int(row.get("iteration", 0)),
                )
            )
        except (TypeError, ValueError):
            pass
        try:
            total_dv.append((float(assessment.get("total_dv_m_s_total")), int(row.get("iteration", 0))))
        except (TypeError, ValueError):
            pass
        if assessment and not bool(assessment.get("pass", True)):
            failed.append(int(row.get("iteration", 0)))
    out: dict[str, Any] = {
        "run_count_in_packet": int(len(runs)),
        "failed_iterations_in_packet": failed,
    }
    if closest:
        out["closest_approach_extremes"] = {
            "min": {"value_km": min(closest)[0], "iteration": min(closest)[1]},
            "max": {"value_km": max(closest)[0], "iteration": max(closest)[1]},
        }
    if duration:
        out["duration_extremes"] = {
            "min": {"value_s": min(duration)[0], "iteration": min(duration)[1]},
            "max": {"value_s": max(duration)[0], "iteration": max(duration)[1]},
        }
    if total_dv:
        out["total_dv_extremes"] = {
            "min": {"value_m_s": min(total_dv)[0], "iteration": min(total_dv)[1]},
            "max": {"value_m_s": max(total_dv)[0], "iteration": max(total_dv)[1]},
        }
    return out


def _figure_data_sources(figure_id: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    agg = dict(payload.get("aggregate_stats", {}) or {})
    commander = dict(payload.get("commander_brief", {}) or {})
    runs = list(payload.get("runs", []) or [])
    analyst = dict(payload.get("analyst_pack", {}) or {})

    campaign_summary = {
        "pass_rate": agg.get("pass_rate", commander.get("p_success")),
        "fail_rate": agg.get("fail_rate", commander.get("p_fail")),
        "duration_s_mean": agg.get("duration_s_mean"),
        "duration_s_min": agg.get("duration_s_min"),
        "duration_s_max": agg.get("duration_s_max"),
        "closest_approach_km_min": agg.get("closest_approach_km_min"),
        "closest_approach_km_mean": agg.get("closest_approach_km_mean"),
        "closest_approach_km_max": agg.get("closest_approach_km_max"),
        "p_keepout_violation": agg.get("p_keepout_violation", commander.get("p_keepout_violation")),
        "p_catastrophic_outcome": agg.get("p_catastrophic_outcome", commander.get("p_catastrophic_outcome")),
        "p_exceed_dv_budget": agg.get("p_exceed_dv_budget", commander.get("p_exceed_dv_budget")),
        "p_exceed_time_budget": agg.get("p_exceed_time_budget", commander.get("p_exceed_time_budget")),
        "guardrail_violation_rate": agg.get("guardrail_violation_rate"),
        "failure_mode_counts": agg.get("failure_mode_counts", {}),
    }
    resource_summary = {
        "total_dv_m_s_mean": agg.get("total_dv_m_s_mean"),
        "total_dv_m_s_p50": agg.get("total_dv_m_s_p50"),
        "total_dv_m_s_p95": agg.get("total_dv_m_s_p95"),
        "total_dv_m_s_p99": agg.get("total_dv_m_s_p99"),
        "by_object": agg.get("by_object", {}),
        "delta_v_budget_m_s_by_object": agg.get("delta_v_budget_m_s_by_object", {}),
        "delta_v_remaining_m_s_by_object": agg.get("delta_v_remaining_m_s_by_object", {}),
        "actuator_diagnostics_by_object": agg.get("actuator_diagnostics_by_object", {}),
    }
    run_extremes = _compact_run_extremes(runs)

    sources: list[dict[str, Any]] = []
    fid = str(figure_id)
    if fid in {
        "run_dashboard",
        "rendezvous_summary",
        "relative_range",
        "trajectory_ric_rect",
        "trajectory_ric_curv",
        "trajectory_ric_rect_2d",
        "trajectory_ric_curv_2d",
        "trajectory_ric_rect_multi",
        "trajectory_ric_curv_multi",
        "trajectory_ric_rect_2d_multi",
        "trajectory_ric_curv_2d_multi",
        "master_monte_carlo_histograms",
        "master_monte_carlo_relative_range_timeseries",
        "master_monte_carlo_ops_dashboard",
        "master_monte_carlo_initial_relative_state_vs_closest_approach",
    }:
        sources.append({"name": "campaign_outcome_summary", "summary": campaign_summary})
        if run_extremes:
            sources.append({"name": "selected_run_extremes", "summary": run_extremes})
    if fid in {
        "control_effort",
        "control_thrust",
        "control_thrust_multi",
        "control_thrust_ric",
        "control_thrust_ric_multi",
        "satellite_delta_v_remaining",
        "master_monte_carlo_delta_v_remaining",
        "master_monte_carlo_ops_dashboard",
        "run_dashboard",
    }:
        sources.append({"name": "resource_and_delta_v_summary", "summary": resource_summary})
    if fid in {
        "estimation_error",
        "estimation_error_components",
        "knowledge_timeline",
        "sensor_access",
        "ground_station_access",
    }:
        sources.append(
            {
                "name": "knowledge_and_estimation_summary",
                "summary": {
                    "knowledge_detection_by_observer": agg.get("knowledge_detection_by_observer", {}),
                    "knowledge_consistency_by_observer": agg.get("knowledge_consistency_by_observer", {}),
                },
            }
        )
    if fid in {
        "ground_track",
        "ground_track_multi",
        "orbit_eci",
        "trajectory_eci_multi",
        "trajectory_ecef",
        "trajectory_ecef_multi",
        "orbital_element_a",
        "orbital_element_ecc",
        "orbital_element_inc",
        "orbital_element_raan",
        "orbital_element_argp",
        "orbital_element_true_anomaly",
        "orbital_elements_summary",
        "orbital_elements_angles",
    }:
        sources.append(
            {
                "name": "trajectory_context_summary",
                "summary": {
                    "duration_s_mean": agg.get("duration_s_mean"),
                    "duration_s_min": agg.get("duration_s_min"),
                    "duration_s_max": agg.get("duration_s_max"),
                    "termination_reason_counts": agg.get("termination_reason_counts", {}),
                },
            }
        )
    if fid in {
        "attitude",
        "quaternion_eci",
        "quaternion_ric",
        "rates_eci",
        "rates_ric",
        "quaternion_error",
        "thrust_alignment_error",
        "attitude_control_summary",
    }:
        sources.append(
            {
                "name": "attitude_guardrail_summary",
                "summary": {
                    "guardrail_events_mean": agg.get("guardrail_events_mean"),
                    "guardrail_events_p95": agg.get("guardrail_events_p95"),
                    "guardrail_violation_rate": agg.get("guardrail_violation_rate"),
                },
            }
        )
    if fid in {
        "rocket_ascent_diagnostics",
        "rocket_gnc_diagnostics",
        "rocket_orbital_elements",
        "rocket_fuel_remaining",
        "rocket_mission_timeline",
        "rocket_downrange_altitude",
        "rocket_maxq_throttle",
        "rocket_tvc_aero_authority",
        "rocket_insertion_scorecard",
    }:
        sources.append(
            {
                "name": "rocket_summary",
                "summary": {
                    "termination_reason_counts": agg.get("termination_reason_counts", {}),
                    "duration_s_mean": agg.get("duration_s_mean"),
                    "top_failure_modes": commander.get("top_failure_modes", []),
                },
            }
        )
    if parameter_rankings := analyst.get("sensitivity_rankings"):
        if fid in {"master_monte_carlo_initial_relative_state_vs_closest_approach", "master_monte_carlo_ops_dashboard"}:
            sources.append(
                {"name": "parameter_driver_summary", "summary": {"sensitivity_rankings": parameter_rankings[:5]}}
            )
    if not sources:
        sources.append({"name": "general_campaign_summary", "summary": campaign_summary})
    return sources


def _build_figure_manifest(
    *,
    cfg: SimulationScenarioConfig,
    payload: dict[str, Any],
    outdir: Path | None,
) -> dict[str, Any]:
    plots_cfg = dict(cfg.outputs.plots or {})
    ai_cfg = dict(cfg.outputs.ai_report or {})
    include_figure_data = bool(ai_cfg.get("include_figure_data", True))
    requested = _requested_figure_ids(cfg)
    artifact_map = dict(payload.get("artifacts", {}) or {})
    generated: list[dict[str, Any]] = []
    for key, path_text in _iter_image_artifacts(artifact_map):
        figure_id = _figure_id_from_artifact_key(str(key), path_text)
        generated.append(
            {
                "artifact_key": str(key),
                "figure_id": figure_id,
                "path": path_text,
                "placeholder": f"[[FIGURE:{figure_id}]]",
                "description": FIGURE_DESCRIPTIONS.get(figure_id, "Generated figure artifact."),
                "data_sources": _figure_data_sources(figure_id, payload) if include_figure_data else [],
                "image_pixels_available": False,
                "analysis_basis": "path and metadata only",
            }
        )

    requested_entries = []
    for figure_id in requested:
        requested_entries.append(
            {
                "figure_id": figure_id,
                "description": FIGURE_DESCRIPTIONS.get(figure_id, "Requested plot from outputs.plots.figure_ids."),
                "data_sources": _figure_data_sources(figure_id, payload) if include_figure_data else [],
                "generated_in_campaign_summary": any(item.get("figure_id") == figure_id for item in generated),
                "expected_location_note": (
                    "For Monte Carlo campaigns, single-run figures are generated inside mc_run_#### directories when plots are enabled. "
                    "Campaign-level figures appear in generated_artifacts when enabled."
                ),
                "image_pixels_available": False,
                "analysis_basis": "config request and numeric payload only",
            }
        )

    return {
        "plots_enabled": bool(plots_cfg.get("enabled", True)),
        "requested_presets": plots_cfg.get("preset", plots_cfg.get("presets", [])),
        "requested_figure_ids": requested,
        "requested_figures": requested_entries,
        "generated_artifacts": generated,
        "output_dir": str(outdir) if outdir is not None else cfg.outputs.output_dir,
        "image_pixels_available": False,
        "figure_data_available": include_figure_data,
        "image_input_note": (
            "The current text-only provider call supplies figure paths and descriptions, not image pixels. "
            "The report may explain what requested figures are intended to show and relate them to numeric data, "
            "but must not claim visual inspection of plotted pixels."
        ),
    }


def _figure_lookup(packet: dict[str, Any]) -> dict[str, dict[str, Any]]:
    manifest = dict(packet.get("figure_manifest", {}) or {})
    lookup: dict[str, dict[str, Any]] = {}
    for fig in list(manifest.get("generated_artifacts", []) or []):
        row = dict(fig or {})
        figure_id = str(row.get("figure_id", "") or "").strip()
        path = str(row.get("path", "") or "").strip()
        if figure_id and path:
            lookup[figure_id] = row
    return lookup


def _markdown_image_for_figure(fig: dict[str, Any], *, base_dir: Path | None = None) -> str:
    figure_id = str(fig.get("figure_id", "") or "")
    path = str(fig.get("path", "") or "")
    title = _figure_title(figure_id)
    description = str(fig.get("description", "") or "").strip()
    alt = title if not description else f"{title}: {description}"
    display_path = path
    if base_dir is not None and path:
        try:
            p = Path(path)
            resolved = p if p.is_absolute() else Path.cwd() / p
            display_path = str(resolved.resolve().relative_to(base_dir.resolve()))
        except ValueError:
            try:
                p = Path(path)
                resolved = p if p.is_absolute() else Path.cwd() / p
                display_path = str(Path(os.path.relpath(str(resolved.resolve()), str(base_dir.resolve()))))
            except Exception:
                display_path = path
        except Exception:
            display_path = path
    if any(ch.isspace() for ch in display_path):
        return f"![{alt}](<{display_path}>)"
    return f"![{alt}]({display_path})"


def _unwrap_image_markdown_fences(markdown: str) -> str:
    pattern = re.compile(r"```(?:markdown|md)?\s*\n(\s*!\[[^\]]+\]\([^)]+\)\s*)\n```", re.IGNORECASE)
    return pattern.sub(lambda match: match.group(1).strip(), markdown)


def _render_figure_placeholders(
    markdown: str,
    packet: dict[str, Any],
    *,
    base_dir: Path | None = None,
) -> tuple[str, list[str], list[str]]:
    lookup = _figure_lookup(packet)
    used: list[str] = []
    unknown: list[str] = []

    def replace(match: re.Match[str]) -> str:
        figure_id = match.group(1).strip()
        fig = lookup.get(figure_id)
        if fig is None:
            unknown.append(figure_id)
            return f"<!-- Unknown figure placeholder: {figure_id} -->"
        used.append(figure_id)
        return _markdown_image_for_figure(fig, base_dir=base_dir)

    rendered = re.sub(r"\[\[FIGURE:([A-Za-z0-9_.:-]+)\]\]", replace, markdown)
    rendered = _unwrap_image_markdown_fences(rendered)
    missing = [figure_id for figure_id in lookup.keys() if figure_id not in set(used)]
    if missing:
        lines = ["", "## Additional Generated Figures"]
        for figure_id in missing:
            fig = lookup[figure_id]
            lines.append("")
            lines.append(_markdown_image_for_figure(fig, base_dir=base_dir))
            desc = str(fig.get("description", "") or "").strip()
            if desc:
                lines.append("")
                lines.append(desc)
        rendered = rendered.rstrip() + "\n" + "\n".join(lines) + "\n"
    return rendered, used, unknown


def _ai_report_quality_checks(
    *,
    report_markdown: str,
    raw_report_markdown: str,
    packet: dict[str, Any],
    inserted_figures: list[str],
    unknown_figure_placeholders: list[str],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    required_sections = ["Executive Summary", "Figure Walk-through", "Inferences Based on the Data"]
    if list(packet.get("user_questions", []) or []):
        required_sections.append("Answers To User Questions")
    section_results = {
        section: bool(re.search(rf"(?im)^#+\s+{re.escape(section)}\s*$", report_markdown))
        for section in required_sections
    }
    lookup = _figure_lookup(packet)
    generated_ids = sorted(lookup.keys())
    missing_generated_figures = [figure_id for figure_id in generated_ids if figure_id not in set(inserted_figures)]
    unreplaced_placeholders = re.findall(r"\[\[FIGURE:([A-Za-z0-9_.:-]+)\]\]", report_markdown)
    schemaish_terms = ["json payload", "json object", "json schema", "packet structure", "api response"]
    schemaish_hits = [term for term in schemaish_terms if term in report_markdown.lower()]
    warnings: list[str] = []
    if (
        not all(section_results.values())
        and not bool(metadata.get("dry_run", False))
        and metadata.get("status") == "ok"
    ):
        warnings.append("Missing one or more required top-level report sections.")
    if unknown_figure_placeholders:
        warnings.append("Report referenced unknown figure placeholders.")
    if unreplaced_placeholders:
        warnings.append("Report contains unreplaced figure placeholders after rendering.")
    if schemaish_hits:
        warnings.append("Report may be describing data structures instead of results.")
    if missing_generated_figures:
        warnings.append(
            "Some generated figures were not explicitly inserted by the model; they may have been appended automatically."
        )

    passed = metadata.get("status") in {"dry_run", "error"} or (
        all(section_results.values())
        and not unknown_figure_placeholders
        and not unreplaced_placeholders
        and not schemaish_hits
    )
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": metadata.get("status"),
        "passed": bool(passed),
        "prompt_profile": metadata.get("prompt_profile"),
        "provider": metadata.get("provider"),
        "model": metadata.get("model"),
        "required_sections": section_results,
        "generated_figure_ids": generated_ids,
        "inserted_figures": list(inserted_figures),
        "missing_generated_figures": missing_generated_figures,
        "unknown_figure_placeholders": list(unknown_figure_placeholders),
        "unreplaced_figure_placeholders": unreplaced_placeholders,
        "schemaish_terms": schemaish_hits,
        "raw_report_chars": int(len(raw_report_markdown)),
        "rendered_report_chars": int(len(report_markdown)),
        "warnings": warnings,
    }

__all__ = [name for name in globals() if not name.startswith("__")]
