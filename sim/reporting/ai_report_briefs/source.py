# ruff: noqa: F401,F403,F405,I001
from ..ai_report_models import *
from ..ai_report_evidence import *

def _fmt_value(value: Any, unit: str = "") -> str:
    if value is None:
        return "not available"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        try:
            if value != value:
                return "not available"
            text = f"{float(value):.4g}"
        except (TypeError, ValueError):
            return "not available"
        return f"{text} {unit}".strip()
    return str(value)


def _percent(value: Any) -> str:
    if isinstance(value, (int, float)):
        try:
            if value != value:
                return "not available"
            return f"{100.0 * float(value):.1f}%"
        except (TypeError, ValueError):
            return "not available"
    return "not available"


def _association_label(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "not available"
    try:
        v = abs(float(value))
    except (TypeError, ValueError):
        return "not available"
    if v != v:
        return "not available"
    if v >= 0.8:
        return "strong"
    if v >= 0.5:
        return "moderate"
    if v >= 0.3:
        return "weak-to-moderate"
    if v > 0.0:
        return "weak"
    return "none"


def _report_source_brief(packet: dict[str, Any]) -> str:
    cfg_summary = dict(packet.get("config_summary", {}) or {})
    payload_kind = str(packet.get("payload_kind", cfg_summary.get("payload_kind", "")) or "").strip().lower()
    from ..ai_report_adapters import _adapter_for_payload_kind

    adapter = _adapter_for_payload_kind(payload_kind)
    if adapter.payload_kind != "monte_carlo":
        return str(adapter.source_brief(packet))

    payload = dict(packet.get("payload", {}) or {})
    agg = dict(payload.get("aggregate_stats", {}) or {})
    commander = dict(payload.get("commander_brief", {}) or {})
    mc_cfg = dict(cfg_summary.get("monte_carlo", {}) or {})
    sim_cfg = dict(cfg_summary.get("simulator", {}) or {})
    objects = dict(cfg_summary.get("objects", {}) or {})
    primary_pair = list(cfg_summary.get("primary_object_pair", []) or [])
    if len(primary_pair) >= 2:
        primary_a_id, primary_b_id = str(primary_pair[0]), str(primary_pair[1])
    else:
        object_ids = list(objects.keys())
        primary_a_id = str(object_ids[0]) if object_ids else "object_a"
        primary_b_id = str(object_ids[1]) if len(object_ids) > 1 else "object_b"
    primary_a = dict(objects.get(primary_a_id, {}) or {})
    primary_b = dict(objects.get(primary_b_id, {}) or {})
    figure_manifest = dict(packet.get("figure_manifest", {}) or {})
    requested_figures = list(figure_manifest.get("requested_figures", []) or [])
    generated_figures = list(figure_manifest.get("generated_artifacts", []) or [])
    parameter_drivers = list(commander.get("top_parameter_drivers", []) or [])
    failure_modes = list(commander.get("top_failure_modes", []) or [])
    outputs = dict(cfg_summary.get("outputs", {}) or {})
    mc_outputs = dict(outputs.get("monte_carlo", {}) or {})
    gates = dict(mc_outputs.get("gates", {}) or {})
    total_dv_mean = agg.get("total_dv_m_s_mean")
    max_dv_gate = gates.get("max_total_dv_m_s")
    max_duration_gate = gates.get("max_duration_s")
    duration_mean = agg.get("duration_s_mean")
    dv_margin = None
    if isinstance(total_dv_mean, (int, float)) and isinstance(max_dv_gate, (int, float)):
        dv_margin = float(max_dv_gate) - float(total_dv_mean)
    time_margin = None
    if isinstance(duration_mean, (int, float)) and isinstance(max_duration_gate, (int, float)):
        time_margin = float(max_duration_gate) - float(duration_mean)

    variation_lines = []
    for variation in list(mc_cfg.get("variations", []) or []):
        v = dict(variation or {})
        mode = str(v.get("mode", ""))
        path = str(v.get("parameter_path", "unknown"))
        details = []
        for key in ("mean", "std", "low", "high", "options"):
            if v.get(key) not in (None, [], ""):
                details.append(f"{key}={v.get(key)}")
        suffix = f" ({', '.join(details)})" if details else ""
        variation_lines.append(f"- {path}, sampled with {mode}{suffix}")
    if not variation_lines:
        variation_lines.append("- No varied parameters were listed.")

    figure_lines = []
    if requested_figures:
        for fig in requested_figures:
            row = dict(fig or {})
            figure_lines.append(
                f"- Requested `{row.get('figure_id', 'unknown')}`: {row.get('description', 'No description available.')}"
            )
            for source in list(row.get("data_sources", []) or [])[:2]:
                srow = dict(source or {})
                figure_lines.append(f"  Data source `{srow.get('name', 'unknown')}`: {srow.get('summary', {})}")
    if generated_figures:
        for fig in generated_figures:
            row = dict(fig or {})
            placeholder = row.get("placeholder")
            figure_lines.append(
                f"- Generated `{row.get('figure_id', 'unknown')}` at `{row.get('path', '')}`: "
                f"{row.get('description', 'No description available.')}"
            )
            if placeholder:
                figure_lines.append(f"  Insert this figure in the report using placeholder: {placeholder}")
            for source in list(row.get("data_sources", []) or [])[:2]:
                srow = dict(source or {})
                figure_lines.append(f"  Data source `{srow.get('name', 'unknown')}`: {srow.get('summary', {})}")
    if not figure_lines:
        figure_lines.append("- No figures were requested or generated for this run.")

    driver_lines = []
    for driver in parameter_drivers[:5]:
        row = dict(driver or {})
        corr_ca = row.get("abs_corr_closest_approach_km")
        driver_lines.append(
            f"- {row.get('parameter_path', 'unknown')}: importance={_fmt_value(row.get('importance_score'))}, "
            f"|corr closest approach|={_fmt_value(corr_ca)} ({_association_label(corr_ca)} association)"
        )
    if not driver_lines:
        driver_lines.append("- No parameter-driver ranking was available.")

    failure_lines = []
    for failure in failure_modes:
        row = dict(failure or {})
        failure_lines.append(
            f"- {row.get('reason', 'unknown')}: count={row.get('count', 0)}, rate={_percent(row.get('rate'))}"
        )
    if not failure_lines:
        failure_lines.append("- No failure modes were reported.")

    return "\n".join(
        [
            "REPORT SOURCE BRIEF",
            "",
            "What was simulated:",
            f"- Scenario: {cfg_summary.get('scenario_name', 'unknown')}",
            f"- Description: {cfg_summary.get('scenario_description', 'not available')}",
            f"- Study type: {cfg_summary.get('payload_kind', packet.get('payload_kind', 'unknown'))}",
            f"- Duration/timestep: {_fmt_value(sim_cfg.get('duration_s'), 's')} at dt={_fmt_value(sim_cfg.get('dt_s'), 's')}",
            f"- {primary_a_id} initial state: {dict(primary_a.get('initial_state', {}) or {})}",
            f"- {primary_b_id} initial state: {dict(primary_b.get('initial_state', {}) or {})}",
            f"- {primary_a_id} orbit controller: {dict(primary_a.get('orbit_control', {}) or {}).get('class_name')}",
            f"- {primary_b_id} orbit controller: {dict(primary_b.get('orbit_control', {}) or {}).get('class_name')}",
            "",
            "Monte Carlo setup:",
            f"- Iterations: {mc_cfg.get('iterations', 'not available')}",
            f"- Parallel workers: {mc_cfg.get('parallel_workers', 'not available')}",
            *variation_lines,
            "",
            "Key deterministic results:",
            f"- Pass rate: {_percent(agg.get('pass_rate', commander.get('p_success')))}",
            f"- Fail rate: {_percent(agg.get('fail_rate', commander.get('p_fail')))}",
            f"- Closest approach: min={_fmt_value(agg.get('closest_approach_km_min'), 'km')}, "
            f"mean={_fmt_value(agg.get('closest_approach_km_mean'), 'km')}, "
            f"max={_fmt_value(agg.get('closest_approach_km_max'), 'km')}",
            f"- Worst-case closest approach: {_fmt_value(commander.get('worst_case_closest_approach_km'), 'km')}",
            f"- Keepout violation probability: {_percent(agg.get('p_keepout_violation', commander.get('p_keepout_violation')))}",
            f"- Catastrophic outcome probability: {_percent(agg.get('p_catastrophic_outcome', commander.get('p_catastrophic_outcome')))}",
            f"- Delta-v budget exceedance probability: {_percent(agg.get('p_exceed_dv_budget', commander.get('p_exceed_dv_budget')))}",
            f"- Time budget exceedance probability: {_percent(agg.get('p_exceed_time_budget', commander.get('p_exceed_time_budget')))}",
            f"- Total delta-v mean: {_fmt_value(total_dv_mean, 'm/s')}",
            f"- Guardrail violation rate: {_percent(agg.get('guardrail_violation_rate'))}",
            f"- Configured gates: {gates if gates else 'not available'}",
            "",
            "Resource margins:",
            f"- Delta-v budget gate: {_fmt_value(max_dv_gate, 'm/s')}",
            f"- Mean delta-v used: {_fmt_value(total_dv_mean, 'm/s')}",
            f"- Mean delta-v margin versus gate: {_fmt_value(dv_margin, 'm/s')}",
            f"- Duration gate: {_fmt_value(max_duration_gate, 's')}",
            f"- Mean duration: {_fmt_value(duration_mean, 's')}",
            f"- Mean time margin versus gate: {_fmt_value(time_margin, 's')}",
            "",
            "Failure modes:",
            *failure_lines,
            "",
            "Parameter drivers:",
            *driver_lines,
            "",
            "Figures:",
            *figure_lines,
            f"- Image pixels available to the LLM: {_fmt_value(figure_manifest.get('image_pixels_available'))}",
            "",
            "Recommended next actions for this run:",
            "- Base recommendations on the deterministic pass/fail rates, configured gates, resource margins, and observed failure modes above.",
            "- If the campaign is narrow or has few varied parameters, recommend broader follow-up runs before drawing mission-level conclusions.",
            "- Do not recommend investigating observed failure modes unless failures appear in the supplied data.",
        ]
    )


def _sensitivity_report_source_brief(packet: dict[str, Any]) -> str:
    cfg_summary = dict(packet.get("config_summary", {}) or {})
    payload = dict(packet.get("payload", {}) or {})
    analysis = dict(payload.get("analysis", {}) or {})
    baseline = dict(payload.get("baseline", {}) or {})
    preflight = dict(analysis.get("preflight", {}) or {})
    rankings = [dict(row or {}) for row in list(payload.get("parameter_rankings", []) or [])]
    parameter_summaries = [dict(row or {}) for row in list(payload.get("parameter_summaries", []) or [])]
    interaction_summaries = [dict(row or {}) for row in list(payload.get("interaction_summaries", []) or [])]
    runs = [dict(row or {}) for row in list(payload.get("runs", []) or [])]
    figure_manifest = dict(packet.get("figure_manifest", {}) or {})
    generated_figures = list(figure_manifest.get("generated_artifacts", []) or [])

    analysis_cfg = dict(cfg_summary.get("analysis", {}) or {})
    sensitivity_cfg = dict(analysis_cfg.get("sensitivity", {}) or {})
    parameters = [dict(row or {}) for row in list(sensitivity_cfg.get("parameters", []) or [])]

    parameter_lines = []
    for param in parameters:
        details = []
        if param.get("values") not in (None, []):
            details.append(f"values={param.get('values')}")
        for key in ("distribution", "low", "high", "mean", "std"):
            if param.get(key) not in (None, "", []):
                details.append(f"{key}={param.get(key)}")
        suffix = f" ({', '.join(details)})" if details else ""
        parameter_lines.append(f"- {param.get('parameter_path', 'unknown')}{suffix}")
    if not parameter_lines:
        parameter_lines.append("- No sensitivity parameters were listed.")

    baseline_lines = []
    if baseline:
        baseline_lines.append(f"- Source: {baseline.get('source', 'unknown')}")
        metrics = dict(baseline.get("metrics", {}) or {})
        for key, value in list(metrics.items())[:6]:
            baseline_lines.append(f"- Baseline `{key}`: {_fmt_value(value)}")
    else:
        baseline_lines.append("- Source: none")

    ranking_lines = []
    for row in rankings[:8]:
        if "max_abs_percent_change_from_baseline" in row:
            ranking_lines.append(
                f"- {row.get('parameter_path', 'unknown')}: max |% baseline change|="
                f"{_fmt_value(row.get('max_abs_percent_change_from_baseline'))}, "
                f"max |delta|={_fmt_value(row.get('max_abs_delta_from_baseline'))}"
            )
        elif "max_abs_metric_span" in row:
            ranking_lines.append(
                f"- {row.get('parameter_path', 'unknown')}: max metric span={_fmt_value(row.get('max_abs_metric_span'))}, "
                f"grid cells={_fmt_value(row.get('value_count'))}"
            )
        else:
            corr = row.get("max_abs_correlation")
            ranking_lines.append(
                f"- {row.get('parameter_path', 'unknown')}: max |correlation|={_fmt_value(corr)} "
                f"({_association_label(corr)} association)"
            )
    if not ranking_lines:
        ranking_lines.append("- No parameter rankings were available.")

    interaction_lines = []
    for summary in interaction_summaries[:4]:
        interaction_lines.append(
            f"- {summary.get('interaction', 'unknown')}: grid="
            f"{summary.get('x_value_count', 0)}x{summary.get('y_value_count', 0)}, "
            f"max metric span={_fmt_value(summary.get('max_abs_metric_span'))}"
        )
        for surface in list(summary.get("metric_surfaces", []) or [])[:3]:
            srow = dict(surface or {})
            interaction_lines.append(
                f"  - `{srow.get('metric_path', 'unknown')}` span={_fmt_value(srow.get('span'))}, "
                f"min={_fmt_value(srow.get('min'))}, max={_fmt_value(srow.get('max'))}"
            )
    if not interaction_lines:
        interaction_lines.append("- No two-parameter interaction surfaces were reported.")

    failed_runs = [row for row in runs if str(row.get("status", "")) == "failed"]
    failure_lines = []
    for row in failed_runs[:8]:
        failure_lines.append(
            f"- Run {row.get('iteration')}: {row.get('sampled_parameters', {})} failed with {row.get('error', 'unknown error')}"
        )
    if not failure_lines:
        failure_lines.append("- No failed sensitivity runs were reported.")

    figure_lines = []
    for fig in generated_figures:
        row = dict(fig or {})
        placeholder = row.get("placeholder")
        figure_lines.append(
            f"- Generated `{row.get('figure_id', 'unknown')}` at `{row.get('path', '')}`: "
            f"{row.get('description', 'Generated figure artifact.')}"
        )
        if placeholder:
            figure_lines.append(f"  Insert this figure in the report using placeholder: {placeholder}")
    if not figure_lines:
        figure_lines.append("- No sensitivity figures were generated.")

    oaat_lines = [
        f"- {row.get('parameter_path', 'unknown')}: values={row.get('value_count', 0)}, "
        f"max |delta|={_fmt_value(row.get('max_abs_delta_from_baseline'))}, "
        f"max |% change|={_fmt_value(row.get('max_abs_percent_change_from_baseline'))}"
        for row in parameter_summaries[:6]
    ]
    if not oaat_lines:
        oaat_lines.append("- No OAAT response summaries were reported.")

    return "\n".join(
        [
            "REPORT SOURCE BRIEF",
            "",
            "What was simulated:",
            f"- Scenario: {cfg_summary.get('scenario_name', 'unknown')}",
            f"- Description: {cfg_summary.get('scenario_description', 'not available')}",
            "- Study type: sensitivity",
            f"- Sensitivity method: {analysis.get('method', sensitivity_cfg.get('method', 'unknown'))}",
            f"- Metrics: {list(analysis.get('metrics', []) or [])}",
            "",
            "Sensitivity setup:",
            *parameter_lines,
            f"- Failure policy: {analysis.get('failure_policy', 'fail_fast')}",
            f"- Preflight errors: {preflight.get('error_count', 0)} across {preflight.get('run_count', analysis.get('run_count', 0))} generated runs",
            "",
            "Run outcomes:",
            f"- Total runs: {analysis.get('run_count', len(runs))}",
            f"- Successful runs: {analysis.get('successful_run_count', 'not available')}",
            f"- Failed runs: {analysis.get('failed_run_count', len(failed_runs))}",
            f"- Parallel requested: {analysis.get('parallel_requested', False)}",
            f"- Parallel active: {analysis.get('parallel_enabled', False)}",
            "",
            "Baseline:",
            *baseline_lines,
            "",
            "Top parameter drivers:",
            *ranking_lines,
            "",
            "OAAT response summaries:",
            *oaat_lines,
            "",
            "Two-parameter interactions:",
            *interaction_lines,
            "",
            "Failed runs:",
            *failure_lines,
            "",
            "Figures:",
            *figure_lines,
            f"- Image pixels available to the LLM: {_fmt_value(figure_manifest.get('image_pixels_available'))}",
            "",
            "Recommended next actions for this sensitivity study:",
            "- Base recommendations on the reported deltas, correlations, grid spans, failed runs, and generated figures above.",
            "- If only one method was run, recommend a complementary OAAT, LHS, or two-parameter grid study when appropriate.",
            "- Do not infer causality beyond the deterministic sensitivity evidence in the supplied report.",
        ]
    )


def _controller_bench_report_source_brief(packet: dict[str, Any]) -> str:
    payload = dict(packet.get("payload", {}) or {})
    figure_manifest = dict(packet.get("figure_manifest", {}) or {})
    generated_figures = list(figure_manifest.get("generated_artifacts", []) or [])
    variants = [dict(row or {}) for row in list(payload.get("variants", []) or [])]
    cases = [dict(row or {}) for row in list(payload.get("cases", []) or [])]
    summaries = [dict(row or {}) for row in list(payload.get("variant_summaries", []) or [])]
    leaderboards = dict(payload.get("leaderboards", {}) or {})
    optimization = dict(payload.get("optimization", {}) or {})
    runs = [dict(row or {}) for row in list(payload.get("runs", []) or [])]
    target = dict(payload.get("controller_target", {}) or {})
    execution = dict(payload.get("execution", {}) or {})

    variant_lines = []
    for summary in summaries:
        variant_lines.append(
            f"- {summary.get('variant_name', 'unknown')}: pass rate={_percent(summary.get('pass_rate'))}, "
            f"passed={summary.get('passed_runs', 0)}/{summary.get('run_count', 0)}, "
            f"metric means={dict(summary.get('metric_means', {}) or {})}, "
            f"objective pass rates={dict(summary.get('objective_pass_rates', {}) or {})}"
        )
    if not variant_lines:
        for variant in variants:
            variant_lines.append(f"- {variant.get('name', 'unknown')}: {variant.get('description', '')}")
    if not variant_lines:
        variant_lines.append("- No controller variants were reported.")

    case_lines = []
    for case in cases:
        objective_names = [
            dict(obj or {}).get("name", dict(obj or {}).get("kind", "unknown"))
            for obj in list(case.get("objectives", []) or [])
        ]
        case_lines.append(
            f"- {case.get('name', 'unknown')}: objectives={objective_names}, config={case.get('config_path', '')}"
        )
    if not case_lines:
        case_lines.append("- No benchmark cases were reported.")

    leaderboard_lines = []
    for objective in list(leaderboards.get("relative_rendezvous", []) or []):
        obj = dict(objective or {})
        leaderboard_lines.append(f"- Objective `{obj.get('objective_name', 'unknown')}`")
        for ranking in list(obj.get("rankings", []) or [])[:5]:
            rank = dict(ranking or {})
            entries = [dict(row or {}) for row in list(rank.get("entries", []) or [])[:3]]
            top = entries[0] if entries else {}
            leaderboard_lines.append(
                f"  - {rank.get('label', rank.get('metric', 'metric'))}: direction={rank.get('direction')}, "
                f"top variant={top.get('variant_name', 'not available')}, value={_fmt_value(top.get('value'))}"
            )
    if not leaderboard_lines:
        leaderboard_lines.append("- No leaderboard rows were reported.")

    failed_run_lines = []
    for run in runs:
        if bool(run.get("passed", False)):
            continue
        failed_run_lines.append(
            f"- {run.get('variant_name', 'variant')} on {run.get('case_name', 'case')}: "
            f"metrics={dict(run.get('metrics', {}) or {})}, failures={run.get('failure_reasons', run.get('failed_criteria', []))}"
        )
    if not failed_run_lines:
        failed_run_lines.append("- No failed controller-bench runs were reported.")

    optimization_lines = []
    if optimization.get("enabled"):
        optimization_lines.extend(
            [
                f"- Algorithm: {optimization.get('algorithm', 'pso')}",
                f"- Best cost: {_fmt_value(optimization.get('best_cost'))}",
                f"- Best parameters: {dict(optimization.get('best_parameters', {}) or {})}",
            ]
        )
    else:
        optimization_lines.append("- Optimization was not enabled for this benchmark.")

    figure_lines = []
    for fig in generated_figures:
        row = dict(fig or {})
        figure_lines.append(
            f"- Generated `{row.get('figure_id', 'unknown')}` at `{row.get('path', '')}`: "
            f"{row.get('description', 'Generated benchmark figure.')}"
        )
        if row.get("placeholder"):
            figure_lines.append(f"  Insert this figure in the report using placeholder: {row.get('placeholder')}")
    if not figure_lines:
        figure_lines.append("- No controller-bench figures were generated.")

    return "\n".join(
        [
            "REPORT SOURCE BRIEF",
            "",
            "What was benchmarked:",
            f"- Suite: {payload.get('suite_name', 'controller_bench')}",
            f"- Description: {payload.get('description', 'not available')}",
            "- Study type: controller_bench",
            f"- Controller target: {target.get('object_id', 'target')}.{target.get('slot', 'controller')}",
            f"- Variants: {len(variants)}",
            f"- Cases: {len(cases)}",
            f"- Runs: {len(runs)}",
            "",
            "Execution:",
            f"- Parallel requested: {execution.get('parallel_requested', False)}",
            f"- Parallel active: {execution.get('parallel_enabled', False)}",
            f"- Workers: {execution.get('parallel_workers', 'not available')}",
            "",
            "Variants and pass rates:",
            *variant_lines,
            "",
            "Benchmark cases:",
            *case_lines,
            "",
            "Leaderboards:",
            *leaderboard_lines,
            "",
            "Optimization:",
            *optimization_lines,
            "",
            "Failed or weak runs:",
            *failed_run_lines[:10],
            "",
            "Figures:",
            *figure_lines,
            f"- Image pixels available to the LLM: {_fmt_value(figure_manifest.get('image_pixels_available'))}",
            "",
            "Recommended next actions for this controller benchmark:",
            "- Base recommendations on pass rates, objective pass rates, metric means, leaderboard rows, failed cases, and optimization evidence above.",
            "- If the benchmark includes few cases or narrow initial conditions, recommend broader follow-up tests before treating a controller as generally best.",
            "- Do not choose a controller based on metrics or objectives that are not present in the supplied data.",
        ]
    )


def _validation_harness_report_source_brief(packet: dict[str, Any]) -> str:
    payload = dict(packet.get("payload", {}) or {})
    figure_manifest = dict(packet.get("figure_manifest", {}) or {})
    generated_figures = list(figure_manifest.get("generated_artifacts", []) or [])
    benchmarks = [dict(row or {}) for row in list(payload.get("benchmarks", []) or [])]
    failed = [row for row in benchmarks if not bool(row.get("passed", False))]

    kind_counts: dict[str, int] = {}
    for bench in benchmarks:
        kind = str(bench.get("kind", "unknown") or "unknown")
        kind_counts[kind] = kind_counts.get(kind, 0) + 1

    benchmark_lines = []
    for bench in benchmarks[:12]:
        evals = [dict(row or {}) for row in list(bench.get("evaluations", []) or [])]
        baseline_evals = [dict(row or {}) for row in list(bench.get("baseline_evaluations", []) or [])]
        failed_evals = [row for row in evals + baseline_evals if not bool(row.get("passed", False))]
        benchmark_lines.append(
            f"- {bench.get('name', 'unknown')} ({bench.get('kind', 'unknown')}): "
            f"passed={_fmt_value(bench.get('passed'))}, checks={len(evals)}, baseline_checks={len(baseline_evals)}, "
            f"failed_checks={len(failed_evals)}, tags={list(bench.get('tags', []) or [])}"
        )
        for row in failed_evals[:4]:
            benchmark_lines.append(
                f"  - Failed `{row.get('metric', 'unknown')}`: actual={_fmt_value(row.get('actual'))}, "
                f"expectation={row.get('expectation', 'not available')}"
            )
        if bench.get("error"):
            benchmark_lines.append(f"  - Error: {bench.get('error')}")
    if not benchmark_lines:
        benchmark_lines.append("- No validation benchmarks were reported.")

    failed_lines = []
    for bench in failed[:10]:
        failed_lines.append(
            f"- {bench.get('name', 'unknown')} ({bench.get('kind', 'unknown')}): {bench.get('description', '')}"
        )
    if not failed_lines:
        failed_lines.append("- No failed validation benchmarks were reported.")

    figure_lines = []
    for fig in generated_figures:
        row = dict(fig or {})
        figure_lines.append(
            f"- Generated `{row.get('figure_id', 'unknown')}` at `{row.get('path', '')}`: "
            f"{row.get('description', 'Generated validation figure.')}"
        )
        if row.get("placeholder"):
            figure_lines.append(f"  Insert this figure in the report using placeholder: {row.get('placeholder')}")
    if not figure_lines:
        figure_lines.append("- No validation figures were generated.")

    return "\n".join(
        [
            "REPORT SOURCE BRIEF",
            "",
            "What was validated:",
            f"- Suite: {payload.get('suite_name', 'validation_harness')}",
            "- Study type: validation_harness",
            f"- Generated UTC: {payload.get('generated_utc', 'not available')}",
            f"- Duration: {_fmt_value(payload.get('duration_s'), 's')}",
            f"- Overall pass: {_fmt_value(payload.get('passed'))}",
            f"- Benchmarks passed: {payload.get('benchmarks_passed', 0)}/{payload.get('benchmarks_total', len(benchmarks))}",
            f"- Benchmark kinds: {kind_counts}",
            "",
            "Benchmark results:",
            *benchmark_lines,
            "",
            "Failed benchmarks:",
            *failed_lines,
            "",
            "Figures:",
            *figure_lines,
            f"- Image pixels available to the LLM: {_fmt_value(figure_manifest.get('image_pixels_available'))}",
            "",
            "Recommended next actions for this validation suite:",
            "- Base recommendations on failed checks, tolerance rules, baseline comparisons, benchmark kinds, and reported errors above.",
            "- If all benchmarks passed, state the validation evidence that passed and the remaining scope limits.",
            "- Do not claim decision-grade validation outside the benchmark envelope represented in the supplied data.",
        ]
    )


def _can_load_sensitivity_outputs(cfg: SimulationScenarioConfig, outdir: Path) -> bool:
    return (
        bool(getattr(getattr(cfg, "analysis", None), "enabled", False))
        and str(getattr(cfg.analysis, "study_type", "")).strip().lower() == "sensitivity"
    )


def _can_load_monte_carlo_outputs(cfg: SimulationScenarioConfig, outdir: Path) -> bool:
    return bool(getattr(cfg.monte_carlo, "enabled", False))


def _can_load_controller_bench_outputs(cfg: SimulationScenarioConfig, outdir: Path) -> bool:
    return (outdir / "controller_bench_summary.json").exists()


def _can_load_validation_harness_outputs(cfg: SimulationScenarioConfig, outdir: Path) -> bool:
    return (outdir / "validation_harness_report.json").exists()

__all__ = [name for name in globals() if not name.startswith("__")]
