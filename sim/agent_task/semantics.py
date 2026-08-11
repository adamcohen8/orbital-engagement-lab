from __future__ import annotations

from sim.agent_task.models import SemanticMetric

SEMANTIC_METRICS: dict[str, SemanticMetric] = {
    "initial_range_km": SemanticMetric(
        name="initial_range_km",
        description="Initial relative separation between a deputy and chief object.",
        units="km",
        table="metrics",
        saved_query="rendezvous_metrics",
        interpretation="Starting range for a rendezvous or relative-motion study.",
    ),
    "final_range_km": SemanticMetric(
        name="final_range_km",
        description="Final relative separation between a deputy and chief object.",
        units="km",
        table="metrics",
        saved_query="rendezvous_metrics",
        interpretation="Lower values generally indicate better terminal proximity for rendezvous cases.",
    ),
    "closest_approach_km": SemanticMetric(
        name="closest_approach_km",
        description="Minimum sampled relative range in the completed run.",
        units="km",
        table="metrics",
        saved_query="rendezvous_closest_approach",
        sql="SELECT time_s, deputy_id, chief_id, range_km FROM relative_state ORDER BY range_km ASC LIMIT 1",
        interpretation="The closest sampled approach; resolution depends on the configured simulation step.",
        caveats=("This is sample-based, not a continuous-time root solve between samples.",),
    ),
    "closest_approach_time_s": SemanticMetric(
        name="closest_approach_time_s",
        description="Simulation time of the minimum sampled relative range.",
        units="s",
        table="metrics",
        saved_query="rendezvous_metrics",
    ),
    "range_rate_km_s": SemanticMetric(
        name="range_rate_km_s",
        description="Relative range-rate sample from the relative-state review table.",
        units="km/s",
        table="relative_state",
        saved_query="rendezvous_closest_approach",
        interpretation="Useful for distinguishing closing, separating, and near-stationary relative motion.",
    ),
    "burn_activity": SemanticMetric(
        name="burn_activity",
        description="Active thrust samples and maximum applied acceleration by object.",
        table="thrust",
        saved_query="burn_activity",
        interpretation="Shows whether a controller or mission module actually commanded thrust.",
    ),
    "ground_access": SemanticMetric(
        name="ground_access",
        description="Ground-station access sample counts, minimum range, and maximum elevation.",
        table="ground_access",
        saved_query="ground_access_summary",
    ),
    "mission_recovery_delta_v_m_s": SemanticMetric(
        name="mission_recovery_delta_v_m_s",
        description="Estimated recovery delta-v from mission-recovery analysis.",
        units="m/s",
        table="mission_recovery_summary",
        saved_query="mission_recovery_summary",
    ),    "validation_passed": SemanticMetric(
        name="validation_passed",
        description="Validation benchmark pass/fail evidence from the workflow review store.",
        table="validation_benchmarks",
        saved_query="validation_benchmarks",
    ),
}


def get_semantic_metric(name: str) -> SemanticMetric | None:
    return SEMANTIC_METRICS.get(str(name or "").strip())


def list_semantic_metrics() -> list[SemanticMetric]:
    return [SEMANTIC_METRICS[name] for name in sorted(SEMANTIC_METRICS)]


def semantic_metric_dicts(names: list[str] | tuple[str, ...]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    seen: set[str] = set()
    for name in names:
        metric = get_semantic_metric(name)
        if metric is None or metric.name in seen:
            continue
        seen.add(metric.name)
        out.append(metric.to_dict())
    return out


def semantic_metric_request_rows(names: list[str] | tuple[str, ...]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw_name in names:
        name = str(raw_name or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        metric = get_semantic_metric(name)
        row: dict[str, object] = {"name": name, "known": metric is not None}
        if metric is not None:
            row.update(
                {
                    "maturity": metric.maturity,
                    "source_tables": list(metric.source_tables),
                    "saved_query": metric.saved_query,
                }
            )
        else:
            row["reason"] = "unknown_semantic_metric"
        out.append(row)
    return out
