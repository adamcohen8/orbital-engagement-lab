from __future__ import annotations

import re
from dataclasses import dataclass

SAVED_QUERY_MATURITY_LEVELS = frozenset({"supported", "prototype", "experimental"})


@dataclass(frozen=True)
class SavedReviewQuery:
    name: str
    description: str
    sql: str
    source_tables: tuple[str, ...] = ()
    maturity: str = "supported"
    allow_empty: bool = False
    max_vm_steps: int = 250_000

    def __post_init__(self) -> None:
        if self.maturity not in SAVED_QUERY_MATURITY_LEVELS:
            allowed = ", ".join(sorted(SAVED_QUERY_MATURITY_LEVELS))
            raise ValueError(f"Unknown saved review query maturity {self.maturity!r}; expected one of: {allowed}")
        if not self.sql.lstrip().upper().startswith(("SELECT", "WITH")):
            raise ValueError(f"Saved review query {self.name!r} must be read-only SELECT/WITH SQL.")
        if int(self.max_vm_steps) <= 0:
            raise ValueError(f"Saved review query {self.name!r} must have a positive VM step budget.")
        if not self.source_tables:
            object.__setattr__(self, "source_tables", _infer_source_tables(self.sql))


def _infer_source_tables(sql: str) -> tuple[str, ...]:
    cte_names = {
        match.group(1)
        for match in re.finditer(r"(?:\bWITH|,)\s+([A-Za-z_][A-Za-z0-9_]*)\s+AS\s*\(", sql, flags=re.IGNORECASE)
    }
    names = {
        match.group(1)
        for match in re.finditer(r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)\b", sql, flags=re.IGNORECASE)
        if match.group(1).lower() not in {"select"} and match.group(1) not in cte_names
    }
    return tuple(sorted(names))


SAVED_REVIEW_QUERIES: dict[str, SavedReviewQuery] = {
    "run_metadata": SavedReviewQuery(
        name="run_metadata",
        description="Run name, duration, timestep, samples, OEL version, and review schema version.",
        sql=(
            "SELECT scenario_name, duration_s, dt_s, samples, oel_version, "
            "review_schema_version FROM run_metadata"
        ),
    ),
    "objects": SavedReviewQuery(
        name="objects",
        description="Active object inventory.",
        sql="SELECT object_id, object_type, role, mass_initial_kg FROM objects ORDER BY object_id",
    ),
    "artifacts": SavedReviewQuery(
        name="artifacts",
        description="Known output artifact inventory.",
        sql="SELECT artifact_type, artifact_id, path FROM artifacts ORDER BY artifact_type, artifact_id",
    ),
    "passive_final_state": SavedReviewQuery(
        name="passive_final_state",
        description="Final ECI state for each object.",
        sql=(
            "SELECT object_id, time_s, pos_x_eci_km, pos_y_eci_km, pos_z_eci_km, "
            "vel_x_eci_km_s, vel_y_eci_km_s, vel_z_eci_km_s FROM object_state "
            "WHERE sample_index = (SELECT MAX(sample_index) FROM object_state) "
            "ORDER BY object_id"
        ),
    ),
    "object_state_first_last": SavedReviewQuery(
        name="object_state_first_last",
        description="First and final ECI position and velocity samples for each object.",
        sql=(
            "WITH bounds AS (SELECT object_id, MIN(sample_index) AS first_i, "
            "MAX(sample_index) AS last_i FROM object_state GROUP BY object_id) "
            "SELECT s.object_id, s.sample_index, s.time_s, s.pos_x_eci_km, "
            "s.pos_y_eci_km, s.pos_z_eci_km, s.vel_x_eci_km_s, "
            "s.vel_y_eci_km_s, s.vel_z_eci_km_s FROM object_state s "
            "JOIN bounds b ON s.object_id = b.object_id "
            "AND s.sample_index IN (b.first_i, b.last_i) "
            "ORDER BY s.object_id, s.sample_index"
        ),
    ),
    "rendezvous_metrics": SavedReviewQuery(
        name="rendezvous_metrics",
        description="Initial range, final range, closest approach, and closest approach time.",
        sql=(
            "SELECT metric_name, value, units, deputy_id, chief_id FROM metrics "
            "WHERE metric_name IN ('initial_range_km', 'final_range_km', "
            "'closest_approach_km', 'closest_approach_time_s') ORDER BY metric_name"
        ),
    ),
    "rendezvous_closest_approach": SavedReviewQuery(
        name="rendezvous_closest_approach",
        description="Closest relative-state sample by range.",
        sql=(
            "SELECT time_s, deputy_id, chief_id, range_km, range_rate_km_s "
            "FROM relative_state ORDER BY range_km ASC LIMIT 1"
        ),
    ),
    "relative_final_state": SavedReviewQuery(
        name="relative_final_state",
        description="Final relative-state sample.",
        sql=(
            "SELECT time_s, deputy_id, chief_id, r_radial_km, i_intrack_km, "
            "c_crosstrack_km, range_km, range_rate_km_s FROM relative_state "
            "WHERE sample_index = (SELECT MAX(sample_index) FROM relative_state) "
            "ORDER BY deputy_id, chief_id"
        ),
    ),
    "burn_activity": SavedReviewQuery(
        name="burn_activity",
        description="Applied acceleration and active burn sample counts by object.",
        sql=(
            "SELECT object_id, COUNT(*) AS samples, SUM(burn_active) AS active_samples, "
            "MAX(accel_norm_km_s2) AS max_accel_km_s2 FROM thrust GROUP BY object_id "
            "ORDER BY object_id"
        ),
        max_vm_steps=500_000,
    ),
    "burn_events": SavedReviewQuery(
        name="burn_events",
        description="Burn start/end events.",
        sql=(
            "SELECT time_s, object_id, event_type, message FROM events "
            "WHERE event_type IN ('burn_start', 'burn_end') ORDER BY time_s, event_id"
        ),
        allow_empty=True,
    ),
    "event_log": SavedReviewQuery(
        name="event_log",
        description="Complete ordered termination and runtime event log.",
        sql=(
            "SELECT time_s, object_id, event_type, severity, message FROM events "
            "ORDER BY time_s, event_id"
        ),
        allow_empty=True,
    ),
    "mission_recovery_summary": SavedReviewQuery(
        name="mission_recovery_summary",
        description="Mission-recovery delta-v, time, propellant, and method for configured recovery analysis.",
        sql=(
            "SELECT object_id, goal, method, recovery_available, recovery_delta_v_m_s, "
            "recovery_time_s, recovery_time_basis, propellant_kg, disturbance_delta_v_m_s, "
            "slot_recovery_found, slot_recovery_time_s, best_slot_time_s "
            "FROM mission_recovery_summary"
        ),
    ),
    "mission_recovery_elements": SavedReviewQuery(
        name="mission_recovery_elements",
        description="Initial and final orbital elements used by configured mission-recovery analysis.",
        sql=(
            "SELECT object_id, state_label, a_km, ecc, inc_deg, raan_deg, argp_deg, "
            "true_anomaly_deg FROM mission_recovery_elements ORDER BY object_id, state_label"
        ),
    ),
    "mission_recovery_candidates": SavedReviewQuery(
        name="mission_recovery_candidates",
        description=(
            "Original-orbit analytical baselines and Orbit Transfer Planner candidates ranked by feasibility, "
            "delta-v, and time."
        ),
        sql=(
            "SELECT candidate_id, object_id, goal, source, source_family, target_basis, "
            "transfer_type, planned_delta_v_m_s, planned_time_s, departure_wait_s, "
            "time_of_flight_s, propellant_kg, feasible, verified, recommended_modes_json "
            "FROM mission_recovery_candidates ORDER BY feasible DESC, planned_delta_v_m_s, planned_time_s"
        ),
    ),
    "mission_recovery_burns": SavedReviewQuery(
        name="mission_recovery_burns",
        description="Burn sequence rows for original-orbit baselines and Orbit Transfer Planner candidates.",
        sql=(
            "SELECT candidate_id, burn_index, start_time_s, duration_s, frame, axis, delta_v_m_s "
            "FROM mission_recovery_burns ORDER BY candidate_id, burn_index"
        ),
    ),
    "ground_access_summary": SavedReviewQuery(
        name="ground_access_summary",
        description="Access sample counts, minimum range, and maximum elevation by station/object.",
        sql=(
            "SELECT station_id, object_id, COUNT(*) AS samples, SUM(access) AS access_samples, "
            "MIN(range_km) AS min_range_km, MAX(elevation_deg) AS max_elevation_deg "
            "FROM ground_access GROUP BY station_id, object_id ORDER BY station_id, object_id"
        ),
    ),
    "ground_access_no_access_reasons": SavedReviewQuery(
        name="ground_access_no_access_reasons",
        description="No-access reason counts by station/object.",
        sql=(
            "SELECT station_id, object_id, reason, COUNT(*) AS samples FROM ground_access "
            "WHERE access = 0 GROUP BY station_id, object_id, reason "
            "ORDER BY station_id, object_id, samples DESC"
        ),
    ),
    "attitude_rates_first_last": SavedReviewQuery(
        name="attitude_rates_first_last",
        description="First and final angular-rate samples by object.",
        sql=(
            "WITH bounds AS (SELECT object_id, MIN(sample_index) AS first_i, "
            "MAX(sample_index) AS last_i FROM object_state GROUP BY object_id) "
            "SELECT s.object_id, s.time_s, s.omega_x_rad_s, s.omega_y_rad_s, "
            "s.omega_z_rad_s FROM object_state s JOIN bounds b ON s.object_id = b.object_id "
            "AND s.sample_index IN (b.first_i, b.last_i) ORDER BY s.object_id, s.sample_index"
        ),
    ),
    "attitude_state_first_last": SavedReviewQuery(
        name="attitude_state_first_last",
        description="First and final quaternion and angular-rate samples by object.",
        sql=(
            "WITH bounds AS (SELECT object_id, MIN(sample_index) AS first_i, "
            "MAX(sample_index) AS last_i FROM object_state GROUP BY object_id) "
            "SELECT s.object_id, s.time_s, s.quat_w, s.quat_x, s.quat_y, s.quat_z, "
            "s.omega_x_rad_s, s.omega_y_rad_s, s.omega_z_rad_s FROM object_state s "
            "JOIN bounds b ON s.object_id = b.object_id "
            "AND s.sample_index IN (b.first_i, b.last_i) ORDER BY s.object_id, s.sample_index"
        ),
    ),
    "workflow_metadata": SavedReviewQuery(
        name="workflow_metadata",
        description="Workflow review type, scenario, status, generation time, schema version, and source config.",
        sql=(
            "SELECT workflow_type, scenario_name, title, status, generated_utc, "
            "review_schema_version, source_config FROM workflow_metadata"
        ),
    ),
    "workflow_artifacts": SavedReviewQuery(
        name="workflow_artifacts",
        description="Workflow review artifact inventory.",
        sql="SELECT artifact_key, artifact_type, path FROM workflow_artifacts ORDER BY artifact_type, artifact_key",
    ),
    "controller_bench_runs": SavedReviewQuery(
        name="controller_bench_runs",
        description="Controller-bench run pass/fail rows and output directories.",
        sql=(
            "SELECT variant_name, case_name, passed, failure_count, output_dir "
            "FROM bench_runs ORDER BY variant_name, case_name"
        ),
    ),
    "controller_bench_leaderboard": SavedReviewQuery(
        name="controller_bench_leaderboard",
        description="Controller-bench leaderboard rows by objective and metric.",
        sql=(
            "SELECT kind, objective, metric, rank, variant_name, value, samples "
            "FROM bench_leaderboard ORDER BY kind, objective, metric, rank"
        ),
    ),
    "controller_bench_failures": SavedReviewQuery(
        name="controller_bench_failures",
        description="Controller-bench objective failures and suggestions.",
        sql=(
            "SELECT variant_name, case_name, objective, metric, reason, failure_mode, suggestion "
            "FROM bench_failures ORDER BY variant_name, case_name, objective"
        ),
    ),
    "sensitivity_rankings": SavedReviewQuery(
        name="sensitivity_rankings",
        description="Sensitivity parameters ranked by effect size.",
        sql=(
            "SELECT parameter_path, metric_path, method, effect_size, rank "
            "FROM sensitivity_rankings ORDER BY rank, parameter_path, metric_path"
        ),
    ),
    "sensitivity_runs": SavedReviewQuery(
        name="sensitivity_runs",
        description="Sensitivity generated-run status and changed parameter rows.",
        sql=(
            "SELECT run_id, status, parameter_path, parameter_value, output_dir "
            "FROM sensitivity_runs ORDER BY run_id"
        ),
    ),
    "campaign_runs": SavedReviewQuery(
        name="campaign_runs",
        description="Monte Carlo campaign runs with pass/fail and key metrics.",
        sql=(
            "SELECT iteration, passed, closest_approach_km, duration_s, total_dv_m_s, output_dir "
            "FROM campaign_runs ORDER BY iteration"
        ),
    ),
    "campaign_metrics": SavedReviewQuery(
        name="campaign_metrics",
        description="Flattened Monte Carlo campaign metrics by iteration.",
        sql="SELECT iteration, metric_name, metric_value FROM campaign_metrics ORDER BY iteration, metric_name",
    ),
    "validation_benchmarks": SavedReviewQuery(
        name="validation_benchmarks",
        description="Validation benchmark pass/fail rows.",
        sql=(
            "SELECT benchmark_name, kind, passed, duration_s, output_dir "
            "FROM validation_benchmarks ORDER BY benchmark_name"
        ),
    ),
    "od_phase9_pilots": SavedReviewQuery(
        name="od_phase9_pilots",
        description="Productized OD pilot capability, claim level, readiness, and evidence packet paths.",
        sql=(
            "SELECT case_id, capability_id, claim_level, status, passed, evidence_packet_path "
            "FROM od_phase9_pilots ORDER BY case_id"
        ),
        maturity="supported",
    ),
    "sequential_od_summary": SavedReviewQuery(
        name="sequential_od_summary",
        description="Sequential OD estimator, observation decisions, and evidence state.",
        sql=(
            "SELECT object_id, estimator, observation_count, accepted_count, rejected_count, evidence_status "
            "FROM sequential_od_run"
        ),
        maturity="supported",
    ),
    "integrated_relative_od_models": SavedReviewQuery(
        name="integrated_relative_od_models",
        description="Integrated relative OD model fit, holdout, and sequential comparison rows.",
        sql=(
            "SELECT model, fit_position_rms_m, holdout_position_rms_m, batch_solver_success, "
            "sequential_position_rms_m FROM relative_od_models ORDER BY model"
        ),
        maturity="supported",
    ),
}


def get_saved_review_query(name: str) -> SavedReviewQuery | None:
    return SAVED_REVIEW_QUERIES.get(str(name or "").strip())


def list_saved_review_queries() -> list[SavedReviewQuery]:
    return [SAVED_REVIEW_QUERIES[name] for name in sorted(SAVED_REVIEW_QUERIES)]
