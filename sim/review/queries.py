from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SavedReviewQuery:
    name: str
    description: str
    sql: str


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
            "ORDER BY time_s DESC LIMIT 1"
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
    ),
    "burn_events": SavedReviewQuery(
        name="burn_events",
        description="Burn start/end events.",
        sql=(
            "SELECT time_s, object_id, event_type, message FROM events "
            "WHERE event_type IN ('burn_start', 'burn_end') ORDER BY time_s, event_id"
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
}


def get_saved_review_query(name: str) -> SavedReviewQuery | None:
    return SAVED_REVIEW_QUERIES.get(str(name or "").strip())


def list_saved_review_queries() -> list[SavedReviewQuery]:
    return [SAVED_REVIEW_QUERIES[name] for name in sorted(SAVED_REVIEW_QUERIES)]
