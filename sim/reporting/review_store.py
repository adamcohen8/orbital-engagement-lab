from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from sim.config import iter_object_sections, relative_reference_for_object
from sim.plotting.style import get_oel_version
from sim.utils.frames import eci_relative_to_ric_rect

REVIEW_SCHEMA_VERSION = "0.5"
REVIEW_SCHEMA_COMPATIBILITY_POLICY = "pre_1_0_additive"
REVIEW_SCHEMA_STABLE_TABLES = (
    "run_metadata",
    "objects",
    "time_samples",
    "object_state",
    "relative_state",
    "thrust",
    "metrics",
    "artifacts",
)


def write_single_run_review_store(
    *,
    payload: dict[str, Any],
    context: Any,
    artifacts: dict[str, Any],
) -> dict[str, str]:
    """Write a SQLite review store for a completed single-run output folder."""

    cfg = context.cfg
    review_cfg = cfg.outputs.review
    if not bool(review_cfg.enabled):
        return {}

    outdir = Path(context.outdir)
    review_dir = outdir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    db_path = review_dir / "run.sqlite"
    tmp_path = review_dir / "run.sqlite.tmp"
    schema_path = review_dir / "schema.json"
    if tmp_path.exists():
        tmp_path.unlink()

    summary = dict(payload.get("summary", {}) or {})
    t_s = np.asarray(context.t_s, dtype=float).reshape(-1)
    truth_hist = {str(k): _as_2d_float(v) for k, v in dict(context.truth_hist or {}).items()}
    thrust_hist = {str(k): _as_2d_float(v) for k, v in dict(context.thrust_hist or {}).items()}
    generated_utc = _utc_stamp()

    try:
        conn = sqlite3.connect(tmp_path)
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            _create_schema(conn)
            _insert_run_metadata(conn, cfg=cfg, summary=summary, outdir=outdir, generated_utc=generated_utc)
            _insert_objects(conn, cfg=cfg, summary=summary)
            _insert_frame_provenance(conn, payload=payload)
            _insert_object_initialization(conn, payload=payload)
            _insert_object_propagation(conn, payload=payload)
            _insert_object_state_frame(conn, payload=payload)
            _insert_time_samples(conn, t_s=t_s)
            _insert_object_state(conn, t_s=t_s, truth_hist=truth_hist)
            _insert_relative_state(conn, t_s=t_s, truth_hist=truth_hist, summary=summary, cfg=cfg)
            _insert_thrust(conn, t_s=t_s, thrust_hist=thrust_hist)
            _insert_ground_access(conn, t_s=t_s, payload=payload)
            _insert_events(conn, t_s=t_s, summary=summary, thrust_hist=thrust_hist)
            _insert_mission_recovery(conn, summary=summary)
            _insert_metrics(conn, summary=summary)
            _insert_artifacts(conn, artifacts=artifacts, outdir=outdir, generated_utc=generated_utc)
            conn.commit()
        finally:
            conn.close()
        tmp_path.replace(db_path)
        _write_schema_json(schema_path, generated_utc=generated_utc)
        return {
            "sqlite": str(db_path),
            "schema_json": str(schema_path),
        }
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def refresh_review_schema(db_path: str | Path) -> Path:
    """Regenerate the schema sidecar after an approved transactional extension."""

    database = Path(db_path).expanduser().resolve()
    if not database.is_file():
        raise FileNotFoundError(f"Review database not found: {database}")
    schema_path = database.parent / "schema.json"
    _write_schema_json(schema_path, generated_utc=_utc_stamp())
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    tables = dict(schema.get("tables", {}) or {})
    with sqlite3.connect(database) as conn:
        table_names = [
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        ]
        for table_name in table_names:
            columns = [
                {"name": str(row[1]), "type": str(row[2] or "")}
                for row in conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()
            ]
            entry = dict(tables.get(table_name, {}) or {})
            entry.setdefault("description", f"Review evidence table: {table_name}.")
            entry["columns"] = columns
            tables[table_name] = entry
    schema["tables"] = tables
    tmp = schema_path.with_name(schema_path.name + ".tmp")
    tmp.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(schema_path)
    return schema_path


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE run_metadata (
            run_id TEXT PRIMARY KEY,
            scenario_name TEXT,
            scenario_description TEXT,
            oel_version TEXT,
            review_schema_version TEXT,
            generated_utc TEXT,
            duration_s REAL,
            dt_s REAL,
            samples INTEGER,
            output_dir TEXT,
            config_path TEXT,
            config_sha256 TEXT,
            config_json TEXT,
            summary_json_path TEXT,
            run_log_json_path TEXT
        );

        CREATE TABLE objects (
            object_id TEXT PRIMARY KEY,
            object_type TEXT,
            enabled INTEGER,
            mass_initial_kg REAL,
            role TEXT
        );

        CREATE TABLE frame_provenance (
            scope TEXT PRIMARY KEY,
            model TEXT,
            legacy_frame_model TEXT,
            time_scale_model TEXT,
            eop_path TEXT,
            tt_minus_utc_s REAL,
            dut1_s REAL,
            xp_arcsec REAL,
            yp_arcsec REAL,
            dat_s REAL,
            ddpsi_rad REAL,
            ddeps_rad REAL,
            polar_motion_applied INTEGER,
            nutation_corrections_applied INTEGER,
            sample_t_s REAL
        );

        CREATE TABLE object_propagation (
            object_id TEXT PRIMARY KEY,
            propagation_method TEXT,
            general_model TEXT,
            native_frame TEXT,
            output_frame TEXT,
            frame_transform TEXT,
            tle_epoch_jd_utc REAL,
            tle_age_start_days REAL,
            tle_age_end_days REAL
        );

        CREATE TABLE object_initialization (
            object_id TEXT PRIMARY KEY,
            source TEXT,
            initialization_model TEXT,
            initialization_propagator_family TEXT,
            initialization_propagator_name TEXT,
            handoff_propagation_method TEXT,
            native_frame TEXT,
            output_frame TEXT,
            frame_transform TEXT,
            tle_epoch_jd_utc REAL,
            initial_jd_utc REAL,
            tle_age_initialization_days REAL,
            propagate_to_initial_epoch INTEGER,
            simulation_duration_s REAL,
            note TEXT
        );

        CREATE TABLE object_state_frame (
            object_id TEXT PRIMARY KEY,
            state_frame TEXT NOT NULL
        );

        CREATE TABLE time_samples (
            sample_index INTEGER PRIMARY KEY,
            time_s REAL NOT NULL
        );

        CREATE TABLE object_state (
            sample_index INTEGER,
            time_s REAL,
            object_id TEXT,
            pos_x_eci_km REAL,
            pos_y_eci_km REAL,
            pos_z_eci_km REAL,
            vel_x_eci_km_s REAL,
            vel_y_eci_km_s REAL,
            vel_z_eci_km_s REAL,
            quat_w REAL,
            quat_x REAL,
            quat_y REAL,
            quat_z REAL,
            omega_x_rad_s REAL,
            omega_y_rad_s REAL,
            omega_z_rad_s REAL,
            mass_kg REAL
        );
        CREATE INDEX idx_object_state_object_time ON object_state(object_id, time_s);

        CREATE TABLE relative_state (
            sample_index INTEGER,
            time_s REAL,
            deputy_id TEXT,
            chief_id TEXT,
            r_radial_km REAL,
            i_intrack_km REAL,
            c_crosstrack_km REAL,
            v_radial_km_s REAL,
            v_intrack_km_s REAL,
            v_crosstrack_km_s REAL,
            range_km REAL,
            range_rate_km_s REAL
        );
        CREATE INDEX idx_relative_state_pair_time ON relative_state(deputy_id, chief_id, time_s);

        CREATE TABLE thrust (
            sample_index INTEGER,
            time_s REAL,
            object_id TEXT,
            accel_x_eci_km_s2 REAL,
            accel_y_eci_km_s2 REAL,
            accel_z_eci_km_s2 REAL,
            accel_norm_km_s2 REAL,
            burn_active INTEGER
        );
        CREATE INDEX idx_thrust_object_time ON thrust(object_id, time_s);

        CREATE TABLE attitude_error (
            sample_index INTEGER,
            time_s REAL,
            object_id TEXT,
            pointing_error_deg REAL,
            quat_error_angle_deg REAL
        );

        CREATE TABLE ground_access (
            sample_index INTEGER,
            time_s REAL,
            station_id TEXT,
            object_id TEXT,
            access INTEGER,
            line_of_sight INTEGER,
            range_km REAL,
            elevation_deg REAL,
            reason TEXT
        );
        CREATE INDEX idx_ground_access_station_object_time ON ground_access(station_id, object_id, time_s);

        CREATE TABLE events (
            event_id TEXT PRIMARY KEY,
            time_s REAL,
            sample_index INTEGER,
            object_id TEXT,
            event_type TEXT,
            severity TEXT,
            message TEXT,
            source TEXT
        );

        CREATE TABLE metrics (
            metric_id TEXT PRIMARY KEY,
            metric_name TEXT,
            object_id TEXT,
            deputy_id TEXT,
            chief_id TEXT,
            value REAL,
            units TEXT,
            source TEXT
        );

        CREATE TABLE mission_recovery_summary (
            object_id TEXT,
            goal TEXT,
            method TEXT,
            assessment_time_s REAL,
            assessment_sample_index INTEGER,
            recovery_available INTEGER,
            recovery_delta_v_m_s REAL,
            recovery_time_s REAL,
            recovery_time_basis TEXT,
            propellant_kg REAL,
            propellant_fraction REAL,
            disturbance_delta_v_m_s REAL,
            disturbance_apsis TEXT,
            slot_recovery_found INTEGER,
            slot_recovery_orbits INTEGER,
            slot_recovery_time_s REAL,
            slot_recovery_phase_error_deg REAL,
            best_slot_orbits INTEGER,
            best_slot_time_s REAL,
            best_slot_phase_error_deg REAL,
            local_orbit_shape_delta_v_m_s REAL,
            local_orbit_shape_position_error_km REAL,
            notes_json TEXT
        );

        CREATE TABLE mission_recovery_elements (
            object_id TEXT,
            state_label TEXT,
            a_km REAL,
            ecc REAL,
            inc_deg REAL,
            raan_deg REAL,
            argp_deg REAL,
            true_anomaly_deg REAL
        );

        CREATE TABLE mission_recovery_candidates (
            candidate_id TEXT,
            object_id TEXT,
            goal TEXT,
            source TEXT,
            source_family TEXT,
            target_basis TEXT,
            description TEXT,
            planned_delta_v_m_s REAL,
            simulated_delta_v_m_s REAL,
            planned_time_s REAL,
            simulated_recovery_time_s REAL,
            propellant_kg REAL,
            propellant_fraction REAL,
            feasible INTEGER,
            verified INTEGER,
            within_tolerances INTEGER,
            score REAL,
            recommended_modes_json TEXT,
            transfer_type TEXT,
            departure_wait_s REAL,
            time_of_flight_s REAL,
            arrival_time_s REAL,
            target_phase_deg REAL,
            lambert_short_way INTEGER,
            lambert_revolutions INTEGER,
            solver_iterations INTEGER,
            solver_residual_s REAL,
            position_residual_km REAL,
            velocity_residual_m_s REAL,
            notes_json TEXT
        );

        CREATE TABLE mission_recovery_burns (
            candidate_id TEXT,
            burn_index INTEGER,
            start_time_s REAL,
            duration_s REAL,
            frame TEXT,
            axis TEXT,
            delta_v_m_s REAL,
            delta_v_eci_m_s_json TEXT
        );

        CREATE TABLE mission_recovery_candidate_elements (
            candidate_id TEXT,
            object_id TEXT,
            a_km REAL,
            ecc REAL,
            inc_deg REAL,
            raan_deg REAL,
            argp_deg REAL,
            true_anomaly_deg REAL,
            element_errors_json TEXT
        );

        CREATE TABLE artifacts (
            artifact_id TEXT PRIMARY KEY,
            artifact_type TEXT,
            path TEXT,
            title TEXT,
            source TEXT,
            created_utc TEXT
        );
        """
    )


def _insert_run_metadata(
    conn: sqlite3.Connection,
    *,
    cfg: Any,
    summary: dict[str, Any],
    outdir: Path,
    generated_utc: str,
) -> None:
    config_json, config_sha256 = _config_json_and_sha256(cfg)
    conn.execute(
        """
        INSERT INTO run_metadata VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(summary.get("scenario_name") or cfg.scenario_name or "single_run"),
            str(summary.get("scenario_name") or cfg.scenario_name or ""),
            str(summary.get("scenario_description") or cfg.scenario_description or ""),
            get_oel_version(),
            REVIEW_SCHEMA_VERSION,
            generated_utc,
            _float_or_none(summary.get("duration_s")),
            _float_or_none(summary.get("dt_s")),
            _int_or_none(summary.get("samples")),
            str(outdir),
            str(summary.get("config_source_path") or ""),
            config_sha256,
            config_json,
            str(outdir / "master_run_summary.json"),
            str(outdir / "master_run_log.json"),
        ),
    )


def _config_json_and_sha256(cfg: Any) -> tuple[str, str]:
    if hasattr(cfg, "to_dict"):
        data = cfg.to_dict()
    else:
        data = {}
    config_json = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return config_json, hashlib.sha256(config_json.encode("utf-8")).hexdigest()


def _insert_objects(conn: sqlite3.Connection, *, cfg: Any, summary: dict[str, Any]) -> None:
    sections = {str(object_id): section for object_id, section in iter_object_sections(cfg)}
    object_ids = [str(item) for item in list(summary.get("objects", []) or [])]
    rows = []
    for object_id in object_ids:
        section = sections.get(object_id)
        specs = dict(getattr(section, "specs", {}) or {}) if section is not None else {}
        rows.append(
            (
                object_id,
                str(getattr(section, "kind", "") or ""),
                int(bool(getattr(section, "enabled", True))),
                _float_or_none(specs.get("mass_kg", specs.get("dry_mass_kg"))),
                str(getattr(section, "role", "") or ""),
            )
        )
    conn.executemany("INSERT INTO objects VALUES (?, ?, ?, ?, ?)", rows)


def _insert_frame_provenance(conn: sqlite3.Connection, *, payload: dict[str, Any]) -> None:
    frame = dict(payload.get("frame_provenance", {}) or {})
    if not frame:
        frame = dict(dict(payload.get("summary", {}) or {}).get("frame_provenance", {}) or {})
    if not frame:
        return
    row = (
        "scenario",
        str(frame.get("model", "") or ""),
        str(frame.get("legacy_frame_model", "") or ""),
        str(frame.get("time_scale_model", "") or ""),
        str(frame.get("eop_path", "") or ""),
        _float_or_none(frame.get("tt_minus_utc_s")),
        _float_or_none(frame.get("dut1_s")),
        _float_or_none(frame.get("xp_arcsec")),
        _float_or_none(frame.get("yp_arcsec")),
        _float_or_none(frame.get("dat_s")),
        _float_or_none(frame.get("ddpsi_rad")),
        _float_or_none(frame.get("ddeps_rad")),
        1 if bool(frame.get("polar_motion_applied", False)) else 0,
        1 if bool(frame.get("nutation_corrections_applied", False)) else 0,
        _float_or_none(frame.get("sample_t_s")),
    )
    conn.execute("INSERT INTO frame_provenance VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", row)


def _insert_object_initialization(conn: sqlite3.Connection, *, payload: dict[str, Any]) -> None:
    rows = []
    for object_id, metadata_raw in sorted(dict(payload.get("object_initialization", {}) or {}).items()):
        metadata = dict(metadata_raw or {})
        rows.append(
            (
                str(object_id),
                str(metadata.get("source", "") or ""),
                str(metadata.get("initialization_model", "") or ""),
                str(metadata.get("initialization_propagator_family", "") or ""),
                str(metadata.get("initialization_propagator_name", "") or ""),
                str(metadata.get("handoff_propagation_method", "") or ""),
                str(metadata.get("native_frame", "") or ""),
                str(metadata.get("output_frame", "") or ""),
                str(metadata.get("frame_transform", "") or ""),
                _float_or_none(metadata.get("tle_epoch_jd_utc")),
                _float_or_none(metadata.get("initial_jd_utc")),
                _float_or_none(metadata.get("tle_age_initialization_days")),
                1 if bool(metadata.get("propagate_to_initial_epoch", False)) else 0,
                _float_or_none(metadata.get("simulation_duration_s")),
                str(metadata.get("note", "") or ""),
            )
        )
    conn.executemany(
        "INSERT INTO object_initialization VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
    )


def _insert_object_propagation(conn: sqlite3.Connection, *, payload: dict[str, Any]) -> None:
    rows = []
    for object_id, metadata_raw in sorted(dict(payload.get("object_propagation", {}) or {}).items()):
        metadata = dict(metadata_raw or {})
        rows.append(
            (
                str(object_id),
                str(metadata.get("propagation_method", "") or ""),
                str(metadata.get("general_model", "") or ""),
                str(metadata.get("native_frame", "") or ""),
                str(metadata.get("output_frame", "") or ""),
                str(metadata.get("frame_transform", "") or ""),
                _float_or_none(metadata.get("tle_epoch_jd_utc")),
                _float_or_none(metadata.get("tle_age_start_days")),
                _float_or_none(metadata.get("tle_age_end_days")),
            )
        )
    conn.executemany("INSERT INTO object_propagation VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)


def _insert_object_state_frame(conn: sqlite3.Connection, *, payload: dict[str, Any]) -> None:
    explicit = dict(payload.get("object_state_frames", {}) or {})
    propagation = dict(payload.get("object_propagation", {}) or {})
    object_ids = sorted(set(str(key) for key in explicit) | set(str(key) for key in propagation))
    rows = []
    for object_id in object_ids:
        frame = explicit.get(object_id)
        if frame is None:
            frame = dict(propagation.get(object_id, {}) or {}).get("state_history_frame", "eci")
        rows.append((object_id, str(frame or "eci").strip().lower()))
    conn.executemany("INSERT OR REPLACE INTO object_state_frame VALUES (?, ?)", rows)


def _insert_time_samples(conn: sqlite3.Connection, *, t_s: np.ndarray) -> None:
    conn.executemany("INSERT INTO time_samples VALUES (?, ?)", [(int(i), _finite_float(t)) for i, t in enumerate(t_s)])


def _insert_object_state(
    conn: sqlite3.Connection,
    *,
    t_s: np.ndarray,
    truth_hist: dict[str, np.ndarray],
) -> None:
    rows = []
    for object_id, hist in truth_hist.items():
        n = int(min(t_s.size, hist.shape[0]))
        for i in range(n):
            row = hist[i]
            rows.append(
                (
                    i,
                    _finite_float(t_s[i]),
                    object_id,
                    _state_value(row, 0),
                    _state_value(row, 1),
                    _state_value(row, 2),
                    _state_value(row, 3),
                    _state_value(row, 4),
                    _state_value(row, 5),
                    _state_value(row, 6),
                    _state_value(row, 7),
                    _state_value(row, 8),
                    _state_value(row, 9),
                    _state_value(row, 10),
                    _state_value(row, 11),
                    _state_value(row, 12),
                    _state_value(row, 13),
                )
            )
    conn.executemany(
        """
        INSERT INTO object_state VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        rows,
    )


def _insert_relative_state(
    conn: sqlite3.Connection,
    *,
    t_s: np.ndarray,
    truth_hist: dict[str, np.ndarray],
    summary: dict[str, Any],
    cfg: Any,
) -> None:
    rows = []
    for deputy_id, chief_id in _relative_review_pairs(cfg=cfg, truth_hist=truth_hist, summary=summary):
        deputy = truth_hist.get(deputy_id)
        chief = truth_hist.get(chief_id)
        if deputy is None or chief is None or deputy.shape[1] < 6 or chief.shape[1] < 6:
            continue
        n = int(min(t_s.size, deputy.shape[0], chief.shape[0]))
        for i in range(n):
            rel = eci_relative_to_ric_rect(deputy[i, :6], chief[i, :6])
            rng = float(np.linalg.norm(rel[:3]))
            range_rate = float(np.dot(rel[:3], rel[3:]) / rng) if rng > 1e-12 else 0.0
            rows.append(
                (
                    i,
                    _finite_float(t_s[i]),
                    deputy_id,
                    chief_id,
                    _finite_float(rel[0]),
                    _finite_float(rel[1]),
                    _finite_float(rel[2]),
                    _finite_float(rel[3]),
                    _finite_float(rel[4]),
                    _finite_float(rel[5]),
                    _finite_float(rng),
                    _finite_float(range_rate),
                )
            )
    conn.executemany("INSERT INTO relative_state VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)


def _relative_review_pairs(
    *,
    cfg: Any,
    truth_hist: dict[str, np.ndarray],
    summary: dict[str, Any],
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []

    def add_pair(deputy_id: str, chief_id: str) -> None:
        pair = (str(deputy_id), str(chief_id))
        if pair[0] == pair[1] or pair[0] not in truth_hist or pair[1] not in truth_hist:
            return
        if pair not in pairs:
            pairs.append(pair)

    primary = [str(item) for item in list(summary.get("primary_object_pair", []) or [])]
    if len(primary) == 2:
        add_pair(primary[0], primary[1])

    for object_id, _section in iter_object_sections(cfg, enabled_only=True):
        reference_id = relative_reference_for_object(cfg, object_id)
        if reference_id:
            add_pair(object_id, reference_id)

    return pairs


def _insert_thrust(conn: sqlite3.Connection, *, t_s: np.ndarray, thrust_hist: dict[str, np.ndarray]) -> None:
    rows = []
    for object_id, hist in thrust_hist.items():
        if hist.shape[1] < 3:
            continue
        n = int(min(t_s.size, hist.shape[0]))
        for i in range(n):
            vec = np.asarray(hist[i, :3], dtype=float)
            norm = float(np.linalg.norm(vec))
            rows.append(
                (
                    i,
                    _finite_float(t_s[i]),
                    object_id,
                    _finite_float(vec[0]),
                    _finite_float(vec[1]),
                    _finite_float(vec[2]),
                    _finite_float(norm),
                    int(norm > 0.0),
                )
            )
    conn.executemany("INSERT INTO thrust VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)


def _insert_ground_access(conn: sqlite3.Connection, *, t_s: np.ndarray, payload: dict[str, Any]) -> None:
    rows = []
    access_root = dict(payload.get("ground_station_access", {}) or {})
    for station_id, station_payload in access_root.items():
        targets = dict(dict(station_payload or {}).get("targets", {}) or {})
        for object_id, object_payload_raw in targets.items():
            object_payload = dict(object_payload_raw or {})
            access = list(object_payload.get("access", []) or [])
            los = list(object_payload.get("line_of_sight", []) or [])
            ranges = list(object_payload.get("range_km", []) or [])
            elevations = list(object_payload.get("elevation_deg", []) or [])
            reasons = list(object_payload.get("reason", []) or [])
            n = max(len(access), len(los), len(ranges), len(elevations), len(reasons))
            n = int(min(n, t_s.size))
            for i in range(n):
                rows.append(
                    (
                        i,
                        _finite_float(t_s[i]),
                        str(station_id),
                        str(object_id),
                        _bool_int(_list_get(access, i)),
                        _bool_int(_list_get(los, i)),
                        _float_or_none(_list_get(ranges, i)),
                        _float_or_none(_list_get(elevations, i)),
                        None if _list_get(reasons, i) is None else str(_list_get(reasons, i)),
                    )
                )
    conn.executemany("INSERT INTO ground_access VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)


def _insert_events(
    conn: sqlite3.Connection,
    *,
    t_s: np.ndarray,
    summary: dict[str, Any],
    thrust_hist: dict[str, np.ndarray],
) -> None:
    rows = []
    if bool(summary.get("terminated_early", False)):
        event_time = _float_or_none(summary.get("termination_time_s"))
        rows.append(
            (
                "termination",
                event_time,
                _sample_index_for_time(t_s, event_time),
                _none_or_str(summary.get("termination_object_id")),
                "termination",
                "warning",
                _none_or_str(summary.get("termination_reason")) or "simulation terminated early",
                "summary",
            )
        )
    for object_id, hist in thrust_hist.items():
        if hist.shape[1] < 3:
            continue
        n = int(min(t_s.size, hist.shape[0]))
        active = np.linalg.norm(hist[:n, :3], axis=1) > 0.0
        previous = False
        for i, current in enumerate(active):
            if bool(current) and not previous:
                rows.append(
                    (
                        f"burn_start:{object_id}:{i}",
                        _finite_float(t_s[i]),
                        i,
                        object_id,
                        "burn_start",
                        "info",
                        f"{object_id} burn interval started",
                        "review_store",
                    )
                )
            if previous and not bool(current):
                rows.append(
                    (
                        f"burn_end:{object_id}:{i}",
                        _finite_float(t_s[i]),
                        i,
                        object_id,
                        "burn_end",
                        "info",
                        f"{object_id} burn interval ended",
                        "review_store",
                    )
                )
            previous = bool(current)
        if previous and n > 0:
            rows.append(
                (
                    f"burn_end:{object_id}:{n - 1}:final",
                    _finite_float(t_s[n - 1]),
                    n - 1,
                    object_id,
                    "burn_end",
                    "info",
                    f"{object_id} burn interval ended at final sample",
                    "review_store",
                )
            )
    conn.executemany("INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)


def _insert_mission_recovery(conn: sqlite3.Connection, *, summary: dict[str, Any]) -> None:
    recovery = dict(summary.get("mission_recovery", {}) or {})
    if not recovery:
        return
    estimate = dict(recovery.get("recovery_estimate", {}) or {})
    conn.execute(
        """
        INSERT INTO mission_recovery_summary VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        (
            _none_or_str(recovery.get("object_id")),
            _none_or_str(recovery.get("goal")),
            _none_or_str(estimate.get("method")),
            _float_or_none(recovery.get("assessment_time_s")),
            _int_or_none(recovery.get("assessment_sample_index")),
            _bool_int(estimate.get("available")),
            _float_or_none(estimate.get("recovery_delta_v_m_s")),
            _float_or_none(estimate.get("recovery_time_s")),
            _none_or_str(estimate.get("recovery_time_basis")),
            _float_or_none(estimate.get("propellant_kg")),
            _float_or_none(estimate.get("propellant_fraction")),
            _float_or_none(estimate.get("disturbance_delta_v_m_s")),
            _none_or_str(estimate.get("disturbance_apsis")),
            _bool_int(estimate.get("slot_recovery_found")),
            _int_or_none(estimate.get("slot_recovery_orbits")),
            _float_or_none(estimate.get("slot_recovery_time_s")),
            _float_or_none(estimate.get("slot_recovery_phase_error_deg")),
            _int_or_none(estimate.get("best_slot_orbits")),
            _float_or_none(estimate.get("best_slot_time_s")),
            _float_or_none(estimate.get("best_slot_phase_error_deg")),
            _float_or_none(estimate.get("local_orbit_shape_delta_v_m_s")),
            _float_or_none(estimate.get("local_orbit_shape_position_error_km")),
            json.dumps(list(estimate.get("notes", []) or []), sort_keys=True),
        ),
    )
    rows = []
    object_id = _none_or_str(recovery.get("object_id"))
    for label in ("initial", "target", "final"):
        elements = dict(recovery.get(f"{label}_elements", {}) or {})
        if elements:
            rows.append(
                (
                    object_id,
                    label,
                    _float_or_none(elements.get("a_km")),
                    _float_or_none(elements.get("ecc")),
                    _float_or_none(elements.get("inc_deg")),
                    _float_or_none(elements.get("raan_deg")),
                    _float_or_none(elements.get("argp_deg")),
                    _float_or_none(elements.get("true_anomaly_deg")),
                )
            )
    conn.executemany("INSERT INTO mission_recovery_elements VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
    _insert_mission_recovery_planner(conn, recovery=recovery)


def _insert_mission_recovery_planner(conn: sqlite3.Connection, *, recovery: dict[str, Any]) -> None:
    planner = dict(recovery.get("planner", {}) or {})
    if not planner:
        return
    object_id = _none_or_str(recovery.get("object_id"))
    recommended = dict(planner.get("recommended", {}) or {})
    recommended_by_candidate: dict[str, list[str]] = {}
    for mode, candidate_id in recommended.items():
        if candidate_id:
            recommended_by_candidate.setdefault(str(candidate_id), []).append(str(mode))
    candidate_rows = []
    burn_rows = []
    element_rows = []
    for candidate_raw in list(planner.get("candidates", []) or []):
        candidate = dict(candidate_raw or {})
        candidate_id = _none_or_str(candidate.get("candidate_id"))
        if not candidate_id:
            continue
        candidate_rows.append(
            (
                candidate_id,
                object_id,
                _none_or_str(candidate.get("goal", recovery.get("goal"))),
                _none_or_str(candidate.get("source")),
                _none_or_str(candidate.get("source_family")),
                _none_or_str(candidate.get("target_basis")),
                _none_or_str(candidate.get("description")),
                _float_or_none(candidate.get("planned_delta_v_m_s")),
                _float_or_none(candidate.get("simulated_delta_v_m_s")),
                _float_or_none(candidate.get("planned_time_s")),
                _float_or_none(candidate.get("simulated_recovery_time_s")),
                _float_or_none(candidate.get("propellant_kg")),
                _float_or_none(candidate.get("propellant_fraction")),
                _bool_int(candidate.get("feasible")),
                _bool_int(candidate.get("verified")),
                _bool_int(candidate.get("within_tolerances")),
                _float_or_none(candidate.get("score")),
                json.dumps(recommended_by_candidate.get(candidate_id, []), sort_keys=True),
                _none_or_str(candidate.get("transfer_type")),
                _float_or_none(candidate.get("departure_wait_s")),
                _float_or_none(candidate.get("time_of_flight_s")),
                _float_or_none(candidate.get("arrival_time_s")),
                _float_or_none(candidate.get("target_phase_deg")),
                _bool_int(candidate.get("lambert_short_way")),
                _int_or_none(candidate.get("lambert_revolutions")),
                _int_or_none(candidate.get("solver_iterations")),
                _float_or_none(candidate.get("solver_residual_s")),
                _float_or_none(candidate.get("position_residual_km")),
                _float_or_none(candidate.get("velocity_residual_m_s")),
                json.dumps(list(candidate.get("notes", []) or []), sort_keys=True),
            )
        )
        for burn_raw in list(candidate.get("burn_sequence", []) or []):
            burn = dict(burn_raw or {})
            burn_rows.append(
                (
                    candidate_id,
                    _int_or_none(burn.get("burn_index")),
                    _float_or_none(burn.get("start_time_s")),
                    _float_or_none(burn.get("duration_s")),
                    _none_or_str(burn.get("frame")),
                    _none_or_str(burn.get("axis")),
                    _float_or_none(burn.get("delta_v_m_s")),
                    json.dumps(list(burn.get("delta_v_eci_m_s", []) or []), sort_keys=True),
                )
            )
        elements = dict(candidate.get("expected_final_elements", {}) or {})
        if elements:
            element_rows.append(
                (
                    candidate_id,
                    object_id,
                    _float_or_none(elements.get("a_km")),
                    _float_or_none(elements.get("ecc")),
                    _float_or_none(elements.get("inc_deg")),
                    _float_or_none(elements.get("raan_deg")),
                    _float_or_none(elements.get("argp_deg")),
                    _float_or_none(elements.get("true_anomaly_deg")),
                    json.dumps(dict(candidate.get("expected_element_errors", {}) or {}), sort_keys=True),
                )
            )
    conn.executemany(
        "INSERT INTO mission_recovery_candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        candidate_rows,
    )
    conn.executemany("INSERT INTO mission_recovery_burns VALUES (?, ?, ?, ?, ?, ?, ?, ?)", burn_rows)
    conn.executemany(
        "INSERT INTO mission_recovery_candidate_elements VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        element_rows,
    )


def _insert_metrics(conn: sqlite3.Connection, *, summary: dict[str, Any]) -> None:
    rows = []
    pair = [str(item) for item in list(summary.get("primary_object_pair", []) or [])]
    deputy_id = pair[0] if len(pair) == 2 else None
    chief_id = pair[1] if len(pair) == 2 else None
    rel_summary = dict(summary.get("relative_range_summary", {}) or {})
    for name, units in (
        ("initial_range_km", "km"),
        ("final_range_km", "km"),
        ("closest_approach_km", "km"),
        ("closest_approach_time_s", "s"),
    ):
        if name in rel_summary:
            rows.append((name, name, None, deputy_id, chief_id, _float_or_none(rel_summary.get(name)), units, "summary"))
    thrust_stats = dict(summary.get("thrust_stats", {}) or {})
    for object_id, stats_raw in thrust_stats.items():
        stats = dict(stats_raw or {})
        for name, units in (
            ("burn_samples", "samples"),
            ("max_accel_km_s2", "km/s^2"),
            ("total_dv_m_s", "m/s"),
        ):
            if name in stats:
                metric_id = f"{object_id}:{name}"
                rows.append((metric_id, name, str(object_id), None, None, _float_or_none(stats.get(name)), units, "summary"))
    recovery = dict(summary.get("mission_recovery", {}) or {})
    estimate = dict(recovery.get("recovery_estimate", {}) or {})
    object_id = str(recovery.get("object_id", "") or "")
    for name, units in (
        ("recovery_delta_v_m_s", "m/s"),
        ("recovery_time_s", "s"),
        ("propellant_kg", "kg"),
        ("local_orbit_shape_position_error_km", "km"),
    ):
        if name in estimate:
            metric_id = f"{object_id}:mission_recovery:{name}" if object_id else f"mission_recovery:{name}"
            rows.append((metric_id, name, object_id or None, None, None, _float_or_none(estimate.get(name)), units, "summary"))
    conn.executemany("INSERT INTO metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)


def _insert_artifacts(
    conn: sqlite3.Connection,
    *,
    artifacts: dict[str, Any],
    outdir: Path,
    generated_utc: str,
) -> None:
    rows = []
    for artifact_type, artifact_id, path in _iter_artifact_paths(artifacts):
        rows.append(
            (
                artifact_id,
                artifact_type,
                _relative_artifact_path(path, outdir),
                artifact_id.replace("_", " ").title(),
                "single_run_artifacts",
                generated_utc,
            )
        )
    conn.executemany("INSERT OR REPLACE INTO artifacts VALUES (?, ?, ?, ?, ?, ?)", rows)


def _write_schema_json(path: Path, *, generated_utc: str) -> None:
    schema = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "generated_utc": generated_utc,
        "format": "sqlite",
        "database": "run.sqlite",
        "compatibility": {
            "policy": REVIEW_SCHEMA_COMPATIBILITY_POLICY,
            "stable_tables": list(REVIEW_SCHEMA_STABLE_TABLES),
            "breaking_change_requires_schema_version_bump": True,
            "reader_guidance": (
                "Prefer table/column discovery through this schema and ReviewWorkspace.schema(); "
                "new tables or nullable columns may be added before 1.0."
            ),
        },
        "tables": {
            "run_metadata": {"description": "One row describing the run and review schema."},
            "objects": {"description": "Active simulation objects."},
            "frame_provenance": {"description": "Scenario-level frame, EOP, and time-scale provenance."},
            "object_initialization": {
                "description": "Per-object initial-state provenance, including TLE recovery handoffs."
            },
            "object_propagation": {"description": "Per-object propagation-family provenance for GP/SP workflows."},
            "object_state_frame": {"description": "Frame label for each object's state rows, e.g. eci or teme."},
            "time_samples": {"description": "Retained sample times."},
            "object_state": {"description": "Truth state histories by object; join object_state_frame for frame labels."},
            "relative_state": {
                "description": "RIC relative state for configured/default review object pairs."
            },
            "thrust": {"description": "Applied acceleration histories by object."},
            "attitude_error": {"description": "Reserved for attitude error histories."},
            "ground_access": {"description": "Ground station access histories."},
            "events": {"description": "Termination and review-derived event rows."},
            "metrics": {"description": "Scalar summary and review metrics."},
            "mission_recovery_summary": {
                "description": "Configured mission-recovery estimate from initial and assessment-state orbit comparison."
            },
            "mission_recovery_elements": {
                "description": "Initial and assessment classical orbital elements used by mission recovery."
            },
            "mission_recovery_candidates": {
                "description": "Planner candidate trade-space rows for configured mission recovery and Orbit Transfer Planner analyses."
            },
            "mission_recovery_burns": {
                "description": "Burn sequence rows for mission-recovery planner candidates."
            },
            "mission_recovery_candidate_elements": {
                "description": "Expected candidate final elements and element-error JSON."
            },
            "artifacts": {"description": "Known artifacts in the output folder."},
        },
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _iter_artifact_paths(artifacts: dict[str, Any], prefix: str = ""):
    for key, value in dict(artifacts or {}).items():
        path_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            yield from _iter_artifact_paths(value, path_key)
        elif isinstance(value, str) and value:
            yield prefix or "artifact", path_key.replace(".", ":"), Path(value)


def _relative_artifact_path(path: Path, outdir: Path) -> str:
    try:
        return path.resolve().relative_to(outdir.resolve()).as_posix()
    except Exception:
        return str(path)


def _as_2d_float(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return np.empty((0, 0), dtype=float)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim > 2:
        return arr.reshape(arr.shape[0], -1)
    return arr


def _state_value(row: np.ndarray, idx: int) -> float | None:
    if idx >= row.size:
        return None
    return _finite_float(row[idx])


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _float_or_none(value: Any) -> float | None:
    return _finite_float(value)


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bool_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(bool(value))


def _none_or_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _list_get(values: list[Any], idx: int) -> Any:
    return values[idx] if idx < len(values) else None


def _sample_index_for_time(t_s: np.ndarray, time_s: float | None) -> int | None:
    if time_s is None or t_s.size <= 0:
        return None
    return int(np.argmin(np.abs(t_s - float(time_s))))


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
