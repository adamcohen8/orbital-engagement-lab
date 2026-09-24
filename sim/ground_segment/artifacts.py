"""Portable ground evidence and read-only review-store integration."""

import json


def write_ground_artifacts(outdir, history):
    path = outdir / "ground_segment.json"
    path.write_text(json.dumps({"schema_version": "1", "samples": history}, indent=2, allow_nan=False) + "\n")
    return {"ground_segment_json": str(path)}


def insert_ground_review(conn, history):
    if not history:
        return
    conn.execute(
        "CREATE TABLE ground_segment_state (time_s REAL, object_id TEXT, orbit_json TEXT, telemetry_json TEXT, tracking_json TEXT, PRIMARY KEY(time_s, object_id))"
    )
    conn.execute(
        "CREATE TABLE ground_segment_contacts (time_s REAL, station_id TEXT, object_id TEXT, tracking INTEGER, downlink INTEGER, uplink INTEGER, access_reason TEXT, disabled_services_json TEXT, PRIMARY KEY(time_s, station_id, object_id))"
    )
    conn.execute(
        "CREATE TABLE ground_segment_packets (time_s REAL, station_id TEXT, object_id TEXT, measurement_time_s REAL, received_time_s REAL, kind TEXT, disposition TEXT, data_json TEXT)"
    )
    for sample in history:
        t = sample["time_s"]
        for oid, state in sample["objects"].items():
            conn.execute(
                "INSERT INTO ground_segment_state VALUES (?, ?, ?, ?, ?)",
                (
                    t,
                    oid,
                    json.dumps(state["orbit"], allow_nan=False),
                    json.dumps(state["telemetry"], allow_nan=False),
                    json.dumps(state["tracking"], allow_nan=False),
                ),
            )
        for sid, targets in sample["contacts"].items():
            for oid, row in targets.items():
                conn.execute(
                    "INSERT INTO ground_segment_contacts VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        t,
                        sid,
                        oid,
                        row["tracking"],
                        row["downlink"],
                        row["uplink"],
                        row["access_reason"],
                        json.dumps(row["disabled_services"]),
                    ),
                )
        for p in sample["packets"]:
            conn.execute(
                "INSERT INTO ground_segment_packets VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    t,
                    p["station_id"],
                    p["object_id"],
                    p["time_s"],
                    p["received_time_s"],
                    p["kind"],
                    p["disposition"],
                    json.dumps(p["data"], allow_nan=False),
                ),
            )
