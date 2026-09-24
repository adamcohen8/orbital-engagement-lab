"""Resource truth evidence writers; no sensor or estimator semantics implied."""

from __future__ import annotations

import csv
import json
from bisect import bisect_left
from pathlib import Path

from sim.spacecraft_resources.model import ENERGY_RATES

COLUMNS = (
    "time_s",
    "interval_start_s",
    "sunlit_fraction",
    "solar_irradiance_w_m2",
    "temperature_k",
    "solar_heat_w",
    "albedo_heat_w",
    "earth_ir_heat_w",
    "internal_heat_w",
    "radiated_heat_w",
    "stored_heat_w",
    "thermal_balance_residual_w",
    "solar_generation_w",
    "load_demand_w",
    "load_served_w",
    "unmet_load_w",
    "curtailed_power_w",
    "battery_charge_w",
    "battery_discharge_w",
    "battery_energy_wh",
    "battery_soc",
    "battery_loss_w",
    "conversion_loss_w",
    "electrical_heat_w",
    "power_balance_residual_w",
) + tuple(key + "_energy_j" for key in ENERGY_RATES)


def write_resource_artifacts(outdir, histories):
    root = Path(outdir)
    json_path = root / "spacecraft_resources.json"
    json_path.write_text(
        json.dumps(
            {"schema_version": 1, "semantics": "simulation_truth", "objects": histories}, indent=2, allow_nan=False
        )
        + "\n"
    )
    csv_path = root / "spacecraft_resources.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("object_id", *COLUMNS))
        writer.writeheader()
        for oid, rows in histories.items():
            writer.writerows({"object_id": oid, **row} for row in rows)
    return {"spacecraft_resources_json": str(json_path), "spacecraft_resources_csv": str(csv_path)}


def insert_resource_review(conn, histories):
    if not histories:
        return
    conn.execute(
        "CREATE TABLE spacecraft_resources (object_id TEXT NOT NULL, "
        + ", ".join(f'"{key}" REAL' for key in COLUMNS)
        + ", PRIMARY KEY(object_id, time_s))"
    )
    conn.executemany(
        "INSERT INTO spacecraft_resources VALUES (" + ",".join("?" for _ in range(len(COLUMNS) + 1)) + ")",
        ((oid, *(row.get(key) for key in COLUMNS)) for oid, rows in histories.items() for row in rows),
    )


def resource_snapshot(histories, time_s):
    result = {}
    for oid, rows in histories.items():
        index = bisect_left(rows, time_s - 1e-9, key=lambda row: row["time_s"])
        if index < len(rows) and abs(rows[index]["time_s"] - time_s) < 1e-9:
            result[oid] = dict(rows[index])
    return result
