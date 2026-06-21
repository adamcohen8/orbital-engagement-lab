from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from sim.dynamics.orbit.epoch import datetime_to_julian_date, julian_date_to_datetime

DEFAULT_ACCESS_REPORT_EPOCH_UTC = datetime(2026, 1, 1, tzinfo=timezone.utc)
DEFAULT_ACCESS_REPORT_JD_UTC = datetime_to_julian_date(DEFAULT_ACCESS_REPORT_EPOCH_UTC)


def access_report_epoch_jd_utc(initial_jd_utc: float | None) -> float:
    return DEFAULT_ACCESS_REPORT_JD_UTC if initial_jd_utc is None else float(initial_jd_utc)


def utc_iso_from_sim_time(t_s: float | None, *, jd_utc_start: float) -> str | None:
    if t_s is None:
        return None
    dt = julian_date_to_datetime(float(jd_utc_start) + float(t_s) / 86400.0)
    if dt.microsecond >= 500_000:
        dt += timedelta(seconds=1)
    dt = dt.replace(microsecond=0)
    return dt.isoformat(timespec="seconds").replace("+00:00", "Z")


def _json_float(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _finite_stat(values: list[Any], *, op: str) -> float | None:
    arr = np.array([float("nan") if value is None else float(value) for value in values], dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    return float(np.min(finite) if op == "min" else np.max(finite))


def extract_access_windows(
    *,
    t_s: np.ndarray,
    target_payload: dict[str, Any],
    jd_utc_start: float,
) -> list[dict[str, Any]]:
    t = np.array(t_s, dtype=float).reshape(-1)
    access = np.array(target_payload.get("access", []), dtype=bool).reshape(-1)
    n = int(min(t.size, access.size))
    if n <= 0:
        return []

    ranges = list(target_payload.get("range_km", []) or [])
    elevations = list(target_payload.get("elevation_deg", []) or [])
    windows: list[dict[str, Any]] = []
    k = 0
    while k < n:
        if not bool(access[k]):
            k += 1
            continue
        start_idx = k
        while k + 1 < n and bool(access[k + 1]):
            k += 1
        end_idx = k
        end_time_idx = min(end_idx + 1, n - 1)
        start_time_s = float(t[start_idx])
        end_time_s = float(t[end_time_idx])
        range_slice = ranges[start_idx : end_idx + 1]
        elevation_slice = elevations[start_idx : end_idx + 1]
        windows.append(
            {
                "start_index": int(start_idx),
                "end_index": int(end_idx),
                "start_time_s": start_time_s,
                "end_time_s": end_time_s,
                "duration_s": float(max(end_time_s - start_time_s, 0.0)),
                "aos_utc": utc_iso_from_sim_time(start_time_s, jd_utc_start=jd_utc_start),
                "los_utc": utc_iso_from_sim_time(end_time_s, jd_utc_start=jd_utc_start),
                "min_range_km": _finite_stat(range_slice, op="min"),
                "max_elevation_deg": _finite_stat(elevation_slice, op="max"),
            }
        )
        k += 1
    return windows


def build_ground_station_access_report_views(
    *,
    ground_station_access: dict[str, Any],
    ground_station_access_summary: dict[str, Any],
    t_s: np.ndarray,
    initial_jd_utc: float | None,
) -> dict[str, Any]:
    jd_utc_start = access_report_epoch_jd_utc(initial_jd_utc)
    station_view: dict[str, Any] = {}
    satellite_view: dict[str, Any] = {}
    for station_id, station_payload_raw in sorted(dict(ground_station_access or {}).items()):
        station_payload = dict(station_payload_raw or {})
        station = dict(station_payload.get("station", {}) or {})
        station_entry = station_view.setdefault(str(station_id), {"station": station, "satellites": {}})
        targets = dict(station_payload.get("targets", {}) or {})
        for object_id, target_payload_raw in sorted(targets.items()):
            target_payload = dict(target_payload_raw or {})
            summary = dict(dict(ground_station_access_summary or {}).get(station_id, {}).get(object_id, {}) or {})
            windows = extract_access_windows(t_s=t_s, target_payload=target_payload, jd_utc_start=jd_utc_start)
            summary["first_access_utc"] = utc_iso_from_sim_time(
                summary.get("first_access_time_s"), jd_utc_start=jd_utc_start
            )
            summary["last_access_utc"] = utc_iso_from_sim_time(
                summary.get("last_access_time_s"), jd_utc_start=jd_utc_start
            )
            pair = {"summary": summary, "windows": windows}
            station_entry["satellites"][str(object_id)] = pair
            sat_entry = satellite_view.setdefault(str(object_id), {"stations": {}})
            sat_entry["stations"][str(station_id)] = {
                "station": station,
                "summary": summary,
                "windows": windows,
            }
    return {
        "epoch_jd_utc": float(jd_utc_start),
        "epoch_utc": utc_iso_from_sim_time(0.0, jd_utc_start=jd_utc_start),
        "by_ground_station": station_view,
        "by_satellite": satellite_view,
    }


def _fmt_duration(seconds: Any) -> str:
    value = _json_float(seconds)
    return "" if value is None else f"{value:.1f}"


def _fmt_metric(value: Any, *, precision: int = 3) -> str:
    x = _json_float(value)
    return "" if x is None else f"{x:.{precision}f}"


def _summary_values_from_windows(
    summary: dict[str, Any],
    windows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not windows:
        return {
            "access_duration_s": summary.get("access_duration_s"),
            "first_access_utc": summary.get("first_access_utc"),
            "last_los_utc": summary.get("last_access_utc"),
            "max_elevation_deg": summary.get("max_elevation_deg"),
            "min_range_km": summary.get("min_range_km"),
        }

    durations = [_json_float(window.get("duration_s")) for window in windows]
    duration_values = [value for value in durations if value is not None]
    max_elevation_deg = _finite_stat([window.get("max_elevation_deg") for window in windows], op="max")
    min_range_km = _finite_stat([window.get("min_range_km") for window in windows], op="min")
    return {
        "access_duration_s": sum(duration_values) if duration_values else summary.get("access_duration_s"),
        "first_access_utc": windows[0].get("aos_utc") or summary.get("first_access_utc"),
        "last_los_utc": windows[-1].get("los_utc") or summary.get("last_access_utc"),
        "max_elevation_deg": max_elevation_deg if max_elevation_deg is not None else summary.get("max_elevation_deg"),
        "min_range_km": min_range_km if min_range_km is not None else summary.get("min_range_km"),
    }


def _summary_table_row(name: str, summary: dict[str, Any], windows: list[dict[str, Any]]) -> str:
    values = _summary_values_from_windows(summary, windows)
    return (
        f"| `{name}` | {len(windows)} | {_fmt_duration(values.get('access_duration_s'))} | "
        f"{values.get('first_access_utc') or ''} | {values.get('last_los_utc') or ''} | "
        f"{_fmt_metric(values.get('max_elevation_deg'), precision=2)} | "
        f"{_fmt_metric(values.get('min_range_km'), precision=3)} |"
    )


def _window_rows(windows: list[dict[str, Any]]) -> list[str]:
    if not windows:
        return ["| none | none | 0.0 |  |  |"]
    return [
        "| "
        f"{window.get('aos_utc') or ''} | "
        f"{window.get('los_utc') or ''} | "
        f"{_fmt_duration(window.get('duration_s'))} | "
        f"{_fmt_metric(window.get('max_elevation_deg'), precision=2)} | "
        f"{_fmt_metric(window.get('min_range_km'), precision=3)} |"
        for window in windows
    ]


def render_satellite_access_report(views: dict[str, Any]) -> str:
    lines = [
        "# Satellite Access Report",
        "",
        f"Epoch UTC: `{views.get('epoch_utc')}`",
        "",
    ]
    by_sat = dict(views.get("by_satellite", {}) or {})
    if not by_sat:
        lines.append("No ground-station access was recorded.")
        return "\n".join(lines) + "\n"
    for sat_id, sat_payload_raw in sorted(by_sat.items()):
        stations = dict(dict(sat_payload_raw or {}).get("stations", {}) or {})
        lines.extend(
            [
                f"## {sat_id}",
                "",
                "| Ground Station | Windows | Access Duration (s) | First AOS UTC | Last LOS UTC | Max Elevation (deg) | Min Range (km) |",
                "|---|---:|---:|---|---|---:|---:|",
            ]
        )
        for station_id, station_payload_raw in sorted(stations.items()):
            station_payload = dict(station_payload_raw or {})
            lines.append(
                _summary_table_row(
                    str(station_id),
                    dict(station_payload.get("summary", {}) or {}),
                    list(station_payload.get("windows", []) or []),
                )
            )
        lines.append("")
        for station_id, station_payload_raw in sorted(stations.items()):
            station_payload = dict(station_payload_raw or {})
            lines.extend(
                [
                    f"### {sat_id} -> {station_id}",
                    "",
                    "| AOS UTC | LOS UTC | Duration (s) | Max Elevation (deg) | Min Range (km) |",
                    "|---|---|---:|---:|---:|",
                    *_window_rows(list(station_payload.get("windows", []) or [])),
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def render_ground_station_access_report(views: dict[str, Any]) -> str:
    lines = [
        "# Ground Station Access Report",
        "",
        f"Epoch UTC: `{views.get('epoch_utc')}`",
        "",
    ]
    by_station = dict(views.get("by_ground_station", {}) or {})
    if not by_station:
        lines.append("No ground-station access was recorded.")
        return "\n".join(lines) + "\n"
    for station_id, station_payload_raw in sorted(by_station.items()):
        satellites = dict(dict(station_payload_raw or {}).get("satellites", {}) or {})
        lines.extend(
            [
                f"## {station_id}",
                "",
                "| Satellite | Windows | Access Duration (s) | First AOS UTC | Last LOS UTC | Max Elevation (deg) | Min Range (km) |",
                "|---|---:|---:|---|---|---:|---:|",
            ]
        )
        for sat_id, sat_payload_raw in sorted(satellites.items()):
            sat_payload = dict(sat_payload_raw or {})
            lines.append(
                _summary_table_row(
                    str(sat_id),
                    dict(sat_payload.get("summary", {}) or {}),
                    list(sat_payload.get("windows", []) or []),
                )
            )
        lines.append("")
        for sat_id, sat_payload_raw in sorted(satellites.items()):
            sat_payload = dict(sat_payload_raw or {})
            lines.extend(
                [
                    f"### {station_id} -> {sat_id}",
                    "",
                    "| AOS UTC | LOS UTC | Duration (s) | Max Elevation (deg) | Min Range (km) |",
                    "|---|---|---:|---:|---:|",
                    *_window_rows(list(sat_payload.get("windows", []) or [])),
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def write_ground_station_access_reports(
    *,
    outdir: Path,
    ground_station_access: dict[str, Any],
    ground_station_access_summary: dict[str, Any],
    t_s: np.ndarray,
    initial_jd_utc: float | None,
) -> tuple[dict[str, str], dict[str, Any]]:
    views = build_ground_station_access_report_views(
        ground_station_access=ground_station_access,
        ground_station_access_summary=ground_station_access_summary,
        t_s=t_s,
        initial_jd_utc=initial_jd_utc,
    )
    if not views["by_ground_station"]:
        return {}, views
    outdir.mkdir(parents=True, exist_ok=True)
    by_satellite_path = outdir / "ground_station_access_by_satellite.md"
    by_station_path = outdir / "ground_station_access_by_station.md"
    by_satellite_path.write_text(render_satellite_access_report(views), encoding="utf-8")
    by_station_path.write_text(render_ground_station_access_report(views), encoding="utf-8")
    return {
        "by_satellite": str(by_satellite_path),
        "by_ground_station": str(by_station_path),
    }, views
