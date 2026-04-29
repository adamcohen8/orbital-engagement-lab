from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from sim.config import GroundStationSection
from sim.dynamics.orbit.environment import EARTH_RADIUS_KM
from sim.dynamics.orbit.frames import ecef_to_eci, eci_to_ecef
from sim.utils.geodesy import ecef_to_enu_rotation, geodetic_to_ecef_km


def _json_float(value: float) -> float | None:
    x = float(value)
    return x if np.isfinite(x) else None


def _first_last_time(t_s: np.ndarray, mask: np.ndarray) -> tuple[float | None, float | None]:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return None, None
    return float(t_s[int(idx[0])]), float(t_s[int(idx[-1])])


def _line_of_sight_from_ground(station_eci_km: np.ndarray, target_eci_km: np.ndarray) -> bool:
    station = np.array(station_eci_km, dtype=float).reshape(3)
    target = np.array(target_eci_km, dtype=float).reshape(3)
    segment = target - station
    denom = float(np.dot(segment, segment))
    if denom <= 0.0:
        return True
    tau = float(-np.dot(station, segment) / denom)
    if tau <= 0.0 or tau >= 1.0:
        return True
    closest = station + tau * segment
    return bool(np.linalg.norm(closest) > EARTH_RADIUS_KM)


def evaluate_ground_station_access(
    *,
    ground_stations: list[GroundStationSection],
    t_s: np.ndarray,
    truth_hist: dict[str, np.ndarray],
    jd_utc_start: float | None = None,
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, dict[str, dict[str, Any]]]]:
    """
    Evaluate passive ground-station access to each simulated object.

    Access is true when the target has geometric line of sight from the station,
    elevation is at least the station minimum, and range is no greater than the
    configured maximum range when one is supplied.
    """
    stations = [station for station in ground_stations if station.enabled]
    tt = np.array(t_s, dtype=float).reshape(-1)
    if not stations or tt.size == 0:
        return {}, {}

    histories: dict[str, dict[str, dict[str, Any]]] = {}
    summaries: dict[str, dict[str, dict[str, Any]]] = {}
    dt = np.diff(tt)
    total_duration_s = float(tt[-1] - tt[0]) if tt.size > 1 else 0.0

    for station in stations:
        station_ecef = geodetic_to_ecef_km(station.lat_deg, station.lon_deg, station.alt_km)
        enu_rot = ecef_to_enu_rotation(station.lat_deg, station.lon_deg)
        min_elev = float(station.min_elevation_deg)
        max_range = station.max_range_km
        station_hist: dict[str, dict[str, Any]] = {}
        station_summary: dict[str, dict[str, Any]] = {}

        for object_id, hist in sorted(truth_hist.items()):
            arr = np.array(hist, dtype=float)
            n = min(tt.size, arr.shape[0])
            access = np.zeros(tt.size, dtype=bool)
            los_ok = np.zeros(tt.size, dtype=bool)
            range_km = np.full(tt.size, np.nan)
            elevation_deg = np.full(tt.size, np.nan)
            reason: list[str] = ["inactive"] * tt.size

            for k in range(n):
                state = arr[k, :]
                if state.size < 3 or not np.all(np.isfinite(state[:3])):
                    continue
                t = float(tt[k])
                target_eci = np.array(state[:3], dtype=float)
                station_eci = ecef_to_eci(station_ecef, t, jd_utc_start=jd_utc_start)
                target_ecef = eci_to_ecef(target_eci, t, jd_utc_start=jd_utc_start)
                rho_ecef = target_ecef - station_ecef
                rng = float(np.linalg.norm(rho_ecef))
                range_km[k] = rng
                if rng <= 0.0:
                    elevation_deg[k] = 90.0
                else:
                    enu = enu_rot @ rho_ecef
                    elevation_deg[k] = float(np.rad2deg(np.arcsin(np.clip(enu[2] / rng, -1.0, 1.0))))

                los_ok[k] = bool(_line_of_sight_from_ground(station_eci, target_eci))
                if not los_ok[k]:
                    reason[k] = "line_of_sight"
                    continue
                if elevation_deg[k] < min_elev:
                    reason[k] = "elevation"
                    continue
                if max_range is not None and rng > float(max_range):
                    reason[k] = "range"
                    continue
                access[k] = True
                reason[k] = "ok"

            first_t, last_t = _first_last_time(tt, access)
            access_duration_s = 0.0
            if tt.size > 1:
                access_duration_s = float(np.sum(dt * access[:-1].astype(float)))
            finite_range = range_km[np.isfinite(range_km)]
            finite_elev = elevation_deg[np.isfinite(elevation_deg)]
            station_hist[str(object_id)] = {
                "access": access.tolist(),
                "line_of_sight": los_ok.tolist(),
                "range_km": [_json_float(x) for x in range_km],
                "elevation_deg": [_json_float(x) for x in elevation_deg],
                "reason": reason,
            }
            station_summary[str(object_id)] = {
                "samples": int(tt.size),
                "access_samples": int(np.count_nonzero(access)),
                "access_fraction": float(np.mean(access)) if access.size else 0.0,
                "access_duration_s": access_duration_s,
                "total_duration_s": total_duration_s,
                "first_access_time_s": first_t,
                "last_access_time_s": last_t,
                "min_range_km": float(np.min(finite_range)) if finite_range.size else None,
                "max_elevation_deg": float(np.max(finite_elev)) if finite_elev.size else None,
            }

        histories[station.id] = {
            "station": asdict(station),
            "targets": station_hist,
        }
        summaries[station.id] = station_summary

    return histories, summaries
