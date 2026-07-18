from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np

from sim.dynamics.orbit.frames import frame_context_from_mapping
from sim.plotting.single_run_context import _payload_arrays, _save_show_close
from sim.utils.figure_size import cap_figsize
from sim.utils.ground_track import ground_track_from_eci_history, split_ground_track_dateline
from sim.utils.plotting_capabilities import _setup_ground_track_axes

ArrayMap = dict[str, np.ndarray]
NestedArrayMap = dict[str, dict[str, np.ndarray]]
OrbitalElementSeriesCache = dict[
    str,
    tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]],
]
RICSummaryFrame = Literal["rectangular", "curvilinear"]

def plot_ground_track_from_payload(
    payload: dict[str, Any] | None = None,
    *,
    t_s: np.ndarray | None = None,
    truth_by_object: ArrayMap | None = None,
    jd_utc_start: float | None = None,
    object_id: str | None = None,
    draw_earth_map: bool = False,
    out_path: str | Path | None = None,
    show: bool = False,
    close: bool = False,
    dpi: int = 150,
) -> plt.Figure:
    if payload is not None:
        t_s, truth_by_object, _, _, _, _ = _payload_arrays(payload)
        frame_context = frame_context_from_mapping(
            dict(payload.get("frame_provenance", {}) or {}),
            jd_utc_start=jd_utc_start,
            source="payload",
        )
    else:
        frame_context = frame_context_from_mapping({}, jd_utc_start=jd_utc_start, source="plot")
    t = np.array([] if t_s is None else t_s, dtype=float).reshape(-1)
    truth = dict(truth_by_object or {})
    ids = [object_id] if object_id and object_id in truth else sorted(truth.keys())
    fig, ax, is_cartopy = _setup_ground_track_axes(title="Ground Track", draw_earth_map=bool(draw_earth_map))
    for oid in ids:
        hist = truth.get(oid)
        if hist is None or hist.shape[1] < 3:
            continue
        n = min(hist.shape[0], t.size)
        lat, lon, _ = ground_track_from_eci_history(
            hist[:n, :3],
            t_s=t[:n],
            jd_utc_start=jd_utc_start,
            frame_context=frame_context,
        )
        lon_p, lat_p = split_ground_track_dateline(lon_deg=lon, lat_deg=lat, jump_threshold_deg=180.0)
        if is_cartopy:
            import cartopy.crs as ccrs  # type: ignore

            ax.plot(lon_p, lat_p, linewidth=1.4, label=oid, transform=ccrs.PlateCarree(), zorder=3)
        else:
            ax.plot(lon_p, lat_p, linewidth=1.4, label=oid)
        finite = np.isfinite(lon) & np.isfinite(lat)
        idx = np.where(finite)[0]
        if idx.size:
            if is_cartopy:
                ax.scatter(
                    [lon[idx[0]]],
                    [lat[idx[0]]],
                    color="green",
                    s=18,
                    transform=ccrs.PlateCarree(),
                    zorder=4,
                )
                ax.scatter(
                    [lon[idx[-1]]],
                    [lat[idx[-1]]],
                    color="red",
                    s=18,
                    transform=ccrs.PlateCarree(),
                    zorder=4,
                )
            else:
                ax.scatter([lon[idx[0]]], [lat[idx[0]]], color="green", s=18)
                ax.scatter([lon[idx[-1]]], [lat[idx[-1]]], color="red", s=18)
    if ids:
        ax.legend(loc="best")
    fig.tight_layout()
    _save_show_close(fig, out_path=out_path, show=show, close=close, dpi=dpi)
    return fig

def plot_ground_station_access(
    payload: dict[str, Any] | None = None,
    *,
    t_s: np.ndarray | None = None,
    ground_station_access: dict[str, Any] | None = None,
    out_path: str | Path | None = None,
    show: bool = False,
    close: bool = False,
    dpi: int = 150,
) -> plt.Figure:
    if payload is not None:
        t_s = np.array(payload.get("time_s", []), dtype=float).reshape(-1)
        ground_station_access = dict(payload.get("ground_station_access", {}) or {})
    t = np.array([] if t_s is None else t_s, dtype=float).reshape(-1)
    access_root = dict(ground_station_access or {})

    pairs: list[tuple[str, str, dict[str, Any]]] = []
    for station_id, station_payload in sorted(access_root.items()):
        targets = dict(dict(station_payload or {}).get("targets", {}) or {})
        for target_id, target_payload in sorted(targets.items()):
            pairs.append((str(station_id), str(target_id), dict(target_payload or {})))

    fig, axes = plt.subplots(3, 1, figsize=cap_figsize(12, 9), sharex=True)
    if not pairs or t.size == 0:
        for ax in axes:
            ax.text(
                0.5,
                0.5,
                "No ground-station access history available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        fig.tight_layout()
        _save_show_close(fig, out_path=out_path, show=show, close=close, dpi=dpi)
        return fig

    y_ticks = []
    y_labels = []
    for row, (station_id, target_id, target_payload) in enumerate(pairs):
        label = f"{station_id}->{target_id}"
        access = np.array(target_payload.get("access", []), dtype=float).reshape(-1)
        los = np.array(target_payload.get("line_of_sight", []), dtype=float).reshape(-1)
        elev_values = list(target_payload.get("elevation_deg", []) or [])
        range_values = list(target_payload.get("range_km", []) or [])
        elev = np.array([float("nan") if value is None else float(value) for value in elev_values], dtype=float)
        rng = np.array([float("nan") if value is None else float(value) for value in range_values], dtype=float)
        n_access = min(t.size, access.size)
        if n_access:
            axes[0].step(t[:n_access], access[:n_access] + row * 1.3, where="post", linewidth=1.5)
        n_los = min(t.size, los.size)
        if n_los:
            axes[0].step(
                t[:n_los],
                0.35 * los[:n_los] + row * 1.3,
                where="post",
                linewidth=0.9,
                linestyle=":",
                alpha=0.65,
            )
        n_elev = min(t.size, elev.size)
        if n_elev:
            axes[1].plot(t[:n_elev], elev[:n_elev], linewidth=1.2, label=label)
        n_rng = min(t.size, rng.size)
        if n_rng:
            axes[2].plot(t[:n_rng], rng[:n_rng], linewidth=1.2, label=label)
        y_ticks.append(row * 1.3 + 0.5)
        y_labels.append(label)

    axes[0].set_title("Ground-Station Access Timeline")
    axes[0].set_ylabel("access")
    axes[0].set_yticks(y_ticks)
    axes[0].set_yticklabels(y_labels)
    axes[0].set_ylim(-0.2, max(y_ticks) + 0.95)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Elevation")
    axes[1].set_ylabel("deg")
    axes[1].grid(True, alpha=0.3)
    if axes[1].lines:
        axes[1].legend(loc="best")

    axes[2].set_title("Slant Range")
    axes[2].set_ylabel("km")
    axes[2].set_xlabel("time (s)")
    axes[2].grid(True, alpha=0.3)
    if axes[2].lines:
        axes[2].legend(loc="best")

    fig.tight_layout()
    _save_show_close(fig, out_path=out_path, show=show, close=close, dpi=dpi)
    return fig
