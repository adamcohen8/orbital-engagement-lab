from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Polygon, Rectangle

from sim.dynamics.orbit.epoch import julian_date_to_datetime
from sim.dynamics.orbit.frames import FrameContext
from sim.plotting.capability_common import _object_role_color
from sim.plotting.style import (
    OEL_DARK_PALETTE,
    OEL_LIGHT_PALETTE,
    current_style_name,
    role_color,
    save_oel_animation,
)
from sim.utils.figure_size import cap_figsize
from sim.utils.ground_track import ground_track_from_eci_history, split_ground_track_dateline

try:
    import cartopy.crs as ccrs  # type: ignore
    import cartopy.feature as cfeature  # type: ignore

    _HAS_CARTOPY = True
except Exception:
    _HAS_CARTOPY = False

PlotMode = Literal["interactive", "save", "both"]
FrameName = Literal["eci", "ecef", "ric_rect", "ric_curv"]
AttitudeFrame = Literal["eci", "ric"]
Layout = Literal["single", "subplots"]

def _map_colors() -> dict[str, str]:
    style_key = current_style_name()
    if style_key == "oel_dark":
        return {
            "ocean": OEL_DARK_PALETTE["panel"],
            "land": OEL_DARK_PALETTE["panel_alt"],
            "land_edge": OEL_DARK_PALETTE["edge"],
            "coast": OEL_DARK_PALETTE["muted_text"],
            "grid": OEL_DARK_PALETTE["grid"],
        }
    if style_key == "oel_light":
        return {
            "ocean": OEL_LIGHT_PALETTE["panel_alt"],
            "land": "#D9E8C6",
            "land_edge": OEL_LIGHT_PALETTE["edge"],
            "coast": OEL_LIGHT_PALETTE["muted_text"],
            "grid": OEL_LIGHT_PALETTE["grid"],
        }
    return {
        "ocean": "#cfe8ff",
        "land": "#dbe7c9",
        "land_edge": "#8aa27a",
        "coast": "#5e6f57",
        "grid": "gray",
    }


def _draw_stylized_earth_map(ax: plt.Axes) -> None:
    colors = _map_colors()
    ocean = Rectangle((-180.0, -90.0), 360.0, 180.0, facecolor=colors["ocean"], edgecolor="none", zorder=0)
    ax.add_patch(ocean)
    continents = [
        [
            (-168, 72),
            (-145, 68),
            (-130, 55),
            (-123, 50),
            (-118, 34),
            (-105, 24),
            (-97, 17),
            (-83, 20),
            (-80, 27),
            (-66, 45),
            (-82, 55),
            (-110, 72),
        ],
        [
            (-81, 12),
            (-72, 8),
            (-66, -5),
            (-62, -18),
            (-58, -33),
            (-54, -54),
            (-69, -56),
            (-76, -40),
            (-78, -20),
            (-81, 0),
        ],
        [
            (-18, 35),
            (2, 37),
            (20, 33),
            (33, 23),
            (40, 8),
            (47, -12),
            (40, -28),
            (28, -35),
            (13, -35),
            (3, -24),
            (-4, -6),
            (-9, 14),
            (-16, 28),
        ],
        [
            (-10, 36),
            (8, 46),
            (30, 56),
            (55, 64),
            (90, 72),
            (120, 66),
            (145, 58),
            (170, 50),
            (155, 40),
            (120, 24),
            (102, 12),
            (80, 8),
            (55, 16),
            (30, 26),
            (18, 32),
            (5, 38),
        ],
        [(72, 23), (85, 22), (95, 15), (103, 8), (106, 2), (102, -4), (90, 2), (82, 8), (75, 16)],
        [(113, -12), (132, -11), (150, -20), (154, -32), (145, -42), (129, -42), (116, -33), (111, -22)],
        [(-56, 82), (-42, 82), (-28, 74), (-34, 62), (-49, 60), (-60, 68)],
        [(-180, -62), (-120, -64), (-60, -66), (0, -68), (60, -66), (120, -64), (180, -62), (180, -90), (-180, -90)],
    ]
    for poly in continents:
        ax.add_patch(
            Polygon(
                poly,
                closed=True,
                facecolor=colors["land"],
                edgecolor=colors["land_edge"],
                linewidth=0.6,
                zorder=1,
            )
        )


def _setup_ground_track_axes(
    *,
    title: str,
    draw_earth_map: bool,
) -> tuple[plt.Figure, Any, bool]:
    colors = _map_colors()
    if draw_earth_map and _HAS_CARTOPY:
        fig = plt.figure(figsize=cap_figsize(11, 5))
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        ax.set_global()
        ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor=colors["ocean"], zorder=0)
        ax.add_feature(
            cfeature.LAND.with_scale("110m"),
            facecolor=colors["land"],
            edgecolor=colors["land_edge"],
            linewidth=0.4,
            zorder=1,
        )
        ax.coastlines(resolution="110m", linewidth=0.5, color=colors["coast"], zorder=2)
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.4, color=colors["grid"], alpha=0.4, linestyle="-"
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 8}
        gl.ylabel_style = {"size": 8}
        ax.set_title(title)
        return fig, ax, True

    fig, ax = plt.subplots(figsize=cap_figsize(11, 5))
    if draw_earth_map:
        _draw_stylized_earth_map(ax)
    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(-90.0, 90.0)
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(np.arange(-180, 181, 30))
    ax.set_yticks(np.arange(-90, 91, 15))
    for xv in np.arange(-180, 181, 30):
        ax.axvline(xv, color=colors["grid"], linewidth=0.35, alpha=0.35, zorder=0)
    for yv in np.arange(-90, 91, 15):
        ax.axhline(yv, color=colors["grid"], linewidth=0.35, alpha=0.35, zorder=0)
    return fig, ax, False

def animate_ground_track(
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    *,
    t_s: np.ndarray | None = None,
    jd_utc_start: float | None = None,
    mode: PlotMode = "interactive",
    out_path: str | None = None,
    fps: float = 30.0,
    speed_multiple: float = 10.0,
    draw_earth_map: bool = True,
    frame_stride: int = 1,
) -> None:
    lon_p, lat_p = split_ground_track_dateline(lon_deg=lon_deg, lat_deg=lat_deg, jump_threshold_deg=180.0)
    t_arr = np.array(t_s, dtype=float).reshape(-1) if t_s is not None else np.arange(len(lon_deg), dtype=float)
    if t_arr.size < len(lon_deg):
        t_arr = np.pad(t_arr, (0, len(lon_deg) - t_arr.size), mode="edge")
    fig, ax, is_cartopy = _setup_ground_track_axes(title="Ground Track Animation", draw_earth_map=draw_earth_map)
    if is_cartopy:
        (line,) = ax.plot(
            [],
            [],
            linewidth=1.4,
            color=role_color("actual"),
            transform=ccrs.PlateCarree(),
            zorder=3,
        )
        (dot,) = ax.plot(
            [],
            [],
            marker="o",
            markersize=4,
            color=role_color("actual"),
            transform=ccrs.PlateCarree(),
            zorder=4,
        )
    else:
        (line,) = ax.plot([], [], linewidth=1.4, color=role_color("actual"), zorder=3)
        (dot,) = ax.plot([], [], marker="o", markersize=4, color=role_color("actual"), zorder=4)
    time_text = ax.text(
        0.01,
        0.99,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        zorder=10,
    )

    stride = int(max(frame_stride, 1))
    frame_ids = np.arange(0, len(lon_p), stride, dtype=int)
    if frame_ids.size == 0 or frame_ids[-1] != (len(lon_p) - 1):
        frame_ids = np.append(frame_ids, len(lon_p) - 1)

    def update(i: int):
        idx = int(frame_ids[i])
        line.set_data(lon_p[: idx + 1], lat_p[: idx + 1])
        dot.set_data([lon_p[idx]], [lat_p[idx]])
        t_now = float(t_arr[min(idx, t_arr.size - 1)])
        if jd_utc_start is not None:
            dt_utc = julian_date_to_datetime(float(jd_utc_start) + t_now / 86400.0)
            time_text.set_text(f"UTC: {dt_utc.strftime('%Y-%m-%d %H:%M:%S')}\nSim t: {t_now:.1f} s")
        else:
            time_text.set_text(f"Sim t: {t_now:.1f} s")
        return [line, dot, time_text]

    interval_ms = 1000.0 / max(float(fps) * max(speed_multiple, 1e-6), 1e-3)
    if mode in ("interactive", "both"):
        # Explicit interactive loop is more reliable than backend animation playback in IDE windows.
        plt.ion()
        fig.show()
        for i in range(int(frame_ids.size)):
            update(i)
            fig.canvas.draw_idle()
            plt.pause(interval_ms / 1000.0)
        plt.ioff()
        plt.show()
    if mode in ("save", "both"):
        ani = animation.FuncAnimation(fig, update, frames=int(frame_ids.size), interval=interval_ms, blit=False)
        if out_path is None:
            raise ValueError("out_path is required when mode is 'save' or 'both'.")
        p = Path(out_path)
        try:
            save_oel_animation(ani, fig, p, fps=fps, artifact_id=p.stem)
        except Exception as exc:
            print(f"Warning: failed to save animation ({exc}).")
        del ani
    plt.close(fig)


def animate_multi_ground_track(
    t_s: np.ndarray,
    truth_hist_by_object: dict[str, np.ndarray],
    *,
    jd_utc_start: float | None = None,
    mode: PlotMode = "interactive",
    out_path: str | None = None,
    fps: float = 30.0,
    speed_multiple: float = 10.0,
    draw_earth_map: bool = True,
    frame_stride: int = 1,
    frame_context: FrameContext | None = None,
) -> None:
    tracks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    tracks_t: dict[str, np.ndarray] = {}
    n_frames = 0
    for oid, hist in truth_hist_by_object.items():
        arr = np.array(hist, dtype=float)
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 3:
            continue
        mask = np.isfinite(arr[:, 0])
        if not np.any(mask):
            continue
        lat, lon, _ = ground_track_from_eci_history(
            arr[:, :3],
            t_s=t_s,
            jd_utc_start=jd_utc_start,
            frame_context=frame_context,
        )
        lon_p, lat_p = split_ground_track_dateline(lon_deg=lon, lat_deg=lat, jump_threshold_deg=180.0)
        tracks[oid] = (lon_p, lat_p)
        t_local = np.array(t_s, dtype=float).reshape(-1)
        if t_local.size < arr.shape[0]:
            t_local = np.pad(t_local, (0, arr.shape[0] - t_local.size), mode="edge")
        # For inserted NaNs at dateline splits, approximate expanded time vector linearly.
        if lon_p.size == t_local.size:
            tracks_t[oid] = t_local
        else:
            tracks_t[oid] = np.linspace(float(t_local[0]), float(t_local[-1]), num=lon_p.size, endpoint=True)
        n_frames = max(n_frames, int(lon_p.size))

    if not tracks:
        return

    fig, ax, is_cartopy = _setup_ground_track_axes(
        title="Ground Track Animation (Multi-Object)",
        draw_earth_map=draw_earth_map,
    )

    line_by_obj: dict[str, Any] = {}
    dot_by_obj: dict[str, Any] = {}
    for oid in sorted(tracks.keys()):
        color = _object_role_color(oid)
        if is_cartopy:
            (line,) = ax.plot([], [], linewidth=1.4, label=oid, color=color, transform=ccrs.PlateCarree(), zorder=3)
            (dot,) = ax.plot([], [], marker="o", markersize=4, color=color, transform=ccrs.PlateCarree(), zorder=4)
        else:
            (line,) = ax.plot([], [], linewidth=1.4, label=oid, color=color, zorder=3)
            (dot,) = ax.plot([], [], marker="o", markersize=4, color=color, zorder=4)
        line_by_obj[oid] = line
        dot_by_obj[oid] = dot
    ax.legend(loc="best")
    time_text = ax.text(
        0.01,
        0.99,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        zorder=10,
    )

    stride = int(max(frame_stride, 1))
    frame_ids = np.arange(0, max(n_frames, 1), stride, dtype=int)
    if frame_ids.size == 0 or frame_ids[-1] != (max(n_frames, 1) - 1):
        frame_ids = np.append(frame_ids, max(n_frames, 1) - 1)

    def update(i: int):
        artists = []
        frame_i = int(frame_ids[i])
        t_now = 0.0
        for oid, (lon_p, lat_p) in tracks.items():
            idx = min(frame_i, lon_p.size - 1)
            line_by_obj[oid].set_data(lon_p[: idx + 1], lat_p[: idx + 1])
            dot_by_obj[oid].set_data([lon_p[idx]], [lat_p[idx]])
            t_track = tracks_t.get(oid)
            if t_track is not None and t_track.size > 0:
                t_now = max(t_now, float(t_track[min(idx, t_track.size - 1)]))
            artists.extend([line_by_obj[oid], dot_by_obj[oid]])
        if jd_utc_start is not None:
            dt_utc = julian_date_to_datetime(float(jd_utc_start) + t_now / 86400.0)
            time_text.set_text(f"UTC: {dt_utc.strftime('%Y-%m-%d %H:%M:%S')}\nSim t: {t_now:.1f} s")
        else:
            time_text.set_text(f"Sim t: {t_now:.1f} s")
        artists.append(time_text)
        return artists

    interval_ms = 1000.0 / max(float(fps) * max(speed_multiple, 1e-6), 1e-3)
    if mode in ("interactive", "both"):
        # Explicit interactive loop is more reliable than backend animation playback in IDE windows.
        plt.ion()
        fig.show()
        for i in range(int(frame_ids.size)):
            update(i)
            fig.canvas.draw_idle()
            plt.pause(interval_ms / 1000.0)
        plt.ioff()
        plt.show()
    if mode in ("save", "both"):
        ani = animation.FuncAnimation(fig, update, frames=int(frame_ids.size), interval=interval_ms, blit=False)
        if out_path is None:
            raise ValueError("out_path is required when mode is 'save' or 'both'.")
        p = Path(out_path)
        try:
            save_oel_animation(ani, fig, p, fps=fps, artifact_id=p.stem)
        except Exception as exc:
            print(f"Warning: failed to save animation ({exc}).")
        del ani
    plt.close(fig)
