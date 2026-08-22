"""Review plotting for rich sampled Earth-coverage footprints."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from sim.plotting.ground_track_plots import _setup_ground_track_axes
from sim.plotting.style import role_color
from sim.utils.ground_track import split_ground_track_dateline

if TYPE_CHECKING:
    from sim.analysis.rich_coverage import RichCoverageResult


def _selected_samples(sample_count: int, maximum: int = 12) -> np.ndarray:
    if sample_count <= maximum:
        return np.arange(sample_count, dtype=np.int64)
    return np.unique(np.linspace(0, sample_count - 1, maximum, dtype=np.int64))


def _boundary_plot_vectors(
    longitude_deg: np.ndarray,
    latitude_deg: np.ndarray,
    hit: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    longitude = np.asarray(longitude_deg, dtype=float).reshape(-1)
    latitude = np.asarray(latitude_deg, dtype=float).reshape(-1)
    valid = np.asarray(hit, dtype=bool).reshape(-1)
    if longitude.size != latitude.size or longitude.size != valid.size:
        raise ValueError("Boundary longitude, latitude, and hit arrays must match.")
    if not longitude.size:
        return longitude, latitude
    closed_longitude = np.concatenate((longitude, longitude[:1]))
    closed_latitude = np.concatenate((latitude, latitude[:1]))
    closed_valid = np.concatenate((valid, valid[:1]))
    closed_longitude = np.where(closed_valid, closed_longitude, np.nan)
    closed_latitude = np.where(closed_valid, closed_latitude, np.nan)
    return split_ground_track_dateline(
        lon_deg=closed_longitude,
        lat_deg=closed_latitude,
        jump_threshold_deg=180.0,
    )


def write_coverage_footprint_plot(
    result: RichCoverageResult,
    output_path: str | Path,
    *,
    sample_indices: np.ndarray | None = None,
    draw_earth_map: bool = True,
) -> Path:
    """Write a ground-track and sampled FOV-boundary review overlay."""

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    footprint = result.footprint_boundary
    if sample_indices is None:
        selected = _selected_samples(result.times_s.size)
    else:
        raw = np.asarray(sample_indices)
        if raw.ndim != 1 or raw.dtype.kind not in {"i", "u"}:
            raise ValueError("sample_indices must be a one-dimensional integer array.")
        selected = raw.astype(np.int64, copy=False)
        if selected.size == 0:
            raise ValueError("sample_indices must not be empty.")
        if np.any(selected < 0) or np.any(selected >= result.times_s.size):
            raise ValueError("sample_indices contains an index outside the result horizon.")
        if selected.size > 1 and np.any(selected[1:] <= selected[:-1]):
            raise ValueError("sample_indices must be unique and strictly increasing.")

    figure, axes, _ = _setup_ground_track_axes(
        title=f"Rich Coverage Footprints — {result.config.analysis_id}",
        draw_earth_map=draw_earth_map,
    )
    ground_lon, ground_lat = split_ground_track_dateline(
        footprint.subsatellite_longitude_deg,
        footprint.subsatellite_geodetic_latitude_deg,
    )
    axes.plot(
        ground_lon,
        ground_lat,
        color=role_color("actual"),
        linewidth=1.6,
        label="Subsatellite track",
        zorder=4,
    )
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.9, selected.size))
    footprint_label_used = False
    for color, sample_index in zip(colors, selected, strict=True):
        boundary_lon, boundary_lat = _boundary_plot_vectors(
            footprint.boundary_longitude_deg[sample_index],
            footprint.boundary_geodetic_latitude_deg[sample_index],
            footprint.boundary_hit[sample_index],
        )
        if not np.any(np.isfinite(boundary_lon)):
            continue
        axes.plot(
            boundary_lon,
            boundary_lat,
            color=color,
            linewidth=1.0,
            alpha=0.9,
            label="Sampled FOV boundary" if not footprint_label_used else None,
            zorder=5,
        )
        footprint_label_used = True
    axes.scatter(
        footprint.subsatellite_longitude_deg[selected],
        footprint.subsatellite_geodetic_latitude_deg[selected],
        color=role_color("actual"),
        s=14,
        zorder=6,
        label="Boundary epochs",
    )
    axes.legend(loc="lower left", fontsize=8, framealpha=0.85)
    axes.text(
        0.985,
        0.965,
        (
            f"Pattern: {result.config.pattern.kind}\n"
            f"Grid: HEALPix NESTED order {result.config.order}\n"
            "Boundaries are sampled review geometry"
        ),
        transform=axes.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 3.0},
        zorder=10,
    )
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return destination


__all__ = ["write_coverage_footprint_plot"]
