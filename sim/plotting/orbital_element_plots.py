from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np

from sim.plotting.single_run_context import _payload_arrays, _save_show_close
from sim.plotting.single_run_math import (
    ORBITAL_ELEMENT_SPECS,
    _classical_orbital_elements_series,
)
from sim.utils.figure_size import cap_figsize

ArrayMap = dict[str, np.ndarray]
NestedArrayMap = dict[str, dict[str, np.ndarray]]
OrbitalElementSeriesCache = dict[
    str,
    tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]],
]
RICSummaryFrame = Literal["rectangular", "curvilinear"]

def _orbital_element_object_ids(truth_by_object: ArrayMap, object_id: str | None) -> list[str]:
    if object_id:
        return [object_id] if object_id in truth_by_object else []
    return sorted(truth_by_object.keys())


def _plot_element_on_axis(
    ax: plt.Axes,
    *,
    t_s: np.ndarray,
    truth_by_object: ArrayMap,
    element_id: str,
    object_id: str | None,
    label_prefix: bool = False,
    series_cache: OrbitalElementSeriesCache | None = None,
) -> bool:
    plotted = False
    for oid in _orbital_element_object_ids(truth_by_object, object_id):
        hist = truth_by_object.get(oid)
        if hist is None:
            continue
        state_history = np.asarray(hist, dtype=float)
        if state_history.ndim == 2:
            state_history = state_history[:, :6]
        cached = None if series_cache is None else series_cache.get(oid)
        if (
            cached is not None
            and cached[0] is hist
            and np.array_equal(cached[1], state_history, equal_nan=True)
        ):
            elements = cached[2]
        else:
            elements = _classical_orbital_elements_series(state_history)
            if series_cache is not None:
                source_snapshot = np.array(state_history, copy=True)
                source_snapshot.setflags(write=False)
                series_cache[oid] = (hist, source_snapshot, elements)
        series = elements.get(element_id)
        if series is None:
            continue
        n = min(t_s.size, series.size)
        if n <= 0:
            continue
        y = np.array(series[:n], dtype=float)
        finite = np.isfinite(y)
        if not np.any(finite):
            continue
        label = f"{oid} {element_id}" if label_prefix else oid
        ax.plot(t_s[:n], y, linewidth=1.2, label=label)
        plotted = True
    return plotted

def plot_orbital_element(
    payload: dict[str, Any] | None = None,
    *,
    element_id: str,
    t_s: np.ndarray | None = None,
    truth_by_object: ArrayMap | None = None,
    object_id: str | None = None,
    orbital_elements_cache: OrbitalElementSeriesCache | None = None,
    out_path: str | Path | None = None,
    show: bool = False,
    close: bool = False,
    dpi: int = 150,
) -> plt.Figure:
    if payload is not None:
        t_s, truth_by_object, _, _, _, _ = _payload_arrays(payload)
    element_key = str(element_id or "").strip()
    if element_key not in ORBITAL_ELEMENT_SPECS:
        valid = ", ".join(sorted(ORBITAL_ELEMENT_SPECS))
        raise ValueError(f"Unknown orbital element '{element_id}'. Valid elements: {valid}")
    t = np.array([] if t_s is None else t_s, dtype=float).reshape(-1)
    truth = dict(truth_by_object or {})
    series_cache = orbital_elements_cache
    title, unit = ORBITAL_ELEMENT_SPECS[element_key]

    fig, ax = plt.subplots(figsize=cap_figsize(10, 5))
    plotted = _plot_element_on_axis(
        ax,
        t_s=t,
        truth_by_object=truth,
        element_id=element_key,
        object_id=object_id,
        series_cache=series_cache,
    )
    if not plotted:
        ax.text(0.5, 0.5, "No valid COE samples available", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(f"{title} Over Time")
    ax.set_xlabel("time (s)")
    ax.set_ylabel(title if not unit else f"{title} ({unit})")
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend(loc="best")
    fig.tight_layout()
    _save_show_close(fig, out_path=out_path, show=show, close=close, dpi=dpi)
    return fig


def plot_orbital_elements_summary(
    payload: dict[str, Any] | None = None,
    *,
    t_s: np.ndarray | None = None,
    truth_by_object: ArrayMap | None = None,
    object_id: str | None = None,
    orbital_elements_cache: OrbitalElementSeriesCache | None = None,
    out_path: str | Path | None = None,
    show: bool = False,
    close: bool = False,
    dpi: int = 150,
) -> plt.Figure:
    if payload is not None:
        t_s, truth_by_object, _, _, _, _ = _payload_arrays(payload)
    t = np.array([] if t_s is None else t_s, dtype=float).reshape(-1)
    truth = dict(truth_by_object or {})
    series_cache = {} if orbital_elements_cache is None else orbital_elements_cache

    fig, axes = plt.subplots(3, 2, figsize=cap_figsize(13, 10), sharex=True)
    for ax, element_key in zip(axes.ravel(), ORBITAL_ELEMENT_SPECS.keys()):
        title, unit = ORBITAL_ELEMENT_SPECS[element_key]
        plotted = _plot_element_on_axis(
            ax,
            t_s=t,
            truth_by_object=truth,
            element_id=element_key,
            object_id=object_id,
            series_cache=series_cache,
        )
        if not plotted:
            ax.text(0.5, 0.5, "No valid samples", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_ylabel(unit or title)
        ax.grid(True, alpha=0.3)
        if plotted:
            ax.legend(loc="best")
    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 1].set_xlabel("time (s)")
    fig.suptitle("Classical Orbital Elements")
    fig.tight_layout()
    _save_show_close(fig, out_path=out_path, show=show, close=close, dpi=dpi)
    return fig


def plot_orbital_elements_angles(
    payload: dict[str, Any] | None = None,
    *,
    t_s: np.ndarray | None = None,
    truth_by_object: ArrayMap | None = None,
    object_id: str | None = None,
    orbital_elements_cache: OrbitalElementSeriesCache | None = None,
    out_path: str | Path | None = None,
    show: bool = False,
    close: bool = False,
    dpi: int = 150,
) -> plt.Figure:
    if payload is not None:
        t_s, truth_by_object, _, _, _, _ = _payload_arrays(payload)
    t = np.array([] if t_s is None else t_s, dtype=float).reshape(-1)
    truth = dict(truth_by_object or {})
    series_cache = {} if orbital_elements_cache is None else orbital_elements_cache
    angle_ids = ("inc", "raan", "argp", "true_anomaly")

    fig, ax = plt.subplots(figsize=cap_figsize(11, 5.5))
    plotted = False
    for element_key in angle_ids:
        plotted = (
            _plot_element_on_axis(
                ax,
                t_s=t,
                truth_by_object=truth,
                element_id=element_key,
                object_id=object_id,
                label_prefix=True,
                series_cache=series_cache,
            )
            or plotted
        )
    if not plotted:
        ax.text(0.5, 0.5, "No valid angular COE samples available", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Orbital Element Angles")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("deg")
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    _save_show_close(fig, out_path=out_path, show=show, close=close, dpi=dpi)
    return fig
