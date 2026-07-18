from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np

from sim.plotting.single_run_context import _array_map, _payload_arrays, _save_show_close
from sim.plotting.single_run_math import _quat_error_series_deg, _thrust_alignment_error_deg_series
from sim.utils.figure_size import cap_figsize

ArrayMap = dict[str, np.ndarray]
NestedArrayMap = dict[str, dict[str, np.ndarray]]
OrbitalElementSeriesCache = dict[
    str,
    tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]],
]
RICSummaryFrame = Literal["rectangular", "curvilinear"]

def plot_attitude_control_summary(
    payload: dict[str, Any] | None = None,
    *,
    t_s: np.ndarray | None = None,
    truth_by_object: ArrayMap | None = None,
    thrust_by_object: ArrayMap | None = None,
    desired_attitude_by_object: ArrayMap | None = None,
    thrust_axis_body_by_object: dict[str, np.ndarray] | None = None,
    object_id: str | None = None,
    out_path: str | Path | None = None,
    show: bool = False,
    close: bool = False,
    dpi: int = 150,
) -> plt.Figure:
    if payload is not None:
        t_s, truth_by_object, thrust_by_object, _, _, _ = _payload_arrays(payload)
        desired_attitude_by_object = _array_map(payload.get("desired_attitude_by_object", {}))
    t = np.array([] if t_s is None else t_s, dtype=float).reshape(-1)
    truth = dict(truth_by_object or {})
    thrust = dict(thrust_by_object or {})
    desired = dict(desired_attitude_by_object or {})
    axes_by_object = dict(thrust_axis_body_by_object or {})
    ids = [object_id] if object_id and object_id in truth else sorted(truth.keys())

    fig, axes = plt.subplots(4, 1, figsize=cap_figsize(12, 10), sharex=True)
    plotted = False
    for oid in ids:
        hist = truth.get(oid)
        if hist is None or hist.ndim != 2 or hist.shape[1] < 13:
            continue
        n = min(hist.shape[0], t.size)
        if n <= 0:
            continue
        q_err = _quat_error_series_deg(
            truth_hist=hist,
            desired_attitude_hist=desired.get(oid),
            n_s=n,
        )
        rate_norm = np.linalg.norm(np.nan_to_num(hist[:n, 10:13], nan=0.0), axis=1)
        u = thrust.get(oid, np.zeros((n, 3), dtype=float))
        n_u = min(u.shape[0], n)
        thrust_mag = np.full(n, np.nan, dtype=float)
        if n_u > 0 and u.ndim == 2 and u.shape[1] >= 3:
            thrust_mag[:n_u] = np.linalg.norm(np.nan_to_num(u[:n_u, :3], nan=0.0), axis=1)
            align = _thrust_alignment_error_deg_series(
                truth_hist=hist,
                thrust_hist=u,
                thrust_axis_body=axes_by_object.get(oid, np.array([1.0, 0.0, 0.0], dtype=float)),
                n_s=n_u,
            )
        else:
            align = np.full(n, np.nan, dtype=float)
        axes[0].plot(t[:n], q_err, linewidth=1.2, label=oid)
        axes[1].plot(t[:n], rate_norm, linewidth=1.2, label=oid)
        axes[2].plot(t[:n], thrust_mag, linewidth=1.2, label=oid)
        axes[3].plot(t[: align.size], align, linewidth=1.2, label=oid)
        plotted = True

    if not plotted:
        for ax in axes:
            ax.text(
                0.5,
                0.5,
                "No attitude-control history available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    axes[0].set_title("Quaternion Tracking Error")
    axes[0].set_ylabel("deg")
    axes[1].set_title("Body Rate Norm")
    axes[1].set_ylabel("rad/s")
    axes[2].set_title("Applied Thrust Magnitude")
    axes[2].set_ylabel("km/s^2")
    axes[3].set_title("Thrust Alignment Error")
    axes[3].set_ylabel("deg")
    axes[3].set_xlabel("time (s)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        if plotted:
            ax.legend(loc="best")
    fig.tight_layout()
    _save_show_close(fig, out_path=out_path, show=show, close=close, dpi=dpi)
    return fig
