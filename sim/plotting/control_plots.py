from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from sim.plotting.capability_common import _show_save_close
from sim.utils.figure_size import cap_figsize

PlotMode = Literal["interactive", "save", "both"]
FrameName = Literal["eci", "ecef", "ric_rect", "ric_curv"]
AttitudeFrame = Literal["eci", "ric"]
Layout = Literal["single", "subplots"]


def plot_control_commands(
    t_s: np.ndarray,
    u_hist: np.ndarray,
    *,
    layout: Layout = "subplots",
    input_labels: list[str] | None = None,
    title: str = "Control Commands",
    y_label: str = "",
    mode: PlotMode = "interactive",
    out_path: str | None = None,
) -> None:
    u = np.array(u_hist, dtype=float)
    if u.ndim != 2:
        raise ValueError("u_hist must be shape (N, M).")
    m = u.shape[1]
    labels = input_labels if input_labels is not None else [f"u{i}" for i in range(m)]
    if len(labels) != m:
        raise ValueError("input_labels length must match u_hist second dimension.")
    if layout == "single":
        fig, ax = plt.subplots(figsize=cap_figsize(10, 5))
        for i in range(m):
            ax.plot(t_s, u[:, i], label=labels[i])
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(y_label if y_label else "Command")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    else:
        fig, axes = plt.subplots(m, 1, figsize=cap_figsize(10, max(3.0, 2.4 * m)), sharex=True)
        if m == 1:
            axes = [axes]
        for i, ax in enumerate(axes):
            ax.plot(t_s, u[:, i], linewidth=1.3)
            ax.set_ylabel(labels[i] if not y_label else f"{labels[i]} ({y_label})")
            ax.grid(True, alpha=0.3)
        axes[0].set_title(title)
        axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    _show_save_close(fig, mode=mode, out_path=out_path)


def plot_multi_control_commands(
    t_s: np.ndarray,
    u_hist_by_object: dict[str, np.ndarray],
    *,
    component_index: int = 0,
    title: str = "Control Command Overlay",
    y_label: str = "",
    mode: PlotMode = "interactive",
    out_path: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=cap_figsize(10, 5))
    for oid, u in u_hist_by_object.items():
        arr = np.array(u, dtype=float)
        if arr.ndim != 2 or arr.shape[1] <= component_index:
            continue
        ax.plot(t_s, arr[:, component_index], label=oid)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(y_label if y_label else f"u[{component_index}]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    _show_save_close(fig, mode=mode, out_path=out_path)
