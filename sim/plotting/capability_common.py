from __future__ import annotations

from time import perf_counter
from typing import Any, Literal

import matplotlib.pyplot as plt

from sim.plotting.style import (
    role_color,
    show_save_close_oel,
)

PlotMode = Literal["interactive", "save", "both"]
FrameName = Literal["eci", "ecef", "ric_rect", "ric_curv"]
AttitudeFrame = Literal["eci", "ric"]
Layout = Literal["single", "subplots"]

def _show_save_close(fig: plt.Figure, *, mode: PlotMode, out_path: str | None, dpi: int = 150) -> None:
    show_save_close_oel(fig, mode=mode, out_path=out_path, dpi=dpi, plt_module=plt, show_block=False)


def _play_interactive_animation(
    fig: plt.Figure,
    *,
    update: Any,
    frame_count: int,
    interval_ms: float,
) -> None:
    if frame_count <= 0:
        return
    dt_s = max(float(interval_ms) / 1000.0, 1e-4)
    plt.ion()
    fig.show()
    t0 = perf_counter()
    i = 0
    while i < frame_count:
        if not plt.fignum_exists(fig.number):
            break
        elapsed_s = perf_counter() - t0
        target_i = min(int(elapsed_s / dt_s), frame_count - 1)
        if target_i < i:
            target_i = i
        update(target_i)
        fig.canvas.draw_idle()
        plt.pause(0.001)
        i = target_i + 1
    plt.ioff()
    plt.show()


def _object_role_color(object_id: str) -> str | None:
    oid = str(object_id or "").strip().lower()
    if "target" in oid or "chief" in oid:
        return role_color("target")
    if "chaser" in oid or "deputy" in oid:
        return role_color("chaser")
    return None
