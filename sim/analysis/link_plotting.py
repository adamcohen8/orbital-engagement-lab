"""Review plots for directed-link evidence."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from sim.plotting.style import role_color

if TYPE_CHECKING:
    from sim.analysis.directed_link import DirectedLinkResult


def write_link_margin_plot(result: DirectedLinkResult, output_path: str | Path) -> Path:
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(9.2, 4.8))
    axes.plot(
        result.samples.time_s,
        result.samples.margin_db,
        color=role_color("actual"),
        linewidth=1.8,
        label="Link margin",
    )
    axes.axhline(
        0.0,
        color=role_color("warning"),
        linewidth=1.2,
        linestyle="--",
        label="Closure threshold",
    )
    axes.fill_between(
        result.samples.time_s,
        result.samples.margin_db,
        0.0,
        where=result.samples.available,
        color=role_color("actual"),
        alpha=0.15,
        interpolate=False,
        label="RF-qualified samples",
    )
    axes.set_title(f"Directed Link Margin — {result.config.link_id}")
    axes.set_xlabel("Analysis time (s)")
    axes.set_ylabel("Margin (dB)")
    axes.grid(True, alpha=0.25)
    axes.legend(loc="best", fontsize=8)
    axes.text(
        0.985,
        0.03,
        "Same-epoch free-space model; sampled evidence",
        transform=axes.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 3.0},
    )
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return destination


__all__ = ["write_link_margin_plot"]
