"""Review plots for directed-link evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from sim.plotting.quality import STRICT_AGENT_PLOT_QUALITY, apply_plot_quality_policy
from sim.plotting.style import (
    add_artifact_footer,
    artifact_metadata,
    oel_plot_context,
    role_color,
    save_oel_figure,
)
from sim.runtime_environment import configure_headless_runtime

if TYPE_CHECKING:
    from sim.analysis.directed_link import DirectedLinkResult


def write_link_margin_plot(
    result: DirectedLinkResult,
    output_path: str | Path,
    *,
    scenario_name: str = "",
    style_name: str = "oel_light",
) -> Path:
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    configure_headless_runtime()
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    artifact_id = "directed_link_margin"
    metadata = artifact_metadata(scenario_name=scenario_name, artifact_id=artifact_id)
    with oel_plot_context(style_name=style_name, metadata=metadata):
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
        figure.tight_layout(rect=(0.0, 0.035, 1.0, 1.0))
        add_artifact_footer(figure, metadata=metadata, artifact_id=artifact_id)
        quality = apply_plot_quality_policy(figure, policy=STRICT_AGENT_PLOT_QUALITY)
        save_oel_figure(
            figure,
            destination,
            dpi=180,
            metadata=metadata,
            artifact_id=artifact_id,
            style_name=style_name,
            bbox_inches="tight",
        )
        plt.close(figure)
    destination.with_suffix(".quality.json").write_text(
        json.dumps(quality.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


__all__ = ["write_link_margin_plot"]
