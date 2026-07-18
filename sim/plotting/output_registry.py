from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sim.plotting.control_outputs import FIGURE_IDS as CONTROL_FIGURE_IDS
from sim.plotting.control_outputs import render_control_outputs
from sim.plotting.knowledge_outputs import FIGURE_IDS as KNOWLEDGE_FIGURE_IDS
from sim.plotting.knowledge_outputs import render_knowledge_outputs
from sim.plotting.output_context import PlotOutputContext
from sim.plotting.rocket_outputs import FIGURE_IDS as ROCKET_FIGURE_IDS
from sim.plotting.rocket_outputs import render_rocket_outputs
from sim.plotting.summary_outputs import FIGURE_IDS as SUMMARY_FIGURE_IDS
from sim.plotting.summary_outputs import render_summary_outputs
from sim.plotting.trajectory_outputs import FIGURE_IDS as TRAJECTORY_FIGURE_IDS
from sim.plotting.trajectory_outputs import render_trajectory_outputs

OutputRenderer = Callable[[PlotOutputContext], dict[str, str]]


@dataclass(frozen=True)
class RendererFamily:
    """Named renderer entry kept explicit for discoverability and stable ordering."""

    name: str
    figure_ids: tuple[str, ...]
    render: OutputRenderer


PLOT_RENDERER_FAMILIES: tuple[RendererFamily, ...] = (
    RendererFamily("summary", SUMMARY_FIGURE_IDS, render_summary_outputs),
    RendererFamily("trajectory", TRAJECTORY_FIGURE_IDS, render_trajectory_outputs),
    RendererFamily("rocket", ROCKET_FIGURE_IDS, render_rocket_outputs),
    RendererFamily("control", CONTROL_FIGURE_IDS, render_control_outputs),
    RendererFamily("knowledge", KNOWLEDGE_FIGURE_IDS, render_knowledge_outputs),
)


def render_plot_outputs(context: PlotOutputContext) -> dict[str, str]:
    out: dict[str, str] = {}
    for family in PLOT_RENDERER_FAMILIES:
        out.update(family.render(context))
    return out
