from __future__ import annotations

import numpy as np

from sim.plotting.output_context import PlotOutputContext
from sim.plotting.style import save_oel_figure
from sim.utils.figure_size import cap_figsize

FIGURE_IDS = ("knowledge_timeline",)


def render_knowledge_outputs(context: PlotOutputContext) -> dict[str, str]:
    cfg = context.cfg
    t_s = context.t_s
    knowledge_hist = context.knowledge_hist
    outdir = context.outdir
    figure_ids = context.figure_ids
    mode = context.mode
    out: dict[str, str] = {}
    if "knowledge_timeline" in figure_ids:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=cap_figsize(10, 5))
        i = 0
        for obs, by_tgt in knowledge_hist.items():
            for tgt, hist in by_tgt.items():
                known = np.any(np.isfinite(hist), axis=1).astype(float)
                ax.plot(t_s, known + i * 1.2, label=f"{obs}->{tgt}")
                i += 1
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Known (offset)")
        ax.set_title("Knowledge Timeline")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        p = outdir / "knowledge_timeline.png"
        if mode in ("save", "both"):
            save_oel_figure(fig, p, dpi=int(cfg.outputs.plots.get("dpi", 150)), artifact_id=p.stem)
            out["knowledge_timeline"] = str(p)
        if mode in ("interactive", "both"):
            plt.show(block=False)
        else:
            plt.close(fig)

    return out
