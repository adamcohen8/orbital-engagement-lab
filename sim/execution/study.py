from __future__ import annotations

from sim.config import SimulationScenarioConfig


def analysis_study_type(cfg: SimulationScenarioConfig) -> str:
    if bool(cfg.analysis.enabled):
        return str(cfg.analysis.study_type or "monte_carlo").strip().lower()
    if bool(cfg.monte_carlo.enabled):
        return "monte_carlo"
    return "single_run"


_analysis_study_type = analysis_study_type
