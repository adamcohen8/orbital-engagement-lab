from __future__ import annotations

from typing import Any


def run_asat_phased_engagement(*args: Any, **kwargs: Any):
    from sim.scenarios.asat_phased_engagement import run_asat_phased_engagement as _run

    return _run(*args, **kwargs)


def run_monte_carlo(*args: Any, **kwargs: Any):
    raise ImportError(
        "Monte Carlo scenario workflows are part of Orbital Engagement Pro. "
        "The public core supports deterministic single-run scenarios."
    )


def __getattr__(name: str) -> Any:
    if name == "MonteCarloConfig":
        raise ImportError("Monte Carlo configuration helpers are part of Orbital Engagement Pro.")
    if name in {"ASATPhasedScenarioConfig", "AgentStrategyConfig", "KnowledgeGateConfig"}:
        import importlib

        _asat = importlib.import_module("sim.scenarios.asat_phased_engagement")
        return getattr(_asat, name)
    raise AttributeError(f"module 'sim.scenarios' has no attribute {name!r}")


__all__ = [
    "run_asat_phased_engagement",
    "run_monte_carlo",
]
