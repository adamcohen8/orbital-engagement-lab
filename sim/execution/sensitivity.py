"""Orbital Engineering Lab Pro sensitivity tools are not included in the public core."""


def _unavailable(*args, **kwargs):
    raise ImportError(
        "Sensitivity campaign orchestration is part of Orbital Engineering Lab Pro. "
        "The public core supports deterministic single-run simulation and scenario YAML."
    )


prepare_sensitivity_runs = _unavailable
run_sensitivity_runs = _unavailable
