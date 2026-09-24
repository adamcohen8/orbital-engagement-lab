"""Orbital Engineering Lab Pro covariance analysis tools are not included in the public core."""


def _unavailable(*args, **kwargs):
    raise ImportError(
        "Covariance analysis is part of Orbital Engineering Lab Pro. "
        "The public core supports deterministic single-run simulation and scenario YAML."
    )


compute_covariance_analysis = _unavailable
run_covariance_analysis = _unavailable
