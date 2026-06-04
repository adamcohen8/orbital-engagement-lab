"""Orbital Engagement Pro covariance analysis tools are not included in the public core."""


def _unavailable(*args, **kwargs):
    raise ImportError(
        "Covariance analysis is part of Orbital Engagement Pro. "
        "The public core supports deterministic single-run simulation and scenario YAML."
    )


compute_covariance_analysis = _unavailable
run_covariance_analysis = _unavailable
