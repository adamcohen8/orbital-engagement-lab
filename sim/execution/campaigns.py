"""Orbital Engagement Pro campaign tools are not included in the public core."""


def _unavailable(*args, **kwargs):
    raise ImportError(
        "Monte Carlo campaign orchestration is part of Orbital Engagement Pro. "
        "The public core supports deterministic single-run simulation and scenario YAML."
    )


can_run_monte_carlo_campaign = _unavailable
prepare_monte_carlo_runs = _unavailable
run_monte_carlo_campaign = _unavailable
run_monte_carlo_runs = _unavailable
run_serial_monte_carlo_runs = _unavailable
sample_monte_carlo_variation = _unavailable
