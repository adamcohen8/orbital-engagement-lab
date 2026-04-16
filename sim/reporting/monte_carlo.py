"""Orbital Engagement Pro Monte Carlo reports are not included in the public core."""


def _unavailable(*args, **kwargs):
    raise ImportError(
        "Monte Carlo reporting is part of Orbital Engagement Pro. "
        "The public core includes single-run outputs and lightweight validation helpers."
    )


build_monte_carlo_report_payload = _unavailable
write_monte_carlo_report_artifacts = _unavailable
apply_monte_carlo_baseline_comparison = _unavailable
