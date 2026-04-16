"""Orbital Engagement Pro Monte Carlo plots are not included in the public core."""


def write_monte_carlo_plot_artifacts(*args, **kwargs):
    raise ImportError(
        "Monte Carlo plot reporting is part of Orbital Engagement Pro. "
        "The public core includes single-run outputs and lightweight validation helpers."
    )
