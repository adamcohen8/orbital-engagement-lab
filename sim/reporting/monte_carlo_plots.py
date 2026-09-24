"""Orbital Engineering Lab Pro Monte Carlo plots are not included in the public core."""


def write_monte_carlo_plot_artifacts(*args, **kwargs):
    raise ImportError(
        "Monte Carlo plot reporting is part of Orbital Engineering Lab Pro. "
        "The public core includes single-run outputs and lightweight validation helpers."
    )
