"""Private/product Monte Carlo plots are not included in the public export."""


def write_monte_carlo_plot_artifacts(*args, **kwargs):
    raise ImportError("Monte Carlo plot reporting is available in the private/product distribution.")
