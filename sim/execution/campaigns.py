"""Private/product campaign tools are not included in the public export."""


def _unavailable(*args, **kwargs):
    raise ImportError("Monte Carlo campaign tools are available in the private/product distribution.")


can_run_monte_carlo_campaign = _unavailable
prepare_monte_carlo_runs = _unavailable
run_monte_carlo_campaign = _unavailable
run_monte_carlo_runs = _unavailable
run_serial_monte_carlo_runs = _unavailable
sample_monte_carlo_variation = _unavailable
