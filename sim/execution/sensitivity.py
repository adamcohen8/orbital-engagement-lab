"""Private/product sensitivity tools are not included in the public export."""


def _unavailable(*args, **kwargs):
    raise ImportError("Sensitivity campaign tools are available in the private/product distribution.")


prepare_sensitivity_runs = _unavailable
run_sensitivity_runs = _unavailable
