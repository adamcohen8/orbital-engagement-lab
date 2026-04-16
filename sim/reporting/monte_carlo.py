"""Private/product Monte Carlo reports are not included in the public export."""


def _unavailable(*args, **kwargs):
    raise ImportError("Monte Carlo reporting is available in the private/product distribution.")


build_monte_carlo_report_payload = _unavailable
write_monte_carlo_report_artifacts = _unavailable
apply_monte_carlo_baseline_comparison = _unavailable
