"""Private/product sensitivity reports are not included in the public export."""


def _unavailable(*args, **kwargs):
    raise ImportError("Sensitivity reporting is available in the private/product distribution.")


analysis_metrics = _unavailable
extract_analysis_metric = _unavailable
extract_analysis_metrics = _unavailable
aggregate_sensitivity_parameter_runs = _unavailable
build_sensitivity_report_payload = _unavailable
write_sensitivity_summary_artifact = _unavailable
