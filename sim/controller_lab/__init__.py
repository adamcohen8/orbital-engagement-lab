"""Orbital Engagement Pro controller-bench tools are not included in the public core."""


def _unavailable(*args, **kwargs):
    raise ImportError(
        "Controller-benchmark suites are part of Orbital Engagement Pro. "
        "The public core supports single-run simulation, controllers, estimators, "
        "scenario YAML, and API workflows."
    )


load_controller_bench_config = _unavailable
run_controller_bench = _unavailable

__all__ = ["load_controller_bench_config", "run_controller_bench"]
