"""Private/product controller-bench tools are not included in the public export."""


def _unavailable(*args, **kwargs):
    raise ImportError("Controller bench is available in the private/product distribution.")


load_controller_bench_config = _unavailable
run_controller_bench = _unavailable

__all__ = ["load_controller_bench_config", "run_controller_bench"]
