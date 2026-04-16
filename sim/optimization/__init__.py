"""Orbital Engagement Pro optimization tools are not included in the public core."""


def _unavailable(*args, **kwargs):
    raise ImportError(
        "Optimization and gain-tuning workflows are part of Orbital Engagement Pro. "
        "The public core supports single-run simulation, controllers, estimators, "
        "scenario YAML, and API workflows."
    )


class _UnavailablePrivateFeature:
    def __init__(self, *args, **kwargs):
        _unavailable()

    def __call__(self, *args, **kwargs):
        _unavailable()


def __getattr__(name: str):
    _unavailable()


ParameterBound = _UnavailablePrivateFeature
OptimizationResult = _UnavailablePrivateFeature
PSOConfig = _UnavailablePrivateFeature
ParticleSwarmOptimizer = _UnavailablePrivateFeature
ControllerAlgorithm = _UnavailablePrivateFeature
AttitudeTuneCase = _UnavailablePrivateFeature
TuneCaseResult = _UnavailablePrivateFeature
GainTuningResult = _UnavailablePrivateFeature
default_case_cost = _unavailable
default_parameter_bounds = _unavailable
preset_tuning_cases = _unavailable
tune_controller_gains = _unavailable


__all__ = [
    "ParameterBound",
    "OptimizationResult",
    "PSOConfig",
    "ParticleSwarmOptimizer",
    "ControllerAlgorithm",
    "AttitudeTuneCase",
    "TuneCaseResult",
    "GainTuningResult",
    "default_case_cost",
    "default_parameter_bounds",
    "preset_tuning_cases",
    "tune_controller_gains",
]
