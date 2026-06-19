"""Orbital Engagement Pro estimated-parameter models are not included in the public core."""


def _unavailable(*args, **kwargs):
    raise ImportError(
        "Estimated-parameter modeling for batch OD is part of Orbital Engagement Pro."
    )


EstimatedParameter = _unavailable
ParameterSet = _unavailable
