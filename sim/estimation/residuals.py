"""Orbital Engagement Pro OD residual helpers are not included in the public core."""


def _unavailable(*args, **kwargs):
    raise ImportError(
        "Orbit-determination residual helpers are part of Orbital Engagement Pro."
    )


ric_position_residuals_m = _unavailable
state_difference_residuals = _unavailable
