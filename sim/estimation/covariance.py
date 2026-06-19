"""Orbital Engagement Pro parameter-estimation covariance tools are not included in the public core."""


def _unavailable(*args, **kwargs):
    raise ImportError(
        "Batch OD covariance and parameter-estimation reporting are part of Orbital Engagement Pro."
    )


covariance_from_jacobian = _unavailable
correlation_from_covariance = _unavailable
