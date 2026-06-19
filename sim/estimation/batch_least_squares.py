"""Orbital Engagement Pro batch OD solver tools are not included in the public core."""


def _unavailable(*args, **kwargs):
    raise ImportError(
        "Batch orbit determination and nonlinear least-squares estimation are part of "
        "Orbital Engagement Pro. The public core supports runtime EKF/UKF state estimation."
    )


BatchLeastSquaresResult = _unavailable
solve_batch_least_squares = _unavailable
