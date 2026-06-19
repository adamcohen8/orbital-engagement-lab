"""Orbital Engagement Pro dynamics OD workflow tools are not included in the public core."""


def _unavailable(*args, **kwargs):
    raise ImportError(
        "Dynamics orbit determination, fit/holdout OD artifacts, and estimated-parameter "
        "workflows are part of Orbital Engagement Pro. The public core supports deterministic "
        "scenario YAML and runtime EKF/UKF state estimation."
    )


build_dynamics_od_quality_gates = _unavailable
build_orbit_od_parameter_set = _unavailable
selected_orbit_od_parameters = _unavailable
solve_dynamics_orbit_determination = _unavailable
