from sim.estimation.aoi import AoITrackingEstimator
from sim.estimation.attitude_ekf import AttitudeEKFEstimator
from sim.estimation.joint_ekf import JointStateEKFEstimator
from sim.estimation.joint_state import JointStateEstimator
from sim.estimation.orbit_ekf import OrbitEKFEstimator
from sim.estimation.orbit_ukf import OrbitUKFEstimator


def _unavailable(*args, **kwargs):
    raise ImportError(
        "General dynamics OD and unrestricted estimated-parameter workflows are part of "
        "Orbital Engagement Pro. The public core supports runtime EKF/UKF state estimation "
        "and the bounded sim.tracking_od TDM fit/holdout workflow."
    )


build_dynamics_od_quality_gates = _unavailable
build_orbit_od_parameter_set = _unavailable
selected_orbit_od_parameters = _unavailable
solve_dynamics_orbit_determination = _unavailable

__all__ = [
    "OrbitEKFEstimator",
    "OrbitUKFEstimator",
    "AttitudeEKFEstimator",
    "JointStateEKFEstimator",
    "JointStateEstimator",
    "AoITrackingEstimator",
    "build_dynamics_od_quality_gates",
    "build_orbit_od_parameter_set",
    "selected_orbit_od_parameters",
    "solve_dynamics_orbit_determination",
]
