# ruff: noqa: F401,F403,F405,I001
from .training_models import *

@dataclass(frozen=True)
class RPOTrainingScore:
    scenario_id: str
    learning_goal: str
    samples: int
    elapsed_s: float
    closest_approach_km: float
    final_range_km: float
    final_goal_error_km: float
    final_relative_speed_km_s: float
    time_inside_keepout_s: float
    approximate_delta_v_m_s: float
    target_delta_v_m_s: float
    burn_axes_satisfied: tuple[str, ...]
    phase_burns_satisfied: tuple[str, ...]
    speed_multiplier_changed: bool
    coast_after_burn_satisfied: bool
    coast_after_burn_s: float
    guided_tutorial_burns_satisfied: tuple[str, ...]
    guided_tutorial_burns_total: int
    guided_tutorial_speed_satisfied: bool
    guided_tutorial_speed_target: float | None
    achieved_time_s: float | None
    min_goal_error_km: float
    final_nmt_radial_amplitude_km: float
    final_nmt_cross_track_amplitude_km: float
    final_nmt_radial_amplitude_error_km: float
    final_nmt_cross_track_amplitude_error_km: float
    final_nmt_drift_velocity_error_km_s: float
    goal_met: bool
    level_passed: bool
    level_failed: bool
    pass_fail_reasons: tuple[str, ...]
    keepout_violation: bool
    hard_speed_limit_violation: bool
    forbidden_region_violation: bool
    forbidden_region_names: tuple[str, ...]
    approach_gate_violation: bool
    approach_gate_names: tuple[str, ...]
    approach_gates_satisfied: int
    approach_gates_total: int
    inspection_gates_satisfied: int
    inspection_gates_total: int
    inspection_gate_names: tuple[str, ...]
    hints: tuple[str, ...]
    final_target_reference_range_km: float = float("nan")
    max_target_reference_range_km: float | None = None
    target_reference_range_violation: bool = False
    sun_angle_violation: bool = False
    sun_angle_constraint_names: tuple[str, ...] = ()
    sun_angle_violation_time_s: float = 0.0
    min_sun_angle_deg: float = float("nan")
    final_sun_angle_deg: float = float("nan")

__all__ = [name for name in globals() if not name.startswith("__")]
