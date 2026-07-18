# ruff: noqa: F401,F403,F405,I001
from .dashboard_common import *
from .geometry import *
from .prediction import *
from .camera import *

from .dashboard_state import DashboardStateMixin
from .dashboard_layout import DashboardLayoutMixin
from .dashboard_prediction import DashboardPredictionMixin
from .dashboard_hud import DashboardHUDMixin
from .dashboard_overlays import DashboardOverlayMixin
from .dashboard_camera import DashboardCameraMixin
from .dashboard_text import DashboardTextMixin

@dataclass
class PygameRPODashboard(
    DashboardStateMixin, DashboardLayoutMixin, DashboardPredictionMixin,
    DashboardHUDMixin, DashboardOverlayMixin, DashboardCameraMixin, DashboardTextMixin,
):
    target_object_id: str = "target"
    chaser_object_id: str = "chaser"
    controlled_object_id: str | None = None
    reference_object_id: str | None = None
    relative_frame: str = "ric"
    keepout_radius_km: float | None = None
    goal_range_km: float | None = None
    goal_range_tolerance_km: float | None = None
    goal_radius_km: float | None = None
    hard_speed_limit_radius_km: float | None = None
    hard_speed_limit_km_s: float | None = None
    goal_relative_ric_km: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    goal_nmt_radial_amplitude_km: float | None = None
    goal_nmt_cross_track_amplitude_km: float = 0.0
    goal_nmt_cross_track_phase_deg: float = 0.0
    goal_nmt_center_ric_km: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    goal_nmt_element_tolerance_km: float | None = None
    fullscreen: bool = True
    max_history: int = 900
    title: str = "Orbital Engagement Lab - RPO Trainer"
    coast_prediction_horizon_s: float = 300.0
    coast_prediction_orbit_fraction: float | None = 1.0
    coast_prediction_dt_s: float = 10.0
    coast_prediction_model: str = "hcw"
    cr3bp_projection_mode: str = "nonlinear"
    cr3bp_coast_prediction_horizon_s: float = 21600.0
    cr3bp_active_prediction_horizon_s: float | None = None
    cr3bp_coast_prediction_horizon_mode: str = "default"
    cr3bp_coast_prediction_dt_s: float = 300.0
    cr3bp_prediction_coast_update_interval_s: float = CR3BP_PREDICTION_COAST_UPDATE_INTERVAL_S
    target_coast_prediction_horizon_s: float | None = None
    target_coast_prediction_dt_s: float | None = None
    show_target_coast_prediction: bool = False
    burn_marker_threshold_km_s2: float = 1.0e-12
    forbidden_regions: tuple[ForbiddenRegionConfig, ...] = ()
    approach_gates: tuple[ApproachGateConfig, ...] = ()
    inspection_gates: tuple[InspectionGateConfig, ...] = ()
    sun_angle_constraints: tuple[SunAngleConstraintConfig, ...] = ()
    plot_overlays_in_zoom: bool = True
    plot_overlays_in_zoom_by_plane: dict[str, bool] = field(default_factory=dict)
    plot_prediction_in_zoom: bool = False
    plot_prediction_zoom_max_span_km: float | None = None
    plot_prediction_full_trajectory_only: bool = False
    plot_axis_scale: dict[str, tuple[float, float]] = field(default_factory=dict)
    plot_fixed_axis_half_span_km: dict[str, tuple[float | None, float | None]] = field(default_factory=dict)
    plot_equal_axis_scale_planes: tuple[str, ...] = ()
    target_centered_plot_planes: tuple[str, ...] = ()
    target_centered_plot_axes: dict[str, tuple[str, ...]] = field(default_factory=dict)
    proximity_ring_plot_planes: tuple[str, ...] = ("RI", "RC", "IC")
    max_target_reference_range_km: float | None = None
    target_reference_object_id: str | None = None
    visual_extrapolation_enabled: bool = True
    visual_extrapolation_max_sim_s: float = VISUAL_EXTRAPOLATION_MAX_SIM_S
    camera_mode: str = "reference"
    camera_rule_mode: str = "default"
    camera_rule_toggle_enabled: bool = False
    target_sprite_path: Path | str | None = None
    chaser_sprite_path: Path | str | None = None
    target_sprite_diameter_km: float = SATELLITE_SPRITE_DIAMETER_KM
    chaser_sprite_diameter_km: float = SATELLITE_SPRITE_DIAMETER_KM
    tutorial_target_path_ric: np.ndarray = field(default_factory=lambda: np.empty((0, 6), dtype=float))
    live_prediction_accel_ric_km_s2: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    live_prediction_elapsed_s: float = 0.0
    operator_projection_transition_duration_s: float = 1.15
    plot_view_modes: dict[str, str] = field(default_factory=dict)
    mission_time_budget_s: float | None = None
    frame_convention: FrameConvention = FrameConvention()

    pass
