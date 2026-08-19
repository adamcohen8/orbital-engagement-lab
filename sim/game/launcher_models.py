# ruff: noqa: F401,F403,F405,I001
from .launcher_common import *

@dataclass(frozen=True)
class GameScenarioOption:
    path: Path
    scenario_id: str
    title: str
    description: str
    learning_goal: str
    player_brief: str
    pass_criteria: tuple[str, ...]
    instructor_notes: tuple[str, ...]
    difficulty: str
    time_budget_s: float | None
    delta_v_budget_m_s: float | None
    goal_speed_km_s: float | None
    target_delta_v_budget_m_s: float | None
    completed_difficulties: tuple[str, ...]
    high_score: int
    level_number: int
    goal_range_km: float | None = None
    controlled_object_id: str = "chaser"
    target_object_id: str = "target"


@dataclass(frozen=True)
class GameProgressRecord:
    completed_difficulties: tuple[str, ...] = ()
    high_score: int = 0


@dataclass(frozen=True)
class GameSettings:
    frame_convention: FrameConvention = FrameConvention()
    presentation_mode: str = "compatibility"
    ask_frame_convention_on_launch: bool = True
    last_game_mode: str | None = None
    operator_burn_scripts: dict[str, OperatorBurnPlan] = field(default_factory=dict)


@dataclass(frozen=True)
class GameLaunchSelection:
    path: Path
    difficulty: str
    music_enabled: bool = True
    record_video: bool = False
    mode: str = "pilot"
    frame_convention: FrameConvention = FrameConvention()
    presentation_mode: str = "compatibility"
    operator_burn_plan: OperatorBurnPlan | None = None
    skip_initial_briefing: bool = False


@dataclass(frozen=True)
class OperatorPlotContext:
    initial_relative_ric_km_s: tuple[float, float, float, float, float, float] | None = None
    training_config: RPOTrainingConfig | None = None
    mean_motion_rad_s: float | None = None
    coast_prediction_model: str = "hcw"
    cr3bp_projection_mode: str = "nonlinear"
    cr3bp_coast_prediction_horizon_s: float | None = None
    cr3bp_coast_prediction_dt_s: float | None = None
    reference_state_eci_km_s: tuple[float, float, float, float, float, float] | None = None
    initial_coast_ric_km_s: tuple[tuple[float, float, float, float, float, float], ...] = ()
    pilot_initial_snapshot: Any | None = None
    pilot_dashboard_kwargs: dict[str, Any] = field(default_factory=dict)
    camera_mode: str = "reference"
    target_centered_plot_planes: tuple[str, ...] = ()
    target_centered_plot_axes: dict[str, tuple[str, ...]] = field(default_factory=dict)
    plot_overlays_in_zoom: bool = True
    plot_overlays_in_zoom_by_plane: dict[str, bool] = field(default_factory=dict)
    plot_axis_scale: dict[str, tuple[float, float]] = field(default_factory=dict)
    plot_fixed_axis_half_span_km: dict[str, tuple[float | None, float | None]] = field(default_factory=dict)
    plot_equal_axis_scale_planes: tuple[str, ...] = ()
    proximity_ring_plot_planes: tuple[str, ...] = ("RI", "RC", "IC")
    _planned_trajectory_cache: dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _planned_trajectory_time_cache: dict[tuple[Any, ...], np.ndarray] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _preview_dashboard: Any | None = field(default=None, init=False, repr=False, compare=False)


@dataclass(frozen=True)
class _RectSpec:
    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height


@dataclass(frozen=True)
class OperatorDisplayState:
    previous_size: tuple[int, int]


@dataclass(frozen=True)
class OperatorTrajectoryProbe:
    state_ric_km_s: tuple[float, float, float, float, float, float]
    time_s: float
    plan_key: tuple[Any, ...]

__all__ = [name for name in globals() if not name.startswith("__")]
