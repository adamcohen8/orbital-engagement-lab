# ruff: noqa: F401,F403,F405,I001
from .runner_common import *

@dataclass(frozen=True)
class GameRunResult:
    config_path: Path
    difficulty: str
    level_passed: bool
    mode: str = "pilot"
    frame_convention: FrameConvention = FrameConvention()
    arcade_score: int = 0
    arcade_seed: int | None = None
    recording_path: Path | None = None
    debrief_path: Path | None = None


@dataclass
class GuidedTutorialRuntime:
    stage_index: int = 0
    active_stage_delta_v_m_s: float = 0.0
    stage_start_rel_ric: np.ndarray | None = None
    stage_start_mean_motion_rad_s: float | None = None
    awaiting_speed_step: bool = False
    wrong_key_active: bool = False


def _game_burn_trace_enabled() -> bool:
    value = str(os.environ.get(GAME_BURN_TRACE_ENV, "") or "").strip().lower()
    return value in {"1", "true", "yes", "on", "debug"}


def _trace_burn_loop(message: str) -> None:
    print(f"[burn-trace] {message}")


@dataclass
class RICPrimerRuntime:
    stage_index: int = 0
    elapsed_s: float = 0.0

    def reset(self) -> None:
        self.stage_index = 0
        self.elapsed_s = 0.0


@dataclass(frozen=True)
class OperatorTutorialStage:
    name: str
    display_label: str
    axis_index: int
    sign: int

    @property
    def plan(self) -> OperatorBurnPlan:
        delta_v = np.zeros(3, dtype=float)
        delta_v[int(self.axis_index)] = float(self.sign) * OPERATOR_TUTORIAL_BURN_DELTA_V_M_S
        return OperatorBurnPlan(
            burns=(
                OperatorBurn(
                    time_s=OPERATOR_TUTORIAL_BURN_TIME_S,
                    delta_v_ric_m_s=tuple(float(value) for value in delta_v),
                ),
            )
        )


@dataclass
class OperatorTutorialRuntime:
    stage_index: int = 0
    awaiting_script: bool = True
    stage_start_sim_s: float | None = None
    completed: bool = False

    def reset(self) -> None:
        self.stage_index = 0
        self.awaiting_script = True
        self.stage_start_sim_s = None
        self.completed = False


@dataclass
class OperatorBurnCinematicRuntime:
    active: bool = False
    hold_until_wall_s: float | None = None

    def reset(self) -> None:
        self.active = False
        self.hold_until_wall_s = None


@dataclass(frozen=True)
class SandboxSetupValues:
    target_a_km: float = 7000.0
    target_ecc: float = 0.0
    target_inc_deg: float = 45.0
    target_raan_deg: float = 0.0
    target_argp_deg: float = 0.0
    target_true_anomaly_deg: float = 0.0
    radial_km: float = 0.0
    in_track_km: float = -3.0
    cross_track_km: float = 0.0
    radial_rate_m_s: float = 0.0
    in_track_rate_m_s: float = 0.0
    cross_track_rate_m_s: float = 0.0

    @property
    def relative_ric_state_km_s(self) -> list[float]:
        return [
            float(self.radial_km),
            float(self.in_track_km),
            float(self.cross_track_km),
            float(self.radial_rate_m_s) / 1000.0,
            float(self.in_track_rate_m_s) / 1000.0,
            float(self.cross_track_rate_m_s) / 1000.0,
        ]


_SANDBOX_TARGET_COE_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("Semimajor Axis", "km", "target_a_km"),
    ("Eccentricity", "", "target_ecc"),
    ("Inclination", "deg", "target_inc_deg"),
    ("RAAN", "deg", "target_raan_deg"),
    ("Argument of Periapsis", "deg", "target_argp_deg"),
    ("True Anomaly", "deg", "target_true_anomaly_deg"),
)


_SANDBOX_CHASER_RIC_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("Radial R", "km", "radial_km"),
    ("In-Track I", "km", "in_track_km"),
    ("Cross-Track C", "km", "cross_track_km"),
    ("Radial Rate dR", "m/s", "radial_rate_m_s"),
    ("In-Track Rate dI", "m/s", "in_track_rate_m_s"),
    ("Cross-Track Rate dC", "m/s", "cross_track_rate_m_s"),
)


_SANDBOX_SETUP_FIELDS = _SANDBOX_TARGET_COE_FIELDS + _SANDBOX_CHASER_RIC_FIELDS

__all__ = [name for name in globals() if not name.startswith("__")]
