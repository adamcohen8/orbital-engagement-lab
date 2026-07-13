from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from sim.api import SimulationSnapshot
from sim.dynamics.orbit.cr3bp import (
    EARTH_MOON_MEAN_MOTION_RAD_S,
    cr3bp_l1_state_km_s,
    cr3bp_moon_state_km_s,
    propagate_cr3bp_reference_stm,
    propagate_cr3bp_state,
)
from sim.dynamics.orbit.elements import rv_to_coe_eci
from sim.estimation.relative_th_ekf import ya_closed_form_transition_matrix
from sim.game.fonts import game_font
from sim.game.formatting import format_distance_km, format_speed_km_s
from sim.game.frame_convention import FrameConvention, frame_convention_display_axis_sign, normalize_frame_convention
from sim.game.training import (
    ApproachGateConfig,
    ForbiddenRegionConfig,
    InspectionGateConfig,
    SunAngleConstraintConfig,
    relative_moon_ric_state_from_arrays,
    relative_ric_state_from_arrays,
)
from sim.utils.frames import (
    eci_relative_to_ric_rect,
    ric_dcm_ir_from_rv,
    ric_rect_state_to_eci,
)

EARTH_MU_KM3_S2 = 398600.4418
EARTH_RADIUS_KM = 6378.137
PLOT_OVERLAY_MARGIN = 1.18
MIN_PLOT_SPAN_KM = 0.005
MAX_TRAIL_DRAW_POINTS = 260
MAX_TARGET_ORBIT_DRAW_POINTS = 1200
MAX_GHOST_DRAW_POINTS = 120
MAX_ACTIVE_CR3BP_GHOST_DRAW_POINTS = 60
TEXT_CACHE_LIMIT = 512
BRIEFING_LINE_HEIGHT_PX = 24
MISSION_BANNER_LINE_HEIGHT_PX = 24
SATELLITE_SPRITE_DIAMETER_KM = 0.006
SATELLITE_ICON_SIZE_PX = 20
MOON_RADIUS_KM = 1737.4
ELLIPTIC_PREDICTION_COAST_UPDATE_INTERVAL_S = 30.0
ELLIPTIC_PREDICTION_BURN_UPDATE_INTERVAL_S = 0.0
ELLIPTIC_REFERENCE_CACHE_POSITION_TOL_KM = 1.0e-3
ELLIPTIC_REFERENCE_CACHE_VELOCITY_TOL_KM_S = 5.0e-6
CR3BP_PREDICTION_COAST_UPDATE_INTERVAL_S = 30.0
CR3BP_PREDICTION_BURN_UPDATE_INTERVAL_S = 0.0
CR3BP_REFERENCE_CACHE_POSITION_TOL_KM = 1.0e-3
CR3BP_REFERENCE_CACHE_VELOCITY_TOL_KM_S = 5.0e-6
CR3BP_RELATIVE_CACHE_POSITION_TOL_KM = 1.0e-3
CR3BP_RELATIVE_CACHE_VELOCITY_TOL_KM_S = 5.0e-6
CR3BP_TARGET_ORBIT_INTERNAL_STEP_S = 120.0
CR3BP_TARGET_ORBIT_MAX_POINTS = 2400
PREDICTION_DENSE_POINT_FRACTION = 2.0 / 3.0
GAME_ASSET_DIR = Path(__file__).resolve().parent / "assets"
TARGET_SPRITE_PATH = GAME_ASSET_DIR / "rpo_target_sprite.png"
CHASER_SPRITE_PATH = GAME_ASSET_DIR / "rpo_chaser_sprite.png"
TARGET_MARKER_COLOR = (245, 92, 92)
CHASER_MARKER_COLOR = (245, 205, 92)
TARGET_TRAIL_COLOR = (215, 86, 86)
CHASER_TRAIL_COLOR = (245, 205, 92)
VELOCITY_VECTOR_COLOR = (106, 155, 210)
COAST_PREDICTION_COLOR = (135, 150, 172)
LIVE_BURN_COLOR = (92, 220, 160)
WEB_VECTOR_VREL_SCALE_PX_PER_KM_S = 75000.0
WEB_VECTOR_THRUST_SCALE_PX = 42.0
WEB_VECTOR_ARROW_HEAD_PX = 8.0
WEB_VECTOR_ARROW_HEAD_ANGLE_RAD = 0.45
VISUAL_EXTRAPOLATION_MAX_SIM_S = 1.0
RIC_PRIMER_STAGES: tuple[dict[str, Any], ...] = (
    {
        "id": "radial",
        "axis_index": 0,
        "title": "Radial Axis",
        "text": "Away from Earth through the target.",
        "hint": "Higher or lower circular orbits map to up/down motion on R.",
        "local_subtitle": "Radial offset in RI",
        "eci_subtitle": "Orbit radius changes",
        "eci_plane": "RC",
        "amplitude_km": 0.65,
    },
    {
        "id": "in_track",
        "axis_index": 1,
        "title": "In-Track Axis",
        "text": "Forward and backward along the target orbit.",
        "hint": "Ahead or behind the target maps to left/right motion on I.",
        "local_subtitle": "Phase offset in RI",
        "eci_subtitle": "Same orbit, phase changes",
        "eci_plane": "RC",
        "amplitude_km": 0.65,
    },
    {
        "id": "cross_track",
        "axis_index": 2,
        "title": "Cross-Track Axis",
        "text": "Out of the target orbital plane.",
        "hint": "Inclination offset maps to left/right motion on C.",
        "local_subtitle": "Plane offset in RC",
        "eci_subtitle": "Inclination side view",
        "eci_plane": "RI",
        "amplitude_km": 0.65,
    },
)


def _ric_primer_stage(stage_index: int) -> dict[str, Any]:
    idx = int(np.clip(int(stage_index), 0, len(RIC_PRIMER_STAGES) - 1))
    return dict(RIC_PRIMER_STAGES[idx])


def _point_on_circle(center: tuple[int, int], radius: float, theta_rad: float) -> tuple[int, int]:
    return (
        int(round(float(center[0]) + float(radius) * float(np.cos(theta_rad)))),
        int(round(float(center[1]) + float(radius) * float(np.sin(theta_rad)))),
    )


def _line_segment(
    center: tuple[int, int],
    half_span_px: float,
    slope: float,
) -> tuple[tuple[int, int], tuple[int, int]]:
    span = float(max(half_span_px, 1.0))
    dx = span / float(np.sqrt(1.0 + float(slope) * float(slope)))
    dy = float(slope) * dx
    return (
        (int(round(float(center[0]) - dx)), int(round(float(center[1]) - dy))),
        (int(round(float(center[0]) + dx)), int(round(float(center[1]) + dy))),
    )


def _point_along_line(
    line: tuple[tuple[int, int], tuple[int, int]],
    fraction: float,
) -> tuple[int, int]:
    frac = float(np.clip(fraction, 0.0, 1.0))
    return (
        int(round(float(line[0][0]) + (float(line[1][0]) - float(line[0][0])) * frac)),
        int(round(float(line[0][1]) + (float(line[1][1]) - float(line[0][1])) * frac)),
    )


def _front_loaded_prediction_times(
    horizon_s: float,
    dt_s: float,
    *,
    max_points: int,
) -> np.ndarray:
    horizon = float(max(horizon_s, 0.0))
    dt = float(max(dt_s, 1.0e-6))
    limit = max(int(max_points), 2)
    dense_count = int(np.floor(horizon / dt)) + 1
    if dense_count <= limit:
        return np.linspace(0.0, horizon, max(dense_count, 2), dtype=float)

    near_count = int(np.clip(round(float(limit) * PREDICTION_DENSE_POINT_FRACTION), 2, limit - 1))
    near_horizon = min(horizon, dt * float(near_count - 1))
    near = np.linspace(0.0, near_horizon, near_count, dtype=float)
    far_count = limit - near_count
    if far_count <= 0 or horizon <= near_horizon:
        return near
    far = np.linspace(near_horizon, horizon, far_count + 1, dtype=float)[1:]
    return np.concatenate((near, far))


def _game_asset_path_or_default(value: Path | str | None, default: Path) -> Path:
    if value is None:
        return default
    raw = str(value or "").strip()
    if not raw:
        return default
    path = Path(raw)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return GAME_ASSET_DIR / path


@dataclass
class _HistoryRingBuffer:
    width: int
    max_rows: int
    data: np.ndarray = field(init=False, repr=False)
    start: int = 0
    count: int = 0

    def __post_init__(self) -> None:
        self.width = int(max(self.width, 1))
        self.max_rows = int(max(self.max_rows, 1))
        self.data = np.zeros((self.max_rows, self.width), dtype=float)

    @classmethod
    def from_rows(cls, rows: Any, *, width: int, max_rows: int) -> _HistoryRingBuffer:
        ring = cls(width=int(width), max_rows=int(max(max_rows, 1)))
        arr = np.asarray(rows, dtype=float).reshape(-1, int(width))
        if arr.size:
            tail = arr[-ring.max_rows :]
            ring.data[: tail.shape[0], :] = tail
            ring.count = int(tail.shape[0])
        return ring

    def append(self, row: Any) -> None:
        row_arr = np.asarray(row, dtype=float).reshape(self.width)
        if self.count < self.max_rows:
            idx = (self.start + self.count) % self.max_rows
            self.count += 1
        else:
            idx = self.start
            self.start = (self.start + 1) % self.max_rows
        self.data[idx, :] = row_arr

    def rows(self) -> np.ndarray:
        if self.count <= 0:
            return np.zeros((0, self.width), dtype=float)
        end = self.start + self.count
        if end <= self.max_rows:
            return self.data[self.start : end, :]
        return np.concatenate((self.data[self.start :, :], self.data[: end % self.max_rows, :]), axis=0)


def _new_history_ring(width: int, max_rows: int) -> _HistoryRingBuffer:
    return _HistoryRingBuffer(width=int(width), max_rows=int(max(max_rows, 1)))


@dataclass
class PygameRPODashboard:
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

    def __post_init__(self) -> None:
        try:
            import pygame
        except ImportError as exc:  # pragma: no cover - exercised only without optional dependency.
            raise RuntimeError("Pygame game backend requires `pygame`. Install with `pip install .[game]`.") from exc
        self.pygame = pygame
        self.frame_convention = normalize_frame_convention(self.frame_convention)
        pygame.init()
        pygame.font.init()
        flags = pygame.FULLSCREEN | pygame.SCALED if self.fullscreen else pygame.RESIZABLE
        self.screen = pygame.display.set_mode((1280, 720), flags)
        pygame.display.set_caption(self.title)
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)
        self.clock = pygame.time.Clock()
        self.font = game_font(pygame, 19)
        self.small_font = game_font(pygame, 15)
        self.large_font = game_font(pygame, 28)
        self.closed = False
        self.t_s: list[float] = []
        self.sample_wall_s: list[float] = []
        self.rel_hist: list[np.ndarray] = []
        self.target_rel_hist: list[np.ndarray] = []
        self.target_reference_rel_hist: list[np.ndarray] = []
        self.target_eci_hist: list[np.ndarray] = []
        self.chaser_eci_hist: list[np.ndarray] = []
        self.thrust_hist: list[np.ndarray] = []
        self.thrust_ric_hist: list[np.ndarray] = []
        max_rows = int(max(self.max_history, 2))
        self._rel_array = _new_history_ring(6, max_rows)
        self._target_rel_array = _new_history_ring(6, max_rows)
        self._target_reference_rel_array = _new_history_ring(6, max_rows)
        self._target_eci_array = _new_history_ring(6, max_rows)
        self._chaser_eci_array = _new_history_ring(6, max_rows)
        self._thrust_ric_array = _new_history_ring(3, max_rows)
        self.mean_motion_rad_s: float | None = None
        self.reference_state_eci: np.ndarray | None = None
        self.target_orbit_reference_state_eci: np.ndarray | None = None
        self.target_true_anomaly_deg: float | None = None
        self.briefing_scroll_px = 0
        self.mission_banner_scroll_px = 0
        self._frame_cache: dict[str, np.ndarray] = {}
        self._raw_frame_cache: dict[str, Any] = {}
        self._frame_cache_dirty = True
        self._render_motion_enabled = False
        self._render_wall_time_s = perf_counter()
        self._render_speed_multiple = 1.0
        self._prediction_cache: dict[str, dict[str, Any]] = {}
        self._briefing_layout_cache: dict[str, Any] = {}
        self._mission_banner_layout_cache: dict[str, Any] = {}
        self._text_cache: dict[tuple[int, str, tuple[int, int, int]], Any] = {}
        self._operator_projection_transition: dict[str, Any] | None = None
        target_sprite_path = _game_asset_path_or_default(self.target_sprite_path, TARGET_SPRITE_PATH)
        chaser_sprite_path = _game_asset_path_or_default(self.chaser_sprite_path, CHASER_SPRITE_PATH)
        self._target_sprite = self._load_marker_sprite(target_sprite_path)
        self._chaser_sprite = self._load_marker_sprite(chaser_sprite_path)
        self._sprite_scale_cache: dict[tuple[str, int], Any] = {}

    def close(self) -> None:
        self.closed = True
        try:
            self.pygame.event.set_grab(False)
            self.pygame.mouse.set_visible(True)
            self.pygame.display.quit()
            self.pygame.quit()
        except Exception:
            pass

    def clear(self) -> None:
        self.t_s.clear()
        if hasattr(self, "sample_wall_s"):
            self.sample_wall_s.clear()
        else:
            self.sample_wall_s = []
        self.rel_hist.clear()
        self.target_rel_hist.clear()
        if hasattr(self, "target_reference_rel_hist"):
            self.target_reference_rel_hist.clear()
        else:
            self.target_reference_rel_hist = []
        if hasattr(self, "target_eci_hist"):
            self.target_eci_hist.clear()
        else:
            self.target_eci_hist = []
        if hasattr(self, "chaser_eci_hist"):
            self.chaser_eci_hist.clear()
        else:
            self.chaser_eci_hist = []
        self.thrust_hist.clear()
        self.thrust_ric_hist.clear()
        max_rows = int(max(self.max_history, 2))
        self._rel_array = _new_history_ring(6, max_rows)
        self._target_rel_array = _new_history_ring(6, max_rows)
        self._target_reference_rel_array = _new_history_ring(6, max_rows)
        self._target_eci_array = _new_history_ring(6, max_rows)
        self._chaser_eci_array = _new_history_ring(6, max_rows)
        self._thrust_ric_array = _new_history_ring(3, max_rows)
        self.mean_motion_rad_s = None
        self.reference_state_eci = None
        self.target_orbit_reference_state_eci = None
        self.target_true_anomaly_deg = None
        self._frame_cache = {}
        self._raw_frame_cache = {}
        self._frame_cache_dirty = True
        self._render_motion_enabled = False
        self._prediction_cache = {}
        self.briefing_scroll_px = 0
        self.mission_banner_scroll_px = 0
        self._mission_banner_layout_cache = {}
        self.tutorial_target_path_ric = np.empty((0, 6), dtype=float)
        self.live_prediction_accel_ric_km_s2 = np.zeros(3, dtype=float)
        self.live_prediction_elapsed_s = 0.0
        self._operator_projection_transition = None

    def set_live_prediction_burn(self, accel_ric_km_s2: np.ndarray, elapsed_s: float) -> None:
        accel = np.array(accel_ric_km_s2, dtype=float).reshape(3)
        elapsed = float(max(elapsed_s, 0.0))
        if not np.all(np.isfinite(accel)):
            accel = np.zeros(3, dtype=float)
        if not np.isfinite(elapsed):
            elapsed = 0.0
        previous_accel = np.array(getattr(self, "live_prediction_accel_ric_km_s2", np.zeros(3)), dtype=float).reshape(3)
        previous_elapsed = float(getattr(self, "live_prediction_elapsed_s", 0.0))
        if not np.allclose(previous_accel, accel, rtol=0.0, atol=1.0e-14) or not np.isclose(
            previous_elapsed,
            elapsed,
            rtol=0.0,
            atol=1.0e-9,
        ):
            self._frame_cache_dirty = True
        self.live_prediction_accel_ric_km_s2 = accel
        self.live_prediction_elapsed_s = elapsed

    def set_operator_projection_transition(
        self,
        pre_burn_rel_ric: np.ndarray,
        post_burn_rel_ric: np.ndarray,
        *,
        duration_s: float | None = None,
    ) -> None:
        pre = np.array(pre_burn_rel_ric, dtype=float).reshape(6)
        post = np.array(post_burn_rel_ric, dtype=float).reshape(6)
        if not np.all(np.isfinite(pre)) or not np.all(np.isfinite(post)):
            return
        duration = (
            float(self.operator_projection_transition_duration_s)
            if duration_s is None
            else float(duration_s)
        )
        self._operator_projection_transition = {
            "pre": pre,
            "post": post,
            "started_wall_s": perf_counter(),
            "duration_s": max(duration, 0.1),
        }
        self._prediction_cache = {}
        self._frame_cache_dirty = True

    def _load_marker_sprite(self, path: Path) -> Any | None:
        try:
            return self.pygame.image.load(str(path)).convert_alpha()
        except Exception:
            return None

    def _draw_satellite_marker(
        self,
        center: tuple[int, int],
        *,
        role: str,
        scale_x: float,
        scale_y: float,
        fallback_radius_px: int,
        force_icon: bool = False,
    ) -> None:
        pygame = self.pygame
        role_key = str(role).strip().lower()
        if role_key == "target":
            sprite = getattr(self, "_target_sprite", None)
            color = TARGET_MARKER_COLOR
            cache_key = "target"
            diameter_km = float(getattr(self, "target_sprite_diameter_km", SATELLITE_SPRITE_DIAMETER_KM))
        else:
            sprite = getattr(self, "_chaser_sprite", None)
            color = CHASER_MARKER_COLOR
            cache_key = "chaser"
            diameter_km = float(getattr(self, "chaser_sprite_diameter_km", SATELLITE_SPRITE_DIAMETER_KM))

        sprite_size = (
            SATELLITE_ICON_SIZE_PX
            if force_icon
            else _satellite_marker_size_px(scale_x, scale_y, diameter_km=diameter_km)
        )
        if sprite is None or sprite_size <= 0:
            pygame.draw.circle(self.screen, color, center, int(fallback_radius_px))
            return

        cache = getattr(self, "_sprite_scale_cache", {})
        self._sprite_scale_cache = cache
        cache_id = (cache_key, int(sprite_size))
        scaled = cache.get(cache_id)
        if scaled is None:
            scaled = pygame.transform.smoothscale(sprite, (int(sprite_size), int(sprite_size)))
            cache[cache_id] = scaled
        rect = scaled.get_rect(center=center)
        self.screen.blit(scaled, rect)
        dot_radius, ring_radius = _satellite_marker_reticle_radii_px(int(sprite_size))
        if dot_radius > 0 and ring_radius > 0:
            pygame.draw.circle(self.screen, (235, 248, 255), center, dot_radius)
            pygame.draw.circle(self.screen, color, center, ring_radius, width=1)

    def reset_briefing_scroll(self) -> None:
        self.briefing_scroll_px = 0

    def scroll_briefing(self, delta_px: int) -> None:
        self.briefing_scroll_px = max(0, int(self.briefing_scroll_px) + int(delta_px))

    def reset_mission_banner_scroll(self) -> None:
        self.mission_banner_scroll_px = 0

    def scroll_mission_banner(self, delta_px: int) -> None:
        self.mission_banner_scroll_px = max(0, int(self.mission_banner_scroll_px) + int(delta_px))

    def push_snapshot(self, snapshot: SimulationSnapshot) -> None:
        target = snapshot.truth.get(self.target_object_id)
        chaser = snapshot.truth.get(self.chaser_object_id)
        reference_id = str(self.reference_object_id or self.target_object_id)
        reference = snapshot.truth.get(reference_id)
        if reference is None:
            reference = target
        target_reference_id = str(getattr(self, "target_reference_object_id", None) or reference_id)
        target_reference = snapshot.truth.get(target_reference_id)
        if target_reference is None:
            target_reference = reference
        if target is None or chaser is None or reference is None or target_reference is None:
            return
        target_state_eci = np.array(target, dtype=float).reshape(-1)[:6]
        chaser_state_eci = np.array(chaser, dtype=float).reshape(-1)[:6]
        target_reference_state_eci = np.array(target_reference, dtype=float).reshape(-1)[:6]
        frame_key = _relative_frame_key(getattr(self, "relative_frame", "ric"))
        if frame_key == "cislunar_l1":
            origin = cr3bp_l1_state_km_s()
            rel = chaser_state_eci - origin
            target_rel = target_state_eci - origin
            target_reference_rel = target_reference_state_eci - origin
        elif frame_key == "moon_ric":
            rel = relative_moon_ric_state_from_arrays(target, chaser)
            target_rel = np.zeros(6, dtype=float)
            target_reference_rel = relative_moon_ric_state_from_arrays(reference, target_reference)
        else:
            rel = relative_ric_state_from_arrays(reference, chaser)
            target_rel = relative_ric_state_from_arrays(reference, target)
            target_reference_rel = relative_ric_state_from_arrays(reference, target_reference)
        self.target_true_anomaly_deg = (
            _true_anomaly_deg_from_state(target) if self._uses_elliptic_prediction_model() else None
        )
        reference_arr = np.array(reference, dtype=float).reshape(-1)
        if frame_key == "cislunar_l1":
            self.reference_state_eci = cr3bp_l1_state_km_s()
            self.mean_motion_rad_s = EARTH_MOON_MEAN_MOTION_RAD_S
        elif frame_key == "moon_ric" and reference_arr.size >= 6:
            self.reference_state_eci = reference_arr[:6].astype(float)
            if getattr(self, "target_orbit_reference_state_eci", None) is None:
                self.target_orbit_reference_state_eci = self.reference_state_eci.copy()
            target_moon = self.reference_state_eci - cr3bp_moon_state_km_s()
            r_norm = float(np.linalg.norm(target_moon[:3]))
            h_norm = float(np.linalg.norm(np.cross(target_moon[:3], target_moon[3:6])))
            if r_norm > 0.0 and np.isfinite(r_norm):
                self.mean_motion_rad_s = h_norm / (r_norm**2)
        elif reference_arr.size >= 6:
            self.reference_state_eci = reference_arr[:6].astype(float)
            r_norm = float(np.linalg.norm(reference_arr[:3]))
            if r_norm > 0.0 and np.isfinite(r_norm):
                self.mean_motion_rad_s = float(np.sqrt(EARTH_MU_KM3_S2 / (r_norm**3)))
        self.t_s.append(float(snapshot.time_s))
        if not hasattr(self, "sample_wall_s"):
            self.sample_wall_s = []
        self.sample_wall_s.append(perf_counter())
        self.rel_hist.append(rel)
        if not hasattr(self, "target_eci_hist"):
            self.target_eci_hist = []
        if not hasattr(self, "chaser_eci_hist"):
            self.chaser_eci_hist = []
        if not hasattr(self, "_target_eci_array"):
            self._target_eci_array = _new_history_ring(6, int(max(self.max_history, 2)))
        if not hasattr(self, "_chaser_eci_array"):
            self._chaser_eci_array = _new_history_ring(6, int(max(self.max_history, 2)))
        self.target_eci_hist.append(target_state_eci.astype(float))
        self.chaser_eci_hist.append(chaser_state_eci.astype(float))
        thrust = snapshot.applied_thrust.get(self.chaser_object_id, np.zeros(3, dtype=float))
        thrust_eci = np.array(thrust, dtype=float).reshape(3)
        self.thrust_hist.append(thrust_eci)
        self.target_rel_hist.append(target_rel)
        if not hasattr(self, "target_reference_rel_hist"):
            self.target_reference_rel_hist = []
        self.target_reference_rel_hist.append(target_reference_rel)
        if frame_key == "cislunar_l1":
            thrust_ric = thrust_eci
        elif frame_key == "moon_ric" and reference_arr.size >= 6:
            target_moon = reference_arr[:6].astype(float) - cr3bp_moon_state_km_s()
            c_ir = ric_dcm_ir_from_rv(target_moon[:3], target_moon[3:6])
            thrust_ric = c_ir.T @ thrust_eci
        elif reference_arr.size >= 6:
            c_ir = ric_dcm_ir_from_rv(reference_arr[:3], reference_arr[3:6])
            thrust_ric = c_ir.T @ thrust_eci
        else:
            thrust_ric = thrust_eci
        self.thrust_ric_hist.append(thrust_ric)
        while len(self.t_s) > int(max(self.max_history, 2)):
            self.t_s.pop(0)
            if self.sample_wall_s:
                self.sample_wall_s.pop(0)
            self.rel_hist.pop(0)
            if self.target_eci_hist:
                self.target_eci_hist.pop(0)
            if self.chaser_eci_hist:
                self.chaser_eci_hist.pop(0)
            if self.target_rel_hist:
                self.target_rel_hist.pop(0)
            if self.target_reference_rel_hist:
                self.target_reference_rel_hist.pop(0)
            if self.thrust_hist:
                self.thrust_hist.pop(0)
            if self.thrust_ric_hist:
                self.thrust_ric_hist.pop(0)
        self._rel_array = _history_array_tail(self._rel_array, rel, width=6, max_rows=int(max(self.max_history, 2)))
        self._target_rel_array = _history_array_tail(
            self._target_rel_array,
            target_rel,
            width=6,
            max_rows=int(max(self.max_history, 2)),
        )
        self._target_reference_rel_array = _history_array_tail(
            getattr(self, "_target_reference_rel_array", _new_history_ring(6, int(max(self.max_history, 2)))),
            target_reference_rel,
            width=6,
            max_rows=int(max(self.max_history, 2)),
        )
        self._target_eci_array = _history_array_tail(
            self._target_eci_array,
            target_state_eci,
            width=6,
            max_rows=int(max(self.max_history, 2)),
        )
        self._chaser_eci_array = _history_array_tail(
            self._chaser_eci_array,
            chaser_state_eci,
            width=6,
            max_rows=int(max(self.max_history, 2)),
        )
        self._thrust_ric_array = _history_array_tail(
            self._thrust_ric_array,
            thrust_ric,
            width=3,
            max_rows=int(max(self.max_history, 2)),
        )
        if frame_key == "moon_ric" and self._uses_cr3bp_prediction_model():
            self._prepare_cr3bp_target_orbit_prediction()
        self._frame_cache_dirty = True

    def draw(
        self,
        *,
        command_status: str = "",
        coach_hint: str = "",
        mission_state: str = "active",
        level_title: str = "",
        mission_metrics: tuple[str, ...] = (),
        objective_checklist: tuple[str, ...] = (),
        speed_multiple: float = 1.0,
        selected_speed_multiple: float | None = None,
        recording_status: str = "",
        briefing_lines: tuple[str, ...] = (),
        debrief_lines: tuple[str, ...] = (),
        debrief_available: bool = True,
        render_motion: bool = False,
        pause_overlay: bool = False,
    ) -> None:
        pygame = self.pygame
        if self.closed:
            return
        width, height = self.screen.get_size()
        self._render_motion_enabled = bool(render_motion)
        self._render_wall_time_s = perf_counter()
        self._render_speed_multiple = float(max(speed_multiple, 0.0))
        if self._operator_projection_transition_active():
            self._frame_cache_dirty = True
        self._prepare_frame_cache()
        self.screen.fill((12, 16, 22))
        top = pygame.Rect(36, 18, width - 72, 84)
        left = pygame.Rect(36, 124, max((width - 108) // 2, 200), max(height - 256, 250))
        right = pygame.Rect(left.right + 36, 78, left.width, left.height)
        right.y = left.y
        hud = pygame.Rect(36, height - 112, width - 72, 86)
        self._draw_top_bar(
            top,
            mission_state=mission_state,
            level_title=level_title,
            mission_metrics=mission_metrics,
            objective_checklist=objective_checklist,
        )
        left_panel, right_panel = self._plot_panel_specs()
        self._draw_panel(left, left_panel[0], x_axis=left_panel[1], y_axis=left_panel[2])
        self._draw_panel(right, right_panel[0], x_axis=right_panel[1], y_axis=right_panel[2])
        self._draw_hud(
            hud,
            command_status=command_status,
            coach_hint=coach_hint,
            speed_multiple=speed_multiple,
            selected_speed_multiple=selected_speed_multiple,
            recording_status=recording_status,
        )
        if briefing_lines:
            self._draw_briefing(briefing_lines)
        elif bool(pause_overlay) and mission_state == "active":
            self._draw_pause_overlay()
        if mission_state in {"passed", "failed"}:
            self._draw_mission_banner(
                mission_state,
                debrief_lines=debrief_lines,
                debrief_available=debrief_available,
            )
        pygame.display.flip()

    def draw_ric_primer(self, *, stage_index: int, elapsed_s: float, recording_status: str = "") -> None:
        pygame = self.pygame
        if self.closed:
            return
        width, height = self.screen.get_size()
        stage = _ric_primer_stage(stage_index)
        self.screen.fill((12, 16, 22))
        top = pygame.Rect(36, 18, width - 72, 104)
        panel_y = top.bottom + 24
        left = pygame.Rect(36, panel_y, max((width - 108) // 2, 200), max(height - panel_y - 130, 250))
        right = pygame.Rect(left.right + 36, left.y, left.width, left.height)
        hud = pygame.Rect(36, height - 112, width - 72, 86)
        self._draw_ric_primer_top_bar(
            top,
            stage=stage,
            stage_index=stage_index,
        )
        if stage["eci_plane"] == "RI":
            self._draw_ric_primer_eci_panel(left, stage=stage, elapsed_s=elapsed_s)
            self._draw_ric_primer_local_panel(right, stage=stage, elapsed_s=elapsed_s, plane="RC")
        else:
            self._draw_ric_primer_local_panel(left, stage=stage, elapsed_s=elapsed_s, plane="RI")
            self._draw_ric_primer_eci_panel(right, stage=stage, elapsed_s=elapsed_s)
        self._draw_ric_primer_hud(
            hud,
            stage=stage,
            stage_index=stage_index,
            recording_status=recording_status,
        )
        pygame.display.flip()

    def tick(self, fps: float = 60.0) -> None:
        self.clock.tick(float(max(fps, 1.0)))

    def _draw_ric_primer_top_bar(self, rect: Any, *, stage: dict[str, Any], stage_index: int) -> None:
        pygame = self.pygame
        pygame.draw.rect(self.screen, (18, 24, 32), rect, border_radius=10)
        pygame.draw.rect(self.screen, (82, 96, 118), rect, width=2, border_radius=10)
        right_width = min(420, max(280, rect.width // 3))
        left_width = max(rect.width - right_width - 34, 160)
        title = self._fit_text_px("RIC FRAME PRIMER", self.large_font, left_width)
        self._text(title, (rect.x + 16, rect.y + 9), self.large_font, (230, 235, 242))
        meta = f"INFO Step {stage_index + 1}/3    INFO {stage['title']}"
        self._text(
            self._fit_text_px(meta, self.small_font, left_width),
            (rect.x + 16, rect.y + 54),
            self.small_font,
            (222, 230, 238),
        )
        x = rect.right - right_width - 16
        self._text("OBJECTIVES", (x, rect.y + 12), self.small_font, (170, 184, 204))
        self._text(
            self._fit_text_px(str(stage["title"]), self.small_font, right_width),
            (x, rect.y + 35),
            self.small_font,
            (150, 235, 170),
        )
        self._text(
            self._fit_text_px(str(stage["text"]), self.small_font, right_width),
            (x, rect.y + 58),
            self.small_font,
            (150, 205, 245),
        )

    def _draw_ric_primer_local_panel(self, rect: Any, *, stage: dict[str, Any], elapsed_s: float, plane: str) -> None:
        pygame = self.pygame
        title = "RI Plane" if plane == "RI" else "RC Plane"
        pygame.draw.rect(self.screen, (20, 27, 36), rect, border_radius=10)
        pygame.draw.rect(self.screen, (80, 92, 110), rect, width=1, border_radius=10)
        subtitle_width = min(240, max(rect.width // 2 - 20, 80))
        subtitle = self._fit_text_px(str(stage["local_subtitle"]), self.small_font, subtitle_width)
        title_width = max(rect.width - subtitle_width - 48, 80)
        self._text(
            self._fit_text_px(title, self.font, title_width),
            (rect.x + 14, rect.y + 10),
            self.font,
            (230, 235, 242),
        )
        self._text(subtitle, (rect.right - subtitle_width - 14, rect.y + 13), self.small_font, (170, 184, 204))
        plot = rect.inflate(-48, -72)
        plot.y += 28
        pygame.draw.rect(self.screen, (8, 11, 16), plot)
        pygame.draw.rect(self.screen, (72, 84, 102), plot, width=1)
        scale = min(plot.width, plot.height) / 2.5
        self._draw_grid(plot, scale=scale)
        x_axis = 1 if plane == "RI" else 2
        y_axis = 0
        x_display_sign = self._axis_display_sign(x_axis)
        y_display_sign = self._axis_display_sign(y_axis)
        self._draw_ric_primer_axes(
            plot,
            x_axis=x_axis,
            y_axis=y_axis,
            active_axis=int(stage["axis_index"]),
            x_display_sign=x_display_sign,
            y_display_sign=y_display_sign,
        )
        point = np.zeros(3, dtype=float)
        point[int(stage["axis_index"])] = float(stage["amplitude_km"]) * float(np.sin(float(elapsed_s) * 1.05))

        def to_px(pos: np.ndarray) -> tuple[int, int]:
            x = float(pos[x_axis]) * x_display_sign
            y = float(pos[y_axis]) * y_display_sign
            return plot.centerx + int(round(x * scale)), plot.centery - int(round(y * scale))

        target = to_px(np.zeros(3, dtype=float))
        chaser = to_px(point)
        self._draw_satellite_marker(target, role="target", scale_x=scale, scale_y=scale, fallback_radius_px=6, force_icon=True)
        self._draw_satellite_marker(
            chaser,
            role="chaser",
            scale_x=scale,
            scale_y=scale,
            fallback_radius_px=7,
            force_icon=True,
        )
        pygame.draw.ellipse(
            self.screen,
            (92, 42, 50),
            pygame.Rect(
                target[0] - int(0.25 * scale),
                target[1] - int(0.25 * scale),
                int(0.5 * scale),
                int(0.5 * scale),
            ),
            width=1,
        )
        self._text("Target", (target[0] + 10, target[1] - 10), self.small_font, TARGET_MARKER_COLOR)
        self._text("Chaser", (chaser[0] + 10, chaser[1] + 14), self.small_font, CHASER_MARKER_COLOR)

    def _draw_ric_primer_axes(
        self,
        plot: Any,
        *,
        x_axis: int,
        y_axis: int,
        active_axis: int,
        x_display_sign: float = 1.0,
        y_display_sign: float = 1.0,
    ) -> None:
        pygame = self.pygame
        colors = {
            0: (150, 235, 170),
            1: (245, 205, 92),
            2: (96, 190, 245),
        }
        x_color = colors.get(x_axis, (90, 104, 124)) if active_axis == x_axis else (90, 104, 124)
        y_color = colors.get(y_axis, (90, 104, 124)) if active_axis == y_axis else (90, 104, 124)
        pygame.draw.line(self.screen, x_color, (plot.left + 36, plot.centery), (plot.right - 36, plot.centery), width=2)
        pygame.draw.line(self.screen, y_color, (plot.centerx, plot.bottom - 32), (plot.centerx, plot.top + 32), width=2)
        x_positive_at_right = float(x_display_sign) >= 0.0
        y_positive_at_top = float(y_display_sign) >= 0.0
        x_arrow_origin = (plot.right - 54, plot.centery) if x_positive_at_right else (plot.left + 54, plot.centery)
        y_arrow_origin = (plot.centerx, plot.top + 56) if y_positive_at_top else (plot.centerx, plot.bottom - 56)
        self._draw_vector(x_arrow_origin, np.array([24.0 * float(x_display_sign), 0.0]), color=x_color, scale=1.0)
        self._draw_vector(y_arrow_origin, np.array([0.0, 24.0 * float(y_display_sign)]), color=y_color, scale=1.0)
        x_plus = self._signed_axis_label_for_plot(x_axis, 1)
        x_minus = self._signed_axis_label_for_plot(x_axis, -1)
        y_plus = self._signed_axis_label_for_plot(y_axis, 1)
        y_minus = self._signed_axis_label_for_plot(y_axis, -1)
        x_left_label = x_minus if x_positive_at_right else x_plus
        x_right_label = x_plus if x_positive_at_right else x_minus
        y_top_label = y_plus if y_positive_at_top else y_minus
        y_bottom_label = y_minus if y_positive_at_top else y_plus
        self._text(x_left_label, (plot.left + 42, plot.centery - 18), self.small_font, x_color)
        self._text(x_right_label, (plot.right - 74, plot.centery - 18), self.small_font, x_color)
        self._text(y_top_label, (plot.centerx + 10, plot.top + 34), self.small_font, y_color)
        self._text(y_bottom_label, (plot.centerx + 10, plot.bottom - 58), self.small_font, y_color)

    def _draw_ric_primer_eci_panel(self, rect: Any, *, stage: dict[str, Any], elapsed_s: float) -> None:
        pygame = self.pygame
        pygame.draw.rect(self.screen, (20, 27, 36), rect, border_radius=10)
        pygame.draw.rect(self.screen, (80, 92, 110), rect, width=1, border_radius=10)
        subtitle_width = min(250, max(rect.width // 2 - 20, 80))
        subtitle = self._fit_text_px(str(stage["eci_subtitle"]), self.small_font, subtitle_width)
        title_width = max(rect.width - subtitle_width - 48, 80)
        self._text(
            self._fit_text_px("ECI Orbit", self.font, title_width),
            (rect.x + 14, rect.y + 10),
            self.font,
            (230, 235, 242),
        )
        self._text(subtitle, (rect.right - subtitle_width - 14, rect.y + 13), self.small_font, (170, 184, 204))
        plot = rect.inflate(-48, -72)
        plot.y += 28
        pygame.draw.rect(self.screen, (8, 11, 16), plot)
        pygame.draw.rect(self.screen, (72, 84, 102), plot, width=1)
        if stage["id"] == "cross_track":
            self._draw_ric_primer_cross_track_side_view(plot, elapsed_s=elapsed_s)
        else:
            self._draw_ric_primer_eci_circles(plot, stage=stage, elapsed_s=elapsed_s)

    def _draw_ric_primer_eci_circles(self, plot: Any, *, stage: dict[str, Any], elapsed_s: float) -> None:
        pygame = self.pygame
        center = plot.center
        radius = min(plot.width, plot.height) * 0.34
        phase = float(np.sin(float(elapsed_s) * 1.05))
        target_theta = -0.35
        radius_offset = 0.14 * phase if stage["id"] == "radial" else 0.0
        phase_offset = 0.34 * phase if stage["id"] == "in_track" else 0.0
        pygame.draw.circle(self.screen, (42, 92, 130), center, int(round(radius)), width=2)
        chaser_radius = radius * (1.0 + radius_offset)
        pygame.draw.circle(
            self.screen,
            (170, 68, 74) if stage["id"] == "radial" else (110, 60, 66),
            center,
            int(round(chaser_radius)),
            width=2,
        )
        self._draw_primer_earth(center, int(round(radius * 0.16)))
        target = _point_on_circle(center, radius, target_theta)
        chaser = _point_on_circle(center, chaser_radius, target_theta + phase_offset)
        self._draw_satellite_marker(target, role="target", scale_x=radius, scale_y=radius, fallback_radius_px=7, force_icon=True)
        self._draw_satellite_marker(
            chaser,
            role="chaser",
            scale_x=radius,
            scale_y=radius,
            fallback_radius_px=7,
            force_icon=True,
        )
        self._text("Target", (target[0] - 62, target[1] + 18), self.small_font, (230, 235, 242))
        self._text("Chaser", (chaser[0] + 10, chaser[1] - 16), self.small_font, (230, 235, 242))
        if stage["id"] == "radial":
            self._text(
                f"Chaser radius {1.0 + radius_offset:.2f}x target orbit",
                (plot.x + 18, plot.bottom - 26),
                self.small_font,
                (170, 184, 204),
            )

    def _draw_ric_primer_cross_track_side_view(self, plot: Any, *, elapsed_s: float) -> None:
        pygame = self.pygame
        center = plot.center
        half_span = min(plot.width, plot.height) * 0.36
        inclination_deg = 10.0 * float(np.sin(float(elapsed_s) * 1.05))
        target_line = _line_segment(center, half_span, -0.5)
        chaser_line = _line_segment(center, half_span, -0.5 - inclination_deg / 22.0)
        self._draw_primer_earth(center, max(24, int(round(min(plot.width, plot.height) * 0.07))))
        pygame.draw.line(self.screen, (96, 174, 224), target_line[0], target_line[1], width=3)
        pygame.draw.line(self.screen, CHASER_MARKER_COLOR, chaser_line[0], chaser_line[1], width=3)
        target = _point_along_line(target_line, 0.86)
        chaser = _point_along_line(chaser_line, 0.86)
        self._draw_satellite_marker(target, role="target", scale_x=half_span, scale_y=half_span, fallback_radius_px=7, force_icon=True)
        self._draw_satellite_marker(
            chaser,
            role="chaser",
            scale_x=half_span,
            scale_y=half_span,
            fallback_radius_px=7,
            force_icon=True,
        )
        self._text("Target", (target[0] - 62, target[1] + 18), self.small_font, (230, 235, 242))
        self._text("Chaser", (chaser[0] + 10, chaser[1] - 16), self.small_font, (230, 235, 242))
        self._text(
            f"Side view: chaser inclination {inclination_deg:.1f} deg",
            (plot.x + 18, plot.bottom - 26),
            self.small_font,
            (170, 184, 204),
        )

    def _draw_primer_earth(self, center: tuple[int, int], radius: int) -> None:
        pygame = self.pygame
        radius = max(int(radius), 18)
        pygame.draw.circle(self.screen, (34, 92, 142), center, radius)
        pygame.draw.circle(self.screen, (220, 240, 255), center, radius, width=1)

    def _draw_ric_primer_hud(
        self,
        rect: Any,
        *,
        stage: dict[str, Any],
        stage_index: int,
        recording_status: str = "",
    ) -> None:
        pygame = self.pygame
        pygame.draw.rect(self.screen, (18, 24, 32), rect, border_radius=10)
        pygame.draw.rect(self.screen, (82, 96, 118), rect, width=1, border_radius=10)
        status_text = str(recording_status or "G Clip").strip()
        pill_width = min(max(self._text_width(self.small_font, status_text) + 22, 86), max(rect.width // 2, 86))
        pill = pygame.Rect(rect.right - pill_width - 12, rect.y + 10, pill_width, 24)
        pygame.draw.rect(self.screen, (34, 48, 62), pill, border_radius=4)
        pygame.draw.rect(self.screen, (104, 130, 156), pill, width=1, border_radius=4)
        self._text(
            self._fit_text_px(status_text, self.small_font, pill.width - 16),
            (pill.x + 8, pill.y + 5),
            self.small_font,
            (220, 234, 246),
        )
        text_width = max(pill.x - rect.x - 32, 120)
        self._text(
            self._fit_text_px(f"Step {stage_index + 1}/3  {stage['title']}", self.font, text_width),
            (rect.x + 16, rect.y + 12),
            self.font,
            (235, 240, 245),
        )
        self._text(
            self._fit_text_px(str(stage["hint"]), self.small_font, rect.width - 32),
            (rect.x + 16, rect.y + 38),
            self.small_font,
            (245, 210, 110),
        )
        self._text(
            self._fit_text_px("Space Next   R Replay   Esc Quit", self.small_font, rect.width - 32),
            (rect.x + 16, rect.y + 60),
            self.small_font,
            (195, 205, 220),
        )

    def _draw_panel(self, rect: Any, title: str, x_axis: int, y_axis: int) -> None:
        pygame = self.pygame
        pygame.draw.rect(self.screen, (20, 27, 36), rect, border_radius=10)
        pygame.draw.rect(self.screen, (80, 92, 110), rect, width=1, border_radius=10)
        self._text(title, (rect.x + 14, rect.y + 10), self.font, (230, 235, 242))
        plot = rect.inflate(-48, -72)
        plot.y += int(getattr(self, "_plot_panel_title_gap_px", 28))
        pygame.draw.rect(self.screen, (8, 11, 16), plot)
        pygame.draw.rect(self.screen, (72, 84, 102), plot, width=1)
        if self._plot_view_mode_for_axes(x_axis=x_axis, y_axis=y_axis) == "eci":
            self._draw_eci_orbit_plane_panel(plot)
            return
        if not self.rel_hist:
            return
        rel = self._frame_cache.get("rel")
        target_rel = self._frame_cache.get("target_rel")
        target_reference_rel = self._frame_cache.get("target_reference_rel")
        if rel is None or target_rel is None:
            rel = _dashboard_history_array(self, "_rel_array", self.rel_hist, width=6)
            target_rel = _dashboard_history_array(
                self,
                "_target_rel_array",
                self.target_rel_hist[-rel.shape[0] :],
                width=6,
            )
        if target_reference_rel is None:
            target_reference_rel = _dashboard_history_array(
                self,
                "_target_reference_rel_array",
                getattr(self, "target_reference_rel_hist", [])[-rel.shape[0] :],
                width=6,
            )
        target_current = target_rel[-1, :3] if target_rel.size else np.zeros(3, dtype=float)
        target_state_for_sun = self._current_target_state_eci_for_sun()
        current_time_s = self._current_time_s()
        target_reference_current = (
            target_reference_rel[-1, :3] if target_reference_rel.size else np.zeros(3, dtype=float)
        )
        ghost = self._frame_cache.get("ghost")
        if ghost is None:
            ghost = self._coast_prediction()
        tutorial_path = np.asarray(self._frame_cache.get("tutorial_path_sample", np.empty((0, 6))), dtype=float)
        if tutorial_path.ndim != 2 or tutorial_path.shape[1] < 3:
            tutorial_path = np.empty((0, 6), dtype=float)
        target_ghost = self._frame_cache.get("target_ghost")
        if target_ghost is None:
            target_ghost = self._target_coast_prediction(target_rel)
        prediction_scales_camera = self._prediction_scales_current_camera()
        scale_ghost = ghost if prediction_scales_camera else np.empty((0, 6), dtype=float)
        scale_target_ghost = target_ghost if prediction_scales_camera else np.empty((0, 6), dtype=float)
        goal = np.array(self.goal_relative_ric_km, dtype=float).reshape(-1)
        nmt = self._frame_cache.get("nmt")
        if nmt is None:
            nmt = self._nmt_points()
        nmt_bounds = self._frame_cache.get("nmt_bounds")
        if nmt_bounds is None:
            nmt_bounds = self._nmt_boundary_points()
        chaser_current = rel[-1, :3]
        camera_center = self._camera_center_ric(
            chaser_current=chaser_current,
            target_current=target_current,
            x_axis=x_axis,
            y_axis=y_axis,
        )
        pts_for_scale = [
            (chaser_current - camera_center)[[x_axis, y_axis]].reshape(1, 2),
            (target_current - camera_center)[[x_axis, y_axis]].reshape(1, 2),
        ]
        pts_for_scale.extend(
            self._camera_rule_scale_points(
                rel=rel,
                target_rel=target_rel,
                ghost=scale_ghost,
                target_ghost=scale_target_ghost,
                x_axis=x_axis,
                y_axis=y_axis,
                camera_center=camera_center,
            )
        )
        if tutorial_path.size:
            pts_for_scale.append((tutorial_path[:, :3] - camera_center.reshape(1, 3))[:, [x_axis, y_axis]])
        min_span_km = self._minimum_plot_span_km(
            x_axis=x_axis,
            y_axis=y_axis,
            offset=target_current - camera_center,
        )
        scale_x, scale_y = self._axis_scales_for_plot(
            plot,
            pts=pts_for_scale,
            min_span_km=min_span_km,
            x_axis=x_axis,
            y_axis=y_axis,
        )
        pixel_cache = self._frame_cache.setdefault("pixel_polyline_cache", {})
        x_display_sign = self._axis_display_sign(x_axis)
        y_display_sign = self._axis_display_sign(y_axis)
        transform_key = (
            int(x_axis),
            int(y_axis),
            round(float(x_display_sign), 1),
            round(float(y_display_sign), 1),
            int(plot.centerx),
            int(plot.centery),
            round(float(scale_x), 12),
            round(float(scale_y), 12),
            tuple(round(float(value), 12) for value in camera_center.reshape(3)),
        )

        def to_px(point: np.ndarray) -> tuple[int, int]:
            shifted = np.array(point, dtype=float).reshape(-1)[:3] - camera_center
            x = float(shifted[x_axis]) * x_display_sign
            y = float(shifted[y_axis]) * y_display_sign
            px = plot.centerx + int(round(x * scale_x))
            py = plot.centery - int(round(y * scale_y))
            return px, py

        def rows_to_px(rows: np.ndarray, cache_key: str = "") -> list[tuple[int, int]]:
            arr = np.asarray(rows, dtype=float).reshape(-1, rows.shape[-1] if hasattr(rows, "shape") else 3)
            if arr.shape[1] < 3:
                return []
            key = None
            if cache_key:
                key = (cache_key, transform_key, id(rows), arr.shape)
                cached_points = pixel_cache.get(key)
                if cached_points is not None:
                    return cached_points
            shifted = arr[:, :3] - camera_center.reshape(1, 3)
            px = np.rint(plot.centerx + shifted[:, int(x_axis)] * x_display_sign * scale_x).astype(int)
            py = np.rint(plot.centery - shifted[:, int(y_axis)] * y_display_sign * scale_y).astype(int)
            points = list(zip(px.tolist(), py.tolist()))
            if key is not None:
                pixel_cache[key] = points
            return points

        def circle_rect(center: tuple[int, int], radius_km: float) -> Any:
            rx = max(1, int(round(float(radius_km) * scale_x)))
            ry = max(1, int(round(float(radius_km) * scale_y)))
            return self.pygame.Rect(center[0] - rx, center[1] - ry, rx * 2, ry * 2)

        plot_transforms = self._frame_cache.setdefault("plot_transforms", {})
        plot_transforms[(int(x_axis), int(y_axis))] = {
            "plot": (int(plot.x), int(plot.y), int(plot.width), int(plot.height)),
            "camera_center": tuple(float(value) for value in camera_center.reshape(3)),
            "scale_x": float(scale_x),
            "scale_y": float(scale_y),
            "x_display_sign": float(x_display_sign),
            "y_display_sign": float(y_display_sign),
        }

        previous_clip = self.screen.get_clip()
        self.screen.set_clip(plot)
        self._draw_cislunar_moon_background(
            plot,
            x_axis=x_axis,
            y_axis=y_axis,
            to_px=to_px,
            scale_x=scale_x,
            scale_y=scale_y,
        )
        self._draw_grid(plot, scale_x=scale_x, scale_y=scale_y)
        self._draw_sun_angle_constraints(
            plot,
            x_axis=x_axis,
            y_axis=y_axis,
            to_px=to_px,
            offset=target_current,
            target_state_eci=target_state_for_sun,
            time_s=current_time_s,
        )
        self._draw_forbidden_regions(plot, x_axis=x_axis, y_axis=y_axis, to_px=to_px, offset=target_current)
        self._draw_inspection_gates(plot, x_axis=x_axis, y_axis=y_axis, to_px=to_px, offset=target_current)
        self._draw_approach_gates(plot, x_axis=x_axis, y_axis=y_axis, to_px=to_px, offset=target_current)
        if self._show_proximity_rings_for_plane(x_axis=x_axis, y_axis=y_axis):
            if self.keepout_radius_km is not None and float(self.keepout_radius_km) > 0.0:
                center = to_px(target_current)
                pygame.draw.ellipse(
                    self.screen, (190, 68, 68), circle_rect(center, float(self.keepout_radius_km)), width=2
                )
            if self.goal_range_km is not None and float(self.goal_range_km) > 0.0:
                center = to_px(target_current)
                outer = float(self.goal_range_km)
                inner = float(self.goal_range_km)
                if self.goal_range_tolerance_km is not None:
                    tol = max(float(self.goal_range_tolerance_km), 0.0)
                    inner = max(float(self.goal_range_km) - tol, 0.0)
                    outer = float(self.goal_range_km) + tol
                pygame.draw.ellipse(self.screen, (78, 178, 112), circle_rect(center, outer), width=2)
                if inner > 0.0 and not np.isclose(inner, outer):
                    pygame.draw.ellipse(self.screen, (78, 178, 112), circle_rect(center, inner), width=2)
            if self.goal_radius_km is not None and float(self.goal_radius_km) > 0.0 and goal.size == 3:
                center = to_px(target_current + goal)
                pygame.draw.ellipse(
                    self.screen, (78, 178, 112), circle_rect(center, float(self.goal_radius_km)), width=2
                )
            if (
                self.max_target_reference_range_km is not None
                and float(self.max_target_reference_range_km) > 0.0
            ):
                self._draw_dashed_ellipse(
                    circle_rect(to_px(target_reference_current), float(self.max_target_reference_range_km)),
                    color=(76, 214, 120),
                    dash_count=72,
                    width=2,
                )
            if self.hard_speed_limit_radius_km is not None and float(self.hard_speed_limit_radius_km) > 0.0:
                center = to_px(target_current)
                pygame.draw.ellipse(
                    self.screen,
                    (232, 194, 74),
                    circle_rect(center, float(self.hard_speed_limit_radius_km)),
                    width=2,
                )
        for boundary_index, boundary in enumerate(nmt_bounds):
            if boundary.size:
                boundary_pts = rows_to_px(boundary, cache_key=f"nmt_boundary:{boundary_index}")
                if len(boundary_pts) >= 2:
                    pygame.draw.lines(self.screen, (78, 178, 112), True, boundary_pts, width=2)
        if _should_draw_nominal_nmt(nmt, nmt_bounds):
            nmt_pts = rows_to_px(nmt, cache_key="nmt")
            self._draw_polyline_dashed(nmt_pts, color=(120, 236, 154), dash_px=18, gap_px=14, width=2)
        if tutorial_path.size:
            tutorial_pts = rows_to_px(tutorial_path, cache_key="tutorial_path")
            self._draw_polyline_dashed(tutorial_pts, color=(92, 240, 132), dash_px=16, gap_px=8, width=3)
        pygame.draw.line(self.screen, (90, 104, 124), (plot.left, plot.centery), (plot.right, plot.centery), width=1)
        pygame.draw.line(self.screen, (90, 104, 124), (plot.centerx, plot.top), (plot.centerx, plot.bottom), width=1)
        pygame.draw.circle(self.screen, (80, 92, 112), (plot.centerx, plot.centery), 4)
        target_px = to_px(target_current)
        target_ghost_sample = self._frame_cache.get("target_ghost_sample", target_ghost)
        if bool(getattr(self, "show_target_coast_prediction", False)) and target_ghost_sample.size:
            target_ghost_pts = rows_to_px(target_ghost_sample, cache_key="target_ghost")
            self._draw_polyline_dashed(target_ghost_pts, color=COAST_PREDICTION_COLOR, dash_px=10, gap_px=7, width=2)
        target_trail_rows = self._frame_cache.get("target_trail", target_rel[-self.max_history :])
        if target_trail_rows.size and len(target_trail_rows) >= 2:
            target_trail = rows_to_px(target_trail_rows, cache_key="target_trail")
            pygame.draw.lines(self.screen, TARGET_TRAIL_COLOR, False, target_trail, width=2)
        self._draw_satellite_marker(target_px, role="target", scale_x=scale_x, scale_y=scale_y, fallback_radius_px=6)

        trail_rows = self._frame_cache.get("rel_trail", rel[-self.max_history :])
        trail = rows_to_px(trail_rows, cache_key="rel_trail")
        ghost_sample = self._frame_cache.get("ghost_sample", ghost)
        if ghost_sample.size:
            ghost_pts = rows_to_px(ghost_sample, cache_key="ghost")
            ghost_color = (
                LIVE_BURN_COLOR
                if bool(self._frame_cache.get("ghost_active_burn", False))
                else COAST_PREDICTION_COLOR
            )
            self._draw_polyline_dashed(ghost_pts, color=ghost_color, dash_px=8, gap_px=8, width=2)
        if len(trail) >= 2:
            pygame.draw.lines(self.screen, CHASER_TRAIL_COLOR, False, trail, width=2)
        self._draw_burn_markers(rel=rel, to_px=to_px, marker_rows=self._frame_cache.get("burn_marker_rel"))
        chaser = trail[-1]
        self._draw_satellite_marker(
            chaser,
            role="chaser",
            scale_x=scale_x,
            scale_y=scale_y,
            fallback_radius_px=7,
        )

        if rel.shape[1] >= 6:
            v_px = self._web_velocity_vector_px(rel[-1], x_axis=x_axis, y_axis=y_axis)
            v_px = v_px * np.array([x_display_sign, y_display_sign], dtype=float)
            self._draw_vector(to_px(rel[-1]), v_px, color=VELOCITY_VECTOR_COLOR, scale=1.0)
        live_accel_ric = np.array(
            getattr(self, "live_prediction_accel_ric_km_s2", np.zeros(3, dtype=float)), dtype=float
        ).reshape(3)
        threshold = float(getattr(self, "burn_marker_threshold_km_s2", 0.0))
        if np.linalg.norm(live_accel_ric) > threshold:
            thrust_vec = self._web_thrust_vector_px(
                live_accel_ric,
                x_axis=x_axis,
                y_axis=y_axis,
                threshold=threshold,
            )
            thrust_vec = thrust_vec * np.array([x_display_sign, y_display_sign], dtype=float)
            controlled_id = str(getattr(self, "controlled_object_id", "") or "")
            target_id = str(getattr(self, "target_object_id", "target") or "target")
            origin = target_px if controlled_id == target_id else chaser
            self._draw_vector(origin, thrust_vec, color=LIVE_BURN_COLOR, scale=1.0)
        self.screen.set_clip(previous_clip)

        self._draw_signed_axis_labels(
            plot,
            x_axis=int(x_axis),
            y_axis=int(y_axis),
            x_display_sign=x_display_sign,
            y_display_sign=y_display_sign,
        )

    def _draw_eci_orbit_plane_panel(self, plot: Any) -> None:
        target_eci = _dashboard_history_array(
            self,
            "_target_eci_array",
            getattr(self, "target_eci_hist", ()),
            width=6,
        )
        chaser_eci = _dashboard_history_array(
            self,
            "_chaser_eci_array",
            getattr(self, "chaser_eci_hist", ()),
            width=6,
        )
        if target_eci.size == 0 or chaser_eci.size == 0:
            return
        frame_key = _relative_frame_key(getattr(self, "relative_frame", "ric"))
        center_state = cr3bp_moon_state_km_s() if frame_key == "moon_ric" else np.zeros(6, dtype=float)
        center_radius_km = MOON_RADIUS_KM if frame_key == "moon_ric" else EARTH_RADIUS_KM
        center_fill = (88, 92, 102) if frame_key == "moon_ric" else (30, 74, 126)
        center_outline = (174, 180, 192) if frame_key == "moon_ric" else (72, 138, 202)
        draw_history_trails = frame_key != "moon_ric"
        panel_label = "Moon-centered target orbit" if frame_key == "moon_ric" else "Earth-centered ECI plane"
        target_centered = target_eci - center_state
        chaser_centered = chaser_eci - center_state
        if frame_key == "moon_ric":
            target_xy = _project_moon_rotating_yz_to_plane(target_centered[:, :3])
            chaser_xy = _project_moon_rotating_yz_to_plane(chaser_centered[:, :3])
        else:
            basis = self._eci_target_plane_basis(target_centered[-1])
            if basis is None:
                self._text("Orbit-plane view unavailable", (plot.x + 16, plot.y + 16), self.small_font, (232, 194, 74))
                return
            i_hat, r_hat, _c_hat = basis
            target_xy = _project_eci_positions_to_plane(target_centered[:, :3], x_hat=i_hat, y_hat=r_hat)
            chaser_xy = _project_eci_positions_to_plane(chaser_centered[:, :3], x_hat=i_hat, y_hat=r_hat)
        if target_xy.size == 0 or chaser_xy.size == 0:
            return
        target_orbit_xy = np.empty((0, 2), dtype=float)
        if frame_key == "moon_ric":
            target_orbit = self._cr3bp_target_orbit_prediction(allow_build=False)
            if target_orbit.size:
                target_orbit_centered = np.array(target_orbit, dtype=float).reshape(-1, 6) - center_state
                target_orbit_xy = _project_moon_rotating_yz_to_plane(target_orbit_centered[:, :3])
        target_radius_km = float(np.linalg.norm(target_centered[-1, :3]))
        half_span_km = max(target_radius_km * 1.18, center_radius_km * 1.25, 1.0)
        target_scale_xy = target_xy if draw_history_trails else target_xy[-1:].copy()
        chaser_scale_xy = chaser_xy if draw_history_trails else chaser_xy[-1:].copy()
        all_xy_rows = [target_scale_xy, chaser_scale_xy, np.array([[0.0, 0.0]], dtype=float)]
        if target_orbit_xy.size and frame_key != "moon_ric":
            all_xy_rows.append(target_orbit_xy)
        all_xy = np.vstack(all_xy_rows)
        max_abs = float(np.nanmax(np.abs(all_xy))) if all_xy.size else 0.0
        if np.isfinite(max_abs):
            half_span_km = max(half_span_km, max_abs * 1.18)
        scale = min(plot.width, plot.height) * 0.5 / half_span_km

        def to_px_xy(xy: np.ndarray) -> tuple[int, int]:
            vec = np.array(xy, dtype=float).reshape(2)
            return (
                plot.centerx + int(round(float(vec[0]) * scale)),
                plot.centery - int(round(float(vec[1]) * scale)),
            )

        def rows_to_px(rows: np.ndarray) -> list[tuple[int, int]]:
            arr = np.array(rows, dtype=float).reshape(-1, 2)
            px = np.rint(plot.centerx + arr[:, 0] * scale).astype(int)
            py = np.rint(plot.centery - arr[:, 1] * scale).astype(int)
            return list(zip(px.tolist(), py.tolist()))

        pygame = self.pygame
        previous_clip = self.screen.get_clip()
        self.screen.set_clip(plot)
        self._draw_grid(plot, scale_x=scale, scale_y=scale)
        if target_orbit_xy.size:
            target_orbit_trail = rows_to_px(_sample_rows(target_orbit_xy, MAX_TARGET_ORBIT_DRAW_POINTS))
            if len(target_orbit_trail) >= 2:
                pygame.draw.lines(self.screen, (66, 76, 92), False, target_orbit_trail, width=1)
        else:
            orbit_radius_px = max(2, int(round(target_radius_km * scale)))
            orbit_rect = pygame.Rect(0, 0, orbit_radius_px * 2, orbit_radius_px * 2)
            orbit_rect.center = plot.center
            pygame.draw.ellipse(self.screen, (66, 76, 92), orbit_rect, width=1)
        pygame.draw.line(self.screen, (90, 104, 124), (plot.left, plot.centery), (plot.right, plot.centery), width=1)
        pygame.draw.line(self.screen, (90, 104, 124), (plot.centerx, plot.top), (plot.centerx, plot.bottom), width=1)
        center_radius_px = max(2, int(round(center_radius_km * scale)))
        pygame.draw.circle(self.screen, center_fill, plot.center, center_radius_px)
        pygame.draw.circle(self.screen, center_outline, plot.center, center_radius_px, width=2)

        if draw_history_trails:
            target_trail = rows_to_px(_sample_rows(target_xy, MAX_TRAIL_DRAW_POINTS))
            chaser_trail = rows_to_px(_sample_rows(chaser_xy, MAX_TRAIL_DRAW_POINTS))
            if len(target_trail) >= 2:
                pygame.draw.lines(self.screen, TARGET_TRAIL_COLOR, False, target_trail, width=2)
            if len(chaser_trail) >= 2:
                pygame.draw.lines(self.screen, CHASER_TRAIL_COLOR, False, chaser_trail, width=2)
        target_px = to_px_xy(target_xy[-1])
        chaser_px = to_px_xy(chaser_xy[-1])
        self._draw_satellite_marker(
            target_px,
            role="target",
            scale_x=scale,
            scale_y=scale,
            fallback_radius_px=6,
        )
        self._draw_satellite_marker(
            chaser_px,
            role="chaser",
            scale_x=scale,
            scale_y=scale,
            fallback_radius_px=7,
        )
        self.screen.set_clip(previous_clip)
        x_label = "T" if frame_key == "moon_ric" else "I"
        y_label = "N" if frame_key == "moon_ric" else "R"
        self._text(x_label, (plot.right - 34, plot.centery + 8), self.small_font, (170, 180, 195))
        self._text(y_label, (plot.centerx + 8, plot.top + 8), self.small_font, (170, 180, 195))
        self._text(panel_label, (plot.x + 14, plot.bottom - 24), self.small_font, (150, 160, 176))

    def _prepare_cr3bp_target_orbit_prediction(self) -> None:
        if _relative_frame_key(getattr(self, "relative_frame", "ric")) != "moon_ric":
            return
        if not self._uses_cr3bp_prediction_model():
            return
        prediction_cache = getattr(self, "_prediction_cache", {})
        if "target_absolute_cr3bp_orbit" in prediction_cache:
            return
        self._cr3bp_target_orbit_prediction(allow_build=True)

    def _cr3bp_target_orbit_prediction(self, *, allow_build: bool = True) -> np.ndarray:
        if _relative_frame_key(getattr(self, "relative_frame", "ric")) != "moon_ric":
            return np.empty((0, 6), dtype=float)
        if not self._uses_cr3bp_prediction_model():
            return np.empty((0, 6), dtype=float)
        reference = self._target_orbit_reference_state()
        if reference is None:
            return np.empty((0, 6), dtype=float)
        horizon = _positive_float_or_none(getattr(self, "target_coast_prediction_horizon_s", None))
        if horizon is None:
            horizon = _positive_float_or_none(getattr(self, "cr3bp_coast_prediction_horizon_s", None))
        if horizon is None:
            n = getattr(self, "mean_motion_rad_s", None)
            if n is None:
                return np.empty((0, 6), dtype=float)
            horizon = self._coast_prediction_horizon_s(float(n))
        if horizon is None or float(horizon) <= 0.0:
            return np.empty((0, 6), dtype=float)
        dt = _positive_float_or_none(getattr(self, "target_coast_prediction_dt_s", None))
        if dt is None:
            dt = _positive_float_or_none(getattr(self, "cr3bp_coast_prediction_dt_s", None))
        if dt is None:
            dt = 3600.0
        dt = max(float(dt), 1.0e-6)
        horizon = float(horizon)
        max_points = CR3BP_TARGET_ORBIT_MAX_POINTS
        count = min(int(np.floor(horizon / dt)) + 1, max_points)
        times = np.linspace(0.0, horizon, max(count, 2), dtype=float)
        cache_key = "target_absolute_cr3bp_orbit"
        prediction_cache = getattr(self, "_prediction_cache", {})
        self._prediction_cache = prediction_cache
        now_s = self._current_time_s()
        cached = prediction_cache.get(cache_key)
        if cached is not None:
            if (
                _cr3bp_reference_cache_valid(cached.get("reference"), reference)
                and float(cached.get("horizon_s", np.nan)) == horizon
                and float(cached.get("dt_s", np.nan)) == dt
            ):
                prediction = cached.get("prediction")
                if prediction is not None:
                    return np.array(prediction, dtype=float)
        if not bool(allow_build):
            return np.empty((0, 6), dtype=float)

        state = reference.copy()
        rows: list[np.ndarray] = []
        current_t = 0.0
        previous_t = 0.0
        for target_t in times:
            step_s = float(target_t - previous_t)
            if step_s > 0.0:
                remaining_s = step_s
                while remaining_s > 1.0e-9:
                    substep_s = min(float(CR3BP_TARGET_ORBIT_INTERNAL_STEP_S), remaining_s)
                    state = propagate_cr3bp_state(state, substep_s, current_t)
                    current_t += substep_s
                    remaining_s -= substep_s
            rows.append(state.copy())
            previous_t = float(target_t)
        prediction = np.vstack(rows) if rows else np.empty((0, 6), dtype=float)
        prediction_cache[cache_key] = {
            "time_s": now_s,
            "prediction": prediction,
            "reference": reference.copy(),
            "horizon_s": horizon,
            "dt_s": dt,
        }
        return prediction

    def _target_orbit_reference_state(self) -> np.ndarray | None:
        reference = getattr(self, "target_orbit_reference_state_eci", None)
        if reference is not None:
            return np.array(reference, dtype=float).reshape(6).copy()
        target_eci = _dashboard_history_array(
            self,
            "_target_eci_array",
            getattr(self, "target_eci_hist", ()),
            width=6,
        )
        if target_eci.size:
            reference = np.array(target_eci[0], dtype=float).reshape(6).copy()
            self.target_orbit_reference_state_eci = reference.copy()
            return reference
        reference = self._reference_cache_state()
        if reference is None:
            return None
        reference = np.array(reference, dtype=float).reshape(6).copy()
        self.target_orbit_reference_state_eci = reference.copy()
        return reference

    def _current_target_state_eci_for_sun(self) -> np.ndarray | None:
        target_eci = _dashboard_history_array(
            self,
            "_target_eci_array",
            getattr(self, "target_eci_hist", ()),
            width=6,
        )
        if target_eci.size:
            return np.array(target_eci[-1], dtype=float).reshape(6).copy()
        reference = self._reference_cache_state()
        if reference is None:
            return None
        return np.array(reference, dtype=float).reshape(6).copy()

    @staticmethod
    def _eci_target_plane_basis(target_state_eci: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        state = np.array(target_state_eci, dtype=float).reshape(-1)
        if state.size < 6:
            return None
        r_vec = state[:3]
        v_vec = state[3:6]
        r_norm = float(np.linalg.norm(r_vec))
        h_vec = np.cross(r_vec, v_vec)
        h_norm = float(np.linalg.norm(h_vec))
        if r_norm <= 0.0 or h_norm <= 0.0 or not np.isfinite(r_norm) or not np.isfinite(h_norm):
            return None
        r_hat = r_vec / r_norm
        c_hat = h_vec / h_norm
        i_hat = np.cross(c_hat, r_hat)
        i_norm = float(np.linalg.norm(i_hat))
        if i_norm <= 0.0 or not np.isfinite(i_norm):
            return None
        i_hat = i_hat / i_norm
        return i_hat.astype(float), r_hat.astype(float), c_hat.astype(float)

    def _plot_panel_specs(self) -> tuple[tuple[str, int, int], tuple[str, int, int]]:
        if _relative_frame_key(getattr(self, "relative_frame", "ric")) == "cislunar_l1":
            return (
                ("Cislunar XY: Tangential Vs Earth-Moon", 1, 0),
                ("Cislunar YZ: Tangential Vs Out-of-Plane", 1, 2),
            )
        return (
            (self._panel_title_for_axes("RI Plane: In-Track Vs Radial", x_axis=1, y_axis=0), 1, 0),
            (self._panel_title_for_axes("RC Plane: Cross-Track Vs Radial", x_axis=2, y_axis=0), 2, 0),
        )

    def _panel_title_for_axes(self, default_title: str, *, x_axis: int, y_axis: int) -> str:
        if self._plot_view_mode_for_axes(x_axis=x_axis, y_axis=y_axis) != "eci":
            return default_title
        plane = _plane_key_for_axes(x_axis=x_axis, y_axis=y_axis) or "RIC"
        if _relative_frame_key(getattr(self, "relative_frame", "ric")) == "moon_ric":
            return f"Moon View ({plane} Swap): Tangential Vs Normal"
        return f"ECI View ({plane} Swap): In-Track Vs Radial"

    def _plot_view_mode_for_axes(self, *, x_axis: int, y_axis: int) -> str:
        if _relative_frame_key(getattr(self, "relative_frame", "ric")) == "cislunar_l1":
            return "ric"
        plane = _plane_key_for_axes(x_axis=x_axis, y_axis=y_axis)
        modes = getattr(self, "plot_view_modes", {}) or {}
        mode = str(modes.get(plane, "ric") or "ric").strip().lower()
        return "eci" if mode == "eci" else "ric"

    def toggle_eci_plot(self, plane: str) -> str:
        key = str(plane or "").strip().upper()
        if key not in {"RI", "RC"}:
            return "ric"
        modes = dict(getattr(self, "plot_view_modes", {}) or {})
        next_mode = "ric" if str(modes.get(key, "ric")).strip().lower() == "eci" else "eci"
        modes[key] = next_mode
        self.plot_view_modes = modes
        if hasattr(self, "_frame_cache_dirty"):
            self._frame_cache_dirty = True
        return next_mode

    def _axis_label_for_plot(self, axis: int) -> str:
        return f"{self._axis_symbol_for_plot(axis)} km"

    def _axis_symbol_for_plot(self, axis: int) -> str:
        if _relative_frame_key(getattr(self, "relative_frame", "ric")) == "cislunar_l1":
            labels = {
                0: "EM",
                1: "T",
                2: "N",
            }
        else:
            labels = {0: "R", 1: "I", 2: "C"}
        return labels.get(int(axis), "")

    def _signed_axis_label_for_plot(self, axis: int, sign: int) -> str:
        symbol = self._axis_symbol_for_plot(axis)
        if not symbol:
            return ""
        prefix = "+" if int(sign) >= 0 else "-"
        return f"{prefix}{symbol}"

    def _draw_signed_axis_labels(
        self,
        plot: Any,
        *,
        x_axis: int,
        y_axis: int,
        x_display_sign: float,
        y_display_sign: float,
    ) -> None:
        color = (170, 180, 195)
        x_plus = self._signed_axis_label_for_plot(x_axis, 1)
        x_minus = self._signed_axis_label_for_plot(x_axis, -1)
        y_plus = self._signed_axis_label_for_plot(y_axis, 1)
        y_minus = self._signed_axis_label_for_plot(y_axis, -1)

        x_plus_right = float(x_display_sign) >= 0.0
        x_left_label = x_minus if x_plus_right else x_plus
        x_right_label = x_plus if x_plus_right else x_minus
        self._text(x_left_label, (plot.left + 10, plot.centery + 8), self.small_font, color)
        right_x = plot.right - self._text_width(self.small_font, x_right_label) - 10
        self._text(x_right_label, (right_x, plot.centery + 8), self.small_font, color)

        y_plus_top = float(y_display_sign) >= 0.0
        y_top_label = y_plus if y_plus_top else y_minus
        y_bottom_label = y_minus if y_plus_top else y_plus
        self._text(y_top_label, (plot.centerx + 8, plot.top + 8), self.small_font, color)
        self._text(y_bottom_label, (plot.centerx + 8, plot.bottom - 24), self.small_font, color)

    def _axis_display_sign(self, axis: int) -> float:
        if _relative_frame_key(getattr(self, "relative_frame", "ric")) == "cislunar_l1":
            return 1.0
        return frame_convention_display_axis_sign(getattr(self, "frame_convention", FrameConvention()), int(axis))

    def _draw_cislunar_moon_background(
        self,
        plot: Any,
        *,
        x_axis: int,
        y_axis: int,
        to_px: Any,
        scale_x: float,
        scale_y: float,
    ) -> None:
        if not _should_draw_cislunar_moon_background(
            relative_frame=getattr(self, "relative_frame", "ric"),
            x_axis=int(x_axis),
            y_axis=int(y_axis),
        ):
            return
        rect_tuple = _scaled_body_rect_tuple(
            center_px=to_px(np.zeros(3, dtype=float)),
            radius_km=MOON_RADIUS_KM,
            scale_x=scale_x,
            scale_y=scale_y,
        )
        pygame = self.pygame
        rect = pygame.Rect(*rect_tuple)
        clipped = rect.clip(plot)
        if clipped.width <= 0 or clipped.height <= 0:
            return
        fill = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        fill.fill((0, 0, 0, 0))
        local_rect = pygame.Rect(0, 0, rect.width, rect.height)
        pygame.draw.ellipse(fill, (150, 158, 166, 42), local_rect)
        pygame.draw.ellipse(fill, (210, 218, 226, 86), local_rect, width=1)
        # A tiny limb and crater pattern gives the disk lunar character while
        # preserving the exact radius set by local_rect.
        if rect.width >= 18 and rect.height >= 18:
            shade = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
            shade.fill((0, 0, 0, 0))
            pygame.draw.ellipse(shade, (54, 60, 68, 38), pygame.Rect(rect.width // 3, 0, rect.width, rect.height))
            fill.blit(shade, (0, 0))
            for frac_x, frac_y, frac_r, alpha in (
                (0.34, 0.42, 0.035, 30),
                (0.58, 0.32, 0.028, 24),
                (0.64, 0.62, 0.045, 28),
                (0.44, 0.68, 0.024, 22),
            ):
                crater = pygame.Rect(0, 0, max(2, int(rect.width * frac_r)), max(2, int(rect.height * frac_r)))
                crater.center = (int(rect.width * frac_x), int(rect.height * frac_y))
                pygame.draw.ellipse(fill, (72, 78, 86, alpha), crater, width=1)
        source = pygame.Rect(clipped.x - rect.x, clipped.y - rect.y, clipped.width, clipped.height)
        self.screen.blit(fill, (clipped.x, clipped.y), source)

    def _prepare_frame_cache(self) -> None:
        if not self.rel_hist:
            self._frame_cache = {}
            self._raw_frame_cache = {}
            self._frame_cache_dirty = False
            return
        if (
            not bool(getattr(self, "_frame_cache_dirty", True))
            and getattr(self, "_frame_cache", None)
            and not bool(getattr(self, "_render_motion_enabled", False))
        ):
            return
        raw_cache = getattr(self, "_raw_frame_cache", {})
        if getattr(self, "_frame_cache_dirty", True) or not raw_cache:
            raw_rel = _dashboard_history_array(self, "_rel_array", self.rel_hist, width=6)
            raw_target_rel = _dashboard_history_array(
                self,
                "_target_rel_array",
                self.target_rel_hist[-raw_rel.shape[0] :],
                width=6,
            )
            raw_target_reference_rel = _dashboard_history_array(
                self,
                "_target_reference_rel_array",
                getattr(self, "target_reference_rel_hist", [])[-raw_rel.shape[0] :],
                width=6,
            )
            thrust = _dashboard_history_array(
                self,
                "_thrust_ric_array",
                self.thrust_ric_hist[-raw_rel.shape[0] :],
                width=3,
            )
            raw_cache = {
                "raw_rel": raw_rel,
                "raw_target_rel": raw_target_rel,
                "raw_target_reference_rel": raw_target_reference_rel,
                "thrust": thrust,
                "target_ghost": self._target_coast_prediction(raw_target_rel),
                "nmt": self._nmt_points(),
                "nmt_bounds": self._nmt_boundary_points(),
            }
            self._raw_frame_cache = raw_cache
        raw_rel = np.asarray(raw_cache["raw_rel"], dtype=float)
        raw_target_rel = np.asarray(raw_cache["raw_target_rel"], dtype=float)
        raw_target_reference_rel = np.asarray(raw_cache["raw_target_reference_rel"], dtype=float)
        thrust = np.asarray(raw_cache["thrust"], dtype=float)
        rel = self._visual_state_rows(raw_rel)
        target_rel = self._visual_state_rows(raw_target_rel)
        target_reference_rel = self._visual_state_rows(raw_target_reference_rel)
        live_accel = np.array(getattr(self, "live_prediction_accel_ric_km_s2", np.zeros(3)), dtype=float).reshape(3)
        active_burn = bool(np.linalg.norm(live_accel) > float(self.burn_marker_threshold_km_s2))
        target_ghost = np.array(raw_cache.get("target_ghost", np.empty((0, 6))), dtype=float)
        ghost_seed = self._live_prediction_seed(raw_rel[-1])
        ghost = self._coast_prediction_from_cached("chaser", ghost_seed, active_burn=active_burn)
        operator_ghost, operator_transition_active = self._operator_projection_transition_ghost()
        if operator_transition_active and operator_ghost.size:
            ghost = operator_ghost
            active_burn = True
        burn_marker_rel = self._burn_marker_rows(rel=rel, thrust=thrust)
        tutorial_path = np.asarray(getattr(self, "tutorial_target_path_ric", np.empty((0, 6))), dtype=float)
        if tutorial_path.ndim != 2 or tutorial_path.shape[1] < 3:
            tutorial_path_sample = np.empty((0, 6), dtype=float)
        else:
            tutorial_path_sample = _sample_rows(tutorial_path, MAX_GHOST_DRAW_POINTS)
        self._frame_cache = {
            "rel": rel,
            "target_rel": target_rel,
            "target_reference_rel": target_reference_rel,
            "thrust": thrust,
            "ghost": ghost,
            "ghost_sample": _sample_rows(ghost, MAX_GHOST_DRAW_POINTS),
            "ghost_active_burn": active_burn,
            "target_ghost": target_ghost,
            "target_ghost_sample": _sample_rows(target_ghost, MAX_GHOST_DRAW_POINTS),
            "rel_trail": _sample_rows(rel[-self.max_history :], MAX_TRAIL_DRAW_POINTS),
            "target_trail": _sample_rows(target_rel[-self.max_history :], MAX_TRAIL_DRAW_POINTS),
            "tutorial_path_sample": tutorial_path_sample,
            "burn_marker_rel": burn_marker_rel,
            "nmt": np.array(raw_cache.get("nmt", np.empty((0, 3))), dtype=float),
            "nmt_bounds": tuple(np.array(row, dtype=float) for row in raw_cache.get("nmt_bounds", ())),
            "pixel_polyline_cache": {},
        }
        self._frame_cache_dirty = False

    def _visual_state_rows(self, rows: np.ndarray) -> np.ndarray:
        arr = np.asarray(rows, dtype=float)
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 6:
            return arr
        if not bool(getattr(self, "visual_extrapolation_enabled", True)):
            return arr
        if not bool(getattr(self, "_render_motion_enabled", False)):
            return arr
        elapsed_sim_s = self._visual_extrapolation_elapsed_sim_s()
        if elapsed_sim_s <= 0.0:
            return arr
        arr = arr.copy()
        arr[-1, :3] = arr[-1, :3] + arr[-1, 3:6] * elapsed_sim_s
        return arr

    def _visual_extrapolation_elapsed_sim_s(self) -> float:
        sample_wall = getattr(self, "sample_wall_s", ())
        if not sample_wall:
            return 0.0
        latest_wall_s = float(sample_wall[-1])
        render_wall_s = float(getattr(self, "_render_wall_time_s", latest_wall_s))
        elapsed_wall_s = max(render_wall_s - latest_wall_s, 0.0)
        speed = max(float(getattr(self, "_render_speed_multiple", 1.0)), 0.0)
        elapsed_sim_s = elapsed_wall_s * speed
        cap = _positive_float_or_none(getattr(self, "visual_extrapolation_max_sim_s", None))
        if cap is None:
            cap = VISUAL_EXTRAPOLATION_MAX_SIM_S
        t_s = getattr(self, "t_s", ())
        if t_s and len(t_s) >= 2:
            latest_dt_s = max(float(t_s[-1]) - float(t_s[-2]), 0.0)
            if latest_dt_s > 0.0:
                cap = min(float(cap), latest_dt_s)
        return float(min(max(elapsed_sim_s, 0.0), max(float(cap), 0.0)))

    def _live_prediction_seed(self, rel0: np.ndarray) -> np.ndarray:
        seed = np.array(rel0, dtype=float).reshape(6)
        accel = np.array(getattr(self, "live_prediction_accel_ric_km_s2", np.zeros(3)), dtype=float).reshape(3)
        elapsed = float(getattr(self, "live_prediction_elapsed_s", 0.0))
        if self._uses_cr3bp_prediction_model():
            if (
                elapsed > 0.0
                and np.all(np.isfinite(accel))
                and float(np.linalg.norm(accel)) > float(self.burn_marker_threshold_km_s2)
            ):
                seed[:3] += seed[3:6] * elapsed + 0.5 * accel * elapsed * elapsed
                seed[3:6] += accel * elapsed
            return seed
        n = getattr(self, "mean_motion_rad_s", None)
        if (
            elapsed <= 0.0
            or n is None
            or not np.isfinite(float(n))
            or float(n) <= 0.0
            or not np.all(np.isfinite(accel))
            or float(np.linalg.norm(accel)) <= float(self.burn_marker_threshold_km_s2)
        ):
            return seed
        return _cw_forced_state(seed, accel, elapsed, float(n))

    def _operator_projection_transition_active(self) -> bool:
        transition = getattr(self, "_operator_projection_transition", None)
        if not transition:
            return False
        started = float(transition.get("started_wall_s", 0.0))
        duration = max(float(transition.get("duration_s", 0.0)), 0.1)
        if perf_counter() - started <= duration:
            return True
        self._operator_projection_transition = None
        self._frame_cache_dirty = True
        return False

    def _operator_projection_transition_ghost(self) -> tuple[np.ndarray, bool]:
        transition = getattr(self, "_operator_projection_transition", None)
        if not transition:
            return np.empty((0, 6), dtype=float), False
        started = float(transition.get("started_wall_s", 0.0))
        duration = max(float(transition.get("duration_s", 0.0)), 0.1)
        alpha = (perf_counter() - started) / duration
        if alpha >= 1.0:
            self._operator_projection_transition = None
            self._frame_cache_dirty = True
            return np.empty((0, 6), dtype=float), False
        alpha = float(min(max(alpha, 0.0), 1.0))
        pre_seed = np.array(transition.get("pre", np.zeros(6)), dtype=float).reshape(6)
        post_seed = np.array(transition.get("post", np.zeros(6)), dtype=float).reshape(6)
        pre_ghost = self._coast_prediction_from_cached(
            "operator_transition_pre",
            pre_seed,
            active_burn=False,
        )
        post_ghost = self._coast_prediction_from_cached(
            "operator_transition_post",
            post_seed,
            active_burn=False,
        )
        if pre_ghost.size == 0 or post_ghost.size == 0:
            return np.empty((0, 6), dtype=float), False
        sample_count = min(pre_ghost.shape[0], post_ghost.shape[0])
        if sample_count <= 0:
            return np.empty((0, 6), dtype=float), False
        blended = pre_ghost[:sample_count] * (1.0 - alpha) + post_ghost[:sample_count] * alpha
        return blended, True

    def _draw_grid(
        self,
        plot: Any,
        *,
        scale: float | None = None,
        scale_x: float | None = None,
        scale_y: float | None = None,
    ) -> None:
        pygame = self.pygame
        if scale_x is None:
            scale_x = scale
        if scale_y is None:
            scale_y = scale
        if scale_x is None or scale_y is None or scale_x <= 0.0 or scale_y <= 0.0:
            return
        visible_span_km = max(plot.width / max(scale_x, 1e-9), plot.height / max(scale_y, 1e-9))
        step_km = self._nice_step(visible_span_km / 6.0)
        if step_km <= 0.0:
            return
        max_x_km = plot.width / max(scale_x, 1e-9)
        max_y_km = plot.height / max(scale_y, 1e-9)
        for k in np.arange(-max_x_km, max_x_km + step_km, step_km):
            x = plot.centerx + int(round(float(k) * scale_x))
            if plot.left <= x <= plot.right:
                pygame.draw.line(self.screen, (30, 38, 50), (x, plot.top), (x, plot.bottom), width=1)
        for k in np.arange(-max_y_km, max_y_km + step_km, step_km):
            y = plot.centery - int(round(float(k) * scale_y))
            if plot.top <= y <= plot.bottom:
                pygame.draw.line(self.screen, (30, 38, 50), (plot.left, y), (plot.right, y), width=1)

    def _range_goal_projection_points(self, *, x_axis: int, y_axis: int, offset: np.ndarray) -> list[np.ndarray]:
        if self.goal_range_km is None:
            return []
        radius = float(self.goal_range_km)
        if self.goal_range_tolerance_km is not None:
            radius += max(float(self.goal_range_tolerance_km), 0.0)
        if radius <= 0.0 or not np.isfinite(radius):
            return []
        center = np.array([offset[x_axis], offset[y_axis]], dtype=float)
        return [
            np.array(
                [
                    center + np.array([radius, 0.0], dtype=float),
                    center + np.array([-radius, 0.0], dtype=float),
                    center + np.array([0.0, radius], dtype=float),
                    center + np.array([0.0, -radius], dtype=float),
                ],
                dtype=float,
            )
        ]

    def _sun_angle_projection_points(
        self, *, x_axis: int, y_axis: int, offset: np.ndarray
    ) -> list[np.ndarray]:
        constraints = tuple(getattr(self, "sun_angle_constraints", ()) or ())
        if not constraints:
            return []
        pts: list[np.ndarray] = []
        target_state_eci = self._current_target_state_eci_for_sun()
        time_s = self._current_time_s()
        for constraint in constraints:
            polygon = _sun_angle_sector_polygon_ric(
                constraint,
                x_axis=x_axis,
                y_axis=y_axis,
                target_state_eci=target_state_eci,
                time_s=time_s,
            )
            if polygon.size:
                polygon = polygon.copy()
                polygon[:, :3] += np.array(offset, dtype=float).reshape(1, 3)
                pts.append(polygon[:, [x_axis, y_axis]])
        return pts

    def _draw_sun_angle_constraints(
        self,
        plot: Any,
        *,
        x_axis: int,
        y_axis: int,
        to_px: Any,
        offset: np.ndarray,
        target_state_eci: np.ndarray | None = None,
        time_s: float | None = None,
    ) -> None:
        constraints = tuple(getattr(self, "sun_angle_constraints", ()) or ())
        if not constraints:
            return
        pygame = self.pygame
        for constraint in constraints:
            if not _constraint_visible_on_plane(constraint, x_axis=x_axis, y_axis=y_axis):
                continue
            polygon = _sun_angle_sector_polygon_ric(
                constraint,
                x_axis=x_axis,
                y_axis=y_axis,
                target_state_eci=target_state_eci,
                time_s=time_s,
            )
            if not polygon.size:
                continue
            polygon = polygon.copy()
            polygon[:, :3] += np.array(offset, dtype=float).reshape(1, 3)
            points = [to_px(row) for row in polygon]
            if len(points) >= 3:
                self._draw_translucent_polygon(plot, points, color=(235, 188, 74, 46))
                pygame.draw.lines(self.screen, (247, 207, 101), True, points, width=1)
            centerline = _sun_angle_centerline_points_ric(
                constraint,
                x_axis=x_axis,
                y_axis=y_axis,
                target_state_eci=target_state_eci,
                time_s=time_s,
            )
            if centerline.size:
                centerline = centerline.copy()
                centerline[:, :3] += np.array(offset, dtype=float).reshape(1, 3)
                line = [to_px(row) for row in centerline]
                if len(line) >= 2:
                    pygame.draw.line(self.screen, (255, 231, 153), line[0], line[1], width=2)

    def _forbidden_region_projection_points(
        self, *, x_axis: int, y_axis: int, offset: np.ndarray
    ) -> list[np.ndarray]:
        pts: list[np.ndarray] = []
        for region in self.forbidden_regions:
            if not _region_visible_on_plane(region, x_axis=x_axis, y_axis=y_axis):
                continue
            if region.kind == "annular_sector":
                polygon = region.sector_polygon_ric()
                if polygon.size:
                    polygon = polygon.copy()
                    polygon[:, :3] += np.array(offset, dtype=float).reshape(1, 3)
                    pts.append(polygon[:, [x_axis, y_axis]])
                continue
            if region.kind == "cylinder":
                polygon = _cylinder_projection_polygon_ric(region, x_axis=x_axis, y_axis=y_axis)
                if polygon.size:
                    polygon = polygon.copy()
                    polygon[:, :3] += np.array(offset, dtype=float).reshape(1, 3)
                    pts.append(polygon[:, [x_axis, y_axis]])
                continue
            if region.kind == "sphere":
                polygon = _sphere_projection_polygon_ric(region, x_axis=x_axis, y_axis=y_axis)
                if polygon.size:
                    polygon = polygon.copy()
                    polygon[:, :3] += np.array(offset, dtype=float).reshape(1, 3)
                    pts.append(polygon[:, [x_axis, y_axis]])
                continue
            bounds = _finite_projected_region_bounds(region, x_axis=x_axis, y_axis=y_axis)
            if bounds is None:
                continue
            lo, hi = bounds
            corners = np.array(
                [
                    [lo[0], lo[1]],
                    [lo[0], hi[1]],
                    [hi[0], lo[1]],
                    [hi[0], hi[1]],
                ],
                dtype=float,
            )
            corners += np.array([offset[x_axis], offset[y_axis]], dtype=float).reshape(1, 2)
            pts.append(corners)
        return pts

    def _draw_forbidden_regions(self, plot: Any, *, x_axis: int, y_axis: int, to_px: Any, offset: np.ndarray) -> None:
        if not self.forbidden_regions:
            return
        pygame = self.pygame
        for region in self.forbidden_regions:
            if not _region_visible_on_plane(region, x_axis=x_axis, y_axis=y_axis):
                continue
            if region.kind == "annular_sector":
                polygon = region.sector_polygon_ric()
                if not polygon.size:
                    continue
                polygon = polygon.copy()
                polygon[:, :3] += np.array(offset, dtype=float).reshape(1, 3)
                points = [to_px(row) for row in polygon]
                if len(points) >= 3:
                    self._draw_translucent_polygon(plot, points, color=(168, 44, 54, 58))
                    pygame.draw.lines(self.screen, (230, 80, 92), True, points, width=1)
                continue
            if region.kind == "cylinder":
                polygon = _cylinder_projection_polygon_ric(region, x_axis=x_axis, y_axis=y_axis)
                if not polygon.size:
                    continue
                polygon = polygon.copy()
                polygon[:, :3] += np.array(offset, dtype=float).reshape(1, 3)
                points = [to_px(row) for row in polygon]
                if len(points) >= 3:
                    self._draw_translucent_polygon(plot, points, color=(168, 44, 54, 58))
                    pygame.draw.lines(self.screen, (230, 80, 92), True, points, width=1)
                continue
            if region.kind == "sphere":
                polygon = _sphere_projection_polygon_ric(region, x_axis=x_axis, y_axis=y_axis)
                if not polygon.size:
                    continue
                polygon = polygon.copy()
                polygon[:, :3] += np.array(offset, dtype=float).reshape(1, 3)
                points = [to_px(row) for row in polygon]
                if len(points) >= 3:
                    self._draw_translucent_polygon(plot, points, color=(168, 44, 54, 58))
                    pygame.draw.lines(self.screen, (230, 80, 92), True, points, width=1)
                continue
            bounds = _finite_projected_region_bounds(region, x_axis=x_axis, y_axis=y_axis)
            if bounds is None:
                continue
            lo, hi = bounds
            p_min = np.zeros(3, dtype=float)
            p_max = np.zeros(3, dtype=float)
            p_min[x_axis] = lo[0] + float(offset[x_axis])
            p_min[y_axis] = lo[1] + float(offset[y_axis])
            p_max[x_axis] = hi[0] + float(offset[x_axis])
            p_max[y_axis] = hi[1] + float(offset[y_axis])
            a = to_px(p_min)
            b = to_px(p_max)
            rect = pygame.Rect(min(a[0], b[0]), min(a[1], b[1]), abs(a[0] - b[0]), abs(a[1] - b[1]))
            clipped = rect.clip(plot)
            if clipped.width <= 0 or clipped.height <= 0:
                continue
            fill = pygame.Surface((clipped.width, clipped.height), pygame.SRCALPHA)
            fill.fill((168, 44, 54, 58))
            self.screen.blit(fill, (clipped.x, clipped.y))
            pygame.draw.rect(self.screen, (230, 80, 92), clipped, width=1)

    def _approach_gate_projection_points(
        self, *, x_axis: int, y_axis: int, offset: np.ndarray
    ) -> list[np.ndarray]:
        pts: list[np.ndarray] = []
        for gate in self.approach_gates:
            bounds = _approach_gate_projected_bounds(gate, x_axis=x_axis, y_axis=y_axis)
            if bounds is None:
                continue
            lo, hi = bounds
            corners = np.array(
                [
                    [lo[0], lo[1]],
                    [lo[0], hi[1]],
                    [hi[0], lo[1]],
                    [hi[0], hi[1]],
                ],
                dtype=float,
            )
            corners += np.array([offset[x_axis], offset[y_axis]], dtype=float).reshape(1, 2)
            pts.append(corners)
        return pts

    def _inspection_gate_projection_points(
        self, *, x_axis: int, y_axis: int, offset: np.ndarray
    ) -> list[np.ndarray]:
        pts: list[np.ndarray] = []
        for gate in self.inspection_gates:
            lo = np.array(gate.center_ric_km, dtype=float).reshape(3) - np.array(
                gate.half_width_ric_km, dtype=float
            ).reshape(3)
            hi = np.array(gate.center_ric_km, dtype=float).reshape(3) + np.array(
                gate.half_width_ric_km, dtype=float
            ).reshape(3)
            corners = np.array(
                [
                    [lo[x_axis], lo[y_axis]],
                    [lo[x_axis], hi[y_axis]],
                    [hi[x_axis], lo[y_axis]],
                    [hi[x_axis], hi[y_axis]],
                ],
                dtype=float,
            )
            corners += np.array([offset[x_axis], offset[y_axis]], dtype=float).reshape(1, 2)
            pts.append(corners)
        return pts

    def _draw_approach_gates(self, plot: Any, *, x_axis: int, y_axis: int, to_px: Any, offset: np.ndarray) -> None:
        if not self.approach_gates:
            return
        pygame = self.pygame
        for gate in self.approach_gates:
            bounds = _approach_gate_projected_bounds(gate, x_axis=x_axis, y_axis=y_axis)
            if bounds is None:
                continue
            lo, hi = bounds
            p_min = np.zeros(3, dtype=float)
            p_max = np.zeros(3, dtype=float)
            p_min[x_axis] = lo[0] + float(offset[x_axis])
            p_min[y_axis] = lo[1] + float(offset[y_axis])
            p_max[x_axis] = hi[0] + float(offset[x_axis])
            p_max[y_axis] = hi[1] + float(offset[y_axis])
            a = to_px(p_min)
            b = to_px(p_max)
            rect = pygame.Rect(min(a[0], b[0]), min(a[1], b[1]), abs(a[0] - b[0]), abs(a[1] - b[1]))
            clipped = rect.clip(plot)
            if clipped.width <= 0 or clipped.height <= 0:
                continue
            fill = pygame.Surface((clipped.width, clipped.height), pygame.SRCALPHA)
            fill.fill((245, 205, 92, 46))
            self.screen.blit(fill, (clipped.x, clipped.y))
            pygame.draw.rect(self.screen, (245, 205, 92), clipped, width=1)

    def _draw_inspection_gates(self, plot: Any, *, x_axis: int, y_axis: int, to_px: Any, offset: np.ndarray) -> None:
        if not self.inspection_gates:
            return
        pygame = self.pygame
        for idx, gate in enumerate(self.inspection_gates, start=1):
            lo = np.array(gate.center_ric_km, dtype=float) - np.array(gate.half_width_ric_km, dtype=float)
            hi = np.array(gate.center_ric_km, dtype=float) + np.array(gate.half_width_ric_km, dtype=float)
            p_min = np.zeros(3, dtype=float)
            p_max = np.zeros(3, dtype=float)
            p_min[x_axis] = lo[x_axis] + float(offset[x_axis])
            p_min[y_axis] = lo[y_axis] + float(offset[y_axis])
            p_max[x_axis] = hi[x_axis] + float(offset[x_axis])
            p_max[y_axis] = hi[y_axis] + float(offset[y_axis])
            a = to_px(p_min)
            b = to_px(p_max)
            rect = pygame.Rect(min(a[0], b[0]), min(a[1], b[1]), abs(a[0] - b[0]), abs(a[1] - b[1]))
            clipped = rect.clip(plot)
            if clipped.width <= 0 or clipped.height <= 0:
                continue
            fill = pygame.Surface((clipped.width, clipped.height), pygame.SRCALPHA)
            fill.fill((78, 178, 112, 42))
            self.screen.blit(fill, (clipped.x, clipped.y))
            pygame.draw.rect(self.screen, (100, 226, 142), clipped, width=1)
            self._text(str(idx), (clipped.x + 5, clipped.y + 4), self.small_font, (170, 250, 190))

    def _draw_hud(
        self,
        rect: Any,
        *,
        command_status: str,
        coach_hint: str,
        speed_multiple: float,
        selected_speed_multiple: float | None = None,
        recording_status: str = "",
    ) -> None:
        pygame = self.pygame
        pygame.draw.rect(self.screen, (18, 24, 32), rect, border_radius=10)
        pygame.draw.rect(self.screen, (82, 96, 118), rect, width=1, border_radius=10)
        if self.rel_hist:
            rel = self.rel_hist[-1]
            rng = float(np.linalg.norm(rel[:3]))
            spd = float(np.linalg.norm(rel[3:]))
            t = self.t_s[-1] if self.t_s else 0.0
            anomaly = self._true_anomaly_indicator_text()
            suffix = f"   {anomaly}" if anomaly else ""
            self._text(
                f"T={t:7.1f}s   Range={format_distance_km(rng)}   Rel Speed={format_speed_km_s(spd)}{suffix}",
                (rect.x + 16, rect.y + 12),
                self.font,
                (235, 240, 245),
            )
        status_text = str(recording_status or "G Clip").strip()
        active = status_text.upper().startswith("REC")
        text_width = self._text_width(self.small_font, status_text)
        pill_width = min(max(text_width + 22, 86), max(rect.width // 2, 86))
        pill = pygame.Rect(rect.right - pill_width - 12, rect.y + 10, pill_width, 24)
        pygame.draw.rect(self.screen, (96, 22, 32) if active else (34, 48, 62), pill, border_radius=4)
        pygame.draw.rect(self.screen, (248, 84, 100) if active else (104, 130, 156), pill, width=1, border_radius=4)
        self._text(
            self._fit_text_px(status_text, self.small_font, pill.width - 16),
            (pill.x + 8, pill.y + 5),
            self.small_font,
            (255, 220, 224) if active else (220, 234, 246),
        )
        hint_text = coach_hint or ""
        hint_is_alert = hint_text.lower().startswith("wrong key")
        if hint_is_alert:
            alert_rect = pygame.Rect(rect.x + 10, rect.y + 34, min(rect.width - 20, 560), 24)
            pygame.draw.rect(self.screen, (96, 28, 34), alert_rect, border_radius=4)
            pygame.draw.rect(self.screen, (242, 92, 104), alert_rect, width=1, border_radius=4)
        self._text(
            hint_text,
            (rect.x + 16, rect.y + 38),
            self.small_font,
            (255, 190, 198) if hint_is_alert else (245, 210, 110),
        )
        selected_speed = None if selected_speed_multiple is None else float(selected_speed_multiple)
        active_speed = float(speed_multiple)
        if selected_speed is not None and not np.isclose(selected_speed, active_speed):
            speed_label = f"Active {active_speed:.0f}x  Coast {selected_speed:.0f}x"
        else:
            speed_label = f"Speed {active_speed:.0f}x"
        footer = f"{speed_label}  Up/Down  Space Pause  R Reset  Esc Select"
        footer_text = self._fit_text_px(
            footer,
            self.small_font,
            min(rect.width - 24, 690),
            preserve_spaces=True,
        )
        footer_width = self._text_width(self.small_font, footer_text)
        footer_x = rect.right - footer_width - 16
        command_line = command_status.splitlines()[0] if command_status else ""
        command_max_width = max(0, footer_x - (rect.x + 16) - 12)
        if command_line and command_max_width > 48:
            command_text = self._fit_text_px(
                command_line,
                self.small_font,
                command_max_width,
                preserve_spaces=True,
            )
            self._text(command_text, (rect.x + 16, rect.y + 60), self.small_font, (195, 205, 220))
        self._text(
            footer_text,
            (footer_x, rect.y + 58),
            self.small_font,
            (220, 160, 160),
        )

    def _draw_top_bar(
        self,
        rect: Any,
        *,
        mission_state: str,
        level_title: str = "",
        mission_metrics: tuple[str, ...],
        objective_checklist: tuple[str, ...] = (),
    ) -> None:
        pygame = self.pygame
        colors = {
            "active": ((18, 24, 32), (82, 96, 118), (230, 235, 242), self._top_bar_label(mission_state, level_title)),
            "passed": ((18, 54, 36), (88, 190, 122), (190, 255, 205), "LEVEL PASSED"),
            "failed": ((62, 24, 28), (220, 94, 94), (255, 205, 205), "LEVEL FAILED"),
        }
        fill, stroke, text_color, label = colors.get(mission_state, colors["active"])
        pygame.draw.rect(self.screen, fill, rect, border_radius=10)
        pygame.draw.rect(self.screen, stroke, rect, width=2, border_radius=10)
        self._text(label, (rect.x + 16, rect.y + 9), self.large_font, text_color)
        x = rect.x + 16
        y = rect.y + 43
        checklist_width = 300 if objective_checklist else 0
        hidden = 0
        for metric in mission_metrics:
            clean, metric_color = self._metric_text_and_color(metric)
            metric_width = max(118, self.small_font.size(clean)[0] + 26)
            if x + metric_width > rect.right - 16 - checklist_width:
                if y > rect.y + 43:
                    hidden += 1
                    continue
                x = rect.x + 16
                y = rect.y + 63
            self._text(clean, (x, y), self.small_font, metric_color)
            x += metric_width
        if hidden:
            self._text(
                f"+{hidden}",
                (rect.right - 34 - checklist_width, rect.y + 63),
                self.small_font,
                (222, 230, 238),
            )
        if objective_checklist:
            self._draw_objective_checklist(rect, objective_checklist)

    @staticmethod
    def _top_bar_label(mission_state: str, level_title: str = "") -> str:
        if str(mission_state or "").strip().lower() != "active":
            return ""
        title = str(level_title or "").strip()
        return title.upper() if title else "LEVEL ACTIVE"

    def _draw_objective_checklist(self, rect: Any, objective_checklist: tuple[str, ...]) -> None:
        x = rect.right - 292
        y = rect.y + 12
        self._text("OBJECTIVES", (x, y), self.small_font, (170, 184, 204))
        y += 18
        for item in objective_checklist[:3]:
            clean, color = self._metric_text_and_color(str(item))
            self._text(clean, (x, y), self.small_font, color)
            y += 17

    def _metric_text_and_color(self, metric: str) -> tuple[str, tuple[int, int, int]]:
        text = str(metric)
        if text.startswith("OK "):
            return text, (150, 235, 170)
        if text.startswith("WARN "):
            return text, (245, 210, 110)
        if text.startswith("FAIL "):
            return text, (255, 150, 150)
        if text.startswith("INFO "):
            return text, (150, 205, 245)
        return text, (222, 230, 238)

    def _draw_briefing(self, lines: tuple[str, ...]) -> None:
        pygame = self.pygame
        width, height = self.screen.get_size()
        rect_w = min(860, max(width - 96, 420))
        rect_h = min(500, max(height - 96, 320))
        rect = pygame.Rect(width // 2 - rect_w // 2, height // 2 - rect_h // 2, rect_w, rect_h)
        pygame.draw.rect(self.screen, (15, 24, 34), rect, border_radius=10)
        pygame.draw.rect(self.screen, (96, 174, 224), rect, width=2, border_radius=10)
        title = self.large_font.render(str(lines[0] if lines else "Mission Brief"), True, (238, 244, 250))
        self.screen.blit(title, (rect.x + 30, rect.y + 24))

        content_rect = pygame.Rect(rect.x + 32, rect.y + 70, rect.width - 76, rect.height - 136)
        wrapped_lines = self._briefing_body_lines(lines[1:], content_rect.width)
        content_height = len(wrapped_lines) * BRIEFING_LINE_HEIGHT_PX
        max_scroll = max(content_height - content_rect.height, 0)
        self.briefing_scroll_px = min(max(int(self.briefing_scroll_px), 0), max_scroll)

        previous_clip = self.screen.get_clip()
        self.screen.set_clip(content_rect)
        y = content_rect.y - self.briefing_scroll_px
        for line in wrapped_lines:
            if y + BRIEFING_LINE_HEIGHT_PX >= content_rect.y and y <= content_rect.bottom:
                self._text(line, (content_rect.x, y), self.font, (206, 218, 232))
            y += BRIEFING_LINE_HEIGHT_PX
        self.screen.set_clip(previous_clip)

        if max_scroll > 0:
            self._draw_briefing_scrollbar(rect, content_rect, max_scroll)

        self._text(
            self._briefing_footer_text(max_scroll > 0),
            (rect.x + 32, rect.bottom - 48),
            self.font,
            (220, 160, 160),
        )

    def _briefing_body_lines(self, lines: tuple[str, ...], width_px: int) -> list[str]:
        cache_key = (tuple(str(line) for line in lines), int(width_px), id(self.font))
        briefing_cache = getattr(self, "_briefing_layout_cache", {})
        if briefing_cache.get("key") == cache_key:
            return list(briefing_cache.get("lines", [""]))
        body: list[str] = []
        for raw in lines:
            body.extend(self._wrap_text_px(str(raw), self.font, width_px))
            body.append("")
        while body and body[-1] == "":
            body.pop()
        wrapped = body or [""]
        self._briefing_layout_cache = {"key": cache_key, "lines": tuple(wrapped)}
        return list(wrapped)

    def _draw_briefing_scrollbar(self, rect: Any, content_rect: Any, max_scroll: int) -> None:
        pygame = self.pygame
        track = pygame.Rect(rect.right - 28, content_rect.y, 5, content_rect.height)
        pygame.draw.rect(self.screen, (42, 58, 76), track, border_radius=3)
        thumb_h = max(int(track.height * content_rect.height / (content_rect.height + max_scroll)), 28)
        thumb_y = track.y + int((track.height - thumb_h) * (self.briefing_scroll_px / max_scroll))
        thumb = pygame.Rect(track.x, thumb_y, track.width, thumb_h)
        pygame.draw.rect(self.screen, (116, 194, 238), thumb, border_radius=3)

    @staticmethod
    def _briefing_footer_text(scrollable: bool) -> str:
        if scrollable:
            return "Scroll To Read. Press Space To Start. Esc Returns To Level Select."
        return "Press Space To Start. Esc Returns To Level Select."

    def _draw_mission_banner(
        self,
        mission_state: str,
        *,
        debrief_lines: tuple[str, ...] = (),
        debrief_available: bool = True,
        ) -> None:
        pygame = self.pygame
        width, height = self.screen.get_size()
        rect = pygame.Rect(width // 2 - 360, height // 2 - 190, 720, 380)
        if mission_state == "passed":
            fill = (24, 86, 48)
            stroke = (108, 232, 142)
            text = "MISSION PASSED"
            base_sub = (
                "D Debrief  R Replay  Esc Quit"
                if debrief_available
                else "R Replay  Esc Quit"
            )
            color = (210, 255, 220)
        else:
            fill = (90, 30, 36)
            stroke = (244, 102, 102)
            text = "MISSION FAILED"
            base_sub = (
                "D Debrief  R Retry  Esc Quit"
                if debrief_available
                else "R Retry  Esc Quit"
            )
            color = (255, 220, 220)
        pygame.draw.rect(self.screen, fill, rect, border_radius=10)
        pygame.draw.rect(self.screen, stroke, rect, width=3, border_radius=10)
        title = self.large_font.render(text, True, color)
        self.screen.blit(title, (rect.centerx - title.get_width() // 2, rect.y + 22))
        content_rect = pygame.Rect(rect.x + 36, rect.y + 70, rect.width - 78, rect.height - 136)
        body_lines = self._mission_banner_body_lines(debrief_lines, content_rect.width)
        content_height = len(body_lines) * MISSION_BANNER_LINE_HEIGHT_PX
        max_scroll = max(content_height - content_rect.height, 0)
        self.mission_banner_scroll_px = min(max(int(getattr(self, "mission_banner_scroll_px", 0)), 0), max_scroll)

        previous_clip = self.screen.get_clip()
        self.screen.set_clip(content_rect)
        y = content_rect.y - self.mission_banner_scroll_px
        for line in body_lines:
            if y + MISSION_BANNER_LINE_HEIGHT_PX >= content_rect.y and y <= content_rect.bottom:
                self._text(str(line), (content_rect.x, y), self.font, color)
            y += MISSION_BANNER_LINE_HEIGHT_PX
        self.screen.set_clip(previous_clip)

        if max_scroll > 0:
            self._draw_mission_banner_scrollbar(rect, content_rect, max_scroll)
        sub = self._fit_text_px(
            self._mission_banner_footer_text(base_sub, max_scroll > 0),
            self.font,
            rect.width - 72,
            preserve_spaces=True,
        )
        subtitle = self.font.render(sub, True, color)
        self.screen.blit(subtitle, (rect.centerx - subtitle.get_width() // 2, rect.bottom - 42))

    def _mission_banner_body_lines(self, lines: tuple[str, ...], width_px: int) -> list[str]:
        cache_key = (tuple(str(line) for line in lines), int(width_px), id(self.font))
        layout_cache = getattr(self, "_mission_banner_layout_cache", {})
        if layout_cache.get("key") == cache_key:
            return list(layout_cache.get("lines", [""]))
        wrapped: list[str] = []
        for raw in lines:
            wrapped.extend(self._wrap_mission_banner_line(str(raw), width_px))
        body = wrapped or [""]
        self._mission_banner_layout_cache = {"key": cache_key, "lines": tuple(body)}
        return list(body)

    def _wrap_mission_banner_line(self, value: str, width_px: int) -> list[str]:
        text = str(value or "")
        if self._text_width(self.font, text) <= width_px:
            return [text]
        label_width_chars = 14
        if len(text) > label_width_chars and text[:label_width_chars].strip():
            label = text[:label_width_chars]
            body = text[label_width_chars:].strip()
            body_width = max(width_px - self._text_width(self.font, label), 80)
            wrapped_body = self._wrap_text_px(body, self.font, body_width)
            if wrapped_body:
                return [label + wrapped_body[0], *(" " * label_width_chars + line for line in wrapped_body[1:])]
        return self._wrap_text_px(text, self.font, width_px)

    def _draw_mission_banner_scrollbar(self, rect: Any, content_rect: Any, max_scroll: int) -> None:
        pygame = self.pygame
        track = pygame.Rect(rect.right - 28, content_rect.y, 5, content_rect.height)
        pygame.draw.rect(self.screen, (112, 50, 56), track, border_radius=3)
        thumb_h = max(int(track.height * content_rect.height / (content_rect.height + max_scroll)), 28)
        thumb_y = track.y + int((track.height - thumb_h) * (self.mission_banner_scroll_px / max_scroll))
        thumb = pygame.Rect(track.x, thumb_y, track.width, thumb_h)
        pygame.draw.rect(self.screen, (255, 172, 172), thumb, border_radius=3)

    @staticmethod
    def _mission_banner_footer_text(base_text: str, scrollable: bool) -> str:
        if scrollable:
            return "Scroll/Page  " + str(base_text)
        return str(base_text)

    def _draw_pause_overlay(self) -> None:
        pygame = self.pygame
        width, height = self.screen.get_size()
        rect_w = min(900, max(width - 128, 560))
        rect_h = min(430, max(height - 152, 340))
        rect = pygame.Rect(width // 2 - rect_w // 2, height // 2 - rect_h // 2, rect_w, rect_h)
        pygame.draw.rect(self.screen, (13, 22, 32), rect, border_radius=10)
        pygame.draw.rect(self.screen, (116, 194, 238), rect, width=2, border_radius=10)

        self._text("PAUSED - RIC MOTION CARD", (rect.x + 28, rect.y + 22), self.large_font, (238, 244, 250))
        self._text(
            "Circular-chief HCW intuition for the local RIC frame.",
            (rect.x + 30, rect.y + 58),
            self.small_font,
            (170, 205, 230),
        )

        column_gap = 28
        left_w = max((rect.width - 76 - column_gap) // 2, 240)
        left_x = rect.x + 30
        right_x = left_x + left_w + column_gap
        section_y = rect.y + 94

        self._draw_pause_text_section(
            "HCW Equations",
            self._pause_overlay_equation_lines(),
            x=left_x,
            y=section_y,
            width_px=left_w,
        )
        takeaway_y = section_y + 138
        self._draw_pause_text_section(
            "Useful Intuition",
            self._pause_overlay_takeaway_lines(),
            x=left_x,
            y=takeaway_y,
            width_px=left_w,
        )

        diagram = pygame.Rect(right_x, section_y, rect.right - right_x - 30, rect.height - 154)
        self._draw_pause_ric_diagram(diagram)
        footer = "Space Resume   Up/Down Speed   R Reset   Esc Level Select"
        self._text(
            self._fit_text_px(footer, self.font, rect.width - 60, preserve_spaces=True),
            (rect.x + 30, rect.bottom - 44),
            self.font,
            (220, 160, 160),
        )

    @staticmethod
    def _pause_overlay_equation_lines() -> tuple[str, ...]:
        return (
            "R'' = 3 n² R + 2 n I' + a_R",
            "I'' = -2 n R' + a_I",
            "C'' = -n² C + a_C",
        )

    @staticmethod
    def _pause_overlay_takeaway_lines() -> tuple[str, ...]:
        return (
            "R is radial: up/down from the target orbit.",
            "I is in-track: ahead/behind along the orbit.",
            "C is cross-track: out of the orbital plane.",
            "Burn, then coast. The curve is the lesson.",
        )

    def _draw_pause_text_section(
        self,
        title: str,
        lines: tuple[str, ...],
        *,
        x: int,
        y: int,
        width_px: int,
    ) -> None:
        self._text(str(title).upper(), (x, y), self.small_font, (150, 205, 245))
        cursor_y = y + 24
        for line in lines:
            for wrapped in self._wrap_text_px(str(line), self.small_font, width_px):
                self._text(wrapped, (x, cursor_y), self.small_font, (222, 230, 238))
                cursor_y += 19

    def _draw_pause_ric_diagram(self, rect: Any) -> None:
        pygame = self.pygame
        pygame.draw.rect(self.screen, (8, 13, 19), rect, border_radius=8)
        pygame.draw.rect(self.screen, (72, 92, 116), rect, width=1, border_radius=8)
        title = self._fit_text_px("Target-Centered RIC Frame", self.small_font, rect.width - 28)
        self._text(title, (rect.x + 14, rect.y + 14), self.small_font, (170, 184, 204))

        origin = (rect.centerx, rect.centery + 28)
        axis_len = max(min(rect.width, rect.height) // 4, 58)
        self._draw_pause_axis(origin, (0, -axis_len), "R", (150, 235, 170), "radial")
        self._draw_pause_axis(origin, (axis_len + 16, 0), "I", (245, 210, 110), "in-track")
        self._draw_pause_axis(origin, (-axis_len // 2, axis_len // 2), "C", (150, 205, 245), "cross-track")

        self._draw_satellite_marker(
            origin,
            role="target",
            scale_x=2000.0,
            scale_y=2000.0,
            fallback_radius_px=7,
            force_icon=True,
        )
        chaser = (origin[0] - axis_len // 3, origin[1] - axis_len // 3)
        self._draw_satellite_marker(
            chaser,
            role="chaser",
            scale_x=2000.0,
            scale_y=2000.0,
            fallback_radius_px=7,
            force_icon=True,
        )
        pygame.draw.arc(
            self.screen,
            (100, 120, 145),
            pygame.Rect(origin[0] - axis_len, origin[1] - axis_len // 2, axis_len * 2, axis_len),
            0.2,
            2.9,
            width=1,
        )
        note = self._fit_text_px(
            "Target-relative, not straight-line flight.",
            self.small_font,
            rect.width - 28,
        )
        self._text(note, (rect.x + 14, rect.bottom - 34), self.small_font, (170, 184, 204))

    def _draw_pause_axis(
        self,
        origin: tuple[int, int],
        delta: tuple[int, int],
        label: str,
        color: tuple[int, int, int],
        description: str,
    ) -> None:
        pygame = self.pygame
        end = (int(origin[0] + delta[0]), int(origin[1] + delta[1]))
        pygame.draw.line(self.screen, color, origin, end, width=3)
        angle = float(np.arctan2(float(end[1] - origin[1]), float(end[0] - origin[0])))
        head = 10.0
        spread = 0.45
        pygame.draw.polygon(
            self.screen,
            color,
            [
                end,
                (
                    int(round(float(end[0]) - head * np.cos(angle - spread))),
                    int(round(float(end[1]) - head * np.sin(angle - spread))),
                ),
                (
                    int(round(float(end[0]) - head * np.cos(angle + spread))),
                    int(round(float(end[1]) - head * np.sin(angle + spread))),
                ),
            ],
        )
        label_pos = (end[0] + 8, end[1] - 8)
        self._text(str(label), label_pos, self.font, color)
        self._text(str(description), (label_pos[0] + 24, label_pos[1] + 2), self.small_font, (205, 216, 230))

    def _draw_vector(
        self,
        origin: tuple[int, int],
        vector: np.ndarray,
        *,
        color: tuple[int, int, int],
        scale: float,
        label: str = "",
    ) -> None:
        pygame = self.pygame
        vec = np.array(vector, dtype=float).reshape(2)
        if not np.all(np.isfinite(vec)) or np.linalg.norm(vec) <= 0.0:
            return
        start = (int(origin[0]), int(origin[1]))
        end = (int(origin[0] + vec[0] * scale), int(origin[1] - vec[1] * scale))
        angle = float(np.arctan2(float(end[1] - start[1]), float(end[0] - start[0])))
        pygame.draw.line(self.screen, color, origin, end, width=2)
        head = float(WEB_VECTOR_ARROW_HEAD_PX)
        spread = float(WEB_VECTOR_ARROW_HEAD_ANGLE_RAD)
        pygame.draw.polygon(
            self.screen,
            color,
            [
                end,
                (
                    int(round(float(end[0]) - head * np.cos(angle - spread))),
                    int(round(float(end[1]) - head * np.sin(angle - spread))),
                ),
                (
                    int(round(float(end[0]) - head * np.cos(angle + spread))),
                    int(round(float(end[1]) - head * np.sin(angle + spread))),
                ),
            ],
        )
        if label:
            self._text(label, (end[0] + 6, end[1] - 8), self.small_font, color)

    @staticmethod
    def _web_velocity_vector_px(rel_state: np.ndarray, *, x_axis: int, y_axis: int) -> np.ndarray:
        state = np.array(rel_state, dtype=float).reshape(-1)
        if state.size < 6:
            return np.zeros(2, dtype=float)
        return np.array(
            [
                float(state[int(x_axis) + 3]) * WEB_VECTOR_VREL_SCALE_PX_PER_KM_S,
                float(state[int(y_axis) + 3]) * WEB_VECTOR_VREL_SCALE_PX_PER_KM_S,
            ],
            dtype=float,
        )

    @staticmethod
    def _web_thrust_vector_px(
        thrust_ric: np.ndarray,
        *,
        x_axis: int,
        y_axis: int,
        threshold: float = 0.0,
    ) -> np.ndarray:
        thrust = np.array(thrust_ric, dtype=float).reshape(-1)
        if thrust.size < 3:
            return np.zeros(2, dtype=float)
        thrust = thrust[:3]
        norm = float(np.linalg.norm(thrust))
        if not np.isfinite(norm) or norm <= max(float(threshold), 0.0):
            return np.zeros(2, dtype=float)
        unit = thrust / norm
        return (
            np.array([float(unit[int(x_axis)]), float(unit[int(y_axis)])], dtype=float)
            * WEB_VECTOR_THRUST_SCALE_PX
        )

    def _burn_marker_rows(self, *, rel: np.ndarray, thrust: np.ndarray) -> np.ndarray:
        thrust_arr = np.array(thrust, dtype=float).reshape(-1, 3)
        if thrust_arr.size == 0:
            return np.empty((0, 6), dtype=float)
        rel_arr = np.array(rel, dtype=float).reshape(-1, 6)
        count = min(rel_arr.shape[0], thrust_arr.shape[0])
        if count <= 0:
            return np.empty((0, 6), dtype=float)
        rel_arr = rel_arr[-count:]
        thrust_arr = thrust_arr[-count:]
        active = np.linalg.norm(thrust_arr, axis=1) > float(self.burn_marker_threshold_km_s2)
        idxs = np.where(active)[0]
        if idxs.size == 0:
            return np.empty((0, 6), dtype=float)
        stride = max(1, int(np.ceil(idxs.size / 80)))
        return rel_arr[idxs[::stride]]

    def _draw_burn_markers(self, *, rel: np.ndarray, to_px: Any, marker_rows: np.ndarray | None = None) -> None:
        if marker_rows is None and not self.thrust_ric_hist:
            return
        pygame = self.pygame
        markers = (
            self._burn_marker_rows(
                rel=rel,
                thrust=_dashboard_history_array(
                    self,
                    "_thrust_ric_array",
                    self.thrust_ric_hist[-rel.shape[0] :],
                    width=3,
                ),
            )
            if marker_rows is None
            else np.array(marker_rows, dtype=float).reshape(-1, 6)
        )
        if markers.size == 0:
            return
        for row in markers:
            pygame.draw.circle(self.screen, (255, 145, 60), to_px(row), 3)

    def _draw_translucent_polygon(
        self, plot: Any, points: list[tuple[int, int]], *, color: tuple[int, int, int, int]
    ) -> None:
        if len(points) < 3 or plot.width <= 0 or plot.height <= 0:
            return
        pygame = self.pygame
        fill = pygame.Surface((plot.width, plot.height), pygame.SRCALPHA)
        local_points = [(int(x) - plot.x, int(y) - plot.y) for x, y in points]
        pygame.draw.polygon(fill, color, local_points)
        self.screen.blit(fill, (plot.x, plot.y))

    def _draw_polyline_dashed(
        self,
        points: list[tuple[int, int]],
        *,
        color: tuple[int, int, int],
        dash_px: int = 8,
        gap_px: int = 6,
        width: int = 1,
    ) -> None:
        if len(points) < 2:
            return
        pygame = self.pygame
        draw_line = pygame.draw.line
        screen = self.screen
        dash = float(dash_px)
        stride = float(dash_px + gap_px)
        for start, end in zip(points[:-1], points[1:]):
            x0 = float(start[0])
            y0 = float(start[1])
            dx = float(end[0]) - x0
            dy = float(end[1]) - y0
            length = hypot(dx, dy)
            if length <= 0.0:
                continue
            ux = dx / length
            uy = dy / length
            pos = 0.0
            while pos < length:
                stop = min(pos + dash, length)
                draw_line(
                    screen,
                    color,
                    (int(x0 + ux * pos), int(y0 + uy * pos)),
                    (int(x0 + ux * stop), int(y0 + uy * stop)),
                    width=width,
                )
                pos += stride

    def _draw_dashed_ellipse(
        self,
        rect: Any,
        *,
        color: tuple[int, int, int],
        dash_count: int = 72,
        width: int = 1,
    ) -> None:
        count = max(int(dash_count), 12)
        cx = float(rect.centerx)
        cy = float(rect.centery)
        rx = float(rect.width) / 2.0
        ry = float(rect.height) / 2.0
        if rx <= 0.0 or ry <= 0.0:
            return
        points: list[tuple[int, int]] = []
        for idx in range(count + 1):
            theta = 2.0 * np.pi * float(idx) / float(count)
            points.append((int(round(cx + rx * np.cos(theta))), int(round(cy + ry * np.sin(theta)))))
        self._draw_polyline_dashed(points, color=color, dash_px=10, gap_px=7, width=width)

    def _coast_prediction(self) -> np.ndarray:
        if not self.rel_hist:
            return np.empty((0, 6), dtype=float)
        live_accel = np.array(getattr(self, "live_prediction_accel_ric_km_s2", np.zeros(3)), dtype=float).reshape(3)
        active_burn = bool(np.linalg.norm(live_accel) > float(self.burn_marker_threshold_km_s2))
        return self._coast_prediction_from_cached(
            "chaser",
            np.array(self.rel_hist[-1], dtype=float).reshape(6),
            active_burn=active_burn,
        )

    def _target_coast_prediction(self, target_rel: np.ndarray | None = None) -> np.ndarray:
        if not bool(getattr(self, "show_target_coast_prediction", False)):
            return np.empty((0, 6), dtype=float)
        rel = np.array(target_rel, dtype=float) if target_rel is not None else np.empty((0, 6), dtype=float)
        if rel.size == 0:
            if not self.target_rel_hist:
                return np.empty((0, 6), dtype=float)
            rel0 = np.array(self.target_rel_hist[-1], dtype=float).reshape(6)
        else:
            rel0 = rel.reshape(-1, 6)[-1]
        return self._coast_prediction_from(
            rel0,
            cr3bp_horizon_s=getattr(self, "target_coast_prediction_horizon_s", None),
            cr3bp_dt_s=getattr(self, "target_coast_prediction_dt_s", None),
        )

    def _coast_prediction_from_cached(self, cache_name: str, rel0: np.ndarray, *, active_burn: bool) -> np.ndarray:
        rel0 = np.array(rel0, dtype=float).reshape(6)
        prediction_cache = getattr(self, "_prediction_cache", {})
        self._prediction_cache = prediction_cache
        if self._uses_cr3bp_prediction_model():
            interval_s = (
                CR3BP_PREDICTION_BURN_UPDATE_INTERVAL_S
                if bool(active_burn)
                else (
                    _positive_float_or_none(getattr(self, "cr3bp_prediction_coast_update_interval_s", None))
                    or CR3BP_PREDICTION_COAST_UPDATE_INTERVAL_S
                )
            )
            now_s = self._current_time_s()
            reference = self._reference_cache_state()
            cached = prediction_cache.get(str(cache_name))
            if cached is not None and interval_s > 0.0:
                age_s = now_s - float(cached.get("time_s", -np.inf))
                if (
                    age_s >= 0.0
                    and age_s < float(interval_s)
                    and _cr3bp_relative_cache_valid(cached.get("rel0"), rel0)
                    and _cr3bp_reference_cache_valid(cached.get("reference"), reference, elapsed_s=age_s)
                ):
                    prediction = cached.get("prediction")
                    if prediction is not None:
                        return np.array(prediction, dtype=float)

            prediction = self._coast_prediction_from(
                rel0,
                cr3bp_horizon_s=(
                    _positive_float_or_none(getattr(self, "cr3bp_active_prediction_horizon_s", None))
                    if bool(active_burn)
                    else None
                ),
                max_draw_points=MAX_ACTIVE_CR3BP_GHOST_DRAW_POINTS if bool(active_burn) else None,
            )
            prediction_cache[str(cache_name)] = {
                "time_s": now_s,
                "rel0": rel0.copy(),
                "prediction": prediction,
                "reference": reference,
            }
            return prediction

        if not self._uses_elliptic_prediction_model():
            prediction = self._coast_prediction_from(rel0)
            prediction_cache[str(cache_name)] = {
                "time_s": self._current_time_s(),
                "rel0": rel0.copy(),
                "prediction": prediction,
                "reference": self._reference_cache_state(),
            }
            return prediction

        interval_s = (
            ELLIPTIC_PREDICTION_BURN_UPDATE_INTERVAL_S
            if bool(active_burn)
            else ELLIPTIC_PREDICTION_COAST_UPDATE_INTERVAL_S
        )
        now_s = self._current_time_s()
        reference = self._reference_cache_state()
        cached = prediction_cache.get(str(cache_name))
        if cached is not None and interval_s > 0.0:
            age_s = now_s - float(cached.get("time_s", -np.inf))
            if age_s >= 0.0 and age_s < float(interval_s):
                prediction = cached.get("prediction")
                if prediction is not None and _elliptic_reference_cache_valid(
                    cached.get("reference"),
                    reference,
                    elapsed_s=age_s,
                ):
                    return np.array(prediction, dtype=float)

        prediction = self._coast_prediction_from(rel0)
        prediction_cache[str(cache_name)] = {
            "time_s": now_s,
            "rel0": rel0.copy(),
            "prediction": prediction,
            "reference": reference,
        }
        return prediction

    def _uses_elliptic_prediction_model(self) -> bool:
        return _coast_prediction_model_key(getattr(self, "coast_prediction_model", "hcw")) in {
            "elliptic_linear",
            "tschauner_hempel",
            "ts",
        }

    def _uses_cr3bp_prediction_model(self) -> bool:
        return _coast_prediction_model_key(getattr(self, "coast_prediction_model", "hcw")) in {
            "cr3bp",
            "cislunar",
            "cislunar_l1",
        }

    def _true_anomaly_indicator_text(self) -> str:
        if not self._uses_elliptic_prediction_model():
            return ""
        anomaly = getattr(self, "target_true_anomaly_deg", None)
        if anomaly is None:
            return ""
        value = float(anomaly)
        if not np.isfinite(value):
            return ""
        return f"Target ν={np.mod(value, 360.0):5.1f} deg"

    def _current_time_s(self) -> float:
        return float(self.t_s[-1]) if self.t_s else 0.0

    def _mission_time_remaining_s(self) -> float | None:
        budget = _positive_float_or_none(getattr(self, "mission_time_budget_s", None))
        if budget is None:
            return None
        start_s = float(self.t_s[0]) if self.t_s else 0.0
        elapsed_s = max(self._current_time_s() - start_s, 0.0)
        return max(float(budget) - elapsed_s, 0.0)

    def _reference_cache_state(self) -> np.ndarray | None:
        reference_state = getattr(self, "reference_state_eci", None)
        if reference_state is None:
            return None
        return np.array(reference_state, dtype=float).reshape(6).copy()

    def _linearized_cr3bp_moon_ric_coast_prediction_cached(
        self,
        rel0: np.ndarray,
        *,
        target_state: np.ndarray,
        times: np.ndarray,
        current_t_s: float,
    ) -> np.ndarray:
        prediction_cache = getattr(self, "_prediction_cache", {})
        self._prediction_cache = prediction_cache
        cache_key = "_linearized_cr3bp_moon_ric_stm_table"
        target = np.array(target_state, dtype=float).reshape(6)
        time_grid = np.array(times, dtype=float).reshape(-1)
        cached = prediction_cache.get(cache_key)
        references: np.ndarray | None = None
        stms: np.ndarray | None = None
        basis_axes: np.ndarray | None = None
        basis_omega: np.ndarray | None = None
        if isinstance(cached, dict):
            cached_times = np.array(cached.get("times", np.empty(0)), dtype=float).reshape(-1)
            if (
                cached_times.shape == time_grid.shape
                and np.allclose(cached_times, time_grid, rtol=0.0, atol=1.0e-9)
                and _cr3bp_reference_cache_valid(cached.get("target_state"), target)
            ):
                references = np.array(cached.get("references", np.empty((0, 6))), dtype=float)
                stms = np.array(cached.get("stms", np.empty((0, 6, 6))), dtype=float)
                basis_axes = np.array(cached.get("basis_axes", np.empty((0, 3, 3))), dtype=float)
                basis_omega = np.array(cached.get("basis_omega", np.empty((0, 3))), dtype=float)
        if references is None or stms is None or references.shape[0] != time_grid.size or stms.shape[0] != time_grid.size:
            references, stms = _linearized_cr3bp_moon_ric_stm_table(
                target_state=target,
                times=time_grid,
                current_t_s=float(current_t_s),
            )
            basis_axes, basis_omega = _moon_ric_basis_rows(references)
            prediction_cache[cache_key] = {
                "target_state": target.copy(),
                "times": time_grid.copy(),
                "references": references,
                "stms": stms,
                "basis_axes": basis_axes,
                "basis_omega": basis_omega,
            }
        elif (
            basis_axes is None
            or basis_omega is None
            or basis_axes.shape != (time_grid.size, 3, 3)
            or basis_omega.shape != (time_grid.size, 3)
        ):
            basis_axes, basis_omega = _moon_ric_basis_rows(references)
            if isinstance(cached, dict):
                cached["basis_axes"] = basis_axes
                cached["basis_omega"] = basis_omega
        return _linearized_cr3bp_moon_ric_projection_from_stm_table(
            rel0,
            target_state=target,
            references=references,
            stms=stms,
            basis_axes=basis_axes,
            basis_omega=basis_omega,
        )

    def _capped_projection_points_for_zoom(
        self,
        points: np.ndarray,
        *,
        x_axis: int,
        y_axis: int,
        camera_center: np.ndarray,
    ) -> np.ndarray:
        projected = np.array(points, dtype=float).reshape(-1, 6)[:, [int(x_axis), int(y_axis)]]
        center = np.array(camera_center, dtype=float).reshape(3)[[int(x_axis), int(y_axis)]]
        shifted = projected - center.reshape(1, 2)
        shifted = shifted[np.all(np.isfinite(shifted), axis=1)]
        if not shifted.size:
            return np.empty((0, 2), dtype=float)
        cap = _positive_float_or_none(getattr(self, "plot_prediction_zoom_max_span_km", None))
        if cap is None:
            return shifted
        cap_value = float(cap)
        return np.clip(shifted, -cap_value, cap_value)

    def _coast_prediction_from(
        self,
        rel0: np.ndarray,
        *,
        cr3bp_horizon_s: float | None = None,
        cr3bp_dt_s: float | None = None,
        max_draw_points: int | None = None,
    ) -> np.ndarray:
        rel0 = np.array(rel0, dtype=float).reshape(6)
        n = self.mean_motion_rad_s
        if n is None or not np.isfinite(float(n)) or float(n) <= 0.0:
            return np.empty((0, 6), dtype=float)
        horizon = self._coast_prediction_horizon_s(float(n))
        if horizon <= 0.0:
            return np.empty((0, 6), dtype=float)
        if self._uses_cr3bp_prediction_model():
            cr3bp_horizon = _positive_float_or_none(cr3bp_horizon_s)
            if cr3bp_horizon is None:
                cr3bp_horizon = _positive_float_or_none(getattr(self, "cr3bp_coast_prediction_horizon_s", None))
            horizon_mode = _cr3bp_coast_prediction_horizon_mode_key(
                getattr(self, "cr3bp_coast_prediction_horizon_mode", "default")
            )
            if horizon_mode == "time_remaining":
                remaining_horizon = self._mission_time_remaining_s()
                if remaining_horizon is not None:
                    horizon = float(remaining_horizon)
                elif cr3bp_horizon is not None:
                    horizon = float(cr3bp_horizon)
                if cr3bp_horizon is not None:
                    horizon = min(float(horizon), float(cr3bp_horizon))
            elif cr3bp_horizon is not None:
                horizon = float(cr3bp_horizon)
            if horizon <= 0.0:
                return np.empty((0, 6), dtype=float)
            cr3bp_dt = _positive_float_or_none(cr3bp_dt_s)
            if cr3bp_dt is None:
                cr3bp_dt = _positive_float_or_none(getattr(self, "cr3bp_coast_prediction_dt_s", None))
            dt = max(
                float(getattr(self, "coast_prediction_dt_s", 10.0)),
                300.0 if cr3bp_dt is None else float(cr3bp_dt),
                1.0e-6,
            )
            point_cap = max(int(max_draw_points or MAX_GHOST_DRAW_POINTS), 2)
            times = _front_loaded_prediction_times(horizon, dt, max_points=point_cap)
            if _relative_frame_key(getattr(self, "relative_frame", "ric")) == "moon_ric":
                target_state = getattr(self, "reference_state_eci", None)
                if target_state is None:
                    return np.empty((0, 6), dtype=float)
                target_state = np.array(target_state, dtype=float).reshape(6)
                if _cr3bp_projection_mode_key(getattr(self, "cr3bp_projection_mode", "nonlinear")) == "linearized":
                    return self._linearized_cr3bp_moon_ric_coast_prediction_cached(
                        rel0,
                        target_state=target_state,
                        times=times,
                        current_t_s=self._current_time_s(),
                    )
                return _nonlinear_cr3bp_moon_ric_coast_prediction(
                    rel0,
                    target_state=target_state,
                    times=times,
                    current_t_s=self._current_time_s(),
                )
            origin = cr3bp_l1_state_km_s()
            state = origin + rel0
            rows: list[np.ndarray] = []
            current_t = self._current_time_s()
            previous_t = 0.0
            for target_t in times:
                step_s = float(target_t - previous_t)
                if step_s > 0.0:
                    state = propagate_cr3bp_state(state, step_s, current_t)
                    current_t += step_s
                rows.append(state - origin)
                previous_t = float(target_t)
            return np.vstack(rows)
        dt = float(max(self.coast_prediction_dt_s, 1.0e-6))
        point_cap = max(int(max_draw_points or MAX_GHOST_DRAW_POINTS), 2)
        times = _front_loaded_prediction_times(horizon, dt, max_points=point_cap)
        if _coast_prediction_model_key(getattr(self, "coast_prediction_model", "hcw")) in {
            "elliptic_linear",
            "tschauner_hempel",
            "ts",
        }:
            reference_state = getattr(self, "reference_state_eci", None)
            if reference_state is not None:
                reference = np.array(reference_state, dtype=float).reshape(6)
                try:
                    prediction = _elliptic_ya_coast_states(rel0, times, reference)
                    if prediction.shape == (times.size, 6) and np.all(np.isfinite(prediction)):
                        return prediction
                except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                    pass
                return _elliptic_linear_coast_states(rel0, times, reference)
        return _cw_coast_states(rel0, times, float(n))

    def _coast_prediction_horizon_s(self, mean_motion_rad_s: float) -> float:
        fraction = self.coast_prediction_orbit_fraction
        n = float(mean_motion_rad_s)
        if fraction is None:
            return float(max(self.coast_prediction_horizon_s, 0.0))
        if not np.isfinite(n) or n <= 0.0:
            return 0.0
        return float(max(fraction, 0.0) * (2.0 * np.pi / n))

    def _nmt_points(self) -> np.ndarray:
        return self._nmt_points_for(
            radial_amplitude_km=getattr(self, "goal_nmt_radial_amplitude_km", None),
            cross_track_amplitude_km=getattr(self, "goal_nmt_cross_track_amplitude_km", 0.0),
        )

    def _nmt_boundary_points(self) -> tuple[np.ndarray, ...]:
        a_r = _positive_float_or_none(getattr(self, "goal_nmt_radial_amplitude_km", None))
        if a_r is None:
            return ()
        tol = _positive_float_or_none(getattr(self, "goal_nmt_element_tolerance_km", None))
        if tol is None:
            return ()
        a_c_raw = getattr(self, "goal_nmt_cross_track_amplitude_km", 0.0)
        try:
            a_c = float(a_c_raw)
        except (TypeError, ValueError):
            a_c = 0.0
        if not np.isfinite(a_c):
            a_c = 0.0
        lower_r = max(float(a_r) - float(tol), 0.0)
        upper_r = float(a_r) + float(tol)
        lower_c = max(abs(a_c) - float(tol), 0.0)
        upper_c = abs(a_c) + float(tol)
        curves: list[np.ndarray] = []
        lower = self._nmt_points_for(radial_amplitude_km=lower_r, cross_track_amplitude_km=lower_c)
        upper = self._nmt_points_for(radial_amplitude_km=upper_r, cross_track_amplitude_km=upper_c)
        if lower.size:
            curves.append(lower)
        if upper.size:
            curves.append(upper)
        return tuple(curves)

    def _nmt_points_for(
        self,
        *,
        radial_amplitude_km: float | None,
        cross_track_amplitude_km: float,
    ) -> np.ndarray:
        if radial_amplitude_km is None:
            return np.empty((0, 3), dtype=float)
        a_r = float(radial_amplitude_km)
        if not np.isfinite(a_r) or a_r <= 0.0:
            return np.empty((0, 3), dtype=float)
        center = np.array(self.goal_nmt_center_ric_km, dtype=float).reshape(-1)
        if center.size != 3:
            center = np.zeros(3, dtype=float)
        a_c = float(cross_track_amplitude_km)
        if not np.isfinite(a_c):
            a_c = 0.0
        phase = np.deg2rad(float(self.goal_nmt_cross_track_phase_deg))
        theta = np.linspace(0.0, 2.0 * np.pi, 181)
        pts = np.zeros((theta.size, 3), dtype=float)
        pts[:, 0] = center[0] + a_r * np.cos(theta)
        pts[:, 1] = center[1] - 2.0 * a_r * np.sin(theta)
        pts[:, 2] = center[2] + a_c * np.cos(theta + phase)
        return pts

    def _scale_for_plot(self, *, pts: list[np.ndarray], min_span_km: float | None = None) -> float:
        finite: list[np.ndarray] = []
        for arr in pts:
            a = np.array(arr, dtype=float).reshape(-1, 2)
            a = a[np.all(np.isfinite(a), axis=1)]
            if a.size:
                finite.append(a)
        span = self._minimum_plot_span_km() if min_span_km is None else float(max(min_span_km, MIN_PLOT_SPAN_KM))
        if finite:
            all_pts = np.vstack(finite)
            span = max(float(np.max(np.abs(all_pts))) * 1.2, span)
        width, height = self.screen.get_size()
        px_span = max(min(width, height) * 0.28, 80.0)
        return float(px_span / max(span, 1e-9))

    def _axis_scales_for_plot(
        self,
        plot: Any,
        *,
        pts: list[np.ndarray],
        min_span_km: float | None,
        x_axis: int,
        y_axis: int,
    ) -> tuple[float, float]:
        axis_scale_x, axis_scale_y = self._axis_scale_for_plane(x_axis=x_axis, y_axis=y_axis)
        fixed_x, fixed_y = self._fixed_axis_half_span_for_plane(x_axis=x_axis, y_axis=y_axis)
        if fixed_x is None and fixed_y is None:
            scale = self._scale_for_plot(pts=pts, min_span_km=min_span_km)
            scale_x = scale * axis_scale_x
            scale_y = scale * axis_scale_y
        else:
            scale_x = self._scale_for_axis(pts=pts, axis_index=0, min_span_km=min_span_km, plot_px=plot.width)
            scale_y = self._scale_for_axis(pts=pts, axis_index=1, min_span_km=min_span_km, plot_px=plot.height)
            scale_x *= axis_scale_x
            scale_y *= axis_scale_y
            if fixed_x is not None:
                scale_x = self._scale_for_fixed_half_span(plot_px=plot.width, half_span_km=fixed_x)
            if fixed_y is not None:
                scale_y = self._scale_for_fixed_half_span(plot_px=plot.height, half_span_km=fixed_y)
        if self._equal_axis_scale_for_plane(x_axis=x_axis, y_axis=y_axis):
            scale_x = scale_y
        return (scale_x, scale_y)

    def _scale_for_axis(
        self,
        *,
        pts: list[np.ndarray],
        axis_index: int,
        min_span_km: float | None,
        plot_px: float,
    ) -> float:
        finite: list[np.ndarray] = []
        for arr in pts:
            a = np.array(arr, dtype=float).reshape(-1, 2)
            a = a[np.all(np.isfinite(a), axis=1)]
            if a.size:
                finite.append(a[:, int(axis_index)])
        span = float(max(MIN_PLOT_SPAN_KM if min_span_km is None else min_span_km, MIN_PLOT_SPAN_KM))
        if finite:
            values = np.concatenate(finite)
            span = max(float(np.max(np.abs(values))) * 1.2, span)
        px_span = max(float(plot_px) * 0.42, 80.0)
        return float(px_span / max(span, 1e-9))

    @staticmethod
    def _scale_for_fixed_half_span(*, plot_px: float, half_span_km: float) -> float:
        span = float(half_span_km)
        if not np.isfinite(span) or span <= 0.0:
            span = MIN_PLOT_SPAN_KM
        return float(max(float(plot_px) * 0.5, 1.0) / span)

    def _axis_scale_for_plane(self, *, x_axis: int, y_axis: int) -> tuple[float, float]:
        key = _plane_key_for_axes(x_axis=x_axis, y_axis=y_axis)
        raw_scales = getattr(self, "plot_axis_scale", {}) or {}
        raw = raw_scales.get(key, (1.0, 1.0))
        try:
            x_scale, y_scale = raw
        except (TypeError, ValueError):
            return (1.0, 1.0)
        x = float(x_scale)
        y = float(y_scale)
        if not np.isfinite(x) or x <= 0.0:
            x = 1.0
        if not np.isfinite(y) or y <= 0.0:
            y = 1.0
        return (x, y)

    def _fixed_axis_half_span_for_plane(self, *, x_axis: int, y_axis: int) -> tuple[float | None, float | None]:
        key = _plane_key_for_axes(x_axis=x_axis, y_axis=y_axis)
        raw_spans = getattr(self, "plot_fixed_axis_half_span_km", {}) or {}
        raw = raw_spans.get(key, (None, None))
        try:
            x_span, y_span = raw
        except (TypeError, ValueError):
            return (None, None)
        return (_positive_float_or_none(x_span), _positive_float_or_none(y_span))

    def _equal_axis_scale_for_plane(self, *, x_axis: int, y_axis: int) -> bool:
        key = _plane_key_for_axes(x_axis=x_axis, y_axis=y_axis)
        planes = tuple(str(plane or "").strip().upper() for plane in getattr(self, "plot_equal_axis_scale_planes", ()))
        return key in planes

    def _show_proximity_rings_for_plane(self, *, x_axis: int, y_axis: int) -> bool:
        key = _plane_key_for_axes(x_axis=x_axis, y_axis=y_axis)
        planes = tuple(str(plane or "").strip().upper() for plane in getattr(self, "proximity_ring_plot_planes", ()))
        return key in planes

    def _camera_rule_mode_key(self) -> str:
        value = str(getattr(self, "camera_rule_mode", "default") or "default").strip().lower()
        if value in {"full", "full_trajectory", "trajectory", "trail", "trail_projection"}:
            return "full_trajectory"
        if value in {"current_pair", "pair", "satellites", "satellites_only", "current"}:
            return "current_pair"
        return "default"

    def _prediction_scales_current_camera(self) -> bool:
        if not bool(getattr(self, "plot_prediction_full_trajectory_only", False)):
            return True
        mode = str(getattr(self, "camera_mode", "reference") or "reference").strip().lower()
        if mode in {"rule_toggle_pair", "camera_rule_pair", "toggle_pair"}:
            return self._camera_rule_mode_key() == "full_trajectory"
        return True

    def toggle_camera_rule_mode(self) -> str:
        next_mode = "current_pair" if self._camera_rule_mode_key() == "full_trajectory" else "full_trajectory"
        self.camera_rule_mode = next_mode
        if hasattr(self, "_frame_cache_dirty"):
            self._frame_cache_dirty = True
        return next_mode

    def _full_trajectory_scale_points(
        self,
        *,
        rel: np.ndarray,
        target_rel: np.ndarray,
        ghost: np.ndarray,
        target_ghost: np.ndarray,
        x_axis: int,
        y_axis: int,
        camera_center: np.ndarray,
    ) -> list[np.ndarray]:
        points: list[np.ndarray] = []
        center = np.array(camera_center, dtype=float).reshape(1, 3)
        rel_arr = np.array(rel, dtype=float)
        if rel_arr.ndim == 2 and rel_arr.shape[1] >= 3:
            points.append((rel_arr[:, :3] - center)[:, [int(x_axis), int(y_axis)]])
        target_arr = np.array(target_rel, dtype=float)
        if target_arr.ndim == 2 and target_arr.shape[1] >= 3:
            points.append((target_arr[:, :3] - center)[:, [int(x_axis), int(y_axis)]])
        ghost_arr = np.array(ghost, dtype=float)
        if ghost_arr.ndim == 2 and ghost_arr.shape[1] >= 3:
            points.append((ghost_arr[:, :3] - center)[:, [int(x_axis), int(y_axis)]])
        target_ghost_arr = np.array(target_ghost, dtype=float)
        if target_ghost_arr.ndim == 2 and target_ghost_arr.shape[1] >= 3:
            points.append((target_ghost_arr[:, :3] - center)[:, [int(x_axis), int(y_axis)]])
        return points

    def _camera_rule_scale_points(
        self,
        *,
        rel: np.ndarray,
        target_rel: np.ndarray,
        ghost: np.ndarray,
        target_ghost: np.ndarray | None = None,
        x_axis: int,
        y_axis: int,
        camera_center: np.ndarray,
    ) -> list[np.ndarray]:
        mode = self._camera_rule_mode_key()
        if mode == "full_trajectory":
            return self._full_trajectory_scale_points(
                rel=rel,
                target_rel=target_rel,
                ghost=ghost,
                target_ghost=np.empty((0, 6), dtype=float) if target_ghost is None else target_ghost,
                x_axis=x_axis,
                y_axis=y_axis,
                camera_center=camera_center,
            )
        if bool(getattr(self, "plot_prediction_in_zoom", False)) and np.array(ghost, dtype=float).size:
            capped = self._capped_projection_points_for_zoom(
                ghost,
                x_axis=x_axis,
                y_axis=y_axis,
                camera_center=camera_center,
            )
            if capped.size:
                return [capped]
        if mode == "current_pair":
            return []
        return []

    def _camera_center_ric(
        self,
        *,
        chaser_current: np.ndarray,
        target_current: np.ndarray,
        x_axis: int | None = None,
        y_axis: int | None = None,
    ) -> np.ndarray:
        mode = str(getattr(self, "camera_mode", "reference") or "reference").strip().lower()
        target = np.array(target_current, dtype=float).reshape(-1)
        chaser = np.array(chaser_current, dtype=float).reshape(-1)
        if mode in {"rule_toggle_pair", "camera_rule_pair", "toggle_pair"}:
            if self._camera_rule_mode_key() == "full_trajectory":
                return np.zeros(3, dtype=float)
            plane = (
                _plane_key_for_axes(x_axis=int(x_axis), y_axis=int(y_axis))
                if x_axis is not None and y_axis is not None
                else ""
            )
            if plane in tuple(str(value or "").strip().upper() for value in getattr(self, "target_centered_plot_planes", ())):
                if target.size >= 3 and np.all(np.isfinite(target[:3])):
                    return target[:3].astype(float)
                return np.zeros(3, dtype=float)
            return _satellite_pair_camera_center(
                chaser=chaser,
                target=target,
                x_axis=x_axis,
                y_axis=y_axis,
                keep_rc_reference_centered=False,
            )
        if mode not in {"target_pair", "satellite_pair", "pair"}:
            return np.zeros(3, dtype=float)
        plane = _plane_key_for_axes(x_axis=int(x_axis), y_axis=int(y_axis)) if x_axis is not None and y_axis is not None else ""
        if plane in tuple(str(value or "").strip().upper() for value in getattr(self, "target_centered_plot_planes", ())):
            if target.size >= 3 and np.all(np.isfinite(target[:3])):
                return target[:3].astype(float)
            return np.zeros(3, dtype=float)
        if {x_axis, y_axis} == {0, 2}:
            return np.zeros(3, dtype=float)
        center = _satellite_pair_camera_center(
            chaser=chaser,
            target=target,
            x_axis=x_axis,
            y_axis=y_axis,
            keep_rc_reference_centered=True,
        )
        axis_overrides = getattr(self, "target_centered_plot_axes", {}) or {}
        raw_override_axes = axis_overrides.get(plane, ())
        override_axes = tuple(str(value or "").strip().lower() for value in raw_override_axes)
        if "x" in override_axes and x_axis is not None:
            center[int(x_axis)] = float(target[int(x_axis)])
        if "y" in override_axes and y_axis is not None:
            center[int(y_axis)] = float(target[int(y_axis)])
        return center

    def _minimum_plot_span_km(
        self,
        *,
        x_axis: int | None = None,
        y_axis: int | None = None,
        offset: np.ndarray | None = None,
    ) -> float:
        span = MIN_PLOT_SPAN_KM
        if x_axis is None or y_axis is None:
            return span
        plane = _plane_key_for_axes(x_axis=int(x_axis), y_axis=int(y_axis))
        overlay_zoom_by_plane = getattr(self, "plot_overlays_in_zoom_by_plane", {}) or {}
        include_overlays = bool(overlay_zoom_by_plane.get(plane, getattr(self, "plot_overlays_in_zoom", True)))
        if not include_overlays:
            return span
        origin_offset = np.zeros(3, dtype=float) if offset is None else np.array(offset, dtype=float).reshape(3)
        overlay_points = [
            *self._forbidden_region_projection_points(x_axis=int(x_axis), y_axis=int(y_axis), offset=origin_offset),
            *self._approach_gate_projection_points(x_axis=int(x_axis), y_axis=int(y_axis), offset=origin_offset),
            *self._inspection_gate_projection_points(x_axis=int(x_axis), y_axis=int(y_axis), offset=origin_offset),
        ]
        frame_cache = getattr(self, "_frame_cache", {})
        nmt = frame_cache.get("nmt")
        if nmt is None:
            nmt = self._nmt_points()
        if nmt.size:
            projected_nmt = nmt[:, [int(x_axis), int(y_axis)]] + origin_offset[
                [int(x_axis), int(y_axis)]
            ].reshape(1, 2)
            overlay_points.append(projected_nmt)
        nmt_boundaries = frame_cache.get("nmt_bounds")
        if nmt_boundaries is None:
            nmt_boundaries = self._nmt_boundary_points()
        for nmt_boundary in nmt_boundaries:
            if nmt_boundary.size:
                projected_nmt = nmt_boundary[:, [int(x_axis), int(y_axis)]] + origin_offset[
                    [int(x_axis), int(y_axis)]
                ].reshape(1, 2)
                overlay_points.append(projected_nmt)
        finite: list[np.ndarray] = []
        for points in overlay_points:
            projected = np.array(points, dtype=float).reshape(-1, 2)
            projected = projected[np.all(np.isfinite(projected), axis=1)]
            if projected.size:
                finite.append(projected)
        if finite:
            span = max(span, float(np.max(np.abs(np.vstack(finite)))) * PLOT_OVERLAY_MARGIN)
        return span

    @staticmethod
    def _nice_step(value: float) -> float:
        if value <= 0.0 or not np.isfinite(value):
            return 1.0
        exp = np.floor(np.log10(value))
        base = value / (10.0**exp)
        if base <= 1.0:
            nice = 1.0
        elif base <= 2.0:
            nice = 2.0
        elif base <= 5.0:
            nice = 5.0
        else:
            nice = 10.0
        return float(nice * (10.0**exp))

    def _text(self, text: str, pos: tuple[int, int], font: Any, color: tuple[int, int, int]) -> None:
        if not text:
            return
        key = (id(font), str(text), tuple(color))
        surf = self._text_cache.get(key)
        if surf is None:
            if len(self._text_cache) >= TEXT_CACHE_LIMIT:
                self._text_cache.clear()
            surf = font.render(str(text), True, color)
            self._text_cache[key] = surf
        self.screen.blit(surf, pos)

    def _wrap_text_px(self, value: str, font: Any, width_px: int) -> list[str]:
        words = str(value or "").split()
        if not words:
            return [""]
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = word if not current else current + " " + word
            if self._text_width(font, candidate) <= width_px:
                current = candidate
                continue
            if current:
                lines.append(current)
            current = self._fit_text_px(word, font, width_px) if self._text_width(font, word) > width_px else word
        if current:
            lines.append(current)
        return lines or [""]

    def _fit_text_px(self, value: str, font: Any, width_px: int, *, preserve_spaces: bool = False) -> str:
        text = str(value or "") if preserve_spaces else " ".join(str(value or "").split())
        if self._text_width(font, text) <= width_px:
            return text
        ellipsis = "..."
        if self._text_width(font, ellipsis) > width_px:
            return ""
        lo = 0
        hi = len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            candidate = text[:mid].rstrip() + ellipsis
            if self._text_width(font, candidate) <= width_px:
                lo = mid
            else:
                hi = mid - 1
        return text[:lo].rstrip() + ellipsis

    @staticmethod
    def _text_width(font: Any, text: str) -> int:
        if hasattr(font, "size"):
            return int(font.size(str(text))[0])
        surf = font.render(str(text), True, (255, 255, 255))
        if hasattr(surf, "get_width"):
            return int(surf.get_width())
        return len(str(text)) * 8

    @staticmethod
    def _wrap_text(value: str, max_chars: int) -> list[str]:
        words = str(value or "").split()
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = word if not current else current + " " + word
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines or [""]


def _finite_projected_region_bounds(
    region: ForbiddenRegionConfig, *, x_axis: int, y_axis: int
) -> tuple[np.ndarray, np.ndarray] | None:
    lower = np.array(region.min_ric_km, dtype=float).reshape(3)
    upper = np.array(region.max_ric_km, dtype=float).reshape(3)
    lo = np.array([lower[x_axis], lower[y_axis]], dtype=float)
    hi = np.array([upper[x_axis], upper[y_axis]], dtype=float)
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        return None
    return lo, hi


def _sample_rows(values: np.ndarray, max_points: int) -> np.ndarray:
    arr = np.array(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    limit = max(int(max_points), 2)
    if arr.shape[0] <= limit:
        return arr
    idx = np.linspace(0, arr.shape[0] - 1, limit, dtype=int)
    idx = np.unique(idx)
    return arr[idx]


def _cylinder_projection_polygon_ric(
    region: ForbiddenRegionConfig, *, x_axis: int, y_axis: int, samples: int = 72
) -> np.ndarray:
    if region.radius_km is None or region.height_km is None:
        return np.zeros((0, 3), dtype=float)
    axis = _region_axis_index(region.axis)
    center = np.array(region.center_ric_km, dtype=float).reshape(3)
    radius = float(region.radius_km)
    half_height = float(region.height_km) / 2.0
    if _cylinder_projection_is_cross_section(region, x_axis=x_axis, y_axis=y_axis):
        theta = np.linspace(0.0, 2.0 * np.pi, max(int(samples), 12), endpoint=True)
        pts = np.tile(center.reshape(1, 3), (theta.size, 1))
        cross_x, cross_y = int(x_axis), int(y_axis)
        pts[:, cross_x] += radius * np.cos(theta)
        pts[:, cross_y] += radius * np.sin(theta)
        return pts
    projected_axes = {int(x_axis), int(y_axis)}
    if axis not in projected_axes:
        return np.zeros((0, 3), dtype=float)
    cross_axis = int(y_axis) if int(x_axis) == axis else int(x_axis)
    corners = np.tile(center.reshape(1, 3), (4, 1))
    corners[:, axis] += np.array([-half_height, half_height, half_height, -half_height], dtype=float)
    corners[:, cross_axis] += np.array([-radius, -radius, radius, radius], dtype=float)
    return corners


def _cylinder_projection_is_cross_section(region: ForbiddenRegionConfig, *, x_axis: int, y_axis: int) -> bool:
    axis = _region_axis_index(region.axis)
    projected_axes = {int(x_axis), int(y_axis)}
    cross_axes = {idx for idx in (0, 1, 2) if idx != axis}
    return projected_axes == cross_axes


def _sphere_projection_polygon_ric(
    region: ForbiddenRegionConfig, *, x_axis: int, y_axis: int, samples: int = 72
) -> np.ndarray:
    if region.radius_km is None:
        return np.zeros((0, 3), dtype=float)
    center = np.array(region.center_ric_km, dtype=float).reshape(3)
    radius = float(region.radius_km)
    theta = np.linspace(0.0, 2.0 * np.pi, max(int(samples), 12), endpoint=True)
    pts = np.tile(center.reshape(1, 3), (theta.size, 1))
    pts[:, int(x_axis)] += radius * np.cos(theta)
    pts[:, int(y_axis)] += radius * np.sin(theta)
    return pts


def _sun_angle_sector_polygon_ric(
    constraint: SunAngleConstraintConfig,
    *,
    x_axis: int,
    y_axis: int,
    samples: int = 72,
    target_state_eci: np.ndarray | None = None,
    time_s: float | None = None,
) -> np.ndarray:
    if not _constraint_visible_on_plane(constraint, x_axis=x_axis, y_axis=y_axis):
        return np.zeros((0, 3), dtype=float)
    center = constraint.allowed_center_at_ric(target_state_eci=target_state_eci, time_s=time_s)
    projected = center[[int(x_axis), int(y_axis)]]
    norm = float(np.linalg.norm(projected))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        return np.zeros((0, 3), dtype=float)
    theta_center = float(np.arctan2(projected[1], projected[0]))
    half = np.deg2rad(float(constraint.allowed_half_angle_deg))
    outer = constraint.beam_radius_km
    if outer is None:
        outer = constraint.max_range_km
    if outer is None:
        outer = 8.0
    inner = 0.0 if constraint.min_range_km is None else max(float(constraint.min_range_km), 0.0)
    outer = max(float(outer), inner + 1.0e-6)
    angles = np.linspace(theta_center - half, theta_center + half, max(int(samples), 12))
    pts: list[np.ndarray] = []
    for radius, seq in ((outer, angles), (inner, angles[::-1])):
        for theta in seq:
            row = np.zeros(3, dtype=float)
            row[int(x_axis)] = radius * float(np.cos(theta))
            row[int(y_axis)] = radius * float(np.sin(theta))
            pts.append(row)
    return np.vstack(pts) if pts else np.zeros((0, 3), dtype=float)


def _sun_angle_centerline_points_ric(
    constraint: SunAngleConstraintConfig,
    *,
    x_axis: int,
    y_axis: int,
    target_state_eci: np.ndarray | None = None,
    time_s: float | None = None,
) -> np.ndarray:
    if not _constraint_visible_on_plane(constraint, x_axis=x_axis, y_axis=y_axis):
        return np.zeros((0, 3), dtype=float)
    center = constraint.allowed_center_at_ric(target_state_eci=target_state_eci, time_s=time_s)
    projected = center[[int(x_axis), int(y_axis)]]
    norm = float(np.linalg.norm(projected))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        return np.zeros((0, 3), dtype=float)
    direction = projected / norm
    outer = constraint.beam_radius_km
    if outer is None:
        outer = constraint.max_range_km
    if outer is None:
        outer = 8.0
    inner = 0.0 if constraint.min_range_km is None else max(float(constraint.min_range_km), 0.0)
    rows = np.zeros((2, 3), dtype=float)
    rows[0, int(x_axis)] = inner * float(direction[0])
    rows[0, int(y_axis)] = inner * float(direction[1])
    rows[1, int(x_axis)] = float(outer) * float(direction[0])
    rows[1, int(y_axis)] = float(outer) * float(direction[1])
    return rows


def _constraint_visible_on_plane(constraint: SunAngleConstraintConfig, *, x_axis: int, y_axis: int) -> bool:
    planes = tuple(constraint.plot_planes or ())
    if not planes:
        return True
    plane = _plane_key_for_axes(x_axis=x_axis, y_axis=y_axis)
    if not plane:
        return False
    return plane in planes


def _region_axis_index(axis: str) -> int:
    key = str(axis or "").strip().upper()
    if key == "R":
        return 0
    if key == "I":
        return 1
    if key == "C":
        return 2
    return 1


def _positive_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result) or result <= 0.0:
        return None
    return result


def _plane_key_for_axes(*, x_axis: int, y_axis: int) -> str:
    axis_set = {int(x_axis), int(y_axis)}
    if axis_set == {0, 1}:
        return "RI"
    if axis_set == {0, 2}:
        return "RC"
    if axis_set == {1, 2}:
        return "IC"
    return ""


def _region_visible_on_plane(region: ForbiddenRegionConfig, *, x_axis: int, y_axis: int) -> bool:
    planes = tuple(region.plot_planes or ())
    if not planes:
        return True
    plane = _plane_key_for_axes(x_axis=x_axis, y_axis=y_axis)
    if not plane:
        return False
    return plane in planes


def _project_eci_positions_to_plane(positions_eci_km: np.ndarray, *, x_hat: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
    positions = np.array(positions_eci_km, dtype=float)
    if positions.size == 0:
        return np.empty((0, 2), dtype=float)
    positions = positions.reshape(-1, positions.shape[-1])
    if positions.shape[1] < 3:
        return np.empty((0, 2), dtype=float)
    x_axis = np.array(x_hat, dtype=float).reshape(3)
    y_axis = np.array(y_hat, dtype=float).reshape(3)
    return np.column_stack((positions[:, :3] @ x_axis, positions[:, :3] @ y_axis)).astype(float)


def _project_moon_rotating_yz_to_plane(moon_centered_positions_km: np.ndarray) -> np.ndarray:
    positions = np.array(moon_centered_positions_km, dtype=float)
    if positions.size == 0:
        return np.empty((0, 2), dtype=float)
    positions = positions.reshape(-1, positions.shape[-1])
    if positions.shape[1] < 3:
        return np.empty((0, 2), dtype=float)
    return positions[:, [1, 2]].astype(float)


def _approach_gate_projected_bounds(
    gate: ApproachGateConfig, *, x_axis: int, y_axis: int
) -> tuple[np.ndarray, np.ndarray] | None:
    axes = {int(x_axis), int(y_axis)}
    if 0 not in axes:
        return None
    lateral_axis = int(x_axis) if int(y_axis) == 0 else int(y_axis)
    if lateral_axis == 1:
        lateral = gate.max_abs_intrack_km
    elif lateral_axis == 2:
        lateral = gate.max_abs_cross_track_km
    else:
        return None
    if lateral is None:
        return None
    lower = np.zeros(3, dtype=float)
    upper = np.zeros(3, dtype=float)
    lower[0] = float(gate.radial_ric_km) - float(gate.radial_tolerance_km)
    upper[0] = float(gate.radial_ric_km) + float(gate.radial_tolerance_km)
    lower[lateral_axis] = -float(lateral)
    upper[lateral_axis] = float(lateral)
    lo = np.array([lower[x_axis], lower[y_axis]], dtype=float)
    hi = np.array([upper[x_axis], upper[y_axis]], dtype=float)
    return lo, hi


def _cw_coast_state(x0: np.ndarray, t_s: float, mean_motion_rad_s: float) -> np.ndarray:
    x, y, z, xd, yd, zd = np.array(x0, dtype=float).reshape(6)
    n = float(mean_motion_rad_s)
    t = float(t_s)
    nt = n * t
    c = float(np.cos(nt))
    s = float(np.sin(nt))
    if abs(n) <= 1.0e-12:
        return np.array([x + xd * t, y + yd * t, z + zd * t, xd, yd, zd], dtype=float)

    xp = (4.0 - 3.0 * c) * x + (s / n) * xd + (2.0 * (1.0 - c) / n) * yd
    yp = 6.0 * (s - nt) * x + y - (2.0 * (1.0 - c) / n) * xd + ((4.0 * s - 3.0 * nt) / n) * yd
    zp = c * z + (s / n) * zd
    xdp = 3.0 * n * s * x + c * xd + 2.0 * s * yd
    ydp = -6.0 * n * (1.0 - c) * x - 2.0 * s * xd + (4.0 * c - 3.0) * yd
    zdp = -n * s * z + c * zd
    return np.array([xp, yp, zp, xdp, ydp, zdp], dtype=float)


def _cw_coast_states(x0: np.ndarray, times_s: np.ndarray, mean_motion_rad_s: float) -> np.ndarray:
    x, y, z, xd, yd, zd = np.array(x0, dtype=float).reshape(6)
    times = np.array(times_s, dtype=float).reshape(-1)
    if times.size == 0:
        return np.empty((0, 6), dtype=float)
    n = float(mean_motion_rad_s)
    if abs(n) <= 1.0e-12:
        out = np.empty((times.size, 6), dtype=float)
        out[:, 0] = x + xd * times
        out[:, 1] = y + yd * times
        out[:, 2] = z + zd * times
        out[:, 3] = xd
        out[:, 4] = yd
        out[:, 5] = zd
        return out
    nt = n * times
    c = np.cos(nt)
    s = np.sin(nt)
    one_minus_c = 1.0 - c
    out = np.empty((times.size, 6), dtype=float)
    out[:, 0] = (4.0 - 3.0 * c) * x + (s / n) * xd + (2.0 * one_minus_c / n) * yd
    out[:, 1] = 6.0 * (s - nt) * x + y - (2.0 * one_minus_c / n) * xd + ((4.0 * s - 3.0 * nt) / n) * yd
    out[:, 2] = c * z + (s / n) * zd
    out[:, 3] = 3.0 * n * s * x + c * xd + 2.0 * s * yd
    out[:, 4] = -6.0 * n * one_minus_c * x - 2.0 * s * xd + (4.0 * c - 3.0) * yd
    out[:, 5] = -n * s * z + c * zd
    return out


def _satellite_marker_size_px(
    scale_x_px_per_km: float,
    scale_y_px_per_km: float,
    *,
    diameter_km: float = SATELLITE_SPRITE_DIAMETER_KM,
) -> int:
    raw_px = float(max(abs(float(scale_x_px_per_km)), abs(float(scale_y_px_per_km)))) * float(max(diameter_km, 0.0))
    if not np.isfinite(raw_px) or raw_px <= 0.0:
        return 0
    return max(int(round(raw_px)), 1)


def _satellite_marker_reticle_radii_px(sprite_size_px: int) -> tuple[int, int]:
    size = int(max(sprite_size_px, 0))
    if size <= 0:
        return 0, 0
    if size < 24:
        return 2, 4
    dot_radius = max(2, min(3, int(round(size * 0.08))))
    ring_radius = max(dot_radius + 2, min(6, int(round(size * 0.18))))
    return dot_radius, ring_radius


def _cw_forced_state(
    x0: np.ndarray,
    accel_ric_km_s2: np.ndarray,
    t_s: float,
    mean_motion_rad_s: float,
    *,
    substep_s: float = 0.1,
) -> np.ndarray:
    state = np.array(x0, dtype=float).reshape(6).copy()
    accel = np.array(accel_ric_km_s2, dtype=float).reshape(3)
    duration = float(max(t_s, 0.0))
    n = float(mean_motion_rad_s)
    if duration <= 0.0 or not np.all(np.isfinite(state)) or not np.all(np.isfinite(accel)):
        return state
    if not np.isfinite(n) or n <= 0.0:
        state[:3] += state[3:6] * duration + 0.5 * accel * duration * duration
        state[3:6] += accel * duration
        return state
    step = float(max(substep_s, 1.0e-6))
    elapsed = 0.0
    while elapsed < duration - 1.0e-12:
        dt = min(step, duration - elapsed)
        r, i, c, rd, idot, cd = state
        rdd = 3.0 * n * n * r + 2.0 * n * idot + accel[0]
        idd = -2.0 * n * rd + accel[1]
        cdd = -n * n * c + accel[2]
        state[3] = rd + rdd * dt
        state[4] = idot + idd * dt
        state[5] = cd + cdd * dt
        state[0] = r + state[3] * dt
        state[1] = i + state[4] * dt
        state[2] = c + state[5] * dt
        elapsed += dt
    return state


def _coast_prediction_model_key(value: str) -> str:
    key = str(value or "hcw").strip().lower().replace("-", "_")
    aliases = {
        "cw": "hcw",
        "cislunar": "cislunar",
        "cislunar_l1": "cislunar_l1",
        "cr3bp": "cr3bp",
        "cr3bp_rotating": "cr3bp",
        "tschauner_hempel": "tschauner_hempel",
        "th": "tschauner_hempel",
        "ts": "ts",
        "elliptic": "elliptic_linear",
        "elliptical": "elliptic_linear",
        "elliptic_linear": "elliptic_linear",
    }
    return aliases.get(key, key or "hcw")


def _cr3bp_projection_mode_key(value: str) -> str:
    key = str(value or "nonlinear").strip().lower().replace("-", "_")
    if key in {"linear", "linearized", "stm", "variational"}:
        return "linearized"
    return "nonlinear"


def _cr3bp_coast_prediction_horizon_mode_key(value: str) -> str:
    key = str(value or "default").strip().lower().replace("-", "_")
    if key in {"time_remaining", "remaining_time", "mission_remaining", "mission_time_remaining"}:
        return "time_remaining"
    return "default"


def _relative_frame_key(value: str) -> str:
    key = str(value or "ric").strip().lower().replace("-", "_")
    if key in {"cislunar", "cislunar_l1", "earth_moon_rotating", "cr3bp", "cr3bp_rotating"}:
        return "cislunar_l1"
    if key in {"moon_ric", "lunar_ric", "target_moon_ric", "target_lunar_ric"}:
        return "moon_ric"
    return "ric"


def _cr3bp_state_to_moon_ric_rect(deputy_state: np.ndarray, chief_state: np.ndarray) -> np.ndarray:
    moon = cr3bp_moon_state_km_s()
    deputy = np.array(deputy_state, dtype=float).reshape(6) - moon
    chief = np.array(chief_state, dtype=float).reshape(6) - moon
    return eci_relative_to_ric_rect(deputy, chief)


def _moon_ric_basis_rows(chief_states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    moon = cr3bp_moon_state_km_s()
    chief_rows = np.array(chief_states, dtype=float).reshape(-1, 6) - moon
    r = chief_rows[:, :3]
    v = chief_rows[:, 3:]
    r_norm = np.maximum(np.linalg.norm(r, axis=1), 1.0e-12)
    r_hat = r / r_norm[:, None]
    h = np.cross(r, v)
    h_norm = np.maximum(np.linalg.norm(h, axis=1), 1.0e-12)
    c_hat = h / h_norm[:, None]
    i_hat = np.cross(c_hat, r_hat)
    i_norm = np.maximum(np.linalg.norm(i_hat, axis=1), 1.0e-12)
    i_hat = i_hat / i_norm[:, None]
    axes = np.stack((r_hat, i_hat, c_hat), axis=2)
    omega = h / np.maximum(np.sum(r * r, axis=1), 1.0e-12)[:, None]
    return axes, omega


def _cr3bp_states_to_moon_ric_rect_rows(
    deputy_states: np.ndarray,
    chief_states: np.ndarray,
    *,
    basis_axes: np.ndarray | None = None,
    basis_omega: np.ndarray | None = None,
) -> np.ndarray:
    deputy_rows = np.array(deputy_states, dtype=float).reshape(-1, 6)
    chief_rows = np.array(chief_states, dtype=float).reshape(-1, 6)
    if deputy_rows.shape[0] != chief_rows.shape[0]:
        raise ValueError("deputy_states and chief_states must have matching row counts")
    if deputy_rows.size == 0:
        return np.empty((0, 6), dtype=float)
    axes = None if basis_axes is None else np.array(basis_axes, dtype=float).reshape(-1, 3, 3)
    omega = None if basis_omega is None else np.array(basis_omega, dtype=float).reshape(-1, 3)
    if axes is None or omega is None or axes.shape[0] != chief_rows.shape[0] or omega.shape[0] != chief_rows.shape[0]:
        axes, omega = _moon_ric_basis_rows(chief_rows)
    dr_eci = deputy_rows[:, :3] - chief_rows[:, :3]
    dv_eci = deputy_rows[:, 3:] - chief_rows[:, 3:]
    omega_cross_dr = np.cross(omega, dr_eci)
    dr_ric = np.einsum("nji,nj->ni", axes, dr_eci)
    dv_ric = np.einsum("nji,nj->ni", axes, dv_eci - omega_cross_dr)
    return np.hstack((dr_ric, dv_ric))


def _moon_ric_rect_state_to_cr3bp(rel_moon_ric: np.ndarray, chief_state: np.ndarray) -> np.ndarray:
    moon = cr3bp_moon_state_km_s()
    chief_abs = np.array(chief_state, dtype=float).reshape(6)
    chief_moon = chief_abs - moon
    deputy_moon = ric_rect_state_to_eci(
        np.array(rel_moon_ric, dtype=float).reshape(6),
        chief_moon[:3],
        chief_moon[3:],
    )
    return deputy_moon + moon


def _nonlinear_cr3bp_moon_ric_coast_prediction(
    rel0: np.ndarray,
    *,
    target_state: np.ndarray,
    times: np.ndarray,
    current_t_s: float,
) -> np.ndarray:
    reference = np.array(target_state, dtype=float).reshape(6)
    deputy = _moon_ric_rect_state_to_cr3bp(rel0, reference)
    rows: list[np.ndarray] = []
    current_t = float(current_t_s)
    previous_t = 0.0
    for target_t in np.array(times, dtype=float).reshape(-1):
        step_s = float(target_t - previous_t)
        if step_s > 0.0:
            deputy = propagate_cr3bp_state(deputy, step_s, current_t)
            reference = propagate_cr3bp_state(reference, step_s, current_t)
            current_t += step_s
        rows.append(_cr3bp_state_to_moon_ric_rect(deputy, reference))
        previous_t = float(target_t)
    return np.vstack(rows) if rows else np.empty((0, 6), dtype=float)


def _linearized_cr3bp_moon_ric_coast_prediction(
    rel0: np.ndarray,
    *,
    target_state: np.ndarray,
    times: np.ndarray,
    current_t_s: float,
) -> np.ndarray:
    reference = np.array(target_state, dtype=float).reshape(6)
    deputy0 = _moon_ric_rect_state_to_cr3bp(rel0, reference)
    delta0 = deputy0 - reference
    stm = np.eye(6, dtype=float)
    rows: list[np.ndarray] = []
    current_t = float(current_t_s)
    previous_t = 0.0
    for target_t in np.array(times, dtype=float).reshape(-1):
        step_s = float(target_t - previous_t)
        if step_s > 0.0:
            reference, stm = propagate_cr3bp_reference_stm(reference, stm, step_s, current_t)
            current_t += step_s
        deputy_linear = reference + stm @ delta0
        rows.append(_cr3bp_state_to_moon_ric_rect(deputy_linear, reference))
        previous_t = float(target_t)
    return np.vstack(rows) if rows else np.empty((0, 6), dtype=float)


def _linearized_cr3bp_moon_ric_stm_table(
    *,
    target_state: np.ndarray,
    times: np.ndarray,
    current_t_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    reference = np.array(target_state, dtype=float).reshape(6)
    stm = np.eye(6, dtype=float)
    references: list[np.ndarray] = []
    stms: list[np.ndarray] = []
    current_t = float(current_t_s)
    previous_t = 0.0
    for target_t in np.array(times, dtype=float).reshape(-1):
        step_s = float(target_t - previous_t)
        if step_s > 0.0:
            reference, stm = propagate_cr3bp_reference_stm(reference, stm, step_s, current_t)
            current_t += step_s
        references.append(reference.copy())
        stms.append(stm.copy())
        previous_t = float(target_t)
    if not references:
        return np.empty((0, 6), dtype=float), np.empty((0, 6, 6), dtype=float)
    return np.vstack(references), np.stack(stms, axis=0)


def _linearized_cr3bp_moon_ric_projection_from_stm_table(
    rel0: np.ndarray,
    *,
    target_state: np.ndarray,
    references: np.ndarray,
    stms: np.ndarray,
    basis_axes: np.ndarray | None = None,
    basis_omega: np.ndarray | None = None,
) -> np.ndarray:
    reference0 = np.array(target_state, dtype=float).reshape(6)
    reference_rows = np.array(references, dtype=float).reshape(-1, 6)
    stm_rows = np.array(stms, dtype=float).reshape(-1, 6, 6)
    if reference_rows.size == 0 or stm_rows.size == 0:
        return np.empty((0, 6), dtype=float)
    deputy0 = _moon_ric_rect_state_to_cr3bp(rel0, reference0)
    delta0 = deputy0 - reference0
    deputy_rows = reference_rows + np.einsum("nij,j->ni", stm_rows, delta0)
    return _cr3bp_states_to_moon_ric_rect_rows(
        deputy_rows,
        reference_rows,
        basis_axes=basis_axes,
        basis_omega=basis_omega,
    )


def _satellite_pair_camera_center(
    *,
    chaser: np.ndarray,
    target: np.ndarray,
    x_axis: int | None,
    y_axis: int | None,
    keep_rc_reference_centered: bool,
) -> np.ndarray:
    if keep_rc_reference_centered and {x_axis, y_axis} == {0, 2}:
        return np.zeros(3, dtype=float)
    chaser_arr = np.array(chaser, dtype=float).reshape(-1)
    target_arr = np.array(target, dtype=float).reshape(-1)
    if chaser_arr.size < 3 or target_arr.size < 3:
        return np.zeros(3, dtype=float)
    pair = np.vstack((chaser_arr[:3], target_arr[:3]))
    if not np.all(np.isfinite(pair)):
        return np.zeros(3, dtype=float)
    center = np.zeros(3, dtype=float)
    axes = (0, 1) if x_axis is None or y_axis is None else (int(x_axis), int(y_axis))
    for axis in axes:
        center[axis] = float(np.mean(pair[:, axis]))
    return center


def _should_draw_cislunar_moon_background(*, relative_frame: str, x_axis: int, y_axis: int) -> bool:
    return _relative_frame_key(relative_frame) == "cislunar_l1" and {int(x_axis), int(y_axis)} == {1, 2}


def _scaled_body_rect_tuple(
    *,
    center_px: tuple[int, int],
    radius_km: float,
    scale_x: float,
    scale_y: float,
) -> tuple[int, int, int, int]:
    rx = max(1, int(round(float(radius_km) * float(scale_x))))
    ry = max(1, int(round(float(radius_km) * float(scale_y))))
    cx, cy = int(center_px[0]), int(center_px[1])
    return (cx - rx, cy - ry, rx * 2, ry * 2)


def _should_draw_nominal_nmt(nmt: np.ndarray, nmt_bounds: tuple[np.ndarray, ...]) -> bool:
    return bool(np.array(nmt).size and not nmt_bounds)


def _true_anomaly_deg_from_state(state: np.ndarray) -> float | None:
    arr = np.array(state, dtype=float).reshape(-1)
    if arr.size < 6:
        return None
    try:
        return float(rv_to_coe_eci(arr[:3], arr[3:6]).true_anomaly_deg)
    except ValueError:
        return None


def _history_array_tail(array: Any, row: np.ndarray, *, width: int, max_rows: int) -> Any:
    rows = int(max(max_rows, 1))
    if isinstance(array, _HistoryRingBuffer) and array.width == int(width) and array.max_rows == rows:
        ring = array
    else:
        source = array.rows() if isinstance(array, _HistoryRingBuffer) else array
        ring = _HistoryRingBuffer.from_rows(source, width=int(width), max_rows=rows)
    ring.append(row)
    return ring


def _dashboard_history_array(
    dashboard: Any,
    attr_name: str,
    fallback_rows: list[np.ndarray],
    *,
    width: int,
) -> np.ndarray:
    cached = getattr(dashboard, attr_name, None)
    if isinstance(cached, _HistoryRingBuffer) and cached.width == int(width):
        return cached.rows()
    if isinstance(cached, np.ndarray) and cached.ndim == 2 and cached.shape[1] == int(width) and cached.size:
        return cached
    if fallback_rows:
        return np.vstack(fallback_rows)
    return np.zeros((0, int(width)), dtype=float)


def _elliptic_linear_coast_states(
    rel0_ric: np.ndarray,
    times_s: np.ndarray,
    chief_state_eci: np.ndarray,
    *,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
) -> np.ndarray:
    """Propagate linearized RIC relative motion along a two-body elliptic chief.

    This is the numerical form of the Tschauner-Hempel idea used for teaching
    overlays: the relative state remains linearized, but the chief radius and
    angular rate vary along the orbit instead of being frozen as in HCW.
    """

    rel = np.array(rel0_ric, dtype=float).reshape(6)
    chief = np.array(chief_state_eci, dtype=float).reshape(6)
    times = np.array(times_s, dtype=float).reshape(-1)
    if times.size == 0:
        return np.empty((0, 6), dtype=float)
    order = np.argsort(times)
    sorted_times = times[order]
    state = np.hstack((chief, rel)).astype(float)
    rows = np.zeros((times.size, 6), dtype=float)
    current_t = 0.0
    max_step_s = max(float(np.max(np.diff(sorted_times))) if sorted_times.size > 1 else 0.0, 1.0)
    max_step_s = min(max(max_step_s, 1.0), 60.0)
    for sorted_idx, target_t in enumerate(sorted_times):
        target = float(max(target_t, current_t))
        while current_t < target:
            h = min(max_step_s, target - current_t)
            state = _rk4_step(_elliptic_linear_derivative, state, h, float(mu_km3_s2))
            current_t += h
        rows[order[sorted_idx]] = state[6:12]
    return rows


def _elliptic_ya_coast_states(
    rel0_ric: np.ndarray,
    times_s: np.ndarray,
    chief_state_eci: np.ndarray,
    *,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
) -> np.ndarray:
    """Propagate elliptic-chief RIC relative motion with the closed-form YA STM."""

    rel = np.array(rel0_ric, dtype=float).reshape(6)
    chief0 = np.array(chief_state_eci, dtype=float).reshape(6)
    times = np.array(times_s, dtype=float).reshape(-1)
    if times.size == 0:
        return np.empty((0, 6), dtype=float)
    order = np.argsort(times)
    sorted_times = times[order]
    chief = chief0.copy()
    rows = np.zeros((times.size, 6), dtype=float)
    current_t = 0.0
    for sorted_idx, target_t in enumerate(sorted_times):
        target = float(max(float(target_t), current_t))
        duration_s = target - current_t
        if duration_s > 0.0:
            chief = _two_body_coast_state(chief, duration_s, mu_km3_s2=float(mu_km3_s2))
            current_t = target
        phi = ya_closed_form_transition_matrix(target, chief0, chief, mu_km3_s2=float(mu_km3_s2))
        rows[order[sorted_idx]] = phi @ rel
    return rows


def _elliptic_reference_cache_valid(
    cached_reference_eci: Any,
    current_reference_eci: Any,
    *,
    elapsed_s: float,
) -> bool:
    if cached_reference_eci is None or current_reference_eci is None:
        return cached_reference_eci is None and current_reference_eci is None
    try:
        cached = np.array(cached_reference_eci, dtype=float).reshape(6)
        current = np.array(current_reference_eci, dtype=float).reshape(6)
    except (TypeError, ValueError):
        return False
    if not np.all(np.isfinite(cached)) or not np.all(np.isfinite(current)):
        return False
    if float(elapsed_s) <= 0.0:
        expected = cached
    else:
        expected = _two_body_coast_state(cached, float(elapsed_s))
    pos_error_km = float(np.linalg.norm(current[:3] - expected[:3]))
    vel_error_km_s = float(np.linalg.norm(current[3:6] - expected[3:6]))
    return bool(
        pos_error_km <= ELLIPTIC_REFERENCE_CACHE_POSITION_TOL_KM
        and vel_error_km_s <= ELLIPTIC_REFERENCE_CACHE_VELOCITY_TOL_KM_S
    )


def _cr3bp_reference_cache_valid(cached_reference: Any, current_reference: Any, *, elapsed_s: float = 0.0) -> bool:
    if cached_reference is None or current_reference is None:
        return cached_reference is None and current_reference is None
    try:
        cached = np.array(cached_reference, dtype=float).reshape(6)
        current = np.array(current_reference, dtype=float).reshape(6)
    except (TypeError, ValueError):
        return False
    if not np.all(np.isfinite(cached)) or not np.all(np.isfinite(current)):
        return False
    expected = cached
    if float(elapsed_s) > 0.0:
        try:
            expected = propagate_cr3bp_state(cached, float(elapsed_s), 0.0)
        except Exception:
            expected = cached
    pos_error_km = float(np.linalg.norm(current[:3] - expected[:3]))
    vel_error_km_s = float(np.linalg.norm(current[3:6] - expected[3:6]))
    return bool(
        pos_error_km <= CR3BP_REFERENCE_CACHE_POSITION_TOL_KM
        and vel_error_km_s <= CR3BP_REFERENCE_CACHE_VELOCITY_TOL_KM_S
    )


def _cr3bp_relative_cache_valid(cached_rel0: Any, current_rel0: np.ndarray) -> bool:
    try:
        cached = np.array(cached_rel0, dtype=float).reshape(6)
        current = np.array(current_rel0, dtype=float).reshape(6)
    except (TypeError, ValueError):
        return False
    if not np.all(np.isfinite(cached)) or not np.all(np.isfinite(current)):
        return False
    pos_error_km = float(np.linalg.norm(current[:3] - cached[:3]))
    vel_error_km_s = float(np.linalg.norm(current[3:6] - cached[3:6]))
    return bool(
        pos_error_km <= CR3BP_RELATIVE_CACHE_POSITION_TOL_KM
        and vel_error_km_s <= CR3BP_RELATIVE_CACHE_VELOCITY_TOL_KM_S
    )


def _two_body_coast_state(
    state_eci: np.ndarray,
    duration_s: float,
    *,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
) -> np.ndarray:
    state = np.array(state_eci, dtype=float).reshape(6)
    duration = float(max(duration_s, 0.0))
    if duration <= 0.0:
        return state.copy()
    current_t = 0.0
    step_s = min(max(duration / 4.0, 1.0), 10.0)
    out = state.astype(float)
    while current_t < duration:
        h = min(step_s, duration - current_t)
        out = _rk4_step(_two_body_derivative, out, h, float(mu_km3_s2))
        current_t += h
    return out


def _two_body_derivative(state: np.ndarray, mu_km3_s2: float) -> np.ndarray:
    r = np.array(state[:3], dtype=float)
    v = np.array(state[3:6], dtype=float)
    r_norm = float(np.linalg.norm(r))
    if r_norm <= 1.0e-9 or not np.isfinite(r_norm):
        return np.zeros_like(state)
    acc = -float(mu_km3_s2) * r / (r_norm**3)
    return np.hstack((v, acc))


def _rk4_step(func: Any, state: np.ndarray, step_s: float, mu_km3_s2: float) -> np.ndarray:
    h = float(step_s)
    k1 = func(state, mu_km3_s2)
    k2 = func(state + 0.5 * h * k1, mu_km3_s2)
    k3 = func(state + 0.5 * h * k2, mu_km3_s2)
    k4 = func(state + h * k3, mu_km3_s2)
    return state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _elliptic_linear_derivative(state: np.ndarray, mu_km3_s2: float) -> np.ndarray:
    chief_r = np.array(state[:3], dtype=float)
    chief_v = np.array(state[3:6], dtype=float)
    rho = np.array(state[6:9], dtype=float)
    rho_dot = np.array(state[9:12], dtype=float)
    r_norm = float(np.linalg.norm(chief_r))
    if r_norm <= 1.0e-9 or not np.isfinite(r_norm):
        return np.zeros_like(state)
    h_vec = np.cross(chief_r, chief_v)
    h_norm = float(np.linalg.norm(h_vec))
    theta_dot = h_norm / max(r_norm * r_norm, 1.0e-12)
    radial_rate = float(np.dot(chief_r, chief_v)) / r_norm
    theta_ddot = -2.0 * theta_dot * radial_rate / r_norm
    omega = np.array([0.0, 0.0, theta_dot], dtype=float)
    omega_dot = np.array([0.0, 0.0, theta_ddot], dtype=float)
    gravity_gradient = (float(mu_km3_s2) / (r_norm**3)) * np.array([2.0 * rho[0], -rho[1], -rho[2]])
    rho_ddot = (
        gravity_gradient
        - 2.0 * np.cross(omega, rho_dot)
        - np.cross(omega_dot, rho)
        - np.cross(omega, np.cross(omega, rho))
    )
    chief_acc = -float(mu_km3_s2) * chief_r / (r_norm**3)
    return np.hstack((chief_v, chief_acc, rho_dot, rho_ddot))
