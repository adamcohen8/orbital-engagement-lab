# ruff: noqa: F401,F403,F405,I001
from .dashboard_common import *
from .geometry import *
from .prediction import *
from .camera import *

class DashboardStateMixin:
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

    def tick(self, fps: float = 60.0) -> None:
        self.clock.tick(float(max(fps, 1.0)))
