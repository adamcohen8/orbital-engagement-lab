"""Opt-in frame pacing, diagnostics, and adaptive presentation policy."""

from __future__ import annotations

import json
import platform
from collections import deque
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

PRESENTATION_MODES = ("compatibility", "standard", "high_refresh", "auto")
PRESENTATION_VSYNC_MODES = ("auto", "on", "off")
DEFAULT_HIGH_REFRESH_FPS = 120.0
DEFAULT_REFRESH_FALLBACK_HZ = 60.0


def normalize_presentation_mode(value: object) -> str:
    mode = str(value or "compatibility").strip().lower().replace("-", "_")
    if mode not in PRESENTATION_MODES:
        raise ValueError(f"presentation mode must be one of {', '.join(PRESENTATION_MODES)}")
    return mode


def normalize_presentation_vsync(value: object) -> str:
    mode = str(value or "auto").strip().lower()
    if mode not in PRESENTATION_VSYNC_MODES:
        raise ValueError(f"presentation VSync mode must be one of {', '.join(PRESENTATION_VSYNC_MODES)}")
    return mode


def positive_fps_or_none(value: object) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not result > 0.0:
        raise ValueError("presentation FPS values must be positive")
    return result


@dataclass(frozen=True, slots=True)
class PresentationSettings:
    mode: str = "compatibility"
    fps_cap: float | None = None
    vsync: str = "auto"
    diagnostics: bool = False
    diagnostics_output: Path | None = None
    high_refresh_ceiling_fps: float = DEFAULT_HIGH_REFRESH_FPS
    refresh_rate_hz: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", normalize_presentation_mode(self.mode))
        object.__setattr__(self, "vsync", normalize_presentation_vsync(self.vsync))
        object.__setattr__(self, "fps_cap", positive_fps_or_none(self.fps_cap))
        object.__setattr__(
            self,
            "high_refresh_ceiling_fps",
            positive_fps_or_none(self.high_refresh_ceiling_fps) or DEFAULT_HIGH_REFRESH_FPS,
        )
        object.__setattr__(self, "refresh_rate_hz", positive_fps_or_none(self.refresh_rate_hz))
        if self.diagnostics_output is not None:
            object.__setattr__(self, "diagnostics_output", Path(self.diagnostics_output))

    @property
    def enabled(self) -> bool:
        return self.mode != "compatibility" or self.diagnostics or self.diagnostics_output is not None

    @property
    def requests_vsync(self) -> bool:
        # Pygame cannot report whether a VSync request is actually honored and
        # its blocking flip is serialized with physics on the game thread.
        # Candidate auto mode therefore uses the existing software pacer;
        # users can still request platform VSync explicitly.
        return self.vsync == "on" and self.mode != "compatibility"


@dataclass(frozen=True, slots=True)
class PresentationQuality:
    name: str
    ghost_draw_points: int
    trail_draw_points: int
    prediction_interval_scale: float
    render_fps_scale: float = 1.0


PRESENTATION_QUALITIES = (
    PresentationQuality("ultra", 180, 360, 1.0),
    PresentationQuality("high", 120, 260, 1.0),
    PresentationQuality("balanced", 90, 190, 1.5),
    PresentationQuality("low", 60, 120, 2.5),
    PresentationQuality("minimum", 36, 72, 4.0, 0.5),
)


def detected_refresh_rate_hz(pygame: Any) -> float | None:
    getter = getattr(getattr(pygame, "display", None), "get_current_refresh_rate", None)
    if callable(getter):
        try:
            refresh = float(getter())
        except (TypeError, ValueError, RuntimeError):
            refresh = 0.0
        if refresh > 1.0:
            return refresh
    return None


def _percentile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = min(max(float(fraction), 0.0), 1.0) * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    alpha = position - lower
    return ordered[lower] * (1.0 - alpha) + ordered[upper] * alpha


class PresentationFrameController:
    """Own display-only pacing policy and non-authoritative diagnostics."""

    _WINDOW = 180
    _MIN_ADAPTIVE_SAMPLES = 45
    _DEGRADE_DWELL_S = 0.75
    _UPGRADE_DWELL_S = 3.0

    def __init__(
        self,
        settings: PresentationSettings,
        *,
        display_refresh_hz: float = DEFAULT_REFRESH_FALLBACK_HZ,
        display_size: tuple[int, int] | None = None,
        fullscreen: bool | None = None,
        vsync_active: bool | None = None,
    ) -> None:
        self.settings = settings
        self.display_refresh_hz = max(float(display_refresh_hz), 1.0)
        self.display_size = display_size
        self.fullscreen = fullscreen
        self.vsync_active = vsync_active
        self._quality_index = 0 if settings.mode in {"high_refresh", "auto"} else 1
        self._frame_work_s: deque[float] = deque(maxlen=self._WINDOW)
        self._frame_interval_s: deque[float] = deque(maxlen=self._WINDOW)
        self._draw_s: deque[float] = deque(maxlen=self._WINDOW)
        self._simulation_step_s: deque[float] = deque(maxlen=self._WINDOW * 4)
        self._snapshot_age_s: deque[float] = deque(maxlen=self._WINDOW)
        self._projection_horizon_s: deque[float] = deque(maxlen=self._WINDOW)
        self._projection_compute_s: deque[float] = deque(maxlen=self._WINDOW)
        self._prediction_recompute_s: deque[float] = deque(maxlen=self._WINDOW)
        self._reconciliation_error_km: deque[float] = deque(maxlen=self._WINDOW)
        self._steps_per_frame: deque[int] = deque(maxlen=self._WINDOW)
        self._frames = 0
        self._frames_without_authoritative_step = 0
        self._catch_up_steps = 0
        self._discarded_backlog_steps = 0
        self._projection_cap_hits = 0
        self._quality_transitions: list[dict[str, object]] = []
        self._scheduler_step_limit = 1
        self._scheduler_compute_limited_frames = 0
        self._last_quality_change_s = perf_counter()
        self._last_target_fps = DEFAULT_REFRESH_FALLBACK_HZ
        self._started_s = perf_counter()
        self._last_observed_frame_s: float | None = None
        self._dashboard: Any | None = None

    @property
    def quality(self) -> PresentationQuality:
        return PRESENTATION_QUALITIES[self._quality_index]

    @property
    def trajectory_aware(self) -> bool:
        return self.settings.mode != "compatibility"

    def apply_to_dashboard(self, dashboard: Any) -> None:
        self._dashboard = dashboard
        quality = self.quality
        dashboard.presentation_mode = self.settings.mode
        dashboard.presentation_ghost_draw_points = quality.ghost_draw_points
        dashboard.presentation_trail_draw_points = quality.trail_draw_points
        dashboard.presentation_prediction_interval_scale = quality.prediction_interval_scale
        dashboard.presentation_controller = self
        if not hasattr(dashboard, "presentation_reconciliation_duration_s"):
            dashboard.presentation_reconciliation_duration_s = 0.08
        if not hasattr(dashboard, "presentation_reconciliation_max_error_km"):
            dashboard.presentation_reconciliation_max_error_km = 0.25

    def target_fps(
        self,
        *,
        compatibility_fps: float,
        recording: bool,
        recording_fps: float,
        static_screen: bool,
    ) -> float:
        if self.settings.mode == "compatibility":
            target = float(compatibility_fps)
        elif recording:
            target = max(float(recording_fps), 1.0)
        elif static_screen:
            target = 15.0
        elif self.settings.mode == "standard":
            target = 60.0
        else:
            target = min(self.display_refresh_hz, self.settings.high_refresh_ceiling_fps)
        if self.settings.mode == "auto" and not recording and not static_screen:
            target *= self.quality.render_fps_scale
        if self.settings.fps_cap is not None and not recording:
            target = min(target, self.settings.fps_cap)
        self._last_target_fps = max(float(target), 1.0)
        return self._last_target_fps

    def record_draw(self, elapsed_s: float) -> None:
        self._draw_s.append(max(float(elapsed_s), 0.0))

    def record_simulation_step(self, elapsed_s: float) -> None:
        self._simulation_step_s.append(max(float(elapsed_s), 0.0))

    def record_projection(
        self,
        *,
        horizon_s: float,
        cap_hit: bool,
        computation_s: float | None = None,
    ) -> None:
        self._projection_horizon_s.append(max(float(horizon_s), 0.0))
        if computation_s is not None:
            self._projection_compute_s.append(max(float(computation_s), 0.0))
        if cap_hit:
            self._projection_cap_hits += 1

    def record_prediction_recompute(self, elapsed_s: float) -> None:
        self._prediction_recompute_s.append(max(float(elapsed_s), 0.0))

    def record_reconciliation_error(self, error_km: float) -> None:
        self._reconciliation_error_km.append(max(float(error_km), 0.0))

    def authoritative_step_limit(self, *, wall_step_s: float, hard_limit: int = 12) -> int:
        """Bound catch-up work so an overloaded simulation cannot starve drawing."""

        frame_budget_s = 1.0 / max(self._last_target_fps, 1.0)
        nominal_limit = max(int(ceil(frame_budget_s / max(float(wall_step_s), 1.0e-9))), 1)
        nominal_limit = min(nominal_limit, max(int(hard_limit), 1))
        if self._simulation_step_s:
            step_cost_s = max(_percentile(self._simulation_step_s, 0.95), 1.0e-9)
            draw_cost_s = _percentile(self._draw_s, 0.95)
            available_s = max(frame_budget_s - draw_cost_s, frame_budget_s * 0.25)
            compute_limit = max(int(available_s // step_cost_s), 1)
        else:
            # Start conservatively until the current session has measured its
            # real physics cost. This prevents the first delayed frame from
            # initiating a catch-up spiral.
            compute_limit = 2
        selected = min(nominal_limit, compute_limit)
        if selected < nominal_limit:
            self._scheduler_compute_limited_frames += 1
        self._scheduler_step_limit = max(selected, 1)
        return self._scheduler_step_limit

    def observe_frame(
        self,
        *,
        work_s: float,
        authoritative_steps: int,
        discarded_backlog_steps: int = 0,
        snapshot_age_s: float | None = None,
    ) -> None:
        duration = max(float(work_s), 0.0)
        now = perf_counter()
        if self._last_observed_frame_s is not None:
            self._frame_interval_s.append(max(now - self._last_observed_frame_s, 0.0))
        self._last_observed_frame_s = now
        steps = max(int(authoritative_steps), 0)
        self._frame_work_s.append(duration)
        self._steps_per_frame.append(steps)
        self._frames += 1
        if steps == 0:
            self._frames_without_authoritative_step += 1
        if steps > 1:
            self._catch_up_steps += steps - 1
        self._discarded_backlog_steps += max(int(discarded_backlog_steps), 0)
        if snapshot_age_s is not None:
            self._snapshot_age_s.append(max(float(snapshot_age_s), 0.0))
        self._adapt_if_needed()

    def _adapt_if_needed(self) -> None:
        if self.settings.mode != "auto" or len(self._frame_work_s) < self._MIN_ADAPTIVE_SAMPLES:
            return
        now = perf_counter()
        elapsed = now - self._last_quality_change_s
        budget_s = 1.0 / max(self._last_target_fps, 1.0)
        p95_s = _percentile(self._frame_work_s, 0.95)
        if p95_s > budget_s * 0.92 and self._quality_index < len(PRESENTATION_QUALITIES) - 1:
            if elapsed >= self._DEGRADE_DWELL_S:
                self._change_quality(self._quality_index + 1, reason="frame_budget_exceeded", now_s=now)
            return
        if p95_s < budget_s * 0.55 and self._quality_index > 0 and elapsed >= self._UPGRADE_DWELL_S:
            self._change_quality(self._quality_index - 1, reason="sustained_headroom", now_s=now)

    def _change_quality(self, index: int, *, reason: str, now_s: float) -> None:
        previous = self.quality.name
        self._quality_index = min(max(int(index), 0), len(PRESENTATION_QUALITIES) - 1)
        self._last_quality_change_s = float(now_s)
        self._quality_transitions.append(
            {
                "elapsed_wall_s": max(float(now_s) - self._started_s, 0.0),
                "from": previous,
                "to": self.quality.name,
                "reason": str(reason),
            }
        )
        if self._dashboard is not None:
            self.apply_to_dashboard(self._dashboard)
            if hasattr(self._dashboard, "_frame_cache_dirty"):
                self._dashboard._frame_cache_dirty = True

    def overlay_lines(self) -> tuple[str, ...]:
        if not self.settings.diagnostics:
            return ()
        frame_p50 = _percentile(self._frame_interval_s, 0.50) * 1.0e3
        frame_p95 = _percentile(self._frame_work_s, 0.95) * 1.0e3
        draw_p95 = _percentile(self._draw_s, 0.95) * 1.0e3
        sim_p95 = _percentile(self._simulation_step_s, 0.95) * 1.0e3
        prediction_p95 = _percentile(self._prediction_recompute_s, 0.95) * 1.0e3
        projection_p95 = _percentile(self._projection_compute_s, 0.95) * 1.0e3
        achieved = 0.0 if frame_p50 <= 0.0 else 1000.0 / frame_p50
        return (
            f"Presentation {self.settings.mode} / {self.quality.name}",
            f"FPS {achieved:5.1f} target {self._last_target_fps:5.1f}  frame p95 {frame_p95:5.1f} ms",
            f"draw p95 {draw_p95:5.1f} ms  physics p95 {sim_p95:5.1f} ms",
            f"prediction p95 {prediction_p95:5.1f} ms  projection p95 {projection_p95:5.1f} ms",
            f"step cap {self._scheduler_step_limit}  catch-up {self._catch_up_steps}  "
            f"dropped backlog {self._discarded_backlog_steps}",
        )

    def summary(self) -> dict[str, object]:
        frame_p50 = _percentile(self._frame_interval_s, 0.50)
        achieved_fps = 0.0 if frame_p50 <= 0.0 else 1.0 / frame_p50
        projection_samples = len(self._projection_horizon_s)
        return {
            "schema": "oel.game.presentation_diagnostics.v1",
            "presentation_mode": self.settings.mode,
            "quality": self.quality.name,
            "target_fps": self._last_target_fps,
            "achieved_fps": achieved_fps,
            "achieved_fps_from_work_time": achieved_fps,
            "display_refresh_hz": self.display_refresh_hz,
            "display_size": None if self.display_size is None else list(self.display_size),
            "fullscreen": self.fullscreen,
            "vsync_preference": self.settings.vsync,
            "vsync_active": self.vsync_active,
            "frame_count": self._frames,
            "frames_without_authoritative_step": self._frames_without_authoritative_step,
            "catch_up_steps": self._catch_up_steps,
            "discarded_backlog_steps": self._discarded_backlog_steps,
            "scheduler_step_limit": self._scheduler_step_limit,
            "scheduler_compute_limited_frames": self._scheduler_compute_limited_frames,
            "projection_cap_hits": self._projection_cap_hits,
            "projection_cap_hit_fraction": (
                0.0 if projection_samples <= 0 else self._projection_cap_hits / projection_samples
            ),
            "steps_per_frame": self._distribution(self._steps_per_frame),
            "frame_work_s": self._distribution(self._frame_work_s),
            "frame_interval_s": self._distribution(self._frame_interval_s),
            "draw_s": self._distribution(self._draw_s),
            "simulation_step_s": self._distribution(self._simulation_step_s),
            "snapshot_age_s": self._distribution(self._snapshot_age_s),
            "projection_horizon_s": self._distribution(self._projection_horizon_s),
            "projection_compute_s": self._distribution(self._projection_compute_s),
            "prediction_recompute_s": self._distribution(self._prediction_recompute_s),
            "reconciliation_error_km": self._distribution(self._reconciliation_error_km),
            "quality_transitions": list(self._quality_transitions),
            "platform": platform.platform(),
            "python": platform.python_version(),
        }

    @staticmethod
    def _distribution(values: Iterable[float]) -> dict[str, float | int]:
        rows = tuple(float(value) for value in values)
        return {
            "samples": len(rows),
            "median": _percentile(rows, 0.50),
            "p95": _percentile(rows, 0.95),
            "max": max(rows, default=0.0),
        }

    def write_summary(self) -> Path | None:
        path = self.settings.diagnostics_output
        if path is None:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.summary(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path


def create_presentation_controller(pygame: Any, dashboard: Any, settings: PresentationSettings) -> PresentationFrameController | None:
    if not settings.enabled:
        return None
    size = None
    try:
        size = tuple(int(value) for value in dashboard.screen.get_size())
    except (AttributeError, TypeError, ValueError):
        pass
    detected_refresh = detected_refresh_rate_hz(pygame)
    refresh = settings.refresh_rate_hz or detected_refresh
    if refresh is None:
        refresh = (
            settings.high_refresh_ceiling_fps
            if settings.mode == "high_refresh"
            else DEFAULT_REFRESH_FALLBACK_HZ
        )
    controller = PresentationFrameController(
        settings,
        display_refresh_hz=refresh,
        display_size=size,
        fullscreen=bool(getattr(dashboard, "fullscreen", False)),
        vsync_active=bool(getattr(dashboard, "presentation_vsync_active", False)),
    )
    controller.apply_to_dashboard(dashboard)
    return controller


__all__ = [
    "DEFAULT_HIGH_REFRESH_FPS",
    "PRESENTATION_MODES",
    "PRESENTATION_QUALITIES",
    "PRESENTATION_VSYNC_MODES",
    "PresentationFrameController",
    "PresentationQuality",
    "PresentationSettings",
    "create_presentation_controller",
    "detected_refresh_rate_hz",
    "normalize_presentation_mode",
    "normalize_presentation_vsync",
]
