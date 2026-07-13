from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Any

import numpy as np

from sim.api import SimulationSnapshot
from sim.dynamics.orbit.cr3bp import EARTH_MOON_MEAN_MOTION_RAD_S, cr3bp_moon_state_km_s, cr3bp_relative_state
from sim.dynamics.orbit.epoch import resolve_sun_moon_positions
from sim.game.formatting import format_distance_km, format_speed_km_s, format_speed_m_s
from sim.utils.frames import eci_relative_to_ric_rect, ric_dcm_ir_from_rv

EARTH_MU_KM3_S2 = 398600.4418
_BURN_AXIS_INDEX = {"radial": 0, "in_track": 1, "cross_track": 2}
_BURN_AXIS_LABEL = {"radial": "Radial", "in_track": "In-track", "cross_track": "Cross-track"}
_BURN_AXIS_SHORT_LABEL = {"radial": "R", "in_track": "I", "cross_track": "C"}
_BURN_AXIS_MIN_COMPONENT_FRACTION = 0.75
OPERATOR_RELAXED_REQUIRED_BURN_AXIS_SCENARIO_IDS = frozenset(
    {
        "rpo_07_elliptic_burn_then_approach",
    }
)


def _segment_crosses_sphere_km(positions_km: np.ndarray, radius_km: float) -> bool:
    pos = np.asarray(positions_km, dtype=float)
    if pos.ndim != 2 or pos.shape[0] < 2 or pos.shape[1] < 3:
        return False
    r2 = float(radius_km) ** 2
    p0 = pos[:-1, :3]
    p1 = pos[1:, :3]
    d = p1 - p0
    denom = np.sum(d * d, axis=1)
    u = np.zeros_like(denom)
    moving = denom > 0.0
    u[moving] = np.clip(-np.sum(p0[moving] * d[moving], axis=1) / denom[moving], 0.0, 1.0)
    closest = p0 + u[:, None] * d
    return bool(np.any(np.sum(closest * closest, axis=1) < r2))


@dataclass(frozen=True)
class ForbiddenRegionConfig:
    name: str
    min_ric_km: np.ndarray
    max_ric_km: np.ndarray
    plot_planes: tuple[str, ...] = ()
    kind: str = "box"
    plane: str = "RI"
    center_ric_km: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    inner_radius_km: float | None = None
    outer_radius_km: float | None = None
    angle_min_deg: float | None = None
    angle_max_deg: float | None = None
    max_abs_out_of_plane_km: float | None = None
    axis: str = "I"
    radius_km: float | None = None
    height_km: float | None = None

    @classmethod
    def from_mapping(cls, raw: dict[str, Any], *, index: int) -> ForbiddenRegionConfig:
        plot_planes = _plot_planes_from_metadata(raw.get("plot_planes"))
        plane = str(raw.get("plane", plot_planes[0] if plot_planes else "RI") or "RI").strip().upper()
        return cls(
            name=str(raw.get("name", f"forbidden_region_{index}") or f"forbidden_region_{index}"),
            min_ric_km=_ric_bound_array(raw.get("min_ric_km"), default=-np.inf, field_name="min_ric_km"),
            max_ric_km=_ric_bound_array(raw.get("max_ric_km"), default=np.inf, field_name="max_ric_km"),
            plot_planes=plot_planes,
            kind=str(raw.get("kind", "box") or "box").strip().lower(),
            plane=plane,
            center_ric_km=_ric_bound_array(raw.get("center_ric_km"), default=0.0, field_name="center_ric_km"),
            inner_radius_km=_optional_float(raw.get("inner_radius_km")),
            outer_radius_km=_optional_float(raw.get("outer_radius_km")),
            angle_min_deg=_optional_float(raw.get("angle_min_deg")),
            angle_max_deg=_optional_float(raw.get("angle_max_deg")),
            max_abs_out_of_plane_km=_optional_float(raw.get("max_abs_out_of_plane_km")),
            axis=str(raw.get("axis", "I") or "I").strip().upper(),
            radius_km=_optional_float(raw.get("radius_km")),
            height_km=_optional_float(raw.get("height_km")),
        )

    def contains_positions(self, ric_positions_km: np.ndarray) -> np.ndarray:
        pos = np.array(ric_positions_km, dtype=float)
        if pos.ndim == 1:
            pos = pos.reshape(1, -1)
        if pos.shape[1] < 3:
            raise ValueError("ric_positions_km must contain R, I, and C components.")
        if self.kind == "annular_sector":
            return self._contains_annular_sector(pos)
        if self.kind == "cylinder":
            return self._contains_cylinder(pos)
        if self.kind == "sphere":
            return self._contains_sphere(pos)
        lower = np.array(self.min_ric_km, dtype=float).reshape(1, 3)
        upper = np.array(self.max_ric_km, dtype=float).reshape(1, 3)
        return np.all((pos[:, :3] >= lower) & (pos[:, :3] <= upper), axis=1)

    def intersects_segment(self, start_ric_km: np.ndarray, end_ric_km: np.ndarray) -> bool:
        """Return whether a straight sample-to-sample segment enters the region."""

        start = np.asarray(start_ric_km, dtype=float).reshape(3)
        end = np.asarray(end_ric_km, dtype=float).reshape(3)
        if not np.all(np.isfinite(start)) or not np.all(np.isfinite(end)):
            return False
        if bool(np.any(self.contains_positions(np.vstack((start, end))))):
            return True
        if self.kind == "sphere":
            if self.radius_km is None:
                return False
            center = np.asarray(self.center_ric_km, dtype=float).reshape(3)
            return _position_segment_sphere_interval(
                start - center,
                end - center,
                radius_km=float(self.radius_km),
            ) is not None
        if self.kind == "cylinder":
            return _position_segment_intersects_cylinder(
                start,
                end,
                center=np.asarray(self.center_ric_km, dtype=float).reshape(3),
                axis=_axis_index(self.axis),
                radius_km=self.radius_km,
                height_km=self.height_km,
            )
        if self.kind == "annular_sector":
            return _position_segment_intersects_annular_sector(start, end, region=self)
        return _position_segment_intersects_bounds(
            start,
            end,
            lower=np.asarray(self.min_ric_km, dtype=float).reshape(3),
            upper=np.asarray(self.max_ric_km, dtype=float).reshape(3),
        )

    def sector_polygon_ric(self, *, samples: int = 64) -> np.ndarray:
        if self.kind != "annular_sector" or self.inner_radius_km is None or self.outer_radius_km is None:
            return np.zeros((0, 3), dtype=float)
        x_axis, y_axis, _ = _plane_axes(self.plane)
        inner = max(float(self.inner_radius_km), 0.0)
        outer = max(float(self.outer_radius_km), inner)
        start = 0.0 if self.angle_min_deg is None else float(self.angle_min_deg)
        end = 360.0 if self.angle_max_deg is None else float(self.angle_max_deg)
        while end <= start:
            end += 360.0
        angles = np.deg2rad(np.linspace(start, end, max(int(samples), 8)))
        center = np.array(self.center_ric_km, dtype=float).reshape(3)
        pts: list[np.ndarray] = []
        for radius, seq in ((outer, angles), (inner, angles[::-1])):
            for theta in seq:
                point = center.copy()
                point[x_axis] += radius * float(np.cos(theta))
                point[y_axis] += radius * float(np.sin(theta))
                pts.append(point)
        return np.vstack(pts) if pts else np.zeros((0, 3), dtype=float)

    def _contains_annular_sector(self, pos: np.ndarray) -> np.ndarray:
        if self.inner_radius_km is None or self.outer_radius_km is None:
            return np.zeros(pos.shape[0], dtype=bool)
        x_axis, y_axis, out_axis = _plane_axes(self.plane)
        delta = pos[:, :3] - np.array(self.center_ric_km, dtype=float).reshape(1, 3)
        xy = delta[:, [x_axis, y_axis]]
        radius = np.linalg.norm(xy, axis=1)
        ok = (radius >= float(self.inner_radius_km)) & (radius <= float(self.outer_radius_km))
        if self.max_abs_out_of_plane_km is not None:
            ok &= np.abs(delta[:, out_axis]) <= float(self.max_abs_out_of_plane_km)
        if self.angle_min_deg is not None or self.angle_max_deg is not None:
            start = 0.0 if self.angle_min_deg is None else float(self.angle_min_deg)
            end = 360.0 if self.angle_max_deg is None else float(self.angle_max_deg)
            angles = np.rad2deg(np.arctan2(xy[:, 1], xy[:, 0]))
            ok &= _angles_in_range_deg(angles, start, end)
        return ok

    def _contains_cylinder(self, pos: np.ndarray) -> np.ndarray:
        if self.radius_km is None or self.height_km is None:
            return np.zeros(pos.shape[0], dtype=bool)
        axis = _axis_index(self.axis)
        cross_axes = tuple(idx for idx in (0, 1, 2) if idx != axis)
        delta = pos[:, :3] - np.array(self.center_ric_km, dtype=float).reshape(1, 3)
        cross_radius = np.linalg.norm(delta[:, cross_axes], axis=1)
        half_height = float(self.height_km) / 2.0
        return (cross_radius <= float(self.radius_km)) & (np.abs(delta[:, axis]) <= half_height)

    def _contains_sphere(self, pos: np.ndarray) -> np.ndarray:
        if self.radius_km is None:
            return np.zeros(pos.shape[0], dtype=bool)
        delta = pos[:, :3] - np.array(self.center_ric_km, dtype=float).reshape(1, 3)
        return np.linalg.norm(delta, axis=1) <= float(self.radius_km)


@dataclass(frozen=True)
class ApproachGateConfig:
    name: str
    radial_ric_km: float
    radial_tolerance_km: float = 0.08
    max_abs_intrack_km: float | None = None
    max_abs_cross_track_km: float | None = None
    max_abs_radial_rate_km_s: float | None = None
    max_total_speed_km_s: float | None = None
    required: bool = True

    @classmethod
    def from_mapping(cls, raw: dict[str, Any], *, index: int) -> ApproachGateConfig:
        if "radial_ric_km" not in raw:
            raise ValueError("Approach gate radial_ric_km is required.")
        return cls(
            name=str(raw.get("name", f"approach_gate_{index}") or f"approach_gate_{index}"),
            radial_ric_km=float(raw["radial_ric_km"]),
            radial_tolerance_km=float(raw.get("radial_tolerance_km", 0.08) or 0.08),
            max_abs_intrack_km=_optional_float(raw.get("max_abs_intrack_km")),
            max_abs_cross_track_km=_optional_float(raw.get("max_abs_cross_track_km")),
            max_abs_radial_rate_km_s=_optional_float(raw.get("max_abs_radial_rate_km_s")),
            max_total_speed_km_s=_optional_float(raw.get("max_total_speed_km_s")),
            required=bool(raw.get("required", True)),
        )

    def samples_near_gate(self, relative_ric_state: np.ndarray) -> np.ndarray:
        rel = np.array(relative_ric_state, dtype=float)
        if rel.ndim == 1:
            rel = rel.reshape(1, -1)
        if rel.shape[1] < 6:
            raise ValueError("relative_ric_state must contain RIC position and velocity.")
        return np.abs(rel[:, 0] - float(self.radial_ric_km)) <= float(self.radial_tolerance_km)

    def samples_satisfying_gate(self, relative_ric_state: np.ndarray) -> np.ndarray:
        rel = np.array(relative_ric_state, dtype=float)
        if rel.ndim == 1:
            rel = rel.reshape(1, -1)
        if rel.shape[1] < 6:
            raise ValueError("relative_ric_state must contain RIC position and velocity.")
        ok = self.samples_near_gate(rel)
        if self.max_abs_intrack_km is not None:
            ok &= np.abs(rel[:, 1]) <= float(self.max_abs_intrack_km)
        if self.max_abs_cross_track_km is not None:
            ok &= np.abs(rel[:, 2]) <= float(self.max_abs_cross_track_km)
        if self.max_abs_radial_rate_km_s is not None:
            ok &= np.abs(rel[:, 3]) <= float(self.max_abs_radial_rate_km_s)
        if self.max_total_speed_km_s is not None:
            ok &= np.linalg.norm(rel[:, 3:6], axis=1) <= float(self.max_total_speed_km_s)
        return ok


@dataclass(frozen=True)
class InspectionGateConfig:
    name: str
    center_ric_km: np.ndarray
    half_width_ric_km: np.ndarray
    max_total_speed_km_s: float | None = None

    @classmethod
    def from_mapping(cls, raw: dict[str, Any], *, index: int) -> InspectionGateConfig:
        return cls(
            name=str(raw.get("name", f"inspection_gate_{index}") or f"inspection_gate_{index}"),
            center_ric_km=_ric_bound_array(raw.get("center_ric_km"), default=0.0, field_name="center_ric_km"),
            half_width_ric_km=_ric_bound_array(
                raw.get("half_width_ric_km", [0.25, 0.25, 0.25]),
                default=0.25,
                field_name="half_width_ric_km",
            ),
            max_total_speed_km_s=_optional_float(raw.get("max_total_speed_km_s")),
        )

    def samples_satisfying_gate(self, relative_ric_state: np.ndarray) -> np.ndarray:
        rel = np.array(relative_ric_state, dtype=float)
        if rel.ndim == 1:
            rel = rel.reshape(1, -1)
        if rel.shape[1] < 6:
            raise ValueError("relative_ric_state must contain RIC position and velocity.")
        center = np.array(self.center_ric_km, dtype=float).reshape(1, 3)
        half_width = np.array(self.half_width_ric_km, dtype=float).reshape(1, 3)
        ok = np.all(np.abs(rel[:, :3] - center) <= half_width, axis=1)
        if self.max_total_speed_km_s is not None:
            ok &= np.linalg.norm(rel[:, 3:6], axis=1) <= float(self.max_total_speed_km_s)
        return ok

    def segment_satisfies_gate(self, start_relative_ric_state: np.ndarray, end_relative_ric_state: np.ndarray) -> bool:
        start = np.array(start_relative_ric_state, dtype=float).reshape(-1)
        end = np.array(end_relative_ric_state, dtype=float).reshape(-1)
        if start.size < 6 or end.size < 6:
            raise ValueError("relative_ric_state must contain RIC position and velocity.")
        center = np.array(self.center_ric_km, dtype=float).reshape(3)
        half_width = np.array(self.half_width_ric_km, dtype=float).reshape(3)
        if not _position_segment_intersects_box(start[:3], end[:3], center=center, half_width=half_width):
            return False
        if self.max_total_speed_km_s is None:
            return True
        endpoint_speed = max(float(np.linalg.norm(start[3:6])), float(np.linalg.norm(end[3:6])))
        return endpoint_speed <= float(self.max_total_speed_km_s)


@dataclass(frozen=True)
class SunAngleConstraintConfig:
    name: str
    sun_direction_ric: np.ndarray
    allowed_center_ric: np.ndarray
    allowed_half_angle_deg: float
    dynamic_sun: bool = False
    allowed_center_mode: str = "configured"
    sun_environment: dict[str, Any] = field(default_factory=dict)
    min_range_km: float | None = None
    max_range_km: float | None = None
    plot_planes: tuple[str, ...] = ("RI", "RC")
    beam_radius_km: float | None = None

    @classmethod
    def from_mapping(cls, raw: dict[str, Any], *, index: int) -> SunAngleConstraintConfig:
        return cls(
            name=str(raw.get("name", f"sun_angle_constraint_{index}") or f"sun_angle_constraint_{index}"),
            sun_direction_ric=_unit_ric_array(
                raw.get("sun_direction_ric", raw.get("sun_vector_ric", [0.0, 1.0, 0.0])),
                field_name="sun_direction_ric",
            ),
            allowed_center_ric=_unit_ric_array(
                raw.get("allowed_center_ric", raw.get("beam_center_ric", [0.0, -1.0, 0.0])),
                field_name="allowed_center_ric",
            ),
            allowed_half_angle_deg=float(raw.get("allowed_half_angle_deg", raw.get("half_angle_deg", 35.0)) or 35.0),
            dynamic_sun=bool(raw.get("dynamic_sun", False)),
            allowed_center_mode=str(raw.get("allowed_center_mode", "configured") or "configured").strip().lower(),
            min_range_km=_optional_float(raw.get("min_range_km")),
            max_range_km=_optional_float(raw.get("max_range_km")),
            plot_planes=_plot_planes_from_metadata(raw.get("plot_planes")) or ("RI", "RC"),
            beam_radius_km=_optional_float(raw.get("beam_radius_km")),
        )

    def with_sun_environment(self, env: dict[str, Any]) -> SunAngleConstraintConfig:
        return replace(self, sun_environment=dict(env or {}))

    def sun_direction_at_ric(self, *, target_state_eci: np.ndarray | None = None, time_s: float | None = None) -> np.ndarray:
        if not self.dynamic_sun or target_state_eci is None:
            return np.array(self.sun_direction_ric, dtype=float).reshape(3)
        target = np.array(target_state_eci, dtype=float).reshape(-1)
        if target.size < 6:
            return np.array(self.sun_direction_ric, dtype=float).reshape(3)
        try:
            sun_pos, _ = resolve_sun_moon_positions(dict(self.sun_environment or {}), 0.0 if time_s is None else float(time_s))
        except Exception:
            return np.array(self.sun_direction_ric, dtype=float).reshape(3)
        sun_vec_eci = np.array(sun_pos, dtype=float).reshape(3) - target[:3]
        norm = float(np.linalg.norm(sun_vec_eci))
        if not np.isfinite(norm) or norm <= 0.0:
            return np.array(self.sun_direction_ric, dtype=float).reshape(3)
        c_ir = ric_dcm_ir_from_rv(target[:3], target[3:6])
        sun_ric = c_ir.T @ (sun_vec_eci / norm)
        sun_norm = float(np.linalg.norm(sun_ric))
        if not np.isfinite(sun_norm) or sun_norm <= 0.0:
            return np.array(self.sun_direction_ric, dtype=float).reshape(3)
        return sun_ric / sun_norm

    def allowed_center_at_ric(
        self, *, target_state_eci: np.ndarray | None = None, time_s: float | None = None
    ) -> np.ndarray:
        if self.allowed_center_mode in {"anti_sun", "antisun", "opposite_sun"}:
            return -self.sun_direction_at_ric(target_state_eci=target_state_eci, time_s=time_s)
        if self.allowed_center_mode in {"sun", "toward_sun"}:
            return self.sun_direction_at_ric(target_state_eci=target_state_eci, time_s=time_s)
        return np.array(self.allowed_center_ric, dtype=float).reshape(3)

    def sun_angles_deg(
        self,
        relative_ric_positions_km: np.ndarray,
        *,
        target_state_eci: np.ndarray | None = None,
        time_s: float | None = None,
    ) -> np.ndarray:
        dirs, valid = _unit_direction_rows(relative_ric_positions_km)
        sun_dir = self.sun_direction_at_ric(target_state_eci=target_state_eci, time_s=time_s)
        dots = np.clip(dirs @ sun_dir.reshape(3), -1.0, 1.0)
        angles = np.rad2deg(np.arccos(dots))
        angles[~valid] = np.nan
        return angles

    def samples_satisfying_constraint(
        self,
        relative_ric_positions_km: np.ndarray,
        *,
        target_state_eci: np.ndarray | None = None,
        time_s: float | None = None,
    ) -> np.ndarray:
        pos = np.array(relative_ric_positions_km, dtype=float)
        if pos.ndim == 1:
            pos = pos.reshape(1, -1)
        if pos.shape[1] < 3:
            raise ValueError("relative_ric_positions_km must contain R, I, and C components.")
        dirs, valid = _unit_direction_rows(pos[:, :3])
        center = self.allowed_center_at_ric(target_state_eci=target_state_eci, time_s=time_s)
        off_axis_deg = np.rad2deg(np.arccos(np.clip(dirs @ center, -1.0, 1.0)))
        ok = valid & (off_axis_deg <= float(self.allowed_half_angle_deg))
        ranges = np.linalg.norm(pos[:, :3], axis=1)
        if self.min_range_km is not None:
            ok &= ranges >= float(self.min_range_km)
        if self.max_range_km is not None:
            ok &= ranges <= float(self.max_range_km)
        return ok


@dataclass(frozen=True)
class RequiredPhaseBurnConfig:
    name: str
    axis: str
    radial_abs_km: float
    radial_tolerance_km: float
    max_abs_intrack_km: float
    threshold_km_s2: float = 1.0e-10
    min_component_fraction: float = _BURN_AXIS_MIN_COMPONENT_FRACTION

    @property
    def label(self) -> str:
        return self.name or f"{_BURN_AXIS_LABEL[self.axis]} phase burn"


@dataclass(frozen=True)
class GuidedTutorialBurnConfig:
    name: str
    axis: str
    sign: int
    delta_v_m_s: float
    label: str = ""
    hint: str = ""

    @property
    def display_label(self) -> str:
        if self.label:
            return self.label
        prefix = "+" if self.sign >= 0 else "-"
        return f"{prefix}{_BURN_AXIS_SHORT_LABEL[self.axis]} burn"


@dataclass(frozen=True)
class GuidedTutorialSpeedStepConfig:
    name: str = "speed_multiplier"
    after_burn_name: str = ""
    target_speed_multiplier: float = 10.0
    label: str = "Speed 10x"
    hint: str = ""


@dataclass(frozen=True)
class RPOTrainingConfig:
    enabled: bool = False
    scenario_id: str = ""
    level_name: str = ""
    learning_goal: str = ""
    relative_frame: str = "ric"
    target_object_id: str = "target"
    chaser_object_id: str = "chaser"
    target_reference_object_id: str = "target_reference"
    keepout_radius_km: float | None = None
    goal_range_km: float | None = None
    goal_range_tolerance_km: float | None = None
    goal_radius_km: float | None = None
    goal_relative_ric_km: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    goal_nmt_radial_amplitude_km: float | None = None
    goal_nmt_cross_track_amplitude_km: float = 0.0
    goal_nmt_cross_track_phase_deg: float = 0.0
    goal_nmt_center_ric_km: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    goal_nmt_tolerance_km: float | None = None
    goal_nmt_element_tolerance_km: float | None = None
    goal_nmt_velocity_tolerance_km_s: float | None = None
    max_cross_track_amplitude_km: float | None = None
    forbidden_regions: tuple[ForbiddenRegionConfig, ...] = ()
    approach_gates: tuple[ApproachGateConfig, ...] = ()
    inspection_gates: tuple[InspectionGateConfig, ...] = ()
    sun_angle_constraints: tuple[SunAngleConstraintConfig, ...] = ()
    max_time_s: float | None = None
    max_goal_speed_km_s: float | None = None
    hard_speed_limit_radius_km: float | None = None
    hard_speed_limit_km_s: float | None = None
    max_delta_v_m_s: float | None = None
    max_target_delta_v_m_s: float | None = None
    max_target_reference_range_km: float | None = None
    fail_on_delta_v_budget: bool = True
    coast_chaser_after_delta_v_budget: bool = False
    survival_goal: bool = False
    sandbox_mode: bool = False
    required_burn_axes: tuple[str, ...] = ()
    required_burn_axis_threshold_km_s2: float = 1.0e-10
    required_burn_axis_min_component_fraction: float = _BURN_AXIS_MIN_COMPONENT_FRACTION
    required_phase_burns: tuple[RequiredPhaseBurnConfig, ...] = ()
    require_speed_multiplier_change: bool = False
    required_coast_after_burn_s: float | None = None
    axis_descriptions: dict[str, str] = field(default_factory=dict)
    tutorial_stage_hints: dict[str, str] = field(default_factory=dict)
    guided_tutorial_burns: tuple[GuidedTutorialBurnConfig, ...] = ()
    guided_tutorial_speed_step: GuidedTutorialSpeedStepConfig | None = None

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> RPOTrainingConfig:
        game_cfg = dict(metadata.get("game", {}) or {})
        raw = dict(game_cfg.get("training", {}) or {})
        if not raw:
            return cls(enabled=False)
        goal = np.array(raw.get("goal_relative_ric_km", [0.0, 0.0, 0.0]), dtype=float).reshape(-1)
        if goal.size != 3:
            raise ValueError("metadata.game.training.goal_relative_ric_km must be length 3.")
        nmt_center = np.array(raw.get("goal_nmt_center_ric_km", [0.0, 0.0, 0.0]), dtype=float).reshape(-1)
        if nmt_center.size != 3:
            raise ValueError("metadata.game.training.goal_nmt_center_ric_km must be length 3.")
        return cls(
            enabled=bool(raw.get("enabled", True)),
            scenario_id=str(raw.get("scenario_id", "") or ""),
            level_name=str(game_cfg.get("level_name", "") or ""),
            learning_goal=str(raw.get("learning_goal", "") or ""),
            relative_frame=str(raw.get("relative_frame", game_cfg.get("relative_frame", "ric")) or "ric")
            .strip()
            .lower(),
            target_object_id=str(raw.get("target_object_id", game_cfg.get("target_object_id", "target")) or "target"),
            chaser_object_id=str(raw.get("chaser_object_id", game_cfg.get("chaser_object_id", "chaser")) or "chaser"),
            target_reference_object_id=str(
                raw.get("target_reference_object_id", game_cfg.get("ric_reference_object_id", "target_reference"))
                or "target_reference"
            ),
            keepout_radius_km=_optional_float(raw.get("keepout_radius_km")),
            goal_range_km=_optional_float(raw.get("goal_range_km")),
            goal_range_tolerance_km=_optional_float(raw.get("goal_range_tolerance_km")),
            goal_radius_km=_optional_float(raw.get("goal_radius_km")),
            goal_relative_ric_km=goal.astype(float),
            goal_nmt_radial_amplitude_km=_optional_float(raw.get("goal_nmt_radial_amplitude_km")),
            goal_nmt_cross_track_amplitude_km=float(raw.get("goal_nmt_cross_track_amplitude_km", 0.0) or 0.0),
            goal_nmt_cross_track_phase_deg=float(raw.get("goal_nmt_cross_track_phase_deg", 0.0) or 0.0),
            goal_nmt_center_ric_km=nmt_center.astype(float),
            goal_nmt_tolerance_km=_optional_float(raw.get("goal_nmt_tolerance_km")),
            goal_nmt_element_tolerance_km=_optional_float(
                raw.get("goal_nmt_element_tolerance_km", raw.get("goal_nmt_tolerance_km"))
            ),
            goal_nmt_velocity_tolerance_km_s=_optional_float(raw.get("goal_nmt_velocity_tolerance_km_s")),
            max_cross_track_amplitude_km=_optional_float(raw.get("max_cross_track_amplitude_km")),
            forbidden_regions=_forbidden_regions_from_metadata(raw.get("forbidden_regions")),
            approach_gates=_approach_gates_from_metadata(raw.get("approach_gates")),
            inspection_gates=_inspection_gates_from_metadata(raw.get("inspection_gates")),
            sun_angle_constraints=_sun_angle_constraints_from_metadata(raw.get("sun_angle_constraints")),
            max_time_s=_optional_float(raw.get("max_time_s")),
            max_goal_speed_km_s=_optional_float(raw.get("max_goal_speed_km_s")),
            hard_speed_limit_radius_km=_optional_float(raw.get("hard_speed_limit_radius_km")),
            hard_speed_limit_km_s=_optional_float(raw.get("hard_speed_limit_km_s")),
            max_delta_v_m_s=_optional_float(raw.get("max_delta_v_m_s")),
            max_target_delta_v_m_s=_optional_float(raw.get("max_target_delta_v_m_s")),
            max_target_reference_range_km=_optional_float(raw.get("max_target_reference_range_km")),
            fail_on_delta_v_budget=bool(raw.get("fail_on_delta_v_budget", True)),
            coast_chaser_after_delta_v_budget=bool(raw.get("coast_chaser_after_delta_v_budget", False)),
            survival_goal=bool(raw.get("survival_goal", False)),
            sandbox_mode=bool(raw.get("sandbox_mode", False)),
            required_burn_axes=_burn_axes_from_metadata(raw.get("required_burn_axes")),
            required_burn_axis_threshold_km_s2=float(raw.get("required_burn_axis_threshold_km_s2", 1.0e-10)),
            required_burn_axis_min_component_fraction=float(
                raw.get("required_burn_axis_min_component_fraction", _BURN_AXIS_MIN_COMPONENT_FRACTION)
            ),
            required_phase_burns=_required_phase_burns_from_metadata(raw.get("required_phase_burns")),
            require_speed_multiplier_change=bool(raw.get("require_speed_multiplier_change", False)),
            required_coast_after_burn_s=_optional_float(raw.get("required_coast_after_burn_s")),
            axis_descriptions=_axis_descriptions_from_metadata(raw.get("axis_descriptions")),
            tutorial_stage_hints=_string_mapping_from_metadata(
                raw.get("tutorial_stage_hints"), "metadata.game.training.tutorial_stage_hints"
            ),
            guided_tutorial_burns=_guided_tutorial_burns_from_metadata(raw.get("guided_tutorial_burns")),
            guided_tutorial_speed_step=_guided_tutorial_speed_step_from_metadata(
                raw.get("guided_tutorial_speed_step")
            ),
        )


def training_config_for_game_mode(config: RPOTrainingConfig, *, game_mode: str) -> RPOTrainingConfig:
    if str(game_mode or "").strip().lower() != "operator":
        return config
    if config.scenario_id in OPERATOR_RELAXED_REQUIRED_BURN_AXIS_SCENARIO_IDS:
        return replace(config, required_burn_axes=())
    return config


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


class RPOTrainingTracker:
    def __init__(self, config: RPOTrainingConfig):
        self.config = config
        self.t_s: list[float] = []
        self.rel_ric_hist: list[np.ndarray] = []
        self.thrust_hist: list[np.ndarray] = []
        self.thrust_ric_hist: list[np.ndarray] = []
        self.target_thrust_hist: list[np.ndarray] = []
        self.target_reference_rel_hist: list[np.ndarray] = []
        self.mean_motion_hist: list[float] = []
        self._speed_multiplier_changed = False
        self._speed_multiplier_change_sample_idx: int | None = None
        self._score_cache: RPOTrainingScore | None = None
        self._inspection_gate_names: list[str] = []
        self._inspection_gate_completed_idx: int | None = None
        self._hard_speed_limit_violation = False
        self._forbidden_region_names: set[str] = set()
        self._burn_axis_first_indices: dict[str, int] = {}
        self._phase_burn_first_indices: dict[str, int] = {}
        self._guided_tutorial_burn_names: list[str] = []
        self._guided_tutorial_speed_complete = False
        self._sun_angle_ok_by_constraint: dict[str, list[bool]] = {}
        self._sun_angle_deg_by_constraint: dict[str, list[float]] = {}
        self._history_capacity = 0
        self._history_count = 0
        self._t_array = np.zeros(0, dtype=float)
        self._rel_array = np.zeros((0, 6), dtype=float)
        self._thrust_array = np.zeros((0, 3), dtype=float)
        self._thrust_ric_array = np.zeros((0, 3), dtype=float)
        self._target_thrust_array = np.zeros((0, 3), dtype=float)
        self._mean_motion_array = np.zeros(0, dtype=float)
        self._range_array = np.zeros(0, dtype=float)
        self._speed_array = np.zeros(0, dtype=float)
        self._goal_error_array = np.zeros(0, dtype=float)
        self._delta_v_interval_km_s_array = np.zeros(0, dtype=float)
        self._target_delta_v_interval_km_s_array = np.zeros(0, dtype=float)
        self._nmt_radial_amplitude_array = np.zeros(0, dtype=float)
        self._nmt_cross_track_amplitude_array = np.zeros(0, dtype=float)
        self._nmt_radial_amplitude_error_array = np.zeros(0, dtype=float)
        self._nmt_cross_track_amplitude_error_array = np.zeros(0, dtype=float)
        self._nmt_drift_velocity_error_array = np.zeros(0, dtype=float)
        self._nmt_element_goal_error_array = np.zeros(0, dtype=float)

    def clear(self, *, reset_guided_tutorial_progress: bool = True) -> None:
        self.t_s.clear()
        self.rel_ric_hist.clear()
        self.thrust_hist.clear()
        self.thrust_ric_hist.clear()
        self.target_thrust_hist.clear()
        self.target_reference_rel_hist.clear()
        self.mean_motion_hist.clear()
        self._speed_multiplier_changed = False
        self._speed_multiplier_change_sample_idx = None
        self._score_cache = None
        self._inspection_gate_names.clear()
        self._inspection_gate_completed_idx = None
        self._hard_speed_limit_violation = False
        self._forbidden_region_names.clear()
        self._burn_axis_first_indices.clear()
        self._phase_burn_first_indices.clear()
        if reset_guided_tutorial_progress:
            self._guided_tutorial_burn_names.clear()
            self._guided_tutorial_speed_complete = False
        self._sun_angle_ok_by_constraint.clear()
        self._sun_angle_deg_by_constraint.clear()
        self._history_count = 0

    def mark_guided_tutorial_burn_complete(self, name: str) -> None:
        stage_name = str(name or "").strip()
        if stage_name and stage_name not in self._guided_tutorial_burn_names:
            self._guided_tutorial_burn_names.append(stage_name)
            self._score_cache = None

    def guided_tutorial_burns_satisfied(self) -> tuple[str, ...]:
        configured = {stage.name for stage in self.config.guided_tutorial_burns}
        return tuple(name for name in self._guided_tutorial_burn_names if name in configured)

    def mark_guided_tutorial_speed_complete(self) -> None:
        if not self._guided_tutorial_speed_complete:
            self._guided_tutorial_speed_complete = True
            self._score_cache = None

    def guided_tutorial_speed_satisfied(self) -> bool:
        return bool(self._guided_tutorial_speed_complete or self.config.guided_tutorial_speed_step is None)

    def record(self, snapshot: SimulationSnapshot) -> None:
        if not self.config.enabled:
            return
        target = snapshot.truth.get(self.config.target_object_id)
        chaser = snapshot.truth.get(self.config.chaser_object_id)
        if target is None or chaser is None:
            return
        rel = relative_state_from_arrays(target, chaser, frame=self.config.relative_frame)
        self.t_s.append(float(snapshot.time_s))
        self.rel_ric_hist.append(rel)
        reference = snapshot.truth.get(self.config.target_reference_object_id)
        if reference is not None:
            target_reference_rel = relative_state_from_arrays(reference, target, frame=self.config.relative_frame)
        else:
            target_reference_rel = np.full(6, np.nan, dtype=float)
        self.target_reference_rel_hist.append(target_reference_rel)
        target_arr = np.array(target, dtype=float).reshape(-1)
        n = float("nan")
        frame_key = _relative_frame_key(self.config.relative_frame)
        if frame_key == "cislunar":
            n = EARTH_MOON_MEAN_MOTION_RAD_S
        elif frame_key == "moon_ric" and target_arr.size >= 6:
            target_moon = target_arr[:6] - cr3bp_moon_state_km_s()
            r_norm = float(np.linalg.norm(target_moon[:3]))
            h_norm = float(np.linalg.norm(np.cross(target_moon[:3], target_moon[3:6])))
            if np.isfinite(r_norm) and r_norm > 0.0:
                n = h_norm / (r_norm**2)
        elif target_arr.size >= 3:
            r_norm = float(np.linalg.norm(target_arr[:3]))
            if np.isfinite(r_norm) and r_norm > 0.0:
                n = float(np.sqrt(EARTH_MU_KM3_S2 / (r_norm**3)))
        self.mean_motion_hist.append(n)
        thrust = snapshot.applied_thrust.get(self.config.chaser_object_id, np.zeros(3, dtype=float))
        thrust_eci = np.array(thrust, dtype=float).reshape(3)
        self.thrust_hist.append(thrust_eci)
        if frame_key == "cislunar":
            self.thrust_ric_hist.append(thrust_eci)
        elif frame_key == "moon_ric" and target_arr.size >= 6:
            target_moon = target_arr[:6] - cr3bp_moon_state_km_s()
            c_ir = ric_dcm_ir_from_rv(target_moon[:3], target_moon[3:6])
            self.thrust_ric_hist.append(c_ir.T @ thrust_eci)
        elif target_arr.size >= 6:
            c_ir = ric_dcm_ir_from_rv(target_arr[:3], target_arr[3:6])
            self.thrust_ric_hist.append(c_ir.T @ thrust_eci)
        else:
            self.thrust_ric_hist.append(np.zeros(3, dtype=float))
        target_thrust = snapshot.applied_thrust.get(self.config.target_object_id, np.zeros(3, dtype=float))
        target_thrust_eci = np.array(target_thrust, dtype=float).reshape(3)
        self.target_thrust_hist.append(target_thrust_eci)
        self._append_history_arrays(
            t_s=float(snapshot.time_s),
            rel=rel,
            thrust=thrust_eci,
            thrust_ric=self.thrust_ric_hist[-1],
            target_thrust=target_thrust_eci,
            mean_motion_rad_s=n,
            target_state=target_arr,
            chaser_state=np.array(chaser, dtype=float).reshape(-1),
        )
        self._record_burn_requirement_sample(rel, self.thrust_ric_hist[-1])
        self._record_hard_speed_limit_sample(rel)
        self._record_forbidden_region_sample(rel)
        self._record_sun_angle_sample(rel, target_arr, float(snapshot.time_s))
        self._record_inspection_gate_sample(rel, target_arr, float(snapshot.time_s))
        self._score_cache = None

    def record_speed_multiplier_change(self) -> None:
        self._speed_multiplier_changed = True
        self._speed_multiplier_change_sample_idx = max(len(self.t_s) - 1, 0) if self.t_s else 0
        self._score_cache = None

    def _record_burn_requirement_sample(self, rel: np.ndarray, thrust_ric: np.ndarray) -> None:
        if not self.config.required_burn_axes and not self.config.required_phase_burns:
            return
        sample_idx = len(self.rel_ric_hist) - 1
        thrust = np.array(thrust_ric, dtype=float).reshape(3)
        thrust_norm = float(np.linalg.norm(thrust))
        if thrust_norm <= 0.0:
            return
        for axis in self.config.required_burn_axes:
            if axis in self._burn_axis_first_indices:
                continue
            axis_idx = _BURN_AXIS_INDEX[axis]
            threshold = max(float(self.config.required_burn_axis_threshold_km_s2), 0.0)
            min_fraction = float(np.clip(self.config.required_burn_axis_min_component_fraction, 0.0, 1.0))
            component = abs(float(thrust[axis_idx]))
            if component > threshold and component >= min_fraction * thrust_norm:
                self._burn_axis_first_indices[axis] = sample_idx
        if not self.config.required_phase_burns:
            return
        rel_arr = np.array(rel, dtype=float).reshape(-1)
        for phase_burn in self.config.required_phase_burns:
            if phase_burn.name in self._phase_burn_first_indices:
                continue
            axis_idx = _BURN_AXIS_INDEX[phase_burn.axis]
            threshold = max(float(phase_burn.threshold_km_s2), 0.0)
            min_fraction = float(np.clip(phase_burn.min_component_fraction, 0.0, 1.0))
            component = abs(float(thrust[axis_idx]))
            radial_error = abs(abs(float(rel_arr[0])) - float(phase_burn.radial_abs_km))
            if (
                component > threshold
                and component >= min_fraction * thrust_norm
                and radial_error <= float(phase_burn.radial_tolerance_km)
                and abs(float(rel_arr[1])) <= float(phase_burn.max_abs_intrack_km)
            ):
                self._phase_burn_first_indices[phase_burn.name] = sample_idx

    def _burn_axis_first_sample_indices(self) -> dict[str, int]:
        return dict(self._burn_axis_first_indices)

    def _burn_axes_satisfied(self) -> tuple[str, ...]:
        first_indices = self._burn_axis_first_sample_indices()
        return tuple(axis for axis in self.config.required_burn_axes if axis in first_indices)

    def _phase_burn_first_sample_indices(self) -> dict[str, int]:
        return dict(self._phase_burn_first_indices)

    def _phase_burns_satisfied(self) -> tuple[str, ...]:
        first_indices = self._phase_burn_first_sample_indices()
        return tuple(
            phase_burn.name for phase_burn in self.config.required_phase_burns if phase_burn.name in first_indices
        )

    def _record_hard_speed_limit_sample(self, rel: np.ndarray) -> None:
        if self._hard_speed_limit_violation:
            return
        if self.config.hard_speed_limit_radius_km is None or self.config.hard_speed_limit_km_s is None:
            return
        current = np.array(rel, dtype=float).reshape(6)
        previous = self.rel_ric_hist[-2] if len(self.rel_ric_hist) >= 2 else None
        self._hard_speed_limit_violation = _hard_speed_limit_sample_violated(
            previous,
            current,
            radius_km=float(self.config.hard_speed_limit_radius_km),
            speed_limit_km_s=float(self.config.hard_speed_limit_km_s),
        )

    def _record_forbidden_region_sample(self, rel: np.ndarray) -> None:
        if len(self._forbidden_region_names) >= len(self.config.forbidden_regions):
            return
        current = np.asarray(rel, dtype=float).reshape(6)[:3]
        previous = self.rel_ric_hist[-2][:3] if len(self.rel_ric_hist) >= 2 else None
        for region in self.config.forbidden_regions:
            if region.name in self._forbidden_region_names:
                continue
            current_inside = bool(region.contains_positions(current)[0])
            segment_crossing = bool(previous is not None and region.intersects_segment(previous, current))
            if current_inside or segment_crossing:
                self._forbidden_region_names.add(region.name)

    def _append_history_arrays(
        self,
        *,
        t_s: float,
        rel: np.ndarray,
        thrust: np.ndarray,
        thrust_ric: np.ndarray,
        target_thrust: np.ndarray,
        mean_motion_rad_s: float,
        target_state: np.ndarray | None = None,
        chaser_state: np.ndarray | None = None,
    ) -> None:
        idx = int(self._history_count)
        if idx >= int(self._history_capacity):
            self._grow_history_arrays(max(idx + 1, 64 if self._history_capacity <= 0 else self._history_capacity * 2))
        self._t_array[idx] = float(t_s)
        rel_arr = np.asarray(rel, dtype=float).reshape(6)
        self._rel_array[idx, :] = rel_arr
        self._thrust_array[idx, :] = np.array(thrust, dtype=float).reshape(3)
        self._thrust_ric_array[idx, :] = np.array(thrust_ric, dtype=float).reshape(3)
        self._target_thrust_array[idx, :] = np.array(target_thrust, dtype=float).reshape(3)
        self._mean_motion_array[idx] = float(mean_motion_rad_s)
        self._delta_v_interval_km_s_array[idx] = self._delta_v_interval_km_s(idx, thrust)
        self._target_delta_v_interval_km_s_array[idx] = self._delta_v_interval_km_s(idx, target_thrust)
        self._range_array[idx] = float(np.sqrt(np.sum(rel_arr[:3] * rel_arr[:3])))
        self._speed_array[idx] = float(np.sqrt(np.sum(rel_arr[3:6] * rel_arr[3:6])))
        self._append_nmt_element_arrays(
            idx=idx,
            rel=rel,
            mean_motion_rad_s=mean_motion_rad_s,
            target_state=target_state,
            chaser_state=chaser_state,
        )
        self._goal_error_array[idx] = self._goal_error_value(idx, rel_arr)
        self._history_count = idx + 1

    def _delta_v_interval_km_s(self, idx: int, thrust_km_s2: np.ndarray) -> float:
        if idx <= 0:
            return 0.0
        dt_s = float(self._t_array[idx] - self._t_array[idx - 1])
        thrust = np.asarray(thrust_km_s2, dtype=float).reshape(3)
        accel_km_s2 = float(np.sqrt(np.sum(thrust * thrust)))
        if not np.isfinite(accel_km_s2) or not np.isfinite(dt_s) or dt_s <= 0.0:
            return float("nan")
        return accel_km_s2 * dt_s

    def _goal_error_value(self, idx: int, rel: np.ndarray) -> float:
        position = np.asarray(rel, dtype=float).reshape(6)[:3]
        if self.config.goal_nmt_radial_amplitude_km is not None:
            if self.config.goal_nmt_tolerance_km is not None:
                return float(
                    nmt_position_error_km(
                        position,
                        radial_amplitude_km=float(self.config.goal_nmt_radial_amplitude_km),
                        cross_track_amplitude_km=float(self.config.goal_nmt_cross_track_amplitude_km),
                        cross_track_phase_deg=float(self.config.goal_nmt_cross_track_phase_deg),
                        center_ric_km=self.config.goal_nmt_center_ric_km,
                    )[0]
                )
            return float(self._nmt_element_goal_error_array[idx])
        if self.config.goal_range_km is not None:
            current_range = float(self._range_array[idx])
            if self.config.goal_range_tolerance_km is None:
                return max(current_range - float(self.config.goal_range_km), 0.0)
            return abs(current_range - float(self.config.goal_range_km))
        if self.config.inspection_gates:
            gate_centers = np.vstack([gate.center_ric_km for gate in self.config.inspection_gates])
            return float(np.min(np.linalg.norm(position.reshape(1, 3) - gate_centers, axis=1)))
        delta = position - self.config.goal_relative_ric_km.reshape(3)
        return float(np.sqrt(np.sum(delta * delta)))

    def _grow_history_arrays(self, capacity: int) -> None:
        new_capacity = int(max(capacity, 1))
        old_count = int(self._history_count)

        def grow_1d(current: np.ndarray, *, fill_value: float = 0.0) -> np.ndarray:
            out = np.full(new_capacity, fill_value, dtype=float)
            if old_count:
                out[:old_count] = current[:old_count]
            return out

        def grow_2d(current: np.ndarray, width: int) -> np.ndarray:
            out = np.zeros((new_capacity, width), dtype=float)
            if old_count:
                out[:old_count, :] = current[:old_count, :]
            return out

        self._t_array = grow_1d(self._t_array)
        self._rel_array = grow_2d(self._rel_array, 6)
        self._thrust_array = grow_2d(self._thrust_array, 3)
        self._thrust_ric_array = grow_2d(self._thrust_ric_array, 3)
        self._target_thrust_array = grow_2d(self._target_thrust_array, 3)
        self._mean_motion_array = grow_1d(self._mean_motion_array)
        self._range_array = grow_1d(self._range_array)
        self._speed_array = grow_1d(self._speed_array)
        self._goal_error_array = grow_1d(self._goal_error_array, fill_value=float("nan"))
        self._delta_v_interval_km_s_array = grow_1d(self._delta_v_interval_km_s_array)
        self._target_delta_v_interval_km_s_array = grow_1d(self._target_delta_v_interval_km_s_array)
        self._nmt_radial_amplitude_array = grow_1d(self._nmt_radial_amplitude_array, fill_value=float("nan"))
        self._nmt_cross_track_amplitude_array = grow_1d(
            self._nmt_cross_track_amplitude_array, fill_value=float("nan")
        )
        self._nmt_radial_amplitude_error_array = grow_1d(
            self._nmt_radial_amplitude_error_array, fill_value=float("nan")
        )
        self._nmt_cross_track_amplitude_error_array = grow_1d(
            self._nmt_cross_track_amplitude_error_array, fill_value=float("nan")
        )
        self._nmt_drift_velocity_error_array = grow_1d(
            self._nmt_drift_velocity_error_array, fill_value=float("nan")
        )
        self._nmt_element_goal_error_array = grow_1d(
            self._nmt_element_goal_error_array, fill_value=float("nan")
        )
        self._history_capacity = new_capacity

    def _append_nmt_element_arrays(
        self,
        *,
        idx: int,
        rel: np.ndarray,
        mean_motion_rad_s: float,
        target_state: np.ndarray | None,
        chaser_state: np.ndarray | None,
    ) -> None:
        if self.config.goal_nmt_radial_amplitude_km is None and self.config.max_cross_track_amplitude_km is None:
            return
        drift_error = _semimajor_axis_drift_velocity_error_km_s(target_state, chaser_state)
        values = _nmt_element_error_values(
            rel,
            mean_motion_rad_s=float(mean_motion_rad_s),
            radial_amplitude_km=float(self.config.goal_nmt_radial_amplitude_km or 0.0),
            cross_track_amplitude_km=float(self.config.goal_nmt_cross_track_amplitude_km),
            center_ric_km=self.config.goal_nmt_center_ric_km,
            drift_velocity_error_km_s=drift_error,
        )
        self._nmt_radial_amplitude_array[idx] = values["radial_amplitude_km"]
        self._nmt_cross_track_amplitude_array[idx] = values["cross_track_amplitude_km"]
        self._nmt_radial_amplitude_error_array[idx] = values["radial_amplitude_error_km"]
        self._nmt_cross_track_amplitude_error_array[idx] = values["cross_track_amplitude_error_km"]
        self._nmt_drift_velocity_error_array[idx] = values["drift_velocity_error_km_s"]
        self._nmt_element_goal_error_array[idx] = _nmt_element_goal_error_km(
            radial_amplitude_error_km=values["radial_amplitude_error_km"],
            cross_track_amplitude_error_km=values["cross_track_amplitude_error_km"],
            include_radial=self.config.goal_nmt_radial_amplitude_km is not None,
            include_cross_track=(
                self.config.goal_nmt_radial_amplitude_km is not None
                or self.config.max_cross_track_amplitude_km is not None
            ),
        )

    def _nmt_element_error_arrays(self, rel: np.ndarray, n_hist: np.ndarray) -> dict[str, np.ndarray]:
        if self._history_arrays_available() and int(self._history_count) >= int(rel.shape[0]):
            count = int(rel.shape[0])
            return {
                "radial_amplitude_km": self._nmt_radial_amplitude_array[:count],
                "cross_track_amplitude_km": self._nmt_cross_track_amplitude_array[:count],
                "radial_amplitude_error_km": self._nmt_radial_amplitude_error_array[:count],
                "cross_track_amplitude_error_km": self._nmt_cross_track_amplitude_error_array[:count],
                "drift_velocity_error_km_s": self._nmt_drift_velocity_error_array[:count],
            }
        return nmt_element_errors(
            rel,
            mean_motion_rad_s=n_hist[: rel.shape[0]],
            radial_amplitude_km=float(self.config.goal_nmt_radial_amplitude_km or 0.0),
            cross_track_amplitude_km=float(self.config.goal_nmt_cross_track_amplitude_km),
            center_ric_km=self.config.goal_nmt_center_ric_km,
        )

    def _nmt_element_goal_error_array_for(self, element_errors: dict[str, np.ndarray]) -> np.ndarray:
        if self._history_arrays_available():
            return self._nmt_element_goal_error_array[: int(self._history_count)]
        return _nmt_element_goal_error_array(
            element_errors,
            include_radial=self.config.goal_nmt_radial_amplitude_km is not None,
            include_cross_track=(
                self.config.goal_nmt_radial_amplitude_km is not None
                or self.config.max_cross_track_amplitude_km is not None
            ),
        )

    def _history_arrays_available(self) -> bool:
        return int(self._history_count) == len(self.rel_ric_hist) and int(self._history_count) > 0

    def _history_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._history_arrays_available():
            count = int(self._history_count)
            return (
                self._rel_array[:count],
                self._t_array[:count],
                self._thrust_array[:count],
                self._target_thrust_array[:count],
                self._mean_motion_array[:count],
            )
        rel = np.vstack(self.rel_ric_hist)
        t = np.array(self.t_s, dtype=float)
        thrust = np.vstack(self.thrust_hist) if self.thrust_hist else np.zeros((rel.shape[0], 3), dtype=float)
        target_thrust = (
            np.vstack(self.target_thrust_hist) if self.target_thrust_hist else np.zeros((rel.shape[0], 3), dtype=float)
        )
        n_hist = np.array(self.mean_motion_hist, dtype=float).reshape(-1)
        return rel, t, thrust, target_thrust, n_hist

    def replay_history(self) -> dict[str, np.ndarray]:
        if int(self._history_count) > 0:
            count = int(self._history_count)
            rel = self._rel_array[:count, :]
            t = self._t_array[:count]
            thrust_ric = self._thrust_ric_array[:count, :]
            target_thrust = self._target_thrust_array[:count, :]
        else:
            rel = np.vstack(self.rel_ric_hist) if self.rel_ric_hist else np.zeros((0, 6), dtype=float)
            t = np.array(self.t_s, dtype=float).reshape(-1)
            thrust_ric = (
                np.vstack(self.thrust_ric_hist) if self.thrust_ric_hist else np.zeros((rel.shape[0], 3), dtype=float)
            )
            target_thrust = (
                np.vstack(self.target_thrust_hist)
                if self.target_thrust_hist
                else np.zeros((rel.shape[0], 3), dtype=float)
            )
            count = int(min(rel.shape[0], t.size, thrust_ric.shape[0], target_thrust.shape[0]))
        return {
            "time_s": t[:count].copy(),
            "relative_ric": rel[:count, :].copy(),
            "chaser_thrust_ric_km_s2": thrust_ric.copy(),
            "target_thrust_eci_km_s2": target_thrust[:count, :].copy(),
        }

    def _missing_tutorial_requirements(self) -> tuple[str, ...]:
        missing: list[str] = []
        satisfied = set(self._burn_axes_satisfied())
        for axis in self.config.required_burn_axes:
            if axis not in satisfied:
                missing.append(f"{_BURN_AXIS_LABEL[axis]} burn")
        phase_satisfied = set(self._phase_burns_satisfied())
        for phase_burn in self.config.required_phase_burns:
            if phase_burn.name not in phase_satisfied:
                missing.append(phase_burn.label)
        if self.config.require_speed_multiplier_change and not self._speed_multiplier_changed:
            missing.append("speed multiplier change")
        if self.config.required_coast_after_burn_s is not None:
            coast_satisfied, _, _ = self._coast_after_burn_status()
            if not coast_satisfied:
                missing.append("coast after a burn")
        if self.config.guided_tutorial_speed_step is not None and not self.guided_tutorial_speed_satisfied():
            missing.append(self.config.guided_tutorial_speed_step.label)
        return tuple(missing)

    def _coast_after_burn_status(self) -> tuple[bool, int | None, float]:
        required_s = self.config.required_coast_after_burn_s
        if required_s is None:
            return True, None, 0.0
        threshold = float(self.config.required_burn_axis_threshold_km_s2)
        if len(self.t_s) < 2 or len(self.thrust_ric_hist) < 2:
            return False, None, 0.0
        t = np.array(self.t_s, dtype=float).reshape(-1)
        thrust = np.vstack(self.thrust_ric_hist)
        n = min(t.size, thrust.shape[0])
        if n < 2:
            return False, None, 0.0
        active = np.linalg.norm(thrust[:n], axis=1) > threshold
        active_idx = np.flatnonzero(active)
        if active_idx.size == 0:
            return False, None, 0.0
        coast_s = 0.0
        best_s = 0.0
        for idx in range(int(active_idx[0]) + 1, n - 1):
            dt = float(t[idx + 1] - t[idx])
            if not np.isfinite(dt) or dt <= 0.0:
                continue
            if active[idx]:
                coast_s = 0.0
                continue
            coast_s += dt
            best_s = max(best_s, coast_s)
            if coast_s >= float(required_s):
                return True, idx + 1, best_s
        return False, None, best_s

    def _record_inspection_gate_sample(
        self,
        rel: np.ndarray,
        target_state_eci: np.ndarray | None = None,
        time_s: float | None = None,
    ) -> None:
        gates = self.config.inspection_gates
        if not gates or len(self._inspection_gate_names) >= len(gates):
            return
        if not self._sun_constraints_satisfied_at(rel[:3], target_state_eci=target_state_eci, time_s=time_s):
            return
        sample_idx = len(self.rel_ric_hist) - 1
        previous = self.rel_ric_hist[sample_idx - 1] if sample_idx > 0 else None
        satisfied = set(self._inspection_gate_names)
        for gate in gates:
            if gate.name in satisfied:
                continue
            current_hits_gate = bool(gate.samples_satisfying_gate(rel.reshape(1, -1))[0])
            segment_hits_gate = bool(previous is not None and gate.segment_satisfies_gate(previous, rel))
            if current_hits_gate or segment_hits_gate:
                self._inspection_gate_names.append(gate.name)
                satisfied.add(gate.name)
                if len(self._inspection_gate_names) >= len(gates):
                    self._inspection_gate_completed_idx = sample_idx
                    break

    def _sun_constraints_satisfied_at(
        self,
        position_ric_km: np.ndarray,
        *,
        target_state_eci: np.ndarray | None = None,
        time_s: float | None = None,
    ) -> bool:
        if not self.config.sun_angle_constraints:
            return True
        position = np.array(position_ric_km, dtype=float).reshape(1, 3)
        for constraint in self.config.sun_angle_constraints:
            if not bool(
                constraint.samples_satisfying_constraint(
                    position,
                    target_state_eci=target_state_eci,
                    time_s=time_s,
                )[0]
            ):
                return False
        return True

    def _inspection_gate_status(self) -> dict[str, Any]:
        return {
            "satisfied": tuple(self._inspection_gate_names),
            "completed_idx": self._inspection_gate_completed_idx,
        }

    def _record_sun_angle_sample(self, rel: np.ndarray, target_state_eci: np.ndarray, time_s: float) -> None:
        if not self.config.sun_angle_constraints:
            return
        position = np.array(rel, dtype=float).reshape(6)[:3].reshape(1, 3)
        for constraint in self.config.sun_angle_constraints:
            ok = bool(
                constraint.samples_satisfying_constraint(
                    position,
                    target_state_eci=target_state_eci,
                    time_s=float(time_s),
                )[0]
            )
            angle = float(
                constraint.sun_angles_deg(
                    position,
                    target_state_eci=target_state_eci,
                    time_s=float(time_s),
                )[0]
            )
            self._sun_angle_ok_by_constraint.setdefault(constraint.name, []).append(ok)
            self._sun_angle_deg_by_constraint.setdefault(constraint.name, []).append(angle)

    def _sun_angle_status_arrays(self, rel: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        ok_by_name: dict[str, np.ndarray] = {}
        angle_by_name: dict[str, np.ndarray] = {}
        count = int(rel.shape[0])
        for constraint in self.config.sun_angle_constraints:
            ok_hist = self._sun_angle_ok_by_constraint.get(constraint.name, [])
            angle_hist = self._sun_angle_deg_by_constraint.get(constraint.name, [])
            if len(ok_hist) == count and len(angle_hist) == count:
                ok_by_name[constraint.name] = np.array(ok_hist, dtype=bool)
                angle_by_name[constraint.name] = np.array(angle_hist, dtype=float)
            else:
                ok_by_name[constraint.name] = constraint.samples_satisfying_constraint(rel[:, :3])
                angle_by_name[constraint.name] = constraint.sun_angles_deg(rel[:, :3])
        return ok_by_name, angle_by_name

    def current_hint(self) -> str:
        return self._current_hint()

    def _current_hint(self, *, inspection_gates_satisfied: int | None = None) -> str:
        if not self.rel_ric_hist:
            return ""
        rel = self.rel_ric_hist[-1]
        r = rel[:3]
        v = rel[3:]
        rng = float(np.linalg.norm(r))
        speed = float(np.linalg.norm(v))
        closing = float(np.dot(r, v)) < 0.0
        cross_track_amplitude = self._current_cross_track_amplitude_km(rel)
        keepout = self.config.keepout_radius_km
        if keepout is not None and rng < float(keepout):
            return "Inside keepout: arrest closing motion and translate away from the target."
        if self.config.sandbox_mode:
            return "Sandbox: Maneuver freely, coast, and watch the relative orbit respond."
        if self.config.sun_angle_constraints:
            for constraint in self.config.sun_angle_constraints:
                ok_hist = self._sun_angle_ok_by_constraint.get(constraint.name, [])
                angle_hist = self._sun_angle_deg_by_constraint.get(constraint.name, [])
                if ok_hist and angle_hist:
                    ok = bool(ok_hist[-1])
                    sun_angle = float(angle_hist[-1])
                else:
                    ok = bool(constraint.samples_satisfying_constraint(r.reshape(1, 3))[0])
                    sun_angle = float(constraint.sun_angles_deg(r.reshape(1, 3))[0])
                if not ok:
                    return f"Outside Sun-angle beam: reenter the amber region before crossing the next gate. Current Sun angle {sun_angle:.0f} deg."
        if self.config.inspection_gates:
            gate_status = self._inspection_gate_status()
            satisfied_names = set(gate_status["satisfied"])
            if len(satisfied_names) >= len(self.config.inspection_gates):
                return "All inspection gates complete: level should complete."
            gate = next(gate for gate in self.config.inspection_gates if gate.name not in satisfied_names)
            delta = np.array(gate.center_ric_km, dtype=float).reshape(3) - r
            return (
                f"Next inspection gate: {gate.name}; drift toward "
                f"R {_format_signed_distance_text(delta[0])}, "
                f"I {_format_signed_distance_text(delta[1])}, "
                f"C {_format_signed_distance_text(delta[2])}."
            )
        if self.config.survival_goal:
            keepout = self.config.keepout_radius_km
            if keepout is not None:
                margin = rng - float(keepout)
                return f"Evade: keep at least {_format_distance_text(float(keepout))} separation. Margin {_format_distance_text(margin)}."
            return "Evade: keep separation until the timer expires."
        missing_requirements = self._missing_tutorial_requirements()
        if missing_requirements:
            staged_hint = self._tutorial_stage_hint()
            if staged_hint:
                return staged_hint
            return f"Tutorial checklist: complete {', '.join(missing_requirements)} before finishing."
        if self.config.goal_nmt_radial_amplitude_km is None and self.config.goal_range_km is not None:
            target_range = float(self.config.goal_range_km)
            tolerance = self.config.goal_range_tolerance_km
            speed_limit = self.config.max_goal_speed_km_s
            range_error = rng - target_range
            if tolerance is None:
                if rng <= target_range:
                    if speed_limit is not None and speed > float(speed_limit):
                        return f"Inside green circle: slow below {_format_speed_text(float(speed_limit))} to finish."
                    if (
                        cross_track_amplitude is not None
                        and self.config.max_cross_track_amplitude_km is not None
                        and cross_track_amplitude > float(self.config.max_cross_track_amplitude_km)
                    ):
                        return (
                            "Inside green circle: damp C amplitude below "
                            f"{_format_distance_text(float(self.config.max_cross_track_amplitude_km))}."
                        )
                    return "Inside green circle with speed under limit: level should complete."
                final_hint = self.config.tutorial_stage_hints.get("final_approach", "")
                if final_hint:
                    return final_hint
                return f"Enter the green circle: close to {_format_distance_text(target_range)} or less."
            if abs(range_error) <= float(tolerance):
                if speed_limit is not None and speed > float(speed_limit):
                    return f"At target range: slow below {_format_speed_text(float(speed_limit))} to finish."
                return "At target range with speed under limit: level should complete."
            if abs(range_error) <= max(float(tolerance) * 2.0, 0.1):
                if speed_limit is not None:
                    return f"Near target range: brake below {_format_speed_text(float(speed_limit))}."
                return "Near target range: settle in the green range band."
            final_hint = self.config.tutorial_stage_hints.get("final_approach", "")
            if final_hint:
                return final_hint
        if self.config.goal_nmt_radial_amplitude_km is None and self.config.goal_radius_km is not None:
            goal = np.array(self.config.goal_relative_ric_km, dtype=float).reshape(-1)
            if goal.size == 3:
                goal_error = float(np.linalg.norm(r - goal))
                goal_radius = float(self.config.goal_radius_km)
                speed_limit = self.config.max_goal_speed_km_s
                if goal_error <= goal_radius:
                    if speed_limit is not None and speed > float(speed_limit):
                        return f"Inside hold box: slow below {_format_speed_text(float(speed_limit))} to finish."
                    if (
                        cross_track_amplitude is not None
                        and self.config.max_cross_track_amplitude_km is not None
                        and cross_track_amplitude > float(self.config.max_cross_track_amplitude_km)
                    ):
                        return (
                            "Inside hold box: damp C amplitude below "
                            f"{_format_distance_text(float(self.config.max_cross_track_amplitude_km))}."
                        )
                    return "Inside hold box with speed under limit: level should complete."
                if goal_error <= max(goal_radius * 2.0, goal_radius + 0.05):
                    if speed_limit is not None:
                        return (
                            "Near hold box: center in the green circle and brake below "
                            f"{_format_speed_text(float(speed_limit))}."
                        )
                    return "Near hold box: center in the green circle."
        if closing and speed > 0.01:
            return "Closing quickly: reduce relative speed before correcting position."
        if abs(float(r[1])) > max(abs(float(r[0])), abs(float(r[2])), 0.1):
            return "In-track error dominates: small along-track burns can create delayed radial effects."
        if speed < 0.001 and rng > 1.0:
            return "Mostly coasting: watch the natural relative drift before burning again."
        return "Pulse gently, coast, and watch the relative motion before the next correction."

    def _tutorial_stage_hint(self) -> str:
        hints = self.config.tutorial_stage_hints
        satisfied = set(self._burn_axes_satisfied())
        for axis in self.config.required_burn_axes:
            if axis not in satisfied:
                return hints.get(axis) or self.config.axis_descriptions.get(axis, "")
        if self.config.require_speed_multiplier_change and not self._speed_multiplier_changed:
            return hints.get("speed_multiplier", "")
        if self.config.required_coast_after_burn_s is not None:
            coast_satisfied, _, coast_s = self._coast_after_burn_status()
            if not coast_satisfied:
                hint = hints.get("coast", "")
                if hint and coast_s > 0.0:
                    return f"{hint} Current coast: {coast_s:.0f} s."
                return hint
        return ""

    def _current_cross_track_amplitude_km(self, rel: np.ndarray) -> float | None:
        if self.config.max_cross_track_amplitude_km is None or not self.mean_motion_hist:
            return None
        n = float(self.mean_motion_hist[-1])
        if not np.isfinite(n) or abs(n) <= 1.0e-12:
            return None
        state = np.array(rel, dtype=float).reshape(6)
        center_c = float(np.array(self.config.goal_nmt_center_ric_km, dtype=float).reshape(3)[2])
        return float(np.sqrt((state[2] - center_c) ** 2 + (state[5] / n) ** 2))

    def score(self) -> RPOTrainingScore:
        if self._score_cache is not None:
            return self._score_cache
        if not self.rel_ric_hist:
            score = RPOTrainingScore(
                scenario_id=self.config.scenario_id,
                learning_goal=self.config.learning_goal,
                samples=0,
                elapsed_s=0.0,
                closest_approach_km=float("nan"),
                final_range_km=float("nan"),
                final_goal_error_km=float("nan"),
                final_relative_speed_km_s=float("nan"),
                time_inside_keepout_s=0.0,
                approximate_delta_v_m_s=0.0,
                target_delta_v_m_s=0.0,
                burn_axes_satisfied=(),
                phase_burns_satisfied=(),
                speed_multiplier_changed=bool(self._speed_multiplier_changed),
                coast_after_burn_satisfied=False,
                coast_after_burn_s=0.0,
                guided_tutorial_burns_satisfied=(),
                guided_tutorial_burns_total=len(self.config.guided_tutorial_burns),
                guided_tutorial_speed_satisfied=self.config.guided_tutorial_speed_step is None,
                guided_tutorial_speed_target=(
                    None
                    if self.config.guided_tutorial_speed_step is None
                    else float(self.config.guided_tutorial_speed_step.target_speed_multiplier)
                ),
                achieved_time_s=None,
                min_goal_error_km=float("nan"),
                final_nmt_radial_amplitude_km=float("nan"),
                final_nmt_cross_track_amplitude_km=float("nan"),
                final_nmt_radial_amplitude_error_km=float("nan"),
                final_nmt_cross_track_amplitude_error_km=float("nan"),
                final_nmt_drift_velocity_error_km_s=float("nan"),
                goal_met=False,
                level_passed=False,
                level_failed=False,
                pass_fail_reasons=("No samples recorded.",),
                keepout_violation=False,
                hard_speed_limit_violation=False,
                forbidden_region_violation=False,
                forbidden_region_names=(),
                sun_angle_violation=False,
                sun_angle_constraint_names=(),
                sun_angle_violation_time_s=0.0,
                min_sun_angle_deg=float("nan"),
                final_sun_angle_deg=float("nan"),
                approach_gate_violation=False,
                approach_gate_names=(),
                approach_gates_satisfied=0,
                approach_gates_total=len(self.config.approach_gates),
                inspection_gates_satisfied=0,
                inspection_gates_total=len(self.config.inspection_gates),
                inspection_gate_names=(),
                hints=(),
            )
            self._score_cache = score
            return score
        rel, t, thrust, target_thrust, n_hist = self._history_arrays()
        burn_axis_first_sample_idx = self._burn_axis_first_sample_indices()
        burn_axes_satisfied = self._burn_axes_satisfied()
        phase_burn_first_sample_idx = self._phase_burn_first_sample_indices()
        phase_burns_satisfied = self._phase_burns_satisfied()
        coast_after_burn_satisfied, coast_after_burn_idx, coast_after_burn_s = self._coast_after_burn_status()
        guided_tutorial_burns_satisfied = self.guided_tutorial_burns_satisfied()
        guided_tutorial_speed_satisfied = self.guided_tutorial_speed_satisfied()
        if self._history_arrays_available():
            count = int(rel.shape[0])
            ranges = self._range_array[:count]
            speeds = self._speed_array[:count]
        else:
            ranges = np.linalg.norm(rel[:, :3], axis=1)
            speeds = np.linalg.norm(rel[:, 3:], axis=1)
        element_errors = None
        if (
            self.config.goal_nmt_radial_amplitude_km is not None
            or self.config.max_cross_track_amplitude_km is not None
        ) and n_hist.size:
            element_errors = self._nmt_element_error_arrays(rel, n_hist)
        if self._history_arrays_available():
            goal_err = self._goal_error_array[: int(rel.shape[0])]
        elif self.config.goal_nmt_radial_amplitude_km is not None:
            if self.config.goal_nmt_tolerance_km is not None:
                goal_err = nmt_position_error_km(
                    rel[:, :3],
                    radial_amplitude_km=float(self.config.goal_nmt_radial_amplitude_km),
                    cross_track_amplitude_km=float(self.config.goal_nmt_cross_track_amplitude_km),
                    cross_track_phase_deg=float(self.config.goal_nmt_cross_track_phase_deg),
                    center_ric_km=self.config.goal_nmt_center_ric_km,
                )
            elif element_errors is not None:
                goal_err = self._nmt_element_goal_error_array_for(element_errors)
            else:
                goal_err = np.linalg.norm(rel[:, :3] - self.config.goal_nmt_center_ric_km.reshape(1, 3), axis=1)
        elif self.config.goal_range_km is not None:
            if self.config.goal_range_tolerance_km is None:
                goal_err = np.maximum(ranges - float(self.config.goal_range_km), 0.0)
            else:
                goal_err = np.abs(ranges - float(self.config.goal_range_km))
        elif self.config.inspection_gates:
            gate_centers = np.vstack([gate.center_ric_km for gate in self.config.inspection_gates])
            goal_err = np.min(np.linalg.norm(rel[:, None, :3] - gate_centers[None, :, :], axis=2), axis=1)
        else:
            goal_err = np.linalg.norm(rel[:, :3] - self.config.goal_relative_ric_km.reshape(1, 3), axis=1)
        keepout_time = 0.0
        keepout_violation = False
        if self.config.keepout_radius_km is not None:
            inside = ranges < float(self.config.keepout_radius_km)
            keepout_violation = bool(np.any(inside)) or _segment_crosses_sphere_km(
                rel[:, :3],
                float(self.config.keepout_radius_km),
            )
            keepout_time = _sampled_dwell_time_s(inside, t)
        hard_speed_limit_violation = False
        if self.config.hard_speed_limit_radius_km is not None and self.config.hard_speed_limit_km_s is not None:
            hard_speed_limit_violation = bool(self._hard_speed_limit_violation)
        if self._history_arrays_available():
            forbidden_region_names = [
                region.name for region in self.config.forbidden_regions if region.name in self._forbidden_region_names
            ]
        else:
            forbidden_region_names = []
            for region in self.config.forbidden_regions:
                sampled_inside = bool(np.any(region.contains_positions(rel[:, :3])))
                segment_crossing = any(
                    region.intersects_segment(rel[idx - 1, :3], rel[idx, :3])
                    for idx in range(1, rel.shape[0])
                )
                if sampled_inside or segment_crossing:
                    forbidden_region_names.append(region.name)
        forbidden_region_violation = bool(forbidden_region_names)
        sun_angle_constraint_names: list[str] = []
        sun_angle_all_ok = np.ones(rel.shape[0], dtype=bool)
        ok_by_name, angle_by_name = self._sun_angle_status_arrays(rel)
        for constraint_name in [constraint.name for constraint in self.config.sun_angle_constraints]:
            ok = ok_by_name.get(constraint_name, np.ones(rel.shape[0], dtype=bool))
            sun_angle_all_ok &= ok
            if not bool(np.all(ok)):
                sun_angle_constraint_names.append(constraint_name)
        sun_angle_violation = bool(sun_angle_constraint_names)
        sun_angle_violation_time_s = _sampled_dwell_time_s(~sun_angle_all_ok, t) if angle_by_name else 0.0
        if angle_by_name:
            sun_angles = np.vstack([angle_by_name[name] for name in angle_by_name])
            finite_angles = sun_angles[np.isfinite(sun_angles)]
            min_sun_angle_deg = float(np.min(finite_angles)) if finite_angles.size else float("nan")
            first_angles = sun_angles[0]
            final_sun_angle_deg = float(first_angles[-1]) if first_angles.size else float("nan")
        else:
            min_sun_angle_deg = float("nan")
            final_sun_angle_deg = float("nan")
        target_reference_range_violation = False
        final_target_reference_range_km = float("nan")
        if self.config.max_target_reference_range_km is not None:
            target_reference_rel = (
                np.vstack(self.target_reference_rel_hist)
                if self.target_reference_rel_hist
                else np.zeros((0, 6), dtype=float)
            )
            if target_reference_rel.size:
                target_reference_ranges = np.linalg.norm(target_reference_rel[:, :3], axis=1)
                finite_target_reference_ranges = target_reference_ranges[np.isfinite(target_reference_ranges)]
                if finite_target_reference_ranges.size:
                    final_target_reference_range_km = float(finite_target_reference_ranges[-1])
                    target_reference_range_violation = bool(
                        np.any(finite_target_reference_ranges > float(self.config.max_target_reference_range_km))
                    )
                else:
                    target_reference_range_violation = True
            else:
                target_reference_range_violation = True
        if self._history_arrays_available():
            count = int(self._history_count)
            dv_intervals = self._delta_v_interval_km_s_array[1:count]
            target_dv_intervals = self._target_delta_v_interval_km_s_array[1:count]
            dv_m_s = float(np.sum(dv_intervals[np.isfinite(dv_intervals)]) * 1.0e3)
            target_dv_m_s = float(
                np.sum(target_dv_intervals[np.isfinite(target_dv_intervals)]) * 1.0e3
            )
        else:
            dv_m_s = _integrated_delta_v_m_s(thrust, t)
            target_dv_m_s = _integrated_delta_v_m_s(target_thrust, t)
        inspection_gate_status = self._inspection_gate_status()
        goal_met_samples = np.ones(rel.shape[0], dtype=bool)
        if self.config.survival_goal:
            goal_met_samples = np.zeros(rel.shape[0], dtype=bool)
            if self.config.max_time_s is not None:
                goal_met_samples |= (t - t[0]) >= float(self.config.max_time_s)
        elif self.config.inspection_gates:
            goal_met_samples = np.zeros(rel.shape[0], dtype=bool)
            if inspection_gate_status["completed_idx"] is not None:
                goal_met_samples[int(inspection_gate_status["completed_idx"]) :] = True
        elif self.config.goal_range_km is not None and self.config.goal_range_tolerance_km is None:
            goal_met_samples &= ranges <= float(self.config.goal_range_km)
        elif self.config.goal_range_km is not None and self.config.goal_range_tolerance_km is not None:
            goal_met_samples &= goal_err <= float(self.config.goal_range_tolerance_km)
        elif self.config.goal_radius_km is not None:
            goal_met_samples &= goal_err <= float(self.config.goal_radius_km)
        if self.config.goal_nmt_tolerance_km is not None:
            goal_met_samples &= goal_err <= float(self.config.goal_nmt_tolerance_km)
        if element_errors is not None and self.config.goal_nmt_element_tolerance_km is not None:
            tol = float(self.config.goal_nmt_element_tolerance_km)
            goal_met_samples &= element_errors["radial_amplitude_error_km"] <= tol
            goal_met_samples &= element_errors["cross_track_amplitude_error_km"] <= tol
        if element_errors is not None and self.config.goal_nmt_velocity_tolerance_km_s is not None:
            goal_met_samples &= element_errors["drift_velocity_error_km_s"] <= float(
                self.config.goal_nmt_velocity_tolerance_km_s
            )
        if element_errors is not None and self.config.max_cross_track_amplitude_km is not None:
            goal_met_samples &= element_errors["cross_track_amplitude_km"] <= float(
                self.config.max_cross_track_amplitude_km
            )
        if self.config.max_time_s is not None and not self.config.survival_goal:
            goal_met_samples &= (t - t[0]) <= float(self.config.max_time_s)
        if self.config.max_goal_speed_km_s is not None:
            goal_met_samples &= speeds <= float(self.config.max_goal_speed_km_s)
        for axis in self.config.required_burn_axes:
            axis_sample_idx = burn_axis_first_sample_idx.get(axis)
            if axis_sample_idx is None:
                goal_met_samples &= False
            else:
                goal_met_samples[: min(axis_sample_idx, goal_met_samples.size)] = False
        for phase_burn in self.config.required_phase_burns:
            phase_sample_idx = phase_burn_first_sample_idx.get(phase_burn.name)
            if phase_sample_idx is None:
                goal_met_samples &= False
            else:
                goal_met_samples[: min(phase_sample_idx, goal_met_samples.size)] = False
        if self.config.require_speed_multiplier_change:
            speed_change_idx = self._speed_multiplier_change_sample_idx
            if speed_change_idx is None:
                goal_met_samples &= False
            else:
                goal_met_samples[: min(speed_change_idx, goal_met_samples.size)] = False
        if self.config.required_coast_after_burn_s is not None:
            if coast_after_burn_idx is None:
                goal_met_samples &= False
            else:
                goal_met_samples[: min(coast_after_burn_idx, goal_met_samples.size)] = False
        achieved_idx = np.flatnonzero(goal_met_samples)
        achieved_time_s = float(t[int(achieved_idx[0])] - t[0]) if achieved_idx.size else None
        gate_eval_end = int(achieved_idx[0]) + 1 if achieved_idx.size else rel.shape[0]
        gate_rel = rel[: max(gate_eval_end, 1)]
        gate_status = _approach_gate_status(self.config.approach_gates, gate_rel)
        budget_ok = True
        reasons: list[str] = []
        if self.config.goal_nmt_radial_amplitude_km is not None:
            objective_name = "NMT target"
        elif self.config.survival_goal:
            objective_name = "survival objective"
        elif self.config.inspection_gates:
            objective_name = "inspection gates"
        elif self.config.goal_range_km is not None:
            objective_name = "range goal"
        else:
            objective_name = "goal"
        if achieved_time_s is None:
            reasons.append(f"{objective_name} not achieved within tolerance.")
        if (
            achieved_time_s is None
            and self.config.max_cross_track_amplitude_km is not None
            and element_errors is not None
            and np.isfinite(element_errors["cross_track_amplitude_km"][-1])
            and element_errors["cross_track_amplitude_km"][-1] > float(self.config.max_cross_track_amplitude_km)
        ):
            reasons.append(
                "Cross-track amplitude above "
                f"{_format_distance_text(float(self.config.max_cross_track_amplitude_km))}."
            )
        time_failed = (
            self.config.max_time_s is not None
            and achieved_time_s is None
            and float(t[-1] - t[0]) >= float(self.config.max_time_s)
        )
        if time_failed:
            reasons.append(f"Time budget exceeded ({float(self.config.max_time_s):.0f} s).")
        dv_failed = False
        if self.config.max_delta_v_m_s is not None and dv_m_s > float(self.config.max_delta_v_m_s):
            if self.config.fail_on_delta_v_budget:
                budget_ok = False
                dv_failed = True
                reasons.append(f"Delta-v budget exceeded ({format_speed_m_s(float(self.config.max_delta_v_m_s))}).")
        target_dv_failed = False
        if self.config.max_target_delta_v_m_s is not None and target_dv_m_s > float(self.config.max_target_delta_v_m_s):
            budget_ok = False
            target_dv_failed = True
            reasons.append(
                f"Target delta-v budget exceeded ({format_speed_m_s(float(self.config.max_target_delta_v_m_s))})."
            )
        if self.config.keepout_radius_km is not None:
            budget_ok = budget_ok and not keepout_violation
            if keepout_violation:
                reasons.append("Keepout was violated.")
        if hard_speed_limit_violation:
            budget_ok = False
            assert self.config.hard_speed_limit_radius_km is not None
            assert self.config.hard_speed_limit_km_s is not None
            reasons.append(
                "Hard speed limit violated inside "
                f"{_format_distance_text(float(self.config.hard_speed_limit_radius_km))}: "
                f"{_format_speed_text(float(self.config.hard_speed_limit_km_s))} max."
            )
        if forbidden_region_violation:
            budget_ok = False
            regions = ", ".join(forbidden_region_names[:3])
            suffix = "..." if len(forbidden_region_names) > 3 else ""
            reasons.append(f"Forbidden region violated: {regions}{suffix}.")
        if target_reference_range_violation:
            budget_ok = False
            reasons.append(
                "Mission-capable radius exceeded "
                f"({_format_distance_text(float(self.config.max_target_reference_range_km))})."
            )
        approach_gate_warnings = list(gate_status["required_violated"])
        approach_gate_names: list[str] = []
        if achieved_time_s is not None:
            approach_gate_names.extend(approach_gate_warnings)
            approach_gate_names.extend(gate_status["required_missed"])
        approach_gate_violation = bool(approach_gate_names)
        if approach_gate_violation:
            budget_ok = False
            gates = ", ".join(approach_gate_names[:3])
            suffix = "..." if len(approach_gate_names) > 3 else ""
            reasons.append(f"R-bar approach gate failed: {gates}{suffix}.")
        requirements_ok = True
        burn_axes_set = set(burn_axes_satisfied)
        for axis in self.config.required_burn_axes:
            if axis not in burn_axes_set:
                requirements_ok = False
                reasons.append(f"{_BURN_AXIS_LABEL[axis]} burn required.")
        phase_burns_set = set(phase_burns_satisfied)
        for phase_burn in self.config.required_phase_burns:
            if phase_burn.name not in phase_burns_set:
                requirements_ok = False
                reasons.append(f"{phase_burn.label} required.")
        if self.config.require_speed_multiplier_change and not self._speed_multiplier_changed:
            requirements_ok = False
            reasons.append("Speed multiplier change required.")
        if self.config.required_coast_after_burn_s is not None and not coast_after_burn_satisfied:
            requirements_ok = False
            reasons.append(f"Coast for {float(self.config.required_coast_after_burn_s):.0f} s after a burn required.")
        guided_tutorial_burn_set = set(guided_tutorial_burns_satisfied)
        for stage in self.config.guided_tutorial_burns:
            if stage.name not in guided_tutorial_burn_set:
                requirements_ok = False
                reasons.append(f"{stage.display_label} tutorial stage required.")
        if self.config.guided_tutorial_speed_step is not None and not guided_tutorial_speed_satisfied:
            requirements_ok = False
            reasons.append(f"{self.config.guided_tutorial_speed_step.label} tutorial step required.")
        if self.config.sandbox_mode:
            sandbox_elapsed = float(t[-1] - t[0]) if t.size >= 2 else 0.0
            level_passed = bool(self.config.max_time_s is not None and sandbox_elapsed >= float(self.config.max_time_s))
            level_failed = False
            reasons = (
                ["Sandbox complete; time limit reached."]
                if level_passed
                else ["Sandbox active; no pass/fail objective."]
            )
        else:
            level_passed = bool(achieved_time_s is not None and budget_ok and requirements_ok)
            level_failed = bool(
                (
                    keepout_violation
                    or hard_speed_limit_violation
                    or forbidden_region_violation
                    or target_reference_range_violation
                    or approach_gate_violation
                    or dv_failed
                    or target_dv_failed
                    or time_failed
                )
                and not level_passed
            )
        goal_met = level_passed
        if level_passed:
            reasons.append("All pass criteria satisfied.")
        final_elements = _final_nmt_element_values(element_errors)
        hints = tuple(
            h for h in (self._current_hint(inspection_gates_satisfied=len(inspection_gate_status["satisfied"])),) if h
        )
        score = RPOTrainingScore(
            scenario_id=self.config.scenario_id,
            learning_goal=self.config.learning_goal,
            samples=int(rel.shape[0]),
            elapsed_s=float(t[-1] - t[0]) if t.size >= 2 else 0.0,
            closest_approach_km=float(np.min(ranges)),
            final_range_km=float(ranges[-1]),
            final_goal_error_km=float(goal_err[-1]),
            final_relative_speed_km_s=float(speeds[-1]),
            time_inside_keepout_s=float(keepout_time),
            approximate_delta_v_m_s=float(dv_m_s),
            target_delta_v_m_s=float(target_dv_m_s),
            burn_axes_satisfied=tuple(burn_axes_satisfied),
            phase_burns_satisfied=tuple(phase_burns_satisfied),
            speed_multiplier_changed=bool(self._speed_multiplier_changed),
            coast_after_burn_satisfied=bool(coast_after_burn_satisfied),
            coast_after_burn_s=float(coast_after_burn_s),
            guided_tutorial_burns_satisfied=tuple(guided_tutorial_burns_satisfied),
            guided_tutorial_burns_total=len(self.config.guided_tutorial_burns),
            guided_tutorial_speed_satisfied=bool(guided_tutorial_speed_satisfied),
            guided_tutorial_speed_target=(
                None
                if self.config.guided_tutorial_speed_step is None
                else float(self.config.guided_tutorial_speed_step.target_speed_multiplier)
            ),
            achieved_time_s=achieved_time_s,
            min_goal_error_km=float(np.min(goal_err)),
            final_nmt_radial_amplitude_km=final_elements["radial_amplitude_km"],
            final_nmt_cross_track_amplitude_km=final_elements["cross_track_amplitude_km"],
            final_nmt_radial_amplitude_error_km=final_elements["radial_amplitude_error_km"],
            final_nmt_cross_track_amplitude_error_km=final_elements["cross_track_amplitude_error_km"],
            final_nmt_drift_velocity_error_km_s=final_elements["drift_velocity_error_km_s"],
            goal_met=bool(goal_met),
            level_passed=bool(level_passed),
            level_failed=bool(level_failed),
            pass_fail_reasons=tuple(reasons),
            keepout_violation=bool(keepout_violation),
            hard_speed_limit_violation=bool(hard_speed_limit_violation),
            forbidden_region_violation=bool(forbidden_region_violation),
            forbidden_region_names=tuple(forbidden_region_names),
            sun_angle_violation=bool(sun_angle_violation),
            sun_angle_constraint_names=tuple(sun_angle_constraint_names),
            sun_angle_violation_time_s=float(sun_angle_violation_time_s),
            min_sun_angle_deg=float(min_sun_angle_deg),
            final_sun_angle_deg=float(final_sun_angle_deg),
            approach_gate_violation=bool(approach_gate_violation),
            approach_gate_names=tuple(approach_gate_names),
            approach_gates_satisfied=len(gate_status["satisfied"]),
            approach_gates_total=len(self.config.approach_gates),
            inspection_gates_satisfied=len(inspection_gate_status["satisfied"]),
            inspection_gates_total=len(self.config.inspection_gates),
            inspection_gate_names=tuple(inspection_gate_status["satisfied"]),
            hints=hints,
            final_target_reference_range_km=float(final_target_reference_range_km),
            max_target_reference_range_km=self.config.max_target_reference_range_km,
            target_reference_range_violation=bool(target_reference_range_violation),
        )
        self._score_cache = score
        return score

    def debrief_text(self) -> str:
        score = self.score()
        lines = [
            "",
            "=" * 72,
            "RPO TRAINER DEBRIEF",
            "=" * 72,
        ]
        if score.scenario_id:
            lines.append(f"Scenario      : {score.scenario_id}")
        if score.learning_goal:
            lines.append(f"Learning Goal : {score.learning_goal}")
        lines.extend(
            [
                f"Samples       : {score.samples}",
                f"Elapsed       : {score.elapsed_s:.1f} s",
                f"Closest App   : {_format_distance_text(score.closest_approach_km)}",
                f"Final Range   : {_format_distance_text(score.final_range_km)}",
                f"Goal Error    : {_format_distance_text(score.final_goal_error_km)}",
                f"Best Goal Err : {_format_distance_text(score.min_goal_error_km)}",
                f"Final Speed   : {_format_speed_text(score.final_relative_speed_km_s)}",
                f"Keepout Time  : {score.time_inside_keepout_s:.1f} s",
                f"Approx dV     : {format_speed_m_s(score.approximate_delta_v_m_s)}",
                f"Target dV     : {format_speed_m_s(score.target_delta_v_m_s)}",
                f"Achieved Time : {_format_optional_time(score.achieved_time_s)}",
                f"Level Passed  : {'Yes' if score.level_passed else 'No'}",
            ]
        )
        if score.forbidden_region_violation:
            lines.append(f"Forbidden Reg : {', '.join(score.forbidden_region_names)}")
        if self.config.sun_angle_constraints:
            lines.append(f"Min Sun Angle : {score.min_sun_angle_deg:.1f} deg")
            lines.append(f"Final Sun Ang : {score.final_sun_angle_deg:.1f} deg")
            lines.append(f"Sun Viol Time : {score.sun_angle_violation_time_s:.1f} s")
        if score.sun_angle_violation:
            lines.append(f"Sun Region Out: {', '.join(score.sun_angle_constraint_names)}")
        if score.approach_gates_total:
            lines.append(f"R-Bar Gates   : {score.approach_gates_satisfied}/{score.approach_gates_total}")
        if score.inspection_gates_total:
            lines.append(f"Inspect Gates : {score.inspection_gates_satisfied}/{score.inspection_gates_total}")
        if score.approach_gate_violation:
            lines.append(f"Gate Failure  : {', '.join(score.approach_gate_names)}")
        if self.config.required_burn_axes:
            axes = ", ".join(_BURN_AXIS_LABEL.get(axis, axis.title()) for axis in score.burn_axes_satisfied)
            axes = axes if axes else "None"
            lines.append(f"Burn Axes     : {axes}")
        if self.config.required_phase_burns:
            burns = ", ".join(score.phase_burns_satisfied) if score.phase_burns_satisfied else "None"
            lines.append(f"Phase Burns   : {burns}")
        if self.config.require_speed_multiplier_change:
            lines.append(f"Speed Changed : {'Yes' if score.speed_multiplier_changed else 'No'}")
        if self.config.goal_nmt_radial_amplitude_km is not None:
            lines.extend(
                [
                    f"NMT Rad Amp   : {_format_distance_text(score.final_nmt_radial_amplitude_km)}",
                    f"NMT Cross Amp : {_format_distance_text(score.final_nmt_cross_track_amplitude_km)}",
                    f"NMT Drift Err : {_format_speed_text(score.final_nmt_drift_velocity_error_km_s)}",
                ]
            )
        elif self.config.max_cross_track_amplitude_km is not None:
            lines.append(f"Cross Amp     : {_format_distance_text(score.final_nmt_cross_track_amplitude_km)}")
        for reason in score.pass_fail_reasons:
            lines.append(f"Pass/Fail     : {reason}")
        for hint in score.hints:
            lines.append(f"Coach Note    : {hint}")
        lines.append("=" * 72)
        return "\n".join(lines)


def relative_ric_state_from_arrays(target_truth: np.ndarray, chaser_truth: np.ndarray) -> np.ndarray:
    target = np.array(target_truth, dtype=float).reshape(-1)
    chaser = np.array(chaser_truth, dtype=float).reshape(-1)
    if target.size < 6 or chaser.size < 6:
        return np.full(6, np.nan, dtype=float)
    return eci_relative_to_ric_rect(chaser[:6], target[:6])


def relative_moon_ric_state_from_arrays(target_truth: np.ndarray, chaser_truth: np.ndarray) -> np.ndarray:
    target = np.array(target_truth, dtype=float).reshape(-1)
    chaser = np.array(chaser_truth, dtype=float).reshape(-1)
    if target.size < 6 or chaser.size < 6:
        return np.full(6, np.nan, dtype=float)
    moon = cr3bp_moon_state_km_s()
    return eci_relative_to_ric_rect(chaser[:6] - moon, target[:6] - moon)


def relative_state_from_arrays(target_truth: np.ndarray, chaser_truth: np.ndarray, *, frame: str = "ric") -> np.ndarray:
    frame_key = _relative_frame_key(frame)
    if frame_key == "cislunar":
        target = np.array(target_truth, dtype=float).reshape(-1)
        chaser = np.array(chaser_truth, dtype=float).reshape(-1)
        if target.size < 6 or chaser.size < 6:
            return np.full(6, np.nan, dtype=float)
        return cr3bp_relative_state(chaser[:6], target[:6])
    if frame_key == "moon_ric":
        return relative_moon_ric_state_from_arrays(target_truth, chaser_truth)
    return relative_ric_state_from_arrays(target_truth, chaser_truth)


def _relative_frame_key(frame: str) -> str:
    key = str(frame or "ric").strip().lower().replace("-", "_")
    if key in {"cislunar", "cislunar_l1", "earth_moon_rotating", "cr3bp", "cr3bp_rotating"}:
        return "cislunar"
    if key in {"moon_ric", "lunar_ric", "target_moon_ric", "target_lunar_ric"}:
        return "moon_ric"
    return "ric"


def nmt_position_error_km(
    relative_ric_km: np.ndarray,
    *,
    radial_amplitude_km: float,
    cross_track_amplitude_km: float = 0.0,
    cross_track_phase_deg: float = 0.0,
    center_ric_km: np.ndarray,
) -> np.ndarray:
    pos = np.array(relative_ric_km, dtype=float)
    if pos.ndim == 1:
        pos = pos.reshape(1, -1)
    if pos.shape[1] < 3:
        raise ValueError("relative_ric_km must contain R, I, and C components.")
    center = np.array(center_ric_km, dtype=float).reshape(3)
    curve = nmt_curve_points_km(
        radial_amplitude_km=radial_amplitude_km,
        cross_track_amplitude_km=cross_track_amplitude_km,
        cross_track_phase_deg=cross_track_phase_deg,
        center_ric_km=center,
    )
    if curve.size == 0:
        return np.linalg.norm(pos[:, :3] - center.reshape(1, 3), axis=1)
    delta = pos[:, None, :3] - curve[None, :, :]
    return np.min(np.linalg.norm(delta, axis=2), axis=1)


def nmt_curve_points_km(
    *,
    radial_amplitude_km: float,
    cross_track_amplitude_km: float = 0.0,
    cross_track_phase_deg: float = 0.0,
    center_ric_km: np.ndarray,
    samples: int = 721,
) -> np.ndarray:
    a_r = float(radial_amplitude_km)
    if not np.isfinite(a_r) or a_r <= 0.0:
        return np.empty((0, 3), dtype=float)
    a_c = float(cross_track_amplitude_km)
    if not np.isfinite(a_c):
        a_c = 0.0
    phase = np.deg2rad(float(cross_track_phase_deg))
    center = np.array(center_ric_km, dtype=float).reshape(3)
    return _cached_nmt_curve_points_km(
        float(a_r),
        float(a_c),
        float(phase),
        (float(center[0]), float(center[1]), float(center[2])),
        int(max(int(samples), 8)),
    ).copy()


@lru_cache(maxsize=64)
def _cached_nmt_curve_points_km(
    radial_amplitude_km: float,
    cross_track_amplitude_km: float,
    phase_rad: float,
    center_ric_km: tuple[float, float, float],
    samples: int,
) -> np.ndarray:
    a_r = float(radial_amplitude_km)
    a_c = float(cross_track_amplitude_km)
    phase = float(phase_rad)
    center = np.array(center_ric_km, dtype=float).reshape(3)
    theta = np.linspace(0.0, 2.0 * np.pi, max(int(samples), 8), endpoint=True)
    pts = np.zeros((theta.size, 3), dtype=float)
    pts[:, 0] = center[0] + a_r * np.cos(theta)
    pts[:, 1] = center[1] - 2.0 * a_r * np.sin(theta)
    pts[:, 2] = center[2] + a_c * np.cos(theta + phase)
    pts.setflags(write=False)
    return pts


def nmt_velocity_error_km_s(
    relative_ric_state: np.ndarray,
    *,
    mean_motion_rad_s: float,
    radial_amplitude_km: float,
    cross_track_amplitude_km: float = 0.0,
    cross_track_phase_deg: float = 0.0,
    center_ric_km: np.ndarray,
) -> float:
    rel = np.array(relative_ric_state, dtype=float).reshape(-1)
    if rel.size < 6:
        raise ValueError("relative_ric_state must contain RIC position and velocity.")
    center = np.array(center_ric_km, dtype=float).reshape(3)
    n = float(mean_motion_rad_s)
    curve = nmt_curve_points_km(
        radial_amplitude_km=radial_amplitude_km,
        cross_track_amplitude_km=cross_track_amplitude_km,
        cross_track_phase_deg=cross_track_phase_deg,
        center_ric_km=center,
    )
    if curve.size == 0 or not np.isfinite(n):
        return float(np.linalg.norm(rel[3:6]))
    idx = int(np.argmin(np.linalg.norm(curve - rel[:3].reshape(1, 3), axis=1)))
    theta = 2.0 * np.pi * idx / max(curve.shape[0] - 1, 1)
    a_r = float(radial_amplitude_km)
    a_c = float(cross_track_amplitude_km)
    phase = np.deg2rad(float(cross_track_phase_deg))
    expected = np.array(
        [
            -a_r * n * np.sin(theta),
            -2.0 * a_r * n * np.cos(theta),
            -a_c * n * np.sin(theta + phase),
        ],
        dtype=float,
    )
    return float(np.linalg.norm(rel[3:6] - expected))


def nmt_element_errors(
    relative_ric_state: np.ndarray,
    *,
    mean_motion_rad_s: np.ndarray | float,
    radial_amplitude_km: float,
    cross_track_amplitude_km: float,
    center_ric_km: np.ndarray,
) -> dict[str, np.ndarray]:
    rel = np.array(relative_ric_state, dtype=float)
    if rel.ndim == 1:
        rel = rel.reshape(1, -1)
    if rel.shape[1] < 6:
        raise ValueError("relative_ric_state must contain RIC position and velocity.")
    n_raw = np.array(mean_motion_rad_s, dtype=float).reshape(-1)
    if n_raw.size == 1:
        n = np.full(rel.shape[0], float(n_raw[0]), dtype=float)
    else:
        n = n_raw[: rel.shape[0]]
        if n.size < rel.shape[0]:
            n = np.pad(n, (0, rel.shape[0] - n.size), constant_values=np.nan)
    center = np.array(center_ric_km, dtype=float).reshape(3)
    pos = rel[:, :3] - center.reshape(1, 3)
    vel = rel[:, 3:6]
    valid_n = np.isfinite(n) & (np.abs(n) > 1.0e-12)
    radial_amp = np.full(rel.shape[0], np.nan, dtype=float)
    cross_amp = np.full(rel.shape[0], np.nan, dtype=float)
    drift_vel_err = np.full(rel.shape[0], np.nan, dtype=float)
    radial_amp[valid_n] = np.sqrt(pos[valid_n, 0] ** 2 + (vel[valid_n, 0] / n[valid_n]) ** 2)
    cross_amp[valid_n] = np.sqrt(pos[valid_n, 2] ** 2 + (vel[valid_n, 2] / n[valid_n]) ** 2)
    drift_vel_err[valid_n] = np.abs(vel[valid_n, 1] + 2.0 * n[valid_n] * pos[valid_n, 0])
    return {
        "radial_amplitude_km": radial_amp,
        "cross_track_amplitude_km": cross_amp,
        "radial_amplitude_error_km": np.abs(radial_amp - float(radial_amplitude_km)),
        "cross_track_amplitude_error_km": np.abs(cross_amp - float(cross_track_amplitude_km)),
        "drift_velocity_error_km_s": drift_vel_err,
    }


def _nmt_element_error_values(
    relative_ric_state: np.ndarray,
    *,
    mean_motion_rad_s: float,
    radial_amplitude_km: float,
    cross_track_amplitude_km: float,
    center_ric_km: np.ndarray,
    drift_velocity_error_km_s: float | None = None,
) -> dict[str, float]:
    rel = np.array(relative_ric_state, dtype=float).reshape(-1)
    center = np.array(center_ric_km, dtype=float).reshape(3)
    if rel.size < 6:
        raise ValueError("relative_ric_state must contain RIC position and velocity.")
    n = float(mean_motion_rad_s)
    if not np.isfinite(n) or abs(n) <= 1.0e-12:
        return {
            "radial_amplitude_km": float("nan"),
            "cross_track_amplitude_km": float("nan"),
            "radial_amplitude_error_km": float("nan"),
            "cross_track_amplitude_error_km": float("nan"),
            "drift_velocity_error_km_s": float("nan"),
        }
    pos = rel[:3] - center
    vel = rel[3:6]
    radial_amp = float(np.sqrt(pos[0] ** 2 + (vel[0] / n) ** 2))
    cross_amp = float(np.sqrt(pos[2] ** 2 + (vel[2] / n) ** 2))
    drift_error = (
        float(drift_velocity_error_km_s)
        if drift_velocity_error_km_s is not None and np.isfinite(float(drift_velocity_error_km_s))
        else abs(float(vel[1]) + 2.0 * n * float(pos[0]))
    )
    return {
        "radial_amplitude_km": radial_amp,
        "cross_track_amplitude_km": cross_amp,
        "radial_amplitude_error_km": abs(radial_amp - float(radial_amplitude_km)),
        "cross_track_amplitude_error_km": abs(cross_amp - float(cross_track_amplitude_km)),
        "drift_velocity_error_km_s": abs(drift_error),
    }


def _semimajor_axis_drift_velocity_error_km_s(
    target_state_eci: np.ndarray | None,
    chaser_state_eci: np.ndarray | None,
    *,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
) -> float | None:
    if target_state_eci is None or chaser_state_eci is None:
        return None
    target_a = _semimajor_axis_km(target_state_eci, mu_km3_s2=mu_km3_s2)
    chaser_a = _semimajor_axis_km(chaser_state_eci, mu_km3_s2=mu_km3_s2)
    if target_a is None or chaser_a is None or target_a <= 0.0:
        return None
    n = float(np.sqrt(float(mu_km3_s2) / (float(target_a) ** 3)))
    return float(abs(0.5 * n * (float(chaser_a) - float(target_a))))


def _semimajor_axis_km(state_eci: np.ndarray, *, mu_km3_s2: float = EARTH_MU_KM3_S2) -> float | None:
    state = np.array(state_eci, dtype=float).reshape(-1)
    if state.size < 6:
        return None
    r_norm = float(np.linalg.norm(state[:3]))
    v_norm = float(np.linalg.norm(state[3:6]))
    if not np.isfinite(r_norm) or not np.isfinite(v_norm) or r_norm <= 0.0:
        return None
    specific_energy = 0.5 * v_norm * v_norm - float(mu_km3_s2) / r_norm
    if not np.isfinite(specific_energy) or abs(specific_energy) <= 1.0e-12:
        return None
    return float(-float(mu_km3_s2) / (2.0 * specific_energy))


def _nmt_element_goal_error_km(
    *,
    radial_amplitude_error_km: float,
    cross_track_amplitude_error_km: float,
    include_radial: bool,
    include_cross_track: bool,
) -> float:
    values = []
    if include_radial:
        values.append(float(radial_amplitude_error_km))
    if include_cross_track:
        values.append(float(cross_track_amplitude_error_km))
    finite = [value for value in values if np.isfinite(value)]
    return float(max(finite)) if finite else float("nan")


def _nmt_element_goal_error_array(
    element_errors: dict[str, np.ndarray],
    *,
    include_radial: bool,
    include_cross_track: bool,
) -> np.ndarray:
    values: list[np.ndarray] = []
    if include_radial:
        values.append(np.array(element_errors["radial_amplitude_error_km"], dtype=float).reshape(-1))
    if include_cross_track:
        values.append(np.array(element_errors["cross_track_amplitude_error_km"], dtype=float).reshape(-1))
    if not values:
        first = next(iter(element_errors.values()), np.zeros(0, dtype=float))
        return np.zeros(np.array(first, dtype=float).reshape(-1).shape, dtype=float)
    return np.nanmax(np.vstack(values), axis=0)


def _final_nmt_element_values(element_errors: dict[str, np.ndarray] | None) -> dict[str, float]:
    keys = (
        "radial_amplitude_km",
        "cross_track_amplitude_km",
        "radial_amplitude_error_km",
        "cross_track_amplitude_error_km",
        "drift_velocity_error_km_s",
    )
    if element_errors is None:
        return {k: float("nan") for k in keys}
    return {k: float(np.array(element_errors[k], dtype=float).reshape(-1)[-1]) for k in keys}


def _format_optional_time(value: float | None) -> str:
    if value is None:
        return "Not Achieved"
    return f"{float(value):.1f} s"


def _format_distance_text(value_km: float) -> str:
    return format_distance_km(value_km)


def _format_speed_text(value_km_s: float) -> str:
    return format_speed_km_s(value_km_s)


def _format_signed_distance_text(value_km: float) -> str:
    sign = "+" if float(value_km) >= 0.0 else "-"
    return sign + format_distance_km(abs(float(value_km)))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _burn_axes_from_metadata(value: Any) -> tuple[str, ...]:
    if value is None or value is False:
        return ()
    if not isinstance(value, list):
        raise ValueError("metadata.game.training.required_burn_axes must be a list.")
    axes: list[str] = []
    for item in value:
        axis = _burn_axis_from_metadata_value(item)
        if axis not in axes:
            axes.append(axis)
    return tuple(axes)


def _burn_axis_from_metadata_value(value: Any) -> str:
    aliases = {
        "r": "radial",
        "radial": "radial",
        "i": "in_track",
        "in_track": "in_track",
        "in-track": "in_track",
        "intrack": "in_track",
        "along_track": "in_track",
        "along-track": "in_track",
        "c": "cross_track",
        "cross_track": "cross_track",
        "cross-track": "cross_track",
        "crosstrack": "cross_track",
    }
    key = str(value or "").strip().lower()
    axis = aliases.get(key)
    if axis is None:
        raise ValueError(f"Unknown burn axis '{value}'.")
    return axis


def _axis_descriptions_from_metadata(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("metadata.game.training.axis_descriptions must be a mapping.")
    descriptions: dict[str, str] = {}
    for raw_axis, raw_text in value.items():
        axis = _burn_axis_from_metadata_value(raw_axis)
        text = str(raw_text or "").strip()
        if text:
            descriptions[axis] = text
    return descriptions


def _string_mapping_from_metadata(value: Any, field_name: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping.")
    return {str(key or "").strip(): str(text or "").strip() for key, text in value.items() if str(text or "").strip()}


def _guided_tutorial_burns_from_metadata(value: Any) -> tuple[GuidedTutorialBurnConfig, ...]:
    if value is None or value is False:
        return ()
    if not isinstance(value, list):
        raise ValueError("metadata.game.training.guided_tutorial_burns must be a list.")
    stages: list[GuidedTutorialBurnConfig] = []
    for idx, raw_value in enumerate(value, start=1):
        if not isinstance(raw_value, dict):
            raise ValueError("metadata.game.training.guided_tutorial_burns entries must be mappings.")
        axis = _burn_axis_from_metadata_value(raw_value.get("axis", "in_track"))
        sign = _sign_from_metadata(raw_value.get("sign", 1))
        name = str(raw_value.get("name", "") or "").strip()
        if not name:
            prefix = "plus" if sign >= 0 else "minus"
            name = f"{prefix}_{axis}_{idx}"
        delta_v = float(raw_value.get("delta_v_m_s", 0.25) or 0.25)
        if not np.isfinite(delta_v) or delta_v <= 0.0:
            raise ValueError("metadata.game.training.guided_tutorial_burns.delta_v_m_s must be positive.")
        stages.append(
            GuidedTutorialBurnConfig(
                name=name,
                axis=axis,
                sign=sign,
                delta_v_m_s=delta_v,
                label=str(raw_value.get("label", "") or "").strip(),
                hint=str(raw_value.get("hint", "") or "").strip(),
            )
        )
    return tuple(stages)


def _guided_tutorial_speed_step_from_metadata(value: Any) -> GuidedTutorialSpeedStepConfig | None:
    if value is None or value is False:
        return None
    if value is True:
        raw: dict[str, Any] = {}
    elif isinstance(value, dict):
        raw = value
    else:
        raise ValueError("metadata.game.training.guided_tutorial_speed_step must be a mapping.")
    target = float(raw.get("target_speed_multiplier", 10.0) or 10.0)
    if not np.isfinite(target) or target <= 0.0:
        raise ValueError("metadata.game.training.guided_tutorial_speed_step.target_speed_multiplier must be positive.")
    return GuidedTutorialSpeedStepConfig(
        name=str(raw.get("name", "speed_multiplier") or "speed_multiplier").strip(),
        after_burn_name=str(raw.get("after_burn_name", "") or "").strip(),
        target_speed_multiplier=target,
        label=str(raw.get("label", "") or "").strip() or f"Speed {target:g}x",
        hint=str(raw.get("hint", "") or "").strip(),
    )


def _sign_from_metadata(value: Any) -> int:
    if isinstance(value, str):
        key = value.strip().lower()
        if key in {"+", "+1", "plus", "positive", "pos"}:
            return 1
        if key in {"-", "-1", "minus", "negative", "neg"}:
            return -1
    return 1 if float(value) >= 0.0 else -1


def _required_phase_burns_from_metadata(value: Any) -> tuple[RequiredPhaseBurnConfig, ...]:
    if value is None or value is False:
        return ()
    if not isinstance(value, list):
        raise ValueError("metadata.game.training.required_phase_burns must be a list.")
    burns: list[RequiredPhaseBurnConfig] = []
    for idx, raw_value in enumerate(value, start=1):
        if not isinstance(raw_value, dict):
            raise ValueError("metadata.game.training.required_phase_burns entries must be mappings.")
        axis = _burn_axis_from_metadata_value(raw_value.get("axis", "cross_track"))
        name = str(raw_value.get("name", f"{axis}_phase_burn_{idx}") or f"{axis}_phase_burn_{idx}")
        burns.append(
            RequiredPhaseBurnConfig(
                name=name,
                axis=axis,
                radial_abs_km=float(raw_value["radial_abs_km"]),
                radial_tolerance_km=float(raw_value.get("radial_tolerance_km", 0.2)),
                max_abs_intrack_km=float(raw_value.get("max_abs_intrack_km", 0.35)),
                threshold_km_s2=float(raw_value.get("threshold_km_s2", 1.0e-10)),
                min_component_fraction=float(
                    raw_value.get("min_component_fraction", _BURN_AXIS_MIN_COMPONENT_FRACTION)
                ),
            )
        )
    return tuple(burns)


def _forbidden_regions_from_metadata(value: Any) -> tuple[ForbiddenRegionConfig, ...]:
    if value is None or value is False:
        return ()
    if not isinstance(value, list):
        raise ValueError("metadata.game.training.forbidden_regions must be a list.")
    regions: list[ForbiddenRegionConfig] = []
    for idx, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            raise ValueError("Each forbidden region must be a mapping.")
        region = ForbiddenRegionConfig.from_mapping(item, index=idx)
        if region.kind == "box" and np.any(region.min_ric_km > region.max_ric_km):
            raise ValueError(f"Forbidden region '{region.name}' has min_ric_km greater than max_ric_km.")
        if region.kind == "annular_sector":
            _validate_annular_sector_region(region)
        elif region.kind == "cylinder":
            _validate_cylinder_region(region)
        elif region.kind == "sphere":
            _validate_sphere_region(region)
        elif region.kind != "box":
            raise ValueError(f"Forbidden region '{region.name}' has unknown kind '{region.kind}'.")
        regions.append(region)
    return tuple(regions)


def _approach_gates_from_metadata(value: Any) -> tuple[ApproachGateConfig, ...]:
    if value is None or value is False:
        return ()
    if not isinstance(value, list):
        raise ValueError("metadata.game.training.approach_gates must be a list.")
    gates: list[ApproachGateConfig] = []
    for idx, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            raise ValueError("Each approach gate must be a mapping.")
        gate = ApproachGateConfig.from_mapping(item, index=idx)
        if gate.radial_tolerance_km <= 0.0:
            raise ValueError(f"Approach gate '{gate.name}' radial_tolerance_km must be positive.")
        gates.append(gate)
    return tuple(gates)


def _inspection_gates_from_metadata(value: Any) -> tuple[InspectionGateConfig, ...]:
    if value is None or value is False:
        return ()
    if not isinstance(value, list):
        raise ValueError("metadata.game.training.inspection_gates must be a list.")
    gates: list[InspectionGateConfig] = []
    for idx, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            raise ValueError("Each inspection gate must be a mapping.")
        gate = InspectionGateConfig.from_mapping(item, index=idx)
        if np.any(np.array(gate.half_width_ric_km, dtype=float) <= 0.0):
            raise ValueError(f"Inspection gate '{gate.name}' half_width_ric_km values must be positive.")
        gates.append(gate)
    return tuple(gates)


def _sun_angle_constraints_from_metadata(value: Any) -> tuple[SunAngleConstraintConfig, ...]:
    if value is None or value is False:
        return ()
    if not isinstance(value, list):
        raise ValueError("metadata.game.training.sun_angle_constraints must be a list.")
    constraints: list[SunAngleConstraintConfig] = []
    for idx, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            raise ValueError("Each sun angle constraint must be a mapping.")
        constraint = SunAngleConstraintConfig.from_mapping(item, index=idx)
        if float(constraint.allowed_half_angle_deg) <= 0.0 or float(constraint.allowed_half_angle_deg) >= 180.0:
            raise ValueError(f"Sun angle constraint '{constraint.name}' allowed_half_angle_deg must be between 0 and 180.")
        if constraint.allowed_center_mode not in {"configured", "anti_sun", "antisun", "opposite_sun", "sun", "toward_sun"}:
            raise ValueError(
                f"Sun angle constraint '{constraint.name}' allowed_center_mode must be configured, anti_sun, or sun."
            )
        if constraint.min_range_km is not None and float(constraint.min_range_km) < 0.0:
            raise ValueError(f"Sun angle constraint '{constraint.name}' min_range_km must be nonnegative.")
        if constraint.max_range_km is not None and float(constraint.max_range_km) <= 0.0:
            raise ValueError(f"Sun angle constraint '{constraint.name}' max_range_km must be positive.")
        if (
            constraint.min_range_km is not None
            and constraint.max_range_km is not None
            and float(constraint.min_range_km) >= float(constraint.max_range_km)
        ):
            raise ValueError(f"Sun angle constraint '{constraint.name}' min_range_km must be less than max_range_km.")
        constraints.append(constraint)
    return tuple(constraints)


def _approach_gate_status(gates: tuple[ApproachGateConfig, ...], relative_ric_state: np.ndarray) -> dict[str, tuple[str, ...]]:
    satisfied: list[str] = []
    violated: list[str] = []
    missed: list[str] = []
    required_violated: list[str] = []
    required_missed: list[str] = []
    for gate in gates:
        near = gate.samples_near_gate(relative_ric_state)
        ok = gate.samples_satisfying_gate(relative_ric_state)
        if bool(np.any(ok)):
            satisfied.append(gate.name)
        elif bool(np.any(near)):
            violated.append(gate.name)
            if gate.required:
                required_violated.append(gate.name)
        else:
            missed.append(gate.name)
            if gate.required:
                required_missed.append(gate.name)
    return {
        "satisfied": tuple(satisfied),
        "violated": tuple(violated),
        "missed": tuple(missed),
        "required_violated": tuple(required_violated),
        "required_missed": tuple(required_missed),
    }


def _inspection_gate_status(gates: tuple[InspectionGateConfig, ...], relative_ric_state: np.ndarray) -> dict[str, Any]:
    if not gates:
        return {"satisfied": (), "completed_idx": None}
    rel = np.array(relative_ric_state, dtype=float)
    if rel.ndim == 1:
        rel = rel.reshape(1, -1)
    if rel.shape[1] < 6:
        raise ValueError("relative_ric_state must contain RIC position and velocity.")
    satisfied: list[str] = []
    completed_idx: int | None = None
    for sample_idx in range(rel.shape[0]):
        if len(satisfied) >= len(gates):
            break
        for gate in gates:
            if gate.name in satisfied:
                continue
            current_hits_gate = bool(gate.samples_satisfying_gate(rel[sample_idx : sample_idx + 1])[0])
            segment_hits_gate = bool(
                sample_idx > 0 and gate.segment_satisfies_gate(rel[sample_idx - 1], rel[sample_idx])
            )
            if current_hits_gate or segment_hits_gate:
                satisfied.append(gate.name)
                if len(satisfied) >= len(gates):
                    completed_idx = sample_idx
                    break
    return {"satisfied": tuple(satisfied), "completed_idx": completed_idx}


def _position_segment_intersects_box(
    start_ric_km: np.ndarray, end_ric_km: np.ndarray, *, center: np.ndarray, half_width: np.ndarray
) -> bool:
    start = np.array(start_ric_km, dtype=float).reshape(3)
    end = np.array(end_ric_km, dtype=float).reshape(3)
    lo = np.array(center, dtype=float).reshape(3) - np.array(half_width, dtype=float).reshape(3)
    hi = np.array(center, dtype=float).reshape(3) + np.array(half_width, dtype=float).reshape(3)
    delta = end - start
    t_min = 0.0
    t_max = 1.0
    for axis in range(3):
        if abs(float(delta[axis])) <= 1.0e-12:
            if start[axis] < lo[axis] or start[axis] > hi[axis]:
                return False
            continue
        inv_delta = 1.0 / float(delta[axis])
        t1 = float((lo[axis] - start[axis]) * inv_delta)
        t2 = float((hi[axis] - start[axis]) * inv_delta)
        t_near = min(t1, t2)
        t_far = max(t1, t2)
        t_min = max(t_min, t_near)
        t_max = min(t_max, t_far)
        if t_min > t_max:
            return False
    return True


def _position_segment_intersects_bounds(
    start_ric_km: np.ndarray,
    end_ric_km: np.ndarray,
    *,
    lower: np.ndarray,
    upper: np.ndarray,
) -> bool:
    start = np.asarray(start_ric_km, dtype=float).reshape(3)
    end = np.asarray(end_ric_km, dtype=float).reshape(3)
    lo = np.asarray(lower, dtype=float).reshape(3)
    hi = np.asarray(upper, dtype=float).reshape(3)
    delta = end - start
    t_min = 0.0
    t_max = 1.0
    for axis in range(3):
        if abs(float(delta[axis])) <= 1.0e-12:
            if start[axis] < lo[axis] or start[axis] > hi[axis]:
                return False
            continue
        if np.isfinite(lo[axis]):
            t_min = max(t_min, float((lo[axis] - start[axis]) / delta[axis])) if delta[axis] > 0 else t_min
            t_max = min(t_max, float((lo[axis] - start[axis]) / delta[axis])) if delta[axis] < 0 else t_max
        if np.isfinite(hi[axis]):
            t_max = min(t_max, float((hi[axis] - start[axis]) / delta[axis])) if delta[axis] > 0 else t_max
            t_min = max(t_min, float((hi[axis] - start[axis]) / delta[axis])) if delta[axis] < 0 else t_min
        if t_min > t_max:
            return False
    return t_max >= 0.0 and t_min <= 1.0


def _position_segment_intersects_cylinder(
    start_ric_km: np.ndarray,
    end_ric_km: np.ndarray,
    *,
    center: np.ndarray,
    axis: int,
    radius_km: float | None,
    height_km: float | None,
) -> bool:
    if radius_km is None or height_km is None:
        return False
    start = np.asarray(start_ric_km, dtype=float).reshape(3) - np.asarray(center, dtype=float).reshape(3)
    end = np.asarray(end_ric_km, dtype=float).reshape(3) - np.asarray(center, dtype=float).reshape(3)
    delta = end - start
    half_height = max(float(height_km), 0.0) / 2.0
    axial = _linear_interval_in_bounds(float(start[axis]), float(delta[axis]), -half_height, half_height)
    if axial is None:
        return False
    cross_axes = tuple(idx for idx in range(3) if idx != int(axis))
    p = start[list(cross_axes)]
    d = delta[list(cross_axes)]
    radial = _quadratic_radius_interval(p, d, max(float(radius_km), 0.0))
    if radial is None:
        return False
    return max(axial[0], radial[0], 0.0) <= min(axial[1], radial[1], 1.0)


def _position_segment_intersects_annular_sector(
    start_ric_km: np.ndarray,
    end_ric_km: np.ndarray,
    *,
    region: ForbiddenRegionConfig,
) -> bool:
    if region.inner_radius_km is None or region.outer_radius_km is None:
        return False
    start = np.asarray(start_ric_km, dtype=float).reshape(3)
    end = np.asarray(end_ric_km, dtype=float).reshape(3)
    delta = end - start
    x_axis, y_axis, out_axis = _plane_axes(region.plane)
    center = np.asarray(region.center_ric_km, dtype=float).reshape(3)
    p = (start - center)[[x_axis, y_axis]]
    d = delta[[x_axis, y_axis]]
    candidates = {0.0, 1.0}
    for radius in (float(region.inner_radius_km), float(region.outer_radius_km)):
        candidates.update(_quadratic_boundary_roots(p, d, max(radius, 0.0)))
    if region.max_abs_out_of_plane_km is not None:
        limit = max(float(region.max_abs_out_of_plane_km), 0.0)
        p_out = float(start[out_axis] - center[out_axis])
        d_out = float(delta[out_axis])
        if abs(d_out) > 1.0e-12:
            candidates.update(((limit - p_out) / d_out, (-limit - p_out) / d_out))
    if region.angle_min_deg is not None or region.angle_max_deg is not None:
        start_deg = 0.0 if region.angle_min_deg is None else float(region.angle_min_deg)
        end_deg = 360.0 if region.angle_max_deg is None else float(region.angle_max_deg)
        for angle_deg in (start_deg, end_deg):
            ray = np.array([np.cos(np.deg2rad(angle_deg)), np.sin(np.deg2rad(angle_deg))], dtype=float)
            denom = float(d[0] * ray[1] - d[1] * ray[0])
            if abs(denom) > 1.0e-12:
                candidates.add(float((p[1] * ray[0] - p[0] * ray[1]) / denom))
    ordered = sorted(float(value) for value in candidates if np.isfinite(value) and -1.0e-12 <= value <= 1.0 + 1.0e-12)
    probes = ordered + [(left + right) / 2.0 for left, right in zip(ordered, ordered[1:], strict=False)]
    points = np.vstack([start + np.clip(value, 0.0, 1.0) * delta for value in probes])
    return bool(np.any(region.contains_positions(points)))


def _linear_interval_in_bounds(value: float, delta: float, lower: float, upper: float) -> tuple[float, float] | None:
    if abs(delta) <= 1.0e-12:
        return (0.0, 1.0) if lower <= value <= upper else None
    t0 = (lower - value) / delta
    t1 = (upper - value) / delta
    return (min(t0, t1), max(t0, t1))


def _quadratic_boundary_roots(position: np.ndarray, delta: np.ndarray, radius: float) -> tuple[float, ...]:
    a = float(np.dot(delta, delta))
    b = 2.0 * float(np.dot(position, delta))
    c = float(np.dot(position, position) - radius * radius)
    if a <= 1.0e-18:
        return ()
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return ()
    root = float(np.sqrt(max(discriminant, 0.0)))
    return ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a))


def _quadratic_radius_interval(
    position: np.ndarray,
    delta: np.ndarray,
    radius: float,
) -> tuple[float, float] | None:
    roots = _quadratic_boundary_roots(position, delta, radius)
    if roots:
        return (min(roots), max(roots))
    return (0.0, 1.0) if float(np.linalg.norm(position)) <= radius else None


def _hard_speed_limit_violated(relative_ric_state: np.ndarray, *, radius_km: float, speed_limit_km_s: float) -> bool:
    rel = np.array(relative_ric_state, dtype=float)
    if rel.ndim == 1:
        rel = rel.reshape(1, -1)
    if rel.shape[1] < 6:
        raise ValueError("relative_ric_state must contain RIC position and velocity.")
    positions = rel[:, :3]
    velocities = rel[:, 3:6]
    ranges = np.linalg.norm(positions, axis=1)
    speeds = np.linalg.norm(velocities, axis=1)
    if bool(np.any((ranges <= float(radius_km)) & (speeds > float(speed_limit_km_s)))):
        return True
    if rel.shape[0] < 2:
        return False
    for idx in range(1, rel.shape[0]):
        interval = _position_segment_sphere_interval(
            positions[idx - 1],
            positions[idx],
            radius_km=float(radius_km),
        )
        if interval is None:
            continue
        u0, u1 = interval
        v0 = velocities[idx - 1]
        dv = velocities[idx] - velocities[idx - 1]
        entry_speed = float(np.linalg.norm(v0 + dv * u0))
        exit_speed = float(np.linalg.norm(v0 + dv * u1))
        if max(entry_speed, exit_speed) > float(speed_limit_km_s):
            return True
    return False


def _hard_speed_limit_sample_violated(
    previous_relative_ric_state: np.ndarray | None,
    current_relative_ric_state: np.ndarray,
    *,
    radius_km: float,
    speed_limit_km_s: float,
) -> bool:
    current = np.array(current_relative_ric_state, dtype=float).reshape(6)
    radius = float(radius_km)
    speed_limit = float(speed_limit_km_s)
    if float(np.linalg.norm(current[:3])) <= radius and float(np.linalg.norm(current[3:6])) > speed_limit:
        return True
    if previous_relative_ric_state is None:
        return False
    previous = np.array(previous_relative_ric_state, dtype=float).reshape(6)
    interval = _position_segment_sphere_interval(previous[:3], current[:3], radius_km=radius)
    if interval is None:
        return False
    u0, u1 = interval
    v0 = previous[3:6]
    dv = current[3:6] - previous[3:6]
    entry_speed = float(np.linalg.norm(v0 + dv * u0))
    exit_speed = float(np.linalg.norm(v0 + dv * u1))
    return bool(max(entry_speed, exit_speed) > speed_limit)


def _position_segment_sphere_interval(
    start_ric_km: np.ndarray,
    end_ric_km: np.ndarray,
    *,
    radius_km: float,
) -> tuple[float, float] | None:
    start = np.array(start_ric_km, dtype=float).reshape(3)
    end = np.array(end_ric_km, dtype=float).reshape(3)
    radius = float(radius_km)
    if radius < 0.0:
        return None
    inside_start = float(np.linalg.norm(start)) <= radius
    inside_end = float(np.linalg.norm(end)) <= radius
    if inside_start and inside_end:
        return (0.0, 1.0)
    delta = end - start
    a = float(np.dot(delta, delta))
    if a <= 1.0e-18:
        return (0.0, 1.0) if inside_start else None
    b = 2.0 * float(np.dot(start, delta))
    c = float(np.dot(start, start) - radius * radius)
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None
    sqrt_disc = float(np.sqrt(max(disc, 0.0)))
    t0 = (-b - sqrt_disc) / (2.0 * a)
    t1 = (-b + sqrt_disc) / (2.0 * a)
    if t0 > t1:
        t0, t1 = t1, t0
    entry = max(0.0, t0)
    exit_ = min(1.0, t1)
    if inside_start:
        entry = 0.0
    if inside_end:
        exit_ = 1.0
    if entry <= exit_ and t1 >= 0.0 and t0 <= 1.0:
        return (float(entry), float(exit_))
    return None


def _ric_bound_array(value: Any, *, default: float, field_name: str) -> np.ndarray:
    if value is None:
        return np.full(3, float(default), dtype=float)
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"Forbidden region {field_name} must be a length-3 list.")
    vals = [float(default) if item is None else float(item) for item in value]
    return np.array(vals, dtype=float).reshape(3)


def _unit_ric_array(value: Any, *, field_name: str) -> np.ndarray:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"Sun angle constraint {field_name} must be a length-3 list.")
    vec = np.array(value, dtype=float).reshape(3)
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"Sun angle constraint {field_name} must be nonzero.")
    return vec / norm


def _unit_direction_rows(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos = np.array(value, dtype=float)
    if pos.ndim == 1:
        pos = pos.reshape(1, -1)
    dirs = np.zeros((pos.shape[0], 3), dtype=float)
    if pos.shape[1] < 3:
        return dirs, np.zeros(pos.shape[0], dtype=bool)
    norm = np.linalg.norm(pos[:, :3], axis=1)
    valid = np.isfinite(norm) & (norm > 0.0)
    dirs[valid, :] = pos[valid, :3] / norm[valid].reshape(-1, 1)
    return dirs, valid


def _validate_annular_sector_region(region: ForbiddenRegionConfig) -> None:
    _plane_axes(region.plane)
    if region.inner_radius_km is None or region.outer_radius_km is None:
        raise ValueError(f"Forbidden region '{region.name}' annular_sector requires inner_radius_km and outer_radius_km.")
    if float(region.inner_radius_km) < 0.0:
        raise ValueError(f"Forbidden region '{region.name}' inner_radius_km must be nonnegative.")
    if float(region.outer_radius_km) <= float(region.inner_radius_km):
        raise ValueError(f"Forbidden region '{region.name}' outer_radius_km must be greater than inner_radius_km.")
    if region.max_abs_out_of_plane_km is not None and float(region.max_abs_out_of_plane_km) < 0.0:
        raise ValueError(f"Forbidden region '{region.name}' max_abs_out_of_plane_km must be nonnegative.")


def _validate_cylinder_region(region: ForbiddenRegionConfig) -> None:
    _axis_index(region.axis)
    if region.radius_km is None or region.height_km is None:
        raise ValueError(f"Forbidden region '{region.name}' cylinder requires radius_km and height_km.")
    if float(region.radius_km) <= 0.0:
        raise ValueError(f"Forbidden region '{region.name}' radius_km must be positive.")
    if float(region.height_km) <= 0.0:
        raise ValueError(f"Forbidden region '{region.name}' height_km must be positive.")


def _validate_sphere_region(region: ForbiddenRegionConfig) -> None:
    if region.radius_km is None:
        raise ValueError(f"Forbidden region '{region.name}' sphere requires radius_km.")
    if float(region.radius_km) <= 0.0:
        raise ValueError(f"Forbidden region '{region.name}' radius_km must be positive.")


def _axis_index(axis: str) -> int:
    key = str(axis or "").strip().upper()
    if key == "R":
        return 0
    if key == "I":
        return 1
    if key == "C":
        return 2
    raise ValueError(f"Forbidden region axis must be one of R, I, or C; got '{axis}'.")


def _plane_axes(plane: str) -> tuple[int, int, int]:
    key = str(plane or "").strip().upper()
    if key == "RI":
        return 1, 0, 2
    if key == "RC":
        return 2, 0, 1
    if key == "IC":
        return 1, 2, 0
    raise ValueError(f"Forbidden region plane must be one of RI, RC, or IC; got '{plane}'.")


def _angles_in_range_deg(angles_deg: np.ndarray, start_deg: float, end_deg: float) -> np.ndarray:
    span = float(end_deg) - float(start_deg)
    if span >= 360.0:
        return np.ones_like(np.array(angles_deg, dtype=float), dtype=bool)
    while span < 0.0:
        span += 360.0
    relative = (np.array(angles_deg, dtype=float) - float(start_deg)) % 360.0
    return relative <= span


def _plot_planes_from_metadata(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("Forbidden region plot_planes must be a list.")
    valid = {"RI", "RC", "IC"}
    planes = tuple(str(item).strip().upper() for item in value if str(item).strip())
    unknown = [item for item in planes if item not in valid]
    if unknown:
        raise ValueError(f"Forbidden region plot_planes contains unknown plane(s): {', '.join(unknown)}.")
    return planes


def _sampled_dwell_time_s(mask: np.ndarray, time_s: np.ndarray) -> float:
    inside = np.array(mask, dtype=bool).reshape(-1)
    t = np.array(time_s, dtype=float).reshape(-1)
    n = min(inside.size, t.size)
    if n < 2:
        return 0.0
    dt = np.diff(t[:n])
    valid = np.isfinite(dt) & (dt > 0.0)
    if not np.any(valid):
        return 0.0
    return float(np.sum(dt[valid] * inside[: n - 1][valid]))


def _integrated_delta_v_m_s(thrust_km_s2: np.ndarray, time_s: np.ndarray) -> float:
    thrust = np.array(thrust_km_s2, dtype=float)
    t = np.array(time_s, dtype=float).reshape(-1)
    n = min(thrust.shape[0], t.size)
    if n < 2:
        return 0.0
    # Snapshot i reports the command applied during the interval ending at t[i].
    accel = np.linalg.norm(thrust[1:n, :], axis=1)
    dt = np.diff(t[:n])
    valid = np.isfinite(accel) & np.isfinite(dt) & (dt > 0.0)
    if not np.any(valid):
        return 0.0
    return float(np.sum(accel[valid] * dt[valid]) * 1.0e3)
