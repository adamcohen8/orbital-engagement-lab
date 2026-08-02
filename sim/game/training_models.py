# ruff: noqa: F401,F403,F405,I001
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from sim.api import SimulationSnapshot
from sim.dynamics.orbit.epoch import resolve_sun_moon_positions
from sim.game.training_geometry import *
from sim.utils.frames import ric_dcm_ir_from_rv

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
            required=_metadata_bool(raw.get("required", True), f"approach_gates[{index}].required"),
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
            dynamic_sun=_metadata_bool(raw.get("dynamic_sun", False), f"sun_angle_constraints[{index}].dynamic_sun"),
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
        if goal.size != 3 or not np.all(np.isfinite(goal)):
            raise ValueError("metadata.game.training.goal_relative_ric_km must be length 3.")
        nmt_center = np.array(raw.get("goal_nmt_center_ric_km", [0.0, 0.0, 0.0]), dtype=float).reshape(-1)
        if nmt_center.size != 3 or not np.all(np.isfinite(nmt_center)):
            raise ValueError("metadata.game.training.goal_nmt_center_ric_km must be length 3.")
        return cls(
            enabled=_metadata_bool(raw.get("enabled", True), "metadata.game.training.enabled"),
            scenario_id=str(raw.get("scenario_id", "") or ""),
            level_name=str(game_cfg.get("level_name", "") or ""),
            learning_goal=str(raw.get("learning_goal", "") or ""),
            relative_frame=_training_relative_frame(
                raw.get("relative_frame", game_cfg.get("relative_frame", "ric"))
            ),
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
            fail_on_delta_v_budget=_metadata_bool(
                raw.get("fail_on_delta_v_budget", True), "metadata.game.training.fail_on_delta_v_budget"
            ),
            coast_chaser_after_delta_v_budget=_metadata_bool(
                raw.get("coast_chaser_after_delta_v_budget", False),
                "metadata.game.training.coast_chaser_after_delta_v_budget",
            ),
            survival_goal=_metadata_bool(raw.get("survival_goal", False), "metadata.game.training.survival_goal"),
            sandbox_mode=_metadata_bool(raw.get("sandbox_mode", False), "metadata.game.training.sandbox_mode"),
            required_burn_axes=_burn_axes_from_metadata(raw.get("required_burn_axes")),
            required_burn_axis_threshold_km_s2=float(raw.get("required_burn_axis_threshold_km_s2", 1.0e-10)),
            required_burn_axis_min_component_fraction=float(
                raw.get("required_burn_axis_min_component_fraction", _BURN_AXIS_MIN_COMPONENT_FRACTION)
            ),
            required_phase_burns=_required_phase_burns_from_metadata(raw.get("required_phase_burns")),
            require_speed_multiplier_change=_metadata_bool(
                raw.get("require_speed_multiplier_change", False),
                "metadata.game.training.require_speed_multiplier_change",
            ),
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

def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    if not np.isfinite(parsed):
        raise ValueError("Training numeric metadata must contain finite values.")
    return parsed


def _metadata_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"{field_name} must be a boolean.")


def _training_relative_frame(value: Any) -> str:
    frame = str(value or "ric").strip().lower().replace("-", "_")
    allowed = {
        "ric",
        "cislunar",
        "cislunar_l1",
        "earth_moon_rotating",
        "cr3bp",
        "cr3bp_rotating",
        "moon_ric",
        "lunar_ric",
        "target_moon_ric",
        "target_lunar_ric",
    }
    if frame not in allowed:
        raise ValueError(f"Unknown metadata.game.training.relative_frame {frame!r}.")
    return frame


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


__all__ = [name for name in globals() if not name.startswith("__")]
