# ruff: noqa: F401,I001
from __future__ import annotations

import sys

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


def _dashboard_dep(name, default):
    facade = sys.modules.get("sim.game.pygame_dashboard")
    return getattr(facade, name, default)


__all__ = [name for name in globals() if not name.startswith("__")]
