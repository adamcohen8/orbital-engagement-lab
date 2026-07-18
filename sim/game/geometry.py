# ruff: noqa: F401,F403,F405,I001
from .dashboard_common import *

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

__all__ = [name for name in globals() if not name.startswith("__")]
