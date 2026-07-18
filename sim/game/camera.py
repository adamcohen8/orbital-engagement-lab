# ruff: noqa: F401,F403,F405,I001
from .dashboard_common import *
from .geometry import *
from .prediction import *

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

__all__ = [name for name in globals() if not name.startswith("__")]
