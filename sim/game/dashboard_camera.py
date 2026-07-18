# ruff: noqa: F401,F403,F405,I001
from .dashboard_common import *
from .geometry import *
from .prediction import *
from .camera import *

class DashboardCameraMixin:
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
