# ruff: noqa: F401,F403,F405,I001
from .dashboard_common import *
from .geometry import *
from .prediction import *
from .camera import *

class DashboardPredictionMixin:
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

    def _coast_prediction(self) -> np.ndarray:
        if not bool(getattr(self, "show_coast_prediction", True)) or not self.rel_hist:
            return np.empty((0, 6), dtype=float)
        live_accel = np.array(getattr(self, "live_prediction_accel_ric_km_s2", np.zeros(3)), dtype=float).reshape(3)
        active_burn = bool(np.linalg.norm(live_accel) > float(self.burn_marker_threshold_km_s2))
        return self._coast_prediction_from_cached(
            "chaser",
            np.array(self.rel_hist[-1], dtype=float).reshape(6),
            active_burn=active_burn,
        )

    def _aerodynamic_sprite_rotation_deg(self, *, x_axis: int, y_axis: int) -> float:
        if not bool(getattr(self, "aerodynamic_control_enabled", False)):
            return 0.0
        if (x_axis, y_axis) == (2, 0):
            return float(getattr(self, "aerodynamic_lift_bank_angle_deg", 0.0))
        if (x_axis, y_axis) != (1, 0):
            return 0.0
        bc_min = float(getattr(self, "aerodynamic_ballistic_coefficient_min_kg_m2", 40.0))
        bc_max = float(getattr(self, "aerodynamic_ballistic_coefficient_max_kg_m2", 200.0))
        bc = float(getattr(self, "aerodynamic_ballistic_coefficient_kg_m2", 100.0))
        span = max(bc_max - bc_min, 1.0e-9)
        low_bc_fraction = 1.0 - float(np.clip((bc - bc_min) / span, 0.0, 1.0))
        return low_bc_fraction * float(getattr(self, "aerodynamic_ri_pitch_max_deg", 24.0))

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
