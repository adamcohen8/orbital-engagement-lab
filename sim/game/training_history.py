# ruff: noqa: F401,F403,F405,I001
from .training_models import *
from .scoring import *
from .training_geometry import *
from .coaching import *
from .criteria import *
from sim.dynamics.orbit.cr3bp import EARTH_MOON_MEAN_MOTION_RAD_S


def _aerodynamic_control_telemetry(
    provider: Any | None,
    *,
    chaser_state: np.ndarray,
) -> dict[str, float | bool]:
    empty: dict[str, float | bool] = {
        "ballistic_coefficient_kg_m2": float("nan"),
        "drag_area_m2": float("nan"),
        "lift_coefficient": float("nan"),
        "lift_area_m2": float("nan"),
        "lift_bank_angle_deg": float("nan"),
        "control_active": False,
    }
    mode = str(getattr(provider, "control_mode", "") or "").strip().lower()
    if mode not in {"aerodynamic", "aero", "aero_control", "aerodynamic_control"}:
        return empty
    values = dict(empty)
    try:
        bc = float(provider.ballistic_coefficient_kg_m2)
        drag_coefficient = float(provider.aerodynamic_drag_coefficient)
        mass_kg = float(chaser_state[13]) if chaser_state.size > 13 else float("nan")
        values.update(
            {
                "ballistic_coefficient_kg_m2": bc,
                "drag_area_m2": mass_kg / (drag_coefficient * bc),
                "lift_coefficient": float(provider.aerodynamic_lift_coefficient),
                "lift_area_m2": float(provider.aerodynamic_lift_area_m2),
                "lift_bank_angle_deg": float(provider.lift_bank_angle_deg),
            }
        )
    except (AttributeError, IndexError, TypeError, ValueError, ZeroDivisionError):
        return empty
    state = getattr(provider, "command_state", None)
    values["control_active"] = bool(
        state is not None
        and (abs(float(getattr(state, "pitch", 0.0))) > 1.0e-9 or abs(float(getattr(state, "roll", 0.0))) > 1.0e-9)
    )
    for key in (
        "ballistic_coefficient_kg_m2",
        "drag_area_m2",
        "lift_coefficient",
        "lift_area_m2",
        "lift_bank_angle_deg",
    ):
        if not np.isfinite(float(values[key])):
            values[key] = float("nan")
    return values


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
        self.aerodynamic_ballistic_coefficient_hist: list[float] = []
        self.aerodynamic_drag_area_hist: list[float] = []
        self.aerodynamic_lift_coefficient_hist: list[float] = []
        self.aerodynamic_lift_area_hist: list[float] = []
        self.aerodynamic_lift_bank_angle_hist: list[float] = []
        self.aerodynamic_control_active_hist: list[bool] = []
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
        self._aerodynamic_ballistic_coefficient_array = np.zeros(0, dtype=float)
        self._aerodynamic_drag_area_array = np.zeros(0, dtype=float)
        self._aerodynamic_lift_coefficient_array = np.zeros(0, dtype=float)
        self._aerodynamic_lift_area_array = np.zeros(0, dtype=float)
        self._aerodynamic_lift_bank_angle_array = np.zeros(0, dtype=float)
        self._aerodynamic_control_active_array = np.zeros(0, dtype=bool)
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
        self.aerodynamic_ballistic_coefficient_hist.clear()
        self.aerodynamic_drag_area_hist.clear()
        self.aerodynamic_lift_coefficient_hist.clear()
        self.aerodynamic_lift_area_hist.clear()
        self.aerodynamic_lift_bank_angle_hist.clear()
        self.aerodynamic_control_active_hist.clear()
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

    def record(self, snapshot: SimulationSnapshot, *, control_telemetry_provider: Any | None = None) -> None:
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
        aerodynamic = _aerodynamic_control_telemetry(
            control_telemetry_provider,
            chaser_state=np.array(chaser, dtype=float).reshape(-1),
        )
        self.aerodynamic_ballistic_coefficient_hist.append(aerodynamic["ballistic_coefficient_kg_m2"])
        self.aerodynamic_drag_area_hist.append(aerodynamic["drag_area_m2"])
        self.aerodynamic_lift_coefficient_hist.append(aerodynamic["lift_coefficient"])
        self.aerodynamic_lift_area_hist.append(aerodynamic["lift_area_m2"])
        self.aerodynamic_lift_bank_angle_hist.append(aerodynamic["lift_bank_angle_deg"])
        self.aerodynamic_control_active_hist.append(bool(aerodynamic["control_active"]))
        self._append_history_arrays(
            t_s=float(snapshot.time_s),
            rel=rel,
            thrust=thrust_eci,
            thrust_ric=self.thrust_ric_hist[-1],
            target_thrust=target_thrust_eci,
            mean_motion_rad_s=n,
            aerodynamic=aerodynamic,
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
        aerodynamic: dict[str, float | bool],
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
        self._aerodynamic_ballistic_coefficient_array[idx] = float(aerodynamic["ballistic_coefficient_kg_m2"])
        self._aerodynamic_drag_area_array[idx] = float(aerodynamic["drag_area_m2"])
        self._aerodynamic_lift_coefficient_array[idx] = float(aerodynamic["lift_coefficient"])
        self._aerodynamic_lift_area_array[idx] = float(aerodynamic["lift_area_m2"])
        self._aerodynamic_lift_bank_angle_array[idx] = float(aerodynamic["lift_bank_angle_deg"])
        self._aerodynamic_control_active_array[idx] = bool(aerodynamic["control_active"])
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
        self._aerodynamic_ballistic_coefficient_array = grow_1d(
            self._aerodynamic_ballistic_coefficient_array,
            fill_value=float("nan"),
        )
        self._aerodynamic_drag_area_array = grow_1d(
            self._aerodynamic_drag_area_array,
            fill_value=float("nan"),
        )
        self._aerodynamic_lift_coefficient_array = grow_1d(
            self._aerodynamic_lift_coefficient_array,
            fill_value=float("nan"),
        )
        self._aerodynamic_lift_area_array = grow_1d(
            self._aerodynamic_lift_area_array,
            fill_value=float("nan"),
        )
        self._aerodynamic_lift_bank_angle_array = grow_1d(
            self._aerodynamic_lift_bank_angle_array,
            fill_value=float("nan"),
        )
        active = np.zeros(new_capacity, dtype=bool)
        if old_count:
            active[:old_count] = self._aerodynamic_control_active_array[:old_count]
        self._aerodynamic_control_active_array = active
        self._range_array = grow_1d(self._range_array)
        self._speed_array = grow_1d(self._speed_array)
        self._goal_error_array = grow_1d(self._goal_error_array, fill_value=float("nan"))
        self._delta_v_interval_km_s_array = grow_1d(self._delta_v_interval_km_s_array)
        self._target_delta_v_interval_km_s_array = grow_1d(self._target_delta_v_interval_km_s_array)
        self._nmt_radial_amplitude_array = grow_1d(self._nmt_radial_amplitude_array, fill_value=float("nan"))
        self._nmt_cross_track_amplitude_array = grow_1d(self._nmt_cross_track_amplitude_array, fill_value=float("nan"))
        self._nmt_radial_amplitude_error_array = grow_1d(
            self._nmt_radial_amplitude_error_array, fill_value=float("nan")
        )
        self._nmt_cross_track_amplitude_error_array = grow_1d(
            self._nmt_cross_track_amplitude_error_array, fill_value=float("nan")
        )
        self._nmt_drift_velocity_error_array = grow_1d(self._nmt_drift_velocity_error_array, fill_value=float("nan"))
        self._nmt_element_goal_error_array = grow_1d(self._nmt_element_goal_error_array, fill_value=float("nan"))
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
            ballistic_coefficient = self._aerodynamic_ballistic_coefficient_array[:count]
            drag_area = self._aerodynamic_drag_area_array[:count]
            lift_coefficient = self._aerodynamic_lift_coefficient_array[:count]
            lift_area = self._aerodynamic_lift_area_array[:count]
            lift_bank_angle = self._aerodynamic_lift_bank_angle_array[:count]
            aerodynamic_active = self._aerodynamic_control_active_array[:count]
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
            ballistic_coefficient = np.asarray(self.aerodynamic_ballistic_coefficient_hist, dtype=float)
            drag_area = np.asarray(self.aerodynamic_drag_area_hist, dtype=float)
            lift_coefficient = np.asarray(self.aerodynamic_lift_coefficient_hist, dtype=float)
            lift_area = np.asarray(self.aerodynamic_lift_area_hist, dtype=float)
            lift_bank_angle = np.asarray(self.aerodynamic_lift_bank_angle_hist, dtype=float)
            aerodynamic_active = np.asarray(self.aerodynamic_control_active_hist, dtype=bool)
        return {
            "time_s": t[:count].copy(),
            "relative_ric": rel[:count, :].copy(),
            "chaser_thrust_ric_km_s2": thrust_ric.copy(),
            "target_thrust_eci_km_s2": target_thrust[:count, :].copy(),
            "aerodynamic_ballistic_coefficient_kg_m2": ballistic_coefficient[:count].copy(),
            "aerodynamic_drag_area_m2": drag_area[:count].copy(),
            "aerodynamic_lift_coefficient": lift_coefficient[:count].copy(),
            "aerodynamic_lift_area_m2": lift_area[:count].copy(),
            "aerodynamic_lift_bank_angle_deg": lift_bank_angle[:count].copy(),
            "aerodynamic_control_active": aerodynamic_active[:count].copy(),
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
                level_failed=True,
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
            self.config.goal_nmt_radial_amplitude_km is not None or self.config.max_cross_track_amplitude_km is not None
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
                    region.intersects_segment(rel[idx - 1, :3], rel[idx, :3]) for idx in range(1, rel.shape[0])
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
            target_dv_m_s = float(np.sum(target_dv_intervals[np.isfinite(target_dv_intervals)]) * 1.0e3)
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
                f"Cross-track amplitude above {_format_distance_text(float(self.config.max_cross_track_amplitude_km))}."
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
