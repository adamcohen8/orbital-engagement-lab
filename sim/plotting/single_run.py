from __future__ import annotations

from functools import wraps
from typing import Any

from sim.plotting import orbital_element_plots as _orbital_element_plots
from sim.plotting.access_plots import (
    plot_ground_station_access as plot_ground_station_access,
)
from sim.plotting.access_plots import (
    plot_ground_track_from_payload as plot_ground_track_from_payload,
)
from sim.plotting.attitude_summary_plots import (
    plot_attitude_control_summary as plot_attitude_control_summary,
)
from sim.plotting.dashboard_plots import (
    plot_control_effort as plot_control_effort,
)
from sim.plotting.dashboard_plots import (
    plot_rendezvous_summary as plot_rendezvous_summary,
)
from sim.plotting.dashboard_plots import (
    plot_rendezvous_summary_curvilinear as plot_rendezvous_summary_curvilinear,
)
from sim.plotting.dashboard_plots import (
    plot_run_dashboard as plot_run_dashboard,
)
from sim.plotting.estimation_plots import (
    plot_estimation_error as plot_estimation_error,
)
from sim.plotting.estimation_plots import (
    plot_estimation_error_components as plot_estimation_error_components,
)
from sim.plotting.estimation_plots import (
    plot_knowledge_filtering as plot_knowledge_filtering,
)
from sim.plotting.estimation_plots import (
    plot_sensor_access as plot_sensor_access,
)
from sim.plotting.orbital_element_plots import (
    ORBITAL_ELEMENT_SPECS as ORBITAL_ELEMENT_SPECS,
)
from sim.plotting.orbital_element_plots import (
    OrbitalElementSeriesCache as OrbitalElementSeriesCache,
)
from sim.plotting.orbital_element_plots import (
    _orbital_element_object_ids as _orbital_element_object_ids,
)
from sim.plotting.reentry_plots import (
    _cross_track_axis_from_truth as _cross_track_axis_from_truth,
)
from sim.plotting.reentry_plots import (
    _mark_reentry_threshold as _mark_reentry_threshold,
)
from sim.plotting.reentry_plots import (
    _plot_cross_track_kinematics as _plot_cross_track_kinematics,
)
from sim.plotting.reentry_plots import (
    _plot_lift_axis_alignment as _plot_lift_axis_alignment,
)
from sim.plotting.reentry_plots import (
    _plot_reentry_series as _plot_reentry_series,
)
from sim.plotting.reentry_plots import (
    plot_atmospheric_pass as plot_atmospheric_pass,
)
from sim.plotting.reentry_plots import (
    plot_reentry_aero as plot_reentry_aero,
)
from sim.plotting.reentry_plots import (
    plot_reentry_summary as plot_reentry_summary,
)
from sim.plotting.reentry_plots import (
    plot_reentry_thermal as plot_reentry_thermal,
)
from sim.plotting.single_run_context import (
    ArrayMap as ArrayMap,
)
from sim.plotting.single_run_context import (
    NestedArrayMap as NestedArrayMap,
)
from sim.plotting.single_run_context import (
    RICSummaryFrame as RICSummaryFrame,
)
from sim.plotting.single_run_context import (
    _array_map as _array_map,
)
from sim.plotting.single_run_context import (
    _as_array as _as_array,
)
from sim.plotting.single_run_context import (
    _choose_reference as _choose_reference,
)
from sim.plotting.single_run_context import (
    _choose_subject as _choose_subject,
)
from sim.plotting.single_run_context import (
    _finite_rows as _finite_rows,
)
from sim.plotting.single_run_context import (
    _nested_array_map as _nested_array_map,
)
from sim.plotting.single_run_context import (
    _object_color as _object_color,
)
from sim.plotting.single_run_context import (
    _payload_arrays as _payload_arrays,
)
from sim.plotting.single_run_context import (
    _payload_reentry_metrics as _payload_reentry_metrics,
)
from sim.plotting.single_run_context import (
    _plot_eci_trajectories as _plot_eci_trajectories,
)
from sim.plotting.single_run_context import (
    _reentry_metric_map as _reentry_metric_map,
)
from sim.plotting.single_run_context import (
    _ric_position as _ric_position,
)
from sim.plotting.single_run_context import (
    _ric_position_for_summary as _ric_position_for_summary,
)
from sim.plotting.single_run_context import (
    _ric_projection_axis_limits as _ric_projection_axis_limits,
)
from sim.plotting.single_run_context import (
    _ric_relative_state as _ric_relative_state,
)
from sim.plotting.single_run_context import (
    _save_show_close as _save_show_close,
)
from sim.plotting.single_run_context import (
    _set_equal_3d as _set_equal_3d,
)
from sim.plotting.single_run_context import (
    _time_for as _time_for,
)
from sim.plotting.single_run_math import (
    _classical_orbital_elements_series as _classical_orbital_elements_series,
)
from sim.plotting.single_run_math import (
    _cumulative_delta_v_m_s as _cumulative_delta_v_m_s,
)
from sim.plotting.single_run_math import (
    _quat_error_angle_deg as _quat_error_angle_deg,
)
from sim.plotting.single_run_math import (
    _quat_error_series_deg as _quat_error_series_deg,
)
from sim.plotting.single_run_math import (
    _safe_angle_deg as _safe_angle_deg,
)
from sim.plotting.single_run_math import (
    _thrust_alignment_error_deg_series as _thrust_alignment_error_deg_series,
)


def _sync_orbital_math() -> None:
    _orbital_element_plots._classical_orbital_elements_series = _classical_orbital_elements_series


@wraps(_orbital_element_plots._plot_element_on_axis)
def _plot_element_on_axis(*args: Any, **kwargs: Any) -> Any:
    _sync_orbital_math()
    return _orbital_element_plots._plot_element_on_axis(*args, **kwargs)


@wraps(_orbital_element_plots.plot_orbital_element)
def plot_orbital_element(*args: Any, **kwargs: Any) -> Any:
    _sync_orbital_math()
    return _orbital_element_plots.plot_orbital_element(*args, **kwargs)


@wraps(_orbital_element_plots.plot_orbital_elements_summary)
def plot_orbital_elements_summary(*args: Any, **kwargs: Any) -> Any:
    _sync_orbital_math()
    return _orbital_element_plots.plot_orbital_elements_summary(*args, **kwargs)


@wraps(_orbital_element_plots.plot_orbital_elements_angles)
def plot_orbital_elements_angles(*args: Any, **kwargs: Any) -> Any:
    _sync_orbital_math()
    return _orbital_element_plots.plot_orbital_elements_angles(*args, **kwargs)
