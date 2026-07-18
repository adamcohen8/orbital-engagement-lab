from __future__ import annotations

from sim.plotting.attitude_animations import (
    animate_battlespace_dashboard as animate_battlespace_dashboard,
)
from sim.plotting.attitude_animations import (
    animate_rectangular_prism_attitude as animate_rectangular_prism_attitude,
)
from sim.plotting.attitude_geometry import (
    _attitude_display_axes as _attitude_display_axes,
)
from sim.plotting.attitude_geometry import (
    _attitude_rotation_history as _attitude_rotation_history,
)
from sim.plotting.attitude_geometry import (
    _marker_frame_faces as _marker_frame_faces,
)
from sim.plotting.attitude_geometry import (
    _permute_face_vertices as _permute_face_vertices,
)
from sim.plotting.attitude_geometry import (
    _rectangular_prism_faces as _rectangular_prism_faces,
)
from sim.plotting.attitude_geometry import (
    _rectangular_prism_frame_vertices as _rectangular_prism_frame_vertices,
)
from sim.plotting.attitude_geometry import (
    _rectangular_prism_vertices_body as _rectangular_prism_vertices_body,
)
from sim.plotting.attitude_geometry import (
    _symmetric_limit_from_arrays as _symmetric_limit_from_arrays,
)
from sim.plotting.attitude_geometry import (
    _thruster_marker_geometry_body as _thruster_marker_geometry_body,
)
from sim.plotting.capability_common import (
    AttitudeFrame as AttitudeFrame,
)
from sim.plotting.capability_common import (
    FrameName as FrameName,
)
from sim.plotting.capability_common import (
    Layout as Layout,
)
from sim.plotting.capability_common import (
    PlotMode as PlotMode,
)
from sim.plotting.capability_common import (
    _object_role_color as _object_role_color,
)
from sim.plotting.capability_common import (
    _play_interactive_animation as _play_interactive_animation,
)
from sim.plotting.capability_common import (
    _show_save_close as _show_save_close,
)
from sim.plotting.control_plots import (
    plot_control_commands as plot_control_commands,
)
from sim.plotting.control_plots import (
    plot_multi_control_commands as plot_multi_control_commands,
)
from sim.plotting.frame_plots import (
    _bottom_center_figure_legend as _bottom_center_figure_legend,
)
from sim.plotting.frame_plots import (
    _draw_earth_sphere_3d as _draw_earth_sphere_3d,
)
from sim.plotting.frame_plots import (
    _draw_ric_reference_origin_2d as _draw_ric_reference_origin_2d,
)
from sim.plotting.frame_plots import (
    _draw_ric_reference_origin_3d as _draw_ric_reference_origin_3d,
)
from sim.plotting.frame_plots import (
    _first_last_finite_indices as _first_last_finite_indices,
)
from sim.plotting.frame_plots import (
    _rates_in_frame as _rates_in_frame,
)
from sim.plotting.frame_plots import (
    _reference_origin_label as _reference_origin_label,
)
from sim.plotting.frame_plots import (
    _ric_2d_plane_axes as _ric_2d_plane_axes,
)
from sim.plotting.frame_plots import (
    _trajectory_in_frame as _trajectory_in_frame,
)
from sim.plotting.frame_plots import (
    _truth_quaternion_in_frame as _truth_quaternion_in_frame,
)
from sim.plotting.frame_plots import (
    plot_body_rates as plot_body_rates,
)
from sim.plotting.frame_plots import (
    plot_multi_ric_2d_projections as plot_multi_ric_2d_projections,
)
from sim.plotting.frame_plots import (
    plot_multi_trajectory_frame as plot_multi_trajectory_frame,
)
from sim.plotting.frame_plots import (
    plot_quaternion_components as plot_quaternion_components,
)
from sim.plotting.frame_plots import (
    plot_ric_2d_projections as plot_ric_2d_projections,
)
from sim.plotting.frame_plots import (
    plot_trajectory_frame as plot_trajectory_frame,
)
from sim.plotting.ground_track_plots import (
    _HAS_CARTOPY as _HAS_CARTOPY,
)
from sim.plotting.ground_track_plots import (
    _draw_stylized_earth_map as _draw_stylized_earth_map,
)
from sim.plotting.ground_track_plots import (
    _map_colors as _map_colors,
)
from sim.plotting.ground_track_plots import (
    _setup_ground_track_axes as _setup_ground_track_axes,
)
from sim.plotting.ground_track_plots import (
    animate_ground_track as animate_ground_track,
)
from sim.plotting.ground_track_plots import (
    animate_multi_ground_track as animate_multi_ground_track,
)
from sim.plotting.prism_animations import (
    animate_multi_rectangular_prism_ric_curv as animate_multi_rectangular_prism_ric_curv,
)
from sim.plotting.prism_animations import (
    animate_side_by_side_rectangular_prism_ric_attitude as animate_side_by_side_rectangular_prism_ric_attitude,
)
from sim.plotting.trajectory_animations import (
    animate_multi_ric_2d_projections as animate_multi_ric_2d_projections,
)
from sim.plotting.trajectory_animations import (
    animate_multi_trajectory_frame as animate_multi_trajectory_frame,
)
from sim.plotting.trajectory_animations import (
    animate_trajectory_frame as animate_trajectory_frame,
)
from sim.utils.plotting import plot_angular_rates as plot_angular_rates_legacy
from sim.utils.plotting import plot_attitude_ric as plot_attitude_ric_legacy
from sim.utils.plotting import plot_attitude_tumble as plot_attitude_tumble_legacy
from sim.utils.plotting import plot_ground_track as plot_ground_track_legacy
from sim.utils.plotting import plot_orbit_eci as plot_orbit_eci_legacy


# Legacy plotting API re-export wrappers to keep one plotting surface.
def plot_orbit_eci(*args, **kwargs):
    return plot_orbit_eci_legacy(*args, **kwargs)


def plot_attitude_tumble(*args, **kwargs):
    return plot_attitude_tumble_legacy(*args, **kwargs)


def plot_attitude_ric(*args, **kwargs):
    return plot_attitude_ric_legacy(*args, **kwargs)


def plot_angular_rates(*args, **kwargs):
    return plot_angular_rates_legacy(*args, **kwargs)


def plot_ground_track(*args, **kwargs):
    return plot_ground_track_legacy(*args, **kwargs)
