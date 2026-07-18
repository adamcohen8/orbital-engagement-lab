"""Static ownership map for OEL's plotting implementation families.

The public compatibility modules intentionally remain stable.  This map lets
maintainers and coding agents find the implementation owner of a plotting
capability without depending on dynamic registration or import side effects.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlottingCapabilityFamily:
    """A coherent plotting family and the façade that continues to expose it."""

    name: str
    module: str
    facade: str
    capabilities: tuple[str, ...]


LOW_LEVEL_CAPABILITY_FAMILIES: tuple[PlottingCapabilityFamily, ...] = (
    PlottingCapabilityFamily(
        "ground_track",
        "sim.plotting.ground_track_plots",
        "sim.utils.plotting_capabilities",
        ("animate_ground_track", "animate_multi_ground_track"),
    ),
    PlottingCapabilityFamily(
        "frame",
        "sim.plotting.frame_plots",
        "sim.utils.plotting_capabilities",
        (
            "plot_quaternion_components",
            "plot_body_rates",
            "plot_trajectory_frame",
            "plot_multi_trajectory_frame",
            "plot_ric_2d_projections",
            "plot_multi_ric_2d_projections",
        ),
    ),
    PlottingCapabilityFamily(
        "control",
        "sim.plotting.control_plots",
        "sim.utils.plotting_capabilities",
        ("plot_control_commands", "plot_multi_control_commands"),
    ),
    PlottingCapabilityFamily(
        "trajectory_animation",
        "sim.plotting.trajectory_animations",
        "sim.utils.plotting_capabilities",
        (
            "animate_multi_ric_2d_projections",
            "animate_trajectory_frame",
            "animate_multi_trajectory_frame",
        ),
    ),
    PlottingCapabilityFamily(
        "attitude_animation",
        "sim.plotting.attitude_animations",
        "sim.utils.plotting_capabilities",
        ("animate_rectangular_prism_attitude", "animate_battlespace_dashboard"),
    ),
    PlottingCapabilityFamily(
        "prism_animation",
        "sim.plotting.prism_animations",
        "sim.utils.plotting_capabilities",
        (
            "animate_multi_rectangular_prism_ric_curv",
            "animate_side_by_side_rectangular_prism_ric_attitude",
        ),
    ),
)


SINGLE_RUN_PLOT_FAMILIES: tuple[PlottingCapabilityFamily, ...] = (
    PlottingCapabilityFamily(
        "dashboard",
        "sim.plotting.dashboard_plots",
        "sim.plotting.single_run",
        (
            "plot_run_dashboard",
            "plot_rendezvous_summary",
            "plot_rendezvous_summary_curvilinear",
            "plot_control_effort",
        ),
    ),
    PlottingCapabilityFamily(
        "estimation",
        "sim.plotting.estimation_plots",
        "sim.plotting.single_run",
        (
            "plot_estimation_error",
            "plot_estimation_error_components",
            "plot_knowledge_filtering",
            "plot_sensor_access",
        ),
    ),
    PlottingCapabilityFamily(
        "access",
        "sim.plotting.access_plots",
        "sim.plotting.single_run",
        ("plot_ground_track_from_payload", "plot_ground_station_access"),
    ),
    PlottingCapabilityFamily(
        "orbital_element",
        "sim.plotting.orbital_element_plots",
        "sim.plotting.single_run",
        (
            "plot_orbital_element",
            "plot_orbital_elements_summary",
            "plot_orbital_elements_angles",
        ),
    ),
    PlottingCapabilityFamily(
        "reentry",
        "sim.plotting.reentry_plots",
        "sim.plotting.single_run",
        (
            "plot_reentry_summary",
            "plot_reentry_aero",
            "plot_reentry_thermal",
            "plot_atmospheric_pass",
        ),
    ),
    PlottingCapabilityFamily(
        "attitude_summary",
        "sim.plotting.attitude_summary_plots",
        "sim.plotting.single_run",
        ("plot_attitude_control_summary",),
    ),
)


PLOTTING_SUPPORT_MODULES: tuple[str, ...] = (
    "sim.plotting.capability_common",
    "sim.plotting.attitude_geometry",
    "sim.plotting.single_run_context",
    "sim.plotting.single_run_math",
)


PLOTTING_CAPABILITY_FAMILIES = LOW_LEVEL_CAPABILITY_FAMILIES + SINGLE_RUN_PLOT_FAMILIES
