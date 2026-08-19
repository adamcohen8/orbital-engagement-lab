__all__ = [
    "plot_attitude_control_summary",
    "plot_control_effort",
    "plot_estimation_error",
    "plot_estimation_error_components",
    "plot_ground_station_access",
    "plot_ground_track_from_payload",
    "plot_knowledge_filtering",
    "plot_atmospheric_pass",
    "plot_orbital_element",
    "plot_orbital_elements_angles",
    "plot_orbital_elements_summary",
    "plot_reentry_aero",
    "plot_reentry_summary",
    "plot_reentry_thermal",
    "plot_rendezvous_summary",
    "plot_rendezvous_summary_curvilinear",
    "plot_run_dashboard",
    "plot_sensor_access",
    "OEL_STYLE_DARK",
    "OEL_STYLE_LIGHT",
    "OEL_ROLE_COLORS",
    "oel_plot_context",
    "role_color",
    "save_oel_animation",
    "ANIMATION_QUALITY_POLICY_VERSION",
    "STRICT_AGENT_ANIMATION_QUALITY",
    "AnimationQualityPolicy",
    "AnimationQualityReport",
    "save_animation_with_quality",
    "show_save_close_oel",
]


def __getattr__(name: str):
    if name in __all__:
        if name in {
            "ANIMATION_QUALITY_POLICY_VERSION",
            "STRICT_AGENT_ANIMATION_QUALITY",
            "AnimationQualityPolicy",
            "AnimationQualityReport",
            "save_animation_with_quality",
        }:
            from sim.plotting import animation_quality

            return getattr(animation_quality, name)
        if name in {
            "OEL_STYLE_DARK",
            "OEL_STYLE_LIGHT",
            "OEL_ROLE_COLORS",
            "oel_plot_context",
            "role_color",
            "save_oel_animation",
            "show_save_close_oel",
        }:
            from sim.plotting import style

            return getattr(style, name)
        from sim.plotting import single_run

        return getattr(single_run, name)
    raise AttributeError(f"module 'sim.plotting' has no attribute {name!r}")
