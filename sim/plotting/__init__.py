__all__ = [
    "plot_control_effort",
    "plot_estimation_error",
    "plot_estimation_error_components",
    "plot_ground_track_from_payload",
    "plot_rendezvous_summary",
    "plot_run_dashboard",
    "plot_sensor_access",
]


def __getattr__(name: str):
    if name in __all__:
        from sim.plotting import single_run

        return getattr(single_run, name)
    raise AttributeError(f"module 'sim.plotting' has no attribute {name!r}")
