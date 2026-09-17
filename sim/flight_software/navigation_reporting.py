"""Publish navigation estimates through the existing typed telemetry boundary."""

from __future__ import annotations

from sim.gnc.attitude_v2 import AttitudeSolution
from sim.gnc.navigation_v2 import OrbitNavigationSolution

from .clocks import clock_tag_elapsed_ns
from .contracts import DiagnosticTelemetry, FrameId, TelemetryField

NAVIGATION_STATE_TOPIC = "oel.navigation_state.v1"


def navigation_telemetry(
    solution: OrbitNavigationSolution | AttitudeSolution,
    inertial_frame: FrameId,
) -> tuple[DiagnosticTelemetry, ...]:
    """Report the solution already used by control, without advancing a filter.

    Generation time is the invocation time; state epoch distinguishes a held
    state from a propagated estimate. Raw sensor packets retain their own epochs.
    """
    orbit = isinstance(solution, OrbitNavigationSolution)
    attitude = solution.attitude if orbit else solution
    epoch = solution.own_state_epoch if orbit else None
    fields = [
        TelemetryField("representation", "navigation_estimate"),
        TelemetryField("frame_id", inertial_frame.name),
        TelemetryField("frame_registry_version", inertial_frame.registry_version),
        TelemetryField("position_available", solution.position_eci_m is not None),
        TelemetryField("velocity_available", solution.velocity_eci_m_s is not None),
        TelemetryField("state_epoch_ticks", None if epoch is None else epoch.ticks),
        TelemetryField(
            "state_age_s",
            None
            if epoch is None
            else (clock_tag_elapsed_ns(solution.generated_at) - clock_tag_elapsed_ns(epoch)) / 1e9,
            "s",
        ),
    ]
    for prefix, axes, values, unit in (
        ("position_", "xyz", solution.position_eci_m, "m"),
        ("velocity_", "xyz", solution.velocity_eci_m_s, "m/s"),
        ("q_", "wxyz", attitude.attitude_quat_bn, None),
        ("omega_", "xyz", attitude.angular_rate_body_rad_s, "rad/s"),
    ):
        suffix = {"m": "_m", "m/s": "_m_s", "rad/s": "_rad_s", None: ""}[unit]
        if values is not None:
            fields.extend(
                TelemetryField(f"{prefix}{axis}{suffix}", float(value), unit)
                for axis, value in zip(axes, values, strict=True)
            )
    return (DiagnosticTelemetry(NAVIGATION_STATE_TOPIC, solution.generated_at, tuple(fields)),)
