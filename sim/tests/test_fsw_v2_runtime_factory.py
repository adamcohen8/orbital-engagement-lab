from __future__ import annotations

import pytest

from sim import SimulationConfig, SimulationSession
from sim.actuators.physical import (
    CmgHardware,
    ContinuousEngineHardware,
    MagnetorquerHardware,
    ReactionWheelHardware,
)
from sim.flight_software import (
    MISSION_LOAD_SCHEMA,
    ClockScale,
    ClockTag,
    GoalDefinition,
    GoalMode,
    MissionLoadManifest,
    OnboardMissionConfigurationLoad,
    to_primitive,
    with_computed_content_hash,
)


def _scenario(stack: str, params: dict | None = None) -> dict:
    return {
        "scenario_name": f"runtime_{stack.replace('.', '_')}",
        "objects": {
            "sat": {
                "enabled": True,
                "kind": "satellite",
                "specs": {"mass_kg": 100.0},
                "initial_state": {
                    "position_eci_km": [7000.0, 0.0, 0.0],
                    "velocity_eci_km_s": [0.0, 7.5, 0.0],
                },
                "flight_software": {
                    "stack": stack,
                    "hardware_profile": (
                        "hardware.passive.v1" if stack == "fsw.passive" else "hardware.ideal_wrench.v1"
                    ),
                    "task_period_s": 0.1,
                    "params": params or {},
                },
            }
        },
        "simulator": {
            "duration_s": 0.2,
            "dt_s": 0.1,
            "termination": {"earth_impact_enabled": False},
            "dynamics": {"attitude": {"enabled": True, "attitude_substep_s": 0.1}},
        },
        "outputs": {
            "mode": "interactive",
            "stats": {"print_summary": False},
            "plots": {"enabled": False, "figure_ids": []},
            "animations": {"enabled": False, "types": []},
        },
    }


@pytest.mark.parametrize(
    ("stack", "params"),
    [
        ("fsw.passive", {}),
        ("fsw.attitude_reference", {"max_torque_n_m": 0.2}),
        (
            "fsw.orbit_reference",
            {"translation_mode": "stationkeeping", "max_acceleration_m_s2": 0.001},
        ),
    ],
)
def test_v2_stack_factory_runs_complete_stack(stack: str, params: dict) -> None:
    session = SimulationSession.from_config(SimulationConfig.from_dict(_scenario(stack, params)))
    result = session.run()
    runtime = session._engine.agents["sat"].flight_software_runtime

    assert result.summary["samples"] == 3
    assert runtime is not None
    assert runtime.stack.identity.stack_id == stack
    assert runtime.evidence.invocations


def test_custom_stack_factory_receives_only_declared_params() -> None:
    raw = _scenario("fsw.passive")
    raw["objects"]["sat"]["flight_software"] = {
        "module": "sim.flight_software.reference_stacks",
        "class_name": "PassiveFlightSoftwareStack",
        "params": {},
    }
    with pytest.raises(RuntimeError, match="Failed to construct requested plugin"):
        SimulationSession.from_config(SimulationConfig.from_dict(raw)).run()


def test_low_thrust_profile_builds_physical_continuous_engine() -> None:
    raw = _scenario(
        "fsw.low_thrust_reference",
        {
            "reference_object_id": "target",
            "translation_mode": "low_thrust_phasing",
            "target_relative_state_ric_m": [0.0, 100.0, 0.0, 0.0, 0.0, 0.0],
            "max_acceleration_m_s2": 0.001,
        },
    )
    raw["objects"]["sat"]["flight_software"]["hardware_profile"] = "hardware.continuous_engine.v1"
    raw["objects"]["sat"]["initial_state"] = {
        "relative_to": "target",
        "relative_ric_rect": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }
    raw["objects"]["target"] = {
        "kind": "satellite",
        "initial_state": {
            "position_eci_km": [7000.0, 0.0, 0.0],
            "velocity_eci_km_s": [0.0, 7.5, 0.0],
        },
        "flight_software": {"stack": "fsw.passive", "hardware_profile": "hardware.passive.v1"},
    }
    session = SimulationSession.from_config(SimulationConfig.from_dict(raw))
    session.run()
    runtime = session._engine.agents["sat"].flight_software_runtime
    assert runtime is not None
    assert isinstance(runtime.hardware["translation"], ContinuousEngineHardware)
    assert runtime.evidence.realizations


@pytest.mark.parametrize(
    ("profile", "hardware_type"),
    (
        ("hardware.reaction_wheels.v1", ReactionWheelHardware),
        ("hardware.magnetorquer.v1", MagnetorquerHardware),
        ("hardware.cmg.v1", CmgHardware),
    ),
)
def test_attitude_profiles_build_and_run_physical_adcs_hardware(
    profile: str,
    hardware_type: type[object],
) -> None:
    raw = _scenario(
        "fsw.attitude_reference",
        {"quaternion_bn": [1.0, 0.0, 0.0, 0.0]},
    )
    raw["objects"]["sat"]["flight_software"]["hardware_profile"] = profile
    raw["objects"]["sat"]["initial_state"]["angular_rate_body_rad_s"] = [0.01, -0.02, 0.015]
    session = SimulationSession.from_config(SimulationConfig.from_dict(raw))
    session.run()
    runtime = session._engine.agents["sat"].flight_software_runtime
    assert runtime is not None
    assert isinstance(runtime.hardware["attitude"], hardware_type)
    assert runtime.evidence.realizations
    assert any(
        sum(value * value for value in realization.realized_torque_n_m) > 0.0
        for realization in runtime.evidence.realizations
    )
    state_field_names = {
        field.name
        for realization in runtime.evidence.realizations
        for field in realization.device_state
    }
    if profile == "hardware.reaction_wheels.v1":
        assert any(name.startswith("wheel_") and name.endswith("_momentum_n_m_s") for name in state_field_names)
        assert any(abs(value) > 0.0 for value in runtime.hardware["attitude"].momentum_n_m_s)
    elif profile == "hardware.magnetorquer.v1":
        assert {"dipole_x_a_m2", "dipole_y_a_m2", "dipole_z_a_m2"} <= state_field_names
    else:
        assert {"gimbal_x_angle_rad", "gimbal_y_angle_rad", "gimbal_z_angle_rad"} <= state_field_names
        assert any(abs(value) > 0.0 for value in runtime.hardware["attitude"].gimbal_angle_rad)


def test_combined_wheel_torquer_profile_builds_operational_momentum_management_hardware() -> None:
    raw = _scenario(
        "fsw.attitude_reference",
        {
            "quaternion_bn": [1.0, 0.0, 0.0, 0.0],
            "momentum_dump_start_fraction": 0.01,
            "momentum_dump_stop_fraction": 0.005,
            "wheel_max_momentum_n_m_s": [0.01],
        },
    )
    raw["objects"]["sat"]["flight_software"]["hardware_profile"] = (
        "hardware.reaction_wheels_magnetorquer.v1"
    )
    raw["objects"]["sat"]["initial_state"]["angular_rate_body_rad_s"] = [0.01, -0.02, 0.015]
    session = SimulationSession.from_config(SimulationConfig.from_dict(raw))
    session.run()
    runtime = session._engine.agents["sat"].flight_software_runtime
    assert runtime is not None
    assert isinstance(runtime.hardware["attitude"], ReactionWheelHardware)
    assert isinstance(runtime.hardware["momentum_dump"], MagnetorquerHardware)
    fields = {
        field.name: field.value
        for output in runtime.evidence.outputs
        for telemetry in output.telemetry
        for field in telemetry.fields
    }
    assert "adcs_operational_mode" in fields
    assert "wheel_momentum_fraction" in fields


def test_rpo_stack_plans_and_commands_autonomous_conjunction_avoidance_end_to_end() -> None:
    raw = _scenario(
        "fsw.rpo_reference",
        {
            "reference_object_id": "target",
            "translation_mode": "ric_hold",
            "target_relative_state_ric_m": [50.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "max_acceleration_m_s2": 0.01,
            "conjunction_avoidance_enabled": True,
            "conjunction_keep_out_radius_m": 100.0,
            "conjunction_prediction_horizon_s": 60.0,
            "conjunction_avoidance_delta_v_m_s": 0.001,
            "conjunction_maneuver_lead_time_s": 0.1,
        },
    )
    raw["objects"]["sat"]["initial_state"] = {
        "relative_to": "target",
        "relative_ric_rect": [0.05, 0.0, 0.0, 0.0, 0.0, 0.0],
    }
    raw["objects"]["target"] = {
        "kind": "satellite",
        "initial_state": {
            "position_eci_km": [7000.0, 0.0, 0.0],
            "velocity_eci_km_s": [0.0, 7.5, 0.0],
        },
        "flight_software": {"stack": "fsw.passive", "hardware_profile": "hardware.passive.v1"},
    }
    session = SimulationSession.from_config(SimulationConfig.from_dict(raw))
    session.run()
    runtime = session._engine.agents["sat"].flight_software_runtime
    assert runtime is not None
    fields = {
        field.name: field.value
        for output in runtime.evidence.outputs
        for telemetry in output.telemetry
        for field in telemetry.fields
    }
    assert fields["maneuver_plan_reason"] == "predicted_keep_out_violation"
    assert fields["maneuver_plan_executed"] is True
    assert any(output.commands for output in runtime.evidence.outputs)


def test_scenario_mission_load_is_delivered_through_onboard_input_queue() -> None:
    raw = _scenario("fsw.orbit_reference", {"translation_mode": "stationkeeping"})
    manifest = MissionLoadManifest(
        "scenario-load",
        1,
        MISSION_LOAD_SCHEMA,
        "fsw.orbit_reference",
        ">=2.0.0,<3.0.0",
        "0" * 64,
        ClockTag("authoring", 0, 1, ClockScale.ONBOARD),
    )
    load = with_computed_content_hash(
        OnboardMissionConfigurationLoad(
            manifest,
            GoalDefinition("maintain-orbit", "orbit.stationkeeping", GoalMode.MAINTENANCE),
        )
    )
    raw["objects"]["sat"]["flight_software"]["mission_load"] = to_primitive(load)

    session = SimulationSession.from_config(SimulationConfig.from_dict(raw))
    session.run()
    runtime = session._engine.agents["sat"].flight_software_runtime

    assert runtime is not None
    load_events = [event for event in runtime.evidence.input_events if event.kind.value == "mission_load"]
    assert len(load_events) == 1
    assert runtime.stack.snapshot().active_load_id == "scenario-load"
    load_fields = {
        field.name: field.value
        for output in runtime.evidence.outputs
        for telemetry in output.telemetry
        for field in telemetry.fields
        if field.name.startswith("mission_load_")
    }
    assert load_fields["mission_load_disposition"] == "accepted"


def test_orbit_reference_executes_scheduled_finite_burn_through_hardware() -> None:
    raw = _scenario(
        "fsw.orbit_reference",
        {
            "translation_mode": "stationkeeping",
            "max_force_n": 1.0,
            "max_acceleration_m_s2": 0.01,
            "scheduled_burns": [
                {
                    "start_time_s": 0.1,
                    "duration_s": 0.1,
                    "frame": "eci",
                    "delta_v_m_s": [0.001, 0.0, 0.0],
                }
            ],
        },
    )
    session = SimulationSession.from_config(SimulationConfig.from_dict(raw))
    session.run()
    runtime = session._engine.agents["sat"].flight_software_runtime

    assert runtime is not None
    active = [
        realization
        for realization in runtime.evidence.realizations
        if realization.interval_start_ns == 100_000_000
    ]
    assert active
    assert active[0].realized_force_n[0] > 0.0


def test_yaml_actions_constraints_and_alpha_beta_filter_reach_the_complete_stack() -> None:
    raw = _scenario(
        "fsw.rpo_reference",
        {
            "reference_object_id": "target",
            "translation_mode": "ric_hold",
            "navigation_filter": "alpha_beta",
            "actions": [
                {
                    "action_id": "hold-first",
                    "mode": "ric_hold",
                    "kind": "timed",
                    "duration_s": 0.1,
                },
                {
                    "action_id": "then-retreat",
                    "mode": "passive_retreat",
                    "kind": "pulsed",
                    "pulse_count": 2,
                },
            ],
            "constraints": [
                {
                    "constraint_id": "fault-free",
                    "kind": "physical_capability",
                    "evaluator_id": "no_active_faults",
                }
            ],
        },
    )
    raw["objects"]["sat"]["initial_state"] = {
        "relative_to": "target",
        "relative_ric_rect": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    }
    raw["objects"]["target"] = {
        "kind": "satellite",
        "specs": {"mass_kg": 100.0},
        "initial_state": {
            "position_eci_km": [7000.0, 0.0, 0.0],
            "velocity_eci_km_s": [0.0, 7.5, 0.0],
        },
        "flight_software": {"stack": "fsw.passive", "hardware_profile": "hardware.passive.v1"},
    }
    session = SimulationSession.from_config(SimulationConfig.from_dict(raw))
    session.run()
    stack = session._engine.agents["sat"].flight_software_runtime.stack
    assert stack.config.navigation_filter.value == "alpha_beta"
    assert [action.action_id for action in stack.config.executive.actions] == ["hold-first", "then-retreat"]
    assert stack.config.executive.constraints[0].evaluator_id == "no_active_faults"
