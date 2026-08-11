from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sim.actuators.command_bus import ActuatorDeviceDefinition, ExpiryBehavior
from sim.actuators.physical import ContinuousEngineHardware, ReactionWheelHardware
from sim.config import scenario_config_from_dict
from sim.core.models import StateTruth
from sim.flight_software import (
    ActuatorCommand,
    ContinuousEngineCommand,
    FrameId,
    PacketId,
    PassiveFlightSoftwareStack,
    PassiveStackConfig,
    ReactionWheelTorqueCommand,
    ValidityInterval,
)
from sim.runtime.satellites.flight_software_runtime import SatelliteFlightSoftwareRuntime
from sim.single_run import _run_single_config


def _actuator_config(tmp_path: Path) -> dict:
    return {
        "scenario_name": "actuator_runtime_integration",
        "objects": {
            "reference": {
                "enabled": True,
                "specs": {"mass_kg": 500.0},
                "initial_state": {
                    "position_eci_km": [7000.0, 0.0, 0.0],
                    "velocity_eci_km_s": [0.0, 7.5, 0.0],
                },
                "flight_software": {
                    "stack": "fsw.passive",
                    "hardware_profile": "hardware.passive.v1",
                },
            },
            "target": {
                "enabled": True,
                "specs": {
                    "mass_kg": 500.0,
                    "dry_mass_kg": 450.0,
                    "fuel_mass_kg": 50.0,
                    "isp_s": 1600.0,
                },
                "initial_state": {
                    "relative_to": "reference",
                    "relative_ric_rect": [0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                },
                "flight_software": {
                    "stack": "fsw.low_thrust_reference",
                    "hardware_profile": "hardware.continuous_engine.v1",
                    "task_period_s": 1.0,
                    "params": {
                        "translation_mode": "low_thrust_phasing",
                        "reference_object_id": "reference",
                        "target_relative_state_ric_m": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        "max_force_n": 0.5,
                        "max_acceleration_m_s2": 0.001,
                    },
                },
            }
        },
        "simulator": {
            "duration_s": 2.0,
            "dt_s": 1.0,
            "termination": {"earth_impact_enabled": False},
            "dynamics": {"attitude": {"enabled": False}},
        },
        "outputs": {
            "output_dir": str(tmp_path),
            "mode": "save",
            "stats": {
                "print_summary": False,
                "save_json": False,
                "save_csv": False,
                "save_full_log": False,
                "controller_debug": True,
            },
            "plots": {"enabled": False, "figure_ids": []},
            "animations": {"enabled": False, "types": []},
        },
    }


def _mixed_propulsion_attitude_runtime(
    *,
    dry_mass_kg: float,
    initial_checkpoint: dict[str, object] | None = None,
    publish_commands: bool = True,
) -> tuple[SatelliteFlightSoftwareRuntime, StateTruth]:
    body = FrameId("BODY", "1")
    inertial = FrameId("ECI", "1")
    devices = (
        ActuatorDeviceDefinition("sat", "engine", body, (ContinuousEngineCommand,), ExpiryBehavior.ZERO),
        ActuatorDeviceDefinition("sat", "wheels", body, (ReactionWheelTorqueCommand,), ExpiryBehavior.ZERO),
    )
    runtime = SatelliteFlightSoftwareRuntime(
        satellite_id="sat",
        stack=PassiveFlightSoftwareStack(PassiveStackConfig("sat")),
        devices=devices,
        hardware={
            "engine": ContinuousEngineHardware("engine", max_thrust_n=1.0, specific_impulse_s=100.0),
            "wheels": ReactionWheelHardware(
                "wheels",
                axes_body=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                max_torque_n_m=(1.0, 1.0, 1.0),
                max_momentum_n_m_s=(1.0, 1.0, 1.0),
            ),
        },
        inertial_frame=inertial,
        body_frame=body,
        task_period_ns=1_000_000_000,
        tick_period_ns=1,
        dry_mass_kg=dry_mass_kg,
        initial_checkpoint=initial_checkpoint,
    )
    if publish_commands:
        issued = runtime.clock_tag(0)
        validity = ValidityInterval(issued, runtime.clock_tag(2_000_000_000))
        runtime.command_bus.publish_all(
            (
                ActuatorCommand(
                    PacketId("test", "boot", 0),
                    "sat",
                    "engine",
                    issued,
                    validity,
                    body,
                    ContinuousEngineCommand(1.0),
                ),
                ActuatorCommand(
                    PacketId("test", "boot", 1),
                    "sat",
                    "wheels",
                    issued,
                    validity,
                    body,
                    ReactionWheelTorqueCommand((0.1, 0.0, 0.0)),
                ),
            ),
            received_at=issued,
        )
    truth = StateTruth(
        np.array([7000.0, 0.0, 0.0]),
        np.array([0.0, 7.5, 0.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.zeros(3),
        100.0,
        0.0,
    )
    return runtime, truth


def test_reaction_wheel_hardware_rejects_nonunit_axes_and_nonfinite_restore() -> None:
    with pytest.raises(ValueError, match="axes must be unit vectors"):
        ReactionWheelHardware(
            "wheels",
            axes_body=((2.0, 0.0, 0.0),),
            max_torque_n_m=(1.0,),
            max_momentum_n_m_s=(1.0,),
        )
    hardware = ReactionWheelHardware(
        "wheels",
        axes_body=((1.0, 0.0, 0.0),),
        max_torque_n_m=(1.0,),
        max_momentum_n_m_s=(1.0,),
    )
    with pytest.raises(ValueError, match="checkpoint momentum is invalid"):
        hardware.restore_state({"momentum_n_m_s": [float("nan")]})


def test_propellant_limit_does_not_scale_independent_reaction_wheel_torque() -> None:
    runtime, truth = _mixed_propulsion_attitude_runtime(dry_mass_kg=99.99999)

    command = runtime.command_interval(truth, start_time_ns=0, end_time_ns=1_000_000_000)
    wheel = next(item for item in command.realizations if item.actuator_id == "wheels")
    engine = next(item for item in command.realizations if item.actuator_id == "engine")

    assert engine.saturated and engine.realized_force_n[0] < 0.02
    np.testing.assert_allclose(command.torque_body_n_m, (-0.1, 0.0, 0.0), atol=1.0e-15)
    np.testing.assert_allclose(wheel.realized_torque_n_m, (-0.1, 0.0, 0.0), atol=1.0e-15)
    np.testing.assert_allclose(runtime.hardware["wheels"].momentum_n_m_s, (0.1, 0.0, 0.0), atol=1.0e-15)


def test_delta_v_limit_does_not_scale_independent_reaction_wheel_torque() -> None:
    runtime, truth = _mixed_propulsion_attitude_runtime(dry_mass_kg=0.0)
    runtime.max_delta_v_m_s = 1.0e-5

    command = runtime.command_interval(truth, start_time_ns=0, end_time_ns=1_000_000_000)
    engine = next(item for item in command.realizations if item.actuator_id == "engine")

    assert engine.saturated
    assert engine.realized_force_n[0] == pytest.approx(0.001)
    np.testing.assert_allclose(command.torque_body_n_m, (-0.1, 0.0, 0.0), atol=1.0e-15)
    assert runtime.used_delta_v_m_s == pytest.approx(1.0e-5)


def test_configured_v2_stack_commands_physical_continuous_engine(tmp_path: Path) -> None:
    payload = _run_single_config(scenario_config_from_dict(_actuator_config(tmp_path)))
    evidence = payload["flight_software_evidence_by_object"]["target"]

    assert evidence["outputs"]
    assert evidence["receipts"]
    assert evidence["realizations"]
    assert payload["summary"]["actuator_diagnostics_summary"]["target"]["actuator_stack_samples"] > 0
    assert payload["summary"]["actuator_diagnostics_summary"]["target"]["max_electric_propulsion_thrust_n"] > 0.0
    mass_history = np.asarray(payload["truth_by_object"]["target"], dtype=float)[:, 13]
    assert mass_history[-1] < mass_history[0]
    assert mass_history[-1] >= 450.0
    assert any(float(row["mass_flow_kg_s"]) > 0.0 for row in evidence["realizations"])


def test_legacy_actuator_preset_does_not_replace_declared_v2_hardware(tmp_path: Path) -> None:
    config = _actuator_config(tmp_path)
    config["objects"]["target"]["specs"]["actuator_preset"] = "BASIC_ELECTRIC_PROPULSION"

    payload = _run_single_config(scenario_config_from_dict(config))
    summary = payload["summary"]["actuator_diagnostics_summary"]["target"]

    assert summary["actuator_stack_samples"] > 0
    assert summary["max_electric_propulsion_thrust_n"] <= 0.5


def test_sensor_release_cadence_is_independent_from_task_and_receipts_do_not_retrigger(
    tmp_path: Path,
) -> None:
    config = _actuator_config(tmp_path)
    config["objects"]["target"]["knowledge"] = {
        "refresh_rate_s": 0.5,
        "targets": ["reference"],
        "sensor_error": {"seed": 4},
    }

    payload = _run_single_config(scenario_config_from_dict(config))
    evidence = payload["flight_software_evidence_by_object"]["target"]
    gnss_times_ns = [
        int(row["payload"]["sample_time"]["ticks"])
        for row in evidence["input_events"]
        if row["kind"] == "measurement" and row["payload"]["sensor_id"] == "gnss"
    ]

    assert gnss_times_ns == [0, 500_000_000, 1_000_000_000, 1_500_000_000, 2_000_000_000]
    assert len(evidence["invocations"]) == 5


def test_runtime_records_skipped_periodic_releases_in_audit_and_checkpoint_state() -> None:
    runtime, truth = _mixed_propulsion_attitude_runtime(dry_mass_kg=0.0)

    runtime.prepare_interval(truth, start_time_ns=0)
    runtime.prepare_interval(truth, start_time_ns=3_000_000_000)

    latest = runtime.review_evidence()["invocations"][-1]
    assert latest["missed_task_releases"] == 2
    assert latest["missed_sensor_releases"] == 2
    checkpoint_state = runtime._runtime_state(run_time_ns=3_000_000_000)
    assert checkpoint_state["missed_task_releases"] == 2
    assert checkpoint_state["missed_sensor_releases"] == 2

    runtime.shutdown(time_ns=3_000_000_000)
    checkpoint = runtime.review_evidence()["snapshots"][-1]
    restored, _ = _mixed_propulsion_attitude_runtime(
        dry_mass_kg=0.0,
        initial_checkpoint=checkpoint,
        publish_commands=False,
    )
    restored_state = restored._runtime_state(run_time_ns=0)
    assert restored_state["missed_task_releases"] == 2
    assert restored_state["missed_sensor_releases"] == 2
