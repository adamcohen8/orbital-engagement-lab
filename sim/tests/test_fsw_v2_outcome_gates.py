from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sim import SimulationConfig, SimulationSession
from sim.dynamics.orbit.elements import rv_to_coe_eci
from sim.gnc.attitude_v2 import AttitudeReferenceMode
from sim.gnc.orbit_v2 import TranslationControlLaw, TranslationMode
from sim.utils.quaternion import quaternion_to_dcm_bn


def _attitude_config(output_dir: Path, mode: AttitudeReferenceMode) -> SimulationConfig:
    params: dict[str, object] = {
        "reference_mode": mode.value,
        "kp": 0.25,
        "kd": 1.0,
        "max_torque_n_m": 0.08,
    }
    if mode is AttitudeReferenceMode.TARGET:
        params["target_position_eci_m"] = [7.0e6, 1.0e7, 0.0]
    elif mode is AttitudeReferenceMode.THRUST:
        params["thrust_direction_eci"] = [0.0, 0.0, 1.0]
    elif mode is AttitudeReferenceMode.RIC:
        params["ric_axis"] = "radial_out"
    elif mode is AttitudeReferenceMode.REPLAY:
        params.update(
            {
                "replay_times_s": [0.0, 60.0],
                "replay_quaternions_bn": [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            }
        )
    return SimulationConfig.from_dict(
        {
            "scenario_name": f"outcome_attitude_{mode.value}",
            "objects": {
                "sat": {
                    "kind": "satellite",
                    "specs": {
                        "mass_kg": 100.0,
                        "mass_properties": {
                            "inertia_reference_point": "center_of_mass",
                            "inertia_kg_m2": [[5.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 7.0]],
                        },
                    },
                    "initial_state": {
                        "position_eci_km": [7000.0, 0.0, 0.0],
                        "velocity_eci_km_s": [0.0, 7.54605329, 0.0],
                        "attitude_quat_bn": [0.7071067811865476, 0.0, 0.0, 0.7071067811865475],
                        "angular_rate_body_rad_s": [0.05, -0.03, 0.02],
                    },
                    "flight_software": {
                        "stack": "fsw.attitude_reference",
                        "hardware_profile": "hardware.ideal_wrench.v1",
                        "task_period_s": 0.1,
                        "params": params,
                    },
                }
            },
            "simulator": {
                "duration_s": 60.0,
                "dt_s": 1.0,
                "dynamics": {
                    "orbit": {"model": "two_body", "orbit_substep_s": 1.0},
                    "attitude": {"enabled": True, "attitude_substep_s": 0.05},
                },
                "termination": {"earth_impact_enabled": False},
            },
            "outputs": {
                "output_dir": str(output_dir),
                "mode": "save",
                "stats": {"print_summary": False, "save_json": False, "save_full_log": False},
                "plots": {"enabled": False, "figure_ids": []},
                "animations": {"enabled": False, "types": []},
            },
        }
    )


@pytest.mark.parametrize("mode", tuple(AttitudeReferenceMode))
def test_every_advertised_attitude_reference_has_a_closed_loop_truth_outcome(
    tmp_path: Path,
    mode: AttitudeReferenceMode,
) -> None:
    result = SimulationSession.from_config(_attitude_config(tmp_path / mode.value, mode)).run()
    final = result.truth["sat"][-1]
    position = final[:3]
    velocity = final[3:6]
    quaternion = final[6:10]
    rate = final[10:13]
    dcm_bn = quaternion_to_dcm_bn(quaternion)

    if mode in {AttitudeReferenceMode.QUATERNION, AttitudeReferenceMode.REPLAY}:
        alignment = abs(float(quaternion[0]))
    else:
        radial = position / np.linalg.norm(position)
        cross_track = np.cross(position, velocity)
        cross_track /= np.linalg.norm(cross_track)
        direction = {
            AttitudeReferenceMode.NADIR: -radial,
            AttitudeReferenceMode.VELOCITY: velocity / np.linalg.norm(velocity),
            AttitudeReferenceMode.SUN: np.array([1.0, 0.0, 0.0]),
            AttitudeReferenceMode.TARGET: np.array([7000.0, 10000.0, 0.0]) - position,
            AttitudeReferenceMode.RIC: radial,
            AttitudeReferenceMode.THRUST: np.array([0.0, 0.0, 1.0]),
        }[mode]
        direction /= np.linalg.norm(direction)
        alignment = float((dcm_bn @ direction)[0])

    assert alignment > 0.995
    assert np.linalg.norm(rate) < 3.0e-3


def _rpo_config(
    output_dir: Path,
    *,
    mode: TranslationMode,
    initial_relative_ric_km_km_s: tuple[float, ...],
    duration_s: float = 600.0,
    control_law: TranslationControlLaw = TranslationControlLaw.REFERENCE_PD,
    target_relative_ric_m_m_s: tuple[float, ...] | None = None,
) -> SimulationConfig:
    params: dict[str, object] = {
        "reference_object_id": "target",
        "translation_mode": mode.value,
        "goal_mode": "maintenance",
        "control_law": control_law.value,
        "kp_position_s2": 4.0e-5,
        "kd_velocity_s_inv": 1.3e-2,
        "mean_motion_rad_s": 0.0010780076,
        "max_acceleration_m_s2": 0.01,
        "position_tolerance_m": 25.0,
        "velocity_tolerance_m_s": 0.1,
    }
    if target_relative_ric_m_m_s is not None:
        params["target_relative_state_ric_m"] = list(target_relative_ric_m_m_s)
    if mode is TranslationMode.WAYPOINT:
        params["waypoints_ric"] = [list(target_relative_ric_m_m_s or (0.0,) * 6)]
    if mode is TranslationMode.PASSIVE_RETREAT:
        params.update({"retreat_speed_m_s": 1.0, "retreat_coast_range_m": 500.0})

    return SimulationConfig.from_dict(
        {
            "scenario_name": f"outcome_{mode.value}_{control_law.value}",
            "objects": {
                "target": {
                    "kind": "satellite",
                    "specs": {"mass_kg": 250.0},
                    "initial_state": {
                        "position_eci_km": [7000.0, 0.0, 0.0],
                        "velocity_eci_km_s": [0.0, 7.54605329, 0.0],
                    },
                    "flight_software": {
                        "stack": "fsw.passive",
                        "hardware_profile": "hardware.passive.v1",
                    },
                },
                "chaser": {
                    "kind": "satellite",
                    "role": "chaser",
                    "specs": {"mass_kg": 100.0},
                    "initial_state": {
                        "relative_to": "target",
                        "relative_ric_rect": list(initial_relative_ric_km_km_s),
                    },
                    "knowledge": {"refresh_rate_s": 0.5, "targets": ["target"]},
                    "flight_software": {
                        "stack": "fsw.rpo_reference",
                        "hardware_profile": "hardware.ideal_wrench.v1",
                        "task_period_s": 0.5,
                        "params": params,
                    },
                },
            },
            "simulator": {
                "duration_s": duration_s,
                "dt_s": 1.0,
                "dynamics": {
                    "orbit": {"model": "two_body", "orbit_substep_s": 0.5},
                    "attitude": {"enabled": False},
                },
                "termination": {"earth_impact_enabled": False},
            },
            "outputs": {
                "output_dir": str(output_dir),
                "mode": "save",
                "stats": {"print_summary": False, "save_json": False, "save_full_log": False},
                "plots": {"enabled": False, "figure_ids": []},
                "animations": {"enabled": False, "types": []},
            },
        }
    )


@pytest.mark.parametrize(
    ("mode", "initial", "target", "duration_s"),
    (
        (TranslationMode.RIC_HOLD, (0.0, 1.0, 0.0, 0.0, 0.0, 0.0), (0.0, 500.0, 0.0, 0.0, 0.0, 0.0), 900.0),
        (TranslationMode.WAYPOINT, (0.0, 1.0, 0.0, 0.0, 0.0, 0.0), (0.0, 500.0, 0.0, 0.0, 0.0, 0.0), 900.0),
        (TranslationMode.R_BAR_APPROACH, (1.0, 0.0, 0.0, 0.0, 0.0, 0.0), None, 600.0),
        (TranslationMode.V_BAR_APPROACH, (0.0, 1.0, 0.0, 0.0, 0.0, 0.0), None, 600.0),
        (TranslationMode.C_BAR_APPROACH, (0.0, 0.0, 1.0, 0.0, 0.0, 0.0), None, 600.0),
        (TranslationMode.TERMINAL_BRAKING, (0.0, 1.0, 0.0, 0.0, 0.0, 0.0), None, 600.0),
        (TranslationMode.PASSIVE_RETREAT, (0.1, 0.0, 0.0, 0.0, 0.0, 0.0), None, 600.0),
    ),
)
def test_every_advertised_rpo_mode_changes_truth_toward_its_goal(
    tmp_path: Path,
    mode: TranslationMode,
    initial: tuple[float, ...],
    target: tuple[float, ...] | None,
    duration_s: float,
) -> None:
    result = SimulationSession.from_config(
        _rpo_config(
            tmp_path / mode.value,
            mode=mode,
            initial_relative_ric_km_km_s=initial,
            duration_s=duration_s,
            target_relative_ric_m_m_s=target,
        )
    ).run()
    relative = result.relative_state("chaser", "target", frame="ric_rect")
    ranges = result.range_between("chaser", "target")

    assert np.all(np.isfinite(relative))
    if mode is TranslationMode.PASSIVE_RETREAT:
        assert ranges[-1] > 5.0 * ranges[0]
    elif target is not None:
        target_km_km_s = np.asarray(target, dtype=float) / 1.0e3
        assert np.linalg.norm(relative[-1] - target_km_km_s) < 0.05
    else:
        assert ranges[-1] < 0.30 * ranges[0]


@pytest.mark.parametrize("control_law", tuple(TranslationControlLaw))
def test_every_selectable_rpo_control_law_has_a_closed_loop_truth_outcome(
    tmp_path: Path,
    control_law: TranslationControlLaw,
) -> None:
    result = SimulationSession.from_config(
        _rpo_config(
            tmp_path / control_law.value,
            mode=TranslationMode.V_BAR_APPROACH,
            initial_relative_ric_km_km_s=(0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
            control_law=control_law,
        )
    ).run()
    ranges = result.range_between("chaser", "target")

    assert np.all(np.isfinite(ranges))
    if control_law is TranslationControlLaw.RMOE_IF_THEN:
        # The shipped RMOE law is deliberately a slow rule-based maintainer,
        # not a terminal rendezvous controller.
        assert ranges[-1] < 0.98 * ranges[0]
    else:
        assert ranges[-1] < 0.30 * ranges[0]


def test_subsecond_navigation_release_does_not_invent_chief_relative_velocity(tmp_path: Path) -> None:
    result = SimulationSession.from_config(
        _rpo_config(
            tmp_path / "stage_sync",
            mode=TranslationMode.RIC_HOLD,
            initial_relative_ric_km_km_s=(0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
            duration_s=2.0,
            target_relative_ric_m_m_s=(0.0, 1000.0, 0.0, 0.0, 0.0, 0.0),
        )
    ).run()
    relative = result.relative_state("chaser", "target", frame="ric_rect")

    # A passive chief and initially co-moving deputy must remain co-moving at
    # the first outer-step boundary.  The old constant-velocity chief retiming
    # produced several metres/second of fictitious relative motion here.
    assert np.linalg.norm(relative[1, 3:]) < 2.0e-3


def _orbit_objects(mode: TranslationMode) -> dict[str, object]:
    params: dict[str, object] = {
        "translation_mode": mode.value,
        "goal_mode": "maintenance",
        "max_acceleration_m_s2": 0.01,
        "kp_position_s2": 4.0e-5,
        "kd_velocity_s_inv": 1.3e-2,
    }
    initial_state: dict[str, object] = {
        "position_eci_km": [7000.0, 0.0, 0.0],
        "velocity_eci_km_s": [0.0, 7.54605329, 0.0],
    }
    objects: dict[str, object] = {}
    if mode is TranslationMode.SCHEDULED_BURN:
        params["max_acceleration_m_s2"] = 0.02
        params["scheduled_burns"] = [
            {"start_time_s": 10.0, "duration_s": 5.0, "frame": "eci", "delta_v_m_s": [0.1, 0.0, 0.0]}
        ]
    elif mode is TranslationMode.STATIONKEEPING:
        params["target_state_eci_m_m_s"] = [7.0e6, 0.0, 0.0, 0.0, 7546.05329, 0.0]
        initial_state["position_eci_km"] = [7001.0, 0.0, 0.0]
        objects["target"] = {
            "kind": "satellite",
            "initial_state": {
                "position_eci_km": [7000.0, 0.0, 0.0],
                "velocity_eci_km_s": [0.0, 7.54605329, 0.0],
            },
            "flight_software": {"stack": "fsw.passive", "hardware_profile": "hardware.passive.v1"},
        }
    elif mode is TranslationMode.ORBITAL_ELEMENTS:
        params.update({"target_semi_major_axis_m": 7_001_000.0, "target_eccentricity": 0.0})
    elif mode is TranslationMode.ATMOSPHERIC_PASS:
        params.update(
            {
                "raise_start_s": 10.0,
                "raise_end_s": 20.0,
                "prograde_acceleration_m_s2": 0.001,
                "min_raise_altitude_m": 0.0,
            }
        )
    else:  # pragma: no cover - the caller owns the advertised-mode matrix
        raise AssertionError(f"unsupported orbit outcome mode {mode}")
    objects["vehicle"] = {
        "kind": "satellite",
        "specs": {"mass_kg": 100.0},
        "initial_state": initial_state,
        "flight_software": {
            "stack": "fsw.orbit_reference",
            "hardware_profile": "hardware.ideal_wrench.v1",
            "task_period_s": 1.0,
            "params": params,
        },
    }
    return objects


@pytest.mark.parametrize(
    "mode",
    (
        TranslationMode.SCHEDULED_BURN,
        TranslationMode.STATIONKEEPING,
        TranslationMode.ORBITAL_ELEMENTS,
        TranslationMode.ATMOSPHERIC_PASS,
    ),
)
def test_every_advertised_orbit_mode_has_a_physical_truth_outcome(
    tmp_path: Path,
    mode: TranslationMode,
) -> None:
    duration_s = 600.0 if mode in {TranslationMode.STATIONKEEPING, TranslationMode.ORBITAL_ELEMENTS} else 45.0
    result = SimulationSession.from_config(
        SimulationConfig.from_dict(
            {
                "scenario_name": f"outcome_orbit_{mode.value}",
                "objects": _orbit_objects(mode),
                "simulator": {
                    "duration_s": duration_s,
                    "dt_s": 1.0,
                    "dynamics": {
                        "orbit": {"model": "two_body", "orbit_substep_s": 1.0},
                        "attitude": {"enabled": False},
                    },
                    "termination": {"earth_impact_enabled": False},
                },
                "outputs": {
                    "output_dir": str(tmp_path / mode.value),
                    "mode": "save",
                    "stats": {"print_summary": False, "save_json": False, "save_full_log": False},
                    "plots": {"enabled": False, "figure_ids": []},
                    "animations": {"enabled": False, "types": []},
                },
            }
        )
    ).run()

    if mode is TranslationMode.STATIONKEEPING:
        ranges = result.range_between("vehicle", "target")
        assert ranges[-1] < 0.30 * ranges[0]
    elif mode is TranslationMode.ORBITAL_ELEMENTS:
        history = result.truth["vehicle"]
        initial_a_m = rv_to_coe_eci(history[0, :3], history[0, 3:6]).a_km * 1.0e3
        final_a_m = rv_to_coe_eci(history[-1, :3], history[-1, 3:6]).a_km * 1.0e3
        assert abs(final_a_m - 7_001_000.0) < abs(initial_a_m - 7_001_000.0)
    else:
        acceleration = np.nan_to_num(result.applied_thrust["vehicle"])
        delivered_delta_v_m_s = float(np.sum(np.linalg.norm(acceleration, axis=1))) * 1.0e3
        expected_delta_v_m_s = 0.1 if mode is TranslationMode.SCHEDULED_BURN else 0.01
        assert delivered_delta_v_m_s == pytest.approx(expected_delta_v_m_s, abs=1.0e-12)
        if mode is TranslationMode.SCHEDULED_BURN:
            evidence = result.payload["flight_software_evidence_by_object"]["vehicle"]
            diagnostics = []
            for output in evidence["outputs"]:
                for packet in output["telemetry"]:
                    diagnostics.append({field["name"]: field["value"] for field in packet["fields"]})
            assert any(row.get("scheduled_burn_pending_receipts", 0) > 0 for row in diagnostics)
            assert diagnostics[-1]["scheduled_burn_receipt_confirmed"] is True
            assert diagnostics[-1]["scheduled_burn_receipt_failed"] is False
            assert diagnostics[-1]["scheduled_burn_pending_receipts"] == 0
            # This parameterized outcome fixture deliberately uses a
            # maintenance goal, so receipt-confirmed completion remains active
            # rather than transitioning to the terminal "achieved" state.
            assert diagnostics[-1]["goal_state"] == "active"


@pytest.mark.parametrize("mode", (TranslationMode.LOW_THRUST_PHASING, TranslationMode.ORBITAL_ELEMENTS))
def test_every_advertised_low_thrust_mode_has_a_continuous_engine_truth_outcome(
    tmp_path: Path,
    mode: TranslationMode,
) -> None:
    params: dict[str, object] = {
        "translation_mode": mode.value,
        "goal_mode": "maintenance",
        "max_acceleration_m_s2": 0.002,
        "max_force_n": 0.2,
    }
    vehicle: dict[str, object] = {
        "kind": "satellite",
        "specs": {"mass_kg": 100.0, "dry_mass_kg": 80.0},
        "initial_state": {
            "position_eci_km": [7000.0, 0.0, 0.0],
            "velocity_eci_km_s": [0.0, 7.54605329, 0.0],
        },
        "flight_software": {
            "stack": "fsw.low_thrust_reference",
            "hardware_profile": "hardware.continuous_engine.v1",
            "task_period_s": 1.0,
            "params": params,
        },
    }
    objects: dict[str, object] = {"vehicle": vehicle}
    if mode is TranslationMode.LOW_THRUST_PHASING:
        params.update(
            {
                "reference_object_id": "target",
                "target_relative_state_ric_m": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "approach_speed_m_s": 0.1,
                "kp_position_s2": 4.0e-5,
                "kd_velocity_s_inv": 1.3e-2,
                "mean_motion_rad_s": 0.0010780076,
                # Keep this short physical-realization smoke inside a
                # bounded-acceleration envelope.  The Supported convergence
                # claim is evaluated by the six-hour WP6 long-arc scenario.
                "max_acceleration_m_s2": 5.0e-4,
            }
        )
        vehicle["initial_state"] = {
            "relative_to": "target",
            "relative_ric_rect": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        }
        vehicle["knowledge"] = {"refresh_rate_s": 1.0, "targets": ["target"]}
        objects["target"] = {
            "kind": "satellite",
            "initial_state": {
                "position_eci_km": [7000.0, 0.0, 0.0],
                "velocity_eci_km_s": [0.0, 7.54605329, 0.0],
            },
            "flight_software": {"stack": "fsw.passive", "hardware_profile": "hardware.passive.v1"},
        }
    else:
        params.update({"target_semi_major_axis_m": 7_001_000.0, "target_eccentricity": 0.0})

    result = SimulationSession.from_config(
        SimulationConfig.from_dict(
            {
                "scenario_name": f"outcome_low_thrust_{mode.value}",
                "objects": objects,
                "simulator": {
                    "duration_s": 600.0,
                    "dt_s": 1.0,
                    "dynamics": {"orbit": {"orbit_substep_s": 1.0}, "attitude": {"enabled": False}},
                    "termination": {"earth_impact_enabled": False},
                },
                "outputs": {
                    "output_dir": str(tmp_path / mode.value),
                    "mode": "save",
                    "stats": {"print_summary": False, "save_json": False, "save_full_log": False},
                    "plots": {"enabled": False, "figure_ids": []},
                    "animations": {"enabled": False, "types": []},
                },
            }
        )
    ).run()

    assert np.any(np.linalg.norm(np.nan_to_num(result.applied_thrust["vehicle"]), axis=1) > 0.0)
    if mode is TranslationMode.LOW_THRUST_PHASING:
        ranges = result.range_between("vehicle", "target")
        # Mean-motion phasing is intentionally a long-arc maneuver; this
        # short smoke proves bounded physical execution without pretending a
        # 600-second translation is the convergence envelope.
        assert np.max(ranges) < 1.1 * ranges[0]
    else:
        history = result.truth["vehicle"]
        initial_a_m = rv_to_coe_eci(history[0, :3], history[0, 3:6]).a_km * 1.0e3
        final_a_m = rv_to_coe_eci(history[-1, :3], history[-1, 3:6]).a_km * 1.0e3
        assert abs(final_a_m - 7_001_000.0) < abs(initial_a_m - 7_001_000.0)
