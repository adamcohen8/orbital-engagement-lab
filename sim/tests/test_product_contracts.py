from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from sim import SimulationConfig, SimulationResult, SimulationSession
from sim.config import AlgorithmPointer, load_simulation_yaml, validate_scenario_plugins
from sim.execution import create_single_run_engine
from sim.execution import service as execution_service
from sim.execution.metrics import closest_approach_from_run_payload


class ContractRecordTimingMission:
    def update(self, *, object_id, truth, t_s, env, **kwargs):
        flags = {
            "object_id": str(object_id),
            "decision_t_s": float(t_s),
            "own_truth_t_s": float(truth.t_s),
            "received_world_truth": "world_truth" in kwargs,
            "env_has_world_truth": "world_truth" in dict(env),
        }
        return {
            "mission_use_integrated_command": True,
            "thrust_eci_km_s2": np.zeros(3, dtype=float),
            "command_mode_flags": flags,
        }


class ContractFailingBridge:
    def step(self, event):
        raise RuntimeError("bridge boom")


def scenario_config_from_dict(data: dict):
    return SimulationConfig.from_dict(data).to_scenario_config()


def _contract_config(output_dir: Path) -> dict:
    return {
        "scenario_name": "contract_smoke",
        "scenario_description": "Contract smoke test scenario",
        "objects": {
            "target": {
                "enabled": True,
                "specs": {"mass_kg": 100.0},
                "initial_state": {
                    "position_eci_km": [7000.0, 0.0, 0.0],
                    "velocity_eci_km_s": [0.0, 7.5, 0.0],
                },
            },
        },
        "simulator": {
            "duration_s": 2.0,
            "dt_s": 1.0,
            "termination": {"earth_impact_enabled": False},
            "dynamics": {"attitude": {"enabled": False}},
        },
        "outputs": {
            "output_dir": str(output_dir),
            "mode": "save",
            "stats": {
                "print_summary": False,
                "save_json": True,
                "save_full_log": True,
            },
            "plots": {"enabled": False, "figure_ids": []},
            "animations": {"enabled": False, "types": []},
        },
        "metadata": {"seed": 123},
    }


def test_validate_scenario_rejects_nonpositive_mass() -> None:
    cfg = scenario_config_from_dict(
        {
            "rocket": {"enabled": False},
            "target": {"enabled": True, "specs": {"mass_kg": 0.0}},
            "chaser": {"enabled": False},
            "simulator": {"duration_s": 1.0, "dt_s": 1.0},
        }
    )

    errs = validate_scenario_plugins(cfg)

    assert any("mass_kg" in err and "> 0" in err for err in errs)


def test_validate_scenario_rejects_zero_absolute_eci_position() -> None:
    cfg = scenario_config_from_dict(
        {
            "rocket": {"enabled": False},
            "target": {
                "enabled": True,
                "specs": {"mass_kg": 100.0},
                "initial_state": {
                    "position_eci_km": [0.0, 0.0, 0.0],
                    "velocity_eci_km_s": [0.0, 0.0, 0.0],
                },
            },
            "chaser": {"enabled": False},
            "simulator": {"duration_s": 1.0, "dt_s": 1.0},
        }
    )

    errs = validate_scenario_plugins(cfg)

    assert any("position_eci_km" in err and "nonzero" in err for err in errs)


def test_engine_contract_keeps_deterministic_time_grid_and_step_snapshots(tmp_path: Path) -> None:
    session = SimulationSession.from_config(SimulationConfig.from_dict(_contract_config(tmp_path)))
    callback_events: list[tuple[int, int]] = []

    initial = session.reset()
    assert initial is not None
    assert initial.step_index == 0
    assert initial.time_s == 0.0
    assert initial.object_ids == ("target",)
    assert session.done is False

    session.run(step_callback=lambda step, total: callback_events.append((step, total)))
    result = session.result

    assert result is not None
    assert callback_events == [(0, 2), (1, 2), (2, 2)]
    assert result.time_s.tolist() == [0.0, 1.0, 2.0]
    assert result.num_steps == 3
    assert result.summary["samples"] == 3
    assert result.summary["duration_s"] == 2.0
    assert result.snapshot(2).time_s == 2.0
    assert result.snapshot(2).object_ids == ("target",)
    assert session.done is True


def test_scenario_yaml_contract_validates_stable_schema_boundaries(tmp_path: Path) -> None:
    preset_dir = tmp_path / "presets"
    preset_dir.mkdir()
    (preset_dir / "target.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "Contract Target",
                "preset_type": "satellite",
                "specs": {
                    "dry_mass_kg": 80.0,
                    "fuel_mass_kg": 20.0,
                    "max_thrust_n": 15.0,
                },
                "knowledge": {"refresh_rate_s": 5.0},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    scenario_path = tmp_path / "scenario.yaml"
    scenario_path.write_text(
        yaml.safe_dump(
            {
                "scenario_name": "yaml_contract",
                "metadata": {"owner": "tests"},
                "objects": {
                    "target": {
                        "enabled": True,
                        "preset": "presets/target.yaml",
                        "specs": {"mass_kg": 95.0},
                        "orbit_control": {
                            "kind": "python",
                            "module": "sim.control.orbit.zero_controller",
                            "class_name": "ZeroController",
                            "params": {},
                        },
                    },
                },
                "simulator": {
                    "duration_s": 4.0,
                    "dt_s": 1.0,
                    "dynamics": {
                        "orbit": {"orbit_substep_s": 0.5},
                        "attitude": {"enabled": False, "attitude_substep_s": 1.0},
                    },
                },
                "outputs": {"output_dir": str(tmp_path / "out"), "mode": "save"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    cfg = load_simulation_yaml(scenario_path)

    assert cfg.scenario_name == "yaml_contract"
    assert cfg.metadata == {"owner": "tests"}
    target = cfg.objects["target"]
    assert target.enabled is True
    assert target.specs["mass_kg"] == 95.0
    assert "dry_mass_kg" not in target.specs
    assert "fuel_mass_kg" not in target.specs
    assert target.specs["max_thrust_n"] == 15.0
    assert target.knowledge["refresh_rate_s"] == 5.0
    assert target.orbit_control == AlgorithmPointer(
        kind="python",
        module="sim.control.orbit.zero_controller",
        class_name="ZeroController",
        function=None,
        file=None,
        params={},
    )

    with pytest.raises(ValueError, match="outputs.stats.save_json"):
        scenario_config_from_dict(
            {
                "simulator": {"duration_s": 2.0, "dt_s": 1.0},
                "outputs": {"stats": {"save_json": "true"}},
            }
        )

    with pytest.raises(ValueError, match="simulator.dt_s must be an integer multiple"):
        scenario_config_from_dict(
            {
                "simulator": {
                    "duration_s": 2.0,
                    "dt_s": 1.0,
                    "dynamics": {"orbit": {"orbit_substep_s": 0.3}},
                }
            }
        )

    with pytest.raises(ValueError, match="Algorithm pointers do not support 'file'"):
        scenario_config_from_dict(
            {
                "target": {"orbit_control": {"file": "controllers.py", "class_name": "Controller"}},
                "simulator": {"duration_s": 2.0, "dt_s": 1.0},
            }
        )


def test_payload_artifact_contract_exposes_stable_summary_histories_and_wrappers(tmp_path: Path) -> None:
    result = SimulationSession.from_config(SimulationConfig.from_dict(_contract_config(tmp_path))).run()
    assert isinstance(result, SimulationResult)

    payload = result.payload
    expected_top_level = {
        "summary",
        "time_s",
        "truth_by_object",
        "target_reference_orbit_truth",
        "belief_by_object",
        "applied_thrust_by_object",
        "applied_torque_by_object",
        "desired_attitude_by_object",
        "knowledge_by_observer",
        "knowledge_detection_by_observer",
        "knowledge_consistency_by_observer",
        "bridge_events_by_object",
        "controller_debug_by_object",
        "rocket_throttle_cmd",
        "rocket_metrics",
    }
    assert expected_top_level.issubset(payload.keys())

    summary = result.summary
    expected_summary = {
        "scenario_name",
        "scenario_description",
        "objects",
        "samples",
        "dt_s",
        "duration_s",
        "terminated_early",
        "termination_reason",
        "termination_time_s",
        "termination_object_id",
        "rocket_insertion_achieved",
        "rocket_insertion_time_s",
        "target_reference_orbit_enabled",
        "thrust_stats",
        "attitude_guardrail_stats",
        "knowledge_detection_by_observer",
        "knowledge_consistency_by_observer",
        "plot_outputs",
        "animation_outputs",
    }
    assert expected_summary.issubset(summary.keys())
    assert summary["scenario_name"] == "contract_smoke"
    assert summary["scenario_description"] == "Contract smoke test scenario"
    assert summary["objects"] == ["target"]
    assert summary["samples"] == len(payload["time_s"]) == 3
    assert summary["terminated_early"] is False
    assert summary["termination_reason"] is None
    assert summary["termination_time_s"] is None
    assert summary["termination_object_id"] is None
    assert summary["plot_outputs"] == {}
    assert summary["animation_outputs"] == {}

    assert result.truth["target"].shape[0] == result.num_steps
    assert result.belief["target"].shape[0] == result.num_steps
    assert result.applied_thrust["target"].shape == (result.num_steps, 3)
    assert result.applied_torque["target"].shape == (result.num_steps, 3)
    assert result.artifacts == {"plots": {}, "animations": {}}
    assert result.metrics["scenario_name"] == "contract_smoke"

    assert (tmp_path / "master_run_summary.json").is_file()
    assert (tmp_path / "master_run_log.json").is_file()


def test_single_run_can_write_binary_history_artifact(tmp_path: Path) -> None:
    raw = _contract_config(tmp_path)
    raw["outputs"]["stats"]["save_history_npz"] = True
    result = SimulationSession.from_config(SimulationConfig.from_dict(raw)).run()

    history_outputs = dict(result.summary["history_binary_outputs"])
    history_path = Path(history_outputs["npz"])

    assert history_path == tmp_path / "master_run_history.npz"
    assert history_path.is_file()
    assert history_outputs["array_count"] >= 5
    with np.load(history_path, allow_pickle=False) as data:
        manifest = json.loads(str(data["manifest_json"].item()))
        assert data["time_s"].tolist() == [0.0, 1.0, 2.0]
        assert data["truth__target"].shape == (3, 14)
        assert data["applied_torque__target"].shape == (3, 3)
        assert manifest["arrays"]["truth__target"]["path"] == "truth_by_object.target"


def test_engine_can_build_run_payload_without_artifact_side_effects(tmp_path: Path) -> None:
    cfg = scenario_config_from_dict(_contract_config(tmp_path))
    engine = create_single_run_engine(cfg)
    while not engine.done:
        engine.step()

    payload = engine.build_run_payload()
    summary = payload["summary"]

    assert summary["samples"] == 3
    assert summary["plot_outputs"] == {}
    assert summary["animation_outputs"] == {}
    assert "output_index_md" not in summary
    assert "output_index_md" not in payload
    assert not (tmp_path / "master_run_summary.json").exists()
    assert not (tmp_path / "master_run_log.json").exists()
    assert not (tmp_path / "README.md").exists()


def test_engine_payload_parts_use_history_views_until_json_materialization(tmp_path: Path) -> None:
    cfg = scenario_config_from_dict(_contract_config(tmp_path))
    engine = create_single_run_engine(cfg)
    while not engine.done:
        engine.step()

    parts = engine._build_payload_parts()

    assert np.shares_memory(parts.t_s, engine.t_s)
    assert np.shares_memory(parts.truth_hist["target"], engine.truth_hist["target"])
    assert np.shares_memory(parts.belief_hist["target"], engine.belief_hist["target"])
    assert np.shares_memory(parts.thrust_hist["target"], engine.thrust_hist["target"])
    assert np.shares_memory(parts.torque_hist["target"], engine.torque_hist["target"])


def test_bridge_runtime_errors_are_recorded_by_default_and_strict_when_requested(tmp_path: Path) -> None:
    raw = _contract_config(tmp_path / "default")
    raw["objects"]["target"]["bridge"] = {
        "enabled": True,
        "module": "sim.tests.test_product_contracts",
        "class_name": "ContractFailingBridge",
    }

    default_result = SimulationSession.from_config(SimulationConfig.from_dict(raw)).run()
    bridge_events = default_result.payload["bridge_events_by_object"]["target"]
    assert bridge_events[0]["bridge_error"] == "bridge boom"

    strict = _contract_config(tmp_path / "strict")
    strict["simulator"]["plugin_validation"] = {"strict_runtime": True}
    strict["objects"]["target"]["bridge"] = raw["objects"]["target"]["bridge"]
    with pytest.raises(RuntimeError, match="target bridge.step failed"):
        SimulationSession.from_config(SimulationConfig.from_dict(strict)).run()


def test_engine_runs_named_multi_satellite_objects_with_relative_initialization(tmp_path: Path) -> None:
    raw = _contract_config(tmp_path)
    raw["objects"] = {
        "red_asset": {
            "kind": "satellite",
            "role": "asset",
            "enabled": True,
            "specs": {"mass_kg": 100.0},
            "initial_state": {
                "position_eci_km": [7000.0, 0.0, 0.0],
                "velocity_eci_km_s": [0.0, 7.5, 0.0],
            },
        },
        "blue_one": {
            "kind": "satellite",
            "role": "interceptor",
            "enabled": True,
            "specs": {"mass_kg": 100.0},
            "initial_state": {
                "relative_to": "red_asset",
                "relative_ric_rect": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
        },
        "blue_two": {
            "kind": "satellite",
            "role": "observer",
            "enabled": True,
            "specs": {"mass_kg": 100.0},
            "initial_state": {
                "position_eci_km": [7010.0, 0.0, 0.0],
                "velocity_eci_km_s": [0.0, 7.49, 0.0],
            },
        },
    }

    result = SimulationSession.from_config(SimulationConfig.from_dict(raw)).run()

    assert result.summary["objects"] == ["blue_one", "blue_two", "red_asset"]
    assert result.summary["reference_object_id"] == "red_asset"
    assert result.summary["primary_object_pair"] == ["blue_one", "red_asset"]
    assert set(result.truth.keys()) == {"blue_one", "blue_two", "red_asset"}
    initial_separation = np.linalg.norm(result.truth["blue_one"][0, 0:3] - result.truth["red_asset"][0, 0:3])
    assert initial_separation == pytest.approx(1.0)
    assert closest_approach_from_run_payload(result.payload) < 2.0


def test_batch_config_temp_yaml_is_cleaned_up_after_serialization_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = scenario_config_from_dict(_contract_config(tmp_path))
    source_path = tmp_path / "scenario.yaml"

    def _fail_safe_dump(*_: object, **__: object) -> None:
        raise RuntimeError("synthetic serialization failure")

    monkeypatch.setattr(execution_service.yaml, "safe_dump", _fail_safe_dump)

    with pytest.raises(RuntimeError, match="synthetic serialization failure"):
        execution_service.SimulationExecutionService().run_batch_from_config(cfg, source_path=source_path)

    assert list(tmp_path.glob("tmp*.yaml")) == []


def test_engine_timing_contract_does_not_expose_world_truth_to_agents(tmp_path: Path) -> None:
    cfg = SimulationConfig.from_dict(
        {
            "scenario_name": "contract_no_world_truth_access",
            "rocket": {"enabled": False},
            "target": {
                "enabled": True,
                "specs": {"mass_kg": 100.0},
                "initial_state": {
                    "position_eci_km": [7000.0, 0.0, 0.0],
                    "velocity_eci_km_s": [1.0, 7.5, 0.0],
                },
            },
            "chaser": {
                "enabled": True,
                "specs": {"mass_kg": 100.0},
                "initial_state": {
                    "position_eci_km": [7100.0, 0.0, 0.0],
                    "velocity_eci_km_s": [0.0, 7.5, 0.0],
                },
                "mission_objectives": [
                    {
                        "module": "sim.tests.test_product_contracts",
                        "class_name": "ContractRecordTimingMission",
                    }
                ],
            },
            "simulator": {
                "duration_s": 1.0,
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
                    "save_full_log": False,
                    "controller_debug": True,
                },
                "plots": {"enabled": False, "figure_ids": []},
                "animations": {"enabled": False, "types": []},
            },
            "monte_carlo": {"enabled": False},
        }
    )

    result = SimulationSession.from_config(cfg).run()
    flags = result.payload["controller_debug_by_object"]["chaser"][0]["mode_flags"]
    target_truth = result.truth["target"]

    assert target_truth[-1, 0] > 7000.5
    assert flags["decision_t_s"] == 0.0
    assert flags["own_truth_t_s"] == 0.0
    assert flags["received_world_truth"] is False
    assert flags["env_has_world_truth"] is False


def test_external_intent_provider_does_not_receive_world_truth(tmp_path: Path) -> None:
    cfg = SimulationConfig.from_dict(
        {
            "scenario_name": "contract_external_intent_no_world_truth_access",
            "rocket": {"enabled": False},
            "target": {
                "enabled": True,
                "specs": {"mass_kg": 100.0},
                "initial_state": {
                    "position_eci_km": [7000.0, 0.0, 0.0],
                    "velocity_eci_km_s": [1.0, 7.5, 0.0],
                },
            },
            "chaser": {
                "enabled": True,
                "specs": {"mass_kg": 100.0},
                "initial_state": {
                    "position_eci_km": [7100.0, 0.0, 0.0],
                    "velocity_eci_km_s": [0.0, 7.5, 0.0],
                },
            },
            "simulator": {
                "duration_s": 1.0,
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
                    "save_full_log": False,
                    "controller_debug": True,
                },
                "plots": {"enabled": False, "figure_ids": []},
                "animations": {"enabled": False, "types": []},
            },
            "monte_carlo": {"enabled": False},
        }
    )
    calls: list[dict[str, object]] = []

    def provider(**kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        return {}

    session = SimulationSession.from_config(cfg)
    session.set_external_intent_provider("chaser", provider)
    result = session.run()

    assert calls
    assert result.truth["target"][-1, 0] > 7000.5
    assert all("world_truth" not in call for call in calls)
    assert all("world_truth" not in dict(call.get("env", {})) for call in calls)
    assert all("own_knowledge" in call for call in calls)


def test_engine_timing_contract_estimates_after_inner_step_propagation(tmp_path: Path) -> None:
    cfg = SimulationConfig.from_dict(
        {
            "scenario_name": "contract_inner_estimation",
            "rocket": {"enabled": False},
            "target": {
                "enabled": True,
                "initial_state": {
                    "position_eci_km": [7000.0, 0.0, 0.0],
                    "velocity_eci_km_s": [0.0, 7.546049108166282, 0.0],
                },
                "knowledge": {
                    "sensor_error": {
                        "pos_sigma_km": [0.0],
                        "vel_sigma_km_s": [0.0],
                        "quat_sigma": [0.0],
                        "omega_sigma_rad_s": [0.0],
                    }
                },
            },
            "chaser": {"enabled": False},
            "simulator": {
                "duration_s": 1.0,
                "dt_s": 1.0,
                "termination": {"earth_impact_enabled": False},
                "dynamics": {
                    "orbit": {"orbit_substep_s": 1.0},
                    "attitude": {"enabled": True, "attitude_substep_s": 0.25},
                },
            },
                "outputs": {
                    "output_dir": str(tmp_path),
                    "mode": "save",
                    "stats": {
                        "print_summary": False,
                        "save_json": False,
                        "save_full_log": False,
                        "controller_debug": True,
                    },
                "plots": {"enabled": False, "figure_ids": []},
                "animations": {"enabled": False, "types": []},
            },
            "monte_carlo": {"enabled": False},
        }
    )

    result = SimulationSession.from_config(cfg).run()
    truth = result.truth["target"][-1, :6]
    belief = result.belief["target"][-1, :6]
    debug_times = [entry["t_s"] for entry in result.payload["controller_debug_by_object"]["target"]]

    assert np.linalg.norm(belief[:3] - truth[:3]) < 1e-9
    assert np.linalg.norm(belief[3:] - truth[3:]) < 1e-12
    assert debug_times == [0.0, 0.25, 0.5, 0.75]
