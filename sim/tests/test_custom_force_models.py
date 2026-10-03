from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from sim.api import SimulationConfig, SimulationSession
from sim.config import scenario_config_from_dict, validate_scenario_plugins
from sim.dynamics.orbit.accelerations import OrbitContext
from sim.dynamics.orbit.custom_force import CustomForceModel
from sim.runtime.satellite_factory import _build_orbit_propagator
from sim.security.sealed_mode import validate_sealed_mode

INITIAL_JD = 2460000.5


class RecordingForce:
    def __init__(self, acceleration_km_s2: float = 0.001) -> None:
        self.value = acceleration_km_s2
        self.epochs: list[float] = []

    def acceleration(self, state_eci_km_km_s: np.ndarray, epoch_jd_utc: float) -> np.ndarray:
        assert state_eci_km_km_s.shape == (6,)
        assert not state_eci_km_km_s.flags.writeable
        self.epochs.append(epoch_jd_utc)
        return np.array([self.value, 0.0, 0.0])


class MissingAcceleration:
    pass


def constant_force(state_eci_km_km_s: np.ndarray, epoch_jd_utc: float) -> np.ndarray:
    assert state_eci_km_km_s.shape == (6,)
    assert epoch_jd_utc >= INITIAL_JD
    return np.array([0.001, 0.0, 0.0])


CONTEXT_SAMPLES: list[tuple[str, float, float]] = []


def context_force(state_eci_km_km_s: np.ndarray, epoch_jd_utc: float, context) -> np.ndarray:
    assert context.object_id == "sat"
    assert context.snapshot_basis == "start_of_step"
    partner = context.other_objects["other"]
    assert partner.state_eci_km_km_s.shape == (6,)
    assert not partner.state_eci_km_km_s.flags.writeable
    assert partner.mass_kg == 100.0
    CONTEXT_SAMPLES.append((context.object_id, epoch_jd_utc, partner.time_s))
    return np.zeros(3)


def _scenario(*, force_models: list | None = None, integrator: str = "rk4") -> dict:
    satellite = {
        "kind": "satellite",
        "runtime_profile": "trajectory_only",
        "specs": {"mass_kg": 100.0},
        "initial_state": {"default_circular_earth": True},
    }
    if force_models is not None:
        satellite["force_models"] = force_models
    return {
        "scenario_name": "custom_force_test",
        "objects": {"sat": satellite},
        "simulator": {
            "duration_s": 2.0,
            "dt_s": 1.0,
            "initial_jd_utc": INITIAL_JD,
            "dynamics": {"orbit": {"integrator": integrator}},
        },
    }


def _class_pointer(**params: float) -> dict:
    return {
        "module": "sim.tests.test_custom_force_models",
        "class_name": "RecordingForce",
        "params": params,
    }


@pytest.mark.parametrize("integrator", ["rk4", "rkf78"])
def test_force_model_receives_integrator_stage_state_and_epoch(integrator: str) -> None:
    config = scenario_config_from_dict(_scenario(force_models=[_class_pointer()], integrator=integrator))
    assert validate_scenario_plugins(config) == []
    propagator = _build_orbit_propagator(config, force_models=config.objects["sat"].force_models)
    plugin = propagator.plugins[0]
    state = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
    original = state.copy()
    context = OrbitContext(mu_km3_s2=0.0, mass_kg=100.0)

    result = propagator.propagate(state, 2.0, 0.0, np.zeros(3), {}, context)

    assert plugin.acceleration.__self__.epochs[0] == INITIAL_JD
    assert plugin.acceleration.__self__.epochs[-1] == pytest.approx(INITIAL_JD + 2.0 / 86400.0)
    assert any(INITIAL_JD < epoch < INITIAL_JD + 2.0 / 86400.0 for epoch in plugin.acceleration.__self__.epochs)
    np.testing.assert_array_equal(state, original)
    assert result[3] == pytest.approx(0.002)


def test_function_pointer_changes_only_configured_object_trajectory(tmp_path) -> None:
    with_force = _scenario(force_models=[{
        "module": "sim.tests.test_custom_force_models", "function": "constant_force"
    }])
    with_force["objects"]["other"] = {
        "kind": "satellite",
        "runtime_profile": "trajectory_only",
        "specs": {"mass_kg": 100.0},
        "initial_state": {"default_circular_earth": True},
    }
    with_force["outputs"] = {"mode": "save", "output_dir": str(tmp_path / "with_force")}
    baseline = _scenario()
    baseline["objects"]["other"] = dict(with_force["objects"]["other"])
    baseline["outputs"] = {"mode": "save", "output_dir": str(tmp_path / "baseline")}
    assert validate_scenario_plugins(scenario_config_from_dict(with_force)) == []

    actual = SimulationSession.from_config(SimulationConfig.from_dict(with_force)).run()
    expected = SimulationSession.from_config(SimulationConfig.from_dict(baseline)).run()

    assert actual.truth["sat"][-1, 3] - expected.truth["sat"][-1, 3] == pytest.approx(0.002, abs=1e-8)
    np.testing.assert_array_equal(actual.truth["other"], expected.truth["other"])


@pytest.mark.parametrize("integrator", ["rk4", "rkf78", "dopri5"])
def test_rust_integrator_preserves_custom_force_scenario(tmp_path, integrator: str) -> None:
    pytest.importorskip("oel_rust_orbit")
    results = []
    for backend in ("python", "rust"):
        raw = _scenario(force_models=[_class_pointer()], integrator=integrator)
        raw["simulator"]["dynamics"]["orbit"]["numeric_backend"] = backend
        raw["outputs"] = {"mode": "save", "output_dir": str(tmp_path / backend)}
        results.append(SimulationSession.from_config(SimulationConfig.from_dict(raw)).run())
    np.testing.assert_allclose(results[1].truth["sat"], results[0].truth["sat"], rtol=0, atol=1e-9)


def test_optional_context_exposes_time_labelled_other_object_truth(tmp_path) -> None:
    CONTEXT_SAMPLES.clear()
    raw = _scenario(force_models=[{
        "module": "sim.tests.test_custom_force_models", "function": "context_force"
    }])
    raw["objects"]["other"] = {
        "kind": "satellite", "runtime_profile": "trajectory_only",
        "specs": {"mass_kg": 100.0},
        "initial_state": {"default_circular_earth": True},
    }
    raw["outputs"] = {"mode": "save", "output_dir": str(tmp_path / "context")}
    SimulationSession.from_config(SimulationConfig.from_dict(raw)).run()
    assert CONTEXT_SAMPLES
    assert any(sample[2] == 0.0 for sample in CONTEXT_SAMPLES)
    assert any(sample[2] == 1.0 for sample in CONTEXT_SAMPLES)


def test_safe_validation_checks_shape_without_import_and_sealed_mode_blocks_module() -> None:
    raw = _scenario(force_models=[{"module": "external_models", "function": "obscure_force"}])
    config = scenario_config_from_dict(raw)
    with patch("sim.config.plugin_validation.importlib.import_module", side_effect=AssertionError("imported")):
        assert validate_scenario_plugins(config, import_plugins=False) == []
    assert any("force_models[0]" in error and "blocks plugin module" in error for error in validate_sealed_mode(config))


def test_force_model_serialization_is_opt_in() -> None:
    baseline = scenario_config_from_dict(_scenario())
    configured = scenario_config_from_dict(_scenario(force_models=[_class_pointer()]))

    assert "force_models" not in baseline.to_dict()["objects"]["sat"]
    assert configured.to_dict()["objects"]["sat"]["force_models"][0]["class_name"] == "RecordingForce"


def test_force_model_requires_eci_onp_and_absolute_epoch() -> None:
    raw = _scenario(force_models=[_class_pointer()])
    raw["simulator"].pop("initial_jd_utc")
    with pytest.raises(ValueError, match="requires simulator.initial_jd_utc"):
        scenario_config_from_dict(raw)
    raw["simulator"]["initial_jd_utc"] = INITIAL_JD
    raw["simulator"]["dynamics"]["orbit"]["model"] = "cr3bp"
    with pytest.raises(ValueError, match="requires ECI ONP special propagation"):
        scenario_config_from_dict(raw)
    raw["simulator"]["dynamics"]["orbit"]["model"] = "two_body"
    raw["objects"]["sat"]["propagation_method"] = "general"
    with pytest.raises(ValueError, match="requires ECI ONP special propagation"):
        scenario_config_from_dict(raw)


def test_force_model_pointer_and_output_errors_are_explicit() -> None:
    raw = _scenario(force_models=[{
        "module": "sim.tests.test_custom_force_models", "class_name": "MissingAcceleration"
    }])
    assert any("missing required callable method 'acceleration'" in error for error in validate_scenario_plugins(scenario_config_from_dict(raw)))
    raw["objects"]["sat"]["force_models"] = [{
        "module": "sim.tests.test_custom_force_models", "function": "constant_force", "params": {"value": 1}
    }]
    assert any("function force models do not accept params" in error for error in validate_scenario_plugins(scenario_config_from_dict(raw), import_plugins=False))
    model = CustomForceModel(lambda state, epoch: [float("nan"), 0.0, 0.0], INITIAL_JD, "bad")
    with pytest.raises(ValueError, match="three finite ECI acceleration"):
        model(0.0, np.zeros(6), {}, None)


def test_optional_numeric_acceleration_keeps_custom_force() -> None:
    raw = _scenario(force_models=[_class_pointer()])
    raw["simulator"]["dynamics"]["orbit"]["j2"] = True
    config = scenario_config_from_dict(raw)
    pointer_list = config.objects["sat"].force_models
    state = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
    context = OrbitContext(mu_km3_s2=398600.4418, mass_kg=100.0)
    environment: dict = {}
    command = np.zeros(3)

    config.simulator.acceleration["mode"] = "off"
    reference = _build_orbit_propagator(config, force_models=pointer_list)
    expected = reference.propagate(state, 1.0, 0.0, command, environment, context)
    config.simulator.acceleration["mode"] = "auto"
    candidate = _build_orbit_propagator(config, force_models=pointer_list)
    actual = candidate.propagate(state, 1.0, 0.0, command, environment, context)

    np.testing.assert_array_equal(actual, expected)
