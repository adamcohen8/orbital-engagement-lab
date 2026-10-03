from __future__ import annotations

from dataclasses import FrozenInstanceError
from unittest.mock import patch

import numpy as np
import pytest

from sim.api import SimulationConfig, SimulationSession
from sim.config import scenario_config_from_dict, validate_scenario_plugins
from sim.security.sealed_mode import validate_sealed_mode

STAGE_TIMES: list[tuple[float, tuple[str, ...]]] = []


def balanced_pair(context):
    ids = tuple(context.states_eci_km_km_s)
    STAGE_TIMES.append((context.time_s, ids))
    assert len(ids) == 2
    assert all(state.shape == (6,) and not state.flags.writeable for state in context.states_eci_km_km_s.values())
    assert context.epoch_jd_utc >= 2460000.5
    with pytest.raises(ValueError):
        context.states_eci_km_km_s[ids[0]][0] = 0.0
    with pytest.raises(TypeError):
        context.states_eci_km_km_s[ids[0]] = np.zeros(6)
    with pytest.raises(FrozenInstanceError):
        context.time_s = -1.0
    a = np.array([1e-6, 0.0, 0.0])
    return {ids[0]: a, ids[1]: -a * context.masses_kg[ids[0]] / context.masses_kg[ids[1]]}


def broken_pair(context):
    return {next(iter(context.states_eci_km_km_s)): np.zeros(3)}


def zero_pair(context):
    return {oid: np.zeros(3) for oid in context.states_eci_km_km_s}


def _scenario(tmp_path, *, pointer=None):
    scenario = {
        "scenario_name": "system_force_test",
        "objects": {
            "a": {
                "kind": "satellite", "runtime_profile": "trajectory_only",
                "specs": {"mass_kg": 100.0},
                "initial_state": {"position_eci_km": [7000.0, 0.0, 0.0], "velocity_eci_km_s": [0.0, 7.5, 0.0]},
            },
            "b": {
                "kind": "satellite", "runtime_profile": "trajectory_only",
                "specs": {"mass_kg": 500.0},
                "initial_state": {"position_eci_km": [7000.0, 0.02, 0.0], "velocity_eci_km_s": [0.0, 7.5, 0.0]},
            },
        },
        "simulator": {
            "duration_s": 2.0, "dt_s": 1.0, "initial_jd_utc": 2460000.5,
            "dynamics": {"orbit": {"integrator": "rk4"}, "attitude": {"enabled": False}},
        },
        "outputs": {"mode": "save", "output_dir": str(tmp_path)},
    }
    if pointer is not None:
        scenario["simulator"]["system_force_models"] = [pointer]
    return scenario


def _pointer(function="balanced_pair"):
    return {"module": "sim.tests.test_system_force_models", "function": function}


def test_system_force_is_stage_synchronized_and_changes_both_objects(tmp_path):
    STAGE_TIMES.clear()
    raw = _scenario(tmp_path / "coupled", pointer=_pointer())
    cfg = scenario_config_from_dict(raw)
    assert validate_scenario_plugins(cfg) == []
    assert cfg.to_dict()["simulator"]["system_force_models"][0]["function"] == "balanced_pair"
    assert "system_force_models" not in scenario_config_from_dict(_scenario(tmp_path / "plain")).to_dict()["simulator"]
    actual = SimulationSession.from_config(SimulationConfig.from_dict(raw)).run()
    baseline = SimulationSession.from_config(SimulationConfig.from_dict(_scenario(tmp_path / "baseline"))).run()
    assert len(STAGE_TIMES) == 8
    assert [item[0] for item in STAGE_TIMES[:4]] == [0.0, 0.5, 0.5, 1.0]
    assert all(item[1] == ("a", "b") for item in STAGE_TIMES)
    assert actual.truth["a"][-1, 3] > baseline.truth["a"][-1, 3]
    assert actual.truth["b"][-1, 3] < baseline.truth["b"][-1, 3]
    impulse_delta = (
        100.0 * (actual.truth["a"][-1, 3] - baseline.truth["a"][-1, 3])
        + 500.0 * (actual.truth["b"][-1, 3] - baseline.truth["b"][-1, 3])
    )
    assert abs(impulse_delta) < 1e-8


def test_system_force_rejects_incomplete_return(tmp_path):
    raw = _scenario(tmp_path / "broken", pointer=_pointer("broken_pair"))
    with pytest.raises(ValueError, match="one acceleration for each object"):
        SimulationSession.from_config(SimulationConfig.from_dict(raw)).run()


def test_zero_system_force_matches_independent_two_body_path(tmp_path):
    coupled = SimulationSession.from_config(SimulationConfig.from_dict(
        _scenario(tmp_path / "zero", pointer=_pointer("zero_pair"))
    )).run()
    baseline = SimulationSession.from_config(SimulationConfig.from_dict(
        _scenario(tmp_path / "ordinary")
    )).run()
    for oid in ("a", "b"):
        np.testing.assert_allclose(coupled.truth[oid], baseline.truth[oid], rtol=0, atol=1e-12)


def test_rust_system_force_stages_match_python(tmp_path):
    pytest.importorskip("oel_rust_orbit")
    results = []
    calls = []
    for backend in ("python", "rust"):
        STAGE_TIMES.clear()
        raw = _scenario(tmp_path / backend, pointer=_pointer())
        raw["simulator"]["dynamics"]["orbit"]["numeric_backend"] = backend
        results.append(SimulationSession.from_config(SimulationConfig.from_dict(raw)).run())
        calls.append(list(STAGE_TIMES))
    assert calls[0] == calls[1]
    for oid in ("a", "b"):
        np.testing.assert_allclose(results[1].truth[oid], results[0].truth[oid], rtol=0, atol=1e-10)


def test_system_force_safe_validation_and_sealed_boundary(tmp_path):
    raw = _scenario(tmp_path, pointer={"module": "external_forces", "function": "pair"})
    cfg = scenario_config_from_dict(raw)
    with patch("sim.config.plugin_validation.importlib.import_module", side_effect=AssertionError("imported")):
        assert validate_scenario_plugins(cfg, import_plugins=False) == []
    assert any("system_force_models[0]" in error for error in validate_sealed_mode(cfg))


@pytest.mark.parametrize("change, error", [
    (lambda raw: raw["simulator"]["dynamics"]["attitude"].update(enabled=True), "attitude.enabled=false"),
    (lambda raw: raw["simulator"]["dynamics"]["orbit"].update(j2=True), "unperturbed two-body"),
    (lambda raw: raw["simulator"]["dynamics"]["orbit"].update(integrator="rkf78"), "with RK4"),
    (lambda raw: raw["objects"]["a"].update(force_models=[_pointer("zero_pair")]), "cannot be combined"),
])
def test_system_force_envelope_rejected_at_config_parse(tmp_path, change, error):
    raw = _scenario(tmp_path, pointer=_pointer())
    change(raw)
    with pytest.raises(ValueError, match=error):
        scenario_config_from_dict(raw)
