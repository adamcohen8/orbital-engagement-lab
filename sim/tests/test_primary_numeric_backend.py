"""Exercise the primary native route and the explicit reference override."""

import numpy as np
import pytest

from sim.api import SimulationConfig
from sim.dynamics.orbit.accelerations import OrbitContext
from sim.dynamics.orbit.propagator import OrbitPropagator
from sim.game.backend import configure_game_backend


def test_omitted_selector_executes_native_and_python_override_matches():
    pytest.importorskip("oel_rust_orbit")
    initial = np.array([7000., 0., 0., 0., 7.5, 0.])
    context = OrbitContext(398600.4418, 100.)
    primary = OrbitPropagator()
    reference = OrbitPropagator(numeric_backend="python")
    command = np.zeros(3)
    actual = primary.propagate(initial, 10., 0., command, {}, context)
    expected = reference.propagate(initial, 10., 0., command, {}, context)
    assert primary.last_numeric_path.startswith("rust_")
    np.testing.assert_allclose(actual, expected, rtol=0., atol=1e-11)


def test_trainer_python_override_reaches_numeric_sections():
    config = SimulationConfig.from_yaml("sim/game/configs/game_training_rpo_00_tutorial.yaml")
    selected = configure_game_backend(config, "python").to_dict()
    assert selected["simulator"]["dynamics"]["orbit"]["numeric_backend"] == "python"
    assert selected["metadata"]["game"]["backend"] == "python"


@pytest.mark.parametrize("field", ["orbit", "attitude", "rocket", "reentry"])
def test_boolean_false_is_not_a_numeric_backend(field):
    from sim.config.scenario.simulator import _normalize_dynamics_section

    with pytest.raises(ValueError, match="must be python or rust"):
        _normalize_dynamics_section({field: {"numeric_backend": False}})


def test_trainer_metadata_python_reaches_physics_without_cli_override():
    config = SimulationConfig.from_yaml("sim/game/configs/game_training_rpo_00_tutorial.yaml")
    raw = config.to_dict()
    raw["metadata"]["game"]["backend"] = "python"
    source = SimulationConfig.from_dict(raw, source_path=config.source_path)
    selected = configure_game_backend(source).to_dict()
    assert selected["simulator"]["dynamics"]["orbit"]["numeric_backend"] == "python"
    assert source.to_dict() == raw


def test_tracking_python_measurement_and_jacobian_need_no_extension(monkeypatch):
    import sim.rust_tracking_backend as native
    from sim.knowledge.object_tracking import _relative_measurement_and_jacobian

    def unavailable(*args, **kwargs):
        raise AssertionError("explicit Python requested native tracking")

    monkeypatch.setattr(native, "_extension", unavailable)
    target = np.array([7001., 2., 3., .1, 7.6, .2])
    observer = np.array([7000., 0., 0., 0., 7.5, 0.])
    measurement, jacobian = _relative_measurement_and_jacobian(
        "relative_angles_range_rate", target, observer, numeric_backend="python"
    )
    assert measurement.shape == (4,)
    assert jacobian.shape == (4, 6)
    assert np.all(np.isfinite(jacobian))
    assert measurement[2] == pytest.approx(np.sqrt(14.))
