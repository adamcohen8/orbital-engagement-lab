"""Scenario selectors for the parallel numerical engines."""

import pytest

from sim.config.scenario.analysis import _parse_covariance_section
from sim.config.scenario.models import _plain_config_data
from sim.config.scenario.simulator import _normalize_dynamics_section, _parse_simulator_section
from sim.config.scenario.validation import _require_rust_symbols


@pytest.mark.parametrize("section", ["orbit", "attitude", "rocket", "reentry"])
def test_numeric_backend_selector_normalizes_without_changing_defaults(section) -> None:
    explicit = _normalize_dynamics_section({section: {"numeric_backend": " RUST "}})
    assert explicit[section]["numeric_backend"] == "rust"
    default = _normalize_dynamics_section({section: {}})
    assert "numeric_backend" not in default[section]


@pytest.mark.parametrize("section", ["orbit", "attitude", "rocket", "reentry"])
def test_numeric_backend_selector_rejects_unknown_value(section) -> None:
    with pytest.raises(ValueError, match=rf"simulator.dynamics.{section}.numeric_backend must be python or rust"):
        _normalize_dynamics_section({section: {"numeric_backend": "unknown"}})


def test_collision_backend_selector_retains_existing_default_payload() -> None:
    block = {"enabled": True, "radii_m": {"one": 1.0, "two": 1.0}}
    default = _parse_simulator_section({"collisions": block})
    assert default.collisions == block
    selected = _parse_simulator_section({"collisions": {**block, "numeric_backend": "RUST"}})
    assert selected.collisions == {**block, "numeric_backend": "rust"}
    with pytest.raises(ValueError, match="simulator.collisions.numeric_backend must be python or rust"):
        _parse_simulator_section({"collisions": {**block, "numeric_backend": "unknown"}})


def test_covariance_backend_selector_routes_normalized_choice() -> None:
    assert _parse_covariance_section({}).numeric_backend == "rust"
    assert _plain_config_data(_parse_covariance_section({}))["numeric_backend"] == "rust"
    assert _plain_config_data(_parse_covariance_section({"numeric_backend": "rust"}))["numeric_backend"] == "rust"
    assert _parse_covariance_section({"numeric_backend": " RUST "}).numeric_backend == "rust"
    with pytest.raises(ValueError, match="analysis.covariance.numeric_backend must be python or rust"):
        _parse_covariance_section({"numeric_backend": "unknown"})


def test_native_selection_rejects_an_older_wheel(monkeypatch) -> None:
    import sys
    from types import SimpleNamespace

    monkeypatch.setitem(sys.modules, "oel_rust_orbit", SimpleNamespace())
    with pytest.raises(ValueError, match="missing kernels: attitude_propagate_exponential_map"):
        _require_rust_symbols("attitude Rust", ("attitude_propagate_exponential_map",))
