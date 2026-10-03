from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from sim.config.plugin_validation import _validate_object_knowledge
from sim.knowledge.object_tracking import TrackedObjectConfig
from sim.runtime.knowledge_factory import _build_knowledge_base


def test_tracking_backend_defaults_to_rust() -> None:
    config = TrackedObjectConfig(target_id="target")
    assert config.numeric_backend == "rust"


def test_tracking_backend_selector_accepts_rust_and_rejects_unknown() -> None:
    assert TrackedObjectConfig(target_id="target", numeric_backend=" RUST ").numeric_backend == "rust"
    with pytest.raises(ValueError, match="knowledge numeric_backend must be python or rust"):
        TrackedObjectConfig(target_id="target", numeric_backend="cuda")


def test_object_knowledge_validation_checks_all_backend_locations() -> None:
    assert _validate_object_knowledge({"numeric_backend": "rust"}, "objects.chaser.knowledge") == []
    errors = _validate_object_knowledge(
        {
            "estimation": {"numeric_backend": "bad", "ekf": {"numeric_backend": "also_bad"}},
            "ekf": {"numeric_backend": "wrong"},
        },
        "objects.chaser.knowledge",
    )
    assert len(errors) == 3
    assert all("numeric_backend must be 'python' or 'rust'" in error for error in errors)


def test_runtime_knowledge_factory_wires_explicit_tracking_backend() -> None:
    agent = SimpleNamespace(
        knowledge={
            "targets": ["target"],
            "numeric_backend": "rust",
            "estimation": {
                "measurement_model": "relative_range_rate",
                "ekf": {"initial_state_eci_km_s": [7001.0, 0.0, 0.0, 0.0, 7.5, 0.0]},
            },
        }
    )
    knowledge = _build_knowledge_base("chaser", agent, 1.0, np.random.default_rng(3))
    assert knowledge is not None
    assert knowledge._tracks["target"].sensor.numeric_backend == "rust"
