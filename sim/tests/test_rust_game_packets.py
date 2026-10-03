"""Exact evidence representation and boundary containment for native conversion."""

from collections import UserDict
from dataclasses import dataclass
from enum import Enum, IntEnum

import numpy as np
import pytest

pytest.importorskip("oel_rust_game")

from sim.api import SimulationConfig
from sim.flight_software.rust_game_packets import trusted_evidence_encoder
from sim.flight_software.schemas import (
    _canonical_primitive_json_bytes,
    _to_primitive_trusted,
    assert_truth_free,
    to_primitive,
)
from sim.game.backend import create_game_physics_session


class IntegerChoice(IntEnum):
    ONE = 1


class StringChoice(str, Enum):
    VALUE = "value"


class NestedChoice(Enum):
    VALUE = (1, "nested")


@dataclass(frozen=True, slots=True)
class Record:
    data: object


@pytest.mark.parametrize(
    "value",
    [
        None, True, False, 2**200, -2**200, -0.0, 1e-300, 1e300,
        'é ☾ \n\t\\ " \u0000', b"\x00\xff\x80", IntegerChoice.ONE, StringChoice.VALUE, NestedChoice.VALUE,
        Record((None, {"z": Record([1.25, b"bytes"]), "a": "é"})),
        UserDict({"unicode": "🌒", "numbers": (-0.0, 1e-7, 1e20)}),
    ],
)
def test_native_conversion_preserves_canonical_bytes(value: object) -> None:
    assert_truth_free(value)
    encoder = trusted_evidence_encoder()
    expected = _canonical_primitive_json_bytes(_to_primitive_trusted(value))
    for _repeat in range(2):  # Exercise the metadata cache too.
        assert _canonical_primitive_json_bytes(encoder.convert(value)) == expected
        batch = encoder.convert_many((value, value))
        assert [_canonical_primitive_json_bytes(item) for item in batch] == [expected, expected]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), {1: "bad"}, object(), Record])
def test_native_conversion_preserves_rejections(value: object) -> None:
    with pytest.raises((TypeError, ValueError)) as python_failure:
        _to_primitive_trusted(value)
    encoder = trusted_evidence_encoder()
    for function in (encoder.convert, lambda item: encoder.convert_many((item,))):
        with pytest.raises(type(python_failure.value)) as native_failure:
            function(value)
        assert str(native_failure.value) == str(python_failure.value)


def test_native_evidence_conversion_does_not_weaken_public_firewall() -> None:
    # Internal trusted conversion is never a substitute for ingress validation.
    fake_truth = type("StateTruth", (), {"__module__": "sim.core.models"})()
    encoder = trusted_evidence_encoder()
    encoder.convert(Record("observable"))
    with pytest.raises(TypeError, match="forbidden simulator-owned"):
        assert_truth_free(Record(fake_truth))
    with pytest.raises(TypeError, match="forbidden simulator-truth field"):
        to_primitive(Record({"truth_state": "hidden"}))


def test_native_converter_handles_cycles_without_crashing() -> None:
    values: list[object] = []
    values.append(values)
    with pytest.raises(RecursionError):
        trusted_evidence_encoder().convert(values)


def test_encoder_can_be_used_on_another_thread() -> None:
    from concurrent.futures import ThreadPoolExecutor

    encoder = trusted_evidence_encoder()
    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(encoder.convert, Record((1, 2))).result() == {"data": [1, 2]}


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_runtime_conversion_and_export_preserve_bytes_and_evidence(backend: str) -> None:
    config = SimulationConfig.from_yaml("sim/game/configs/game_training_rpo_10_defensive_target_demo.yaml")
    session = create_game_physics_session(config, backend=backend)
    session.reset(seed=1)
    for _tick in range(12):
        snapshot = session.step(dt_s=0.5)
        assert all(np.isfinite(state).all() for state in snapshot.truth.values())
    seen_native = False
    for agent in session._engine.agents.values():
        runtime = agent.flight_software_runtime
        if runtime is None:
            continue
        native = runtime._native_evidence_encoder
        seen_native |= native is not None
        before = _canonical_primitive_json_bytes(_to_primitive_trusted(runtime.evidence.invocations))
        actual = runtime.review_evidence()
        assert _canonical_primitive_json_bytes(runtime.evidence.invocations) == before
        for name, values in actual.items():
            expected = _to_primitive_trusted(getattr(runtime.evidence, name))
            assert _canonical_primitive_json_bytes(values) == _canonical_primitive_json_bytes(expected)
    assert seen_native == (backend == "rust")
