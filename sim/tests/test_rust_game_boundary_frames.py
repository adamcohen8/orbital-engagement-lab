"""Native boundary containment and exact reductions at gameplay decisions."""

from collections import UserDict
from dataclasses import dataclass, fields
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("oel_rust_game")

from sim.flight_software.contracts import TelemetryField
from sim.flight_software.rust_game_frames import frame_converter, quaternion_to_dcm_bn, training_frame_sample
from sim.flight_software.rust_game_packets import boundary_validator
from sim.flight_software.schemas import assert_truth_free
from sim.utils.frames import eci_relative_to_ric_rect, ric_dcm_ir_from_rv
from sim.utils.quaternion import quaternion_to_dcm_bn as python_dcm


@dataclass
class External:
    value: object


def _result(function, value):
    try:
        function(value)
        return None
    except Exception as error:
        return type(error), str(error)


def test_native_firewall_retains_diagnostics_and_live_external_metadata() -> None:
    native = boundary_validator()
    truth = type("StateTruth", (), {"__module__": "sim.core.models"})()
    values = [
        truth, object(), External, External(truth),
        TelemetryField("hidden", truth), TelemetryField("hidden", {"STATE_TRUTH": None}),
        TelemetryField("nested", UserDict({"items": [External(truth)]})),
        TelemetryField("data", External({"nested": (truth,)})),
        float("nan"),
    ]
    for value in values:
        assert _result(native.check, value) == _result(assert_truth_free, value)
    record = External("observable")
    native.check(record)
    original_name = fields(External)[0].name
    try:
        fields(External)[0].name = "replacement"
        record.replacement = truth  # type: ignore[attr-defined]
        assert _result(native.check, record) == _result(assert_truth_free, record)
        assert _result(native.check, record)[0] is TypeError
    finally:
        fields(External)[0].name = original_name


def test_shared_cycles_across_native_python_walkers_do_not_hide_truth() -> None:
    values: list[object] = []
    record = External(TelemetryField("shared", values))
    values.extend((record, record, {"observable": "ok"}))
    native = boundary_validator()
    native.check(values)
    values.append(type("StateTruth", (), {"__module__": "sim.core.models"})())
    assert _result(native.check, values) == _result(assert_truth_free, values)
    assert _result(native.check, values)[0] is TypeError


@pytest.mark.parametrize("nested", [False, True])
def test_external_predicate_accesses_match_python(nested) -> None:
    visits = []

    class Wrapper:
        @property
        def __class__(self):
            visits.append("class")
            return type(self)

    value = Wrapper()
    if nested:
        value = TelemetryField("external", value)
    visits.clear()
    expected = _result(assert_truth_free, value)
    expected_visits = visits.copy()
    visits.clear()
    assert _result(boundary_validator().check, value) == expected
    assert visits == expected_visits


@pytest.mark.parametrize("name", [
    "test_runtime_keeps_recursive_output_firewall_for_external_stack_subclasses",
    "test_runtime_rejects_an_unsupported_output_contract_before_publication",
    "test_runtime_checks_publicly_enqueued_open_kind_at_point_of_use",
])
def test_native_runtime_retains_ingress_and_egress_firewalls(monkeypatch, name) -> None:
    from sim.tests import test_fsw_truth_firewall_dynamic as regression

    class NativePassiveStack(regression.PassiveFlightSoftwareStack):
        numeric_backend = "rust"

    monkeypatch.setattr(regression, "PassiveFlightSoftwareStack", NativePassiveStack)
    getattr(regression, name)()


def test_deep_boundary_graph_retains_python_recursion_failure() -> None:
    value: object = None
    for _depth in range(1200):
        value = [value]
    assert _result(boundary_validator().check, value)[0] is RecursionError
    assert _result(assert_truth_free, value)[0] is RecursionError


def test_mapping_key_formatting_and_unicode_diagnostics_remain_exact() -> None:
    class Key:
        def __str__(self):
            return "TRUTH_STATE"

        def __format__(self, _spec):
            return "formatted🌒"

    value = {"\ud800": [{Key(): None}]}
    assert _result(boundary_validator().check, value) == _result(assert_truth_free, value)


def test_native_frame_samples_preserve_all_bytes() -> None:
    rng = np.random.default_rng(921)
    for _sample in range(300):
        target = rng.normal(size=6) * [7000, 7000, 7000, 7, 7, 7]
        chaser = target + rng.normal(size=6) * [1, 1, 1, .001, .001, .001]
        reference = target + rng.normal(size=6)
        thrust = rng.normal(size=3) * 1e-5
        expected = (
            eci_relative_to_ric_rect(chaser, target, numeric_backend="python"),
            eci_relative_to_ric_rect(target, reference, numeric_backend="python"),
            ric_dcm_ir_from_rv(target[:3], target[3:]).T @ thrust,
        )
        actual = training_frame_sample(target, chaser, reference, thrust)
        assert all(a.tobytes() == b.tobytes() for a, b in zip(actual, expected, strict=True))
        assert np.asarray(frame_converter().relative_state(chaser.tolist(), target.tolist())).tobytes() == expected[0].tobytes()


def test_quaternion_batch_retains_numpy_scalar_square_rounding() -> None:
    rng = np.random.default_rng(921)
    values = rng.normal(size=(2000, 4))
    values *= np.exp(rng.uniform(-20, 20, size=(2000, 1)))
    # This valid normalized scalar's NumPy power differs from x*x by one ULP.
    values = np.vstack((values, [3.40415213, 1.58780239, 2.93658662, 3.39063256]))
    expected = np.asarray([python_dcm(value) for value in values])
    actual = np.asarray(frame_converter().quaternion_matrices(values.tolist()))
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("value", [[1, 0, 0, 0], [0, 0, 0, 0], [np.nan, 0, 0, 0], [np.inf, 0, 0, 0], [1, 2, 3], [-1, -0., 0., 0.]])
def test_quaternion_guardrails_match_existing_owner(value) -> None:
    assert quaternion_to_dcm_bn(value).tobytes() == python_dcm(np.asarray(value, dtype=float)).tobytes()


@pytest.mark.parametrize("target", [[0, 0, 0, 0, 0, 0], [7000, 0, 0, 7, 0, 0], [np.nan, 0, 0, 0, 7, 0]])
def test_undefined_frame_diagnostics_match(target) -> None:
    target = np.asarray(target, dtype=float)
    deputy = np.asarray([7001, 0, 0, 0, 7, 0], dtype=float)
    assert _result(lambda v: frame_converter().relative_state(deputy.tolist(), v.tolist()), target) == _result(lambda v: eci_relative_to_ric_rect(deputy, v, numeric_backend="python"), target)


def test_existing_accelerated_frame_owner_is_preserved(monkeypatch) -> None:
    from sim.flight_software.rust_game_frames import relative_ric_state
    from sim.utils import frames

    expected = np.asarray([1, 2, 3, 4, 5, 6.])
    monkeypatch.setattr(frames, "_frame_acceleration_enabled", lambda: True)
    monkeypatch.setattr(frames, "eci_relative_to_ric_rect", lambda *_args: expected)
    assert relative_ric_state(expected, expected) is expected


@pytest.mark.parametrize("invalid", ["target", "reference", "thrust"])
def test_invalid_history_sample_retains_exception_and_partial_histories(invalid) -> None:
    from sim.game.training import RPOTrainingConfig, RPOTrainingTracker

    state = np.asarray([7000, 0, 0, 0, 7, 0.])
    snapshot = SimpleNamespace(
        time_s=0., truth={"target": state, "chaser": state + [1, 0, 0, 0, 0, 0], "reference": state},
        applied_thrust={},
    )
    if invalid == "thrust":
        snapshot.applied_thrust["chaser"] = [1, 2]
    else:
        snapshot.truth[invalid] = np.zeros(6)
    config = RPOTrainingConfig(enabled=True, target_reference_object_id="reference")
    python = RPOTrainingTracker(config, numeric_backend="python")
    native = RPOTrainingTracker(config, numeric_backend="rust")
    assert _result(native.record, snapshot) == _result(python.record, snapshot)
    for field in ("t_s", "rel_ric_hist", "target_reference_rel_hist", "target_state_eci_hist", "thrust_hist", "thrust_ric_hist"):
        assert np.asarray(getattr(native, field)).tobytes() == np.asarray(getattr(python, field)).tobytes()
