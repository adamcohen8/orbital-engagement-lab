"""Containment and diagnostic contracts for cached boundary record validation."""

from dataclasses import dataclass, fields

import pytest

from sim.flight_software.contracts import TelemetryField
from sim.flight_software.schemas import assert_truth_free, canonical_json_bytes


@dataclass(slots=True)
class Envelope:
    payload: object


def _hidden_truth() -> object:
    return type("StateTruth", (), {"__module__": "sim.core.models"})()


def test_cached_validator_rechecks_actual_values_after_mutation() -> None:
    record = Envelope({"samples": [1.0, {"observable": "ok"}]})
    assert_truth_free(record)
    record.payload["samples"][1]["observable"] = _hidden_truth()  # type: ignore[index]
    with pytest.raises(TypeError) as failure:
        assert_truth_free(record)
    assert str(failure.value) == (
        "$.payload.samples[1].observable contains forbidden simulator-owned value sim.core.models.StateTruth"
    )


def test_specialized_boundary_record_rechecks_nested_mutable_values() -> None:
    values: list[object] = ["observable"]
    record = TelemetryField("external", values)
    assert_truth_free(record)
    values.append({"hidden": _hidden_truth()})
    with pytest.raises(TypeError) as failure:
        assert_truth_free(record)
    assert str(failure.value) == (
        "$.value[1].hidden contains forbidden simulator-owned value sim.core.models.StateTruth"
    )


def test_external_dataclass_metadata_changes_remain_visible_after_warmup() -> None:
    @dataclass
    class ExternalRecord:
        original: object

    record = ExternalRecord("observable")
    assert_truth_free(record)
    fields(ExternalRecord)[0].name = "replacement"
    record.replacement = _hidden_truth()  # type: ignore[attr-defined]
    with pytest.raises(TypeError) as failure:
        assert_truth_free(record)
    assert str(failure.value) == (
        "$.replacement contains forbidden simulator-owned value sim.core.models.StateTruth"
    )


@pytest.mark.parametrize("key", ["simulator_truth", "STATE_TRUTH", "truth_state", "World_Truth"])
def test_cached_validator_retains_truth_field_checks(key: str) -> None:
    assert_truth_free(Envelope({"ordinary": 1}))
    with pytest.raises(TypeError) as failure:
        assert_truth_free(Envelope({key: None}))
    assert str(failure.value) == f"$.payload.{key} is a forbidden simulator-truth field"


def test_cycles_and_shared_references_do_not_hide_later_truth() -> None:
    values: list[object] = []
    record = Envelope(values)
    values.extend([record, record, {"data": [1, 2]}])
    assert_truth_free(record)
    values.append({"nested": _hidden_truth()})
    with pytest.raises(TypeError) as failure:
        assert_truth_free(record)
    assert str(failure.value).startswith("$.payload[3].nested contains forbidden simulator-owned value")


def test_inherited_slotted_dataclass_retains_field_order_and_errors() -> None:
    @dataclass(slots=True)
    class ExtendedEnvelope(Envelope):
        later: object

    assert_truth_free(ExtendedEnvelope((), b"observable"))
    with pytest.raises(TypeError) as failure:
        assert_truth_free(ExtendedEnvelope({"truth_state": 0}, _hidden_truth()))
    assert str(failure.value) == "$.payload.truth_state is a forbidden simulator-truth field"


@pytest.mark.parametrize("field_name", ["K", "class", "payload'); raise RuntimeError('unsafe"])
def test_dynamic_field_names_are_looked_up_without_normalization_or_execution(field_name: str) -> None:
    # Dataclasses built by external adapters can supply nonstandard metadata.
    # A Unicode Kelvin sign must not become ASCII K during source compilation.
    @dataclass
    class ExternalRecord:
        original: object

    fields(ExternalRecord)[0].name = field_name
    record = ExternalRecord(None)
    setattr(record, field_name, {"hidden": _hidden_truth()})
    record.K = "observable"  # type: ignore[attr-defined]
    with pytest.raises(TypeError) as failure:
        assert_truth_free(record)
    assert str(failure.value) == (
        f"$.{field_name}.hidden contains forbidden simulator-owned value sim.core.models.StateTruth"
    )


def test_class_objects_and_open_wrappers_are_still_rejected() -> None:
    assert_truth_free(Envelope(None))
    for value, suffix in ((Envelope, "builtins.type"), (object(), "builtins.object")):
        with pytest.raises(TypeError) as failure:
            assert_truth_free(Envelope(value))
        assert str(failure.value) == f"$.payload contains unsupported boundary wrapper {suffix}"


def test_firewall_and_serialization_keep_separate_nonfinite_contracts() -> None:
    record = Envelope(float("nan"))
    assert_truth_free(record)
    with pytest.raises(ValueError, match="finite"):
        canonical_json_bytes(record)
