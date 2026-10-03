"""Optional native conversion for internal evidence already checked at ingress/egress.

Public serialization retains its Python owner. Native validation delegates
external wrappers and containers to the same recursive Python walker.
The native converter produces the same primitives for the existing JSON writer.
"""

from __future__ import annotations

from .contracts import BOUNDARY_RECORD_TYPES
from .rust_game_backend import extension
from .schemas import (
    _BOUNDARY_SCALAR_TYPES,
    _FORBIDDEN_FIELD_NAMES,
    _assert_truth_free,
    _type_guard_info,
)


def _evidence_type_fields(value_type: type[object]) -> tuple[str, tuple[str, ...] | None]:
    qualified, _forbidden, record_fields = _type_guard_info(value_type)
    names = None if record_fields is None else tuple(item.name for item in record_fields)
    return qualified, names


def trusted_evidence_encoder():
    """Create a runtime-owned cache; explicit Rust selection fails closed."""

    return _EvidenceEncoder()


class _EvidenceEncoder:
    """Recreate the nonsemantic native metadata cache in each worker process."""

    def __init__(self):
        self._native = extension().TrustedPacketEncoder(_evidence_type_fields)

    def convert(self, value):
        return self._native.convert(value)

    def convert_many(self, values):
        return self._native.convert_many(values)

    def __reduce__(self):
        return (trusted_evidence_encoder, ())


def _forbidden_key_error(path: list[str], key: object) -> None:
    raise TypeError(f"{''.join(path)}.{key} is a forbidden simulator-truth field")


def boundary_validator():
    """Keep external reflection and shared cycle tracking with the Python owner."""

    return _BoundaryValidator()


class _BoundaryValidator:
    def __init__(self):
        self._native = _new_native_validator()

    def check(self, value):
        return self._native.check(value)

    def check_many(self, values):
        return self._native.check_many(values)

    def __reduce__(self):
        return (boundary_validator, ())


def _new_native_validator():
    records = {}
    for value_type in BOUNDARY_RECORD_TYPES:
        _qualified, forbidden, record_fields = _type_guard_info(value_type)
        if not forbidden and record_fields is not None:
            records[value_type] = tuple(item.name for item in record_fields)
    return extension().BoundaryValidator(
        records, _BOUNDARY_SCALAR_TYPES, _FORBIDDEN_FIELD_NAMES,
        _assert_truth_free, _forbidden_key_error,
    )
