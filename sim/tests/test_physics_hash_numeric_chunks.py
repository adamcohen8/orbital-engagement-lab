from __future__ import annotations

import json

import numpy as np
import pytest

from sim.performance.suite import _deterministic_payload, _iter_deterministic_json


@pytest.mark.parametrize("dtype", [np.bool_, np.int8, np.int64, np.uint64, np.float16, np.float32, np.float64])
@pytest.mark.parametrize("length", [0, 1, 1023, 1024, 1025, 2051])
def test_numeric_chunk_encoding_preserves_canonical_bytes(dtype, length):
    values = np.arange(length).astype(dtype)
    if np.issubdtype(dtype, np.floating) and length > 5:
        values[:5] = [np.nan, np.inf, -np.inf, -0.0, 1e-10]
    if dtype == np.uint64 and length:
        values[-1] = np.iinfo(np.uint64).max
    for array in (values, values[::-1], values[::2]):
        expected = json.dumps(_deterministic_payload(array), sort_keys=True, separators=(",", ":"), allow_nan=True)
        assert "".join(_iter_deterministic_json(array)) == expected


def test_object_arrays_keep_nested_field_filtering():
    array = np.array([{"keep": 1, "elapsed_ms": 9}, {"value": np.array([2.0, -0.0])}], dtype=object)
    assert "".join(_iter_deterministic_json(array)) == '[{"keep":1},{"value":[2.0,-0.0]}]'


def test_ndarray_subclasses_retain_recursive_encoding():
    class CustomArray(np.ndarray):
        def tolist(self):
            raise AssertionError("Subclass tolist must not replace scalar iteration")

    array = np.array([1.0, -0.0]).view(CustomArray)
    assert "".join(_iter_deterministic_json(array)) == "[1.0,-0.0]"
