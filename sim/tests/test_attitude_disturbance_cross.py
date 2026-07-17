from __future__ import annotations

import numpy as np

from sim.dynamics.attitude.disturbances import _cross3, _norm3


def test_specialized_cross3_matches_numpy_bit_for_bit() -> None:
    rng = np.random.default_rng(20260716)
    vectors = rng.standard_normal((1000, 6))

    for row in vectors:
        expected = np.cross(row[:3], row[3:])
        actual = _cross3(row[:3], row[3:])
        np.testing.assert_array_equal(actual, expected)


def test_specialized_norm3_matches_numpy_bit_for_bit() -> None:
    rng = np.random.default_rng(20260716)
    vectors = rng.standard_normal((1000, 3))

    for vector in vectors:
        expected = float(np.linalg.norm(vector))
        actual = _norm3(vector)
        np.testing.assert_array_equal(actual, expected)
