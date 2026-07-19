from __future__ import annotations

import numpy as np
import pytest

from sim.utils.quaternion import normalize_quaternion, quaternion_to_dcm_bn


def _reference_quaternion_to_dcm_bn(q_bn: np.ndarray) -> np.ndarray:
    q0, q1, q2, q3 = normalize_quaternion(q_bn)
    return np.array(
        [
            [1.0 - 2.0 * (q2**2 + q3**2), 2.0 * (q1 * q2 + q0 * q3), 2.0 * (q1 * q3 - q0 * q2)],
            [2.0 * (q1 * q2 - q0 * q3), 1.0 - 2.0 * (q1**2 + q3**2), 2.0 * (q2 * q3 + q0 * q1)],
            [2.0 * (q1 * q3 + q0 * q2), 2.0 * (q2 * q3 - q0 * q1), 1.0 - 2.0 * (q1**2 + q2**2)],
        ]
    )


@pytest.mark.parametrize(
    "quaternion",
    (
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.9961947018177665, 0.0, 0.08715570015903389, 0.0]),
        np.array([2.0, -3.0, 4.0, -5.0]),
        np.zeros(4),
        np.array([np.nan, 0.0, 0.0, 0.0]),
        np.array([1.0, 2.0, 3.0]),
    ),
)
def test_quaternion_to_dcm_preserves_reference_output_bits(quaternion: np.ndarray) -> None:
    expected = _reference_quaternion_to_dcm_bn(quaternion)
    actual = quaternion_to_dcm_bn(quaternion)

    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
