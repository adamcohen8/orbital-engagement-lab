from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from sim.core.models import Command, StateBelief, StateTruth

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_runtime_uses_numpy_2_approved_by_the_python_minor_constraint() -> None:
    assert int(np.__version__.split(".", 1)[0]) == 2

    constraint = REPO_ROOT / "constraints" / f"py{sys.version_info.major}{sys.version_info.minor}.txt"
    numpy_line = next(
        line.strip()
        for line in constraint.read_text(encoding="utf-8").splitlines()
        if line.lower().startswith("numpy==")
    )
    assert numpy_line.startswith("numpy==2.")


def test_public_physics_structures_preserve_explicit_float64_precision() -> None:
    truth = StateTruth(
        position_eci_km=np.asarray([7000, 0, 0], dtype=np.float64),
        velocity_eci_km_s=np.asarray([0, 7.5, 0], dtype=np.float64),
        attitude_quat_bn=np.asarray([1, 0, 0, 0], dtype=np.float64),
        angular_rate_body_rad_s=np.zeros(3, dtype=np.float64),
        mass_kg=100.0,
        t_s=0.0,
    )
    belief = StateBelief(
        state=np.concatenate((truth.position_eci_km, truth.velocity_eci_km_s)),
        covariance=np.eye(6, dtype=np.float64),
        last_update_t_s=0.0,
    )

    assert truth.copy().position_eci_km.dtype == np.dtype(np.float64)
    assert belief.state.dtype == np.dtype(np.float64)
    assert Command.zero().thrust_eci_km_s2.dtype == np.dtype(np.float64)
    assert Command.zero().torque_body_nm.dtype == np.dtype(np.float64)


def test_nep50_scalar_promotion_is_explicit_at_float64_physics_boundary() -> None:
    state = np.asarray([7000.0, 0.0, 0.0], dtype=np.float64)
    delta = np.float32(0.125)
    updated = state + np.float64(delta)

    assert updated.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(updated, np.asarray([7000.125, 0.125, 0.125], dtype=np.float64))


def test_default_integer_width_is_platform_independent_on_supported_64_bit_hosts() -> None:
    if sys.maxsize <= 2**32:
        pytest.skip("OEL's declared desktop compatibility matrix is 64-bit.")

    assert np.dtype(int) == np.dtype(np.intp)
    assert np.dtype(np.intp).itemsize == 8
