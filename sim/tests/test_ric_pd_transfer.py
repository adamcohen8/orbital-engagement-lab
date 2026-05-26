import numpy as np
import pytest

from sim.control.orbit.ric_pd import RICPDTransferController
from sim.core.models import StateBelief


def _belief(rel_ric_curv: np.ndarray) -> StateBelief:
    state = np.hstack(
        (
            np.array(rel_ric_curv, dtype=float).reshape(6),
            np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0], dtype=float),
        )
    )
    return StateBelief(state=state, covariance=np.eye(12), last_update_t_s=0.0)


def test_ric_pd_transfer_acquires_then_coasts_on_matched_velocity() -> None:
    ctrl = RICPDTransferController(
        max_accel_km_s2=6.0e-5,
        mean_motion_rad_s=0.001078,
        transfer_time_s=4800.0,
        velocity_deadband_m_s=0.03,
    )
    initial_rel = np.array([0.0, -10.0, 0.0, 0.0, 0.0, 0.001], dtype=float)

    burn_cmd = ctrl.act(_belief(initial_rel), t_s=0.0, budget_ms=1.0)

    assert burn_cmd.mode_flags["mode"] == "ric_pd_transfer"
    assert burn_cmd.mode_flags["phase"] == "guided_burn"
    assert np.linalg.norm(burn_cmd.mode_flags["velocity_error_ric_km_s"]) > 0.0

    matched_rel = initial_rel.copy()
    matched_rel[3:] = np.array(burn_cmd.mode_flags["target_velocity_ric_km_s"], dtype=float)
    coast_cmd = ctrl.act(_belief(matched_rel), t_s=1.0, budget_ms=1.0)

    assert coast_cmd.mode_flags["phase"] == "coast"
    np.testing.assert_allclose(coast_cmd.mode_flags["accel_ric_km_s2"], np.zeros(3), atol=1e-15)


def test_ric_pd_transfer_rejects_nonpositive_transfer_time() -> None:
    with pytest.raises(ValueError, match="transfer_time_s must be positive"):
        RICPDTransferController(
            max_accel_km_s2=1.0e-4,
            mean_motion_rad_s=0.001078,
            transfer_time_s=0.0,
        )
