import numpy as np
import pytest

from sim.control.orbit.ric_pd import RICPDTransferController
from sim.core.models import StateBelief
from sim.utils.frames import ric_curv_to_rect


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


def test_controller_adapter_and_subordinate_guidance_api_share_the_same_law() -> None:
    controller = RICPDTransferController(
        max_accel_km_s2=6.0e-5,
        mean_motion_rad_s=0.001078,
        transfer_time_s=4800.0,
    )
    guidance = RICPDTransferController(
        max_accel_km_s2=6.0e-5,
        mean_motion_rad_s=0.001078,
        transfer_time_s=4800.0,
    )
    relative_curv = np.array([0.0, -10.0, 0.0, 0.0, 0.0, 0.001], dtype=float)
    relative_rect = ric_curv_to_rect(relative_curv, r0_km=7000.0)
    chief_position = np.array([7000.0, 0.0, 0.0])
    chief_velocity = np.array([0.0, 7.5, 0.0])

    command = controller.act(_belief(relative_curv), t_s=0.0, budget_ms=1.0)
    result = guidance.guide_relative_state(
        relative_rect,
        chief_position,
        chief_velocity,
        t_s=0.0,
    )

    assert result.mode_flags["phase"] == command.mode_flags["phase"]
    np.testing.assert_allclose(result.acceleration_eci_km_s2, command.thrust_eci_km_s2)


def test_ric_pd_transfer_rejects_nonpositive_transfer_time() -> None:
    with pytest.raises(ValueError, match="transfer_time_s must be positive"):
        RICPDTransferController(
            max_accel_km_s2=1.0e-4,
            mean_motion_rad_s=0.001078,
            transfer_time_s=0.0,
        )


def test_ric_pd_transfer_exposes_correction_final_brake_and_terminal_cleanup_phases() -> None:
    ctrl = RICPDTransferController(
        max_accel_km_s2=6.0e-5,
        mean_motion_rad_s=0.001078,
        transfer_time_s=100.0,
        correction_interval_s=10.0,
        final_brake_start_s=20.0,
        terminal_start_s=0.0,
        terminal_range_km=0.0,
    )
    initial_rel = np.array([0.0, -10.0, 0.0, 0.0, 0.0, 0.001], dtype=float)
    acquisition = ctrl.act(_belief(initial_rel), t_s=0.0, budget_ms=1.0)
    target_velocity = np.asarray(acquisition.mode_flags["target_velocity_ric_km_s"], dtype=float)
    matched_rel = initial_rel.copy()
    matched_rel[3:] = target_velocity

    correction = ctrl.act(_belief(matched_rel), t_s=10.0, budget_ms=1.0)
    final_brake = ctrl.act(_belief(matched_rel), t_s=85.0, budget_ms=1.0)
    terminal = ctrl.act(_belief(matched_rel), t_s=100.0, budget_ms=1.0)

    assert correction.mode_flags["phase"] == "guided_burn"
    assert final_brake.mode_flags["phase"] == "final_brake"
    assert terminal.mode_flags["phase"] == "terminal_cleanup"


def test_ric_pd_transfer_snapshot_restores_guidance_phase_state() -> None:
    ctrl = RICPDTransferController(
        max_accel_km_s2=6.0e-5,
        mean_motion_rad_s=0.001078,
        transfer_time_s=4800.0,
    )
    initial_rel = np.array([0.0, -10.0, 0.0, 0.0, 0.0, 0.001], dtype=float)
    ctrl.act(_belief(initial_rel), t_s=100.0, budget_ms=1.0)
    snapshot = ctrl.snapshot_state()
    restored = RICPDTransferController(
        max_accel_km_s2=6.0e-5,
        mean_motion_rad_s=0.001078,
        transfer_time_s=4800.0,
    )
    restored.restore_state(snapshot)

    expected = ctrl.act(_belief(initial_rel), t_s=200.0, budget_ms=1.0)
    actual = restored.act(_belief(initial_rel), t_s=200.0, budget_ms=1.0)

    assert actual.mode_flags["phase"] == expected.mode_flags["phase"]
    np.testing.assert_allclose(actual.thrust_eci_km_s2, expected.thrust_eci_km_s2)
