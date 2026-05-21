import unittest

import numpy as np

from sim.control.orbit import RMOEIfThenController, estimate_rmoes_from_rect_ric
from sim.core.models import StateBelief


def _belief(relative_curv_state: list[float]) -> StateBelief:
    state = np.zeros(12)
    state[0:6] = np.array(relative_curv_state, dtype=float)
    state[6:12] = np.array([7000.0, 0.0, 0.0, 0.0, 7.546053, 0.0])
    return StateBelief(state=state, covariance=np.eye(12), last_update_t_s=0.0)


class TestRMOEIfThenController(unittest.TestCase):
    def test_rmoe_estimator_reports_radial_center_and_drift(self):
        rmoes = estimate_rmoes_from_rect_ric(
            np.array([0.0, 0.0, 0.0, 0.0, -1.0e-3, 0.0]),
            mean_motion_rad_s=1.0e-3,
        )

        self.assertAlmostEqual(rmoes["radial_center_km"], 1.0)
        self.assertAlmostEqual(rmoes["in_track_drift_rate_km_s"], -3.0e-3)
        self.assertAlmostEqual(rmoes["cross_track_amplitude_km"], 0.0)

    def test_ahead_of_target_raises_radial_center_for_negative_drift(self):
        ctrl = RMOEIfThenController(
            mean_motion_rad_s=1.0e-3,
            max_accel_km_s2=1.0,
            max_drift_rate_m_s=0.02,
            gains={"drift": 1.0},
            target={"in_track_center_km": 0.0},
        )

        cmd = ctrl.act(_belief([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]), t_s=0.0, budget_ms=1.0)

        self.assertEqual(cmd.mode_flags["rmoe_mode"], "shape_drift")
        self.assertLess(cmd.mode_flags["desired_drift_rate_km_s"], 0.0)
        self.assertLess(cmd.mode_flags["accel_ric_km_s2"][1], 0.0)

    def test_behind_target_lowers_radial_center_for_positive_drift(self):
        ctrl = RMOEIfThenController(
            mean_motion_rad_s=1.0e-3,
            max_accel_km_s2=1.0,
            max_drift_rate_m_s=0.02,
            gains={"drift": 1.0},
            target={"in_track_center_km": 0.0},
        )

        cmd = ctrl.act(_belief([0.0, -1.0, 0.0, 0.0, 0.0, 0.0]), t_s=0.0, budget_ms=1.0)

        self.assertEqual(cmd.mode_flags["rmoe_mode"], "shape_drift")
        self.assertGreater(cmd.mode_flags["desired_drift_rate_km_s"], 0.0)
        self.assertGreater(cmd.mode_flags["accel_ric_km_s2"][1], 0.0)

    def test_close_zone_limits_existing_drift_before_other_trims(self):
        ctrl = RMOEIfThenController(
            mean_motion_rad_s=1.0e-3,
            max_accel_km_s2=1.0,
            max_drift_rate_m_s=0.01,
            close_zone_m=50.0,
            gains={"drift": 1.0},
            target={"in_track_center_km": 0.0, "cross_track_amplitude_km": 1.0},
        )

        cmd = ctrl.act(_belief([0.0, 0.01, 0.0, 0.0, 2.0e-5, 5.0e-4]), t_s=0.0, budget_ms=1.0)

        self.assertEqual(cmd.mode_flags["rmoe_mode"], "limit_drift")
        self.assertLess(cmd.mode_flags["accel_ric_km_s2"][1], 0.0)
        self.assertEqual(cmd.mode_flags["accel_ric_km_s2"][2], 0.0)

    def test_close_zone_tapers_drift_instead_of_stopping_behind_target(self):
        ctrl = RMOEIfThenController(
            mean_motion_rad_s=1.0e-3,
            max_accel_km_s2=1.0,
            max_drift_rate_m_s=0.05,
            close_zone_m=50.0,
            gains={"drift": 1.0},
            target={"in_track_center_km": 0.0},
        )

        cmd = ctrl.act(_belief([0.0, -0.02, 0.0, 0.0, 0.0, 0.0]), t_s=0.0, budget_ms=1.0)

        self.assertEqual(cmd.mode_flags["rmoe_mode"], "shape_drift")
        self.assertAlmostEqual(cmd.mode_flags["desired_drift_rate_km_s"], 2.0e-5)
        self.assertGreater(cmd.mode_flags["accel_ric_km_s2"][1], 0.0)

    def test_radial_center_trim_increases_in_track_velocity_when_target_is_higher(self):
        ctrl = RMOEIfThenController(
            mean_motion_rad_s=1.0e-3,
            max_accel_km_s2=1.0,
            gains={"radial_center": 1.0},
            target={"radial_center_km": 0.1},
            tolerances={"radial_center_km": 0.01, "in_track_drift_rate_km_s": 1.0},
        )

        cmd = ctrl.act(_belief([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), t_s=0.0, budget_ms=1.0)

        self.assertEqual(cmd.mode_flags["rmoe_mode"], "trim_radial_center")
        self.assertLess(cmd.mode_flags["accel_ric_km_s2"][1], 0.0)

    def test_cross_track_amplitude_burn_waits_for_c_zero_gate(self):
        ctrl = RMOEIfThenController(
            mean_motion_rad_s=1.0e-3,
            max_accel_km_s2=1.0,
            cross_track_burn_gate_m=50.0,
            gains={"cross_track_amplitude": 1.0},
            target={"cross_track_amplitude_km": 1.0},
        )

        away = ctrl.act(_belief([0.0, 0.0, 0.2, 0.0, 0.0, 5.0e-4]), t_s=0.0, budget_ms=1.0)
        at_gate = ctrl.act(_belief([0.0, 0.0, 0.0, 0.0, 0.0, 5.0e-4]), t_s=0.0, budget_ms=1.0)

        self.assertNotEqual(away.mode_flags["rmoe_mode"], "trim_cross_track_amplitude")
        self.assertEqual(at_gate.mode_flags["rmoe_mode"], "trim_cross_track_amplitude")
        self.assertGreater(at_gate.mode_flags["accel_ric_km_s2"][2], 0.0)

    def test_zero_max_accel_commands_zero_thrust(self):
        ctrl = RMOEIfThenController(mean_motion_rad_s=1.0e-3, max_accel_km_s2=0.0, target={})

        cmd = ctrl.act(_belief([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]), t_s=0.0, budget_ms=1.0)

        self.assertTrue(np.allclose(cmd.mode_flags["accel_ric_km_s2"], np.zeros(3)))
        self.assertEqual(cmd.mode_flags["limit_scale"], 0.0)


if __name__ == "__main__":
    unittest.main()
