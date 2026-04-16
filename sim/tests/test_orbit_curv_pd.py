import unittest

import numpy as np

from sim.control.orbit import CurvilinearRICPDController, curv_accel_to_rect
from sim.core.models import StateBelief


class TestCurvilinearRICPDController(unittest.TestCase):
    def test_origin_accel_maps_to_rect_and_eci(self):
        ctrl = CurvilinearRICPDController(
            max_accel_km_s2=1.0,
            kp=[1.0e-6, 2.0e-6, 3.0e-6],
            kd=[1.0e-3, 2.0e-3, 3.0e-3],
        )
        state = np.zeros(12)
        state[0:6] = np.array([1.0, -2.0, 0.5, 1.0e-3, -2.0e-3, 3.0e-3])
        state[6:12] = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
        belief = StateBelief(state=state, covariance=np.eye(12), last_update_t_s=0.0)

        cmd = ctrl.act(belief, t_s=0.0, budget_ms=1.0)

        expected_curv = -(
            np.diag([1.0e-6, 2.0e-6, 3.0e-6]) @ state[0:3]
            + np.diag([1.0e-3, 2.0e-3, 3.0e-3]) @ state[3:6]
        )
        expected_rect = curv_accel_to_rect(expected_curv, position_curv_km=state[0:3], r0_km=7000.0)
        self.assertTrue(np.allclose(cmd.mode_flags["accel_curv_ric_km_s2"], expected_curv, atol=1e-15))
        self.assertTrue(np.allclose(cmd.mode_flags["accel_rect_ric_km_s2"], expected_rect, atol=1e-15))
        self.assertTrue(np.allclose(cmd.thrust_eci_km_s2, expected_rect, atol=1e-12))
        self.assertEqual(cmd.mode_flags["mode"], "curvilinear_ric_pd")

    def test_accel_limit_scales_curv_and_rect_commands(self):
        ctrl = CurvilinearRICPDController(max_accel_km_s2=2.0e-6, kp=[1.0e-5, 0.0, 0.0], kd=0.0)
        state = np.zeros(12)
        state[0:6] = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        state[6:12] = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
        belief = StateBelief(state=state, covariance=np.eye(12), last_update_t_s=0.0)

        cmd = ctrl.act(belief, t_s=0.0, budget_ms=1.0)

        self.assertAlmostEqual(np.linalg.norm(cmd.mode_flags["accel_rect_ric_km_s2"]), 2.0e-6)
        self.assertAlmostEqual(np.linalg.norm(cmd.thrust_eci_km_s2), 2.0e-6)

    def test_zero_max_accel_commands_zero_thrust(self):
        ctrl = CurvilinearRICPDController(max_accel_km_s2=0.0, kp=[1.0e-5, 0.0, 0.0], kd=0.0)
        state = np.zeros(12)
        state[0:6] = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        state[6:12] = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
        belief = StateBelief(state=state, covariance=np.eye(12), last_update_t_s=0.0)

        cmd = ctrl.act(belief, t_s=0.0, budget_ms=1.0)

        self.assertTrue(np.allclose(cmd.mode_flags["accel_curv_ric_km_s2"], np.zeros(3)))
        self.assertTrue(np.allclose(cmd.mode_flags["accel_rect_ric_km_s2"], np.zeros(3)))
        self.assertTrue(np.allclose(cmd.thrust_eci_km_s2, np.zeros(3)))
        self.assertEqual(cmd.mode_flags["linear_feedback_debug"]["limit_scale"], 0.0)

    def test_negative_max_accel_is_rejected(self):
        with self.assertRaises(ValueError):
            CurvilinearRICPDController(max_accel_km_s2=-1.0)


if __name__ == "__main__":
    unittest.main()
