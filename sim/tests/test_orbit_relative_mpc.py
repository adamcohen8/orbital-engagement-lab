import unittest

import numpy as np

from sim.control.orbit import RelativeOrbitMPCController
from sim.core.models import StateBelief
from sim.dynamics.orbit.two_body import propagate_two_body_rk4
from sim.utils.frames import (
    eci_relative_to_ric_rect,
    ric_angular_rate_eci_from_rv,
    ric_dcm_ir_from_rv,
    ric_rect_state_to_eci,
    ric_rect_to_curv,
)


class TestRelativeOrbitMPCController(unittest.TestCase):
    def test_ric_frame_rejects_zero_angular_momentum(self):
        with self.assertRaisesRegex(ValueError, "zero angular momentum"):
            ric_dcm_ir_from_rv(np.array([7000.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))

    def test_ric_rect_inertial_round_trip_preserves_velocity(self):
        x_target = np.array([7000.0, 0.0, 0.0, 0.0, 7.546049108166282, 0.0], dtype=float)
        x_rel_rect = np.array([1.2, -0.8, 0.2, -0.002, 0.0015, 0.0007], dtype=float)
        x_chaser = ric_rect_state_to_eci(x_rel_rect, x_target[:3], x_target[3:])
        x_rel_back = eci_relative_to_ric_rect(x_chaser, x_target)
        self.assertTrue(np.allclose(x_rel_back, x_rel_rect, atol=1e-12))

    def test_acceleration_aware_ric_rate_includes_cross_track_plane_precession(self):
        r = np.array([7000.0, 0.0, 0.0], dtype=float)
        v = np.array([0.0, 7.5, 0.0], dtype=float)
        acceleration = np.array([0.0, 0.0, 1.0e-6], dtype=float)
        expected = np.array([1.0e-6 / 7.5, 0.0, 7.5 / 7000.0], dtype=float)

        omega = ric_angular_rate_eci_from_rv(r, v, chief_accel_eci_km_s2=acceleration)

        np.testing.assert_allclose(omega, expected, rtol=0.0, atol=1.0e-15)

    def test_acceleration_aware_ric_state_transforms_round_trip(self):
        chief = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0], dtype=float)
        acceleration = np.array([0.0, 0.0, 1.0e-6], dtype=float)
        relative = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=float)

        deputy = ric_rect_state_to_eci(
            relative,
            chief[:3],
            chief[3:],
            chief_accel_eci_km_s2=acceleration,
        )
        recovered = eci_relative_to_ric_rect(
            deputy,
            chief,
            chief_accel_eci_km_s2=acceleration,
        )

        expected_velocity_term = -(1.0e-6 / 7.5)
        np.testing.assert_allclose(deputy[4], chief[4] + expected_velocity_term, atol=1.0e-15)
        np.testing.assert_allclose(recovered, relative, rtol=0.0, atol=1.0e-12)

    def test_internal_relative_state_matches_shared_ric_transform(self):
        x_target = np.array([7000.0, 10.0, -5.0, -0.01, 7.546049108166282, 0.02], dtype=float)
        x_rel_rect = np.array([1.2, -0.8, 0.2, -0.002, 0.0015, 0.0007], dtype=float)
        x_chaser = ric_rect_state_to_eci(x_rel_rect, x_target[:3], x_target[3:])

        x_rel_internal = RelativeOrbitMPCController._relative_rect_ric(x_chaser=x_chaser, x_target=x_target)

        self.assertTrue(np.allclose(x_rel_internal, x_rel_rect, atol=1e-12))

    def test_zero_relative_state_commands_near_zero(self):
        ctrl = RelativeOrbitMPCController(
            max_accel_km_s2=2e-5,
            horizon_steps=4,
            step_dt_s=10.0,
            max_iterations=1,
        )
        state = np.zeros(12, dtype=float)
        state[6:12] = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0], dtype=float)
        belief = StateBelief(state=state, covariance=np.eye(12), last_update_t_s=0.0)
        cmd = ctrl.act(belief, t_s=0.0, budget_ms=5.0)
        self.assertEqual(cmd.mode_flags["mode"], "relative_orbit_mpc")
        self.assertTrue(np.allclose(cmd.thrust_eci_km_s2, np.zeros(3), atol=1e-10))

    def test_saturates_to_max_accel(self):
        ctrl = RelativeOrbitMPCController(
            max_accel_km_s2=1e-6,
            horizon_steps=4,
            step_dt_s=10.0,
            max_iterations=1,
        )
        state = np.zeros(12, dtype=float)
        state[0:6] = np.array([1.2, -0.5, 0.4, 0.01, -0.015, 0.005], dtype=float)
        state[6:12] = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0], dtype=float)
        belief = StateBelief(state=state, covariance=np.eye(12), last_update_t_s=0.0)
        cmd = ctrl.act(belief, t_s=0.0, budget_ms=5.0)
        self.assertLessEqual(np.linalg.norm(cmd.thrust_eci_km_s2), 1e-6 + 1e-12)

    def test_nonzero_relative_state_generates_nonzero_thrust_with_tight_budget(self):
        ctrl = RelativeOrbitMPCController(
            max_accel_km_s2=5e-5,
            horizon_steps=6,
            step_dt_s=10.0,
            max_iterations=1,
        )
        state = np.zeros(12, dtype=float)
        state[0:6] = np.array([8.0, -3.0, 2.0, 0.0010, -0.0006, 0.0004], dtype=float)
        state[6:12] = np.array([7000.0, 0.0, 0.0, 0.0, 7.546049108166282, 0.0], dtype=float)
        belief = StateBelief(state=state, covariance=np.eye(12), last_update_t_s=0.0)
        cmd = ctrl.act(belief, t_s=0.0, budget_ms=1e-6)
        self.assertGreater(np.linalg.norm(cmd.thrust_eci_km_s2), 1e-10)

    def test_closed_loop_reduces_relative_distance(self):
        ctrl = RelativeOrbitMPCController(
            max_accel_km_s2=3e-5,
            horizon_steps=5,
            step_dt_s=10.0,
            max_iterations=1,
        )

        x_target = np.array([7000.0, 0.0, 0.0, 0.0, 7.546049108166282, 0.0], dtype=float)
        x_chaser = np.array([7001.2, -0.8, 0.2, -0.002, 7.5449, 0.0007], dtype=float)
        dt_s = 10.0

        d0 = None
        d_final = None
        for k in range(30):
            x_rel_rect = eci_relative_to_ric_rect(x_chaser, x_target)
            x_rel_curv = ric_rect_to_curv(x_rel_rect, r0_km=float(np.linalg.norm(x_target[:3])))
            belief = StateBelief(
                state=np.hstack((x_rel_curv, x_target)),
                covariance=np.eye(12),
                last_update_t_s=k * dt_s,
            )
            cmd = ctrl.act(belief, t_s=k * dt_s, budget_ms=5.0)
            x_target = propagate_two_body_rk4(
                x_eci=x_target,
                dt_s=dt_s,
                mu_km3_s2=398600.4418,
                accel_cmd_eci_km_s2=np.zeros(3, dtype=float),
            )
            x_chaser = propagate_two_body_rk4(
                x_eci=x_chaser,
                dt_s=dt_s,
                mu_km3_s2=398600.4418,
                accel_cmd_eci_km_s2=cmd.thrust_eci_km_s2,
            )
            d = float(np.linalg.norm(x_chaser[:3] - x_target[:3]))
            if d0 is None:
                d0 = d
            d_final = d

        self.assertIsNotNone(d0)
        self.assertIsNotNone(d_final)
        self.assertLess(d_final, d0)


if __name__ == "__main__":
    unittest.main()
