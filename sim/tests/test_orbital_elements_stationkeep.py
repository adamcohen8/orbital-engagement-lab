import unittest

import numpy as np

from sim.control.orbit.baseline import (
    OrbitalElementsFeedbackController,
    SemiMajorAxisEccentricityController,
    StationkeepingController,
)
from sim.core.models import StateBelief, StateTruth
from sim.dynamics.orbit.elements import coe_to_rv_eci, orbital_element_feedback_accel, rv_to_coe_eci
from sim.mission.modules import (
    ControllerPointingExecution,
    OrbitalElementsStationKeepMissionStrategy,
    OrbitalElementsTrackingMissionStrategy,
)


def _truth_from_coes(**coes) -> StateTruth:
    r, v = coe_to_rv_eci(**coes)
    return StateTruth(
        position_eci_km=r,
        velocity_eci_km_s=v,
        attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        angular_rate_body_rad_s=np.zeros(3, dtype=float),
        mass_kg=250.0,
        t_s=0.0,
    )


class OrbitalElementsStationKeepTests(unittest.TestCase):
    def test_rv_to_coe_roundtrips_true_anomaly_for_elliptic_state(self) -> None:
        coes = {
            "a_km": 7050.0,
            "ecc": 0.01,
            "inc_deg": 42.0,
            "raan_deg": 11.0,
            "argp_deg": 25.0,
            "true_anomaly_deg": 33.0,
        }
        r, v = coe_to_rv_eci(**coes)

        out = rv_to_coe_eci(r, v)

        self.assertAlmostEqual(out.a_km, coes["a_km"], places=7)
        self.assertAlmostEqual(out.ecc, coes["ecc"], places=10)
        self.assertAlmostEqual(out.true_anomaly_deg, coes["true_anomaly_deg"], places=7)

    def test_orbital_elements_stationkeep_uses_current_phase_not_target_true_anomaly(self) -> None:
        truth = _truth_from_coes(
            a_km=7050.0,
            ecc=0.01,
            inc_deg=42.0,
            raan_deg=11.0,
            argp_deg=25.0,
            true_anomaly_deg=33.0,
        )
        strategy = OrbitalElementsStationKeepMissionStrategy(
            target_coes={
                "a_km": 7000.0,
                "ecc": 0.001,
                "inc_deg": 45.0,
                "raan_deg": 12.0,
                "argp_deg": 10.0,
                "true_anomaly_deg": 180.0,
            },
            max_accel_km_s2=2.0e-6,
        )

        out = strategy.update(truth=truth)

        expected_r, expected_v = coe_to_rv_eci(
            a_km=7000.0,
            ecc=0.001,
            inc_deg=45.0,
            raan_deg=12.0,
            argp_deg=10.0,
            true_anomaly_deg=33.0,
        )
        np.testing.assert_allclose(out["desired_state_eci_6"], np.hstack((expected_r, expected_v)), rtol=0.0, atol=1e-7)
        self.assertEqual(out["mission_mode"]["phase_mode"], "current_true_anomaly")
        self.assertLessEqual(float(np.linalg.norm(out["fallback_thrust_eci_km_s2"])), 2.0e-6 + 1e-15)

    def test_controller_pointing_execution_updates_stationkeeping_target(self) -> None:
        truth = _truth_from_coes(
            a_km=7050.0,
            ecc=0.01,
            inc_deg=42.0,
            raan_deg=11.0,
            argp_deg=25.0,
            true_anomaly_deg=33.0,
        )
        strategy = OrbitalElementsStationKeepMissionStrategy(
            target_coes={
                "a_km": 7000.0,
                "ecc": 0.001,
                "inc_deg": 45.0,
                "raan_deg": 12.0,
                "argp_deg": 10.0,
            },
            max_accel_km_s2=2.0e-6,
        )
        intent = strategy.update(truth=truth)
        controller = StationkeepingController(target_state=np.zeros(6), max_accel_km_s2=2.0e-6)
        execution = ControllerPointingExecution(require_attitude_alignment=False)
        belief = StateBelief(
            state=np.hstack(
                (
                    truth.position_eci_km,
                    truth.velocity_eci_km_s,
                    truth.attitude_quat_bn,
                    truth.angular_rate_body_rad_s,
                )
            ),
            covariance=np.eye(13),
            last_update_t_s=0.0,
        )

        out = execution.update(
            intent=intent,
            truth=truth,
            t_s=0.0,
            orbit_controller=controller,
            attitude_controller=None,
            orb_belief=belief,
            att_belief=None,
        )

        np.testing.assert_allclose(controller.target_state, intent["desired_state_eci_6"])
        self.assertGreater(float(np.linalg.norm(out["thrust_eci_km_s2"])), 0.0)

    def test_sma_ecc_controller_raises_lower_circular_orbit(self) -> None:
        truth = _truth_from_coes(
            a_km=6950.0,
            ecc=0.0,
            inc_deg=45.0,
            raan_deg=0.0,
            argp_deg=0.0,
            true_anomaly_deg=180.0,
        )
        belief = StateBelief(
            state=np.hstack((truth.position_eci_km, truth.velocity_eci_km_s)),
            covariance=np.eye(6),
            last_update_t_s=0.0,
        )
        controller = SemiMajorAxisEccentricityController(target_a_km=7000.0, target_ecc=0.0)

        cmd = controller.act(belief, 0.0, 2.0)

        c_ir = np.array(
            [
                truth.position_eci_km / np.linalg.norm(truth.position_eci_km),
                truth.velocity_eci_km_s / np.linalg.norm(truth.velocity_eci_km_s),
                np.cross(truth.position_eci_km, truth.velocity_eci_km_s)
                / np.linalg.norm(np.cross(truth.position_eci_km, truth.velocity_eci_km_s)),
            ]
        ).T
        accel_ric = c_ir.T @ cmd.thrust_eci_km_s2
        self.assertGreater(accel_ric[1], 0.0)
        self.assertAlmostEqual(float(cmd.mode_flags["a_error_km"]), 50.0, places=6)

    def test_sma_ecc_controller_zero_max_accel_zeroes_command(self) -> None:
        truth = _truth_from_coes(
            a_km=6950.0,
            ecc=0.0,
            inc_deg=45.0,
            raan_deg=0.0,
            argp_deg=0.0,
            true_anomaly_deg=180.0,
        )
        belief = StateBelief(
            state=np.hstack((truth.position_eci_km, truth.velocity_eci_km_s)),
            covariance=np.eye(6),
            last_update_t_s=0.0,
        )
        controller = SemiMajorAxisEccentricityController(
            target_a_km=7000.0,
            target_ecc=0.0,
            max_accel_km_s2=0.0,
        )

        cmd = controller.act(belief, 0.0, 2.0)

        np.testing.assert_allclose(cmd.thrust_eci_km_s2, np.zeros(3), rtol=0.0, atol=0.0)

    def test_orbital_element_feedback_handles_circular_target(self) -> None:
        r, v = coe_to_rv_eci(
            a_km=6950.0,
            ecc=0.0,
            inc_deg=44.0,
            raan_deg=0.0,
            argp_deg=0.0,
            true_anomaly_deg=0.0,
        )

        result = orbital_element_feedback_accel(
            np.hstack((r, v)),
            {
                "a_km": 7000.0,
                "ecc": 0.0,
                "inc_deg": 45.0,
                "raan_deg": 0.0,
                "argp_deg": 0.0,
            },
            max_accel_km_s2=5.0e-5,
        )

        self.assertTrue(np.all(np.isfinite(result.accel_eci_km_s2)))
        self.assertGreater(float(np.dot(result.accel_eci_km_s2, v)), 0.0)
        self.assertGreater(float(np.linalg.norm(result.hhat_error)), 0.0)
        self.assertLessEqual(float(np.linalg.norm(result.accel_eci_km_s2)), 5.0e-5 + 1e-15)

    def test_orbital_element_feedback_zero_max_accel_zeroes_command(self) -> None:
        r, v = coe_to_rv_eci(
            a_km=6950.0,
            ecc=0.0,
            inc_deg=44.0,
            raan_deg=0.0,
            argp_deg=0.0,
            true_anomaly_deg=0.0,
        )

        result = orbital_element_feedback_accel(
            np.hstack((r, v)),
            {
                "a_km": 7000.0,
                "ecc": 0.0,
                "inc_deg": 45.0,
                "raan_deg": 0.0,
                "argp_deg": 0.0,
            },
            max_accel_km_s2=0.0,
        )

        np.testing.assert_allclose(result.accel_eci_km_s2, np.zeros(3), rtol=0.0, atol=0.0)

    def test_orbital_element_feedback_rejects_unknown_controlled_element(self) -> None:
        r, v = coe_to_rv_eci(
            a_km=6950.0,
            ecc=0.0,
            inc_deg=44.0,
            raan_deg=0.0,
            argp_deg=0.0,
            true_anomaly_deg=0.0,
        )

        with self.assertRaisesRegex(ValueError, "Unsupported controlled_elements"):
            orbital_element_feedback_accel(
                np.hstack((r, v)),
                {"a_km": 7000.0},
                controlled_elements=("semiMajorAxis",),
            )

    def test_orbital_element_feedback_rejects_missing_target_fields(self) -> None:
        r, v = coe_to_rv_eci(
            a_km=6950.0,
            ecc=0.0,
            inc_deg=44.0,
            raan_deg=0.0,
            argp_deg=0.0,
            true_anomaly_deg=0.0,
        )

        with self.assertRaisesRegex(ValueError, "target_coes is missing required field"):
            orbital_element_feedback_accel(
                np.hstack((r, v)),
                {"a_km": 7000.0},
                controlled_elements=("a", "inc"),
            )

    def test_orbital_elements_tracking_strategy_emits_finite_fallback_thrust(self) -> None:
        truth = _truth_from_coes(
            a_km=6950.0,
            ecc=0.0,
            inc_deg=44.0,
            raan_deg=0.0,
            argp_deg=0.0,
            true_anomaly_deg=0.0,
        )
        strategy = OrbitalElementsTrackingMissionStrategy(
            target_coes={
                "a_km": 7000.0,
                "ecc": 0.0,
                "inc_deg": 45.0,
                "raan_deg": 0.0,
                "argp_deg": 0.0,
            },
            max_accel_km_s2=5.0e-5,
        )

        out = strategy.update(truth=truth)

        accel = np.array(out["fallback_thrust_eci_km_s2"], dtype=float)
        self.assertTrue(np.all(np.isfinite(accel)))
        self.assertGreater(float(np.linalg.norm(accel)), 0.0)
        self.assertEqual(out["mission_mode"]["strategy"], "orbital_elements_tracking")

    def test_orbital_elements_feedback_controller_emits_mode_flags(self) -> None:
        truth = _truth_from_coes(
            a_km=6950.0,
            ecc=0.0,
            inc_deg=44.0,
            raan_deg=0.0,
            argp_deg=0.0,
            true_anomaly_deg=0.0,
        )
        belief = StateBelief(
            state=np.hstack((truth.position_eci_km, truth.velocity_eci_km_s)),
            covariance=np.eye(6),
            last_update_t_s=0.0,
        )
        controller = OrbitalElementsFeedbackController(
            target_coes={
                "a_km": 7000.0,
                "ecc": 0.0,
                "inc_deg": 45.0,
                "raan_deg": 0.0,
                "argp_deg": 0.0,
            }
        )

        cmd = controller.act(belief, 0.0, 2.0)

        self.assertEqual(cmd.mode_flags["mode"], "orbital_elements_feedback")
        self.assertTrue(np.all(np.isfinite(cmd.thrust_eci_km_s2)))


if __name__ == "__main__":
    unittest.main()
