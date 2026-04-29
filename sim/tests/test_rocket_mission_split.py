from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from sim.core.models import StateBelief, StateTruth
from sim.mission.modules import (
    RocketGoNowExecution,
    RocketGoWhenPossibleExecution,
    RocketPredefinedOrbitMissionStrategy,
    RocketPursuitMissionStrategy,
    RocketWaitOptimalExecution,
)


def _truth() -> StateTruth:
    return StateTruth(
        position_eci_km=np.array([7000.0, 0.0, 0.0], dtype=float),
        velocity_eci_km_s=np.array([0.0, 7.5, 0.0], dtype=float),
        attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        angular_rate_body_rad_s=np.zeros(3, dtype=float),
        mass_kg=500.0,
        t_s=0.0,
    )


def _belief_with_velocity_y(vy_km_s: float) -> StateBelief:
    truth = _truth()
    return StateBelief(
        state=np.array([*truth.position_eci_km, 0.0, float(vy_km_s), 0.0], dtype=float),
        covariance=np.eye(6),
        last_update_t_s=truth.t_s,
    )


class TestRocketMissionSplit(unittest.TestCase):
    def test_rocket_pursuit_strategy_reports_goal(self):
        out = RocketPursuitMissionStrategy(target_id="target").update(truth=_truth())
        self.assertEqual(out["orbital_goal"], "pursuit")
        self.assertEqual(out["target_id"], "target")
        self.assertEqual(out["mission_mode"]["orbital_goal"], "pursuit")

    def test_rocket_predefined_strategy_reports_goal(self):
        out = RocketPredefinedOrbitMissionStrategy(predef_target_alt_km=500.0, predef_target_ecc=0.01).update(truth=_truth())
        self.assertEqual(out["orbital_goal"], "predefined_orbit")
        self.assertEqual(out["predefined_orbit_goal"]["target_alt_km"], 500.0)
        self.assertEqual(out["predefined_orbit_goal"]["target_ecc"], 0.01)

    def test_rocket_launch_executions_gate_authorization(self):
        base_intent = {"mission_mode": {"orbital_goal": "pursuit"}, "target_id": "target"}
        self.assertTrue(RocketGoNowExecution().update(intent=base_intent)["launch_authorized"])

        wait_exec = RocketWaitOptimalExecution(window_period_s=100.0, window_open_duration_s=10.0)
        self.assertTrue(wait_exec.update(intent=base_intent, t_s=5.0)["launch_authorized"])
        self.assertFalse(wait_exec.update(intent=base_intent, t_s=20.0)["launch_authorized"])

        with patch("sim.mission.modules._estimate_stack_delta_v_m_s", return_value=150.0):
            go_exec = RocketGoWhenPossibleExecution(go_when_possible_margin_m_s=25.0)
            self.assertTrue(
                go_exec.update(
                    intent=base_intent,
                    truth=_truth(),
                    own_knowledge={"target": _belief_with_velocity_y(7.6)},
                    world_truth={},
                    rocket_state=object(),
                    rocket_vehicle_cfg=object(),
                )["launch_authorized"]
            )

        with patch("sim.mission.modules._estimate_stack_delta_v_m_s", return_value=150.0):
            go_exec = RocketGoWhenPossibleExecution(go_when_possible_margin_m_s=25.0)
            self.assertFalse(
                go_exec.update(
                    intent=base_intent,
                    truth=_truth(),
                    own_knowledge={"target": _belief_with_velocity_y(7.63)},
                    world_truth={},
                    rocket_state=object(),
                    rocket_vehicle_cfg=object(),
                )["launch_authorized"]
            )


if __name__ == "__main__":
    unittest.main()
