import unittest

import numpy as np

from sim.core.models import StateBelief, StateTruth
from sim.mission.modules import MissionExecutiveStrategy, PursuitMissionStrategy


def _truth(position_eci_km: list[float], mass_kg: float = 120.0, t_s: float = 0.0) -> StateTruth:
    return StateTruth(
        position_eci_km=np.array(position_eci_km, dtype=float),
        velocity_eci_km_s=np.array([0.0, 7.5, 0.0], dtype=float),
        attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        angular_rate_body_rad_s=np.zeros(3, dtype=float),
        mass_kg=float(mass_kg),
        t_s=float(t_s),
    )


def _mode(name: str) -> dict:
    return {
        "name": name,
        "mission_strategy": {
            "module": "sim.mission.modules",
            "class_name": "SafeHoldMissionStrategy",
            "params": {"attitude_mode": "hold_current"},
        },
        "mission_execution": {
            "module": "sim.mission.modules",
            "class_name": "SafeHoldExecution",
            "params": {},
        },
    }


def _belief(position_eci_km: list[float]) -> StateBelief:
    truth = _truth(position_eci_km)
    return StateBelief(
        state=np.hstack((truth.position_eci_km, truth.velocity_eci_km_s)),
        covariance=np.eye(6),
        last_update_t_s=truth.t_s,
    )


class MissionExecutiveTests(unittest.TestCase):
    def test_world_truth_is_not_targeting_fallback(self) -> None:
        strategy = PursuitMissionStrategy(target_id="target", use_knowledge_for_targeting=False, max_accel_km_s2=1e-6)

        out = strategy.update(
            truth=_truth([7000.0, 0.0, 0.0]),
            own_knowledge={},
            world_truth={"target": _truth([7001.0, 0.0, 0.0])},
        )

        self.assertTrue(np.allclose(np.array(out["fallback_thrust_eci_km_s2"], dtype=float), np.array([1e-6, 0.0, 0.0])))

    def test_range_transition_obeys_min_mode_duration(self) -> None:
        executive = MissionExecutiveStrategy(
            initial_mode="hold",
            modes=[_mode("hold"), _mode("defend")],
            transitions=[
                {
                    "from_mode": "hold",
                    "to_mode": "defend",
                    "trigger": "range_lt",
                    "target_id": "target",
                    "use_knowledge_for_targeting": True,
                    "threshold_km": 10.0,
                    "min_mode_duration_s": 5.0,
                }
            ],
        )
        own_knowledge = {"target": _belief([7008.0, 0.0, 0.0])}

        out0 = executive.update(truth=_truth([7000.0, 0.0, 0.0], t_s=0.0), own_knowledge=own_knowledge, world_truth={}, t_s=0.0)
        self.assertEqual(out0["mission_mode"]["executive_mode"], "hold")

        out1 = executive.update(truth=_truth([7000.0, 0.0, 0.0], t_s=3.0), own_knowledge=own_knowledge, world_truth={}, t_s=3.0)
        self.assertEqual(out1["mission_mode"]["executive_mode"], "hold")

        out2 = executive.update(truth=_truth([7000.0, 0.0, 0.0], t_s=5.0), own_knowledge=own_knowledge, world_truth={}, t_s=5.0)
        self.assertEqual(out2["mission_mode"]["executive_mode"], "defend")

    def test_range_hysteresis_requires_reset_threshold_to_rearm(self) -> None:
        executive = MissionExecutiveStrategy(
            initial_mode="hold",
            modes=[_mode("hold"), _mode("defend")],
            transitions=[
                {
                    "from_mode": "hold",
                    "to_mode": "defend",
                    "trigger": "range_lt",
                    "target_id": "target",
                    "use_knowledge_for_targeting": True,
                    "threshold_km": 10.0,
                    "reset_threshold_km": 15.0,
                },
                {
                    "from_mode": "defend",
                    "to_mode": "hold",
                    "trigger": "range_gt",
                    "target_id": "target",
                    "use_knowledge_for_targeting": True,
                    "threshold_km": 15.0,
                    "reset_threshold_km": 10.0,
                },
            ],
        )

        close_knowledge = {"target": _belief([7008.0, 0.0, 0.0])}
        mid_knowledge = {"target": _belief([7012.0, 0.0, 0.0])}
        far_knowledge = {"target": _belief([7016.0, 0.0, 0.0])}

        out0 = executive.update(truth=_truth([7000.0, 0.0, 0.0], t_s=0.0), own_knowledge=close_knowledge, world_truth={}, t_s=0.0)
        self.assertEqual(out0["mission_mode"]["executive_mode"], "defend")

        out1 = executive.update(truth=_truth([7000.0, 0.0, 0.0], t_s=1.0), own_knowledge=mid_knowledge, world_truth={}, t_s=1.0)
        self.assertEqual(out1["mission_mode"]["executive_mode"], "defend")

        out2 = executive.update(truth=_truth([7000.0, 0.0, 0.0], t_s=2.0), own_knowledge=far_knowledge, world_truth={}, t_s=2.0)
        self.assertEqual(out2["mission_mode"]["executive_mode"], "hold")

        out3 = executive.update(truth=_truth([7000.0, 0.0, 0.0], t_s=3.0), own_knowledge=mid_knowledge, world_truth={}, t_s=3.0)
        self.assertEqual(out3["mission_mode"]["executive_mode"], "hold")

        out4 = executive.update(truth=_truth([7000.0, 0.0, 0.0], t_s=4.0), own_knowledge=far_knowledge, world_truth={}, t_s=4.0)
        self.assertEqual(out4["mission_mode"]["executive_mode"], "hold")

        out5 = executive.update(truth=_truth([7000.0, 0.0, 0.0], t_s=5.0), own_knowledge=close_knowledge, world_truth={}, t_s=5.0)
        self.assertEqual(out5["mission_mode"]["executive_mode"], "defend")

    def test_fuel_hysteresis_uses_reset_threshold_fraction(self) -> None:
        executive = MissionExecutiveStrategy(
            initial_mode="nominal",
            modes=[_mode("nominal"), _mode("safe_hold")],
            transitions=[
                {
                    "from_mode": "nominal",
                    "to_mode": "safe_hold",
                    "trigger": "fuel_below_fraction",
                    "threshold": 0.2,
                    "reset_threshold_fraction": 0.4,
                },
                {
                    "from_mode": "safe_hold",
                    "to_mode": "nominal",
                    "trigger": "fuel_below_fraction",
                    "threshold": -1.0,
                    "reset_threshold_fraction": 0.2,
                },
            ],
        )

        out0 = executive.update(
            truth=_truth([7000.0, 0.0, 0.0], mass_kg=103.0, t_s=0.0),
            own_knowledge={},
            world_truth={},
            dry_mass_kg=100.0,
            fuel_capacity_kg=20.0,
            t_s=0.0,
        )
        self.assertEqual(out0["mission_mode"]["executive_mode"], "safe_hold")

        out1 = executive.update(
            truth=_truth([7000.0, 0.0, 0.0], mass_kg=126.0, t_s=1.0),
            own_knowledge={},
            world_truth={},
            dry_mass_kg=100.0,
            fuel_capacity_kg=20.0,
            t_s=1.0,
        )
        self.assertEqual(out1["mission_mode"]["executive_mode"], "safe_hold")

        out2 = executive.update(
            truth=_truth([7000.0, 0.0, 0.0], mass_kg=130.0, t_s=2.0),
            own_knowledge={},
            world_truth={},
            dry_mass_kg=100.0,
            fuel_capacity_kg=20.0,
            t_s=2.0,
        )
        self.assertEqual(out2["mission_mode"]["executive_mode"], "safe_hold")


if __name__ == "__main__":
    unittest.main()
