import unittest

import numpy as np

from sim.core.models import StateBelief
from sim.presets.rockets import BASIC_TWO_STAGE_STACK
from sim.rocket import (
    GuidanceCommand,
    HoldAttitudeGuidance,
    MaxQThrottleLimiterGuidance,
    RocketSimConfig,
    RocketState,
    RocketVehicleConfig,
    TVCSteeringGuidance,
)
from sim.runtime_support import AgentRuntime, _rocket_state_to_truth
from sim.single_run_support import _RocketStepper


class TestRocketGuidance(unittest.TestCase):
    def _vehicle(self) -> RocketVehicleConfig:
        return RocketVehicleConfig(stack=BASIC_TWO_STAGE_STACK, payload_mass_kg=0.0)

    def _state(self, r_eci_km: np.ndarray, v_eci_km_s: np.ndarray) -> RocketState:
        return RocketState(
            t_s=0.0,
            position_eci_km=np.array(r_eci_km, dtype=float),
            velocity_eci_km_s=np.array(v_eci_km_s, dtype=float),
            attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            angular_rate_body_rad_s=np.zeros(3),
            mass_kg=1000.0,
            active_stage_index=0,
            stage_prop_remaining_kg=np.array([100.0, 0.0]),
            payload_attached=True,
        )

    def test_maxq_limiter_reduces_throttle_when_q_is_high(self):
        sim_cfg = RocketSimConfig(max_time_s=1.0, dt_s=1.0, atmosphere_model="ussa1976")
        base = HoldAttitudeGuidance(throttle=1.0)
        lim = MaxQThrottleLimiterGuidance(base_guidance=base, max_q_pa=45_000.0, min_throttle=0.0)
        r = np.array([6378.137, 0.0, 0.0], dtype=float)
        v = np.array([0.0, 7.8, 0.0], dtype=float)
        s = self._state(r, v)
        cmd = lim.command(s, sim_cfg, self._vehicle())
        self.assertLess(cmd.throttle, 1.0)
        self.assertGreaterEqual(cmd.throttle, 0.0)

    def test_maxq_limiter_leaves_throttle_when_q_is_low(self):
        sim_cfg = RocketSimConfig(max_time_s=1.0, dt_s=1.0, atmosphere_model="ussa1976")
        base = HoldAttitudeGuidance(throttle=0.7)
        lim = MaxQThrottleLimiterGuidance(base_guidance=base, max_q_pa=45_000.0, min_throttle=0.0)
        r = np.array([6378.137 + 400.0, 0.0, 0.0], dtype=float)
        omega = np.array([0.0, 0.0, 7.2921159e-5], dtype=float)
        v_atm = np.cross(omega, r)
        s = self._state(r, v_atm)
        cmd = lim.command(s, sim_cfg, self._vehicle())
        self.assertAlmostEqual(cmd.throttle, 0.7, places=8)

    def test_maxq_limiter_accounts_for_wind(self):
        base = HoldAttitudeGuidance(throttle=0.7)
        lim = MaxQThrottleLimiterGuidance(base_guidance=base, max_q_pa=100.0, min_throttle=0.0)
        sim_cfg = RocketSimConfig(
            max_time_s=1.0,
            dt_s=1.0,
            atmosphere_model="ussa1976",
            wind_enu_m_s=np.array([80.0, 0.0, 0.0]),
        )
        r = np.array([6378.137, 0.0, 0.0], dtype=float)
        omega = np.array([0.0, 0.0, 7.2921159e-5], dtype=float)
        v = np.cross(omega, r)
        s = self._state(r, v)
        cmd = lim.command(s, sim_cfg, self._vehicle())
        self.assertLess(cmd.throttle, 0.7)

    def test_tvc_wrapper_converts_attitude_target_to_body_vector_command(self):
        class _Base:
            def command(self, state, sim_cfg, vehicle_cfg):
                return GuidanceCommand(
                    throttle=0.6,
                    attitude_quat_bn_cmd=np.array([np.cos(np.pi / 8), 0.0, np.sin(np.pi / 8), 0.0]),
                    torque_body_nm_cmd=np.zeros(3),
                )

        sim_cfg = RocketSimConfig(max_time_s=1.0, dt_s=1.0)
        s = self._state(np.array([6378.137, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))
        cmd = TVCSteeringGuidance(base_guidance=_Base()).command(s, sim_cfg, self._vehicle())
        self.assertIsNone(cmd.attitude_quat_bn_cmd)
        self.assertIsNotNone(cmd.thrust_vector_body_cmd)
        self.assertGreater(np.linalg.norm(cmd.thrust_vector_body_cmd[1:]), 1e-6)

    def test_rocket_stepper_preserves_tvc_vector_when_overriding_throttle(self):
        class _VectorGuidance:
            def command(self, state, sim_cfg, vehicle_cfg):
                return GuidanceCommand(
                    throttle=1.0,
                    torque_body_nm_cmd=np.zeros(3),
                    thrust_vector_body_cmd=np.array([1.0, 1.0, 0.0]),
                )

        class _DecisionContexts:
            def outer_context(self, **kwargs):
                return kwargs

        class _Engine:
            dt = 1.0
            zero3 = np.zeros(3)
            decision_contexts = _DecisionContexts()

            def _run_agent_decision(self, *args, **kwargs):
                return {"guidance_throttle": 0.5}

        from sim.rocket import RocketAscentSimulator

        sim_cfg = RocketSimConfig(
            max_time_s=1.0,
            dt_s=1.0,
            enable_drag=False,
            enable_j2=False,
            enable_j3=False,
            enable_j4=False,
            tvc_time_constant_s=0.01,
            tvc_rate_limit_deg_s=180.0,
            tvc_max_gimbal_deg=45.0,
        )
        guidance = _VectorGuidance()
        rocket_sim = RocketAscentSimulator(
            sim_cfg=sim_cfg,
            vehicle_cfg=self._vehicle(),
            guidance=guidance,
        )
        rocket_state = rocket_sim.initial_state()
        truth = _rocket_state_to_truth(rocket_state)
        agent = AgentRuntime(
            object_id="rocket",
            kind="rocket",
            enabled=True,
            active=True,
            truth=truth,
            belief=StateBelief(
                state=np.hstack((truth.position_eci_km, truth.velocity_eci_km_s)),
                covariance=np.eye(6),
                last_update_t_s=0.0,
            ),
            sensor=None,
            estimator=None,
            orbit_controller=None,
            attitude_controller=None,
            dynamics=None,
            knowledge_base=None,
            bridge=None,
            mission_strategy=None,
            mission_execution=None,
            rocket_sim=rocket_sim,
            rocket_state=rocket_state,
            rocket_guidance=guidance,
            deploy_source=None,
            deploy_time_s=None,
            deploy_dv_body_m_s=None,
            mission_modules=[],
            waiting_for_launch=False,
        )

        result = _RocketStepper(_Engine()).step(agent=agent, world_truth_decision={}, t_s=0.0, t_next=1.0)

        self.assertAlmostEqual(result.throttle, 0.5, places=12)
        self.assertGreater(agent.rocket_state.thrust_vector_body[1], 0.1)
        self.assertGreater(abs(result.thrust_eci_km_s2[1]), 1e-9)


if __name__ == "__main__":
    unittest.main()
