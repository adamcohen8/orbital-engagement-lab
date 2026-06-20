import unittest
import warnings
from unittest.mock import patch

import numpy as np

from sim.dynamics.orbit.environment import EARTH_RADIUS_KM
from sim.presets.rockets import RocketStackPreset, RocketStagePreset
from sim.rocket import (
    GuidanceCommand,
    HoldAttitudeGuidance,
    RocketAeroConfig,
    RocketAscentSimulator,
    RocketSimConfig,
    RocketState,
    RocketVehicleConfig,
)


class TestRocketAscentEngine(unittest.TestCase):
    def _tiny_stack(self) -> RocketStackPreset:
        s1 = RocketStagePreset(
            name="s1",
            dry_mass_kg=100.0,
            propellant_mass_kg=200.0,
            max_thrust_n=2.0e5,
            isp_s=280.0,
            burn_time_s=20.0,
            diameter_m=1.5,
            length_m=8.0,
        )
        s2 = RocketStagePreset(
            name="s2",
            dry_mass_kg=40.0,
            propellant_mass_kg=80.0,
            max_thrust_n=7.0e4,
            isp_s=310.0,
            burn_time_s=30.0,
            diameter_m=1.2,
            length_m=5.0,
        )
        return RocketStackPreset(name="tiny", stages=(s1, s2))

    def test_mass_decreases_and_stage_progresses(self):
        sim_cfg = RocketSimConfig(
            dt_s=0.5,
            max_time_s=200.0,
            enable_drag=False,
            enable_j2=False,
            enable_j3=False,
            enable_j4=False,
        )
        vehicle_cfg = RocketVehicleConfig(stack=self._tiny_stack(), payload_mass_kg=20.0)
        sim = RocketAscentSimulator(
            sim_cfg=sim_cfg, vehicle_cfg=vehicle_cfg, guidance=HoldAttitudeGuidance(throttle=1.0)
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            out = sim.run()

        self.assertLess(out.mass_kg[-1], out.mass_kg[0])
        self.assertGreaterEqual(int(np.max(out.active_stage_index)), 1)

    def test_returns_result_arrays_consistent(self):
        sim_cfg = RocketSimConfig(
            dt_s=1.0,
            max_time_s=10.0,
            enable_drag=False,
            enable_j2=False,
            enable_j3=False,
            enable_j4=False,
        )
        vehicle_cfg = RocketVehicleConfig(stack=self._tiny_stack(), payload_mass_kg=0.0)
        sim = RocketAscentSimulator(
            sim_cfg=sim_cfg, vehicle_cfg=vehicle_cfg, guidance=HoldAttitudeGuidance(throttle=0.5)
        )
        out = sim.run()
        n = out.time_s.size
        self.assertEqual(out.position_eci_km.shape, (n, 3))
        self.assertEqual(out.velocity_eci_km_s.shape, (n, 3))
        self.assertEqual(out.attitude_quat_bn.shape, (n, 4))
        self.assertEqual(out.angular_rate_body_rad_s.shape, (n, 3))
        self.assertEqual(out.mass_kg.shape, (n,))
        self.assertEqual(out.latitude_deg.shape, (n,))
        self.assertEqual(out.longitude_deg.shape, (n,))
        self.assertEqual(out.wind_body_m_s.shape, (n, 3))
        self.assertEqual(out.tvc_gimbal_deg.shape, (n,))

    def test_generic_drag_step_reports_dynamic_pressure_and_mach(self):
        sim_cfg = RocketSimConfig(
            dt_s=1.0,
            max_time_s=1.0,
            enable_drag=False,
            enable_j2=False,
            enable_j3=False,
            enable_j4=False,
            aero=RocketAeroConfig(enabled=True),
            atmosphere_env={
                "drag_frame_model": "hpop_like",
                "jd_utc_start": 2460310.5,
            },
        )
        vehicle_cfg = RocketVehicleConfig(stack=self._tiny_stack(), payload_mass_kg=0.0)
        sim = RocketAscentSimulator(
            sim_cfg=sim_cfg, vehicle_cfg=vehicle_cfg, guidance=HoldAttitudeGuidance(throttle=0.0)
        )
        state = RocketState(
            t_s=0.0,
            position_eci_km=np.array([EARTH_RADIUS_KM + 50.0, 0.0, 0.0], dtype=float),
            velocity_eci_km_s=np.array([0.0, 7.0, 0.0], dtype=float),
            attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            angular_rate_body_rad_s=np.zeros(3),
            mass_kg=300.0,
            active_stage_index=0,
            stage_prop_remaining_kg=np.array([200.0, 80.0], dtype=float),
        )

        with patch(
            "sim.rocket.engine.atmosphere_relative_velocity_eci_km_s",
            return_value=np.array([0.0, 2.0, 0.0], dtype=float),
        ) as rel_vel:
            out = sim.step(state, GuidanceCommand(throttle=0.0), dt_s=1.0)

        self.assertGreater(float(out._last_step_q_dyn_pa), 0.0)
        self.assertGreater(float(out._last_step_mach), 0.0)
        self.assertEqual(rel_vel.call_args.kwargs["frame_model"], "hpop_like")
        self.assertIsNone(rel_vel.call_args.kwargs["eop_path"])

    def test_rocket_step_reports_tvc_torque_telemetry(self):
        sim_cfg = RocketSimConfig(
            dt_s=1.0,
            max_time_s=1.0,
            enable_drag=False,
            enable_j2=False,
            enable_j3=False,
            enable_j4=False,
            aero=RocketAeroConfig(enabled=False),
            tvc_pivot_offset_body_m=np.array([0.0, 1.0, 0.0], dtype=float),
        )
        vehicle_cfg = RocketVehicleConfig(stack=self._tiny_stack(), payload_mass_kg=0.0)
        sim = RocketAscentSimulator(
            sim_cfg=sim_cfg, vehicle_cfg=vehicle_cfg, guidance=HoldAttitudeGuidance(throttle=1.0)
        )
        state = sim.initial_state()

        out = sim.step(state, GuidanceCommand(throttle=1.0), dt_s=1.0)

        self.assertGreater(float(np.linalg.norm(out._last_step_torque_body_nm)), 0.0)

    def test_stagewise_aero_geometry_updates_with_stage(self):
        sim_cfg = RocketSimConfig(
            dt_s=1.0,
            max_time_s=1.0,
            enable_drag=True,
            enable_j2=False,
            enable_j3=False,
            enable_j4=False,
            area_ref_m2=None,
            use_stagewise_aero_geometry=True,
        )
        vehicle_cfg = RocketVehicleConfig(stack=self._tiny_stack(), payload_mass_kg=0.0)
        sim = RocketAscentSimulator(
            sim_cfg=sim_cfg, vehicle_cfg=vehicle_cfg, guidance=HoldAttitudeGuidance(throttle=0.0)
        )

        cfg0 = sim._resolve_aero_config_for_stage(0)
        cfg1 = sim._resolve_aero_config_for_stage(1)
        a0_expected = np.pi * 0.25 * (1.5**2)
        a1_expected = np.pi * 0.25 * (1.2**2)
        self.assertAlmostEqual(cfg0.reference_area_m2, a0_expected, places=10)
        self.assertAlmostEqual(cfg1.reference_area_m2, a1_expected, places=10)
        self.assertAlmostEqual(cfg0.reference_length_m, 8.0, places=10)
        self.assertAlmostEqual(cfg1.reference_length_m, 5.0, places=10)

    def test_area_override_takes_priority_over_stage_geometry(self):
        sim_cfg = RocketSimConfig(
            dt_s=1.0,
            max_time_s=1.0,
            enable_drag=True,
            enable_j2=False,
            enable_j3=False,
            enable_j4=False,
            area_ref_m2=4.2,
            use_stagewise_aero_geometry=True,
        )
        vehicle_cfg = RocketVehicleConfig(stack=self._tiny_stack(), payload_mass_kg=0.0)
        sim = RocketAscentSimulator(
            sim_cfg=sim_cfg, vehicle_cfg=vehicle_cfg, guidance=HoldAttitudeGuidance(throttle=0.0)
        )

        cfg0 = sim._resolve_aero_config_for_stage(0)
        cfg1 = sim._resolve_aero_config_for_stage(1)
        self.assertAlmostEqual(cfg0.reference_area_m2, 4.2, places=12)
        self.assertAlmostEqual(cfg1.reference_area_m2, 4.2, places=12)

    def test_wgs84_launch_outputs_geodetic_lat_lon(self):
        sim_cfg = RocketSimConfig(
            dt_s=1.0,
            max_time_s=0.0 + 1.0,
            enable_drag=False,
            enable_j2=False,
            enable_j3=False,
            enable_j4=False,
            launch_lat_deg=28.5,
            launch_lon_deg=-80.6,
            use_wgs84_geodesy=True,
        )
        vehicle_cfg = RocketVehicleConfig(stack=self._tiny_stack(), payload_mass_kg=0.0)
        sim = RocketAscentSimulator(
            sim_cfg=sim_cfg, vehicle_cfg=vehicle_cfg, guidance=HoldAttitudeGuidance(throttle=0.0)
        )
        out = sim.run()
        self.assertAlmostEqual(out.latitude_deg[0], 28.5, places=3)
        self.assertAlmostEqual(out.longitude_deg[0], -80.6, places=3)
        self.assertAlmostEqual(out.altitude_km[0], 0.0, places=3)

    def test_wind_is_reflected_in_logged_body_wind(self):
        sim_cfg = RocketSimConfig(
            dt_s=1.0,
            max_time_s=1.0,
            enable_drag=True,
            enable_j2=False,
            enable_j3=False,
            enable_j4=False,
            wind_enu_m_s=np.array([30.0, 0.0, 0.0]),
        )
        vehicle_cfg = RocketVehicleConfig(stack=self._tiny_stack(), payload_mass_kg=0.0)
        sim = RocketAscentSimulator(
            sim_cfg=sim_cfg, vehicle_cfg=vehicle_cfg, guidance=HoldAttitudeGuidance(throttle=0.0)
        )
        out = sim.run()
        self.assertGreater(np.linalg.norm(out.wind_body_m_s[1]), 1.0)

    def test_tvc_command_lags_and_generates_gimbal_angle(self):
        class _TvcGuidance:
            def command(self, state, sim_cfg, vehicle_cfg):
                return GuidanceCommand(
                    throttle=1.0,
                    torque_body_nm_cmd=np.zeros(3),
                    thrust_vector_body_cmd=np.array([1.0, 0.1, 0.0]),
                )

        sim_cfg = RocketSimConfig(
            dt_s=0.1,
            max_time_s=0.2,
            enable_drag=False,
            enable_j2=False,
            enable_j3=False,
            enable_j4=False,
            attitude_mode="dynamic",
            tvc_time_constant_s=0.5,
            tvc_rate_limit_deg_s=10.0,
            tvc_max_gimbal_deg=5.0,
            tvc_pivot_offset_body_m=np.array([0.0, 0.0, -1.0]),
        )
        vehicle_cfg = RocketVehicleConfig(stack=self._tiny_stack(), payload_mass_kg=0.0)
        sim = RocketAscentSimulator(sim_cfg=sim_cfg, vehicle_cfg=vehicle_cfg, guidance=_TvcGuidance())
        out = sim.run()
        self.assertGreater(out.tvc_gimbal_deg[1], 0.0)
        self.assertLessEqual(out.tvc_gimbal_deg[1], 1.1)

    def test_engine_performance_increases_toward_vacuum(self):
        stage = RocketStagePreset(
            name="perf",
            dry_mass_kg=10.0,
            propellant_mass_kg=50.0,
            max_thrust_n=1200.0,
            isp_s=300.0,
            burn_time_s=20.0,
            diameter_m=1.0,
            length_m=3.0,
            sea_level_thrust_n=1000.0,
            vacuum_thrust_n=1200.0,
            sea_level_isp_s=250.0,
            vacuum_isp_s=300.0,
        )
        stack = RocketStackPreset(name="perf_stack", stages=(stage,))
        sim_cfg = RocketSimConfig(
            dt_s=0.5,
            max_time_s=0.5,
            enable_drag=False,
            enable_j2=False,
            enable_j3=False,
            enable_j4=False,
        )
        vehicle_cfg = RocketVehicleConfig(stack=stack, payload_mass_kg=0.0)
        sim = RocketAscentSimulator(
            sim_cfg=sim_cfg, vehicle_cfg=vehicle_cfg, guidance=HoldAttitudeGuidance(throttle=1.0)
        )
        sea_state = sim.initial_state()
        high_state = sea_state.copy()
        high_state.position_eci_km = np.array([EARTH_RADIUS_KM + 200.0, 0.0, 0.0], dtype=float)
        high_state.velocity_eci_km_s = np.zeros(3)
        sea_next = sim.step(sea_state, GuidanceCommand(throttle=1.0), dt_s=0.5)
        high_next = sim.step(high_state, GuidanceCommand(throttle=1.0), dt_s=0.5)
        self.assertGreater(high_next._last_step_thrust_n, sea_next._last_step_thrust_n)

    def test_early_termination_history_includes_post_step_state(self):
        sim_cfg = RocketSimConfig(
            dt_s=1.0,
            max_time_s=5.0,
            enable_drag=False,
            enable_j2=False,
            enable_j3=False,
            enable_j4=False,
            terminate_on_earth_impact=True,
            earth_impact_radius_km=1.0e9,
            use_wgs84_geodesy=False,
        )
        vehicle_cfg = RocketVehicleConfig(stack=self._tiny_stack(), payload_mass_kg=0.0)
        sim = RocketAscentSimulator(
            sim_cfg=sim_cfg, vehicle_cfg=vehicle_cfg, guidance=HoldAttitudeGuidance(throttle=0.0)
        )

        out = sim.run()

        self.assertTrue(out.terminated_early)
        self.assertEqual(out.termination_reason, "earth_impact")
        self.assertEqual(out.time_s.tolist(), [0.0, 1.0])
        self.assertAlmostEqual(out.termination_time_s, out.time_s[-1], places=12)
        self.assertGreater(np.linalg.norm(out.position_eci_km[-1]), 1.0)
        self.assertGreater(np.linalg.norm(out.velocity_eci_km_s[-1] - out.velocity_eci_km_s[0]), 0.0)

    def test_propellant_depletion_limits_average_step_thrust(self):
        stage = RocketStagePreset(
            name="short",
            dry_mass_kg=10.0,
            propellant_mass_kg=1.0,
            max_thrust_n=1000.0,
            isp_s=100.0,
            burn_time_s=1.0,
            diameter_m=1.0,
            length_m=3.0,
        )
        stack = RocketStackPreset(name="short_stack", stages=(stage,))
        sim_cfg = RocketSimConfig(
            dt_s=10.0,
            max_time_s=10.0,
            enable_drag=False,
            enable_j2=False,
            enable_j3=False,
            enable_j4=False,
            terminate_on_earth_impact=False,
        )
        vehicle_cfg = RocketVehicleConfig(stack=stack, payload_mass_kg=100.0)
        sim = RocketAscentSimulator(
            sim_cfg=sim_cfg, vehicle_cfg=vehicle_cfg, guidance=HoldAttitudeGuidance(throttle=1.0)
        )

        next_state = sim.step(sim.initial_state(), GuidanceCommand(throttle=1.0), dt_s=10.0)

        expected_average_thrust_n = stage.propellant_mass_kg * stage.isp_s * 9.80665 / 10.0
        self.assertAlmostEqual(next_state._last_step_thrust_n, expected_average_thrust_n, places=6)
        self.assertEqual(next_state.active_stage_index, 1)
        self.assertAlmostEqual(next_state.stage_prop_remaining_kg[0], 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
