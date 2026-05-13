from __future__ import annotations

import unittest

import numpy as np

from sim.config import scenario_config_from_dict
from sim.presets.rockets import BASIC_TWO_STAGE_STACK
from sim.rocket import HoldAttitudeGuidance, RocketAscentSimulator, RocketSimConfig, RocketVehicleConfig
from sim.rocket.navigation import build_rocket_nav_state
from sim.single_run import _SingleRunEngine


class TestRocketNavigation(unittest.TestCase):
    def test_nav_state_reports_launch_site_and_energy_terms(self):
        sim_cfg = RocketSimConfig(
            dt_s=1.0,
            max_time_s=1.0,
            launch_lat_deg=28.5,
            launch_lon_deg=-80.6,
            launch_alt_km=0.1,
            enable_drag=False,
            enable_j2=False,
            enable_j3=False,
            enable_j4=False,
            use_wgs84_geodesy=True,
        )
        vehicle_cfg = RocketVehicleConfig(stack=BASIC_TWO_STAGE_STACK, payload_mass_kg=150.0)
        sim = RocketAscentSimulator(
            sim_cfg=sim_cfg,
            vehicle_cfg=vehicle_cfg,
            guidance=HoldAttitudeGuidance(throttle=0.0),
        )

        nav = build_rocket_nav_state(sim.initial_state(), sim_cfg, vehicle_cfg)

        self.assertAlmostEqual(nav.latitude_deg, 28.5, places=3)
        self.assertAlmostEqual(nav.longitude_deg, -80.6, places=3)
        self.assertAlmostEqual(nav.altitude_km, 0.1, places=3)
        self.assertGreater(nav.speed_km_s, 0.0)
        self.assertTrue(np.isfinite(nav.apoapsis_alt_km))
        self.assertTrue(np.isfinite(nav.eccentricity))
        self.assertAlmostEqual(nav.propellant_remaining_fraction, 1.0, places=12)
        self.assertFalse(nav.stages_complete)

    def test_single_run_payload_includes_rocket_gnc_metrics(self):
        cfg = scenario_config_from_dict(
            {
                "scenario_name": "rocket_gnc_metrics_test",
                "objects": {
                    "rocket": {
                        "enabled": True,
                        "role": "rocket",
                        "kind": "rocket",
                        "specs": {
                            "preset_stack": "BASIC_TWO_STAGE_STACK",
                            "payload_mass_kg": 150.0,
                            "thrust_axis_body": [1.0, 0.0, 0.0],
                        },
                        "initial_state": {
                            "launch_lat_deg": 28.5,
                            "launch_lon_deg": -80.6,
                            "launch_alt_km": 0.1,
                            "launch_azimuth_deg": 90.0,
                        },
                    }
                },
                "simulator": {
                    "scenario_type": "rocket_ascent",
                    "duration_s": 3.0,
                    "dt_s": 1.0,
                    "dynamics": {
                        "orbit": {"j2": False, "j3": False, "j4": False, "drag": False, "srp": False},
                        "rocket": {"aero_model_enabled": True, "attitude_mode": "cheater"},
                    },
                    "termination": {"earth_impact_enabled": False},
                },
                "outputs": {
                    "output_dir": "outputs/test_rocket_gnc_metrics",
                    "mode": "interactive",
                    "stats": {"save_json": False},
                    "plots": {"enabled": False},
                    "animations": {"enabled": False},
                },
                "monte_carlo": {"enabled": False},
            }
        )

        payload = _SingleRunEngine(cfg).run()
        metrics = payload["rocket_metrics"]
        summary = payload["summary"]["rocket_metrics_summary"]

        for key in (
            "altitude_km",
            "speed_km_s",
            "flight_path_angle_deg",
            "apoapsis_alt_km",
            "periapsis_alt_km",
            "tvc_gimbal_deg",
            "propellant_remaining_fraction",
        ):
            self.assertIn(key, metrics)
            self.assertEqual(len(metrics[key]), len(payload["time_s"]))
        self.assertIn("final_apoapsis_alt_km", summary)
        self.assertIn("max_dynamic_pressure_pa", summary)


if __name__ == "__main__":
    unittest.main()
