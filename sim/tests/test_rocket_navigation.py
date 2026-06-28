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

    def test_process_pool_rocket_step_updates_parent_runtime_state(self):
        base = {
            "scenario_name": "rocket_process_pool_parent_state",
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
                },
                "target": {
                    "enabled": True,
                    "kind": "satellite",
                    "role": "target",
                    "specs": {"mass_kg": 100.0},
                    "initial_state": {
                        "position_eci_km": [7000.0, 0.0, 0.0],
                        "velocity_eci_km_s": [0.0, 7.546, 0.0],
                    },
                },
            },
            "simulator": {
                "duration_s": 3.0,
                "dt_s": 1.0,
                "resource_profile": "off",
                "dynamics": {
                    "orbit": {"j2": False, "j3": False, "j4": False, "drag": False, "srp": False},
                    "attitude": {"enabled": False},
                    "rocket": {"aero_model_enabled": False, "attitude_mode": "cheater"},
                },
                "termination": {"earth_impact_enabled": False},
            },
            "outputs": {
                "mode": "interactive",
                "stats": {"save_json": False, "save_full_log": False},
                "plots": {"enabled": False, "figure_ids": []},
                "animations": {"enabled": False, "types": []},
            },
        }
        serial_root = dict(base)
        serial_root["simulator"] = dict(base["simulator"])
        serial_root["simulator"]["execution"] = {"object_parallelism": {"enabled": False, "backend": "serial"}}
        serial_payload = _SingleRunEngine(scenario_config_from_dict(serial_root)).run()

        process_root = dict(base)
        process_root["simulator"] = dict(base["simulator"])
        process_root["simulator"]["execution"] = {
            "object_parallelism": {
                "enabled": True,
                "backend": "process_pool",
                "workers": 2,
                "min_objects": 2,
            }
        }
        try:
            process_payload = _SingleRunEngine(scenario_config_from_dict(process_root)).run()
        except RuntimeError as exc:
            if "ProcessPoolObjectStepExecutor is unavailable" in str(exc):
                self.skipTest(str(exc))
            raise

        serial_rocket = np.asarray(serial_payload["truth_by_object"]["rocket"], dtype=float)
        process_rocket = np.asarray(process_payload["truth_by_object"]["rocket"], dtype=float)
        self.assertEqual(process_payload["summary"]["runtime_profile"]["executor"]["object_step_backend"], "process_pool")
        self.assertGreater(float(np.linalg.norm(serial_rocket[-1, :3] - serial_rocket[0, :3])), 0.0)
        self.assertTrue(np.allclose(process_rocket[-1, :6], serial_rocket[-1, :6], rtol=0.0, atol=1e-9))


if __name__ == "__main__":
    unittest.main()
