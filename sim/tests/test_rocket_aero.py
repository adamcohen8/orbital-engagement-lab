import unittest

import numpy as np

from sim.aero import RocketAeroConfig as SharedRocketAeroConfig
from sim.aero import resolve_vehicle_aero_properties
from sim.config import scenario_config_from_dict
from sim.rocket.aero import RocketAeroConfig, compute_aero_loads, compute_aero_state
from sim.runtime_support import _create_rocket_runtime, _create_satellite_runtime


class TestRocketAero(unittest.TestCase):
    def test_rocket_aero_reexports_shared_model(self):
        self.assertIs(RocketAeroConfig, SharedRocketAeroConfig)

    def test_rocket_runtime_accepts_object_level_aero_specs(self):
        cfg = scenario_config_from_dict(
            {
                "rocket": {
                    "enabled": True,
                    "specs": {
                        "aero": {
                            "reference_area_m2": 3.5,
                            "reference_length_m": 11.0,
                            "cp_offset_body_m": [-1.0, 0.0, 0.0],
                            "cd": 0.44,
                        }
                    },
                    "initial_state": {
                        "launch_lat_deg": 0.0,
                        "launch_lon_deg": 0.0,
                        "launch_alt_km": 0.0,
                    },
                },
                "target": {"enabled": False},
                "chaser": {"enabled": False},
                "simulator": {
                    "duration_s": 1.0,
                    "dt_s": 1.0,
                    "dynamics": {"rocket": {"use_stagewise_aero_geometry": False}},
                    "termination": {"earth_impact_enabled": False},
                },
            }
        )
        runtime = _create_rocket_runtime(cfg)
        aero = runtime.rocket_sim.sim_cfg.aero

        self.assertAlmostEqual(aero.reference_area_m2, 3.5)
        self.assertAlmostEqual(aero.reference_length_m, 11.0)
        self.assertAlmostEqual(aero.cd_base, 0.44)
        self.assertTrue(np.allclose(aero.cp_offset_body_m, np.array([-1.0, 0.0, 0.0])))

    def test_rocket_object_level_aero_area_overrides_stagewise_area(self):
        cfg = scenario_config_from_dict(
            {
                "rocket": {
                    "enabled": True,
                    "specs": {
                        "aero": {
                            "reference_area_m2": 3.5,
                            "reference_length_m": 11.0,
                        }
                    },
                    "initial_state": {
                        "launch_lat_deg": 0.0,
                        "launch_lon_deg": 0.0,
                        "launch_alt_km": 0.0,
                    },
                },
                "target": {"enabled": False},
                "chaser": {"enabled": False},
                "simulator": {
                    "duration_s": 1.0,
                    "dt_s": 1.0,
                    "termination": {"earth_impact_enabled": False},
                },
            }
        )
        runtime = _create_rocket_runtime(cfg)
        sim = runtime.rocket_sim

        self.assertAlmostEqual(sim.sim_cfg.area_ref_m2, 3.5)
        self.assertAlmostEqual(sim._resolve_aero_config_for_stage(0).reference_area_m2, 3.5)

    def test_shared_aero_rejects_negative_physical_values(self):
        for specs in (
            {"aero": {"reference_area_m2": -0.1}},
            {"aero": {"drag_area_m2": -2.0}},
            {"aero": {"cd": -1.0}},
            {"aero": {"nose_radius_m": -0.4}},
            {"aero": {"reference_length_m": 0.0}},
            {"aero": {"lift_area_m2": -5.0}},
        ):
            with self.assertRaises(ValueError):
                resolve_vehicle_aero_properties(specs)

    def test_flat_aero_alias_overrides_nested_value(self):
        props = resolve_vehicle_aero_properties({"aero": {"cd": 2.2, "reference_length_m": 4.0}, "cd": 1.1})

        self.assertAlmostEqual(props.cd, 1.1)
        self.assertAlmostEqual(props.reference_length_m, 4.0)

    def test_satellite_object_level_cp_offset_feeds_drag_disturbance_torque(self):
        cfg = scenario_config_from_dict(
            {
                "chaser": {
                    "enabled": True,
                    "specs": {
                        "mass_kg": 100.0,
                        "aero": {
                            "drag_area_m2": 2.0,
                            "cd": 2.1,
                            "cp_offset_body_m": [0.1, -0.2, 0.3],
                        },
                    },
                    "initial_state": {
                        "position_eci_km": [7000.0, 0.0, 0.0],
                        "velocity_eci_km_s": [0.0, 7.5, 0.0],
                    },
                },
                "target": {"enabled": False},
                "rocket": {"enabled": False},
                "simulator": {
                    "duration_s": 1.0,
                    "dt_s": 1.0,
                    "dynamics": {
                        "attitude": {
                            "enabled": True,
                            "disturbance_torques": {
                                "drag": True,
                                "gravity_gradient": False,
                                "magnetic": False,
                                "srp": False,
                            },
                        }
                    },
                },
            }
        )
        runtime = _create_satellite_runtime("chaser", cfg.chaser, cfg, np.random.default_rng(123))
        disturbance_cfg = runtime.dynamics.disturbance_model.config

        self.assertAlmostEqual(disturbance_cfg.drag_area_m2, 2.0)
        self.assertAlmostEqual(disturbance_cfg.drag_cd, 2.1)
        self.assertTrue(np.allclose(disturbance_cfg.drag_cp_offset_body_m, np.array([0.1, -0.2, 0.3])))

    def test_state_zero_relative_speed(self):
        s = compute_aero_state(
            rho_kg_m3=1.225,
            pressure_pa=101325.0,
            temperature_k=288.15,
            sound_speed_m_s=340.0,
            v_rel_body_m_s=np.zeros(3),
            alpha_limit_deg=20.0,
            beta_limit_deg=20.0,
        )
        self.assertAlmostEqual(s.dynamic_pressure_pa, 0.0, places=12)
        self.assertAlmostEqual(s.speed_m_s, 0.0, places=12)
        self.assertAlmostEqual(s.mach, 0.0, places=12)
        self.assertAlmostEqual(s.alpha_rad, 0.0, places=12)
        self.assertAlmostEqual(s.beta_rad, 0.0, places=12)

    def test_state_angle_limits_apply(self):
        s = compute_aero_state(
            rho_kg_m3=1.0,
            pressure_pa=90000.0,
            temperature_k=260.0,
            sound_speed_m_s=320.0,
            v_rel_body_m_s=np.array([10.0, 9.0, 20.0]),
            alpha_limit_deg=5.0,
            beta_limit_deg=3.0,
        )
        self.assertAlmostEqual(s.alpha_rad, np.deg2rad(5.0), places=12)
        self.assertAlmostEqual(s.beta_rad, np.deg2rad(3.0), places=12)
        self.assertGreater(s.dynamic_pressure_pa, 0.0)
        self.assertGreater(s.mach, 0.0)

    def test_loads_zero_when_disabled_or_no_q(self):
        cfg = RocketAeroConfig(enabled=False)
        s = compute_aero_state(
            rho_kg_m3=1.2,
            pressure_pa=101325.0,
            temperature_k=288.0,
            sound_speed_m_s=340.0,
            v_rel_body_m_s=np.array([100.0, 0.0, 0.0]),
            alpha_limit_deg=20.0,
            beta_limit_deg=20.0,
        )
        loads = compute_aero_loads(np.array([100.0, 0.0, 0.0]), s, cfg)
        self.assertTrue(np.allclose(loads.force_body_n, np.zeros(3)))
        self.assertTrue(np.allclose(loads.moment_body_nm, np.zeros(3)))

    def test_drag_and_cp_offset_moment(self):
        cfg = RocketAeroConfig(
            enabled=True,
            reference_area_m2=2.0,
            reference_length_m=1.0,
            cp_offset_body_m=np.array([0.0, 1.0, 0.0]),
            cd_base=0.5,
            cd_alpha2=0.0,
            cd_supersonic=0.5,
            transonic_peak_cd=0.0,
            cl_alpha_per_rad=0.0,
            cy_beta_per_rad=0.0,
            cm_alpha_per_rad=0.0,
            cn_beta_per_rad=0.0,
            cl_roll_per_rad=0.0,
        )
        s = compute_aero_state(
            rho_kg_m3=1.2,
            pressure_pa=101325.0,
            temperature_k=288.0,
            sound_speed_m_s=340.0,
            v_rel_body_m_s=np.array([100.0, 0.0, 0.0]),
            alpha_limit_deg=20.0,
            beta_limit_deg=20.0,
        )
        loads = compute_aero_loads(np.array([100.0, 0.0, 0.0]), s, cfg)

        q = 0.5 * 1.2 * 100.0 * 100.0
        fx_expected = -q * cfg.reference_area_m2 * cfg.cd_base
        self.assertAlmostEqual(loads.force_body_n[0], fx_expected, places=6)
        self.assertAlmostEqual(loads.force_body_n[1], 0.0, places=10)
        self.assertAlmostEqual(loads.force_body_n[2], 0.0, places=10)
        self.assertAlmostEqual(loads.moment_body_nm[0], 0.0, places=8)
        self.assertAlmostEqual(loads.moment_body_nm[1], 0.0, places=8)
        self.assertAlmostEqual(loads.moment_body_nm[2], -fx_expected, places=6)


if __name__ == "__main__":
    unittest.main()
