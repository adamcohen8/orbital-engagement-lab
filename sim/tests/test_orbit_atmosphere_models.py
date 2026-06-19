import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np

from sim.aero.core import atmosphere_relative_velocity_eci_km_s
from sim.dynamics.orbit.accelerations import accel_drag, accel_lift, accel_srp
from sim.dynamics.orbit.atmosphere import (
    _altitude_km_from_eci,
    _spherical_lat_lon_deg_from_eci,
    density_exponential,
    density_from_model,
    density_harris_priester,
    density_jacchia70,
    density_msis86,
    density_ussa1976,
)
from sim.dynamics.orbit.eclipse import resolve_srp_geometry, srp_shadow_factor
from sim.dynamics.orbit.environment import EARTH_ROT_RATE_RAD_S
from sim.dynamics.orbit.epoch import AU_KM, datetime_to_julian_date, gmst_angle_rad_from_jd
from sim.dynamics.orbit.msis86_backend import msis86_density as msis86_backend_density


class TestOrbitAtmosphereModels(unittest.TestCase):
    @staticmethod
    def _write_minimal_jb2008_tables(sol_path: Path, dtc_path: Path, dt_utc: datetime) -> None:
        jd_floor = int(np.floor(datetime_to_julian_date(dt_utc) - 1.0))
        day_of_year = int(dt_utc.timetuple().tm_yday)
        sol_row = [0.0, 0.0, float(jd_floor), 150.0, 150.0, 140.0, 140.0, 130.0, 130.0, 120.0, 120.0]
        dtc_row = [float(dt_utc.year), float(day_of_year)] + [0.0] * 24
        np.savetxt(sol_path, np.array([sol_row, sol_row], dtype=float), fmt="%.6f")
        np.savetxt(dtc_path, np.array([dtc_row, dtc_row], dtype=float), fmt="%.6f")

    @staticmethod
    def _write_minimal_jb2006_tables(sol_path: Path, ap_path: Path, dt_utc: datetime) -> None:
        jd_floor = int(np.floor(datetime_to_julian_date(dt_utc)))
        rows = []
        for offset in range(-6, 2):
            rows.append(
                [
                    0.0,
                    0.0,
                    float(jd_floor + offset),
                    150.0 + offset,
                    150.0,
                    140.0 + offset,
                    140.0,
                    130.0 + offset,
                    130.0,
                    120.0,
                    120.0,
                ]
            )
        lag_dt = dt_utc.replace(tzinfo=timezone.utc)
        day_of_year = int(lag_dt.timetuple().tm_yday)
        ap_row = [float(lag_dt.year), float(day_of_year), 0.0, 0.0] + [4.0] * 8
        np.savetxt(sol_path, np.array(rows, dtype=float), fmt="%.6f")
        np.savetxt(ap_path, np.array([ap_row, ap_row], dtype=float), fmt="%.6f")

    @staticmethod
    def _write_minimal_nrlmsise00_sw_table(sw_path: Path) -> datetime:
        rows = []
        for day in range(7, 11):
            row = np.zeros(33, dtype=float)
            row[0] = 2024.0
            row[1] = 1.0
            row[2] = float(day)
            row[14:22] = 4.0 + day
            row[22] = 4.0 + day
            row[27] = 150.0 + day
            row[29] = 145.0 + day
            rows.append(row)
        np.savetxt(sw_path, np.array(rows, dtype=float), fmt="%.6f")
        return datetime(2024, 1, 10, 12, 0, 0, tzinfo=timezone.utc)

    def test_ussa1976_density_reasonable_at_sea_level(self):
        r = np.array([6378.137, 0.0, 0.0], dtype=float)
        rho = density_ussa1976(r, t_s=0.0)
        self.assertGreater(rho, 1.0)
        self.assertLess(rho, 1.4)

    def test_density_models_selectable(self):
        r = np.array([6778.137, 0.0, 0.0], dtype=float)
        rho_exp = density_from_model("exponential", r, 0.0, env={})
        rho_ussa = density_from_model("ussa1976", r, 0.0, env={})
        self.assertGreaterEqual(rho_exp, 0.0)
        self.assertGreaterEqual(rho_ussa, 0.0)

    def test_density_exponential_remains_positive_above_180_km(self):
        r_200km = np.array([6378.137 + 200.0, 0.0, 0.0], dtype=float)
        rho = density_exponential(r_200km, t_s=0.0)
        self.assertGreater(rho, 0.0)

    def test_density_exponential_skips_ecef_conversion(self):
        r = np.array([6378.137 + 200.0, 0.0, 0.0], dtype=float)
        with patch("sim.dynamics.orbit.atmosphere.eci_to_ecef_harmonic", side_effect=AssertionError("should not be called")):
            rho = density_exponential(r, t_s=123.0)
            rho_from_model = density_from_model("exponential", r, 123.0, env={"geodetic_model": "wgs84"})
        self.assertGreater(rho, 0.0)
        self.assertAlmostEqual(rho_from_model, rho)

    def test_density_nrlmsise00_callable_hook(self):
        calls = []

        def _fn(alt_km, lat_deg, lon_deg, dt_utc, env):
            calls.append((alt_km, lat_deg, lon_deg, dt_utc))
            return 1.23e-11

        env = {
            "nrlmsise00_density_callable": _fn,
            "atmo_epoch_utc": datetime(2024, 1, 1, tzinfo=timezone.utc),
        }
        r = np.array([7000.0, 0.0, 0.0], dtype=float)
        rho = density_from_model("nrlmsise00", r, t_s=60.0, env=env)
        self.assertAlmostEqual(rho, 1.23e-11)
        self.assertEqual(len(calls), 1)

    def test_density_nrlmsise00_builtin_backend_returns_finite_density(self):
        r = np.array([6378.137 + 400.0, 0.0, 0.0], dtype=float)
        env = {
            "atmo_epoch_utc": datetime(2024, 3, 20, 12, 0, 0, tzinfo=timezone.utc),
            "geodetic_model": "wgs84",
            "f107": 150.0,
            "f107a": 150.0,
            "ap": 4.0,
        }

        rho = density_from_model("nrlmsise00", r, 0.0, env=env)

        self.assertTrue(np.isfinite(rho))
        self.assertGreater(rho, 0.0)

    def test_density_nrlmsise00_is_solar_flux_sensitive_and_local(self):
        r = np.array([6378.137 + 400.0, 0.0, 0.0], dtype=float)
        base_env = {
            "atmo_epoch_utc": datetime(2024, 3, 20, 12, 0, 0, tzinfo=timezone.utc),
            "geodetic_model": "wgs84",
            "ap": 4.0,
        }

        rho_low = density_from_model("nrlmsise00", r, 0.0, env={**base_env, "f107": 90.0, "f107a": 90.0})
        rho_high = density_from_model("nrlmsise00", r, 0.0, env={**base_env, "f107": 220.0, "f107a": 220.0})

        self.assertTrue(np.isfinite(rho_low))
        self.assertTrue(np.isfinite(rho_high))
        self.assertGreater(rho_high, rho_low)

    def test_density_nrlmsise00_reads_hpop_style_sw_table(self):
        r = np.array([6378.137 + 400.0, 0.0, 0.0], dtype=float)
        with tempfile.TemporaryDirectory() as td:
            sw_path = Path(td) / "SW-All.txt"
            dt_utc = self._write_minimal_nrlmsise00_sw_table(sw_path)
            rho = density_from_model(
                "nrlmsise00",
                r,
                0.0,
                env={
                    "atmo_epoch_utc": dt_utc,
                    "nrlmsise00_sw_path": str(sw_path),
                    "geodetic_model": "wgs84",
                },
            )

        self.assertTrue(np.isfinite(rho))
        self.assertGreater(rho, 0.0)

    def test_density_msis86_callable_hook(self):
        calls = []

        def _fn(alt_km, lat_deg, lon_deg, dt_utc, env):
            calls.append((alt_km, lat_deg, lon_deg, dt_utc))
            return 9.87e-12

        env = {
            "msis86_density_callable": _fn,
            "atmo_epoch_utc": datetime(2024, 1, 1, tzinfo=timezone.utc),
        }
        r = np.array([7000.0, 0.0, 0.0], dtype=float)
        rho = density_from_model("hpop_msis86", r, t_s=60.0, env=env)
        self.assertAlmostEqual(rho, 9.87e-12)
        self.assertEqual(len(calls), 1)

    def test_density_msis86_builtin_backend_returns_finite_density(self):
        r = np.array([6378.137 + 400.0, 0.0, 0.0], dtype=float)
        env = {
            "atmo_epoch_utc": datetime(2024, 3, 20, 12, 0, 0, tzinfo=timezone.utc),
            "geodetic_model": "wgs84",
            "f107": 150.0,
            "f107a": 150.0,
            "ap": 4.0,
        }

        rho = density_msis86(r, 0.0, env=env)

        self.assertTrue(np.isfinite(rho))
        self.assertGreater(rho, 0.0)

    def test_density_msis86_is_solar_flux_sensitive_and_local(self):
        r = np.array([6378.137 + 400.0, 0.0, 0.0], dtype=float)
        base_env = {
            "atmo_epoch_utc": datetime(2024, 3, 20, 12, 0, 0, tzinfo=timezone.utc),
            "geodetic_model": "wgs84",
            "ap": 4.0,
        }

        rho_low = density_from_model("msis86", r, 0.0, env={**base_env, "f107": 90.0, "f107a": 90.0})
        rho_high = density_from_model("msis-86", r, 0.0, env={**base_env, "f107": 220.0, "f107a": 220.0})

        self.assertTrue(np.isfinite(rho_low))
        self.assertTrue(np.isfinite(rho_high))
        self.assertGreater(rho_high, rho_low)

    def test_density_msis86_reads_hpop_style_sw_table(self):
        r = np.array([6378.137 + 400.0, 0.0, 0.0], dtype=float)
        with tempfile.TemporaryDirectory() as td:
            sw_path = Path(td) / "SW-All.txt"
            dt_utc = self._write_minimal_nrlmsise00_sw_table(sw_path)
            rho = density_from_model(
                "msis86",
                r,
                0.0,
                env={
                    "atmo_epoch_utc": dt_utc,
                    "msis86_sw_path": str(sw_path),
                    "geodetic_model": "wgs84",
                },
            )

        self.assertTrue(np.isfinite(rho))
        self.assertGreater(rho, 0.0)

    def test_density_msis86_uses_hpop_angle_compatibility(self):
        r = np.array([5100.0, 3400.0, 2800.0], dtype=float)
        dt_utc = datetime(2024, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        jd_utc = datetime_to_julian_date(dt_utc)
        env = {
            "atmo_epoch_utc": dt_utc,
            "geodetic_model": "wgs84",
            "f107": 150.0,
            "f107a": 150.0,
            "ap": 4.0,
        }

        alt_km = _altitude_km_from_eci(r, 0.0, env=env)
        lat_deg, lon_deg = _spherical_lat_lon_deg_from_eci(r, 0.0, env=env)
        lst_hr = ((np.radians(lon_deg) + gmst_angle_rad_from_jd(jd_utc)) % (2.0 * np.pi)) * 24.0 / (2.0 * np.pi)
        expected = msis86_backend_density(
            alt_km,
            np.radians(lat_deg),
            np.radians(lon_deg),
            dt_utc,
            {**env, "msis86_lst_hr": lst_hr},
        )

        rho = density_msis86(r, 0.0, env=env)

        self.assertAlmostEqual(rho, expected, delta=max(expected, 1.0e-30) * 1.0e-12)

    def test_density_jb2008_callable_hook(self):
        calls = []

        def _fn(alt_km, lat_deg, lon_deg, dt_utc, env):
            calls.append((alt_km, lat_deg, lon_deg, dt_utc))
            return 4.56e-12

        env = {
            "jb2008_density_callable": _fn,
            "atmo_epoch_utc": datetime(2024, 1, 1, tzinfo=timezone.utc),
        }
        r = np.array([7000.0, 0.0, 0.0], dtype=float)
        rho = density_from_model("jb2008", r, t_s=60.0, env=env)
        self.assertAlmostEqual(rho, 4.56e-12)
        self.assertEqual(len(calls), 1)

    def test_density_jb2008_builtin_backend_returns_finite_density(self):
        r = np.array([7000.0, 0.0, 0.0], dtype=float)
        dt_utc = datetime(2024, 1, 1, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as td:
            sol_path = Path(td) / "SOLFSMY.txt"
            dtc_path = Path(td) / "DTCFILE.txt"
            self._write_minimal_jb2008_tables(sol_path, dtc_path, dt_utc)
            rho = density_from_model(
                "jb2008",
                r,
                t_s=60.0,
                env={
                    "atmo_epoch_utc": dt_utc,
                    "jb2008_sol_path": str(sol_path),
                    "jb2008_dtc_path": str(dtc_path),
                    "geodetic_model": "wgs84",
                },
            )
        self.assertTrue(np.isfinite(rho))
        self.assertGreaterEqual(rho, 0.0)

    def test_density_jb2006_builtin_backend_returns_finite_density(self):
        r = np.array([7000.0, 0.0, 0.0], dtype=float)
        dt_utc = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as td:
            sol_path = Path(td) / "SOLFSMY.txt"
            ap_path = Path(td) / "SOLRESAP.txt"
            self._write_minimal_jb2006_tables(sol_path, ap_path, dt_utc)
            rho = density_from_model(
                "jb2006",
                r,
                t_s=60.0,
                env={
                    "atmo_epoch_utc": dt_utc,
                    "jb2006_sol_path": str(sol_path),
                    "jb2006_ap_path": str(ap_path),
                    "geodetic_model": "wgs84",
                },
            )
        self.assertTrue(np.isfinite(rho))
        self.assertGreaterEqual(rho, 0.0)

    def test_density_jb2006_callable_hook(self):
        calls = []

        def _fn(alt_km, lat_deg, lon_deg, dt_utc, env):
            calls.append((alt_km, lat_deg, lon_deg, dt_utc))
            return 7.89e-12

        env = {
            "jb2006_density_callable": _fn,
            "atmo_epoch_utc": datetime(2024, 1, 1, tzinfo=timezone.utc),
        }
        r = np.array([7000.0, 0.0, 0.0], dtype=float)
        rho = density_from_model("jb2006", r, t_s=60.0, env=env)
        self.assertAlmostEqual(rho, 7.89e-12)
        self.assertEqual(len(calls), 1)

    def test_density_jacchia70_builtin_backend_returns_finite_density(self):
        r = np.array([6378.137 + 400.0, 0.0, 0.0], dtype=float)
        dt_utc = datetime(2024, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        env = {
            "atmo_epoch_utc": dt_utc,
            "geodetic_model": "wgs84",
            "jacchia70_f10": 150.0,
            "jacchia70_f10b": 150.0,
            "jacchia70_ap": 4.0,
        }

        rho = density_jacchia70(r, t_s=0.0, env=env)

        self.assertTrue(np.isfinite(rho))
        self.assertGreater(rho, 0.0)

    def test_density_jacchia70_is_solar_flux_sensitive_and_local(self):
        r = np.array([6378.137 + 400.0, 0.0, 0.0], dtype=float)
        base_env = {
            "atmo_epoch_utc": datetime(2024, 3, 20, 12, 0, 0, tzinfo=timezone.utc),
            "geodetic_model": "wgs84",
            "jacchia70_ap": 4.0,
        }
        rho_low = density_from_model(
            "jacchia70", r, 0.0, env={**base_env, "jacchia70_f10": 90.0, "jacchia70_f10b": 90.0}
        )
        rho_high = density_from_model(
            "jacchia-70", r, 0.0, env={**base_env, "jacchia70_f10": 220.0, "jacchia70_f10b": 220.0}
        )

        self.assertTrue(np.isfinite(rho_low))
        self.assertTrue(np.isfinite(rho_high))
        self.assertGreater(rho_high, rho_low)

    def test_density_jacchia70_callable_hook(self):
        calls = []

        def _fn(alt_km, lat_deg, lon_deg, dt_utc, env):
            calls.append((alt_km, lat_deg, lon_deg, dt_utc))
            return 2.34e-12

        env = {
            "jacchia70_density_callable": _fn,
            "atmo_epoch_utc": datetime(2024, 1, 1, tzinfo=timezone.utc),
        }
        r = np.array([7000.0, 0.0, 0.0], dtype=float)
        rho = density_from_model("hpop_jacchia70", r, t_s=60.0, env=env)

        self.assertAlmostEqual(rho, 2.34e-12)
        self.assertEqual(len(calls), 1)

    def test_harris_priester_density_is_local_and_f107_sensitive(self):
        r = np.array([6378.137 + 400.0, 0.0, 0.0], dtype=float)
        env_low = {"sun_pos_eci_km": np.array([1.0, 0.0, 0.0]), "harris_priester_f107": 65}
        env_high = {"sun_pos_eci_km": np.array([1.0, 0.0, 0.0]), "harris_priester_f107": 275}

        rho_low = density_harris_priester(r, t_s=0.0, env=env_low)
        rho_high = density_from_model("harris_priester", r, 0.0, env=env_high)

        self.assertTrue(np.isfinite(rho_low))
        self.assertTrue(np.isfinite(rho_high))
        self.assertGreater(rho_low, 0.0)
        self.assertGreater(rho_high, rho_low)

    def test_harris_priester_aliases_select_same_model(self):
        r = np.array([6378.137 + 500.0, 0.0, 0.0], dtype=float)
        env = {"sun_pos_eci_km": np.array([1.0, 0.0, 0.0]), "harris_priester_f107": 175}

        rho_named = density_from_model("harris_priester", r, 0.0, env=env)
        rho_alias = density_from_model("hp", r, 0.0, env=env)

        self.assertAlmostEqual(rho_named, rho_alias)

    def test_harris_priester_uses_wgs84_geodetic_height_when_requested(self):
        r = np.array([3506.788211789612, 4884.292725366271, 2667.959407741529], dtype=float)
        env = {"sun_pos_eci_km": np.array([1.0, 0.0, 0.0]), "harris_priester_f107": 175}

        rho_spherical = density_harris_priester(r, t_s=0.0, env=env)
        rho_geodetic = density_harris_priester(r, t_s=0.0, env={**env, "geodetic_model": "wgs84"})

        self.assertTrue(np.isfinite(rho_spherical))
        self.assertTrue(np.isfinite(rho_geodetic))
        self.assertGreater(rho_spherical, rho_geodetic)

    def test_drag_uses_rotating_atmosphere_relative_velocity(self):
        r = np.array([7000.0, 0.0, 0.0], dtype=float)
        omega = np.array([0.0, 0.0, EARTH_ROT_RATE_RAD_S], dtype=float)
        v_atm = np.cross(omega, r)
        a = accel_drag(
            r_eci_km=r,
            v_eci_km_s=v_atm,  # matches corotating atmosphere speed at position
            t_s=0.0,
            mass_kg=100.0,
            area_m2=1.0,
            cd=2.2,
            env={"density_kg_m3": density_exponential(r, 0.0)},
        )
        self.assertTrue(np.linalg.norm(a) < 1e-14)

    def test_drag_hpop_like_frame_matches_shared_relative_velocity_helper(self):
        r = np.array([5100.0, -3200.0, 2600.0], dtype=float)
        v = np.array([3.9, 5.8, -1.2], dtype=float)
        rho = 2.0e-11
        mass_kg = 250.0
        area_m2 = 1.7
        cd = 2.1
        env = {
            "density_kg_m3": rho,
            "drag_area_m2": area_m2,
            "drag_frame_model": "hpop_like",
            "jd_utc_start": 2460310.5,
        }

        a = accel_drag(
            r_eci_km=r,
            v_eci_km_s=v,
            t_s=123.0,
            mass_kg=mass_kg,
            area_m2=area_m2,
            cd=cd,
            env=env,
        )
        v_rel_m_s = (
            atmosphere_relative_velocity_eci_km_s(
                r,
                v,
                t_s=123.0,
                frame_model="hpop_like",
                jd_utc_start=2460310.5,
            )
            * 1000.0
        )
        expected = -0.5 * rho * cd * area_m2 / mass_kg * float(np.linalg.norm(v_rel_m_s)) * v_rel_m_s / 1000.0

        np.testing.assert_allclose(a, expected, rtol=0.0, atol=1e-18)

    def test_lift_projects_attitude_vector_perpendicular_to_relative_wind(self):
        r = np.array([7000.0, 0.0, 0.0], dtype=float)
        v = np.array([0.0, 7.6, 0.0], dtype=float)

        a = accel_lift(
            r_eci_km=r,
            v_eci_km_s=v,
            t_s=0.0,
            mass_kg=100.0,
            area_m2=2.0,
            cl=0.8,
            lift_direction_eci=np.array([0.0, 0.0, 1.0]),
            env={"density_kg_m3": 1.0e-9},
        )

        self.assertGreater(a[2], 0.0)
        self.assertAlmostEqual(float(np.dot(a, v)), 0.0, delta=1e-12)

    def test_lift_returns_zero_when_axis_is_along_relative_wind(self):
        r = np.array([7000.0, 0.0, 0.0], dtype=float)
        v = np.array([0.0, 7.6, 0.0], dtype=float)

        a = accel_lift(
            r_eci_km=r,
            v_eci_km_s=v,
            t_s=0.0,
            mass_kg=100.0,
            area_m2=2.0,
            cl=0.8,
            lift_direction_eci=np.array([0.0, 1.0, 0.0]),
            env={"density_kg_m3": 1.0e-9},
        )

        np.testing.assert_allclose(a, np.zeros(3), atol=1e-15)

    def test_srp_scales_with_sun_spacecraft_distance(self):
        r = np.array([6878.137, 0.0, 0.0], dtype=float)
        a_1au = accel_srp(
            r_eci_km=r,
            mass_kg=100.0,
            area_m2=1.0,
            cr=1.0,
            t_s=0.0,
            env={"sun_pos_eci_km": np.array([AU_KM, 0.0, 0.0]), "srp_shadow_model": "none"},
        )
        a_half_au = accel_srp(
            r_eci_km=r,
            mass_kg=100.0,
            area_m2=1.0,
            cr=1.0,
            t_s=0.0,
            env={"sun_pos_eci_km": np.array([0.5 * AU_KM, 0.0, 0.0]), "srp_shadow_model": "none"},
        )
        ratio = float(np.linalg.norm(a_half_au) / max(np.linalg.norm(a_1au), 1e-18))
        self.assertGreater(ratio, 3.95)
        self.assertLess(ratio, 4.05)

    def test_srp_accel_accepts_precomputed_geometry_bundle(self):
        r = np.array([6878.137, 0.0, 0.0], dtype=float)
        env = {"sun_pos_eci_km": np.array([AU_KM, 0.0, 0.0]), "srp_shadow_model": "conical"}
        geometry = resolve_srp_geometry(r, 0.0, env)
        shadow = srp_shadow_factor(r, 0.0, env, srp_geometry=geometry)

        a_direct = accel_srp(
            r_eci_km=r,
            mass_kg=100.0,
            area_m2=1.0,
            cr=1.0,
            t_s=0.0,
            env=env,
        )
        a_cached = accel_srp(
            r_eci_km=r,
            mass_kg=100.0,
            area_m2=1.0,
            cr=1.0,
            t_s=0.0,
            env={
                "srp_geometry": geometry,
                "srp_sun_dir_eci": geometry["sun_dir_sc_eci"],
                "srp_distance_scale": geometry["distance_scale"],
                "srp_shadow_factor": shadow,
                "srp_area_m2": 1.0,
                "srp_shadow_model": "conical",
            },
        )
        np.testing.assert_allclose(a_cached, a_direct, rtol=0.0, atol=1e-15)


if __name__ == "__main__":
    unittest.main()
