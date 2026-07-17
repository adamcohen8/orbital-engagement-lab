from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.core.models import StateTruth
from sim.dynamics.orbit.atmosphere import density_from_model
from sim.dynamics.orbit.eclipse import srp_shadow_factor
from sim.dynamics.orbit.environment import EARTH_ROT_RATE_RAD_S, srp_pressure_n_m2
from sim.dynamics.spacecraft_geometry import GeometryAreaProfile, RectangularPrismGeometry
from sim.utils.quaternion import quaternion_to_dcm_bn

# Earth dipole field parameter in T*m^3 for a simple centered dipole model.
EARTH_MAGNETIC_DIPOLE_T_M3 = 7.94e15


def _cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the cross product of two three-vectors without generic axis setup."""
    out = np.empty(3, dtype=float)
    out[0] = a[1] * b[2] - a[2] * b[1]
    out[1] = a[2] * b[0] - a[0] * b[2]
    out[2] = a[0] * b[1] - a[1] * b[0]
    return out


def _norm3(vector: np.ndarray) -> float:
    """Return the Euclidean norm of a three-vector without generic linalg dispatch."""
    return float(np.sqrt(np.dot(vector, vector)))


@dataclass(frozen=True)
class DisturbanceTorqueConfig:
    use_gravity_gradient: bool = True
    use_magnetic: bool = True
    use_drag: bool = True
    use_srp: bool = True
    magnetic_dipole_body_a_m2: np.ndarray = field(default_factory=lambda: np.array([0.05, 0.0, 0.0]))
    drag_area_m2: float = 1.5
    drag_cd: float = 2.2
    drag_cp_offset_body_m: np.ndarray = field(default_factory=lambda: np.array([0.05, 0.02, -0.01]))
    drag_facet_normal_body: np.ndarray | None = None
    drag_facets: tuple[dict, ...] | None = None
    srp_area_m2: float = 1.0
    srp_cr: float = 1.3
    srp_cp_offset_body_m: np.ndarray = field(default_factory=lambda: np.array([-0.02, 0.03, 0.01]))
    sun_dir_eci: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    srp_facet_normal_body: np.ndarray | None = None
    srp_facets: tuple[dict, ...] | None = None
    use_rectangular_prism_faces: bool = False
    rectangular_prism_dims_m: tuple[float, float, float] | None = None
    geometry_area_profile: GeometryAreaProfile | None = None
    center_of_mass_body_m: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass(frozen=True)
class DisturbanceTorqueModel:
    mu_km3_s2: float
    inertia_kg_m2: np.ndarray
    config: DisturbanceTorqueConfig = field(default_factory=DisturbanceTorqueConfig)

    def total_torque_body_nm(self, state: StateTruth, env: dict | None = None) -> np.ndarray:
        env = env or {}
        tau = np.zeros(3)
        c_bn = None

        if self.config.use_gravity_gradient:
            c_bn = quaternion_to_dcm_bn(state.attitude_quat_bn) if c_bn is None else c_bn
            tau += self._gravity_gradient_torque(state, c_bn)
        if self.config.use_magnetic:
            c_bn = quaternion_to_dcm_bn(state.attitude_quat_bn) if c_bn is None else c_bn
            tau += self._magnetic_torque(state, c_bn, env)
        if self.config.use_drag:
            c_bn = quaternion_to_dcm_bn(state.attitude_quat_bn) if c_bn is None else c_bn
            tau += self._drag_torque(state, env, c_bn)
        if self.config.use_srp:
            c_bn = quaternion_to_dcm_bn(state.attitude_quat_bn) if c_bn is None else c_bn
            tau += self._srp_torque(state, env, c_bn)

        return tau

    def _gravity_gradient_torque(self, state: StateTruth, c_bn: np.ndarray) -> np.ndarray:
        r_i_m = state.position_eci_km * 1e3
        r_norm_m = _norm3(r_i_m)
        if r_norm_m == 0.0:
            return np.zeros(3)
        r_hat_b = c_bn @ (r_i_m / r_norm_m)
        mu_m3_s2 = self.mu_km3_s2 * 1e9
        return 3.0 * mu_m3_s2 / (r_norm_m**3) * _cross3(r_hat_b, self.inertia_kg_m2 @ r_hat_b)

    def _magnetic_torque(self, state: StateTruth, c_bn: np.ndarray, env: dict) -> np.ndarray:
        if "magnetic_field_eci_t" in env:
            b_eci = np.asarray(env["magnetic_field_eci_t"], dtype=float).reshape(3)
        else:
            r_i_m = state.position_eci_km * 1e3
            r_norm_m = _norm3(r_i_m)
            if r_norm_m == 0.0:
                return np.zeros(3)

            m_eci = np.array([0.0, 0.0, EARTH_MAGNETIC_DIPOLE_T_M3])
            r_hat = r_i_m / r_norm_m
            b_eci = (3.0 * r_hat * np.dot(m_eci, r_hat) - m_eci) / (r_norm_m**3)
        b_body = c_bn @ b_eci
        return _cross3(self.config.magnetic_dipole_body_a_m2, b_body)

    def _drag_torque(self, state: StateTruth, env: dict, c_bn: np.ndarray) -> np.ndarray:
        if "density_kg_m3" in env:
            rho = float(env["density_kg_m3"])
        else:
            rho = density_from_model(
                str(env.get("atmosphere_model", "exponential")).lower(),
                state.position_eci_km,
                state.t_s,
                env=env,
            )
        if "drag_v_rel_eci_m_s" in env:
            v_rel_eci_m_s = np.asarray(env["drag_v_rel_eci_m_s"], dtype=float)
            v_norm = (
                float(env["drag_v_rel_norm_m_s"])
                if "drag_v_rel_norm_m_s" in env
                else _norm3(v_rel_eci_m_s)
            )
        else:
            omega_raw = env.get("drag_earth_rotation_rad_s", EARTH_ROT_RATE_RAD_S)
            from sim.aero.core import atmosphere_relative_velocity_eci_km_s

            v_rel_eci_km_s = atmosphere_relative_velocity_eci_km_s(
                state.position_eci_km,
                state.velocity_eci_km_s,
                t_s=float(state.t_s),
                earth_rotation_rad_s=float(EARTH_ROT_RATE_RAD_S if omega_raw is None else omega_raw),
                frame_model=str(env.get("drag_frame_model", "inertial_z")),
                jd_utc_start=env.get("jd_utc_start"),
                eop_path=env.get("drag_eop_path"),
                dut1_s=env.get("dut1_s"),
                xp_arcsec=env.get("xp_arcsec"),
                yp_arcsec=env.get("yp_arcsec"),
                dat_s=env.get("dat_s"),
                tt_minus_utc_s=env.get("tt_minus_utc_s"),
                ddpsi_rad=float(env.get("ddpsi_rad", 0.0) or 0.0),
                ddeps_rad=float(env.get("ddeps_rad", 0.0) or 0.0),
            )
            v_rel_eci_m_s = v_rel_eci_km_s * 1e3
            v_norm = _norm3(v_rel_eci_m_s)
        if v_norm == 0.0 or rho <= 0.0:
            return np.zeros(3)

        v_rel_body = c_bn @ v_rel_eci_m_s
        if self.config.geometry_area_profile is not None:
            q_dyn = 0.5 * rho * (v_norm**2) * self.config.drag_cd
            return self.config.geometry_area_profile.pressure_torque_sum_body_nm(
                -v_rel_body,
                q_dyn,
                moment_origin_body_m=self.config.center_of_mass_body_m,
            )
        if self._rect_prism_geometry is not None and self.config.use_rectangular_prism_faces:
            q_dyn = 0.5 * rho * (v_norm**2) * self.config.drag_cd
            return self._rect_prism_geometry.face_torque_sum_body_nm(
                -v_rel_body,
                q_dyn,
                moment_origin_body_m=self.config.center_of_mass_body_m,
            )

        if self.config.drag_facets:
            torque = np.zeros(3)
            for facet in self.config.drag_facets:
                torque += self._drag_facet_torque_body_nm(facet, rho, v_norm, v_rel_body, self.config.drag_cd)
            return torque

        if self.config.drag_facet_normal_body is not None:
            facet = {
                "area_m2": self.config.drag_area_m2,
                "drag_cd": self.config.drag_cd,
                "normal_body": self.config.drag_facet_normal_body,
                "cp_offset_body_m": self.config.drag_cp_offset_body_m,
            }
            return self._drag_facet_torque_body_nm(facet, rho, v_norm, v_rel_body, self.config.drag_cd)

        f_drag_mag = 0.5 * rho * (v_norm**2) * self.config.drag_cd * self.config.drag_area_m2
        f_drag_body = -f_drag_mag * (v_rel_body / v_norm)
        return _cross3(self.config.drag_cp_offset_body_m, f_drag_body)

    def _srp_torque(self, state: StateTruth, env: dict, c_bn: np.ndarray) -> np.ndarray:
        sun_dir_eci = np.asarray(
            env.get("sun_dir_eci_unit", env.get("sun_dir_eci", self.config.sun_dir_eci)), dtype=float
        )
        n = _norm3(sun_dir_eci)
        if n == 0.0:
            return np.zeros(3)
        if "sun_dir_eci_unit" not in env:
            sun_dir_eci = sun_dir_eci / n
        if "srp_shadow_factor" in env:
            shadow = float(env["srp_shadow_factor"])
        else:
            shadow = float(srp_shadow_factor(r_sc_eci_km=state.position_eci_km, t_s=state.t_s, env=env))
        if shadow <= 0.0:
            return np.zeros(3)

        sun_dir_body = c_bn @ sun_dir_eci
        distance_scale = float(env.get("srp_distance_scale", 1.0))
        p_srp_base = srp_pressure_n_m2(env) * distance_scale * self.config.srp_cr * shadow
        if self.config.geometry_area_profile is not None:
            return self.config.geometry_area_profile.pressure_torque_sum_body_nm(
                -sun_dir_body,
                p_srp_base,
                moment_origin_body_m=self.config.center_of_mass_body_m,
            )
        if self._rect_prism_geometry is not None and self.config.use_rectangular_prism_faces:
            return self._rect_prism_geometry.face_torque_sum_body_nm(
                -sun_dir_body,
                p_srp_base,
                moment_origin_body_m=self.config.center_of_mass_body_m,
            )

        if self.config.srp_facets:
            torque = np.zeros(3)
            for facet in self.config.srp_facets:
                torque += self._srp_facet_torque_body_nm(facet, p_srp_base, sun_dir_body)
            return torque

        if self.config.srp_facet_normal_body is not None:
            facet = {
                "area_m2": self.config.srp_area_m2,
                "normal_body": self.config.srp_facet_normal_body,
                "cp_offset_body_m": self.config.srp_cp_offset_body_m,
            }
            return self._srp_facet_torque_body_nm(facet, p_srp_base, sun_dir_body)

        force_mag = p_srp_base * self.config.srp_area_m2
        f_srp_body = -force_mag * sun_dir_body
        return _cross3(self.config.srp_cp_offset_body_m, f_srp_body)

    @staticmethod
    def _srp_facet_torque_body_nm(facet: dict, pressure_n_m2: float, sun_dir_body: np.ndarray) -> np.ndarray:
        normal_body = np.asarray(facet["normal_body"], dtype=float).reshape(3)
        normal_norm = _norm3(normal_body)
        if normal_norm <= 0.0:
            return np.zeros(3)
        normal_body = normal_body / normal_norm
        incident_dir_body = -sun_dir_body
        illum = max(0.0, -float(np.dot(normal_body, incident_dir_body)))
        if illum <= 0.0:
            return np.zeros(3)
        area_m2 = float(facet["area_m2"])
        cp_offset_body_m = np.asarray(facet["cp_offset_body_m"], dtype=float).reshape(3)
        force_body = pressure_n_m2 * area_m2 * illum * incident_dir_body
        return _cross3(cp_offset_body_m, force_body)

    @staticmethod
    def _drag_facet_torque_body_nm(
        facet: dict, density_kg_m3: float, v_norm_m_s: float, v_rel_body_m_s: np.ndarray, default_cd: float
    ) -> np.ndarray:
        normal_body = np.asarray(facet["normal_body"], dtype=float).reshape(3)
        normal_norm = _norm3(normal_body)
        if normal_norm <= 0.0 or v_norm_m_s <= 0.0:
            return np.zeros(3)
        normal_body = normal_body / normal_norm
        v_hat_body = np.asarray(v_rel_body_m_s, dtype=float).reshape(3) / float(v_norm_m_s)
        projected_area_m2 = float(facet["area_m2"]) * max(0.0, float(np.dot(normal_body, v_hat_body)))
        if projected_area_m2 <= 0.0:
            return np.zeros(3)
        drag_cd = float(facet.get("drag_cd", facet.get("cd", default_cd)))
        cp_offset_body_m = np.asarray(facet["cp_offset_body_m"], dtype=float).reshape(3)
        force_body = -0.5 * float(density_kg_m3) * projected_area_m2 * drag_cd * float(v_norm_m_s) ** 2 * v_hat_body
        return _cross3(cp_offset_body_m, force_body)

    @property
    def _rect_prism_geometry(self) -> RectangularPrismGeometry | None:
        dims = self.config.rectangular_prism_dims_m
        if dims is None:
            return None
        return RectangularPrismGeometry(lx_m=float(dims[0]), ly_m=float(dims[1]), lz_m=float(dims[2]))
