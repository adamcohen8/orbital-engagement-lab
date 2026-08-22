from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sim.dynamics.orbit.environment import EARTH_ROT_RATE_RAD_S
from sim.dynamics.orbit.frames import (
    FRAME_MODEL_IAU76_80_EOP,
    FRAME_MODEL_SIMPLE_GMST,
    FrameContext,
    eci_to_ecef_rotation,
    eci_to_ecef_rotation_derivative_context,
    eci_to_ecef_rotation_hpop_like,
    normalize_frame_model,
)


@dataclass(frozen=True)
class AeroState:
    density_kg_m3: float
    relative_speed_m_s: float
    dynamic_pressure_pa: float
    relative_velocity_eci_km_s: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass(frozen=True)
class AeroLoadScalars:
    drag_accel_m_s2: float
    lift_accel_m_s2: float
    lift_to_drag: float


@dataclass(frozen=True)
class VehicleAeroProperties:
    reference_area_m2: float = 1.0
    drag_area_m2: float = 1.0
    lift_area_m2: float | None = None
    cd: float = 2.2
    cl: float = 0.0
    nose_radius_m: float = 0.5
    reference_length_m: float = 1.0
    lift_axis_body: np.ndarray | None = None
    cp_offset_body_m: np.ndarray = field(default_factory=lambda: np.zeros(3))


def aero_spec_get(specs: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    nested = dict(specs.get("aero", {}) or {}) if isinstance(specs.get("aero", {}), dict) else {}
    for key in keys:
        if key in specs and specs[key] is not None:
            return specs[key]
    for key in keys:
        if key in nested and nested[key] is not None:
            return nested[key]
    return default


def _aero_spec_float(
    specs: dict[str, Any],
    keys: tuple[str, ...],
    *,
    default: float,
    min_value: float | None = 0.0,
) -> float:
    value = aero_spec_get(specs, keys, default)
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"specs.aero.{keys[0]} must be finite.")
    if min_value is not None:
        min_val = float(min_value)
        if out < min_val:
            raise ValueError(f"specs.aero.{keys[0]} must be >= {min_val}.")
    return out


def aero_spec_vector3(
    specs: dict[str, Any],
    keys: tuple[str, ...],
    *,
    default: np.ndarray | list[float] | tuple[float, float, float] | None = None,
    normalize: bool = False,
    field_name: str = "aero vector",
) -> np.ndarray | None:
    value = aero_spec_get(specs, keys, default)
    if value is None:
        return None
    arr = np.array(value, dtype=float).reshape(3)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"specs.{field_name} must contain finite values.")
    if normalize:
        norm = float(np.linalg.norm(arr))
        if norm <= 0.0:
            raise ValueError(f"specs.{field_name} must be non-zero.")
        arr = arr / norm
    return arr


def resolve_vehicle_aero_properties(
    specs: dict[str, Any],
    *,
    default_reference_area_m2: float = 1.0,
    default_cd: float = 2.2,
    default_cl: float = 0.0,
    default_nose_radius_m: float = 0.5,
    default_reference_length_m: float = 1.0,
) -> VehicleAeroProperties:
    reference_area_m2 = _aero_spec_float(
        specs,
        ("reference_area_m2", "area_ref_m2", "area_m2"),
        default=default_reference_area_m2,
    )
    drag_area_m2 = _aero_spec_float(specs, ("drag_area_m2",), default=reference_area_m2)
    lift_area_raw = aero_spec_get(specs, ("lift_area_m2",), None)
    if lift_area_raw is None:
        lift_area_m2 = None
    else:
        lift_area_m2 = float(lift_area_raw)
        if not np.isfinite(lift_area_m2):
            raise ValueError("specs.aero.lift_area_m2 must be finite.")
        if lift_area_m2 < 0.0:
            raise ValueError("specs.aero.lift_area_m2 must be >= 0.0.")
    cp_offset = aero_spec_vector3(
        specs,
        ("cp_offset_body_m", "center_of_pressure_offset_body_m"),
        default=np.zeros(3),
        field_name="aero.cp_offset_body_m",
    )
    return VehicleAeroProperties(
        reference_area_m2=reference_area_m2,
        drag_area_m2=drag_area_m2,
        lift_area_m2=lift_area_m2,
        cd=_aero_spec_float(specs, ("cd", "drag_cd"), default=default_cd),
        cl=_aero_spec_float(
            specs,
            ("cl", "lift_coefficient", "coefficient_of_lift"),
            default=default_cl,
            min_value=None,
        ),
        nose_radius_m=_aero_spec_float(
            specs,
            ("reentry_nose_radius_m", "nose_radius_m"),
            default=default_nose_radius_m,
            min_value=1.0e-9,
        ),
        reference_length_m=_aero_spec_float(
            specs,
            ("reference_length_m",),
            default=default_reference_length_m,
            min_value=1.0e-12,
        ),
        lift_axis_body=aero_spec_vector3(
            specs,
            ("lift_axis_body", "lift_vector_body"),
            normalize=True,
            field_name="aero.lift_axis_body",
        ),
        cp_offset_body_m=np.zeros(3) if cp_offset is None else cp_offset,
    )


def atmosphere_relative_velocity_eci_km_s(
    r_eci_km: np.ndarray,
    v_eci_km_s: np.ndarray,
    *,
    t_s: float = 0.0,
    earth_rotation_rad_s: float = EARTH_ROT_RATE_RAD_S,
    frame_model: str = "inertial_z",
    jd_utc_start: float | None = None,
    eop_path: str | None = None,
    dut1_s: float | None = None,
    xp_arcsec: float | None = None,
    yp_arcsec: float | None = None,
    dat_s: float | None = None,
    tt_minus_utc_s: float | None = None,
    ddpsi_rad: float = 0.0,
    ddeps_rad: float = 0.0,
    eop_extrapolation: str = "error",
) -> np.ndarray:
    r = (
        r_eci_km
        if isinstance(r_eci_km, np.ndarray) and r_eci_km.dtype == np.float64 and r_eci_km.shape == (3,)
        else np.asarray(r_eci_km, dtype=float).reshape(3)
    )
    v = (
        v_eci_km_s
        if isinstance(v_eci_km_s, np.ndarray) and v_eci_km_s.dtype == np.float64 and v_eci_km_s.shape == (3,)
        else np.asarray(v_eci_km_s, dtype=float).reshape(3)
    )
    model = normalize_frame_model(frame_model)
    if model in {FRAME_MODEL_SIMPLE_GMST, FRAME_MODEL_IAU76_80_EOP}:
        if model == FRAME_MODEL_IAU76_80_EOP:
            rot = eci_to_ecef_rotation_hpop_like(
                float(t_s),
                jd_utc_start=None if jd_utc_start is None else float(jd_utc_start),
                eop_path=None if eop_path is None else str(eop_path),
                dut1_s=dut1_s,
                xp_arcsec=xp_arcsec,
                yp_arcsec=yp_arcsec,
                dat_s=dat_s,
                tt_minus_utc_s=tt_minus_utc_s,
                ddpsi_rad=ddpsi_rad,
                ddeps_rad=ddeps_rad,
                eop_extrapolation=eop_extrapolation,
            )
        else:
            rot = eci_to_ecef_rotation(
                float(t_s),
                jd_utc_start=None if jd_utc_start is None else float(jd_utc_start),
            )
        r_frame = rot @ r
        if model == FRAME_MODEL_IAU76_80_EOP:
            context = FrameContext(
                model=model,
                jd_utc_start=None if jd_utc_start is None else float(jd_utc_start),
                eop_path=None if eop_path is None else str(eop_path),
                eop_extrapolation=eop_extrapolation,
                tt_minus_utc_s=(
                    69.184 if tt_minus_utc_s is None else float(tt_minus_utc_s)
                ),
                dut1_s=dut1_s,
                xp_arcsec=xp_arcsec,
                yp_arcsec=yp_arcsec,
                dat_s=dat_s,
                ddpsi_rad=ddpsi_rad,
                ddeps_rad=ddeps_rad,
                source="atmosphere_relative_velocity",
            )
            rot_dot = eci_to_ecef_rotation_derivative_context(float(t_s), context)
            # A stationary atmosphere has zero ECEF velocity.  The canonical
            # ECI->ECEF state transform therefore gives the complete relative
            # velocity, including precession/nutation/polar-motion derivatives.
            return rot.T @ (rot @ v + rot_dot @ r)
        v_frame = rot @ v
        v_atm_frame_km_s = np.array(
            [
                -float(earth_rotation_rad_s) * float(r_frame[1]),
                float(earth_rotation_rad_s) * float(r_frame[0]),
                0.0,
            ],
            dtype=float,
        )
        return rot.T @ (v_frame - v_atm_frame_km_s)

    v_atm_eci_km_s = np.array(
        [
            -float(earth_rotation_rad_s) * float(r[1]),
            float(earth_rotation_rad_s) * float(r[0]),
            0.0,
        ],
        dtype=float,
    )
    return v - v_atm_eci_km_s


def dynamic_pressure_pa(density_kg_m3: float, speed_m_s: float) -> float:
    return float(0.5 * max(float(density_kg_m3), 0.0) * max(float(speed_m_s), 0.0) ** 2)


def compute_aero_load_scalars(
    *,
    density_kg_m3: float,
    speed_m_s: float,
    mass_kg: float,
    drag_area_m2: float,
    cd: float,
    lift_area_m2: float | None = None,
    cl: float = 0.0,
) -> AeroLoadScalars:
    q_dyn_pa = dynamic_pressure_pa(density_kg_m3, speed_m_s)
    mass = max(float(mass_kg), 1e-12)
    drag_accel = q_dyn_pa * max(float(cd), 0.0) * max(float(drag_area_m2), 0.0) / mass
    lift_area = max(float(drag_area_m2), 0.0) if lift_area_m2 is None else max(float(lift_area_m2), 0.0)
    lift_accel = q_dyn_pa * lift_area * abs(float(cl)) / mass
    lift_to_drag = float("nan") if drag_accel <= 0.0 else float(lift_accel / drag_accel)
    return AeroLoadScalars(
        drag_accel_m_s2=float(drag_accel),
        lift_accel_m_s2=float(lift_accel),
        lift_to_drag=lift_to_drag,
    )


def sutton_graves_heat_rate_w_m2(
    *,
    density_kg_m3: float,
    speed_m_s: float,
    nose_radius_m: float,
    coefficient: float,
) -> float:
    rho = max(float(density_kg_m3), 0.0)
    radius = max(float(nose_radius_m), 1e-9)
    return float(coefficient) * float(np.sqrt(rho / radius)) * max(float(speed_m_s), 0.0) ** 3
