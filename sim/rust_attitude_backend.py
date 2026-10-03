"""Optional Rust attitude and disturbance kernels.

The Python attitude implementation remains the reference path.  This module
only crosses into Rust after a caller explicitly selects ``numeric_backend=
"rust"``; custom disturbance models and unsupported geometry still use their
existing Python implementations.
"""

from __future__ import annotations

import math
import struct

import numpy as np

from sim.rust_orbit_backend import _extension


def _fixed(value: np.ndarray | list[float], shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    return np.ascontiguousarray(array)


def _flat(value: np.ndarray | list[float], shape: tuple[int, ...], name: str) -> list[float]:
    return _fixed(value, shape, name).reshape(-1).tolist()


def _native_counts(values: list[int] | np.ndarray) -> np.ndarray:
    counts = np.asarray(values, dtype=np.int64).reshape(-1)
    if counts.size != 6:
        raise RuntimeError("Rust attitude kernel returned invalid guardrail counts")
    return counts.copy()


def propagate_attitude_exponential_map(
    quat_bn: np.ndarray,
    omega_body_rad_s: np.ndarray,
    inertia_kg_m2: np.ndarray,
    torque_body_nm: np.ndarray,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Advance one attitude exponential-map step in Rust."""

    if not math.isfinite(float(dt_s)):
        raise ValueError("dt_s must be finite")
    native = _extension()
    packed = getattr(native, "attitude_step_packed", None)
    if packed is not None:
        data = b"".join((
            _fixed(quat_bn, (4,), "quaternion").astype("<f8", copy=False).tobytes(),
            _fixed(omega_body_rad_s, (3,), "body rate").astype("<f8", copy=False).tobytes(),
            _fixed(inertia_kg_m2, (3, 3), "inertia").astype("<f8", copy=False).tobytes(),
            _fixed(torque_body_nm, (3,), "torque").astype("<f8", copy=False).tobytes(),
        ))
        result, counts = packed(data, float(dt_s))
        values = np.frombuffer(result, dtype="<f8").copy()
        return values[:4], values[4:], counts
    function = getattr(native, "attitude_propagate_exponential_map", None)
    if function is None:
        raise RuntimeError("Rust attitude backend requires an oel_rust_orbit wheel with attitude kernels")
    q_next, omega_next, counts = function(
        _flat(quat_bn, (4,), "quaternion"),
        _flat(omega_body_rad_s, (3,), "body rate"),
        _flat(inertia_kg_m2, (3, 3), "inertia"),
        _flat(torque_body_nm, (3,), "torque"),
        float(dt_s),
    )
    return (
        _fixed(q_next, (4,), "Rust quaternion"),
        _fixed(omega_next, (3,), "Rust body rate"),
        _native_counts(counts),
    )


def rigid_body_derivatives(
    quat_bn: np.ndarray,
    omega_body_rad_s: np.ndarray,
    inertia_kg_m2: np.ndarray,
    torque_body_nm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate rigid-body quaternion/rate derivatives in Rust."""

    native = _extension()
    packed = getattr(native, "attitude_step_packed", None)
    if packed is not None:
        data = b"".join((
            _fixed(quat_bn, (4,), "quaternion").astype("<f8", copy=False).tobytes(),
            _fixed(omega_body_rad_s, (3,), "body rate").astype("<f8", copy=False).tobytes(),
            _fixed(inertia_kg_m2, (3, 3), "inertia").astype("<f8", copy=False).tobytes(),
            _fixed(torque_body_nm, (3,), "torque").astype("<f8", copy=False).tobytes(),
        ))
        result, counts = packed(data, None)
        values = np.frombuffer(result, dtype="<f8").copy()
        return values[:4], values[4:], counts
    function = getattr(native, "attitude_rigid_body_derivatives", None)
    if function is None:
        raise RuntimeError("Rust attitude backend requires an oel_rust_orbit wheel with attitude kernels")
    q_dot, omega_dot, counts = function(
        _flat(quat_bn, (4,), "quaternion"),
        _flat(omega_body_rad_s, (3,), "body rate"),
        _flat(inertia_kg_m2, (3, 3), "inertia"),
        _flat(torque_body_nm, (3,), "torque"),
    )
    return (
        _fixed(q_dot, (4,), "Rust quaternion derivative"),
        _fixed(omega_dot, (3,), "Rust body rate derivative"),
        _native_counts(counts),
    )


def _facet_buffers(
    normals: np.ndarray,
    areas: np.ndarray,
    coefficients: np.ndarray | None,
    offsets: np.ndarray,
    *,
    name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    normals_array = np.asarray(normals, dtype=np.float64)
    areas_array = np.asarray(areas, dtype=np.float64).reshape(-1)
    offsets_array = np.asarray(offsets, dtype=np.float64)
    if normals_array.ndim != 2 or normals_array.shape[1] != 3:
        raise ValueError(f"{name} normals must have shape (n, 3)")
    if offsets_array.ndim != 2 or offsets_array.shape != normals_array.shape:
        raise ValueError(f"{name} offsets must have shape {normals_array.shape}")
    if areas_array.shape != (normals_array.shape[0],):
        raise ValueError(f"{name} areas must have shape ({normals_array.shape[0]},)")
    if coefficients is None:
        coefficient_array = np.ones_like(areas_array)
    else:
        coefficient_array = np.asarray(coefficients, dtype=np.float64).reshape(-1)
        if coefficient_array.shape != areas_array.shape:
            raise ValueError(f"{name} coefficients must have shape {areas_array.shape}")
    return (
        normals_array,
        areas_array,
        coefficient_array,
        offsets_array,
    )


def _facet_arrays(normals, areas, coefficients, offsets, *, name):
    buffers = _facet_buffers(normals, areas, coefficients, offsets, name=name)
    return tuple(value.reshape(-1).tolist() for value in buffers)


def builtin_disturbance_torque(
    *,
    quat_bn: np.ndarray,
    position_eci_km: np.ndarray,
    inertia_kg_m2: np.ndarray,
    mu_km3_s2: float,
    enabled: np.ndarray,
    magnetic_dipole_body_a_m2: np.ndarray,
    magnetic_field_eci_t: np.ndarray,
    magnetic_field_provided: bool,
    density_kg_m3: float,
    drag_v_rel_eci_m_s: np.ndarray,
    drag_v_rel_norm_m_s: float,
    drag_mode: int,
    drag_area_m2: float,
    drag_cd: float,
    drag_cp_offset_body_m: np.ndarray,
    drag_facet_normals_body: np.ndarray,
    drag_facet_areas_m2: np.ndarray,
    drag_facet_cd: np.ndarray,
    drag_facet_cp_offsets_body_m: np.ndarray,
    sun_dir_eci_unit: np.ndarray,
    srp_pressure_scaled_n_m2: float,
    srp_mode: int,
    srp_area_m2: float,
    srp_cp_offset_body_m: np.ndarray,
    srp_facet_normals_body: np.ndarray,
    srp_facet_areas_m2: np.ndarray,
    srp_facet_cp_offsets_body_m: np.ndarray,
) -> np.ndarray:
    """Evaluate the ordered built-in disturbance torque sum in Rust."""

    enabled_array = np.asarray(enabled, dtype=np.int64).reshape(-1)
    if enabled_array.shape != (4,):
        raise ValueError("enabled must have shape (4,)")
    drag_normals, drag_areas, drag_coefficients, drag_offsets = _facet_arrays(
        drag_facet_normals_body,
        drag_facet_areas_m2,
        drag_facet_cd,
        drag_facet_cp_offsets_body_m,
        name="drag",
    )
    srp_normals, srp_areas, _, srp_offsets = _facet_arrays(
        srp_facet_normals_body,
        srp_facet_areas_m2,
        None,
        srp_facet_cp_offsets_body_m,
        name="SRP",
    )
    native = _extension()
    function = getattr(native, "attitude_builtin_disturbance_torque", None)
    if function is None:
        raise RuntimeError("Rust attitude backend requires an oel_rust_orbit wheel with disturbance kernels")
    result = function(
        _flat(quat_bn, (4,), "quaternion"),
        _flat(position_eci_km, (3,), "position"),
        _flat(inertia_kg_m2, (3, 3), "inertia"),
        float(mu_km3_s2),
        enabled_array.tolist(),
        _flat(magnetic_dipole_body_a_m2, (3,), "magnetic dipole"),
        _flat(magnetic_field_eci_t, (3,), "magnetic field"),
        bool(magnetic_field_provided),
        float(density_kg_m3),
        _flat(drag_v_rel_eci_m_s, (3,), "drag velocity"),
        float(drag_v_rel_norm_m_s),
        int(drag_mode),
        float(drag_area_m2),
        float(drag_cd),
        _flat(drag_cp_offset_body_m, (3,), "drag center of pressure"),
        drag_normals,
        drag_areas,
        drag_coefficients,
        drag_offsets,
        _flat(sun_dir_eci_unit, (3,), "sun direction"),
        float(srp_pressure_scaled_n_m2),
        int(srp_mode),
        float(srp_area_m2),
        _flat(srp_cp_offset_body_m, (3,), "SRP center of pressure"),
        srp_normals,
        srp_areas,
        srp_offsets,
    )
    return _fixed(result, (3,), "Rust disturbance torque")


def propagate_builtin_disturbances(
    *,
    quat_bn: np.ndarray,
    omega_body_rad_s: np.ndarray,
    inertia_kg_m2: np.ndarray,
    command_torque_body_nm: np.ndarray,
    substeps_s: np.ndarray,
    position_eci_km: np.ndarray,
    mu_km3_s2: float,
    enabled: np.ndarray,
    magnetic_dipole_body_a_m2: np.ndarray,
    magnetic_field_eci_t: np.ndarray,
    magnetic_field_provided: bool,
    density_kg_m3: float,
    drag_v_rel_eci_m_s: np.ndarray,
    drag_v_rel_norm_m_s: float,
    drag_mode: int,
    drag_area_m2: float,
    drag_cd: float,
    drag_cp_offset_body_m: np.ndarray,
    drag_facet_normals_body: np.ndarray,
    drag_facet_areas_m2: np.ndarray,
    drag_facet_cd: np.ndarray,
    drag_facet_cp_offsets_body_m: np.ndarray,
    sun_dir_eci_unit: np.ndarray,
    srp_pressure_scaled_n_m2: float,
    srp_mode: int,
    srp_area_m2: float,
    srp_cp_offset_body_m: np.ndarray,
    srp_facet_normals_body: np.ndarray,
    srp_facet_areas_m2: np.ndarray,
    srp_facet_cp_offsets_body_m: np.ndarray,
    prepared_context=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Propagate all built-in disturbance substeps through one Rust call."""

    if prepared_context is not None:
        data = b"".join((
            _fixed(quat_bn, (4,), "quaternion").astype("<f8", copy=False).tobytes(),
            _fixed(omega_body_rad_s, (3,), "body rate").astype("<f8", copy=False).tobytes(),
            _fixed(command_torque_body_nm, (3,), "command torque").astype("<f8", copy=False).tobytes(),
            _fixed(position_eci_km, (3,), "position").astype("<f8", copy=False).tobytes(),
            _fixed(magnetic_field_eci_t, (3,), "magnetic field").astype("<f8", copy=False).tobytes(),
            struct.pack("<dd", bool(magnetic_field_provided), float(density_kg_m3)),
            _fixed(drag_v_rel_eci_m_s, (3,), "drag velocity").astype("<f8", copy=False).tobytes(),
            struct.pack("<d", float(drag_v_rel_norm_m_s)),
            _fixed(sun_dir_eci_unit, (3,), "sun direction").astype("<f8", copy=False).tobytes(),
            struct.pack("<d", float(srp_pressure_scaled_n_m2)),
        ))
        raw, counts = prepared_context.step(data, np.asarray(substeps_s, dtype="<f8").reshape(-1).tobytes())
        values = np.frombuffer(raw, dtype="<f8").copy()
        return values[:4], values[4:], counts
    enabled_array = np.asarray(enabled, dtype=np.int64).reshape(-1)
    if enabled_array.shape != (4,):
        raise ValueError("enabled must have shape (4,)")
    substeps = np.asarray(substeps_s, dtype=np.float64).reshape(-1)
    native = _extension()
    packed = getattr(native, "attitude_propagate_disturbances_packed", None)
    if packed is not None:
        drag = _facet_buffers(drag_facet_normals_body, drag_facet_areas_m2, drag_facet_cd,
                              drag_facet_cp_offsets_body_m, name="drag")
        srp = _facet_buffers(srp_facet_normals_body, srp_facet_areas_m2, None,
                             srp_facet_cp_offsets_body_m, name="SRP")
        def buffer(value):
            return value.tobytes() if np.little_endian else value.astype("<f8", copy=False).tobytes()

        data = b"".join((
            buffer(_fixed(quat_bn, (4,), "quaternion")), buffer(_fixed(omega_body_rad_s, (3,), "body rate")),
            buffer(_fixed(inertia_kg_m2, (3, 3), "inertia")), buffer(_fixed(command_torque_body_nm, (3,), "command torque")),
            buffer(_fixed(position_eci_km, (3,), "position")), struct.pack("<d", float(mu_km3_s2)),
            buffer(enabled_array.astype(np.float64)), buffer(_fixed(magnetic_dipole_body_a_m2, (3,), "magnetic dipole")),
            buffer(_fixed(magnetic_field_eci_t, (3,), "magnetic field")),
            struct.pack("<dd", bool(magnetic_field_provided), float(density_kg_m3)),
            buffer(_fixed(drag_v_rel_eci_m_s, (3,), "drag velocity")),
            struct.pack("<dddd", float(drag_v_rel_norm_m_s), int(drag_mode), float(drag_area_m2), float(drag_cd)),
            buffer(_fixed(drag_cp_offset_body_m, (3,), "drag center of pressure")),
            buffer(_fixed(sun_dir_eci_unit, (3,), "sun direction")),
            struct.pack("<ddd", float(srp_pressure_scaled_n_m2), int(srp_mode), float(srp_area_m2)),
            buffer(_fixed(srp_cp_offset_body_m, (3,), "SRP center of pressure")),
            *(buffer(value) for value in drag), buffer(srp[0]), buffer(srp[1]), buffer(srp[3]),
        ))
        raw, counts = packed(data, np.asarray(substeps, dtype="<f8").tobytes(), int(drag[1].size), int(srp[1].size))
        values = np.frombuffer(raw, dtype="<f8").copy()
        return values[:4], values[4:], counts
    drag_normals, drag_areas, drag_coefficients, drag_offsets = _facet_arrays(
        drag_facet_normals_body,
        drag_facet_areas_m2,
        drag_facet_cd,
        drag_facet_cp_offsets_body_m,
        name="drag",
    )
    srp_normals, srp_areas, _, srp_offsets = _facet_arrays(
        srp_facet_normals_body,
        srp_facet_areas_m2,
        None,
        srp_facet_cp_offsets_body_m,
        name="SRP",
    )
    native = _extension()
    function = getattr(native, "attitude_propagate_builtin_disturbances", None)
    if function is None:
        raise RuntimeError("Rust attitude backend requires an oel_rust_orbit wheel with disturbance kernels")
    q_next, omega_next, counts = function(
        _flat(quat_bn, (4,), "quaternion"),
        _flat(omega_body_rad_s, (3,), "body rate"),
        _flat(inertia_kg_m2, (3, 3), "inertia"),
        _flat(command_torque_body_nm, (3,), "command torque"),
        substeps.tolist(),
        _flat(position_eci_km, (3,), "position"),
        float(mu_km3_s2),
        enabled_array.tolist(),
        _flat(magnetic_dipole_body_a_m2, (3,), "magnetic dipole"),
        _flat(magnetic_field_eci_t, (3,), "magnetic field"),
        bool(magnetic_field_provided),
        float(density_kg_m3),
        _flat(drag_v_rel_eci_m_s, (3,), "drag velocity"),
        float(drag_v_rel_norm_m_s),
        int(drag_mode),
        float(drag_area_m2),
        float(drag_cd),
        _flat(drag_cp_offset_body_m, (3,), "drag center of pressure"),
        drag_normals,
        drag_areas,
        drag_coefficients,
        drag_offsets,
        _flat(sun_dir_eci_unit, (3,), "sun direction"),
        float(srp_pressure_scaled_n_m2),
        int(srp_mode),
        float(srp_area_m2),
        _flat(srp_cp_offset_body_m, (3,), "SRP center of pressure"),
        srp_normals,
        srp_areas,
        srp_offsets,
    )
    return (
        _fixed(q_next, (4,), "Rust quaternion"),
        _fixed(omega_next, (3,), "Rust body rate"),
        _native_counts(counts),
    )


def prepare_builtin_disturbance_context(
    *, inertia_kg_m2, mu_km3_s2, enabled, magnetic_dipole_body_a_m2,
    drag_mode, drag_area_m2, drag_cd, drag_cp_offset_body_m,
    drag_facet_normals_body, drag_facet_areas_m2, drag_facet_cd, drag_facet_cp_offsets_body_m,
    srp_mode, srp_area_m2, srp_cp_offset_body_m,
    srp_facet_normals_body, srp_facet_areas_m2, srp_facet_cp_offsets_body_m,
):
    """Validate and retain one immutable native built-in torque configuration."""
    factory = getattr(_extension(), "BuiltinDisturbanceContext", None)
    if factory is None:
        return None
    drag = _facet_buffers(drag_facet_normals_body, drag_facet_areas_m2, drag_facet_cd,
                          drag_facet_cp_offsets_body_m, name="drag")
    srp = _facet_buffers(srp_facet_normals_body, srp_facet_areas_m2, None,
                         srp_facet_cp_offsets_body_m, name="SRP")
    fixed = np.zeros(54, dtype="<f8")
    fixed[7:16] = _fixed(inertia_kg_m2, (3, 3), "inertia").reshape(-1)
    fixed[22] = float(mu_km3_s2)
    enabled_array = np.asarray(enabled, dtype=np.int64).reshape(-1)
    if enabled_array.shape != (4,):
        raise ValueError("enabled must have shape (4,)")
    fixed[23:27] = enabled_array
    fixed[27:30] = _fixed(magnetic_dipole_body_a_m2, (3,), "magnetic dipole")
    fixed[39:42] = [int(drag_mode), float(drag_area_m2), float(drag_cd)]
    fixed[42:45] = _fixed(drag_cp_offset_body_m, (3,), "drag center of pressure")
    fixed[49:51] = [int(srp_mode), float(srp_area_m2)]
    fixed[51:54] = _fixed(srp_cp_offset_body_m, (3,), "SRP center of pressure")
    data = fixed.tobytes() + b"".join(value.astype("<f8", copy=False).tobytes()
                                    for value in (*drag, srp[0], srp[1], srp[3]))
    return factory(data, int(drag[1].size), int(srp[1].size))
