"""Public dynamics API with lazy implementation imports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "OrbitalAttitudeDynamics": ("sim.dynamics.model", "OrbitalAttitudeDynamics"),
    "GeometryAreaProfile": ("sim.dynamics.spacecraft_geometry", "GeometryAreaProfile"),
    "GeometryProfileLookup": ("sim.dynamics.spacecraft_geometry", "GeometryProfileLookup"),
    "RectangularPrismGeometry": ("sim.dynamics.spacecraft_geometry", "RectangularPrismGeometry"),
}

for _name in (
    "EARTH_MU_KM3_S2",
    "EARTH_RADIUS_KM",
    "EARTH_J2",
    "EARTH_J3",
    "EARTH_J4",
    "CR3BPSystem",
    "EARTH_MOON_CR3BP",
    "EARTH_MOON_DISTANCE_KM",
    "EARTH_MOON_MEAN_MOTION_RAD_S",
    "EARTH_MOON_MU",
    "AtmosphereModelName",
    "SphericalHarmonicTerm",
    "OrbitContext",
    "OrbitPropagator",
    "cr3bp_system",
    "cr3bp_l1_position_km",
    "cr3bp_l1_state_km_s",
    "cr3bp_halo_seed_state_km_s",
    "propagate_cr3bp_state",
    "density_exponential",
    "density_ussa1976",
    "density_msis86",
    "density_nrlmsise00",
    "density_jacchia70",
    "density_jb2006",
    "density_jb2008",
    "density_harris_priester",
    "density_from_model",
    "parse_spherical_harmonic_terms",
    "accel_spherical_harmonics_terms",
    "load_hpop_ggm03_terms",
    "load_icgem_gfc_terms",
    "load_real_earth_gravity_terms",
    "j2_plugin",
    "j3_plugin",
    "j4_plugin",
    "spherical_harmonics_plugin",
    "drag_plugin",
    "srp_plugin",
    "third_body_moon_plugin",
    "third_body_sun_plugin",
):
    _EXPORTS[_name] = ("sim.dynamics.orbit", _name)

del _name

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
