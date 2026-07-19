from __future__ import annotations

from sim.config.scenario.models import (
    SimulationScenarioConfig,
)
from sim.config.scenario.primitives import (
    _parse_bool,
)

__all__ = [
    '_validate_physics_runtime_settings',
    '_validate_object_references',
]

def _validate_physics_runtime_settings(cfg: SimulationScenarioConfig) -> None:
    orbit = dict((cfg.simulator.dynamics or {}).get("orbit", {}) or {})
    reentry = dict((cfg.simulator.dynamics or {}).get("reentry", {}) or {})
    propagation_method = str(orbit.get("propagation_method", "special") or "special").strip().lower()
    if propagation_method not in {"special", "general"}:
        raise ValueError("simulator.dynamics.orbit.propagation_method must be one of: special, general.")
    integrator = str(orbit.get("integrator", "rk4") or "rk4").strip().lower()
    if integrator not in {"rk4", "rkf78", "dopri5", "adaptive"}:
        raise ValueError("simulator.dynamics.orbit.integrator must be one of: adaptive, dopri5, rk4, rkf78.")

    env = dict(cfg.simulator.environment or {})
    if _parse_bool(reentry.get("enabled", False), "simulator.dynamics.reentry.enabled"):
        if not _parse_bool(orbit.get("drag", False), "simulator.dynamics.orbit.drag"):
            raise ValueError(
                "simulator.dynamics.reentry.enabled requires simulator.dynamics.orbit.drag=true "
                "so reported atmospheric loads and heating use a trajectory propagated with drag."
            )
        reentry_atmosphere = str(reentry.get("atmosphere_model", "") or "").strip().lower().replace("-", "_")
        environment_atmosphere = str(env.get("atmosphere_model", "") or "").strip().lower().replace("-", "_")
        atmosphere_aliases = {
            "hp": "harris_priester",
            "hpop_harris_priester": "harris_priester",
            "hpop_msis86": "msis86",
            "hpop_jacchia70": "jacchia70",
        }
        reentry_atmosphere = atmosphere_aliases.get(reentry_atmosphere, reentry_atmosphere)
        environment_atmosphere = atmosphere_aliases.get(environment_atmosphere, environment_atmosphere)
        if (
            env.get("density_kg_m3") is None
            and reentry_atmosphere
            and environment_atmosphere
            and reentry_atmosphere != environment_atmosphere
        ):
            raise ValueError(
                "simulator.dynamics.reentry.atmosphere_model must match "
                "simulator.environment.atmosphere_model. Configure the shared atmospheric model under "
                "simulator.environment; the reentry field is a compatibility alias only."
            )
    if _parse_bool(orbit.get("drag", False), "simulator.dynamics.orbit.drag") or _parse_bool(
        orbit.get("lift", False), "simulator.dynamics.orbit.lift"
    ):
        if env.get("atmosphere_model") in (None, "") and env.get("density_kg_m3") is None:
            raise ValueError(
                "simulator.dynamics.orbit drag/lift requires simulator.environment.atmosphere_model "
                "or density_kg_m3; atmosphere selection is explicit."
            )
    ephemeris_mode = env.get("ephemeris_mode")
    if ephemeris_mode not in (None, ""):
        mode = str(ephemeris_mode).strip().lower()
        if mode not in {
            "analytic_enhanced",
            "enhanced",
            "analytic_simple",
            "simple",
            "de440",
            "hpop_de440",
            "de440_hpop",
            "spice",
            "spiceypy",
        }:
            raise ValueError(
                "simulator.environment.ephemeris_mode must be one of: analytic_enhanced, analytic_simple, "
                "de440, hpop_de440, de440_hpop, spice, spiceypy."
            )

    model = str(orbit.get("model", "two_body") or "two_body").strip().lower()
    if model not in {"two_body", "cr3bp"}:
        raise ValueError("simulator.dynamics.orbit.model must be one of: cr3bp, two_body.")
    if model == "cr3bp":
        unsupported = []
        for key in ("j2", "j3", "j4", "drag", "srp", "third_body_sun", "third_body_moon", "lift"):
            if _parse_bool(orbit.get(key, False), f"simulator.dynamics.orbit.{key}"):
                unsupported.append(key)
        if unsupported:
            raise ValueError(
                "simulator.dynamics.orbit.model=cr3bp does not support two-body perturbation flags: "
                + ", ".join(sorted(unsupported))
                + "."
            )

    sh = dict(orbit.get("spherical_harmonics", {}) or {})
    if _parse_bool(sh.get("enabled", False), "simulator.dynamics.orbit.spherical_harmonics.enabled"):
        degree = int(sh.get("degree", 0) or 0)
        source = str(sh.get("source", sh.get("model", "")) or "").strip().lower()
        terms = list(sh.get("terms", []) or [])
        has_terms = bool(terms)
        has_path = sh.get("coeff_path") not in (None, "") or sh.get("source_path") not in (None, "")
        if has_terms and (has_path or source):
            raise ValueError(
                "simulator.dynamics.orbit.spherical_harmonics must use either inline terms or a coefficient source, "
                "not both."
            )
        if not has_terms and degree < 2:
            raise ValueError(
                "File-backed simulator.dynamics.orbit.spherical_harmonics requires degree >= 2; "
                "inline terms infer degree and order when omitted."
            )
        supported_sources = {"hpop", "hpop_ggm03", "ggm03", "icgem", "gfc", "egm96"}
        if source and source not in supported_sources:
            choices = ", ".join(sorted(supported_sources))
            raise ValueError(
                f"simulator.dynamics.orbit.spherical_harmonics.source must be one of: {choices}."
            )
        if not has_terms and not source and not has_path:
            raise ValueError(
                "simulator.dynamics.orbit.spherical_harmonics.enabled requires inline terms or a supported "
                "coefficient source/path; degree and order alone do not define a gravity field."
            )
        if source in {"icgem", "gfc"} and not has_path:
            raise ValueError(
                "ICGEM spherical harmonics require spherical_harmonics.coeff_path or source_path."
            )
def _validate_object_references(cfg: SimulationScenarioConfig) -> None:
    objects = dict(cfg.objects or {})
    for object_id, section in objects.items():
        initial_state = dict(getattr(section, "initial_state", {}) or {})
        reference_id = str(initial_state.get("relative_to", "") or "").strip()
        if not reference_id:
            continue
        path = f"objects.{object_id}.initial_state.relative_to"
        if reference_id == str(object_id):
            raise ValueError(f"{path} cannot reference the same object.")
        reference = objects.get(reference_id)
        if reference is None:
            raise ValueError(f"{path} references unknown object '{reference_id}'.")
        if not bool(getattr(reference, "enabled", False)):
            raise ValueError(f"{path} references disabled object '{reference_id}'.")
