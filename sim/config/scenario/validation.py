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
    '_validate_orbital_analysis_references',
]


def _validate_orbital_analysis_references(cfg: SimulationScenarioConfig) -> None:
    section = cfg.outputs.orbital_analysis
    if not bool(section.enabled):
        return
    if cfg.simulator.initial_jd_utc is None:
        raise ValueError("outputs.orbital_analysis requires simulator.initial_jd_utc.")
    if not section.coverage and not section.directed_links:
        raise ValueError("outputs.orbital_analysis.enabled requires coverage or directed_links entries.")
    objects = dict(cfg.objects or {})

    def require_object(object_id: str, path: str) -> object:
        item = objects.get(object_id)
        if item is None:
            raise ValueError(f"{path} references unknown object {object_id!r}.")
        if not bool(getattr(item, "enabled", False)):
            raise ValueError(f"{path} references disabled object {object_id!r}.")
        return item

    attitude_enabled = bool(
        dict(dict(cfg.simulator.dynamics or {}).get("attitude", {}) or {}).get("enabled", True)
    )
    propagation_method = str(
        dict(dict(cfg.simulator.dynamics or {}).get("orbit", {}) or {}).get(
            "propagation_method", "special"
        )
        or "special"
    ).strip().lower()

    def require_achieved_attitude(item: object, path: str) -> None:
        trajectory_only = str(getattr(item, "runtime_profile", "") or "").strip().lower() == "trajectory_only"
        if propagation_method == "general":
            raise ValueError(
                f"{path} requires achieved attitude, but general OGP propagation retains only static "
                "initial attitude. Use special ONP propagation or an attitude-independent product."
            )
        if not attitude_enabled or trajectory_only:
            raise ValueError(f"{path} requires achieved attitude from enabled non-trajectory-only dynamics.")

    for index, item in enumerate(section.coverage):
        path = f"outputs.orbital_analysis.coverage[{index}].source_object_id"
        source = require_object(str(item["source_object_id"]), path)
        require_achieved_attitude(source, path)
    for index, item in enumerate(section.directed_links):
        for endpoint in ("tx", "rx"):
            object_path = f"outputs.orbital_analysis.directed_links[{index}].{endpoint}_object_id"
            endpoint_object = require_object(str(item[f"{endpoint}_object_id"]), object_path)
            terminal = dict(item[f"{endpoint}_terminal"] or {})
            pattern = dict(terminal.get("pattern", {}) or {})
            if str(pattern.get("kind", "constant") or "constant").lower() != "constant":
                require_achieved_attitude(endpoint_object, object_path)

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
    if not objects:
        objects = {
            "rocket": cfg.rocket,
            "chaser": cfg.chaser,
            "target": cfg.target,
        }
    relative_forms = {
        "relative_to_target_ric",
        "relative_ric_rect",
        "relative_ric_curv",
        "relative_to_target_cislunar",
        "relative_cislunar",
    }
    enabled_rockets = [
        object_id
        for object_id, section in objects.items()
        if bool(getattr(section, "enabled", False)) and str(getattr(section, "kind", "")).strip().lower() == "rocket"
    ]
    relative_dependencies: dict[str, str] = {}
    for object_id, section in objects.items():
        initial_state = dict(getattr(section, "initial_state", {}) or {})
        if initial_state.get("source") in {"rocket_deployment", "rocket_insertion"} and not enabled_rockets:
            raise ValueError(
                f"objects.{object_id}.initial_state.source requires at least one enabled rocket object."
            )
        reference_id = str(initial_state.get("relative_to", "") or "").strip()
        selected_forms = relative_forms.intersection(initial_state)
        if not reference_id and selected_forms:
            target_defaulted = (
                bool(selected_forms.intersection({"relative_to_target_ric", "relative_to_target_cislunar"}))
                or str(object_id) == "chaser"
            )
            if target_defaulted and "target" in objects:
                reference_id = "target"
            else:
                path = f"objects.{object_id}.initial_state.relative_to"
                raise ValueError(f"{path} is required for relative initial-state form '{sorted(selected_forms)[0]}'.")
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
        if selected_forms:
            relative_dependencies[str(object_id)] = reference_id

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(object_id: str, chain: list[str]) -> None:
        if object_id in visited:
            return
        if object_id in visiting:
            start = chain.index(object_id)
            cycle = chain[start:] + [object_id]
            raise ValueError("Relative initial-state reference cycle: " + " -> ".join(cycle))
        visiting.add(object_id)
        reference_id = relative_dependencies.get(object_id)
        if reference_id in relative_dependencies:
            visit(str(reference_id), [*chain, str(reference_id)])
        visiting.remove(object_id)
        visited.add(object_id)

    for object_id in sorted(relative_dependencies):
        visit(object_id, [object_id])
