from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable

from sim.config.profiles import profile_choices

DEFAULT_HELP_SCOPE = "public"


@dataclass(frozen=True)
class ConfigOption:
    value: str
    description: str


@dataclass(frozen=True)
class ConfigHelpEntry:
    path: str
    title: str
    description: str
    options: tuple[ConfigOption, ...] = ()
    option_loader: Callable[[], tuple[ConfigOption, ...]] | None = None
    aliases: tuple[str, ...] = ()
    example: str = ""
    notes: tuple[str, ...] = ()
    visibility: str = "public"

    def resolved_options(self) -> tuple[ConfigOption, ...]:
        if self.option_loader is None:
            return self.options
        return self.option_loader()

    def search_terms(self) -> tuple[str, ...]:
        return (self.path, self.title, *self.aliases)


def _option(value: str, description: str) -> ConfigOption:
    return ConfigOption(value=value, description=description)


def _scope_allows(entry: ConfigHelpEntry, scope: str) -> bool:
    normalized = str(scope or DEFAULT_HELP_SCOPE).strip().lower()
    if normalized == "all":
        return True
    if normalized == "pro":
        return entry.visibility in {"public", "pro"}
    return entry.visibility == "public"


def _figure_id_options() -> tuple[ConfigOption, ...]:
    from sim.master_outputs import AVAILABLE_FIGURE_IDS

    descriptions = {
        "run_dashboard": "Single-run summary dashboard.",
        "rendezvous_summary": "Rendezvous-focused relative motion summary.",
        "rendezvous_summary_curvilinear": "Curvilinear-RIC rendezvous summary with range/speed and delta-v.",
        "ground_track_multi": "Multi-object ground tracks.",
        "trajectory_ric_curv_2d_multi": "Multi-object curvilinear-RIC 2D relative trajectory.",
        "trajectory_ric_rect_2d_multi_target_burns": "Rectangular-RIC 2D trajectory with target-burn markers.",
        "trajectory_ric_curv_2d_multi_target_burns": "Curvilinear-RIC 2D trajectory with target-burn markers.",
        "relative_range": "Range between relevant objects over time.",
        "control_effort": "Control effort or delta-v usage over time.",
        "orbital_elements_summary": "Six-panel classical orbital element history.",
        "satellite_delta_v_remaining": "Remaining satellite delta-v budget over time.",
        "ground_station_access": "Ground-station access, elevation, and range.",
    }
    return tuple(
        _option(figure_id, descriptions.get(figure_id, "Generated single-run plot artifact."))
        for figure_id in AVAILABLE_FIGURE_IDS
    )


def _plot_preset_options() -> tuple[ConfigOption, ...]:
    from sim.master_outputs import PLOT_PRESETS

    descriptions = {
        "minimal": "Smallest useful plot bundle.",
        "orbit": "Orbit trajectory, ground track, and orbital-elements plots.",
        "rendezvous": "RPO-oriented relative trajectory, range, and control plots.",
        "attitude": "Attitude state, rates, error, and control summary.",
        "estimation": "Estimator, knowledge, and sensor-access diagnostics.",
        "access": "Ground-station access and ground-track plots.",
        "rocket": "Rocket ascent, GNC, insertion, and resource diagnostics.",
        "debug": "Every available figure ID.",
    }
    return tuple(_option(name, descriptions.get(name, ", ".join(ids))) for name, ids in sorted(PLOT_PRESETS.items()))


def _animation_type_options() -> tuple[ConfigOption, ...]:
    from sim.master_outputs import AVAILABLE_ANIMATION_TYPES

    return tuple(_option(name, "Animation artifact type.") for name in AVAILABLE_ANIMATION_TYPES)


def _profile_options() -> tuple[ConfigOption, ...]:
    descriptions = {
        "fast": "Lower-cost settings for interactive experiments.",
        "ops": "Default operational balance of cost and fidelity.",
        "high_fidelity": "Tighter integration settings for validation-style runs.",
    }
    return tuple(_option(name, descriptions.get(name, "Simulation fidelity profile.")) for name in profile_choices())


def _satellite_preset_options() -> tuple[ConfigOption, ...]:
    from sim.app.services import _discover_named_presets

    return tuple(
        _option(name, "Reusable satellite hardware preset.")
        for name in _discover_named_presets("sim.presets.satellites", "SatellitePreset")
    )


def _rocket_preset_options() -> tuple[ConfigOption, ...]:
    from sim.app.services import _discover_named_presets

    return tuple(
        _option(name, "Reusable rocket stack preset.")
        for name in _discover_named_presets("sim.presets.rockets", "RocketStackPreset")
    )


CONFIG_HELP_ENTRIES: tuple[ConfigHelpEntry, ...] = (
    ConfigHelpEntry(
        path="simulator.environment.ephemeris_mode",
        title="Ephemeris Mode",
        description="Controls how Sun/Moon and other time-dependent ephemeris positions are resolved.",
        aliases=("ephemeris model", "emphemeris model", "sun moon model", "environment ephemeris"),
        options=(
            _option("analytic_enhanced", "Default low-cost analytic Sun/Moon model with improved approximations."),
            _option(
                "analytic_simple", "Older lightweight analytic Sun/Moon model; useful for repeatable legacy cases."
            ),
            _option("external", "Use env['ephemeris_callable'] supplied by Python code."),
            _option("spice", "Use spiceypy and configured SPICE kernels, or env['spice_ephemeris_callable']."),
        ),
        example='simulator:\n  environment:\n    ephemeris_mode: "analytic_enhanced"',
        notes=("Aliases accepted by lower-level code include simple/enhanced and spiceypy/callable forms.",),
    ),
    ConfigHelpEntry(
        path="simulator.dynamics.orbit.model",
        title="Orbit Base Model",
        description="Names the base orbit propagation model. Two-body is the default; CR3BP is an opt-in rotating-frame model for cislunar teaching cases.",
        aliases=("orbit model", "dynamics model", "propagation model"),
        options=(
            _option(
                "two_body",
                "Central-body Keplerian gravity baseline. Add j2, drag, SRP, or spherical_harmonics for higher fidelity.",
            ),
            _option(
                "cr3bp",
                "Circular restricted three-body propagation in a configured rotating-frame system such as earth_moon.",
            ),
        ),
        example='simulator:\n  dynamics:\n    orbit:\n      model: "two_body"\n      j2: true\n      drag: false',
    ),
    ConfigHelpEntry(
        path="simulator.dynamics.orbit.integrator",
        title="Orbit Integrator",
        description="Numerical integrator used by the orbit propagator.",
        aliases=("integrator", "orbit integration", "propagator integrator"),
        options=(
            _option("rk4", "Fixed-step fourth-order Runge-Kutta; normal starting point."),
            _option("rkf78", "Adaptive RKF78 path for tighter propagation checks."),
            _option("dopri5", "Adaptive Dormand-Prince 5 path."),
            _option("adaptive", "Alias that currently routes to RKF78 behavior."),
        ),
        example="simulator:\n  dynamics:\n    orbit:\n      integrator: rk4",
    ),
    ConfigHelpEntry(
        path="simulator.dynamics.rocket.atmosphere_model",
        title="Atmosphere Model",
        description="Density model used by drag/aerodynamic calculations when atmosphere-dependent forces are enabled.",
        aliases=("atmosphere", "atmospheric model", "density model", "drag atmosphere"),
        options=(
            _option("exponential", "Simple exponential atmosphere; cheapest and most robust."),
            _option("ussa1976", "U.S. Standard Atmosphere 1976 table-style model; default for rocket aero configs."),
            _option(
                "msis86",
                "Local MSIS-86 backend copied from MATLAB HPOP; supports direct env inputs or HPOP-style SW-All table input.",
            ),
            _option(
                "nrlmsise00",
                "Local NRLMSISE-00 backend copied from MATLAB HPOP; supports env-provided inputs or callable override.",
            ),
            _option(
                "jacchia70",
                "Local Jacchia-70 backend copied from MATLAB HPOP; supports direct env inputs and HPOP-style solar tables.",
            ),
            _option(
                "jb2006",
                "Local Jacchia-Bowman 2006 backend copied from MATLAB HPOP; supports HPOP-style SOL and geomagnetic inputs.",
            ),
            _option(
                "jb2008",
                "Local Jacchia-Bowman 2008 backend copied from MATLAB HPOP; supports HPOP-style SOL and DTC inputs.",
            ),
            _option(
                "harris_priester",
                "Local Harris-Priester backend using the bundled HPOP coefficient table.",
            ),
        ),
        example="simulator:\n  dynamics:\n    rocket:\n      atmosphere_model: ussa1976",
    ),
    ConfigHelpEntry(
        path="simulator.dynamics.orbit.srp_shadow_model",
        title="SRP Shadow Model",
        description="Eclipse/shadow model used when solar radiation pressure is enabled.",
        aliases=("shadow model", "eclipse model", "srp eclipse"),
        options=(
            _option("conical", "Conical Earth shadow approximation; normal fidelity-oriented choice."),
            _option("cylindrical", "Simpler cylindrical shadow approximation."),
            _option("none", "Disable SRP shadow gating."),
        ),
        example="simulator:\n  dynamics:\n    orbit:\n      srp: true\n      srp_shadow_model: conical",
    ),
    ConfigHelpEntry(
        path="outputs.mode",
        title="Output Mode",
        description="Controls whether plots and artifacts are shown interactively, saved, or both.",
        aliases=("output mode", "plot mode", "display mode"),
        options=(
            _option("interactive", "Show interactive windows where supported."),
            _option("save", "Write artifacts to outputs.output_dir without opening windows."),
            _option("both", "Save artifacts and show interactive windows."),
        ),
        example='outputs:\n  mode: "save"',
    ),
    ConfigHelpEntry(
        path="outputs.plots.preset",
        title="Plot Preset",
        description="Expands to a maintained bundle of figure IDs. You can combine presets with explicit figure_ids.",
        aliases=("plot preset", "plot presets", "figure preset", "outputs plots preset"),
        option_loader=_plot_preset_options,
        example='outputs:\n  plots:\n    preset: "rendezvous"',
    ),
    ConfigHelpEntry(
        path="outputs.plots.figure_ids",
        title="Figure IDs",
        description="Explicit single-run plot artifacts to request.",
        aliases=("figure ids", "figures", "plot ids", "available plots"),
        option_loader=_figure_id_options,
        example='outputs:\n  plots:\n    figure_ids: ["run_dashboard", "relative_range"]',
    ),
    ConfigHelpEntry(
        path="outputs.plots.style",
        title="Plot Style",
        description="Controls the visual identity applied to saved single-run plot artifacts.",
        aliases=("plot style", "plot theme", "artifact style", "artifact theme"),
        options=(
            _option("oel_dark", "Branded dark OEL artifact style for screen review and demos."),
            _option("oel_light", "Branded light OEL artifact style for reports and print-friendly review."),
            _option("matplotlib", "Use Matplotlib defaults without OEL artifact styling."),
        ),
        example='outputs:\n  plots:\n    style: "oel_dark"',
    ),
    ConfigHelpEntry(
        path="outputs.animations.types",
        title="Animation Types",
        description="Animation artifacts to request when outputs.animations.enabled is true.",
        aliases=("animation types", "animations", "animation ids"),
        option_loader=_animation_type_options,
        example='outputs:\n  animations:\n    enabled: true\n    types: ["ground_track_multi"]',
    ),
    ConfigHelpEntry(
        path="outputs.animations.style",
        title="Animation Style",
        description="Controls the visual identity applied to saved animation artifacts. Defaults to outputs.plots.style.",
        aliases=("animation style", "animation theme", "movie style", "movie theme"),
        options=(
            _option("oel_dark", "Branded dark OEL artifact style for screen review and demos."),
            _option("oel_light", "Branded light OEL artifact style for reports and print-friendly review."),
            _option("matplotlib", "Use Matplotlib defaults without OEL artifact styling."),
        ),
        example='outputs:\n  animations:\n    style: "oel_dark"',
    ),
    ConfigHelpEntry(
        path="outputs.review",
        title="Output Review Store",
        description=(
            "Enables the durable SQLite review store for completed single-run outputs. "
            "This is the planned data layer for the Output Review Workbench."
        ),
        aliases=("review store", "output review", "orw", "sqlite review", "outputs review"),
        options=(
            _option("enabled", "Write review/run.sqlite and review/schema.json beside normal artifacts."),
            _option("detail", "compact, standard, or full. Initial tables focus on standard single-run review."),
            _option("strict", "Raise if review-store writing fails instead of preserving normal artifacts."),
        ),
        example='outputs:\n  review:\n    enabled: true\n    detail: "standard"',
    ),
    ConfigHelpEntry(
        path="objects.<id>.kind",
        title="Object Kind",
        description="Declares the runtime object family for a configured object.",
        aliases=("agent kind", "object kind", "vehicle kind"),
        options=(
            _option("satellite", "Spacecraft/satellite object using orbital and optional attitude dynamics."),
            _option("rocket", "Launch/ascent vehicle using rocket-specific dynamics and guidance."),
        ),
        example="objects:\n  chaser:\n    kind: satellite",
    ),
    ConfigHelpEntry(
        path="objects.<id>.specs.preset_satellite",
        title="Satellite Preset",
        description="Named Python satellite hardware preset used by compatibility config paths.",
        aliases=("satellite preset", "preset satellite", "spacecraft preset"),
        option_loader=_satellite_preset_options,
        example="objects:\n  target:\n    specs:\n      preset_satellite: BASIC_SATELLITE",
    ),
    ConfigHelpEntry(
        path="objects.<id>.specs.preset_stack",
        title="Rocket Stack Preset",
        description="Named Python rocket stack preset used by rocket configs.",
        aliases=("rocket preset", "stack preset", "preset stack"),
        option_loader=_rocket_preset_options,
        example="rocket:\n  specs:\n    preset_stack: BASIC_TWO_STAGE_STACK",
    ),
    ConfigHelpEntry(
        path="profile",
        title="Simulation Profile",
        description="Shared fidelity profile used by Python builders and profile-aware workflows.",
        aliases=("fidelity profile", "simulation profile"),
        option_loader=_profile_options,
        example='profile: "ops"',
    ),
)


def _load_pro_config_help_entries() -> tuple[ConfigHelpEntry, ...]:
    try:
        from sim.config.pro_help import PRO_CONFIG_HELP_ENTRIES
    except Exception:
        return ()
    return tuple(PRO_CONFIG_HELP_ENTRIES)


CONFIG_HELP_ENTRIES = (*CONFIG_HELP_ENTRIES, *_load_pro_config_help_entries())


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.lower())).strip()


def _score(query: str, entry: ConfigHelpEntry) -> float:
    q = _normalize(query)
    if not q:
        return 0.0
    best = 0.0
    q_tokens = set(q.split())
    for term_raw in entry.search_terms():
        term = _normalize(term_raw)
        if not term:
            continue
        term_tokens = set(term.split())
        if q == term:
            return 1.0
        if q in term or term in q:
            best = max(best, 0.92)
        overlap = len(q_tokens & term_tokens) / max(1, len(q_tokens | term_tokens))
        best = max(best, 0.55 * overlap + 0.45 * SequenceMatcher(None, q, term).ratio())
    return best


def find_config_help(
    query: str,
    *,
    limit: int = 5,
    scope: str = DEFAULT_HELP_SCOPE,
) -> list[tuple[float, ConfigHelpEntry]]:
    """Return matching config help entries ranked by fuzzy relevance."""
    entries = [entry for entry in CONFIG_HELP_ENTRIES if _scope_allows(entry, scope)]
    ranked = sorted(((_score(query, entry), entry) for entry in entries), key=lambda item: item[0], reverse=True)
    return [(score, entry) for score, entry in ranked[:limit] if score >= 0.35]


def list_config_help_entries(*, scope: str = DEFAULT_HELP_SCOPE) -> list[ConfigHelpEntry]:
    return [entry for entry in CONFIG_HELP_ENTRIES if _scope_allows(entry, scope)]


def load_config_help_context(path: str | Path) -> dict[str, Any]:
    """Load YAML as plain data for contextual help without resolving plugins or presets."""
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError("PyYAML is required to inspect config files. Install with `pip install pyyaml`.") from exc

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config YAML root must be a mapping/object: {config_path}")
    return dict(raw)


def _is_placeholder_token(token: str) -> bool:
    return token.startswith("<") and token.endswith(">")


def _resolve_config_values(value: Any, tokens: list[str], actual_path: str = "") -> list[tuple[str, Any]]:
    if not tokens:
        return [(actual_path, value)]

    token = tokens[0]
    rest = tokens[1:]
    out: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        if _is_placeholder_token(token):
            for key, child in value.items():
                child_path = f"{actual_path}.{key}" if actual_path else str(key)
                out.extend(_resolve_config_values(child, rest, child_path))
        elif token in value:
            child_path = f"{actual_path}.{token}" if actual_path else token
            out.extend(_resolve_config_values(value[token], rest, child_path))
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            child_path = f"{actual_path}[{idx}]" if actual_path else f"[{idx}]"
            out.extend(_resolve_config_values(child, tokens, child_path))
    return out


def _format_config_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None or isinstance(value, (bool, int, float)):
        return repr(value)
    try:
        import yaml  # type: ignore

        rendered = yaml.safe_dump(value, sort_keys=False, default_flow_style=True).strip()
        return rendered.rstrip("\n")
    except Exception:
        return repr(value)


def _context_label(config_path: str | Path | None) -> str:
    if config_path is None:
        return "Current Config"
    return f"Current Config ({config_path})"


def format_config_help(
    query: str,
    *,
    max_options: int = 80,
    config_data: dict[str, Any] | None = None,
    config_path: str | Path | None = None,
    scope: str = DEFAULT_HELP_SCOPE,
) -> str:
    matches = find_config_help(query, limit=5, scope=scope)
    if not matches:
        return f"No config help match for {query!r}.\n\nRun `python config_help.py --list` to see known help topics."

    score, entry = matches[0]
    lines = [
        f"{entry.title}",
        f"Field: {entry.path}",
        "",
        *textwrap.wrap(entry.description, width=88),
        "",
    ]
    if config_data is not None:
        lines.append(_context_label(config_path))
        values = _resolve_config_values(config_data, entry.path.split("."))
        if values:
            for actual_path, value in values:
                lines.append(f"  {actual_path}: {_format_config_value(value)}")
        else:
            lines.append(f"  {entry.path}: not set in this config")
        lines.append("")

    lines.append("Options:")
    options = entry.resolved_options()
    if not options:
        lines.append("  (No fixed option list; this field accepts user-supplied values.)")
    else:
        width = min(max(len(opt.value) for opt in options), 34)
        for opt in options[:max_options]:
            wrapped = textwrap.wrap(opt.description, width=max(40, 84 - width))
            lines.append(f"  {opt.value:<{width}}  {wrapped[0] if wrapped else ''}")
            for continuation in wrapped[1:]:
                lines.append(f"  {'':<{width}}  {continuation}")
        if len(options) > max_options:
            lines.append(
                f"  ... {len(options) - max_options} more options hidden; refine the query for a narrower list."
            )
    if entry.notes:
        lines.extend(["", "Notes:"])
        for note in entry.notes:
            for idx, wrapped in enumerate(textwrap.wrap(note, width=84)):
                prefix = "  - " if idx == 0 else "    "
                lines.append(f"{prefix}{wrapped}")
    if entry.example:
        lines.extend(["", "Example:", textwrap.indent(entry.example, "  ")])
    if score < 0.75 and len(matches) > 1:
        lines.extend(["", "Other possible matches:"])
        for _, other in matches[1:4]:
            lines.append(f"  - {other.path} ({other.title})")
    return "\n".join(lines)


def format_config_help_list(*, scope: str = DEFAULT_HELP_SCOPE) -> str:
    lines = ["Known config help topics:"]
    entries = list_config_help_entries(scope=scope)
    if not entries:
        return "Known config help topics: none"
    width = max(len(entry.path) for entry in entries)
    for entry in sorted(entries, key=lambda item: item.path):
        lines.append(f"  {entry.path:<{width}}  {entry.title}")
    return "\n".join(lines)
