# ruff: noqa: F401,F403,F405,I001
from .ai_report_models import *
from .ai_report_briefs import *

REPORT_PAYLOAD_ADAPTERS: tuple[ReportPayloadAdapter, ...] = (
    ReportPayloadAdapter(
        payload_kind="sensitivity",
        default_prompt_profile="sensitivity_analysis_report",
        summary_filenames=("master_analysis_sensitivity_summary.json",),
        can_load=_can_load_sensitivity_outputs,
        source_brief=_sensitivity_report_source_brief,
    ),
    ReportPayloadAdapter(
        payload_kind="controller_bench",
        default_prompt_profile="controller_bench_report",
        summary_filenames=("controller_bench_summary.json",),
        can_load=_can_load_controller_bench_outputs,
        source_brief=_controller_bench_report_source_brief,
    ),
    ReportPayloadAdapter(
        payload_kind="validation_harness",
        default_prompt_profile="validation_evidence_summary",
        summary_filenames=("validation_harness_report.json",),
        can_load=_can_load_validation_harness_outputs,
        source_brief=_validation_harness_report_source_brief,
    ),
    ReportPayloadAdapter(
        payload_kind="monte_carlo",
        default_prompt_profile="commander_summary",
        summary_filenames=("master_monte_carlo_summary.json",),
        can_load=_can_load_monte_carlo_outputs,
        source_brief=_report_source_brief,
    ),
)


def _adapter_for_payload_kind(payload_kind: str) -> ReportPayloadAdapter:
    kind = str(payload_kind or "monte_carlo").strip().lower()
    for adapter in REPORT_PAYLOAD_ADAPTERS:
        if adapter.payload_kind == kind:
            return adapter
    for adapter in REPORT_PAYLOAD_ADAPTERS:
        if adapter.payload_kind == "monte_carlo":
            return adapter
    raise ValueError("No Monte Carlo AI report adapter is registered.")


def _adapter_for_outputs(cfg: SimulationScenarioConfig, outdir: Path) -> ReportPayloadAdapter:
    for adapter in REPORT_PAYLOAD_ADAPTERS:
        if bool(adapter.can_load(cfg, outdir)):
            return adapter
    supported = ", ".join(adapter.payload_kind for adapter in REPORT_PAYLOAD_ADAPTERS)
    raise ValueError(f"AI report generation from existing outputs currently supports: {supported}.")


def _merged_ai_report_config(
    cfg: SimulationScenarioConfig,
    ai_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    merged = dict(cfg.outputs.ai_report or {})
    if ai_options:
        for key, value in dict(ai_options).items():
            if key not in {"allow_custom_endpoint", "allow_external_ai_prompt_files"}:
                merged[str(key)] = value
        if "allow_external_ai_prompt_files" in dict(ai_options):
            merged["allow_external_ai_prompt_files"] = bool(dict(ai_options).get("allow_external_ai_prompt_files"))
    return merged


def _agent_config_summary(agent: Any) -> dict[str, Any]:
    control = getattr(agent, "orbit_control", None)
    attitude_control = getattr(agent, "attitude_control", None)
    return {
        "enabled": bool(getattr(agent, "enabled", False)),
        "role": str(getattr(agent, "role", "")),
        "initial_state": deepcopy(dict(getattr(agent, "initial_state", {}) or {})),
        "orbit_control": {
            "module": getattr(control, "module", None) if control is not None else None,
            "class_name": getattr(control, "class_name", None) if control is not None else None,
        },
        "attitude_control": {
            "module": getattr(attitude_control, "module", None) if attitude_control is not None else None,
            "class_name": getattr(attitude_control, "class_name", None) if attitude_control is not None else None,
        },
    }


def _config_summary(cfg: SimulationScenarioConfig, payload_kind: str) -> dict[str, Any]:
    cfg_dict = cfg.to_dict()
    return {
        "scenario_name": cfg.scenario_name,
        "scenario_description": cfg.scenario_description,
        "payload_kind": str(payload_kind),
        "objects": {
            object_id: _agent_config_summary(section) for object_id, section in configured_objects(cfg).items()
        },
        "primary_object_pair": list(default_pair_object_ids(cfg) or ()),
        "simulator": {
            "duration_s": cfg.simulator.duration_s,
            "dt_s": cfg.simulator.dt_s,
            "dynamics": deepcopy(dict(cfg.simulator.dynamics or {})),
            "environment": deepcopy(dict(cfg.simulator.environment or {})),
            "termination": deepcopy(dict(cfg.simulator.termination or {})),
        },
        "analysis": deepcopy(dict(cfg_dict.get("analysis", {}) or {})),
        "monte_carlo": {
            "enabled": bool(cfg.monte_carlo.enabled),
            "iterations": int(cfg.monte_carlo.iterations),
            "base_seed": int(cfg.monte_carlo.base_seed),
            "parallel_enabled": bool(cfg.monte_carlo.parallel_enabled),
            "parallel_workers": int(cfg.monte_carlo.parallel_workers),
            "variations": deepcopy(list(dict(cfg_dict.get("monte_carlo", {}) or {}).get("variations", []) or [])),
        },
        "outputs": {
            "output_dir": cfg.outputs.output_dir,
            "mode": cfg.outputs.mode,
            "plots": deepcopy(dict(cfg.outputs.plots or {})),
            "monte_carlo": deepcopy(dict(cfg.outputs.monte_carlo or {})),
            "ai_report": deepcopy(dict(cfg.outputs.ai_report or {})),
        },
    }

def _read_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_ai_report_payload_from_outputs(
    *,
    cfg: SimulationScenarioConfig,
    output_dir: str | Path | None = None,
) -> tuple[dict[str, Any], str, Path]:
    outdir = Path(output_dir if output_dir is not None else cfg.outputs.output_dir)
    adapter = _adapter_for_outputs(cfg, outdir)
    payload_kind = adapter.payload_kind
    path = None
    for filename in adapter.summary_filenames:
        candidate = outdir / filename
        if candidate.exists():
            path = candidate
            break
    if path is None:
        path = outdir / adapter.summary_filenames[0]
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find saved {payload_kind} summary at {path}. "
            "Run the simulation with saved aggregate summaries enabled before estimating or creating an AI report."
        )
    payload = _read_json_file(path)
    artifacts = dict(payload.get("artifacts", {}) or {})
    artifacts.setdefault("summary_json", str(path))
    payload["artifacts"] = artifacts
    return payload, payload_kind, outdir

__all__ = [name for name in globals() if not name.startswith("__")]
