# ruff: noqa: F401,F403,F405,I001
import hashlib
from urllib.parse import urlsplit, urlunsplit

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


def _sanitized_endpoint(value: str) -> str:
    try:
        parsed = urlsplit(str(value))
        host = parsed.hostname or ""
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
        return urlunsplit((parsed.scheme, host, parsed.path, "", ""))
    except ValueError:
        return str(value).split("?", 1)[0].split("#", 1)[0]


def _sanitize_endpoint_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _sanitized_endpoint(item)
            if str(key).strip().lower() == "endpoint" and isinstance(item, str)
            else _sanitize_endpoint_fields(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_endpoint_fields(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_endpoint_fields(item) for item in value)
    return value

def _read_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_config_sha256(cfg: SimulationScenarioConfig) -> str:
    canonical = json.dumps(cfg.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_ai_report_payload_provenance(
    *,
    cfg: SimulationScenarioConfig,
    payload: dict[str, Any],
    payload_kind: str,
    summary_path: Path,
) -> dict[str, Any]:
    expected_name = str(cfg.scenario_name or "").strip()
    actual_name = str(payload.get("scenario_name", payload.get("suite_name", "")) or "").strip()
    source_path_value = getattr(cfg, "source_path", None)
    source_path = Path(source_path_value).expanduser().resolve() if source_path_value else None

    if not actual_name:
        raise ValueError(
            f"Saved {payload_kind} summary at {summary_path} has no scenario/suite identity; rerun the workflow."
        )
    report_config_is_workflow_config = payload_kind != "validation_harness"
    if report_config_is_workflow_config and actual_name and expected_name and actual_name != expected_name:
        raise ValueError(
            f"Saved {payload_kind} summary scenario {actual_name!r} does not match current config "
            f"scenario {expected_name!r}; rerun the workflow before preparing an AI report."
        )

    provenance: dict[str, Any] = {
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": _sha256_file(summary_path),
        "summary_mtime_ns": int(summary_path.stat().st_mtime_ns),
        "scenario_name": actual_name or expected_name,
    }
    config_sha256 = _canonical_config_sha256(cfg)
    stored_sha256 = str(dict(payload.get("reproducibility", {}) or {}).get("config_sha256", "") or "")
    if report_config_is_workflow_config and stored_sha256 and stored_sha256 != config_sha256:
        raise ValueError(
            f"Saved {payload_kind} summary config hash does not match the current config; "
            "rerun the workflow before preparing an AI report."
        )
    provenance.update(
        {
            "config_sha256": config_sha256,
            "stored_config_sha256": stored_sha256 or None,
        }
    )
    if not report_config_is_workflow_config:
        if source_path is not None and not source_path.is_file():
            raise FileNotFoundError(f"Current AI report settings config does not exist: {source_path}")
        provenance.update(
            {
                "report_settings_scenario_name": expected_name,
                "report_settings_config_path": str(source_path) if source_path is not None else None,
                "config_verification": "workflow_summary_identity_and_digest_plus_report_settings_config",
            }
        )
        return provenance
    if source_path is None:
        provenance["config_verification"] = (
            "scenario_and_hash" if stored_sha256 else "scenario_only_programmatic_config"
        )
        return provenance
    if not source_path.is_file():
        raise FileNotFoundError(f"Current AI report source config does not exist: {source_path}")

    payload_config_text = str(payload.get("config_path", payload.get("source_config_path", "")) or "").strip()
    if not payload_config_text:
        raise ValueError(
            f"Saved {payload_kind} summary at {summary_path} has no source config path; rerun the workflow."
        )
    payload_config = Path(payload_config_text).expanduser().resolve()
    if payload_config != source_path:
        raise ValueError(
            f"Saved {payload_kind} summary source config {payload_config} does not match current config "
            f"{source_path}; rerun the workflow before preparing an AI report."
        )
    if not stored_sha256 and summary_path.stat().st_mtime_ns < source_path.stat().st_mtime_ns:
        raise ValueError(
            f"Saved {payload_kind} summary at {summary_path} is older than current config {source_path}; "
            "rerun the workflow before preparing an AI report."
        )

    payload_generated_utc = str(
        payload.get("generated_utc", dict(payload.get("reproducibility", {}) or {}).get("generated_utc", "")) or ""
    ).strip()
    if payload_generated_utc:
        try:
            generated_at = datetime.fromisoformat(payload_generated_utc.replace("Z", "+00:00"))
            if generated_at.tzinfo is None:
                generated_at = generated_at.replace(tzinfo=timezone.utc)
        except ValueError as exc:
            raise ValueError(
                f"Saved {payload_kind} summary has an invalid generated_utc timestamp; rerun the workflow."
            ) from exc
        if not stored_sha256 and source_path.stat().st_mtime > generated_at.timestamp():
            raise ValueError(
                f"Saved {payload_kind} summary predates current config {source_path}; "
                "rerun the workflow before preparing an AI report."
            )

    provenance.update(
        {
            "config_path": str(source_path),
            "config_sha256": config_sha256,
            "config_mtime_ns": int(source_path.stat().st_mtime_ns),
            "payload_generated_utc": payload_generated_utc or None,
            "stored_config_sha256": stored_sha256 or None,
            "config_verification": "path_scenario_timestamp_and_hash"
            if stored_sha256
            else "path_scenario_and_timestamp",
        }
    )
    return provenance


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
    payload["ai_report_source_provenance"] = _validate_ai_report_payload_provenance(
        cfg=cfg,
        payload=payload,
        payload_kind=payload_kind,
        summary_path=path,
    )
    artifacts = dict(payload.get("artifacts", {}) or {})
    artifacts.setdefault("summary_json", str(path))
    payload["artifacts"] = artifacts
    return payload, payload_kind, outdir

__all__ = [name for name in globals() if not name.startswith("__")]
