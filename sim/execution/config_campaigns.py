# ruff: noqa: F401,F403,F405,I001
from .campaign_common import *

@dataclass(frozen=True)
class ConfigCampaignRunSpec:
    """One scenario config scheduled by a config-queue campaign."""

    index: int
    name: str
    config_path: Path
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConfigCampaignSpec:
    """Thin Pro campaign that queues existing scenario YAML configs."""

    name: str
    description: str
    config_path: Path
    runs: tuple[ConfigCampaignRunSpec, ...]
    execution: dict[str, Any]
    outputs: dict[str, Any]


def _truthy(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _resolve_campaign_input_path(raw_path: str | Path, *, base_dir: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _resolve_campaign_output_root(
    raw_path: str | Path,
    *,
    campaign_name: str,
    path_policy: ConfigPathPolicy,
) -> Path:
    text = str(raw_path or "").strip()
    return path_policy.resolve_output_dir(text or f"outputs/campaigns/{campaign_name}", purpose="outputs.root")


def load_config_campaign(campaign_path: str | Path) -> ConfigCampaignSpec:
    """Load a Pro config-queue campaign YAML file."""

    source = Path(campaign_path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Campaign YAML root must be a mapping: {source}")

    name = str(raw.get("name") or source.stem).strip() or source.stem
    description = str(raw.get("description", "") or "").strip()
    execution_raw = raw.get("execution", {}) or {}
    outputs_raw = raw.get("outputs", {}) or {}
    if not isinstance(execution_raw, dict):
        raise ValueError("campaign execution must be a mapping/object.")
    if not isinstance(outputs_raw, dict):
        raise ValueError("campaign outputs must be a mapping/object.")

    runs_raw = raw.get("runs")
    if not isinstance(runs_raw, list) or not runs_raw:
        raise ValueError("campaign runs must be a non-empty list.")

    runs: list[ConfigCampaignRunSpec] = []
    for idx, item in enumerate(runs_raw):
        if isinstance(item, str):
            entry = {"config": item}
        elif isinstance(item, dict):
            entry = dict(item)
        else:
            raise ValueError(f"campaign runs[{idx}] must be a config path or mapping/object.")
        if not _truthy(entry.get("enabled"), default=True):
            continue
        config_text = str(entry.get("config", "") or "").strip()
        if not config_text:
            raise ValueError(f"campaign runs[{idx}].config is required.")
        config_path = _resolve_campaign_input_path(config_text, base_dir=source.parent)
        run_name = str(entry.get("name") or config_path.stem).strip() or config_path.stem
        metadata = {k: v for k, v in entry.items() if k not in {"config", "name", "enabled"}}
        runs.append(ConfigCampaignRunSpec(index=len(runs), name=run_name, config_path=config_path, metadata=metadata))

    if not runs:
        raise ValueError("campaign has no enabled runs.")

    execution = dict(execution_raw)
    mode = str(execution.get("mode", "serial") or "serial").strip().lower()
    if mode not in {"serial", "parallel"}:
        raise ValueError("campaign execution.mode must be 'serial' or 'parallel'.")
    execution["mode"] = mode
    execution["max_workers"] = int(max(int(execution.get("max_workers", 0) or 0), 0))
    execution["validate_first"] = _truthy(execution.get("validate_first"), default=True)
    execution["stop_on_failure"] = _truthy(execution.get("stop_on_failure"), default=True)

    outputs = dict(outputs_raw)
    path_policy = ConfigPathPolicy.default(config_path=source)
    outputs["root"] = str(
        _resolve_campaign_output_root(outputs.get("root", ""), campaign_name=name, path_policy=path_policy)
    )
    outputs["include_payloads"] = _truthy(outputs.get("include_payloads"), default=False)

    return ConfigCampaignSpec(
        name=name,
        description=description,
        config_path=source,
        runs=tuple(runs),
        execution=execution,
        outputs=outputs,
    )


def _validate_campaign_run(spec: ConfigCampaignRunSpec, *, import_plugins: bool = True) -> dict[str, Any]:
    errors: list[str] = []
    cfg = None
    try:
        cfg = scenario_config_from_dict(load_simulation_yaml(spec.config_path).to_dict())
    except Exception as exc:
        errors.append(str(exc))
    if cfg is not None and import_plugins:
        strict_plugins = bool(cfg.simulator.plugin_validation.get("strict", True))
        plugin_errors = validate_scenario_plugins(cfg, import_plugins=True)
        if plugin_errors and strict_plugins:
            errors.extend(plugin_errors)
    return {
        "index": int(spec.index),
        "name": spec.name,
        "config_path": str(spec.config_path),
        "scenario_name": str(getattr(cfg, "scenario_name", "") or "") if cfg is not None else "",
        "output_dir": str(getattr(getattr(cfg, "outputs", None), "output_dir", "") or "") if cfg is not None else "",
        "valid": not errors,
        "errors": errors,
    }


def validate_config_campaign(campaign_path: str | Path, *, import_plugins: bool = True) -> dict[str, Any]:
    """Validate a config-queue campaign and its scheduled scenario configs."""

    require_pro_feature(FEATURE_CAMPAIGNS)
    errors: list[str] = []
    try:
        campaign = load_config_campaign(campaign_path)
    except Exception as exc:
        return {
            "valid": False,
            "campaign_path": str(Path(campaign_path).expanduser()),
            "name": "",
            "description": "",
            "run_count": 0,
            "runs": [],
            "errors": [str(exc)],
        }

    runs = [_validate_campaign_run(run, import_plugins=import_plugins) for run in campaign.runs]
    for run in runs:
        errors.extend(f"{run['name']}: {err}" for err in list(run.get("errors", []) or []))
    return {
        "valid": not errors,
        "campaign_path": str(campaign.config_path),
        "name": campaign.name,
        "description": campaign.description,
        "run_count": len(campaign.runs),
        "execution": dict(campaign.execution),
        "outputs": dict(campaign.outputs),
        "runs": runs,
        "errors": errors,
    }


def _output_index_from_payload(payload: dict[str, Any]) -> str:
    run = dict(payload.get("run", {}) or {})
    artifacts = dict(payload.get("artifacts", {}) or {})
    for candidate in (
        run.get("output_index_md"),
        payload.get("output_index_md"),
        artifacts.get("output_index_md"),
        artifacts.get("index_md"),
    ):
        text = str(candidate or "").strip()
        if text:
            return text
    output_dir = str(run.get("output_dir") or payload.get("output_dir") or "").strip()
    if output_dir:
        candidate_path = Path(output_dir) / "index.md"
        if candidate_path.exists():
            return str(candidate_path)
    return ""


def _output_dir_from_payload(payload: dict[str, Any]) -> str:
    run = dict(payload.get("run", {}) or {})
    artifacts = dict(payload.get("artifacts", {}) or {})
    for candidate in (run.get("output_dir"), payload.get("output_dir"), artifacts.get("output_dir")):
        text = str(candidate or "").strip()
        if text:
            return text
    index_path = _output_index_from_payload(payload)
    return str(Path(index_path).parent) if index_path else ""


def _short_campaign_payload_summary(payload: dict[str, Any]) -> dict[str, Any]:
    if str(dict(payload.get("analysis", {}) or {}).get("study_type", "")) == "sensitivity":
        analysis = dict(payload.get("analysis", {}) or {})
        return {
            "workflow": "sensitivity",
            "scenario_name": str(payload.get("scenario_name", "")),
            "run_count": int(analysis.get("run_count", len(payload.get("runs", []) or [])) or 0),
            "output_dir": _output_dir_from_payload(payload),
            "output_index_md": _output_index_from_payload(payload),
        }
    if bool(dict(payload.get("monte_carlo", {}) or {}).get("enabled", False)):
        mc = dict(payload.get("monte_carlo", {}) or {})
        agg = dict(payload.get("aggregate_stats", {}) or {})
        return {
            "workflow": "monte_carlo",
            "scenario_name": str(payload.get("scenario_name", "")),
            "run_count": int(mc.get("iterations", len(payload.get("runs", []) or [])) or 0),
            "pass_rate": agg.get("pass_rate"),
            "output_dir": _output_dir_from_payload(payload),
            "output_index_md": _output_index_from_payload(payload),
        }

    run = dict(payload.get("run", {}) or {})
    thrust = dict(run.get("thrust_stats", {}) or {})
    total_dv_m_s = 0.0
    for stats in thrust.values():
        try:
            total_dv_m_s += float(dict(stats or {}).get("total_dv_m_s", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
    return {
        "workflow": "single_run",
        "scenario_name": str(payload.get("scenario_name", run.get("scenario_name", "")) or ""),
        "duration_s": run.get("duration_s"),
        "samples": run.get("samples"),
        "terminated_early": bool(run.get("terminated_early", False)),
        "termination_reason": run.get("termination_reason"),
        "total_dv_m_s": total_dv_m_s,
        "output_dir": _output_dir_from_payload(payload),
        "output_index_md": _output_index_from_payload(payload),
    }


def _run_config_campaign_worker(task: dict[str, Any]) -> dict[str, Any]:
    from sim.execution.service import run_simulation_config_file

    started = time.perf_counter()
    spec = dict(task.get("spec", {}) or {})
    run_index = int(spec.get("index", 0))
    run_name = str(spec.get("name", ""))
    config_path = str(spec.get("config_path", ""))
    progress_queue = task.get("progress_queue")
    if progress_queue is None:
        progress_queue = worker_progress_queue()
    emit_every = int(task.get("progress_emit_every", 20) or 20)
    emit_every = max(1, emit_every)
    last_emit = -(10**9)

    def _emit_progress(event: str, **payload: Any) -> None:
        if progress_queue is None:
            return
        try:
            progress_queue.put(
                {
                    "event": event,
                    "pid": int(os.getpid()),
                    "iteration": run_index,
                    "run_index": run_index,
                    "run_name": run_name,
                    **payload,
                }
            )
        except Exception:
            pass

    def _on_step(step: int, total: int) -> None:
        nonlocal last_emit
        s = max(int(step), 0)
        t = max(int(total), 0)
        should_emit = (s == 0) or (t > 0 and s >= t) or (s - last_emit >= emit_every)
        if not should_emit:
            return
        last_emit = s
        _emit_progress("step", step=int(s), total=int(t))

    try:
        _emit_progress("start", step=0, total=0)
        payload = run_simulation_config_file(
            config_path=config_path,
            step_callback=_on_step if progress_queue is not None else None,
        )
        _emit_progress("done")
        result = {
            "index": run_index,
            "name": run_name,
            "config_path": config_path,
            "status": "passed",
            "wall_time_s": float(time.perf_counter() - started),
            "summary": _short_campaign_payload_summary(dict(payload or {})),
        }
        if bool(task.get("include_payload", False)):
            result["payload"] = payload
        return result
    except Exception as exc:
        _emit_progress("done")
        return {
            "index": run_index,
            "name": run_name,
            "config_path": config_path,
            "status": "failed",
            "wall_time_s": float(time.perf_counter() - started),
            "error": f"{type(exc).__name__}: {exc}",
            "summary": {},
        }


def _campaign_run_task(spec: ConfigCampaignRunSpec, *, include_payload: bool) -> dict[str, Any]:
    return {
        "spec": {
            "index": int(spec.index),
            "name": spec.name,
            "config_path": str(spec.config_path),
            "metadata": dict(spec.metadata),
        },
        "include_payload": bool(include_payload),
    }


def _write_config_campaign_artifacts(campaign: ConfigCampaignSpec, payload: dict[str, Any]) -> dict[str, str]:
    outdir = Path(str(campaign.outputs.get("root", ""))).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    summary_json = outdir / "campaign_summary.json"
    runs_csv = outdir / "campaign_runs.csv"
    summary_md = outdir / "campaign_summary.md"

    rows = []
    for run in list(payload.get("runs", []) or []):
        summary = dict(run.get("summary", {}) or {})
        rows.append(
            {
                "index": int(run.get("index", 0)),
                "name": str(run.get("name", "")),
                "status": str(run.get("status", "")),
                "workflow": str(summary.get("workflow", "")),
                "scenario_name": str(summary.get("scenario_name", "")),
                "config_path": str(run.get("config_path", "")),
                "output_dir": str(summary.get("output_dir", "")),
                "total_dv_m_s": summary.get("total_dv_m_s", ""),
                "duration_s": summary.get("duration_s", ""),
                "wall_time_s": run.get("wall_time_s", ""),
                "error": str(run.get("error", "")),
            }
        )
    with runs_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "name",
                "status",
                "workflow",
                "scenario_name",
                "config_path",
                "output_dir",
                "total_dv_m_s",
                "duration_s",
                "wall_time_s",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"# {campaign.name} Campaign",
        "",
        campaign.description,
        "",
        f"- Runs: {payload['summary']['run_count']}",
        f"- Passed: {payload['summary']['passed_count']}",
        f"- Failed: {payload['summary']['failed_count']}",
        f"- Execution: {payload['execution']['mode']}",
        "",
        "| Run | Status | Workflow | Scenario | Output |",
        "|---|---:|---|---|---|",
    ]
    for run in list(payload.get("runs", []) or []):
        summary = dict(run.get("summary", {}) or {})
        output = str(summary.get("output_index_md") or summary.get("output_dir") or "")
        lines.append(
            f"| {run.get('name', '')} | {run.get('status', '')} | "
            f"{summary.get('workflow', '')} | {summary.get('scenario_name', '')} | {output} |"
        )
    summary_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return {
        "output_dir": str(outdir),
        "summary_json": str(summary_json),
        "runs_csv": str(runs_csv),
        "summary_md": str(summary_md),
        "output_index_md": str(summary_md),
    }
