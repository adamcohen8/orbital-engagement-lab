# ruff: noqa: F401,F403,F405,I001
from .campaign_common import *
from .monte_carlo_preparation import prepare_monte_carlo_runs

def can_run_monte_carlo_campaign(cfg: SimulationScenarioConfig) -> bool:
    """Return whether this campaign slice is owned by the execution package."""
    return bool(cfg.monte_carlo.enabled)


def _metric_gate_entries(gates: Any) -> list[dict[str, Any]]:
    if isinstance(gates, list):
        return [dict(item) for item in gates if isinstance(item, dict)]
    if not isinstance(gates, dict):
        return []
    out: list[dict[str, Any]] = []
    for key in ("metric_gates", "metrics", "generic"):
        raw = gates.get(key)
        if isinstance(raw, list):
            out.extend(dict(item) for item in raw if isinstance(item, dict))
    return out


def _metric_gates_need_payload(gates: Any) -> bool:
    for gate in _metric_gate_entries(gates):
        metric_path = str(gate.get("metric", "") or gate.get("path", "") or "").strip()
        if metric_path.startswith("payload."):
            return True
        if metric_path in {"derived.final_altitude_km_min", "derived.final_altitude_km_max"}:
            return True
    return False


def _mc_checkpoint_dir(outdir: Path) -> Path:
    return outdir / "mc_checkpoints"


def _mc_checkpoint_path(outdir: Path, iteration: int) -> Path:
    return _mc_checkpoint_dir(outdir) / f"iteration_{int(iteration):04d}.json"


def _config_fingerprint(config_dict: dict[str, Any]) -> str:
    from sim.execution.provenance import runtime_implementation_digest

    payload = json.dumps(
        {
            "config": _json_safe(config_dict),
            "runtime_implementation_digest": runtime_implementation_digest(),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_mc_checkpoint(outdir: Path, iteration: int, config_hash: str | None = None) -> dict[str, Any] | None:
    path = _mc_checkpoint_path(outdir, iteration)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Ignoring unreadable Monte Carlo checkpoint %s: %s", path, exc)
        return None
    if not isinstance(raw, dict):
        return None
    result = dict(raw.get("result", raw) or {})
    if int(result.get("iteration", iteration)) != int(iteration):
        return None
    stored_hash = str(raw.get("config_hash", "") or "")
    if config_hash is not None and stored_hash != str(config_hash):
        if stored_hash:
            logger.info("Ignoring stale Monte Carlo checkpoint %s: config hash changed.", path)
        else:
            logger.info("Ignoring legacy Monte Carlo checkpoint %s: missing config hash.", path)
        return None
    result["resumed_from_checkpoint"] = True
    return result


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        fv = float(value)
        return fv if np.isfinite(fv) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _write_mc_checkpoint(outdir: Path, iteration: int, result: dict[str, Any], config_hash: str) -> None:
    path = _mc_checkpoint_path(outdir, iteration)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        str(path),
        {
            "checkpoint_version": 1,
            "iteration": int(iteration),
            "config_hash": str(config_hash),
            "result": _json_safe(result),
        },
    )


def clear_monte_carlo_checkpoints(output_dir: str | Path) -> int:
    checkpoint_dir = _mc_checkpoint_dir(Path(output_dir))
    if not checkpoint_dir.exists():
        return 0
    count = sum(1 for path in checkpoint_dir.glob("iteration_*.json") if path.is_file())
    shutil.rmtree(checkpoint_dir)
    return int(count)


def monte_carlo_checkpoint_status(*, cfg: SimulationScenarioConfig, root: dict[str, Any] | None = None) -> dict[str, Any]:
    outdir = Path(cfg.outputs.output_dir)
    prepared = prepare_monte_carlo_runs(cfg=cfg, root=root or cfg.to_dict(), outdir=outdir)
    matching: list[int] = []
    stale: list[int] = []
    legacy: list[int] = []
    missing: list[int] = []
    for item in prepared:
        iteration = int(item["iteration"])
        path = _mc_checkpoint_path(outdir, iteration)
        if not path.exists():
            missing.append(iteration)
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            stale.append(iteration)
            continue
        if not isinstance(raw, dict):
            stale.append(iteration)
            continue
        stored_hash = str(raw.get("config_hash", "") or "")
        current_hash = str(item.get("config_hash", "") or _config_fingerprint(dict(item["config_dict"])))
        if not stored_hash:
            legacy.append(iteration)
        elif stored_hash == current_hash:
            matching.append(iteration)
        else:
            stale.append(iteration)
    return {
        "checkpoint_dir": str(_mc_checkpoint_dir(outdir)),
        "total": int(len(prepared)),
        "matching": matching,
        "stale": stale,
        "legacy": legacy,
        "missing": missing,
        "matching_count": int(len(matching)),
        "stale_count": int(len(stale)),
        "legacy_count": int(len(legacy)),
        "missing_count": int(len(missing)),
    }

__all__ = [name for name in globals() if not name.startswith("__")]
