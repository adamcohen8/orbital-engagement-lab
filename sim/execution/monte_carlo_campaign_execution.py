# ruff: noqa: F401,F403,F405,I001
from .campaign_common import *
from .monte_carlo_checkpoints import *
from .monte_carlo_preparation import *

def run_serial_monte_carlo_runs(
    *,
    cfg: SimulationScenarioConfig,
    root: dict[str, Any],
    outdir: Path,
    strict_plugins: bool,
    collect_payload: bool = False,
    step_callback: StepCallback | None = None,
    batch_callback: BatchCallback | None = None,
    relative_range_writer: MonteCarloRelativeRangePlotWriter | None = None,
    resource_governor: ResourceGovernor | None = None,
    checkpoint: bool = True,
    initial_completed: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    require_pro_feature(FEATURE_CAMPAIGNS)
    prepared = prepare_monte_carlo_runs(cfg=cfg, root=root, outdir=outdir)
    completed: dict[int, dict[str, Any]] = dict(initial_completed or {})
    total_iters = int(cfg.monte_carlo.iterations)
    completed_count = len(completed)

    for item in prepared:
        iteration = int(item["iteration"])
        if iteration in completed:
            continue
        if checkpoint:
            cached = _load_mc_checkpoint(outdir, iteration, str(item.get("config_hash", "")))
            if cached is not None:
                completed[iteration] = cached
                if relative_range_writer is not None:
                    marker = getattr(relative_range_writer, "mark_checkpoint_omitted", None)
                    if callable(marker):
                        marker(iteration)
                completed_count += 1
                if batch_callback is not None:
                    try:
                        batch_callback(completed_count, total_iters)
                    except Exception as exc:
                        logger.warning("Disabling Monte Carlo callback after runtime error: %s", exc)
                        batch_callback = None
                continue
        if resource_governor is not None:
            resource_governor.wait_for_capacity(
                context=f"Monte Carlo iteration {iteration + 1}/{total_iters}",
                include_load=False,
            )
        ci = scenario_config_from_dict(dict(item["config_dict"]))
        if strict_plugins:
            errs = validate_scenario_plugins(ci)
            if errs:
                msg = f"Plugin validation failed in Monte Carlo iteration {iteration}:\n- " + "\n- ".join(errs)
                raise ValueError(msg)
        run_payload = _compat_run_single_config(ci, step_callback=step_callback)
        if relative_range_writer is not None and relative_range_writer.should_collect_iteration(iteration):
            relative_range_writer.add_run(
                iteration=iteration,
                series=_compat_relative_range_series(run_payload),
            )
        completed[iteration] = {
            "iteration": iteration,
            "summary": run_payload["summary"],
            "closest_approach_km": _compat_closest_approach(run_payload),
            "derived": {
                "relative_motion": relative_motion_summary_from_run_payload(run_payload),
            },
        }
        rel = dict(completed[iteration]["derived"]["relative_motion"])
        for key in ("initial_range_km", "final_range_km", "final_relative_speed_m_s", "max_relative_speed_m_s"):
            if key in rel:
                completed[iteration]["derived"][key] = rel.get(key)
        if collect_payload:
            completed[iteration]["payload"] = run_payload
        if checkpoint:
            _write_mc_checkpoint(outdir, iteration, completed[iteration], str(item.get("config_hash", "")))
        completed_count += 1
        if batch_callback is not None:
            try:
                batch_callback(completed_count, total_iters)
            except Exception as exc:
                logger.warning("Disabling Monte Carlo callback after runtime error: %s", exc)
                batch_callback = None

    return {
        "prepared": prepared,
        "completed": completed,
        "parallel_active": False,
        "parallel_fallback_reason": None,
        "checkpoint": {
            "enabled": bool(checkpoint),
            "checkpoint_dir": str(_mc_checkpoint_dir(outdir)),
            "resumed_count": int(sum(1 for item in completed.values() if bool(item.get("resumed_from_checkpoint")))),
            "written_count": int(sum(1 for item in completed.values() if not bool(item.get("resumed_from_checkpoint")))),
        },
    }


def run_monte_carlo_runs(
    *,
    cfg: SimulationScenarioConfig,
    root: dict[str, Any],
    outdir: Path,
    strict_plugins: bool,
    mc_out_cfg: dict[str, Any],
    step_callback: StepCallback | None = None,
    batch_callback: BatchCallback | None = None,
    batch_progress_callback: BatchProgressCallback | None = None,
    relative_range_writer: MonteCarloRelativeRangePlotWriter | None = None,
) -> dict[str, Any]:
    total_iters = int(cfg.monte_carlo.iterations)
    parallel_enabled = bool(cfg.monte_carlo.parallel_enabled)
    max_workers_cfg = int(cfg.monte_carlo.parallel_workers or 0)
    default_workers = max(1, (os.cpu_count() or 1) - 1)
    parallel_workers = max_workers_cfg if max_workers_cfg > 0 else default_workers
    parallel_workers = max(1, min(parallel_workers, max(total_iters, 1)))
    parallel_active = bool(parallel_enabled and total_iters > 1)
    parallel_fallback_reason: str | None = None
    prepared = prepare_monte_carlo_runs(cfg=cfg, root=root, outdir=outdir)
    hash_by_idx = {int(item["iteration"]): str(item.get("config_hash", "")) for item in prepared}
    completed: dict[int, dict[str, Any]] = {}
    collect_payload = _metric_gates_need_payload(mc_out_cfg.get("gates", {}) or {})
    checkpoint = checkpoint_enabled(cfg, mc_out_cfg)
    checkpoint_status_before = monte_carlo_checkpoint_status(cfg=cfg, root=root) if checkpoint else {}
    resource_governor = ResourceGovernor(cfg, emit=lambda msg: logger.warning("%s", msg))
    hierarchy = plan_hierarchical_execution(
        task_roots=[dict(item.get("config_dict", {}) or {}) for item in prepared],
        task_count=total_iters,
        requested_campaign_workers=parallel_workers,
        profile=resource_governor.profile,
    )
    parallel_workers = int(hierarchy.campaign_workers)
    if resource_governor.profile.force_serial or parallel_workers <= 1:
        parallel_active = False
        if parallel_enabled:
            parallel_fallback_reason = f"hierarchical_planner:{hierarchy.reason}"

    if parallel_active:
        manager = None
        progress_queue = None
        thread_env_prev = set_parallel_worker_thread_limits(default_threads="1")
        hierarchy_env_prev = apply_hierarchical_worker_env(hierarchy)
        try:
            resource_governor.wait_for_capacity(context="parallel Monte Carlo launch")
            if batch_progress_callback is not None:
                try:
                    manager = mp.Manager()
                    progress_queue = manager.Queue()
                except (OSError, PermissionError, EOFError) as exc:
                    logger.warning(
                        "Parallel Monte Carlo progress transport is unavailable; continuing without live progress: %s",
                        exc,
                    )
                    manager = None
                    progress_queue = None
            for item in prepared:
                iteration = int(item["iteration"])
                if checkpoint:
                    cached = _load_mc_checkpoint(outdir, iteration, str(item.get("config_hash", "")))
                    if cached is not None:
                        completed[iteration] = cached
                        if relative_range_writer is not None:
                            marker = getattr(relative_range_writer, "mark_checkpoint_omitted", None)
                            if callable(marker):
                                marker(iteration)
            tasks = [
                {
                    "iteration": item["iteration"],
                    "config_dict": item["config_dict"],
                    "strict_plugins": strict_plugins,
                    "progress_emit_every": int(mc_out_cfg.get("parallel_progress_emit_every_steps", 20) or 20),
                    "collect_relative_range_series": bool(
                        relative_range_writer is not None
                        and relative_range_writer.should_collect_iteration(int(item["iteration"]))
                    ),
                    "collect_payload": collect_payload,
                }
                for item in prepared
                if int(item["iteration"]) not in completed
            ]
            if not tasks:
                parallel_active = False
            pool_kwargs = {}
            if progress_queue is not None:
                pool_kwargs = {
                    "initializer": initialize_worker_progress_queue,
                    "initargs": (progress_queue,),
                }
            if tasks:
                with ProcessPoolExecutor(max_workers=parallel_workers, **pool_kwargs) as executor:
                    for fut, task in iter_bounded_futures(
                        executor,
                        run_mc_iteration_from_dict,
                        tasks,
                        max_in_flight=2 * parallel_workers,
                    ):
                        if fut is not None:
                            idx = int(task["iteration"])
                            result = dict(fut.result())
                            if relative_range_writer is not None:
                                relative_range_writer.add_run(
                                    iteration=idx,
                                    series=result.pop("relative_range_series", None),
                                )
                            completed[idx] = result
                            if checkpoint:
                                _write_mc_checkpoint(outdir, idx, result, hash_by_idx.get(idx, ""))
                            if batch_callback is not None:
                                try:
                                    batch_callback(len(completed), total_iters)
                                except Exception as exc:
                                    logger.warning("Disabling Monte Carlo callback after runtime error: %s", exc)
                                    batch_callback = None
                        if progress_queue is not None:
                            while True:
                                try:
                                    evt = progress_queue.get_nowait()
                                except queue_mod.Empty:
                                    break
                                except Exception:
                                    break
                                if batch_progress_callback is not None:
                                    try:
                                        batch_progress_callback(dict(evt or {}))
                                    except Exception as exc:
                                        logger.warning(
                                            "Disabling Monte Carlo progress callback after runtime error: %s", exc
                                        )
                                        batch_progress_callback = None
        except (
            OSError,
            PermissionError,
            NotImplementedError,
            EOFError,
            BrokenProcessPool,
            ResourcePressureError,
        ) as exc:
            parallel_active = False
            parallel_fallback_reason = format_parallel_fallback_reason(exc)
            logger.warning("Parallel Monte Carlo unavailable, falling back to serial execution: %s", exc)
        finally:
            if progress_queue is not None:
                try:
                    while True:
                        evt = progress_queue.get_nowait()
                        if batch_progress_callback is not None:
                            batch_progress_callback(dict(evt or {}))
                except Exception:
                    pass
            if manager is not None:
                try:
                    manager.shutdown()
                except Exception:
                    pass
            restore_env_vars(thread_env_prev)
            restore_hierarchical_worker_env(hierarchy_env_prev)

    if not parallel_active:
        serial_result = run_serial_monte_carlo_runs(
            cfg=cfg,
            root=root,
            outdir=outdir,
            strict_plugins=strict_plugins,
            collect_payload=collect_payload,
            step_callback=step_callback,
            batch_callback=batch_callback,
            relative_range_writer=relative_range_writer,
            resource_governor=resource_governor,
            checkpoint=checkpoint,
            initial_completed=completed,
        )
        prepared = list(serial_result.get("prepared", []) or [])
        completed = dict(serial_result.get("completed", {}) or {})

    checkpoint_meta = {
        "enabled": bool(checkpoint),
        "checkpoint_dir": str(_mc_checkpoint_dir(outdir)),
        "resumed_count": int(sum(1 for item in completed.values() if bool(dict(item or {}).get("resumed_from_checkpoint")))),
        "completed_count": int(len(completed)),
        "matching_before_count": int(dict(checkpoint_status_before).get("matching_count", 0) or 0),
        "stale_before_count": int(dict(checkpoint_status_before).get("stale_count", 0) or 0),
        "legacy_before_count": int(dict(checkpoint_status_before).get("legacy_count", 0) or 0),
    }
    return {
        "prepared": prepared,
        "completed": completed,
        "parallel_active": bool(parallel_active),
        "parallel_workers": int(parallel_workers if parallel_active else 1),
        "hierarchical_execution": hierarchy.payload(),
        "parallel_fallback_reason": parallel_fallback_reason,
        "checkpoint": checkpoint_meta,
        "resource_wait": resource_governor.telemetry(),
    }

__all__ = [name for name in globals() if not name.startswith("__")]
