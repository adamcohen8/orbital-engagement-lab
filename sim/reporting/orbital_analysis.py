"""Evidence-only scenario adapters for coverage and directed link analysis."""

from __future__ import annotations

import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from sim.analysis.directed_link import (
    DirectedLinkConfig,
    LinkTerminal,
    TerminalPattern,
    write_directed_link_artifacts,
)
from sim.analysis.global_coverage import (
    GlobalCoverageConfig,
    write_global_coverage_artifacts,
)
from sim.analysis.history_adapters import (
    AnalysisHistory,
    evaluate_history_directed_link,
    evaluate_history_global_coverage,
    history_from_single_run,
)
from sim.dynamics.orbit.frames import frame_context_from_mapping


def _only(mapping: dict[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"Unknown field(s) in {path}: {', '.join(unknown)}")


def _artifact_paths(value: Any) -> dict[str, Any]:
    if hasattr(value, "__dataclass_fields__"):
        return {
            key: str(item) if isinstance(item, Path) else item
            for key, item in asdict(value).items()
            if key != "output_dir"
        }
    return {}


def _replace_derived_directory(path: Path, *, root: Path) -> None:
    resolved = path.resolve()
    resolved.relative_to(root.resolve())
    if resolved.exists():
        shutil.rmtree(resolved)


def _coverage_interval_rows(product: Any) -> list[dict[str, Any]]:
    if product.refined_intervals:
        return [asdict(row) for row in product.refined_intervals]
    sparse = product.cell_metrics.intervals
    rows: list[dict[str, Any]] = []
    for cell_offset, cell_index in enumerate(sparse.cell_index):
        first = int(sparse.interval_offset[cell_offset])
        stop = int(sparse.interval_offset[cell_offset + 1])
        for interval_index, sparse_index in enumerate(range(first, stop)):
            start_index = int(sparse.start_sample_index[sparse_index])
            end_index = int(sparse.end_sample_index_exclusive[sparse_index])
            start_censored = start_index == 0
            end_censored = end_index == product.times_s.size
            start_s = float(product.times_s[start_index])
            end_s = float(product.times_s[-1] if end_censored else product.times_s[end_index])
            rows.append({
                "cell_index": int(cell_index), "interval_index": interval_index,
                "start_s": start_s, "end_s": end_s, "duration_s": max(0.0, end_s - start_s),
                "start_censored": start_censored, "end_censored": end_censored,
                "acquisition_disposition": "study_start_censored" if start_censored else "sample_bounded",
                "loss_disposition": "study_end_censored" if end_censored else "sample_bounded",
                "acquisition_reason": "available" if start_censored else "not_covered",
                "loss_reason": "available" if end_censored else "not_covered",
            })
    return rows


def _terminal(value: Any, *, asset_id: str, path: str) -> LinkTerminal:
    data = dict(value or {})
    _only(data, {"terminal_id", "quat_body_from_terminal", "pattern"}, path)
    pattern_data = dict(data.get("pattern", {}) or {})
    _only(pattern_data, {"kind", "gain_dbi", "half_angle_deg"}, f"{path}.pattern")
    kind = str(pattern_data.get("kind", "constant"))
    half_angle = pattern_data.get("half_angle_deg")
    return LinkTerminal(
        terminal_id=str(data.get("terminal_id") or ""),
        asset_id=asset_id,
        parent_frame="body",
        quat_parent_from_terminal=tuple(data.get("quat_body_from_terminal", (1.0, 0.0, 0.0, 0.0))),
        pattern=TerminalPattern(
            kind=kind,
            gain_dbi=float(pattern_data.get("gain_dbi")),
            half_angle_rad=None if half_angle is None else float(np.deg2rad(float(half_angle))),
        ),
    )


def _histories(context: Any, *, object_ids: set[str]) -> dict[str, AnalysisHistory]:
    attitude_enabled = bool(
        dict(dict(context.cfg.simulator.dynamics or {}).get("attitude", {}) or {}).get("enabled", True)
    )
    propagation = dict(getattr(context, "object_propagation", {}) or {})
    histories: dict[str, AnalysisHistory] = {}
    for object_id in sorted(object_ids):
        if object_id not in dict(context.truth_hist or {}):
            raise ValueError(f"Orbital analysis references object {object_id!r} without retained truth history.")
        truth = dict(context.truth_hist or {})[object_id]
        frame = str(dict(context.object_state_frames or {}).get(object_id, "eci") or "eci").lower()
        if frame != "eci":
            raise ValueError(f"Orbital analysis requires canonical ECI history for {object_id!r}; received {frame!r}.")
        product_kind = "ogp_scenario_history" if object_id in propagation else "onp_completed_run"
        object_config = dict(getattr(context.cfg, "objects", {}) or {}).get(object_id)
        object_attitude_enabled = product_kind != "ogp_scenario_history" and attitude_enabled and str(
            getattr(object_config, "runtime_profile", "") or ""
        ).strip().lower() != "trajectory_only"
        histories[object_id] = history_from_single_run(
            object_id=object_id,
            times_s=context.t_s,
            truth_state=truth,
            initial_jd_utc=float(context.cfg.simulator.initial_jd_utc),
            attitude_enabled=object_attitude_enabled,
            state_provider_id=f"scenario:{context.cfg.scenario_name}:{object_id}",
            product_kind=product_kind,
        )
    return histories


def run_scenario_orbital_analysis(*, context: Any) -> dict[str, Any]:
    section = context.cfg.outputs.orbital_analysis
    if not bool(section.enabled):
        return {}
    if context.cfg.simulator.initial_jd_utc is None:
        raise ValueError("outputs.orbital_analysis requires simulator.initial_jd_utc.")
    referenced_object_ids = {
        str(item.get("source_object_id") or "") for item in section.coverage
    }
    for item in section.directed_links:
        referenced_object_ids.update(
            {str(item.get("tx_object_id") or ""), str(item.get("rx_object_id") or "")}
        )
    histories = _histories(context, object_ids=referenced_object_ids)
    frame_context = frame_context_from_mapping(
        dict(getattr(context.cfg.simulator, "frames", {}) or {}),
        jd_utc_start=context.cfg.simulator.initial_jd_utc,
        source="scenario_orbital_analysis",
    )
    root = Path(context.outdir) / "orbital_analysis"
    result: dict[str, Any] = {"schema_version": "oel.scenario-orbital-analysis.v1", "coverage": [], "directed_links": []}

    coverage_allowed = {
        "analysis_id", "source_object_id", "sensor_id", "order", "half_angle_deg",
        "quat_body_from_sensor", "max_range_km", "chunk_size", "max_working_memory_bytes",
        "max_cell_time_comparisons", "transition_time_tolerance_s", "transition_max_iterations",
        "max_transition_refinement_evaluations",
        "include_cell_csv",
    }
    for index, raw in enumerate(section.coverage):
        data = dict(raw)
        _only(data, coverage_allowed, f"outputs.orbital_analysis.coverage[{index}]")
        source_id = str(data.get("source_object_id") or "")
        if source_id not in histories:
            raise ValueError(f"Coverage source_object_id {source_id!r} is not an active scenario object.")
        history = histories[source_id]
        if history.attitude_quat_bn is None:
            raise ValueError("Scenario coverage requires achieved attitude; simulator attitude dynamics are disabled.")
        config = GlobalCoverageConfig(
            analysis_id=str(data.get("analysis_id") or ""), source_asset_id=source_id,
            state_provider_id=history.state_provider_id, attitude_source_kind="achieved",
            attitude_provider_id=str(history.attitude_provider_id), sensor_id=str(data.get("sensor_id") or ""),
            order=int(data["order"]), half_angle_rad=float(np.deg2rad(float(data.get("half_angle_deg")))),
            quat_body_from_sensor=tuple(data.get("quat_body_from_sensor", (1.0, 0.0, 0.0, 0.0))),
            max_range_km=data.get("max_range_km"), chunk_size=int(data.get("chunk_size", 8192)),
            max_working_memory_bytes=int(data.get("max_working_memory_bytes", 512 * 1024 * 1024)),
            max_cell_time_comparisons=int(data.get("max_cell_time_comparisons", 300_000_000)),
            max_transition_refinement_evaluations=int(
                data.get("max_transition_refinement_evaluations", 5_000_000)
            ),
            transition_time_tolerance_s=(
                None
                if "transition_time_tolerance_s" not in data
                else float(data["transition_time_tolerance_s"])
            ),
            transition_max_iterations=(
                None
                if "transition_max_iterations" not in data
                else int(data["transition_max_iterations"])
            ),
        )
        product = evaluate_history_global_coverage(
            config, history=history, frame_context=frame_context
        )
        destination = root / "coverage" / config.analysis_id
        _replace_derived_directory(destination, root=Path(context.outdir))
        artifacts = write_global_coverage_artifacts(
            product, destination, include_cell_csv=bool(data.get("include_cell_csv", False))
        )
        result["coverage"].append({
            "analysis_id": config.analysis_id, "source_object_id": source_id,
            "state_provider_id": history.state_provider_id, "product_kind": history.product_kind,
            "attitude_source_kind": history.attitude_source_kind,
            "attitude_provider_id": history.attitude_provider_id,
            "refinement_source": product.refinement_provider_id or "sample_bounded",
            "summary": product.summary,
            "samples": [
                {"sample_index": i, "time_s": float(time_s), "covered_cell_count": int(product.covered_cell_count[i]),
                 "instantaneous_covered_fraction": float(product.instantaneous_covered_fraction[i])}
                for i, time_s in enumerate(product.times_s)
            ],
            "intervals": _coverage_interval_rows(product),
            "transitions": [asdict(row) for row in product.refined_transitions],
            "artifacts": _artifact_paths(artifacts),
            "input_evidence_sha256": product.input_evidence_sha256,
            "semantic_sha256": product.interval_semantic_sha256,
        })

    link_allowed = {
        "analysis_id", "link_id", "tx_object_id", "rx_object_id", "tx_terminal", "rx_terminal",
        "carrier_frequency_hz", "tx_power_w", "data_rate_bps", "system_noise_temperature_k",
        "required_eb_n0_db", "tx_line_loss_db", "rx_line_loss_db", "misc_loss_db", "max_range_km",
        "transition_time_tolerance_s", "transition_max_iterations", "include_margin_plot",
    }
    for index, raw in enumerate(section.directed_links):
        data = dict(raw)
        _only(data, link_allowed, f"outputs.orbital_analysis.directed_links[{index}]")
        tx_id, rx_id = str(data.get("tx_object_id") or ""), str(data.get("rx_object_id") or "")
        if tx_id not in histories or rx_id not in histories:
            raise ValueError("Directed link endpoints must name active scenario objects.")
        tx_terminal = _terminal(data.get("tx_terminal"), asset_id=tx_id, path=f"directed_links[{index}].tx_terminal")
        rx_terminal = _terminal(data.get("rx_terminal"), asset_id=rx_id, path=f"directed_links[{index}].rx_terminal")
        config = DirectedLinkConfig(
            analysis_id=str(data.get("analysis_id") or ""), link_id=str(data.get("link_id") or ""),
            tx_terminal=tx_terminal, rx_terminal=rx_terminal,
            carrier_frequency_hz=float(data.get("carrier_frequency_hz")), tx_power_w=float(data.get("tx_power_w")),
            data_rate_bps=float(data.get("data_rate_bps")),
            system_noise_temperature_k=float(data.get("system_noise_temperature_k")),
            required_eb_n0_db=float(data.get("required_eb_n0_db")),
            tx_line_loss_db=float(data.get("tx_line_loss_db", 0.0)),
            rx_line_loss_db=float(data.get("rx_line_loss_db", 0.0)), misc_loss_db=float(data.get("misc_loss_db", 0.0)),
            max_range_km=data.get("max_range_km"),
            transition_time_tolerance_s=(
                None
                if "transition_time_tolerance_s" not in data
                else float(data["transition_time_tolerance_s"])
            ),
            transition_max_iterations=(
                None
                if "transition_max_iterations" not in data
                else int(data["transition_max_iterations"])
            ),
        )
        tx_history, rx_history = histories[tx_id], histories[rx_id]
        product = evaluate_history_directed_link(
            config, tx_history=tx_history, rx_history=rx_history, frame_context=frame_context
        )
        destination = root / "directed_links" / config.analysis_id
        _replace_derived_directory(destination, root=Path(context.outdir))
        artifacts = write_directed_link_artifacts(
            product, destination, include_margin_plot=bool(data.get("include_margin_plot", True))
        )
        result["directed_links"].append({
            "analysis_id": config.analysis_id, "link_id": config.link_id,
            "tx_object_id": tx_id, "rx_object_id": rx_id,
            "tx_state_provider_id": tx_history.state_provider_id, "rx_state_provider_id": rx_history.state_provider_id,
            "tx_attitude_source_kind": tx_history.attitude_source_kind,
            "tx_attitude_provider_id": tx_history.attitude_provider_id,
            "rx_attitude_source_kind": rx_history.attitude_source_kind,
            "rx_attitude_provider_id": rx_history.attitude_provider_id,
            "refinement_source": {
                "method": product.summary["transition_refinement"]["method"],
                "provider_id": product.refinement_provider_id,
            },
            "summary": product.summary,
            "samples": [
                {"sample_index": i, "time_s": float(product.samples.time_s[i]), "range_km": float(product.samples.range_km[i]),
                 "margin_db": float(product.samples.margin_db[i]), "available": bool(product.samples.available[i]),
                 "primary_reason": product.samples.primary_reason[i]}
                for i in range(product.samples.time_s.size)
            ],
            "windows": [asdict(row) for row in product.windows],
            "transitions": [asdict(row) for row in product.transitions],
            "artifacts": _artifact_paths(artifacts), "input_evidence_sha256": product.input_evidence_sha256,
            "semantic_sha256": product.semantic_sha256,
        })
    return result


__all__ = ["run_scenario_orbital_analysis"]
