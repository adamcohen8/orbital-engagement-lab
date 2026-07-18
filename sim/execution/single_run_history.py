"""Allocation growth and compaction for single-run history arrays."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from sim.resource_limits import HistoryMemoryEstimate, enforce_history_memory_budget

if TYPE_CHECKING:
    from sim.single_run import _SingleRunEngine


class SingleRunHistoryStore:
    """Own history resizing and retention while the engine owns lifecycle state."""

    def __init__(self, engine: _SingleRunEngine) -> None:
        self.engine = engine

    def ensure_belief_width(self, aid: str, width: int) -> None:
        engine = self.engine
        hist = engine.belief_hist[aid]
        if hist.shape[1] >= int(width):
            return
        extra_columns = int(width) - int(hist.shape[1])
        extra_array_bytes = int(engine.n * extra_columns * np.dtype(float).itemsize)
        next_estimate = HistoryMemoryEstimate(
            samples=engine.history_memory_estimate.samples,
            active_objects=engine.history_memory_estimate.active_objects,
            knowledge_pairs=engine.history_memory_estimate.knowledge_pairs,
            array_bytes=engine.history_memory_estimate.array_bytes + extra_array_bytes,
            estimated_peak_bytes=engine.history_memory_estimate.estimated_peak_bytes + (2 * extra_array_bytes),
            limit_bytes=engine.history_memory_estimate.limit_bytes,
        )
        enforce_history_memory_budget(next_estimate)
        engine.history_memory_estimate = next_estimate
        expanded = np.full((engine.n, int(width)), np.nan)
        if hist.shape[1] > 0:
            expanded[:, : hist.shape[1]] = hist
        engine.belief_hist[aid] = expanded

    @staticmethod
    def grow_axis0(arr: np.ndarray | None, rows: int, *, fill: float = np.nan) -> np.ndarray | None:
        if arr is None:
            return None
        if arr.shape[0] >= int(rows):
            return arr
        shape = (int(rows), *arr.shape[1:])
        expanded = np.full(shape, fill, dtype=arr.dtype)
        expanded[: arr.shape[0], ...] = arr
        return expanded

    @staticmethod
    def compact_axis0_latest(
        arr: np.ndarray | None,
        *,
        start: int,
        count: int,
        fill: float = np.nan,
    ) -> np.ndarray | None:
        if arr is None:
            return None
        retained = arr[int(start) : int(start) + int(count), ...].copy()
        arr[...] = fill
        arr[: int(count), ...] = retained
        return arr

    @staticmethod
    def compact_event_history_latest(
        rows: list[dict[str, Any]], *, retained_start_time_s: float
    ) -> list[dict[str, Any]]:
        threshold = float(retained_start_time_s) - 1.0e-9
        retained: list[dict[str, Any]] = []
        for row in rows:
            event_t = row.get("interval_end_t_s", row.get("t_s")) if isinstance(row, dict) else None
            try:
                t_s = float(event_t)
            except (TypeError, ValueError):
                retained.append(row)
                continue
            if t_s >= threshold:
                retained.append(row)
        return retained

    def compact_if_needed(self, *, keep_latest: int | None = None) -> None:
        engine = self.engine
        if engine.history_mode != "dynamic" or engine.current_index < engine.n - 1:
            return
        if engine.n < engine.max_history_samples:
            return
        if keep_latest is None:
            keep_latest = max(1, (int(engine.n) * 3) // 4)
        keep = int(max(1, min(int(keep_latest), engine.current_index + 1)))
        start = int(engine.current_index - keep + 1)
        retained_start_time_s = float(engine.t_s[start])
        engine.t_s = self.compact_axis0_latest(engine.t_s, start=start, count=keep)
        engine.target_reference_orbit_hist = self.compact_axis0_latest(
            engine.target_reference_orbit_hist,
            start=start,
            count=keep,
        )
        for name in ("truth_hist", "belief_hist", "thrust_hist", "torque_hist", "desired_attitude_hist", "throttle_hist"):
            histories = getattr(engine, name)
            setattr(
                engine,
                name,
                {aid: self.compact_axis0_latest(hist, start=start, count=keep) for aid, hist in histories.items()},
            )
        engine.rocket_stage_hist = self.compact_axis0_latest(engine.rocket_stage_hist, start=start, count=keep)
        engine.rocket_q_dyn_hist = self.compact_axis0_latest(engine.rocket_q_dyn_hist, start=start, count=keep)
        engine.rocket_mach_hist = self.compact_axis0_latest(engine.rocket_mach_hist, start=start, count=keep)
        engine.rocket_metric_hists = {
            key: self.compact_axis0_latest(hist, start=start, count=keep)
            for key, hist in engine.rocket_metric_hists.items()
        }
        engine.reentry_metric_hists = {
            aid: {key: self.compact_axis0_latest(hist, start=start, count=keep) for key, hist in metrics.items()}
            for aid, metrics in engine.reentry_metric_hists.items()
        }
        engine.knowledge_hist = {
            obs: {tgt: self.compact_axis0_latest(hist, start=start, count=keep) for tgt, hist in by_tgt.items()}
            for obs, by_tgt in engine.knowledge_hist.items()
        }
        engine.knowledge_measurement_hist = {
            obs: {tgt: self.compact_axis0_latest(hist, start=start, count=keep) for tgt, hist in by_tgt.items()}
            for obs, by_tgt in engine.knowledge_measurement_hist.items()
        }
        engine.controller_debug_hist = {
            aid: self.compact_event_history_latest(rows, retained_start_time_s=retained_start_time_s)
            for aid, rows in engine.controller_debug_hist.items()
        }
        engine.bridge_hist = {
            aid: self.compact_event_history_latest(rows, retained_start_time_s=retained_start_time_s)
            for aid, rows in engine.bridge_hist.items()
        }
        engine.sample_offset += start
        engine.current_index = keep - 1

    def ensure_sample_capacity(self, sample_index: int) -> None:
        engine = self.engine
        needed = int(sample_index) + 1
        if needed <= engine.n:
            return
        grow_to = max(needed, int(max(engine.n * 2, engine.n + 1)))
        if engine.history_mode == "dynamic":
            grow_to = min(grow_to, engine.max_history_samples)
            if needed > grow_to:
                raise RuntimeError("dynamic history compaction did not free space for the next sample.")
        engine.n = grow_to
        engine.t_s = self.grow_axis0(engine.t_s, grow_to)
        engine.target_reference_orbit_hist = self.grow_axis0(engine.target_reference_orbit_hist, grow_to)
        for name in ("truth_hist", "belief_hist", "thrust_hist", "torque_hist", "desired_attitude_hist", "throttle_hist"):
            histories = getattr(engine, name)
            setattr(engine, name, {aid: self.grow_axis0(hist, grow_to) for aid, hist in histories.items()})
        engine.rocket_stage_hist = self.grow_axis0(engine.rocket_stage_hist, grow_to)
        engine.rocket_q_dyn_hist = self.grow_axis0(engine.rocket_q_dyn_hist, grow_to)
        engine.rocket_mach_hist = self.grow_axis0(engine.rocket_mach_hist, grow_to)
        engine.rocket_metric_hists = {
            key: self.grow_axis0(hist, grow_to) for key, hist in engine.rocket_metric_hists.items()
        }
        engine.reentry_metric_hists = {
            aid: {key: self.grow_axis0(hist, grow_to) for key, hist in metrics.items()}
            for aid, metrics in engine.reentry_metric_hists.items()
        }
        engine.knowledge_hist = {
            obs: {tgt: self.grow_axis0(hist, grow_to) for tgt, hist in by_tgt.items()}
            for obs, by_tgt in engine.knowledge_hist.items()
        }
        engine.knowledge_measurement_hist = {
            obs: {tgt: self.grow_axis0(hist, grow_to) for tgt, hist in by_tgt.items()}
            for obs, by_tgt in engine.knowledge_measurement_hist.items()
        }
        engine.history_memory_estimate = engine._estimate_history_memory()
        enforce_history_memory_budget(engine.history_memory_estimate)
