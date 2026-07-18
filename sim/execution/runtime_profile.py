"""Single-run wall-clock profiling with stable payload formatting."""

from __future__ import annotations

from typing import Any


class _RuntimeProfiler:
    _OBJECT_WALL_STAGES = frozenset(
        {
            "rocket_step",
            "general_propagation_step",
            "satellite_step",
            "bridge_step",
        }
    )

    def __init__(self, *, object_ids: list[str], enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.stage_seconds: dict[str, float] = {}
        self.stage_counts: dict[str, int] = {}
        self.object_stage_seconds: dict[str, dict[str, float]] = {str(oid): {} for oid in object_ids}
        self.object_stage_counts: dict[str, dict[str, int]] = {str(oid): {} for oid in object_ids}

    def record_stage(self, stage: str, elapsed_s: float) -> None:
        if not self.enabled:
            return
        elapsed = float(elapsed_s)
        if elapsed < 0.0:
            return
        key = str(stage)
        self.stage_seconds[key] = float(self.stage_seconds.get(key, 0.0) + elapsed)
        self.stage_counts[key] = int(self.stage_counts.get(key, 0) + 1)

    def record_object(self, object_id: str, stage: str, elapsed_s: float) -> None:
        if not self.enabled:
            return
        elapsed = float(elapsed_s)
        if elapsed < 0.0:
            return
        oid = str(object_id)
        key = str(stage)
        by_stage = self.object_stage_seconds.setdefault(oid, {})
        by_stage[key] = float(by_stage.get(key, 0.0) + elapsed)
        by_count = self.object_stage_counts.setdefault(oid, {})
        by_count[key] = int(by_count.get(key, 0) + 1)

    def payload(self, *, completed_steps: int, object_count: int) -> dict[str, Any]:
        steps = int(max(completed_steps, 0))
        total_step_wall_s = float(self.stage_seconds.get("step_wall", 0.0))
        stage_totals = {
            key: {
                "total_s": float(value),
                "count": int(self.stage_counts.get(key, 0)),
                "mean_ms": _mean_ms(value, self.stage_counts.get(key, 0)),
                "share_of_step_wall": _safe_share(value, total_step_wall_s),
            }
            for key, value in sorted(self.stage_seconds.items())
        }
        object_totals: dict[str, Any] = {}
        for oid, by_stage in sorted(self.object_stage_seconds.items()):
            wall_total = float(
                sum(value for key, value in by_stage.items() if key in self._OBJECT_WALL_STAGES)
            )
            nested_total = float(
                sum(value for key, value in by_stage.items() if key not in self._OBJECT_WALL_STAGES)
            )
            object_totals[oid] = {
                "total_s": wall_total,
                "mean_ms_per_completed_step": _mean_ms(wall_total, steps),
                "nested_stage_total_s": nested_total,
                "stages": {
                    key: {
                        "total_s": float(value),
                        "count": int(self.object_stage_counts.get(oid, {}).get(key, 0)),
                        "mean_ms": _mean_ms(value, self.object_stage_counts.get(oid, {}).get(key, 0)),
                    }
                    for key, value in sorted(by_stage.items())
                },
            }
        slowest = sorted(
            (
                {"object_id": oid, "total_s": float(data["total_s"])}
                for oid, data in object_totals.items()
                if float(data["total_s"]) > 0.0
            ),
            key=lambda item: (-float(item["total_s"]), str(item["object_id"])),
        )[:10]
        return {
            "schema_version": 1,
            "enabled": bool(self.enabled),
            "completed_steps": steps,
            "object_count": int(object_count),
            "total_step_wall_s": total_step_wall_s,
            "mean_step_wall_ms": _mean_ms(total_step_wall_s, steps),
            "stage_totals": stage_totals,
            "object_totals": object_totals,
            "slowest_objects": slowest,
            "notes": [
                "Profiler timings use wall-clock perf_counter measurements inside the single-run engine.",
                "Object total_s is the non-overlapping object wall time; nested_stage_total_s is diagnostic detail.",
            ],
        }


def _mean_ms(total_s: float, count: int | None) -> float:
    n = int(count or 0)
    return 0.0 if n <= 0 else float(total_s) * 1000.0 / float(n)


def _safe_share(value: float, total: float) -> float:
    denom = float(total)
    return 0.0 if denom <= 0.0 else float(value) / denom
