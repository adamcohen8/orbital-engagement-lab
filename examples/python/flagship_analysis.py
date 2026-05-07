from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim import SimulationConfig, SimulationWorkspace


def _as_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _total_dv_m_s(summary: dict[str, Any], object_id: str) -> float:
    thrust_stats = dict(summary.get("thrust_stats", {}) or {})
    object_stats = dict(thrust_stats.get(object_id, {}) or {})
    return _as_float(object_stats.get("total_dv_m_s"))


def build_flagship_metrics(config_path: Path, output_dir: Path | None = None) -> dict[str, Any]:
    workspace = SimulationWorkspace()
    cfg = SimulationConfig.from_yaml(config_path)
    if output_dir is not None:
        cfg = cfg.with_output_dir(output_dir)

    validation = workspace.validate(cfg)
    if not validation["ok"]:
        raise RuntimeError(f"Flagship config validation failed: {validation['errors']}")

    result = workspace.run(cfg)
    pair = result.primary_pair or ("chaser", "target")
    deputy, chief = pair
    rel = result.relative_state(deputy, chief, frame="ric_rect")
    ranges_km = result.range_between(deputy, chief)
    final_range_km = float(ranges_km[-1]) if ranges_km.size else float("nan")
    final_speed_m_s = float(np.linalg.norm(rel[-1, 3:]) * 1000.0) if rel.size else float("nan")

    metrics = {
        "scenario_name": result.summary.get("scenario_name"),
        "deputy": deputy,
        "chief": chief,
        "samples": result.num_steps,
        "duration_s": _as_float(result.summary.get("duration_s")),
        "initial_range_km": float(ranges_km[0]) if ranges_km.size else float("nan"),
        "final_range_km": final_range_km,
        "final_relative_speed_m_s": final_speed_m_s,
        "min_range_km": result.min_range(deputy, chief),
        "time_of_min_range_s": result.time_of_min_range(deputy, chief),
        "chaser_total_dv_m_s": _total_dv_m_s(result.summary, deputy),
        "keepout_1km_violation_count": len(result.keepout_violations(deputy, chief, radius_km=1.0)),
        "close_approach_100m": result.collision_event(deputy, chief, radius_km=0.1),
        "output_dir": str(result.config.scenario.outputs.output_dir),
    }

    analysis_dir = Path(result.config.scenario.outputs.output_dir) / "custom_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = analysis_dir / "flagship_metrics.json"
    csv_path = analysis_dir / "flagship_metrics.csv"

    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, json.dumps(value) if isinstance(value, dict) else value])

    metrics["artifacts"] = {
        "metrics_json": str(metrics_path),
        "metrics_csv": str(csv_path),
    }
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the flagship HCW PD 10 km analysis workflow.")
    parser.add_argument(
        "--config",
        default="configs/hcw_pd_10km_experiment.yaml",
        help="Scenario YAML to run.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output directory override.",
    )
    args = parser.parse_args()

    metrics = build_flagship_metrics(
        Path(args.config),
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
