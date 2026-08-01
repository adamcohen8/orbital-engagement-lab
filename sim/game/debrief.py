from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from sim.game.training import RPOTrainingConfig
from sim.platform_compat import open_folder
from sim.plotting.style import OELArtifactMetadata, artifact_metadata, role_color, save_oel_figure


def game_debrief_path(
    *,
    scenario_id: str,
    difficulty: str,
    attempt_index: int,
    output_dir: str | Path | None = None,
    timestamp: datetime | None = None,
) -> Path:
    return game_debrief_attempt_dir(
        scenario_id=scenario_id,
        difficulty=difficulty,
        attempt_index=attempt_index,
        output_dir=output_dir,
        timestamp=timestamp,
    ) / "report.md"


def game_debrief_attempt_dir(
    *,
    scenario_id: str,
    difficulty: str,
    attempt_index: int,
    output_dir: str | Path | None = None,
    timestamp: datetime | None = None,
) -> Path:
    root = Path(output_dir) if output_dir is not None else Path("outputs") / "game_debriefs"
    scenario = _slug(scenario_id or "game")
    diff = _slug(difficulty or "easy")
    stamp = (timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return root / scenario / f"attempt_{max(int(attempt_index), 1):03d}_{diff}_{stamp}"


def next_game_debrief_attempt_index(
    *,
    scenario_id: str,
    output_dir: str | Path | None = None,
) -> int:
    root = Path(output_dir) if output_dir is not None else Path("outputs") / "game_debriefs"
    level_dir = root / _slug(scenario_id or "game")
    if not level_dir.exists():
        return 1
    attempts: list[int] = []
    for path in level_dir.iterdir():
        if not path.is_dir():
            continue
        match = re.match(r"attempt_(\d+)", path.name)
        if match:
            attempts.append(int(match.group(1)))
    return max(attempts, default=0) + 1


def write_game_debrief(
    path: str | Path,
    *,
    config: RPOTrainingConfig,
    score: Any,
    difficulty: str,
    objective_checklist: tuple[str, ...] = (),
    arcade_score: int = 0,
    arcade_seed: int | None = None,
    arcade_round_index: int | None = None,
    recording_path: str | Path | None = None,
    replay_history: dict[str, Any] | None = None,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    attempt_dir = output.parent
    summary_path = attempt_dir / "summary.json"
    plots_dir = attempt_dir / "plots"
    payload = game_debrief_payload(
        config=config,
        score=score,
        difficulty=difficulty,
        objective_checklist=objective_checklist,
        arcade_score=arcade_score,
        arcade_seed=arcade_seed,
        arcade_round_index=arcade_round_index,
        recording_path=recording_path,
        replay_history=replay_history,
    )
    plot_paths = write_game_debrief_plots(
        plots_dir,
        config=config,
        event_timeline=payload["event_timeline"],
        replay_history=replay_history or {},
    )
    payload["artifacts"]["report_path"] = str(output)
    payload["artifacts"]["summary_path"] = str(summary_path)
    payload["artifacts"]["plot_paths"] = {key: str(path) for key, path in plot_paths.items()}
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.write_text(
        _markdown_report(
            payload,
            plot_paths=plot_paths,
            report_path=output,
        ),
        encoding="utf-8",
    )
    return output


def game_debrief_payload(
    *,
    config: RPOTrainingConfig,
    score: Any,
    difficulty: str,
    objective_checklist: tuple[str, ...] = (),
    arcade_score: int = 0,
    arcade_seed: int | None = None,
    arcade_round_index: int | None = None,
    recording_path: str | Path | None = None,
    replay_history: dict[str, Any] | None = None,
) -> dict[str, Any]:
    replay = replay_history or {}
    replay_metrics = _replay_metrics(replay)
    return {
        "schema_version": 1,
        "scenario_id": str(config.scenario_id or getattr(score, "scenario_id", "") or ""),
        "display_title": _debrief_display_title(config=config, score=score),
        "learning_goal": str(config.learning_goal or getattr(score, "learning_goal", "") or ""),
        "difficulty": str(difficulty or "easy"),
        "level_passed": bool(getattr(score, "level_passed", False)),
        "level_failed": bool(getattr(score, "level_failed", False)),
        "score": {
            "arcade_score": int(max(arcade_score, 0)),
            "arcade_seed": None if arcade_seed is None else int(arcade_seed),
            "arcade_round_index": None if arcade_round_index is None else int(arcade_round_index),
        },
        "metrics": {
            "samples": int(getattr(score, "samples", 0)),
            "elapsed_s": _float_or_none(getattr(score, "elapsed_s", None)),
            "achieved_time_s": _float_or_none(getattr(score, "achieved_time_s", None)),
            "closest_approach_km": _float_or_none(getattr(score, "closest_approach_km", None)),
            "final_range_km": _float_or_none(getattr(score, "final_range_km", None)),
            "final_goal_error_km": _float_or_none(getattr(score, "final_goal_error_km", None)),
            "final_relative_speed_km_s": _float_or_none(getattr(score, "final_relative_speed_km_s", None)),
            "time_inside_keepout_s": _float_or_none(getattr(score, "time_inside_keepout_s", None)),
            "approximate_delta_v_m_s": _float_or_none(getattr(score, "approximate_delta_v_m_s", None)),
            "target_delta_v_m_s": _float_or_none(getattr(score, "target_delta_v_m_s", None)),
            "min_sun_angle_deg": _float_or_none(getattr(score, "min_sun_angle_deg", None)),
            "final_sun_angle_deg": _float_or_none(getattr(score, "final_sun_angle_deg", None)),
            "sun_angle_violation_time_s": _float_or_none(getattr(score, "sun_angle_violation_time_s", None)),
            "min_goal_error_km": _float_or_none(getattr(score, "min_goal_error_km", None)),
            "initial_range_km": replay_metrics["initial_range_km"],
            "max_relative_speed_km_s": replay_metrics["max_relative_speed_km_s"],
            "mean_relative_speed_km_s": replay_metrics["mean_relative_speed_km_s"],
            "largest_command_accel_m_s2": replay_metrics["largest_command_accel_m_s2"],
            "burn_count": replay_metrics["burn_count"],
            "active_control_time_s": replay_metrics["active_control_time_s"],
            "remaining_delta_v_m_s": _remaining_budget(
                getattr(config, "max_delta_v_m_s", None), getattr(score, "approximate_delta_v_m_s", None)
            ),
            "remaining_target_delta_v_m_s": _remaining_budget(
                getattr(config, "max_target_delta_v_m_s", None), getattr(score, "target_delta_v_m_s", None)
            ),
        },
        "violations": {
            "keepout": bool(getattr(score, "keepout_violation", False)),
            "hard_speed_limit": bool(getattr(score, "hard_speed_limit_violation", False)),
            "forbidden_region": bool(getattr(score, "forbidden_region_violation", False)),
            "forbidden_region_names": list(getattr(score, "forbidden_region_names", ()) or ()),
            "sun_angle": bool(getattr(score, "sun_angle_violation", False)),
            "sun_angle_constraint_names": list(getattr(score, "sun_angle_constraint_names", ()) or ()),
            "approach_gate": bool(getattr(score, "approach_gate_violation", False)),
            "approach_gate_names": list(getattr(score, "approach_gate_names", ()) or ()),
        },
        "objectives": {
            "checklist": list(objective_checklist),
            "pass_fail_reasons": list(getattr(score, "pass_fail_reasons", ()) or ()),
            "hints": list(getattr(score, "hints", ()) or ()),
            "burn_axes_satisfied": list(getattr(score, "burn_axes_satisfied", ()) or ()),
            "speed_multiplier_changed": bool(getattr(score, "speed_multiplier_changed", False)),
            "approach_gates_satisfied": int(getattr(score, "approach_gates_satisfied", 0)),
            "approach_gates_total": int(getattr(score, "approach_gates_total", 0)),
            "inspection_gates_satisfied": int(getattr(score, "inspection_gates_satisfied", 0)),
            "inspection_gates_total": int(getattr(score, "inspection_gates_total", 0)),
            "inspection_gate_names": list(getattr(score, "inspection_gate_names", ()) or ()),
        },
        "artifacts": {
            "recording_path": None if recording_path is None else str(recording_path),
        },
        "event_timeline": _event_timeline(config=config, score=score, replay_history=replay),
        "replay": replay,
    }


def tracker_replay_history(tracker: Any) -> dict[str, Any]:
    replay = tracker.replay_history() if hasattr(tracker, "replay_history") else {}
    if replay:
        return {
            "time_s": _array_list(replay.get("time_s", [])),
            "relative_ric": _array_list(replay.get("relative_ric", [])),
            "chaser_thrust_ric_km_s2": _array_list(replay.get("chaser_thrust_ric_km_s2", [])),
            "target_thrust_eci_km_s2": _array_list(replay.get("target_thrust_eci_km_s2", [])),
        }
    return {
        "time_s": _array_list(getattr(tracker, "t_s", [])),
        "relative_ric": _array_list(getattr(tracker, "rel_ric_hist", [])),
        "chaser_thrust_ric_km_s2": _array_list(getattr(tracker, "thrust_ric_hist", [])),
        "target_thrust_eci_km_s2": _array_list(getattr(tracker, "target_thrust_hist", [])),
    }


def write_game_debrief_plots(
    output_dir: str | Path,
    *,
    config: RPOTrainingConfig,
    replay_history: dict[str, Any],
    event_timeline: list[dict[str, Any]] = (),
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Polygon, Rectangle
    except Exception as exc:
        (output / "plot_error.txt").write_text(f"Could not create matplotlib plots: {exc}\n", encoding="utf-8")
        return {}

    t, rel, thrust = _history_arrays_from_replay(replay_history)
    if t.size == 0 or rel.shape[0] == 0:
        return {}

    paths: dict[str, Path] = {}
    metadata = artifact_metadata(
        scenario_name=str(getattr(config, "scenario_id", "") or getattr(config, "name", "") or "game_debrief")
    )
    plot_style = {
        "trajectory": role_color("actual", style_name="oel_dark"),
        "start": role_color("coast", style_name="oel_dark"),
        "end": role_color("chaser", style_name="oel_dark"),
        "burn": role_color("burn", style_name="oel_dark"),
        "keepout": role_color("warning", style_name="oel_dark"),
        "goal": role_color("desired", style_name="oel_dark"),
        "forbidden": role_color("warning", style_name="oel_dark"),
        "gate": role_color("safety_zone", style_name="oel_dark"),
    }

    timeline_path = _save_event_timeline_plot(
        plt,
        output / "mission_timeline.png",
        event_timeline,
        elapsed_s=float(t[-1] - t[0]) if t.size >= 2 else 0.0,
        metadata=metadata,
    )
    if timeline_path is not None:
        paths["mission_timeline"] = timeline_path

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.6), constrained_layout=False)
    for ax, plane in zip(axes, ("RI", "RC", "IC")):
        x_idx, y_idx, x_label, y_label = _plane_axes(plane)
        ax.plot(rel[:, x_idx], rel[:, y_idx], color=plot_style["trajectory"], linewidth=1.8, label="Trajectory")
        ax.scatter(rel[0, x_idx], rel[0, y_idx], color=plot_style["start"], marker="o", s=38, label="Start", zorder=4)
        ax.scatter(rel[-1, x_idx], rel[-1, y_idx], color=plot_style["end"], marker="x", s=54, label="End", zorder=4)
        burn_idx = _burn_sample_indices(thrust)
        if burn_idx.size:
            ax.scatter(
                rel[burn_idx, x_idx],
                rel[burn_idx, y_idx],
                color=plot_style["burn"],
                marker=".",
                s=18,
                alpha=0.8,
                label="Burn samples",
                zorder=3,
            )
        _draw_geometry_overlays(ax, config=config, plane=plane, patches=(Circle, Polygon, Rectangle), colors=plot_style)
        ax.set_title(f"{plane} Plane")
        ax.set_xlabel(f"{x_label} position (km)")
        ax.set_ylabel(f"{y_label} position (km)")
        ax.grid(True, alpha=0.28)
        ax.axis("equal")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.015),
            ncols=max(len(labels), 1),
            frameon=False,
            fontsize=8,
            handlelength=1.8,
            columnspacing=1.1,
        )
    fig.suptitle("2D RIC Trajectory", y=0.98)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.84, bottom=0.24, wspace=0.20)
    paths["ric_2d"] = output / "ric_2d_plots.png"
    save_oel_figure(
        fig,
        paths["ric_2d"],
        dpi=160,
        metadata=metadata,
        artifact_id=paths["ric_2d"].stem,
        style_name="oel_dark",
    )
    plt.close(fig)

    range_km = np.linalg.norm(rel[:, :3], axis=1)
    rel_speed_m_s = np.linalg.norm(rel[:, 3:6], axis=1) * 1000.0
    cumulative_dv_m_s = _cumulative_delta_v_m_s(thrust, t)
    thrust_m_s2 = thrust * 1000.0

    paths["relative_range"] = _save_time_plot(
        plt,
        output / "relative_range_vs_time.png",
        t,
        [(range_km, "Relative range", "#1f77b4")],
        title="Relative Range vs Time",
        ylabel="Range (km)",
        metadata=metadata,
    )
    paths["relative_velocity"] = _save_time_plot(
        plt,
        output / "relative_velocity_vs_time.png",
        t,
        [(rel_speed_m_s, "Relative speed", "#d62728")],
        title="Relative Velocity vs Time",
        ylabel="Speed (m/s)",
        metadata=metadata,
    )
    paths["cumulative_delta_v"] = _save_time_plot(
        plt,
        output / "cumulative_delta_v_vs_time.png",
        t,
        [(cumulative_dv_m_s, "Cumulative dV", "#2ca02c")],
        title="Cumulative Delta V vs Time",
        ylabel="Delta V (m/s)",
        metadata=metadata,
    )
    paths["control_commands"] = _save_time_plot(
        plt,
        output / "control_commands_vs_time.png",
        t,
        [
            (thrust_m_s2[:, 0], "R command", "#1f77b4"),
            (thrust_m_s2[:, 1], "I command", "#ff7f0e"),
            (thrust_m_s2[:, 2], "C command", "#2ca02c"),
        ],
        title="Control Commands vs Time",
        ylabel="Applied acceleration (m/s^2)",
        metadata=metadata,
    )
    return paths


def open_game_debrief_folder(path: str | Path | None) -> bool:
    if path is None:
        return False
    target = Path(path)
    folder = target if target.is_dir() else target.parent
    if not folder.exists():
        return False
    try:
        open_folder(folder)
    except Exception:
        return False
    return True


def _markdown_report(
    payload: dict[str, Any],
    *,
    plot_paths: dict[str, Path],
    report_path: Path,
) -> str:
    display_title = str(payload.get("display_title") or "").strip()
    if not display_title:
        display_title = _debrief_title_from_scenario_id(str(payload.get("scenario_id") or "Game Debrief"))
    passed = bool(payload.get("level_passed"))
    failed = bool(payload.get("level_failed"))
    outcome = "PASS" if passed else "FAIL" if failed else "INCOMPLETE"
    reasons = list(payload.get("objectives", {}).get("pass_fail_reasons", []) or [])
    metrics = dict(payload.get("metrics", {}) or {})
    stats = [
        ("Difficulty", str(payload.get("difficulty", ""))),
        ("Arcade Score", _int_text(payload.get("score", {}).get("arcade_score"))),
        ("Elapsed Time", _seconds_text(metrics.get("elapsed_s"))),
        ("Time to Requirements", _seconds_text(metrics.get("achieved_time_s"))),
        ("Initial Range", _km_text(metrics.get("initial_range_km"))),
        ("Closest Approach", _km_text(metrics.get("closest_approach_km"))),
        ("Final Range", _km_text(metrics.get("final_range_km"))),
        ("Best Goal Error", _km_text(metrics.get("min_goal_error_km"))),
        ("Final Relative Speed", _speed_text(metrics.get("final_relative_speed_km_s"))),
        ("Max Relative Speed", _speed_text(metrics.get("max_relative_speed_km_s"))),
        ("Delta V Used", _meters_per_second_text(metrics.get("approximate_delta_v_m_s"))),
        ("Remaining Delta V", _meters_per_second_text(metrics.get("remaining_delta_v_m_s"))),
        ("Target Delta V Used", _meters_per_second_text(metrics.get("target_delta_v_m_s"))),
        ("Burn Count", _int_text(metrics.get("burn_count"))),
        ("Active Control Time", _seconds_text(metrics.get("active_control_time_s"))),
        ("Largest Command", _accel_text(metrics.get("largest_command_accel_m_s2"))),
        ("Keepout Time", _seconds_text(metrics.get("time_inside_keepout_s"))),
    ]
    stats_rows = "\n".join(
        f"| {_md_table_text(label)} | {_md_table_text(value)} |"
        for label, value in stats
        if value and value != "--"
    )
    checklist = list(payload.get("objectives", {}).get("checklist", []) or [])
    plot_blocks = "\n\n".join(
        f"## {_md_text(_plot_title(key))}\n\n![{_md_alt_text(_plot_title(key))}]({path.relative_to(report_path.parent).as_posix()})"
        for key, path in plot_paths.items()
    )
    plot_note = "" if plot_blocks else "No plots were generated for this attempt.\n"
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    failure_reason = "None" if passed else "; ".join(str(reason) for reason in reasons[:3]) or "Not recorded"
    reason_lines = _md_bullets(reasons, empty="None recorded.")
    checklist_lines = _md_bullets(checklist, empty="None recorded.")
    return f"""# {_md_text(display_title)}

Generated: {generated}<br>
Difficulty: {_md_text(str(payload.get("difficulty", "")))}<br>
Outcome: **{outcome}**

## Outcome

**Pass/Failure:** {outcome}<br>
**Failure reason:** {_md_text(failure_reason)}

### Reasons

{reason_lines}

## Stats Summary

| Metric | Value |
| --- | --- |
{stats_rows}

## Objective Checklist

{checklist_lines}

{plot_note}{plot_blocks}
"""


def _history_arrays_from_replay(replay_history: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.array(replay_history.get("time_s", []), dtype=float).reshape(-1)
    rel = np.array(replay_history.get("relative_ric", []), dtype=float)
    thrust = np.array(replay_history.get("chaser_thrust_ric_km_s2", []), dtype=float)
    if rel.size == 0:
        rel = np.zeros((0, 6), dtype=float)
    if rel.ndim == 1:
        rel = rel.reshape(1, -1)
    if rel.shape[1] < 6:
        rel = np.pad(rel, ((0, 0), (0, 6 - rel.shape[1])), constant_values=np.nan)
    if thrust.size == 0:
        thrust = np.zeros((rel.shape[0], 3), dtype=float)
    if thrust.ndim == 1:
        thrust = thrust.reshape(1, -1)
    if thrust.shape[1] < 3:
        thrust = np.pad(thrust, ((0, 0), (0, 3 - thrust.shape[1])), constant_values=0.0)
    n = min(t.size, rel.shape[0], thrust.shape[0])
    return t[:n], rel[:n, :6], thrust[:n, :3]


def _replay_metrics(replay_history: dict[str, Any]) -> dict[str, float | int | None]:
    t, rel, thrust = _history_arrays_from_replay(replay_history)
    if t.size == 0 or rel.shape[0] == 0:
        return {
            "initial_range_km": None,
            "max_relative_speed_km_s": None,
            "mean_relative_speed_km_s": None,
            "largest_command_accel_m_s2": None,
            "burn_count": 0,
            "active_control_time_s": 0.0,
        }
    ranges = np.linalg.norm(rel[:, :3], axis=1)
    speeds = np.linalg.norm(rel[:, 3:6], axis=1)
    thrust_norm = np.linalg.norm(thrust, axis=1)
    active = thrust_norm > _burn_threshold_km_s2(thrust)
    return {
        "initial_range_km": _float_or_none(ranges[0]),
        "max_relative_speed_km_s": _float_or_none(np.nanmax(speeds)),
        "mean_relative_speed_km_s": _float_or_none(np.nanmean(speeds)),
        "largest_command_accel_m_s2": _float_or_none(np.nanmax(thrust_norm) * 1000.0),
        "burn_count": int(_burn_segments(active)),
        "active_control_time_s": _float_or_none(_active_time_s(active, t)),
    }


def _event_timeline(*, config: RPOTrainingConfig, score: Any, replay_history: dict[str, Any]) -> list[dict[str, Any]]:
    t, _, thrust = _history_arrays_from_replay(replay_history)
    events: list[dict[str, Any]] = []
    if t.size:
        events.append({"time_s": float(t[0]), "label": "Attempt started."})
    active = np.linalg.norm(thrust, axis=1) > _burn_threshold_km_s2(thrust) if thrust.size else np.zeros(0, dtype=bool)
    for start_idx, end_idx in _active_segments(active)[:12]:
        if int(end_idx) <= 0:
            continue
        start_sample_idx = max(int(start_idx) - 1, 0)
        end_sample_idx = min(int(end_idx), t.size - 1)
        start_t = float(t[start_sample_idx])
        end_t = float(t[end_sample_idx])
        if not np.isfinite(end_t) or end_t <= start_t:
            end_t = start_t
        events.append(
            {
                "time_s": start_t,
                "start_time_s": start_t,
                "end_time_s": end_t,
                "label": "Control input",
                "kind": "interval",
            }
        )
    achieved = _float_or_none(getattr(score, "achieved_time_s", None))
    if achieved is not None:
        events.append({"time_s": achieved, "label": "Mission requirements first satisfied."})
    for name in list(getattr(score, "inspection_gate_names", ()) or ())[:8]:
        events.append({"time_s": None, "label": f"Inspection gate completed: {name}."})
    for reason in list(getattr(score, "pass_fail_reasons", ()) or ())[:5]:
        events.append({"time_s": _float_or_none(getattr(score, "elapsed_s", None)), "label": str(reason)})
    events.sort(key=_event_sort_time)
    return events


def _event_sort_time(event: dict[str, Any]) -> float:
    value = _float_or_none(event.get("time_s"))
    return float("inf") if value is None else float(value)


def _draw_geometry_overlays(ax: Any, *, config: RPOTrainingConfig, plane: str, patches: tuple[Any, Any, Any], colors: dict[str, str]) -> None:
    Circle, Polygon, Rectangle = patches
    x_idx, y_idx, _, _ = _plane_axes(plane)
    if config.keepout_radius_km is not None:
        ax.add_patch(Circle((0.0, 0.0), float(config.keepout_radius_km), fill=False, color=colors["keepout"], linestyle="--", linewidth=1.2, label="Keepout"))
    if config.goal_range_km is not None:
        ax.add_patch(Circle((0.0, 0.0), float(config.goal_range_km), fill=False, color=colors["goal"], linestyle="-.", linewidth=1.2, label="Goal range"))
    if config.goal_radius_km is not None:
        center = np.array(config.goal_relative_ric_km, dtype=float).reshape(3)
        ax.add_patch(Circle((center[x_idx], center[y_idx]), float(config.goal_radius_km), fill=False, color=colors["goal"], linestyle="-.", linewidth=1.2, label="Goal"))
    if config.goal_nmt_radial_amplitude_km is not None:
        try:
            from sim.game.training import nmt_curve_points_km

            curve = nmt_curve_points_km(
                radial_amplitude_km=float(config.goal_nmt_radial_amplitude_km),
                cross_track_amplitude_km=float(config.goal_nmt_cross_track_amplitude_km),
                cross_track_phase_deg=float(config.goal_nmt_cross_track_phase_deg),
                center_ric_km=np.array(config.goal_nmt_center_ric_km, dtype=float),
            )
            ax.plot(curve[:, x_idx], curve[:, y_idx], color=colors["goal"], linestyle="--", linewidth=1.2, label="NMT goal")
        except Exception:
            pass
    for region in config.forbidden_regions:
        _draw_forbidden_region(ax, region=region, plane=plane, patches=(Polygon, Rectangle), color=colors["forbidden"])
    for gate in config.approach_gates:
        if x_idx == 0:
            ax.axvspan(
                float(gate.radial_ric_km) - float(gate.radial_tolerance_km),
                float(gate.radial_ric_km) + float(gate.radial_tolerance_km),
                color=colors["gate"],
                alpha=0.12,
            )
        elif y_idx == 0:
            ax.axhspan(
                float(gate.radial_ric_km) - float(gate.radial_tolerance_km),
                float(gate.radial_ric_km) + float(gate.radial_tolerance_km),
                color=colors["gate"],
                alpha=0.12,
            )
    for gate in config.inspection_gates:
        center = np.array(gate.center_ric_km, dtype=float).reshape(3)
        half = np.array(gate.half_width_ric_km, dtype=float).reshape(3)
        ax.add_patch(
            Rectangle(
                (center[x_idx] - half[x_idx], center[y_idx] - half[y_idx]),
                2.0 * half[x_idx],
                2.0 * half[y_idx],
                fill=False,
                color=colors["goal"],
                linestyle=":",
                linewidth=1.0,
            )
        )


def _draw_forbidden_region(ax: Any, *, region: Any, plane: str, patches: tuple[Any, Any], color: str) -> None:
    Polygon, Rectangle = patches
    x_idx, y_idx, _, _ = _plane_axes(plane)
    kind = str(getattr(region, "kind", "box") or "box")
    if kind == "annular_sector":
        pts = region.sector_polygon_ric()
        if pts.size:
            ax.add_patch(Polygon(pts[:, [x_idx, y_idx]], closed=True, color=color, alpha=0.16))
        return
    if kind == "cylinder":
        center = np.array(getattr(region, "center_ric_km", np.zeros(3)), dtype=float).reshape(3)
        radius = getattr(region, "radius_km", None)
        height = getattr(region, "height_km", None)
        if radius is None or height is None:
            return
        axis = _axis_index(str(getattr(region, "axis", "I") or "I"))
        if axis not in {x_idx, y_idx}:
            from matplotlib.patches import Circle

            ax.add_patch(Circle((center[x_idx], center[y_idx]), float(radius), color=color, alpha=0.16))
            return
        half = np.array([float(radius), float(radius), float(radius)], dtype=float)
        half[axis] = float(height) / 2.0
        ax.add_patch(
            Rectangle(
                (center[x_idx] - half[x_idx], center[y_idx] - half[y_idx]),
                2.0 * half[x_idx],
                2.0 * half[y_idx],
                color=color,
                alpha=0.16,
            )
        )
        return
    lower = np.array(getattr(region, "min_ric_km", np.full(3, -np.inf)), dtype=float).reshape(3)
    upper = np.array(getattr(region, "max_ric_km", np.full(3, np.inf)), dtype=float).reshape(3)
    if np.all(np.isfinite([lower[x_idx], lower[y_idx], upper[x_idx], upper[y_idx]])):
        ax.add_patch(
            Rectangle(
                (lower[x_idx], lower[y_idx]),
                upper[x_idx] - lower[x_idx],
                upper[y_idx] - lower[y_idx],
                color=color,
                alpha=0.16,
            )
        )


def _save_time_plot(
    plt: Any,
    path: Path,
    t_s: np.ndarray,
    series: list[tuple[np.ndarray, str, str]],
    *,
    title: str,
    ylabel: str,
    metadata: OELArtifactMetadata | None = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=False)
    for values, label, color in series:
        ax.plot(t_s, np.array(values, dtype=float), label=label, color=color, linewidth=1.6)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.28)
    if len(series) > 1:
        ax.legend(loc="best")
    fig.subplots_adjust(left=0.11, right=0.975, top=0.86, bottom=0.24)
    save_oel_figure(
        fig,
        path,
        dpi=160,
        metadata=metadata,
        artifact_id=path.stem,
        style_name="oel_dark",
    )
    plt.close(fig)
    return path


def _save_event_timeline_plot(
    plt: Any,
    path: Path,
    events: list[dict[str, Any]],
    *,
    elapsed_s: float,
    metadata: OELArtifactMetadata | None = None,
) -> Path | None:
    from matplotlib.lines import Line2D

    finite_events = [
        {
            "time_s": float(event.get("time_s")),
            "start_time_s": _float_or_none(event.get("start_time_s")),
            "end_time_s": _float_or_none(event.get("end_time_s")),
            "label": str(event.get("label", "") or "").strip(),
            "kind": str(event.get("kind", "point") or "point").strip().lower(),
        }
        for event in events
        if event.get("time_s") is not None and _float_or_none(event.get("time_s")) is not None
    ]
    if not finite_events:
        return None
    max_time = max(
        float(elapsed_s),
        *(float(event["time_s"]) for event in finite_events),
        *(float(event["end_time_s"]) for event in finite_events if event.get("end_time_s") is not None),
        1.0,
    )
    point_events = [event for event in finite_events if event["kind"] != "interval"]
    interval_events = [event for event in finite_events if event["kind"] == "interval"]
    fig_height = max(3.6, min(5.4, 2.6 + 0.16 * len(point_events)))
    fig, ax = plt.subplots(figsize=(11, fig_height), constrained_layout=False)
    timeline_y = 0.0
    interval_y = -0.18
    ax.hlines(timeline_y, 0.0, max_time, color="#CBD5E1", linewidth=2.4, zorder=1)
    for event in interval_events:
        start_t = float(event["start_time_s"] if event.get("start_time_s") is not None else event["time_s"])
        end_t = float(event["end_time_s"] if event.get("end_time_s") is not None else event["time_s"])
        if end_t <= start_t:
            end_t = start_t + max(0.01 * max_time, 1.0)
        width = min(end_t, max_time) - max(start_t, 0.0)
        if width <= 0.0:
            continue
        ax.broken_barh(
            [(max(start_t, 0.0), width)],
            (interval_y - 0.055, 0.11),
            facecolors="#ff7f0e",
            edgecolors="#bf5b13",
            alpha=0.86,
            linewidth=0.8,
            zorder=3,
        )
    for idx, event in enumerate(point_events):
        time_s = float(event["time_s"])
        lane_y = 0.46 if idx % 2 == 0 else 0.78
        marker_color = "#2ca02c" if "satisfied" in event["label"].lower() or "criteria" in event["label"].lower() else "#1f77b4"
        ax.vlines(time_s, timeline_y, lane_y - 0.06, color=marker_color, linewidth=1.2, alpha=0.8, zorder=2)
        ax.scatter([time_s], [timeline_y], color=marker_color, s=34, zorder=3)
        label = _timeline_label(event["label"])
        if time_s > max_time * 0.88:
            ha = "right"
            x_offset = -8
        elif time_s < max_time * 0.12:
            ha = "left"
            x_offset = 8
        else:
            ha = "center"
            x_offset = 0
        ax.annotate(
            f"{time_s:.0f}s\n{label}",
            xy=(time_s, lane_y),
            xytext=(x_offset, 0),
            textcoords="offset points",
            ha=ha,
            va="bottom",
            fontsize=8,
            color="#E5E7EB",
            rotation=0,
            clip_on=False,
        )
    ax.set_title("Mission Timeline")
    ax.set_xlabel("Time (s)")
    ax.set_ylim(-0.42, 1.05)
    ax.set_yticks([])
    ax.set_xlim(-0.02 * max_time, 1.08 * max_time)
    ax.grid(True, axis="x", alpha=0.25)
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)
    if interval_events:
        ax.legend(
            handles=[
                Line2D([0], [0], color="#ff7f0e", linewidth=4, label="Control input intervals"),
                Line2D([0], [0], marker="o", color="none", markerfacecolor="#1f77b4", markersize=6, label="Mission events"),
            ],
            loc="upper left",
            frameon=False,
        )
    fig.subplots_adjust(left=0.055, right=0.97, top=0.82, bottom=0.24)
    save_oel_figure(
        fig,
        path,
        dpi=160,
        metadata=metadata,
        artifact_id=path.stem,
        style_name="oel_dark",
    )
    plt.close(fig)
    return path


def _timeline_label(value: str) -> str:
    text = _md_text(value)
    replacements = {
        "Attempt started.": "Start",
        "Control input": "Burn",
        "Mission requirements first satisfied.": "Requirements met",
        "All pass criteria satisfied.": "Pass criteria satisfied",
    }
    return replacements.get(text, text[:42] + "..." if len(text) > 45 else text)


def _plane_axes(plane: str) -> tuple[int, int, str, str]:
    mapping = {
        "RI": (1, 0, "I", "R"),
        "RC": (2, 0, "C", "R"),
        "IC": (1, 2, "I", "C"),
    }
    return mapping.get(str(plane or "RI").upper(), mapping["RI"])


def _axis_index(axis: str) -> int:
    return {"R": 0, "I": 1, "C": 2}.get(str(axis or "I").upper(), 1)


def _burn_threshold_km_s2(thrust: np.ndarray) -> float:
    if thrust.size == 0:
        return 1.0e-12
    peak = float(np.nanmax(np.linalg.norm(thrust, axis=1)))
    return max(peak * 1.0e-4, 1.0e-12)


def _burn_sample_indices(thrust: np.ndarray) -> np.ndarray:
    if thrust.size == 0:
        return np.zeros(0, dtype=int)
    return np.flatnonzero(np.linalg.norm(thrust, axis=1) > _burn_threshold_km_s2(thrust))


def _burn_segments(active: np.ndarray) -> int:
    if active.size == 0:
        return 0
    return int(np.count_nonzero(active & ~np.r_[False, active[:-1]]))


def _active_segments(active: np.ndarray) -> list[tuple[int, int]]:
    flags = np.array(active, dtype=bool).reshape(-1)
    if flags.size == 0:
        return []
    starts = np.flatnonzero(flags & ~np.r_[False, flags[:-1]])
    ends = np.flatnonzero(flags & ~np.r_[flags[1:], False])
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _active_time_s(active: np.ndarray, t_s: np.ndarray) -> float:
    if active.size < 2 or t_s.size < 2:
        return 0.0
    dt = np.diff(t_s)
    valid = np.isfinite(dt) & (dt > 0.0)
    # Snapshot i reports the command applied during the interval ending at t[i].
    return float(np.sum(dt[valid] * active[1:][valid]))


def _cumulative_delta_v_m_s(thrust_km_s2: np.ndarray, t_s: np.ndarray) -> np.ndarray:
    if thrust_km_s2.shape[0] == 0:
        return np.zeros(0, dtype=float)
    dv = np.zeros(thrust_km_s2.shape[0], dtype=float)
    if thrust_km_s2.shape[0] < 2 or t_s.size < 2:
        return dv
    norms = np.linalg.norm(thrust_km_s2, axis=1)
    dt = np.diff(t_s)
    n = min(norms.size, t_s.size)
    increments = np.zeros(max(n - 1, 0), dtype=float)
    valid = np.isfinite(norms[1:n]) & np.isfinite(dt[: n - 1]) & (dt[: n - 1] > 0.0)
    increments[valid] = norms[1:n][valid] * dt[: n - 1][valid] * 1000.0
    dv[1:n] = np.cumsum(increments)
    if n < dv.size:
        dv[n:] = dv[n - 1]
    return dv


def _remaining_budget(budget_m_s: Any, used_m_s: Any) -> float | None:
    if budget_m_s is None or used_m_s is None:
        return None
    return _float_or_none(max(float(budget_m_s) - float(used_m_s), 0.0))


def _plot_title(key: str) -> str:
    return {
        "ric_2d": "2D RIC Plots",
        "mission_timeline": "Mission Timeline",
        "relative_range": "Relative Range vs Time",
        "relative_velocity": "Relative Velocity vs Time",
        "cumulative_delta_v": "Cumulative Delta V vs Time",
        "control_commands": "Control Commands vs Time",
    }.get(str(key), str(key).replace("_", " ").title())


def _debrief_display_title(*, config: RPOTrainingConfig, score: Any) -> str:
    level_name = str(getattr(config, "level_name", "") or "").strip()
    if level_name:
        return f"RPO Trainer {level_name} Debrief"
    scenario_id = str(getattr(config, "scenario_id", "") or getattr(score, "scenario_id", "") or "").strip()
    return _debrief_title_from_scenario_id(scenario_id or "Game Debrief")


def _debrief_title_from_scenario_id(scenario_id: str) -> str:
    text = str(scenario_id or "Game Debrief").strip().replace("-", " ").replace("_", " ")
    title = " ".join(part for part in text.split() if part).title() or "Game Debrief"
    return title if title.lower().endswith("debrief") else f"{title} Debrief"


def _md_text(value: Any) -> str:
    return str(value).replace("\r\n", " ").replace("\n", " ").strip()


def _md_table_text(value: Any) -> str:
    return _md_text(value).replace("|", "\\|")


def _md_alt_text(value: Any) -> str:
    return _md_text(value).replace("[", "").replace("]", "")


def _md_bullets(values: list[Any], *, empty: str) -> str:
    lines = [f"- {_md_text(value)}" for value in values if str(value).strip()]
    return "\n".join(lines) if lines else f"- {_md_text(empty)}"


def _seconds_text(value: Any) -> str:
    value_f = _float_or_none(value)
    return "--" if value_f is None else f"{value_f:.2f} s"


def _km_text(value: Any) -> str:
    value_f = _float_or_none(value)
    return "--" if value_f is None else f"{value_f:.4f} km"


def _speed_text(value_km_s: Any) -> str:
    value_f = _float_or_none(value_km_s)
    return "--" if value_f is None else f"{value_f * 1000.0:.4f} m/s"


def _meters_per_second_text(value_m_s: Any) -> str:
    value_f = _float_or_none(value_m_s)
    return "--" if value_f is None else f"{value_f:.4f} m/s"


def _accel_text(value_m_s2: Any) -> str:
    value_f = _float_or_none(value_m_s2)
    return "--" if value_f is None else f"{value_f:.6f} m/s^2"


def _int_text(value: Any) -> str:
    if value is None:
        return "--"
    try:
        return f"{int(value):,}"
    except Exception:
        return "--"


def _array_list(value: Any) -> list[Any]:
    arr = np.array(value, dtype=float)
    if arr.size == 0:
        return []
    if arr.ndim == 0:
        return [float(arr)]
    return arr.tolist()


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    out = float(value)
    return out if np.isfinite(out) else None


def _slug(value: str) -> str:
    text = str(value or "").strip().lower()
    out = []
    last_sep = False
    for ch in text:
        if ch.isalnum():
            out.append(ch)
            last_sep = False
        elif not last_sep:
            out.append("_")
            last_sep = True
    return "".join(out).strip("_") or "game"
