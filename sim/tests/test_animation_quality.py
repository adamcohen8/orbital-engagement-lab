from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib import animation

from integrations.oel_mcp import PublicOELMCPHandlers
from integrations.oel_mcp.execution import ExecutionApprovalPolicy
from sim.plotting.animation_quality import (
    animation_time_decimal_places,
    fixed_time_text_width,
    format_animation_time,
    save_animation_with_quality,
    select_contact_sheet_frames,
)
from sim.plotting.ground_track_plots import animate_ground_track
from sim.plotting.style import artifact_metadata, oel_plot_context
from sim.plotting.trajectory_animations import animate_multi_ric_2d_projections
from sim.review import (
    ReviewAnimationSpec,
    plan_review_animation,
    render_review_animation,
)


def _write_review_store(root: Path, *, samples: int = 7) -> None:
    review_dir = root / "review"
    review_dir.mkdir(parents=True)
    with sqlite3.connect(review_dir / "run.sqlite") as conn:
        conn.execute("CREATE TABLE run_metadata (scenario_name TEXT)")
        conn.execute("INSERT INTO run_metadata VALUES ('animation_quality_fixture')")
        conn.execute(
            "CREATE TABLE relative_state ("
            "time_s REAL, deputy_id TEXT, chief_id TEXT, "
            "r_radial_km REAL, i_intrack_km REAL, c_crosstrack_km REAL)"
        )
        rows = []
        for index in range(samples):
            time_s = float(index)
            rows.append(
                (
                    time_s,
                    "chaser",
                    "target",
                    0.2 * np.cos(0.5 * time_s),
                    -1.0 + 0.3 * time_s,
                    0.1 * np.sin(0.5 * time_s),
                )
            )
        conn.executemany("INSERT INTO relative_state VALUES (?, ?, ?, ?, ?, ?)", rows)


def test_animation_time_format_is_stable_and_suppresses_negative_zero() -> None:
    times = [0.0, 0.25, 1.0]
    decimals = animation_time_decimal_places(times)
    width = fixed_time_text_width(times, decimal_places=decimals)

    assert decimals == 2
    assert format_animation_time(-0.0001, decimal_places=decimals, width=width).strip() == "0.00"
    assert len({len(format_animation_time(value, decimal_places=decimals, width=width)) for value in times}) == 1


def test_contact_sheet_selection_is_bounded_and_keeps_semantic_frames() -> None:
    selected = select_contact_sheet_frames(101, key_frame_indices=(17, 83), maximum=9)

    assert selected[0] == 0
    assert selected[-1] == 100
    assert 17 in selected
    assert 83 in selected
    assert len(selected) == 9


def test_quality_save_writes_decodable_gif_receipt_and_contact_sheet(tmp_path: Path) -> None:
    target = tmp_path / "quality.gif"
    times = np.arange(4, dtype=float)
    metadata = artifact_metadata(scenario_name="quality_save", artifact_id="quality")
    with oel_plot_context(style_name="oel_light", metadata=metadata):
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.set_xlim(0.0, 3.0)
        ax.set_ylim(0.0, 3.0)
        ax.set_title("Animation quality fixture")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Value")
        (line,) = ax.plot([], [], label="evidence")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
        fig.tight_layout(rect=(0.02, 0.07, 0.78, 0.96))

        def update(index: int):
            line.set_data(times[: index + 1], times[: index + 1])
            return [line]

        movie = animation.FuncAnimation(fig, update, frames=4, interval=200, blit=False)
        report = save_animation_with_quality(
            movie,
            fig,
            target,
            update=update,
            frame_values=range(4),
            frame_times_s=times,
            fps=5,
            camera_policy="fixed",
            metadata=metadata,
            artifact_id="quality",
            style_name="oel_light",
            format_limits={(0, "x"): (0.0, 3.0), (0, "y"): (0.0, 3.0)},
        )
        plt.close(fig)

    assert report.automated_status == "passed"
    assert report.encoding.decode_ok is True
    assert report.encoding.frame_count == 4
    assert target.is_file()
    assert target.with_suffix(".contact-sheet.png").is_file()
    receipt = json.loads(target.with_suffix(".quality.json").read_text(encoding="utf-8"))
    assert receipt["visual_qa_status"] == "pending_agent_review"
    assert receipt["failed_checks"] == []


def test_quality_save_writes_decodable_portable_mp4(tmp_path: Path) -> None:
    target = tmp_path / "portable.mp4"
    times = np.arange(3, dtype=float)
    metadata = artifact_metadata(scenario_name="mp4_quality_save", artifact_id="portable")
    with oel_plot_context(style_name="oel_dark", metadata=metadata):
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.set_xlim(0.0, 2.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_title("Portable MP4 fixture")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Value")
        (line,) = ax.plot([], [])
        fig.tight_layout(rect=(0.02, 0.07, 0.98, 0.96))

        def update(index: int):
            line.set_data(times[: index + 1], np.sin(times[: index + 1]))
            return [line]

        movie = animation.FuncAnimation(fig, update, frames=3, interval=200, blit=False)
        report = save_animation_with_quality(
            movie,
            fig,
            target,
            update=update,
            frame_values=range(3),
            frame_times_s=times,
            fps=5,
            camera_policy="fixed",
            metadata=metadata,
            artifact_id="portable",
            style_name="oel_dark",
            format_limits={(0, "x"): (0.0, 2.0), (0, "y"): (-1.0, 1.0)},
        )
        plt.close(fig)

    assert report.automated_status == "passed"
    assert report.encoding.format == "mp4"
    assert report.encoding.decode_ok is True
    assert report.encoding.frame_count == 3
    assert report.encoding.width_px == 400
    assert report.encoding.height_px == 300


def test_review_animation_plan_render_and_stale_plan_boundary(tmp_path: Path) -> None:
    _write_review_store(tmp_path)
    spec = ReviewAnimationSpec(
        recipe_id="relative_position_ric_2d",
        artifact_id="agent_ric",
        style_name="oel_light",
        file_format="gif",
        fps=5.0,
        camera_policy="fit_history",
    )
    plan = plan_review_animation(tmp_path, spec)

    assert plan["status"] == "planned"
    assert plan["render_ready"] is True
    assert plan["render_authorized"] is False
    assert plan["render_frame_count"] == 7
    artifact = render_review_animation(
        tmp_path,
        spec,
        animation_plan_id=plan["animation_plan_id"],
        path=tmp_path / "review" / "animations" / "agent_ric.gif",
    )
    assert artifact.path.is_file()
    assert artifact.contact_sheet_path.is_file()
    assert artifact.quality_receipt_path.is_file()
    assert artifact.qa["automated_status"] == "passed"
    assert artifact.qa["encoding"]["frame_count"] == 7
    generated = json.loads((tmp_path / "review" / "generated_artifacts.json").read_text(encoding="utf-8"))
    assert generated["artifacts"][-1]["artifact_type"] == "animation"

    with sqlite3.connect(tmp_path / "review" / "run.sqlite") as conn:
        conn.execute(
            "INSERT INTO relative_state VALUES (?, ?, ?, ?, ?, ?)",
            (8.0, "chaser", "target", 0.0, 1.4, 0.0),
        )
    with pytest.raises(ValueError, match="stale or does not match"):
        render_review_animation(
            tmp_path,
            spec,
            animation_plan_id=plan["animation_plan_id"],
            path=tmp_path / "review" / "animations" / "stale.gif",
        )


def test_review_animation_planner_increases_stride_to_resource_limit(tmp_path: Path) -> None:
    _write_review_store(tmp_path, samples=650)
    spec = ReviewAnimationSpec(
        recipe_id="relative_position_ric_2d",
        artifact_id="bounded",
        file_format="gif",
        fps=10.0,
    )

    plan = plan_review_animation(tmp_path, spec)

    assert plan["source_frame_count"] == 650
    assert plan["render_frame_count"] <= 300
    assert plan["effective_frame_stride"] > 1
    assert plan["encoded_duration_s"] <= 30.0


def test_review_animation_refuses_truncated_evidence(tmp_path: Path) -> None:
    _write_review_store(tmp_path, samples=8)
    spec = ReviewAnimationSpec(
        recipe_id="relative_position_ric_2d",
        artifact_id="truncated",
        file_format="gif",
        fps=5.0,
        max_rows=3,
    )
    plan = plan_review_animation(tmp_path, spec)

    assert plan["truncated"] is True
    assert plan["render_ready"] is False
    with pytest.raises(ValueError, match="query is truncated"):
        render_review_animation(
            tmp_path,
            spec,
            animation_plan_id=plan["animation_plan_id"],
            path=tmp_path / "review" / "animations" / "truncated.gif",
        )


def test_pilot_ground_track_animation_emits_quality_bundle(tmp_path: Path) -> None:
    target = tmp_path / "ground.gif"

    animate_ground_track(
        np.array([-20.0, -5.0, 10.0, 25.0]),
        np.array([0.0, 5.0, 10.0, 15.0]),
        t_s=np.array([0.0, 1.0, 2.0, 3.0]),
        mode="save",
        out_path=str(target),
        fps=5.0,
        draw_earth_map=False,
    )

    receipt = json.loads(target.with_suffix(".quality.json").read_text(encoding="utf-8"))
    assert receipt["automated_status"] == "passed"
    assert receipt["camera_policy"] == "fixed"
    assert target.with_suffix(".contact-sheet.png").is_file()


def test_pilot_ric_animation_uses_stable_follow_camera(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "ric.gif"
    relative = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.3, 0.0],
            [0.2, 0.6, 0.1],
            [0.3, 0.9, 0.1],
        ],
        dtype=float,
    )
    monkeypatch.setattr(
        "sim.plotting.trajectory_animations._trajectory_in_frame",
        lambda **_kwargs: relative.copy(),
    )

    animate_multi_ric_2d_projections(
        np.array([0.0, 1.0, 2.0, 3.0]),
        {"chaser": np.zeros((4, 14), dtype=float)},
        frame="ric_rect",
        reference_truth_hist=np.zeros((4, 14), dtype=float),
        planes=["ri"],
        mode="save",
        out_path=str(target),
        fps=5.0,
    )

    receipt = json.loads(target.with_suffix(".quality.json").read_text(encoding="utf-8"))
    assert receipt["automated_status"] == "passed"
    assert receipt["camera_policy"] == "follow"
    camera_check = next(item for item in receipt["checks"] if item["check_id"] == "camera_policy_stable")
    assert camera_check["passed"] is True


def test_mcp_animation_plan_and_render_are_content_bound(tmp_path: Path) -> None:
    _write_review_store(tmp_path)
    handlers = PublicOELMCPHandlers(
        read_roots=(tmp_path,),
        write_roots=(tmp_path,),
        approval_policy=ExecutionApprovalPolicy(write_approval_ids=frozenset({"animation-write"})),
    )
    arguments = {
        "output_dir": str(tmp_path),
        "recipe_id": "relative_position_ric_2d",
        "artifact_id": "mcp_ric",
        "style": "oel_light",
        "format": "gif",
        "fps": 5,
        "frame_stride": 1,
        "camera_policy": "fit_history",
        "handling": {"marking": "PUBLIC", "release_scope": "public"},
    }

    planned = handlers.plan_review_animation(**arguments)
    assert planned["status"] == "completed"
    rendered = handlers.render_review_animation(
        **arguments,
        animation_plan_id=planned["result"]["animation_plan_id"],
        approval={"approval_id": "animation-write", "scope": "write"},
    )

    assert rendered["status"] == "completed"
    assert rendered["effects"] == {
        "reads": True,
        "writes": True,
        "executes": False,
        "external_communication": False,
    }
    assert rendered["result"]["artifact"]["qa"]["automated_status"] == "passed"
    assert Path(rendered["result"]["manifest_path"]).is_file()
