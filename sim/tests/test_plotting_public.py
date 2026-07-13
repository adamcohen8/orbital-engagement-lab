from __future__ import annotations

from pathlib import Path

import pytest

try:
    import matplotlib

    matplotlib.use("Agg")
except Exception as exc:  # pragma: no cover - depends on local optional plotting stack
    pytest.skip(f"matplotlib is not usable in this environment: {exc}", allow_module_level=True)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

import sim.plotting.single_run as single_run_plotting
import sim.plotting.style as oel_style
from sim import SimulationConfig
from sim.master_outputs import PLOT_PRESETS, plot_outputs
from sim.plotting import (
    plot_attitude_control_summary,
    plot_control_effort,
    plot_estimation_error,
    plot_estimation_error_components,
    plot_ground_station_access,
    plot_ground_track_from_payload,
    plot_knowledge_filtering,
    plot_orbital_element,
    plot_orbital_elements_angles,
    plot_orbital_elements_summary,
    plot_rendezvous_summary,
    plot_rendezvous_summary_curvilinear,
    plot_run_dashboard,
    plot_sensor_access,
)
from sim.plotting.style import artifact_metadata, oel_plot_context, save_oel_animation, save_oel_figure
from sim.utils.plotting import _draw_earth_sphere_3d
from sim.utils.plotting_capabilities import plot_multi_ric_2d_projections


def scenario_config_from_dict(data: dict):
    return SimulationConfig.from_dict(data).to_scenario_config()


def _hist(pos: np.ndarray) -> np.ndarray:
    n = pos.shape[0]
    hist = np.zeros((n, 14), dtype=float)
    hist[:, :3] = pos
    hist[:, 3:6] = np.array([0.0, 7.4, 1.0], dtype=float)
    hist[:, 6] = 1.0
    hist[:, 10:13] = np.array([0.0, 0.0, 0.01], dtype=float)
    hist[:, 13] = 100.0
    return hist


def _payload() -> dict[str, object]:
    t = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    target = _hist(
        np.array(
            [
                [7000.0, 0.0, 0.0],
                [7000.0, 7.5, 0.0],
                [6999.99, 15.0, 0.0],
                [6999.98, 22.5, 0.0],
            ],
            dtype=float,
        )
    )
    chaser = target.copy()
    chaser[:, 0] += np.array([1.0, 0.8, 0.5, 0.2], dtype=float)
    thrust = np.zeros((t.size, 3), dtype=float)
    thrust[:, 0] = [0.0, 1.0e-6, 1.0e-6, 0.0]
    belief = chaser.copy()
    belief[:, 0] += 0.01
    belief[:, 3] += 1.0e-4
    knowledge = chaser[:, :6].copy()
    knowledge[0, :] = np.nan
    knowledge[2, 0] += 0.02
    measurements = chaser[:, :6].copy()
    measurements[:, 0] += np.array([0.03, -0.02, 0.01, -0.01], dtype=float)
    measurements[:, 3] += np.array([1.0e-4, -1.0e-4, 5.0e-5, -5.0e-5], dtype=float)
    ground_access = {
        "equator_prime": {
            "station": {"id": "equator_prime"},
            "targets": {
                "target": {
                    "access": [True, True, False, False],
                    "line_of_sight": [True, True, True, False],
                    "range_km": [700.0, 720.0, 900.0, None],
                    "elevation_deg": [80.0, 45.0, 5.0, None],
                    "reason": ["ok", "ok", "elevation", "line_of_sight"],
                }
            },
        }
    }
    return {
        "summary": {"scenario_name": "plot_test"},
        "time_s": t.tolist(),
        "truth_by_object": {"target": target.tolist(), "chaser": chaser.tolist()},
        "belief_by_object": {"chaser": belief.tolist()},
        "applied_thrust_by_object": {"target": np.zeros_like(thrust).tolist(), "chaser": thrust.tolist()},
        "desired_attitude_by_object": {
            "target": np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=float), (t.size, 1)).tolist(),
            "chaser": np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=float), (t.size, 1)).tolist(),
        },
        "knowledge_by_observer": {"chaser": {"target": knowledge.tolist()}},
        "knowledge_measurements_by_observer": {"chaser": {"target": measurements.tolist()}},
        "ground_station_access": ground_access,
        "target_reference_orbit_truth": [],
    }


def test_oel_plot_style_adds_public_safe_footer(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "styled_artifact.png"
    metadata = artifact_metadata(
        scenario_name="style_smoke",
        generated_utc="2026-05-26T00:00:00Z",
        version_text="0.9.0",
    )
    with oel_plot_context(style_name="oel_dark", metadata=metadata):
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot([0.0, 1.0], [0.0, 1.0], label="actual")
        ax.legend(loc="best")
        save_oel_figure(fig, path, dpi=80, artifact_id="style_smoke_plot")
        footer_texts = [item.get_text() for item in fig.texts]
        plt.close(fig)

    assert path.exists()
    assert path.parent.exists()
    assert any("Orbital Engagement Lab" in text for text in footer_texts)
    assert any("scenario: style_smoke" in text for text in footer_texts)


def test_oel_version_falls_back_when_distribution_metadata_is_malformed(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_malformed_metadata(_package_name: str) -> str:
        raise TypeError("missing distribution metadata")

    monkeypatch.setattr(oel_style, "version", _raise_malformed_metadata)

    assert oel_style.get_oel_version() != "unknown"


def test_oel_version_prefers_source_pyproject_over_installed_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    def _stale_distribution_metadata(_package_name: str) -> str:
        return "0.0.0"

    monkeypatch.setattr(oel_style, "version", _stale_distribution_metadata)

    assert oel_style.get_oel_version() == oel_style._version_from_source_pyproject()


def test_oel_animation_save_adds_public_safe_footer(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "styled_animation.gif"
    metadata = artifact_metadata(
        scenario_name="animation_style_smoke",
        generated_utc="2026-05-26T00:00:00Z",
        version_text="0.9.0",
    )
    captured: dict[str, object] = {}

    class _AnimationStub:
        def save(self, path_str: str, *, fps: float, **kwargs: object) -> None:
            captured["path"] = path_str
            captured["fps"] = fps
            captured["kwargs"] = kwargs

    with oel_plot_context(style_name="oel_light", metadata=metadata):
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot([0.0, 1.0], [1.0, 0.0])
        save_oel_animation(_AnimationStub(), fig, path, fps=4.0, artifact_id="style_smoke_movie")
        footer_texts = [item.get_text() for item in fig.texts]
        plt.close(fig)

    assert path.parent.exists()
    assert captured["path"] == str(path)
    assert captured["fps"] == 4.0
    assert any("Orbital Engagement Lab" in text for text in footer_texts)
    assert any("artifact: style_smoke_movie" in text for text in footer_texts)


def test_dark_style_earth_sphere_uses_visible_transparent_colors() -> None:
    calls: dict[str, dict[str, object]] = {}

    class _AxesStub:
        def plot_surface(self, *_args: object, **kwargs: object) -> None:
            calls["surface"] = dict(kwargs)

        def plot_wireframe(self, *_args: object, **kwargs: object) -> None:
            calls["wireframe"] = dict(kwargs)

    with oel_plot_context(style_name="oel_dark"):
        _draw_earth_sphere_3d(_AxesStub())  # type: ignore[arg-type]

    assert calls["surface"]["color"] == "#7DD3FC"
    assert calls["wireframe"]["color"] == "#E0F2FE"
    assert 0.15 < float(calls["surface"]["alpha"]) < 0.3
    assert 0.25 < float(calls["wireframe"]["alpha"]) < 0.45


def test_run_dashboard_trajectory_panel_keeps_earth_centered() -> None:
    payload = _payload()

    fig = plot_run_dashboard(payload, close=False)
    ax_traj = fig.axes[0]
    x_center = float(np.mean(ax_traj.get_xlim()))
    y_center = float(np.mean(ax_traj.get_ylim()))
    z_center = float(np.mean(ax_traj.get_zlim()))
    plt.close(fig)

    assert np.isclose(x_center, 0.0, atol=1e-6)
    assert np.isclose(y_center, 0.0, atol=1e-6)
    assert np.isclose(z_center, 0.0, atol=1e-6)


def test_payload_plotting_api_writes_expected_artifacts(tmp_path: Path) -> None:
    payload = _payload()
    outputs = {
        "dashboard": tmp_path / "dashboard.png",
        "rendezvous": tmp_path / "rendezvous.png",
        "rendezvous_curvilinear": tmp_path / "rendezvous_curvilinear.png",
        "control": tmp_path / "control.png",
        "estimation": tmp_path / "estimation.png",
        "estimation_components": tmp_path / "estimation_components.png",
        "knowledge_filtering": tmp_path / "knowledge_filtering.png",
        "sensor_access": tmp_path / "sensor_access.png",
        "ground_station_access": tmp_path / "ground_station_access.png",
        "attitude_control_summary": tmp_path / "attitude_control_summary.png",
        "orbital_a": tmp_path / "orbital_a.png",
        "orbital_ecc": tmp_path / "orbital_ecc.png",
        "orbital_inc": tmp_path / "orbital_inc.png",
        "orbital_raan": tmp_path / "orbital_raan.png",
        "orbital_argp": tmp_path / "orbital_argp.png",
        "orbital_true_anomaly": tmp_path / "orbital_true_anomaly.png",
        "orbital_summary": tmp_path / "orbital_summary.png",
        "orbital_angles": tmp_path / "orbital_angles.png",
        "ground": tmp_path / "ground.png",
        "ground_map": tmp_path / "ground_map.png",
    }

    plot_run_dashboard(payload, out_path=outputs["dashboard"], close=True)
    plot_rendezvous_summary(payload, out_path=outputs["rendezvous"], close=True)
    plot_rendezvous_summary_curvilinear(payload, out_path=outputs["rendezvous_curvilinear"], close=True)
    plot_control_effort(payload, out_path=outputs["control"], close=True)
    plot_estimation_error(payload, out_path=outputs["estimation"], close=True)
    plot_estimation_error_components(payload, out_path=outputs["estimation_components"], close=True)
    plot_knowledge_filtering(
        payload,
        knowledge_noise_by_observer={
            "chaser": {
                "pos_sigma_km": [0.02, 0.02, 0.02],
                "vel_sigma_km_s": [1.0e-4, 1.0e-4, 1.0e-4],
            }
        },
        out_path=outputs["knowledge_filtering"],
        close=True,
    )
    plot_sensor_access(payload, out_path=outputs["sensor_access"], close=True)
    plot_ground_station_access(payload, out_path=outputs["ground_station_access"], close=True)
    plot_attitude_control_summary(payload, out_path=outputs["attitude_control_summary"], close=True)
    plot_orbital_element(payload, element_id="a", out_path=outputs["orbital_a"], close=True)
    plot_orbital_element(payload, element_id="ecc", out_path=outputs["orbital_ecc"], close=True)
    plot_orbital_element(payload, element_id="inc", out_path=outputs["orbital_inc"], close=True)
    plot_orbital_element(payload, element_id="raan", out_path=outputs["orbital_raan"], close=True)
    plot_orbital_element(payload, element_id="argp", out_path=outputs["orbital_argp"], close=True)
    plot_orbital_element(
        payload,
        element_id="true_anomaly",
        out_path=outputs["orbital_true_anomaly"],
        close=True,
    )
    plot_orbital_elements_summary(payload, out_path=outputs["orbital_summary"], close=True)
    plot_orbital_elements_angles(payload, out_path=outputs["orbital_angles"], close=True)
    plot_ground_track_from_payload(payload, out_path=outputs["ground"], close=True)
    plot_ground_track_from_payload(payload, draw_earth_map=True, out_path=outputs["ground_map"], close=True)

    for path in outputs.values():
        assert path.exists()
        assert path.stat().st_size > 0


def test_rendezvous_summary_scales_ric_projection_axes_independently() -> None:
    payload = _payload()
    target = np.array(payload["truth_by_object"]["target"], dtype=float)
    chaser = target.copy()
    chaser[:, 0] += np.array([0.0, 0.05, 0.10, 0.15], dtype=float)
    chaser[:, 1] += np.array([0.0, 6.0, 12.0, 18.0], dtype=float)
    payload["truth_by_object"] = {"target": target.tolist(), "chaser": chaser.tolist()}

    fig = plot_rendezvous_summary(payload, close=False)
    ax_ri = fig.axes[0]
    x_span = float(np.diff(ax_ri.get_xlim())[0])
    y_span = float(np.diff(ax_ri.get_ylim())[0])
    plt.close(fig)

    assert x_span > 5.0 * y_span


def test_curvilinear_rendezvous_summary_combines_range_speed_and_delta_v() -> None:
    payload = _payload()

    fig = plot_rendezvous_summary_curvilinear(payload, close=False)
    titles = [ax.get_title() for ax in fig.axes]
    axes_count = len(fig.axes)
    plt.close(fig)

    assert "I-R Projection" in titles
    assert "Relative Range and Speed" in titles
    assert "Curvilinear RIC Components" in titles
    assert "Cumulative Delta-V" in titles
    assert axes_count == 7


def test_ric_2d_projection_can_mark_target_burns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _payload()
    t = np.array(payload["time_s"], dtype=float)
    target = np.array(payload["truth_by_object"]["target"], dtype=float)
    chaser = np.array(payload["truth_by_object"]["chaser"], dtype=float)
    target_thrust = np.zeros((t.size, 3), dtype=float)
    target_thrust[1:3, 0] = 1.0e-6
    scatter_calls: list[dict[str, object]] = []
    original_scatter = Axes.scatter

    def _scatter_spy(self: Axes, *args: object, **kwargs: object) -> object:
        scatter_calls.append(dict(kwargs))
        return original_scatter(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "scatter", _scatter_spy)
    out_path = tmp_path / "ric_burn_markers.png"

    plot_multi_ric_2d_projections(
        t,
        {"chaser": chaser},
        frame="ric_curv",
        reference_truth_hist=target,
        burn_marker_by_object={"target": target_thrust},
        burn_marker_object_ids=["target"],
        mode="save",
        out_path=str(out_path),
    )

    assert out_path.exists()
    assert any(call.get("label") == "target burn" and call.get("color") == "#F97316" for call in scatter_calls)


def test_plot_outputs_expands_public_plot_presets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert "run_dashboard" in PLOT_PRESETS["minimal"]
    cfg = scenario_config_from_dict(
        {
            "scenario_name": "plot_outputs_test",
            "target": {"enabled": True},
            "chaser": {"enabled": True},
            "ground_stations": [
                {
                    "id": "equator_prime",
                    "lat_deg": 0.0,
                    "lon_deg": 0.0,
                    "alt_km": 0.0,
                    "min_elevation_deg": 0.0,
                }
            ],
            "simulator": {"duration_s": 3.0, "dt_s": 1.0},
            "outputs": {
                "output_dir": str(tmp_path),
                "mode": "save",
                "plots": {
                    "enabled": True,
                    "preset": "minimal",
                    "figure_ids": [
                        "rendezvous_summary",
                        "rendezvous_summary_curvilinear",
                        "control_effort",
                        "trajectory_ric_rect_2d_multi_target_burns",
                        "trajectory_ric_curv_2d_multi_target_burns",
                        "estimation_error",
                        "estimation_error_components",
                        "knowledge_filtering",
                        "sensor_access",
                        "ground_station_access",
                        "attitude_control_summary",
                        "orbital_element_a",
                        "orbital_element_ecc",
                        "orbital_element_inc",
                        "orbital_element_raan",
                        "orbital_element_argp",
                        "orbital_element_true_anomaly",
                        "orbital_elements_summary",
                        "orbital_elements_angles",
                        "ground_track_multi",
                    ],
                    "reference_object_id": "target",
                    "draw_earth_map": True,
                    "orbital_elements_object_id": "target",
                },
            },
            "monte_carlo": {"enabled": False},
        }
    )
    payload = _payload()
    t = np.array(payload["time_s"], dtype=float)
    truth = {k: np.array(v, dtype=float) for k, v in dict(payload["truth_by_object"]).items()}
    thrust = {k: np.array(v, dtype=float) for k, v in dict(payload["applied_thrust_by_object"]).items()}
    belief = {k: np.array(v, dtype=float) for k, v in dict(payload["belief_by_object"]).items()}
    knowledge = {
        obs: {tgt: np.array(arr, dtype=float) for tgt, arr in by_tgt.items()}
        for obs, by_tgt in dict(payload["knowledge_by_observer"]).items()
    }
    measurements = {
        obs: {tgt: np.array(arr, dtype=float) for tgt, arr in by_tgt.items()}
        for obs, by_tgt in dict(payload["knowledge_measurements_by_observer"]).items()
    }
    coe_calls = 0
    original_coe_series = single_run_plotting._classical_orbital_elements_series

    def _counted_coe_series(*args: object, **kwargs: object) -> dict[str, np.ndarray]:
        nonlocal coe_calls
        coe_calls += 1
        return original_coe_series(*args, **kwargs)

    monkeypatch.setattr(single_run_plotting, "_classical_orbital_elements_series", _counted_coe_series)

    out = plot_outputs(
        cfg=cfg,
        t_s=t,
        truth_hist=truth,
        target_reference_orbit_truth=None,
        thrust_hist=thrust,
        desired_attitude_hist=None,
        knowledge_hist=knowledge,
        rocket_metrics=None,
        outdir=tmp_path,
        resolve_rocket_stack=lambda specs: None,
        resolve_satellite_isp_s=lambda specs: 0.0,
        belief_hist=belief,
        knowledge_measurement_hist=measurements,
    )

    assert coe_calls == 1
    assert set(out) >= {
        "run_dashboard",
        "rendezvous_summary",
        "rendezvous_summary_curvilinear",
        "control_effort",
        "trajectory_ric_rect_2d_multi_target_burns",
        "trajectory_ric_curv_2d_multi_target_burns",
        "estimation_error",
        "estimation_error_components",
        "knowledge_filtering",
        "sensor_access",
        "ground_station_access",
        "attitude_control_summary",
        "orbital_element_a",
        "orbital_element_ecc",
        "orbital_element_inc",
        "orbital_element_raan",
        "orbital_element_argp",
        "orbital_element_true_anomaly",
        "orbital_elements_summary",
        "orbital_elements_angles",
        "ground_track_multi",
    }
    for artifact in out.values():
        path = Path(artifact)
        assert path.exists()
        assert path.stat().st_size > 0


def test_orbital_element_cache_preserves_line_data_exactly() -> None:
    payload = _payload()
    t_s = np.asarray(payload["time_s"], dtype=float)
    truth = {
        object_id: np.asarray(history, dtype=float)
        for object_id, history in dict(payload["truth_by_object"]).items()
    }
    fig_uncached, axes_uncached = plt.subplots(3, 2)
    fig_cached, axes_cached = plt.subplots(3, 2)
    shared_cache: single_run_plotting.OrbitalElementSeriesCache = {}

    for uncached_ax, cached_ax, element_id in zip(
        axes_uncached.ravel(),
        axes_cached.ravel(),
        single_run_plotting.ORBITAL_ELEMENT_SPECS,
    ):
        single_run_plotting._plot_element_on_axis(
            uncached_ax,
            t_s=t_s,
            truth_by_object=truth,
            element_id=element_id,
            object_id=None,
        )
        single_run_plotting._plot_element_on_axis(
            cached_ax,
            t_s=t_s,
            truth_by_object=truth,
            element_id=element_id,
            object_id=None,
            series_cache=shared_cache,
        )
        assert [line.get_label() for line in cached_ax.lines] == [line.get_label() for line in uncached_ax.lines]
        for cached_line, uncached_line in zip(cached_ax.lines, uncached_ax.lines):
            assert np.array_equal(cached_line.get_xdata(), uncached_line.get_xdata(), equal_nan=True)
            assert np.array_equal(cached_line.get_ydata(), uncached_line.get_ydata(), equal_nan=True)

    assert set(shared_cache) == set(truth)
    assert all(entry[1].shape[1] == 6 for entry in shared_cache.values())
    plt.close(fig_uncached)
    plt.close(fig_cached)


def test_orbital_element_cache_invalidates_for_new_history_with_same_object_id() -> None:
    payload = _payload()
    t_s = np.asarray(payload["time_s"], dtype=float)
    first_history = np.asarray(dict(payload["truth_by_object"])["target"], dtype=float)
    second_history = first_history.copy()
    second_history[:, 0] += 1000.0
    shared_cache: single_run_plotting.OrbitalElementSeriesCache = {}

    first_fig = plot_orbital_element(
        t_s=t_s,
        truth_by_object={"sat": first_history},
        element_id="a",
        orbital_elements_cache=shared_cache,
    )
    second_fig = plot_orbital_element(
        t_s=t_s,
        truth_by_object={"sat": second_history},
        element_id="a",
        orbital_elements_cache=shared_cache,
    )

    expected_second = single_run_plotting._classical_orbital_elements_series(second_history)["a"]
    assert np.array_equal(second_fig.axes[0].lines[0].get_ydata(), expected_second, equal_nan=True)
    assert not np.array_equal(
        first_fig.axes[0].lines[0].get_ydata(),
        second_fig.axes[0].lines[0].get_ydata(),
        equal_nan=True,
    )
    plt.close(first_fig)
    plt.close(second_fig)


def test_orbital_element_cache_invalidates_for_in_place_history_change() -> None:
    payload = _payload()
    t_s = np.asarray(payload["time_s"], dtype=float)
    history = np.asarray(dict(payload["truth_by_object"])["target"], dtype=float)
    shared_cache: single_run_plotting.OrbitalElementSeriesCache = {}

    first_fig = plot_orbital_element(
        t_s=t_s,
        truth_by_object={"sat": history},
        element_id="a",
        orbital_elements_cache=shared_cache,
    )
    first_series = np.array(first_fig.axes[0].lines[0].get_ydata(), copy=True)
    history[:, 0] += 1000.0
    second_fig = plot_orbital_element(
        t_s=t_s,
        truth_by_object={"sat": history},
        element_id="a",
        orbital_elements_cache=shared_cache,
    )

    expected_second = single_run_plotting._classical_orbital_elements_series(history)["a"]
    assert np.array_equal(second_fig.axes[0].lines[0].get_ydata(), expected_second, equal_nan=True)
    assert not np.array_equal(first_series, second_fig.axes[0].lines[0].get_ydata(), equal_nan=True)
    plt.close(first_fig)
    plt.close(second_fig)


def test_orbital_element_cache_accepts_float_convertible_object_history() -> None:
    payload = _payload()
    t_s = np.asarray(payload["time_s"], dtype=float)
    history = np.asarray(dict(payload["truth_by_object"])["target"], dtype=object)

    fig = plot_orbital_elements_summary(
        t_s=t_s,
        truth_by_object={"sat": history},
    )

    expected = single_run_plotting._classical_orbital_elements_series(history)["a"]
    assert np.array_equal(fig.axes[0].lines[0].get_ydata(), expected, equal_nan=True)
    plt.close(fig)


def test_orbital_element_plots_tolerate_invalid_coe_samples(tmp_path: Path) -> None:
    invalid_hist = np.zeros((4, 14), dtype=float)
    payload = {
        "time_s": [0.0, 1.0, 2.0, 3.0],
        "truth_by_object": {"invalid": invalid_hist.tolist()},
        "target_reference_orbit_truth": [],
    }
    outputs = {
        "single": tmp_path / "invalid_orbital_element.png",
        "summary": tmp_path / "invalid_orbital_summary.png",
        "angles": tmp_path / "invalid_orbital_angles.png",
    }

    plot_orbital_element(payload, element_id="raan", out_path=outputs["single"], close=True)
    plot_orbital_elements_summary(payload, out_path=outputs["summary"], close=True)
    plot_orbital_elements_angles(payload, out_path=outputs["angles"], close=True)

    for path in outputs.values():
        assert path.exists()
        assert path.stat().st_size > 0


def test_rocket_story_plot_outputs_write_artifacts(tmp_path: Path) -> None:
    cfg = scenario_config_from_dict(
        {
            "scenario_name": "rocket_story_plot_test",
            "rocket": {
                "enabled": True,
                "kind": "rocket",
                "initial_state": {
                    "launch_lat_deg": 28.5,
                    "launch_lon_deg": -80.6,
                    "launch_alt_km": 0.0,
                },
                "base_guidance": {
                    "kind": "python",
                    "module": "sim.rocket.guidance",
                    "class_name": "OpenLoopPitchProgramGuidance",
                    "params": {"pitch_start_s": 1.0, "pitch_end_s": 3.0},
                },
                "guidance_modifiers": [
                    {
                        "kind": "python",
                        "module": "sim.rocket.guidance",
                        "class_name": "MaxQThrottleLimiterGuidance",
                        "params": {"max_q_pa": 55000.0},
                    }
                ],
            },
            "target": {"enabled": False},
            "chaser": {"enabled": False},
            "simulator": {
                "duration_s": 4.0,
                "dt_s": 1.0,
                "initial_jd_utc": 2460310.5,
                "dynamics": {
                    "rocket": {
                        "target_altitude_km": 350.0,
                        "target_altitude_tolerance_km": 40.0,
                        "target_eccentricity_max": 0.05,
                        "tvc_max_gimbal_deg": 5.0,
                    }
                },
                "termination": {"earth_impact_enabled": False},
            },
            "outputs": {
                "output_dir": str(tmp_path),
                "mode": "save",
                "plots": {
                    "enabled": True,
                    "figure_ids": [
                        "rocket_mission_timeline",
                        "rocket_downrange_altitude",
                        "rocket_maxq_throttle",
                        "rocket_tvc_aero_authority",
                        "rocket_insertion_scorecard",
                    ],
                },
            },
            "monte_carlo": {"enabled": False},
        }
    )
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    rocket_hist = _hist(
        np.array(
            [
                [6378.137, 0.0, 0.0],
                [6400.0, 5.0, 0.0],
                [6500.0, 50.0, 5.0],
                [6650.0, 150.0, 20.0],
                [6725.0, 300.0, 35.0],
            ],
            dtype=float,
        )
    )
    rocket_metrics = {
        "altitude_km": np.array([0.0, 22.0, 125.0, 280.0, 350.0]),
        "speed_km_s": np.array([0.0, 0.8, 2.5, 5.0, 7.6]),
        "apoapsis_alt_km": np.array([0.0, 60.0, 180.0, 330.0, 360.0]),
        "periapsis_alt_km": np.array([-20.0, -10.0, 50.0, 250.0, 340.0]),
        "eccentricity": np.array([0.9, 0.6, 0.2, 0.08, 0.02]),
        "q_dyn_pa": np.array([0.0, 20000.0, 56000.0, 15000.0, 100.0]),
        "mach": np.array([0.0, 0.8, 2.2, 8.0, 20.0]),
        "stage_index": np.array([0.0, 0.0, 1.0, 1.0, 2.0]),
        "throttle_cmd": np.array([1.0, 1.0, 0.45, 1.0, 0.0]),
        "tvc_gimbal_deg": np.array([0.0, 1.0, 3.0, 2.0, 0.0]),
        "alpha_deg": np.array([0.0, 1.5, 2.5, 1.0, 0.1]),
        "beta_deg": np.array([0.0, 0.5, 1.0, 0.4, 0.0]),
        "aero_force_n": np.array([0.0, 12000.0, 45000.0, 9000.0, 0.0]),
        "aero_moment_nm": np.array([0.0, 1000.0, 2500.0, 700.0, 0.0]),
        "thrust_to_weight": np.array([1.4, 1.3, 1.1, 0.9, 0.0]),
        "propellant_remaining_fraction": np.array([1.0, 0.8, 0.45, 0.18, 0.08]),
    }

    out = plot_outputs(
        cfg=cfg,
        t_s=t,
        truth_hist={"rocket": rocket_hist},
        target_reference_orbit_truth=None,
        thrust_hist={"rocket": np.zeros((t.size, 3), dtype=float)},
        desired_attitude_hist=None,
        knowledge_hist={},
        rocket_metrics=rocket_metrics,
        outdir=tmp_path,
        resolve_rocket_stack=lambda specs: None,
        resolve_satellite_isp_s=lambda specs: 0.0,
        belief_hist={},
    )

    assert set(out) >= {
        "rocket_mission_timeline",
        "rocket_downrange_altitude",
        "rocket_maxq_throttle",
        "rocket_tvc_aero_authority",
        "rocket_insertion_scorecard",
    }
    for artifact in out.values():
        path = Path(artifact)
        assert path.exists()
        assert path.stat().st_size > 0
