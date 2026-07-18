from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np

from sim.plotting.single_run_context import _nested_array_map, _payload_arrays, _save_show_close
from sim.utils.figure_size import cap_figsize

ArrayMap = dict[str, np.ndarray]
NestedArrayMap = dict[str, dict[str, np.ndarray]]
OrbitalElementSeriesCache = dict[
    str,
    tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]],
]
RICSummaryFrame = Literal["rectangular", "curvilinear"]

def plot_estimation_error(
    payload: dict[str, Any] | None = None,
    *,
    t_s: np.ndarray | None = None,
    truth_by_object: ArrayMap | None = None,
    belief_by_object: ArrayMap | None = None,
    out_path: str | Path | None = None,
    show: bool = False,
    close: bool = False,
    dpi: int = 150,
) -> plt.Figure:
    if payload is not None:
        t_s, truth_by_object, _, belief_by_object, _, _ = _payload_arrays(payload)
    t = np.array([] if t_s is None else t_s, dtype=float).reshape(-1)
    truth = dict(truth_by_object or {})
    belief = dict(belief_by_object or {})
    fig, axes = plt.subplots(2, 1, figsize=cap_figsize(11, 7), sharex=True)
    plotted = False
    for oid, b in sorted(belief.items()):
        x = truth.get(oid)
        if x is None or x.shape[1] < 6 or b.shape[1] < 6:
            continue
        n = min(x.shape[0], b.shape[0], t.size)
        pos_err = np.linalg.norm(b[:n, :3] - x[:n, :3], axis=1)
        vel_err = np.linalg.norm(b[:n, 3:6] - x[:n, 3:6], axis=1)
        axes[0].plot(t[:n], pos_err, label=oid)
        axes[1].plot(t[:n], vel_err, label=oid)
        plotted = True
    if not plotted:
        for ax in axes:
            ax.text(0.5, 0.5, "No belief/truth pair available", ha="center", va="center", transform=ax.transAxes)
    axes[0].set_title("Position Estimation Error")
    axes[0].set_ylabel("km")
    axes[0].grid(True, alpha=0.3)
    axes[1].set_title("Velocity Estimation Error")
    axes[1].set_ylabel("km/s")
    axes[1].set_xlabel("time (s)")
    axes[1].grid(True, alpha=0.3)
    if plotted:
        axes[0].legend(loc="best")
        axes[1].legend(loc="best")
    fig.tight_layout()
    _save_show_close(fig, out_path=out_path, show=show, close=close, dpi=dpi)
    return fig


def plot_estimation_error_components(
    payload: dict[str, Any] | None = None,
    *,
    t_s: np.ndarray | None = None,
    truth_by_object: ArrayMap | None = None,
    belief_by_object: ArrayMap | None = None,
    out_path: str | Path | None = None,
    show: bool = False,
    close: bool = False,
    dpi: int = 150,
) -> plt.Figure:
    if payload is not None:
        t_s, truth_by_object, _, belief_by_object, _, _ = _payload_arrays(payload)
    t = np.array([] if t_s is None else t_s, dtype=float).reshape(-1)
    truth = dict(truth_by_object or {})
    belief = dict(belief_by_object or {})
    fig, axes = plt.subplots(2, 1, figsize=cap_figsize(12, 8), sharex=True)
    plotted = False
    pos_labels = ("x", "y", "z")
    vel_labels = ("vx", "vy", "vz")
    for oid, b in sorted(belief.items()):
        x = truth.get(oid)
        if x is None or x.shape[1] < 6 or b.shape[1] < 6:
            continue
        n = min(x.shape[0], b.shape[0], t.size)
        err = b[:n, :6] - x[:n, :6]
        for i, label in enumerate(pos_labels):
            axes[0].plot(t[:n], err[:, i], linewidth=1.0, label=f"{oid} {label}")
        for i, label in enumerate(vel_labels):
            axes[1].plot(t[:n], err[:, i + 3], linewidth=1.0, label=f"{oid} {label}")
        plotted = True
    if not plotted:
        for ax in axes:
            ax.text(0.5, 0.5, "No belief/truth pair available", ha="center", va="center", transform=ax.transAxes)
    axes[0].set_title("Position Estimation Error Components")
    axes[0].set_ylabel("km")
    axes[0].grid(True, alpha=0.3)
    axes[1].set_title("Velocity Estimation Error Components")
    axes[1].set_ylabel("km/s")
    axes[1].set_xlabel("time (s)")
    axes[1].grid(True, alpha=0.3)
    if plotted:
        axes[0].legend(loc="best", ncol=2)
        axes[1].legend(loc="best", ncol=2)
    fig.tight_layout()
    _save_show_close(fig, out_path=out_path, show=show, close=close, dpi=dpi)
    return fig


def plot_knowledge_filtering(
    payload: dict[str, Any] | None = None,
    *,
    t_s: np.ndarray | None = None,
    truth_by_object: ArrayMap | None = None,
    knowledge_by_observer: NestedArrayMap | None = None,
    knowledge_measurements_by_observer: NestedArrayMap | None = None,
    knowledge_noise_by_observer: dict[str, dict[str, Any]] | None = None,
    out_path: str | Path | None = None,
    show: bool = False,
    close: bool = False,
    dpi: int = 150,
) -> plt.Figure:
    if payload is not None:
        t_s, truth_by_object, _, _, knowledge_by_observer, _ = _payload_arrays(payload)
        knowledge_measurements_by_observer = _nested_array_map(payload.get("knowledge_measurements_by_observer", {}))
    t = np.array([] if t_s is None else t_s, dtype=float).reshape(-1)
    truth = dict(truth_by_object or {})
    estimates = dict(knowledge_by_observer or {})
    measurements = dict(knowledge_measurements_by_observer or {})
    noise_by_observer = dict(knowledge_noise_by_observer or {})
    fig, axes = plt.subplots(2, 3, figsize=cap_figsize(15, 8), constrained_layout=True)
    ax_range, ax_pos, ax_vel, ax_pos_hist, ax_vel_hist, ax_norm_hist = axes.reshape(-1)
    plotted = False
    pos_hist_values: list[np.ndarray] = []
    vel_hist_values: list[np.ndarray] = []
    norm_hist_values: list[np.ndarray] = []
    pos_sigma_markers_m: list[tuple[float, float]] = []
    vel_sigma_markers_mm_s: list[tuple[float, float]] = []
    for obs, by_target in sorted(estimates.items()):
        for target, estimate in sorted(by_target.items()):
            target_truth = truth.get(target)
            observer_truth = truth.get(obs)
            measurement = measurements.get(obs, {}).get(target)
            if target_truth is None or target_truth.shape[1] < 6 or estimate.shape[1] < 6:
                continue
            n = min(target_truth.shape[0], estimate.shape[0], t.size)
            if n <= 0:
                continue
            label = f"{obs}->{target}"
            if observer_truth is not None and observer_truth.shape[1] >= 3:
                nr = min(n, observer_truth.shape[0])
                truth_range = np.linalg.norm(target_truth[:nr, :3] - observer_truth[:nr, :3], axis=1)
                estimate_range = np.linalg.norm(estimate[:nr, :3] - observer_truth[:nr, :3], axis=1)
                ax_range.plot(t[:nr], truth_range, color="black", linewidth=1.4, label=f"{label} truth")
                ax_range.plot(t[:nr], estimate_range, linewidth=1.1, label=f"{label} estimate")
                if measurement is not None and measurement.shape[1] >= 3:
                    nm = min(nr, measurement.shape[0])
                    meas_range = np.linalg.norm(measurement[:nm, :3] - observer_truth[:nm, :3], axis=1)
                    valid_meas = np.all(np.isfinite(measurement[:nm, :3]), axis=1)
                    ax_range.scatter(t[:nm][valid_meas], meas_range[valid_meas], s=8, alpha=0.35, label=f"{label} meas")
            pos_err_est = np.linalg.norm(estimate[:n, :3] - target_truth[:n, :3], axis=1)
            vel_err_est = np.linalg.norm(estimate[:n, 3:6] - target_truth[:n, 3:6], axis=1)
            ax_pos.plot(t[:n], pos_err_est, linewidth=1.2, label=f"{label} estimate")
            ax_vel.plot(t[:n], vel_err_est * 1000.0, linewidth=1.2, label=f"{label} estimate")
            if measurement is not None and measurement.shape[1] >= 6:
                nm = min(n, measurement.shape[0])
                meas_pos_err = measurement[:nm, :3] - target_truth[:nm, :3]
                meas_vel_err = measurement[:nm, 3:6] - target_truth[:nm, 3:6]
                valid_pos = np.all(np.isfinite(meas_pos_err), axis=1)
                valid_vel = np.all(np.isfinite(meas_vel_err), axis=1)
                ax_pos.scatter(
                    t[:nm][valid_pos],
                    np.linalg.norm(meas_pos_err[valid_pos], axis=1),
                    s=8,
                    alpha=0.35,
                    label=f"{label} measurement",
                )
                ax_vel.scatter(
                    t[:nm][valid_vel],
                    np.linalg.norm(meas_vel_err[valid_vel], axis=1) * 1000.0,
                    s=8,
                    alpha=0.35,
                    label=f"{label} measurement",
                )
                if np.any(valid_pos):
                    pos_hist_values.append(meas_pos_err[valid_pos].reshape(-1))
                if np.any(valid_vel):
                    vel_hist_values.append(meas_vel_err[valid_vel].reshape(-1))
                noise = noise_by_observer.get(obs, {})
                pos_sigma = np.array(noise.get("pos_sigma_km", []), dtype=float).reshape(-1)
                vel_sigma = np.array(noise.get("vel_sigma_km_s", []), dtype=float).reshape(-1)
                pos_bias = np.array(noise.get("pos_bias_km", np.zeros(3)), dtype=float).reshape(-1)
                vel_bias = np.array(noise.get("vel_bias_km_s", np.zeros(3)), dtype=float).reshape(-1)
                if pos_sigma.size in (1, 3) and np.any(pos_sigma > 0.0) and np.any(valid_pos):
                    ps = np.full(3, float(pos_sigma[0])) if pos_sigma.size == 1 else pos_sigma[:3]
                    pb = (
                        np.zeros(3, dtype=float)
                        if pos_bias.size == 0
                        else np.full(3, float(pos_bias[0]))
                        if pos_bias.size == 1
                        else pos_bias[:3]
                    )
                    usable = ps > 0.0
                    norm_hist_values.append(((meas_pos_err[valid_pos][:, usable] - pb[usable]) / ps[usable]).reshape(-1))
                    pos_sigma_markers_m.append((float(np.mean(pb)) * 1000.0, float(np.sqrt(np.mean(ps**2))) * 1000.0))
                if vel_sigma.size in (1, 3) and np.any(vel_sigma > 0.0) and np.any(valid_vel):
                    vs = np.full(3, float(vel_sigma[0])) if vel_sigma.size == 1 else vel_sigma[:3]
                    vb = (
                        np.zeros(3, dtype=float)
                        if vel_bias.size == 0
                        else np.full(3, float(vel_bias[0]))
                        if vel_bias.size == 1
                        else vel_bias[:3]
                    )
                    usable = vs > 0.0
                    norm_hist_values.append(((meas_vel_err[valid_vel][:, usable] - vb[usable]) / vs[usable]).reshape(-1))
                    vel_sigma_markers_mm_s.append(
                        (float(np.mean(vb)) * 1.0e6, float(np.sqrt(np.mean(vs**2))) * 1.0e6)
                    )
            plotted = True

    def _unique_sigma_markers(markers: list[tuple[float, float]]) -> list[tuple[float, float]]:
        unique: list[tuple[float, float]] = []
        for bias, sigma in markers:
            if not any(np.isclose(bias, b, rtol=1e-9, atol=1e-12) and np.isclose(sigma, s, rtol=1e-9, atol=1e-12) for b, s in unique):
                unique.append((bias, sigma))
        return unique

    if pos_hist_values:
        values_m = np.concatenate(pos_hist_values) * 1000.0
        finite = values_m[np.isfinite(values_m)]
        if finite.size:
            ax_pos_hist.hist(finite, bins=40, density=True, alpha=0.75, color="tab:blue", label="residuals")
            ax_pos_hist.axvline(float(np.mean(finite)), color="black", linestyle="--", linewidth=1.0, label="mean")
            for marker_idx, (bias_m, sigma_m) in enumerate(_unique_sigma_markers(pos_sigma_markers_m)):
                label = "+/- cfg sigma" if marker_idx == 0 else None
                ax_pos_hist.axvline(bias_m - sigma_m, color="tab:red", linestyle=":", linewidth=1.0, label=label)
                ax_pos_hist.axvline(bias_m + sigma_m, color="tab:red", linestyle=":", linewidth=1.0)
            ax_pos_hist.legend(loc="best")
    if vel_hist_values:
        values_mm_s = np.concatenate(vel_hist_values) * 1.0e6
        finite = values_mm_s[np.isfinite(values_mm_s)]
        if finite.size:
            ax_vel_hist.hist(finite, bins=40, density=True, alpha=0.75, color="tab:orange", label="residuals")
            ax_vel_hist.axvline(float(np.mean(finite)), color="black", linestyle="--", linewidth=1.0, label="mean")
            for marker_idx, (bias_mm_s, sigma_mm_s) in enumerate(_unique_sigma_markers(vel_sigma_markers_mm_s)):
                label = "+/- cfg sigma" if marker_idx == 0 else None
                ax_vel_hist.axvline(
                    bias_mm_s - sigma_mm_s, color="tab:red", linestyle=":", linewidth=1.0, label=label
                )
                ax_vel_hist.axvline(bias_mm_s + sigma_mm_s, color="tab:red", linestyle=":", linewidth=1.0)
            ax_vel_hist.legend(loc="best")
    if norm_hist_values:
        finite = np.concatenate(norm_hist_values)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            ax_norm_hist.hist(finite, bins=40, density=True, alpha=0.7, color="tab:green", label="normalized residuals")
            x = np.linspace(-4.0, 4.0, 241)
            pdf = np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)
            ax_norm_hist.plot(x, pdf, color="black", linewidth=1.1, label="N(0,1)")
            ax_norm_hist.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
            ax_norm_hist.legend(loc="best")
    if not plotted:
        for ax in axes.reshape(-1):
            ax.text(
                0.5,
                0.5,
                "No truth/measurement/estimate chain available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
    ax_range.set_title("Truth / Measurement / Estimate Range")
    ax_range.set_ylabel("km")
    ax_pos.set_title("Position Error Norm")
    ax_pos.set_ylabel("km")
    ax_vel.set_title("Velocity Error Norm")
    ax_vel.set_ylabel("m/s")
    ax_vel.set_xlabel("time (s)")
    ax_pos_hist.set_title("Position Measurement Residuals")
    ax_pos_hist.set_xlabel("measurement - truth (m)")
    ax_pos_hist.set_ylabel("density")
    ax_vel_hist.set_title("Velocity Measurement Residuals")
    ax_vel_hist.set_xlabel("measurement - truth (mm/s)")
    ax_vel_hist.set_ylabel("density")
    ax_norm_hist.set_title("Normalized Measurement Residuals")
    ax_norm_hist.set_xlabel("(measurement - truth - bias) / sigma")
    ax_norm_hist.set_ylabel("density")
    for ax in axes.reshape(-1):
        ax.grid(True, alpha=0.3)
    for ax in (ax_range, ax_pos, ax_vel):
        if ax.lines or ax.collections:
            ax.legend(loc="best")
    _save_show_close(fig, out_path=out_path, show=show, close=close, dpi=dpi)
    return fig


def plot_sensor_access(
    payload: dict[str, Any] | None = None,
    *,
    t_s: np.ndarray | None = None,
    truth_by_object: ArrayMap | None = None,
    knowledge_by_observer: NestedArrayMap | None = None,
    out_path: str | Path | None = None,
    show: bool = False,
    close: bool = False,
    dpi: int = 150,
) -> plt.Figure:
    if payload is not None:
        t_s, truth_by_object, _, _, knowledge_by_observer, _ = _payload_arrays(payload)
    t = np.array([] if t_s is None else t_s, dtype=float).reshape(-1)
    truth = dict(truth_by_object or {})
    knowledge = dict(knowledge_by_observer or {})
    pairs: list[tuple[str, str, np.ndarray]] = []
    for obs, by_target in sorted(knowledge.items()):
        for target, hist in sorted(by_target.items()):
            if hist.ndim == 2 and hist.shape[0] > 0:
                pairs.append((obs, target, hist))

    fig, axes = plt.subplots(3, 1, figsize=cap_figsize(12, 9), sharex=True)
    if not pairs:
        for ax in axes:
            ax.text(0.5, 0.5, "No knowledge history available", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        _save_show_close(fig, out_path=out_path, show=show, close=close, dpi=dpi)
        return fig

    y_ticks = []
    y_labels = []
    for row, (obs, target, hist) in enumerate(pairs):
        n = min(hist.shape[0], t.size)
        known = np.any(np.isfinite(hist[:n, :]), axis=1).astype(float)
        axes[0].step(t[:n], known + row * 1.25, where="post", linewidth=1.4)
        y_ticks.append(row * 1.25 + 0.5)
        y_labels.append(f"{obs}->{target}")

        obs_truth = truth.get(obs)
        target_truth = truth.get(target)
        if (
            obs_truth is not None
            and target_truth is not None
            and obs_truth.shape[1] >= 3
            and target_truth.shape[1] >= 3
        ):
            nr = min(obs_truth.shape[0], target_truth.shape[0], t.size)
            rel = target_truth[:nr, :3] - obs_truth[:nr, :3]
            axes[1].plot(t[:nr], np.linalg.norm(rel, axis=1), label=f"{obs}->{target}")

        if hist.shape[1] >= 6 and target_truth is not None and target_truth.shape[1] >= 6:
            ne = min(hist.shape[0], target_truth.shape[0], t.size)
            err = hist[:ne, :6] - target_truth[:ne, :6]
            finite = np.all(np.isfinite(err[:, :3]), axis=1)
            pos_err = np.full(ne, np.nan, dtype=float)
            pos_err[finite] = np.linalg.norm(err[finite, :3], axis=1)
            axes[2].plot(t[:ne], pos_err, label=f"{obs}->{target}")

    axes[0].set_title("Sensor / Knowledge Access Timeline")
    axes[0].set_ylabel("access")
    axes[0].set_yticks(y_ticks)
    axes[0].set_yticklabels(y_labels)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-0.2, max(y_ticks) + 0.95)

    axes[1].set_title("Observer-Target Range")
    axes[1].set_ylabel("km")
    axes[1].grid(True, alpha=0.3)
    if axes[1].lines:
        axes[1].legend(loc="best")

    axes[2].set_title("Knowledge Position Error vs Target Truth")
    axes[2].set_ylabel("km")
    axes[2].set_xlabel("time (s)")
    axes[2].grid(True, alpha=0.3)
    if axes[2].lines:
        axes[2].legend(loc="best")

    fig.tight_layout()
    _save_show_close(fig, out_path=out_path, show=show, close=close, dpi=dpi)
    return fig
