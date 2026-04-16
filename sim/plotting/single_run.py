from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from sim.dynamics.orbit.environment import EARTH_RADIUS_KM
from sim.utils.figure_size import cap_figsize
from sim.utils.frames import ric_dcm_ir_from_rv
from sim.utils.ground_track import ground_track_from_eci_history, split_ground_track_dateline
from sim.utils.plotting import _draw_earth_sphere_3d


ArrayMap = dict[str, np.ndarray]
NestedArrayMap = dict[str, dict[str, np.ndarray]]


def _as_array(value: Any, *, cols: int | None = None) -> np.ndarray:
    arr = np.array(value if value is not None else [], dtype=float)
    if arr.ndim == 1 and arr.size == 0:
        return np.zeros((0, 0 if cols is None else cols), dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _array_map(value: Any) -> ArrayMap:
    if not isinstance(value, dict):
        return {}
    out: ArrayMap = {}
    for key, arr in value.items():
        parsed = _as_array(arr)
        if parsed.ndim == 2 and parsed.shape[0] > 0:
            out[str(key)] = parsed
    return out


def _nested_array_map(value: Any) -> NestedArrayMap:
    if not isinstance(value, dict):
        return {}
    out: NestedArrayMap = {}
    for outer, inner in value.items():
        if not isinstance(inner, dict):
            continue
        parsed_inner = _array_map(inner)
        if parsed_inner:
            out[str(outer)] = parsed_inner
    return out


def _payload_arrays(payload: dict[str, Any]) -> tuple[np.ndarray, ArrayMap, ArrayMap, ArrayMap, NestedArrayMap, np.ndarray | None]:
    t_s = np.array(payload.get("time_s", []), dtype=float).reshape(-1)
    truth = _array_map(payload.get("truth_by_object", {}))
    thrust = _array_map(payload.get("applied_thrust_by_object", {}))
    belief = _array_map(payload.get("belief_by_object", {}))
    knowledge = _nested_array_map(payload.get("knowledge_by_observer", {}))
    ref = _as_array(payload.get("target_reference_orbit_truth", []), cols=14)
    ref_out = ref if ref.ndim == 2 and ref.shape[0] > 0 and ref.shape[1] >= 6 else None
    return t_s, truth, thrust, belief, knowledge, ref_out


def _save_show_close(fig: plt.Figure, *, out_path: str | Path | None, show: bool, close: bool, dpi: int) -> None:
    if out_path is not None:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=int(dpi))
    if show:
        plt.show(block=False)
    if close:
        plt.close(fig)


def _choose_reference(
    truth_by_object: ArrayMap,
    target_reference_orbit_truth: np.ndarray | None,
    reference_object_id: str | None,
) -> tuple[str, np.ndarray] | tuple[None, None]:
    if target_reference_orbit_truth is not None and target_reference_orbit_truth.shape[1] >= 6:
        return "reference", target_reference_orbit_truth
    if reference_object_id and reference_object_id in truth_by_object:
        return reference_object_id, truth_by_object[reference_object_id]
    if "target" in truth_by_object:
        return "target", truth_by_object["target"]
    if truth_by_object:
        key = sorted(truth_by_object.keys())[0]
        return key, truth_by_object[key]
    return None, None


def _choose_subject(truth_by_object: ArrayMap, reference_id: str | None, object_id: str | None = None) -> tuple[str, np.ndarray] | tuple[None, None]:
    if object_id and object_id in truth_by_object:
        return object_id, truth_by_object[object_id]
    preferred = [k for k in ("chaser", "target", "rocket") if k in truth_by_object and k != reference_id]
    if preferred:
        key = preferred[0]
        return key, truth_by_object[key]
    for key in sorted(truth_by_object.keys()):
        if key != reference_id:
            return key, truth_by_object[key]
    if truth_by_object:
        key = sorted(truth_by_object.keys())[0]
        return key, truth_by_object[key]
    return None, None


def _ric_position(subject: np.ndarray, reference: np.ndarray) -> np.ndarray:
    n = min(subject.shape[0], reference.shape[0])
    out = np.full((n, 3), np.nan, dtype=float)
    for k in range(n):
        rv_ref = reference[k, :6]
        rv_sub = subject[k, :6]
        if not (np.all(np.isfinite(rv_ref)) and np.all(np.isfinite(rv_sub))):
            continue
        c_ir = ric_dcm_ir_from_rv(rv_ref[:3], rv_ref[3:6])
        out[k, :] = c_ir.T @ (rv_sub[:3] - rv_ref[:3])
    return out


def _finite_rows(arr: np.ndarray, cols: int = 3) -> np.ndarray:
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < cols:
        return np.zeros(0, dtype=bool)
    return np.all(np.isfinite(arr[:, :cols]), axis=1)


def _set_equal_3d(ax: Any, points: list[np.ndarray]) -> None:
    finite_parts = []
    for arr in points:
        a = np.array(arr, dtype=float)
        if a.ndim == 2 and a.shape[1] >= 3:
            finite = a[np.all(np.isfinite(a[:, :3]), axis=1), :3]
            if finite.size:
                finite_parts.append(finite)
    if not finite_parts:
        lim = EARTH_RADIUS_KM * 1.25
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        return
    pts = np.vstack(finite_parts)
    center = np.mean(pts, axis=0)
    span = max(float(np.max(np.ptp(pts, axis=0))), 1.0)
    half = 0.6 * span
    if np.linalg.norm(center) < EARTH_RADIUS_KM * 1.1:
        half = max(half, EARTH_RADIUS_KM * 1.15)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1, 1, 1))


def _plot_eci_trajectories(ax: Any, truth_by_object: ArrayMap) -> None:
    _draw_earth_sphere_3d(ax)
    plotted: list[np.ndarray] = []
    for oid, hist in sorted(truth_by_object.items()):
        if hist.shape[1] < 3:
            continue
        r = hist[:, :3]
        mask = _finite_rows(r)
        if not np.any(mask):
            continue
        ax.plot(r[mask, 0], r[mask, 1], r[mask, 2], linewidth=1.2, label=oid)
        idx = np.where(mask)[0]
        ax.scatter([r[idx[0], 0]], [r[idx[0], 1]], [r[idx[0], 2]], color="green", s=18)
        ax.scatter([r[idx[-1], 0]], [r[idx[-1], 1]], [r[idx[-1], 2]], color="red", s=18)
        plotted.append(r)
    earth_extent = np.array(
        [
            [-EARTH_RADIUS_KM, -EARTH_RADIUS_KM, -EARTH_RADIUS_KM],
            [EARTH_RADIUS_KM, EARTH_RADIUS_KM, EARTH_RADIUS_KM],
        ],
        dtype=float,
    )
    _set_equal_3d(ax, plotted + [earth_extent])
    ax.set_xlabel("ECI x (km)")
    ax.set_ylabel("ECI y (km)")
    ax.set_zlabel("ECI z (km)")
    ax.set_title("Trajectory")
    if plotted:
        ax.legend(loc="best")


def _time_for(arr: np.ndarray, t_s: np.ndarray) -> np.ndarray:
    n = min(arr.shape[0], t_s.size)
    return t_s[:n]


def _cumulative_delta_v_m_s(t_s: np.ndarray, accel_km_s2: np.ndarray) -> np.ndarray:
    mag = np.linalg.norm(np.nan_to_num(accel_km_s2, nan=0.0), axis=1)
    if mag.size == 0:
        return mag
    dt = np.diff(t_s[: mag.size], prepend=t_s[0] if t_s.size else 0.0)
    dt = np.clip(dt, 0.0, None)
    return np.cumsum(mag * dt) * 1000.0


def plot_run_dashboard(
    payload: dict[str, Any] | None = None,
    *,
    t_s: np.ndarray | None = None,
    truth_by_object: ArrayMap | None = None,
    thrust_by_object: ArrayMap | None = None,
    belief_by_object: ArrayMap | None = None,
    target_reference_orbit_truth: np.ndarray | None = None,
    reference_object_id: str | None = None,
    object_id: str | None = None,
    out_path: str | Path | None = None,
    show: bool = False,
    close: bool = False,
    dpi: int = 150,
) -> plt.Figure:
    if payload is not None:
        t_s, truth_by_object, thrust_by_object, belief_by_object, _, target_reference_orbit_truth = _payload_arrays(payload)
    t = np.array([] if t_s is None else t_s, dtype=float).reshape(-1)
    truth = dict(truth_by_object or {})
    thrust = dict(thrust_by_object or {})
    ref_id, ref = _choose_reference(truth, target_reference_orbit_truth, reference_object_id)
    subj_id, subj = _choose_subject(truth, ref_id, object_id)

    fig = plt.figure(figsize=cap_figsize(14, 9))
    ax_traj = fig.add_subplot(2, 3, 1, projection="3d")
    _plot_eci_trajectories(ax_traj, truth)

    ax_range = fig.add_subplot(2, 3, 2)
    ax_ric = fig.add_subplot(2, 3, 3)
    ax_thrust = fig.add_subplot(2, 3, 4)
    ax_dv = fig.add_subplot(2, 3, 5)
    ax_rate = fig.add_subplot(2, 3, 6)

    if subj is not None and ref is not None:
        n = min(subj.shape[0], ref.shape[0], t.size)
        rel = subj[:n, :3] - ref[:n, :3]
        rel_v = subj[:n, 3:6] - ref[:n, 3:6]
        rng = np.linalg.norm(rel, axis=1)
        spd = np.linalg.norm(rel_v, axis=1)
        ax_range.plot(t[:n], rng, label="range")
        ax_range_t = ax_range.twinx()
        ax_range_t.plot(t[:n], spd, color="tab:orange", label="speed")
        ax_range.set_ylabel("range (km)")
        ax_range_t.set_ylabel("relative speed (km/s)")
        ax_range.set_title(f"Relative Motion ({subj_id} vs {ref_id})")
        ax_range.grid(True, alpha=0.3)

        ric = _ric_position(subj[:n, :], ref[:n, :])
        labels = ("R", "I", "C")
        for i, label in enumerate(labels):
            ax_ric.plot(t[: ric.shape[0]], ric[:, i], label=label)
        ax_ric.set_title("RIC Position Components")
        ax_ric.set_xlabel("time (s)")
        ax_ric.set_ylabel("km")
        ax_ric.grid(True, alpha=0.3)
        ax_ric.legend(loc="best")
    else:
        ax_range.text(0.5, 0.5, "No relative pair available", ha="center", va="center", transform=ax_range.transAxes)
        ax_ric.text(0.5, 0.5, "No RIC reference available", ha="center", va="center", transform=ax_ric.transAxes)

    for oid, u in sorted(thrust.items()):
        if u.ndim != 2 or u.shape[1] < 3:
            continue
        n = min(u.shape[0], t.size)
        mag = np.linalg.norm(np.nan_to_num(u[:n, :3], nan=0.0), axis=1)
        ax_thrust.plot(t[:n], mag, label=oid)
        ax_dv.plot(t[:n], _cumulative_delta_v_m_s(t[:n], u[:n, :3]), label=oid)
    ax_thrust.set_title("Applied Thrust Magnitude")
    ax_thrust.set_xlabel("time (s)")
    ax_thrust.set_ylabel("km/s^2")
    ax_thrust.grid(True, alpha=0.3)
    ax_dv.set_title("Cumulative Delta-V")
    ax_dv.set_xlabel("time (s)")
    ax_dv.set_ylabel("m/s")
    ax_dv.grid(True, alpha=0.3)
    if thrust:
        ax_thrust.legend(loc="best")
        ax_dv.legend(loc="best")

    for oid, hist in sorted(truth.items()):
        if hist.ndim != 2 or hist.shape[1] < 13:
            continue
        n = min(hist.shape[0], t.size)
        rate = np.linalg.norm(np.nan_to_num(hist[:n, 10:13], nan=0.0), axis=1)
        ax_rate.plot(t[:n], rate, label=oid)
    ax_rate.set_title("Body Rate Norm")
    ax_rate.set_xlabel("time (s)")
    ax_rate.set_ylabel("rad/s")
    ax_rate.grid(True, alpha=0.3)
    if truth:
        ax_rate.legend(loc="best")

    fig.suptitle("Run Dashboard")
    fig.tight_layout()
    _save_show_close(fig, out_path=out_path, show=show, close=close, dpi=dpi)
    return fig


def plot_rendezvous_summary(
    payload: dict[str, Any] | None = None,
    *,
    t_s: np.ndarray | None = None,
    truth_by_object: ArrayMap | None = None,
    target_reference_orbit_truth: np.ndarray | None = None,
    reference_object_id: str | None = None,
    object_id: str | None = None,
    keepout_radius_km: float | None = None,
    out_path: str | Path | None = None,
    show: bool = False,
    close: bool = False,
    dpi: int = 150,
) -> plt.Figure:
    if payload is not None:
        t_s, truth_by_object, _, _, _, target_reference_orbit_truth = _payload_arrays(payload)
    t = np.array([] if t_s is None else t_s, dtype=float).reshape(-1)
    truth = dict(truth_by_object or {})
    ref_id, ref = _choose_reference(truth, target_reference_orbit_truth, reference_object_id)
    subj_id, subj = _choose_subject(truth, ref_id, object_id)

    fig, axes = plt.subplots(2, 3, figsize=cap_figsize(14, 8))
    if subj is None or ref is None:
        for ax in axes.ravel():
            ax.text(0.5, 0.5, "No rendezvous pair available", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        _save_show_close(fig, out_path=out_path, show=show, close=close, dpi=dpi)
        return fig

    n = min(subj.shape[0], ref.shape[0], t.size)
    ric = _ric_position(subj[:n, :], ref[:n, :])
    rel = subj[:n, :3] - ref[:n, :3]
    rel_v = subj[:n, 3:6] - ref[:n, 3:6]
    rng = np.linalg.norm(rel, axis=1)
    spd = np.linalg.norm(rel_v, axis=1)

    planes = ((1, 0, "I", "R"), (1, 2, "I", "C"), (2, 0, "C", "R"))
    for ax, (ix, iy, xlab, ylab) in zip(axes[0], planes):
        ax.plot(ric[:, ix], ric[:, iy], linewidth=1.3)
        if ric.shape[0]:
            ax.scatter([ric[0, ix]], [ric[0, iy]], color="green", s=24, label="start")
            ax.scatter([ric[-1, ix]], [ric[-1, iy]], color="red", s=24, label="end")
        if keepout_radius_km is not None and np.isfinite(float(keepout_radius_km)) and float(keepout_radius_km) > 0.0:
            circ = plt.Circle((0.0, 0.0), float(keepout_radius_km), color="tab:red", fill=False, linestyle="--", alpha=0.6)
            ax.add_patch(circ)
        ax.set_xlabel(f"{xlab} (km)")
        ax.set_ylabel(f"{ylab} (km)")
        ax.set_title(f"{xlab}-{ylab} Projection")
        ax.grid(True, alpha=0.3)
        ax.axis("equal")

    axes[1, 0].plot(t[:n], rng)
    axes[1, 0].set_title("Relative Range")
    axes[1, 0].set_ylabel("km")
    axes[1, 0].set_xlabel("time (s)")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(t[:n], spd)
    axes[1, 1].set_title("Relative Speed")
    axes[1, 1].set_ylabel("km/s")
    axes[1, 1].set_xlabel("time (s)")
    axes[1, 1].grid(True, alpha=0.3)

    for i, label in enumerate(("R", "I", "C")):
        axes[1, 2].plot(t[:n], ric[:, i], label=label)
    axes[1, 2].set_title("RIC Components")
    axes[1, 2].set_ylabel("km")
    axes[1, 2].set_xlabel("time (s)")
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend(loc="best")

    fig.suptitle(f"Rendezvous Summary ({subj_id} vs {ref_id})")
    fig.tight_layout()
    _save_show_close(fig, out_path=out_path, show=show, close=close, dpi=dpi)
    return fig


def plot_control_effort(
    payload: dict[str, Any] | None = None,
    *,
    t_s: np.ndarray | None = None,
    thrust_by_object: ArrayMap | None = None,
    object_id: str | None = None,
    max_accel_km_s2: float | None = None,
    out_path: str | Path | None = None,
    show: bool = False,
    close: bool = False,
    dpi: int = 150,
) -> plt.Figure:
    if payload is not None:
        t_s, _, thrust_by_object, _, _, _ = _payload_arrays(payload)
    t = np.array([] if t_s is None else t_s, dtype=float).reshape(-1)
    thrust = dict(thrust_by_object or {})
    ids = [object_id] if object_id and object_id in thrust else sorted(thrust.keys())

    fig, axes = plt.subplots(2, 1, figsize=cap_figsize(11, 7), sharex=True)
    for oid in ids:
        u = thrust.get(oid)
        if u is None or u.ndim != 2 or u.shape[1] < 3:
            continue
        n = min(u.shape[0], t.size)
        labels = ("x", "y", "z")
        for i, label in enumerate(labels):
            axes[0].plot(t[:n], u[:n, i], linewidth=1.0, label=f"{oid} {label}")
        mag = np.linalg.norm(np.nan_to_num(u[:n, :3], nan=0.0), axis=1)
        axes[1].plot(t[:n], mag, label=f"{oid} |a|")
        axes[1].plot(t[:n], _cumulative_delta_v_m_s(t[:n], u[:n, :3]), linestyle="--", label=f"{oid} dv")
    if max_accel_km_s2 is not None and np.isfinite(float(max_accel_km_s2)) and float(max_accel_km_s2) > 0.0:
        axes[1].axhline(float(max_accel_km_s2), color="tab:red", linestyle=":", label="max accel")
    axes[0].set_title("Applied Thrust Components")
    axes[0].set_ylabel("km/s^2")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    axes[1].set_title("Magnitude and Cumulative Delta-V")
    axes[1].set_ylabel("km/s^2 / m/s")
    axes[1].set_xlabel("time (s)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")
    fig.tight_layout()
    _save_show_close(fig, out_path=out_path, show=show, close=close, dpi=dpi)
    return fig


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
        if obs_truth is not None and target_truth is not None and obs_truth.shape[1] >= 3 and target_truth.shape[1] >= 3:
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


def plot_ground_track_from_payload(
    payload: dict[str, Any] | None = None,
    *,
    t_s: np.ndarray | None = None,
    truth_by_object: ArrayMap | None = None,
    jd_utc_start: float | None = None,
    object_id: str | None = None,
    out_path: str | Path | None = None,
    show: bool = False,
    close: bool = False,
    dpi: int = 150,
) -> plt.Figure:
    if payload is not None:
        t_s, truth_by_object, _, _, _, _ = _payload_arrays(payload)
    t = np.array([] if t_s is None else t_s, dtype=float).reshape(-1)
    truth = dict(truth_by_object or {})
    ids = [object_id] if object_id and object_id in truth else sorted(truth.keys())
    fig, ax = plt.subplots(figsize=cap_figsize(11, 5))
    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(-90.0, 90.0)
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title("Ground Track")
    ax.grid(True, alpha=0.3)
    for oid in ids:
        hist = truth.get(oid)
        if hist is None or hist.shape[1] < 3:
            continue
        n = min(hist.shape[0], t.size)
        lat, lon, _ = ground_track_from_eci_history(hist[:n, :3], t_s=t[:n], jd_utc_start=jd_utc_start)
        lon_p, lat_p = split_ground_track_dateline(lon_deg=lon, lat_deg=lat, jump_threshold_deg=180.0)
        ax.plot(lon_p, lat_p, label=oid)
        finite = np.isfinite(lon) & np.isfinite(lat)
        idx = np.where(finite)[0]
        if idx.size:
            ax.scatter([lon[idx[0]]], [lat[idx[0]]], color="green", s=18)
            ax.scatter([lon[idx[-1]]], [lat[idx[-1]]], color="red", s=18)
    if ids:
        ax.legend(loc="best")
    fig.tight_layout()
    _save_show_close(fig, out_path=out_path, show=show, close=close, dpi=dpi)
    return fig
