from __future__ import annotations

import numpy as np

from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2, EARTH_RADIUS_KM
from sim.plotting.output_context import PlotOutputContext
from sim.plotting.output_helpers import (
    _first_true_time,
    _haversine_distance_km,
    _last_finite_value,
    _max_abs_finite_value,
    _max_finite_value,
    _orbital_elements_basic,
    _rocket_launch_site,
    _rocket_metric_array,
    _rocket_target_altitude_cfg,
)
from sim.plotting.style import save_oel_figure
from sim.utils.figure_size import cap_figsize
from sim.utils.ground_track import ground_track_from_eci_history

FIGURE_IDS = (
    "rocket_ascent_diagnostics",
    "rocket_gnc_diagnostics",
    "rocket_orbital_elements",
    "rocket_fuel_remaining",
    "rocket_mission_timeline",
    "rocket_downrange_altitude",
    "rocket_maxq_throttle",
    "rocket_tvc_aero_authority",
    "rocket_insertion_scorecard",
)


def render_rocket_outputs(context: PlotOutputContext) -> dict[str, str]:
    cfg = context.cfg
    t_s = context.t_s
    truth_hist = context.truth_hist
    thrust_hist = context.thrust_hist
    rocket_metrics = context.rocket_metrics
    outdir = context.outdir
    resolve_rocket_stack = context.resolve_rocket_stack
    figure_ids = context.figure_ids
    frame_context = context.frame_context
    mode = context.mode
    out: dict[str, str] = {}
    if "rocket_ascent_diagnostics" in figure_ids and "rocket" in truth_hist:
        import matplotlib.pyplot as plt

        x = truth_hist["rocket"]
        r = x[:, 0:3]
        v = x[:, 3:6]
        m = x[:, 13]
        alt_km = np.linalg.norm(r, axis=1) - EARTH_RADIUS_KM
        speed_km_s = np.linalg.norm(v, axis=1)
        q_dyn = np.zeros_like(t_s)
        mach = np.zeros_like(t_s)
        stage = np.zeros_like(t_s)
        throttle = np.zeros_like(t_s)
        if rocket_metrics is not None:
            if "q_dyn_pa" in rocket_metrics:
                q_dyn = np.array(rocket_metrics["q_dyn_pa"], dtype=float).reshape(-1)[: t_s.size]
            if "mach" in rocket_metrics:
                mach = np.array(rocket_metrics["mach"], dtype=float).reshape(-1)[: t_s.size]
            if "stage_index" in rocket_metrics:
                stage = np.array(rocket_metrics["stage_index"], dtype=float).reshape(-1)[: t_s.size]
            if "throttle_cmd" in rocket_metrics:
                throttle = np.array(rocket_metrics["throttle_cmd"], dtype=float).reshape(-1)[: t_s.size]
        a_cmd = np.linalg.norm(np.nan_to_num(thrust_hist.get("rocket", np.zeros((t_s.size, 3))), nan=0.0), axis=1)

        fig, ax = plt.subplots(4, 1, figsize=cap_figsize(11, 11), sharex=True)

        ax0r = ax[0].twinx()
        l00 = ax[0].plot(t_s, alt_km, label="altitude (km)", color="tab:blue")
        l01 = ax0r.plot(t_s, speed_km_s, label="speed (km/s)", color="tab:orange")
        ax[0].set_ylabel("altitude (km)")
        ax0r.set_ylabel("speed (km/s)")
        ax[0].set_title("Rocket Ascent: Altitude and Speed")
        ax[0].grid(True, alpha=0.3)
        ax[0].legend(l00 + l01, [ln.get_label() for ln in (l00 + l01)], loc="best")

        ax1r = ax[1].twinx()
        l10 = ax[1].plot(t_s, q_dyn, label="q_dyn (Pa)", color="tab:green")
        l11 = ax1r.plot(t_s, mach, label="Mach", color="tab:red")
        ax[1].set_ylabel("dynamic pressure (Pa)")
        ax1r.set_ylabel("Mach")
        ax[1].set_title("Dynamic Pressure and Mach")
        ax[1].grid(True, alpha=0.3)
        ax[1].legend(l10 + l11, [ln.get_label() for ln in (l10 + l11)], loc="best")

        ax2r = ax[2].twinx()
        l20 = ax[2].plot(t_s, m, label="mass (kg)", color="tab:purple")
        l21 = ax2r.step(t_s, stage, where="post", label="stage index", color="tab:brown")
        ax[2].set_ylabel("mass (kg)")
        ax2r.set_ylabel("stage index")
        ax[2].set_title("Mass and Stage")
        ax[2].grid(True, alpha=0.3)
        ax[2].legend(l20 + l21, [ln.get_label() for ln in (l20 + l21)], loc="best")

        ax3r = ax[3].twinx()
        l30 = ax[3].plot(t_s, throttle, label="throttle", color="tab:cyan")
        l31 = ax3r.plot(t_s, a_cmd, label="|a_cmd| (km/s^2)", color="tab:gray")
        ax[3].set_ylabel("throttle")
        ax3r.set_ylabel("|a_cmd| (km/s^2)")
        ax[3].set_xlabel("time (s)")
        ax[3].set_title("Throttle and Commanded Acceleration")
        ax[3].grid(True, alpha=0.3)
        ax[3].legend(l30 + l31, [ln.get_label() for ln in (l30 + l31)], loc="best")
        fig.tight_layout()
        p = outdir / "rocket_ascent_diagnostics.png"
        if mode in ("save", "both"):
            save_oel_figure(fig, p, dpi=int(cfg.outputs.plots.get("dpi", 150)), artifact_id=p.stem)
            out["rocket_ascent_diagnostics"] = str(p)
        if mode == "save":
            plt.close(fig)

    if "rocket_gnc_diagnostics" in figure_ids and "rocket" in truth_hist:
        import matplotlib.pyplot as plt

        def _metric(name: str, default: float = 0.0) -> np.ndarray:
            if rocket_metrics is None or name not in rocket_metrics:
                return np.full(t_s.size, default, dtype=float)
            arr = np.array(rocket_metrics[name], dtype=float).reshape(-1)
            out_arr = np.full(t_s.size, np.nan, dtype=float)
            n = min(t_s.size, arr.size)
            out_arr[:n] = arr[:n]
            return out_arr

        fpa = _metric("flight_path_angle_deg")
        vertical_speed = _metric("vertical_speed_km_s")
        alpha = _metric("alpha_deg")
        beta = _metric("beta_deg")
        tvc = _metric("tvc_gimbal_deg")
        twr = _metric("thrust_to_weight")
        apo = _metric("apoapsis_alt_km", np.nan)
        peri = _metric("periapsis_alt_km", np.nan)

        fig, ax = plt.subplots(4, 1, figsize=cap_figsize(11, 11), sharex=True)

        ax0r = ax[0].twinx()
        l00 = ax[0].plot(t_s, fpa, label="flight path angle (deg)", color="tab:blue")
        l01 = ax0r.plot(t_s, vertical_speed, label="vertical speed (km/s)", color="tab:orange")
        ax[0].set_ylabel("FPA (deg)")
        ax0r.set_ylabel("vertical speed (km/s)")
        ax[0].set_title("Rocket GNC: Flight-Path State")
        ax[0].grid(True, alpha=0.3)
        ax[0].legend(l00 + l01, [ln.get_label() for ln in (l00 + l01)], loc="best")

        l10 = ax[1].plot(t_s, alpha, label="alpha (deg)", color="tab:red")
        l11 = ax[1].plot(t_s, beta, label="beta (deg)", color="tab:purple")
        ax[1].set_ylabel("angle (deg)")
        ax[1].set_title("Aero Angles")
        ax[1].grid(True, alpha=0.3)
        ax[1].legend(l10 + l11, [ln.get_label() for ln in (l10 + l11)], loc="best")

        ax2r = ax[2].twinx()
        l20 = ax[2].plot(t_s, tvc, label="TVC gimbal (deg)", color="tab:green")
        l21 = ax2r.plot(t_s, twr, label="thrust-to-weight", color="tab:brown")
        ax[2].set_ylabel("gimbal (deg)")
        ax2r.set_ylabel("T/W")
        ax[2].set_title("Control Authority")
        ax[2].grid(True, alpha=0.3)
        ax[2].legend(l20 + l21, [ln.get_label() for ln in (l20 + l21)], loc="best")

        l30 = ax[3].plot(t_s, apo, label="apogee alt (km)", color="tab:cyan")
        l31 = ax[3].plot(t_s, peri, label="perigee alt (km)", color="tab:gray")
        ax[3].set_ylabel("altitude (km)")
        ax[3].set_xlabel("time (s)")
        ax[3].set_title("Targeting Energy")
        ax[3].grid(True, alpha=0.3)
        ax[3].legend(l30 + l31, [ln.get_label() for ln in (l30 + l31)], loc="best")

        fig.tight_layout()
        p = outdir / "rocket_gnc_diagnostics.png"
        if mode in ("save", "both"):
            save_oel_figure(fig, p, dpi=int(cfg.outputs.plots.get("dpi", 150)), artifact_id=p.stem)
            out["rocket_gnc_diagnostics"] = str(p)
        if mode == "save":
            plt.close(fig)

    if "rocket_orbital_elements" in figure_ids and "rocket" in truth_hist:
        import matplotlib.pyplot as plt

        x = truth_hist["rocket"]
        a_km = np.full(t_s.size, np.nan, dtype=float)
        e = np.full(t_s.size, np.nan, dtype=float)
        for k in range(min(t_s.size, x.shape[0])):
            a_km[k], e[k] = _orbital_elements_basic(x[k, 0:3], x[k, 3:6], EARTH_MU_KM3_S2)

        fig, ax = plt.subplots(2, 1, figsize=cap_figsize(10, 7), sharex=True)
        ax[0].plot(t_s, a_km)
        ax[0].set_ylabel("a (km)")
        ax[0].set_title("Rocket Orbital Elements")
        ax[0].grid(True, alpha=0.3)

        ax[1].plot(t_s, e)
        ax[1].set_ylabel("e")
        ax[1].set_xlabel("time (s)")
        ax[1].grid(True, alpha=0.3)
        fig.tight_layout()
        p = outdir / "rocket_orbital_elements.png"
        if mode in ("save", "both"):
            save_oel_figure(fig, p, dpi=int(cfg.outputs.plots.get("dpi", 150)), artifact_id=p.stem)
            out["rocket_orbital_elements"] = str(p)
        if mode == "save":
            plt.close(fig)

    if "rocket_fuel_remaining" in figure_ids and "rocket" in truth_hist:
        import matplotlib.pyplot as plt

        x = truth_hist["rocket"]
        m = np.array(x[:, 13], dtype=float).reshape(-1)
        stack = resolve_rocket_stack(dict(cfg.rocket.specs or {}))
        payload_kg = float((cfg.rocket.specs or {}).get("payload_mass_kg", 150.0))
        dry_total_kg = float(sum(float(s.dry_mass_kg) for s in stack.stages) + payload_kg)
        prop0_kg = float(sum(float(s.propellant_mass_kg) for s in stack.stages))
        if prop0_kg > 0.0:
            fuel_rem_kg = np.clip(m - dry_total_kg, 0.0, prop0_kg)
            fuel_pct = 100.0 * fuel_rem_kg / prop0_kg
        else:
            fuel_pct = np.zeros_like(m)

        fig, ax = plt.subplots(figsize=cap_figsize(10, 4.5))
        ax.plot(t_s, fuel_pct, linewidth=1.6)
        ax.set_ylim(-1.0, 101.0)
        ax.set_ylabel("Fuel Remaining (%)")
        ax.set_xlabel("time (s)")
        ax.set_title("Rocket Fuel Remaining")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = outdir / "rocket_fuel_remaining.png"
        if mode in ("save", "both"):
            save_oel_figure(fig, p, dpi=int(cfg.outputs.plots.get("dpi", 150)), artifact_id=p.stem)
            out["rocket_fuel_remaining"] = str(p)
        if mode == "save":
            plt.close(fig)

    if "rocket_mission_timeline" in figure_ids and "rocket" in truth_hist:
        import matplotlib.pyplot as plt

        x = truth_hist["rocket"]
        target_alt_km, alt_tol_km, ecc_max = _rocket_target_altitude_cfg(cfg)
        alt_km = _rocket_metric_array(rocket_metrics, "altitude_km", t_s.size)
        if not np.any(np.isfinite(alt_km)):
            alt_km = np.linalg.norm(x[:, 0:3], axis=1) - EARTH_RADIUS_KM
        apo = _rocket_metric_array(rocket_metrics, "apoapsis_alt_km", t_s.size)
        peri = _rocket_metric_array(rocket_metrics, "periapsis_alt_km", t_s.size)
        ecc = _rocket_metric_array(rocket_metrics, "eccentricity", t_s.size)
        q_dyn = _rocket_metric_array(rocket_metrics, "q_dyn_pa", t_s.size, 0.0)
        stage = _rocket_metric_array(rocket_metrics, "stage_index", t_s.size, 0.0)

        events: list[tuple[float, str, str]] = [(float(t_s[0]) if t_s.size else 0.0, "Liftoff", "tab:green")]
        guidance = getattr(cfg.rocket, "base_guidance", None)
        guidance_params = dict(getattr(guidance, "params", {}) or {})
        for key, label in (("pitch_start_s", "Pitch start"), ("pitch_end_s", "Pitch complete")):
            value = guidance_params.get(key)
            if value is not None:
                events.append((float(value), label, "tab:blue"))
        finite_q = np.isfinite(q_dyn)
        if np.any(finite_q):
            i_q = int(np.nanargmax(np.where(finite_q, q_dyn, np.nan)))
            events.append((float(t_s[i_q]), "Max Q", "tab:red"))
        finite_stage = np.isfinite(stage)
        if np.any(finite_stage):
            for idx in np.flatnonzero(np.diff(stage[finite_stage]) > 0.5):
                event_t = float(t_s[np.flatnonzero(finite_stage)[idx + 1]])
                events.append((event_t, "Stage event", "tab:purple"))
        insertion_mask = np.zeros(t_s.size, dtype=bool)
        if np.isfinite(target_alt_km) and np.isfinite(alt_tol_km) and np.isfinite(ecc_max):
            altitude_ok = np.abs(alt_km - target_alt_km) <= alt_tol_km
            orbit_ok = np.isfinite(apo) & np.isfinite(peri) & (ecc <= ecc_max)
            insertion_mask = altitude_ok & orbit_ok
        insertion_t = _first_true_time(t_s, insertion_mask)
        if insertion_t is not None:
            events.append((insertion_t, "Insertion band", "tab:orange"))
        if t_s.size:
            events.append((float(t_s[-1]), "Final sample", "tab:gray"))

        fig, ax = plt.subplots(figsize=cap_figsize(11, 3.8))
        ax.axhline(0.0, color="0.3", linewidth=1.4)
        for i, (event_t, label, color) in enumerate(sorted(events, key=lambda row: row[0])):
            offset = 0.34 if i % 2 == 0 else -0.34
            ax.vlines(event_t, 0.0, offset, color=color, linewidth=1.6)
            ax.scatter([event_t], [0.0], color=color, s=42, zorder=3)
            ax.text(event_t, offset, label, ha="center", va="bottom" if offset > 0 else "top", fontsize=8)
        ax.set_xlabel("Time (s)")
        ax.set_yticks([])
        ax.set_ylim(-0.9, 0.9)
        ax.set_title("Rocket Mission Timeline")
        ax.grid(True, axis="x", alpha=0.25)
        fig.tight_layout()
        p = outdir / "rocket_mission_timeline.png"
        if mode in ("save", "both"):
            save_oel_figure(fig, p, dpi=int(cfg.outputs.plots.get("dpi", 150)), artifact_id=p.stem)
            out["rocket_mission_timeline"] = str(p)
        if mode in ("interactive", "both"):
            plt.show(block=False)
        else:
            plt.close(fig)

    if "rocket_downrange_altitude" in figure_ids and "rocket" in truth_hist:
        import matplotlib.pyplot as plt

        x = truth_hist["rocket"]
        launch_site = _rocket_launch_site(cfg)
        lat, lon, _ = ground_track_from_eci_history(
            x[:, 0:3],
            t_s=t_s[: x.shape[0]],
            jd_utc_start=cfg.simulator.initial_jd_utc,
            frame_context=frame_context,
        )
        if launch_site is None:
            lat0, lon0 = float(lat[0]), float(lon[0])
        else:
            lat0, lon0 = launch_site
        downrange_km = _haversine_distance_km(lat0, lon0, lat, lon)
        alt_km = _rocket_metric_array(rocket_metrics, "altitude_km", t_s.size)
        if not np.any(np.isfinite(alt_km)):
            alt_km = np.linalg.norm(x[:, 0:3], axis=1) - EARTH_RADIUS_KM
        speed = _rocket_metric_array(rocket_metrics, "speed_km_s", t_s.size)

        fig, ax = plt.subplots(figsize=cap_figsize(10, 5.5))
        n = min(downrange_km.size, alt_km.size, t_s.size)
        if n > 0 and np.any(np.isfinite(downrange_km[:n]) & np.isfinite(alt_km[:n])):
            if np.any(np.isfinite(speed[:n])):
                sc = ax.scatter(downrange_km[:n], alt_km[:n], c=speed[:n], s=9, cmap="viridis")
                fig.colorbar(sc, ax=ax, label="speed (km/s)")
            else:
                ax.plot(downrange_km[:n], alt_km[:n], linewidth=1.5)
            ax.scatter([downrange_km[0]], [alt_km[0]], color="tab:green", s=35, label="start")
            ax.scatter([downrange_km[n - 1]], [alt_km[n - 1]], color="tab:red", s=35, label="final")
            ax.legend(loc="best")
        else:
            ax.text(0.5, 0.5, "No valid downrange/altitude samples", transform=ax.transAxes, ha="center")
        ax.set_xlabel("Downrange distance (km)")
        ax.set_ylabel("Altitude (km)")
        ax.set_title("Rocket Altitude vs Downrange Distance")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = outdir / "rocket_downrange_altitude.png"
        if mode in ("save", "both"):
            save_oel_figure(fig, p, dpi=int(cfg.outputs.plots.get("dpi", 150)), artifact_id=p.stem)
            out["rocket_downrange_altitude"] = str(p)
        if mode in ("interactive", "both"):
            plt.show(block=False)
        else:
            plt.close(fig)

    if "rocket_maxq_throttle" in figure_ids and "rocket" in truth_hist:
        import matplotlib.pyplot as plt

        q_dyn = _rocket_metric_array(rocket_metrics, "q_dyn_pa", t_s.size, 0.0)
        throttle = _rocket_metric_array(rocket_metrics, "throttle_cmd", t_s.size, 0.0)
        mach = _rocket_metric_array(rocket_metrics, "mach", t_s.size, 0.0)
        alt_km = _rocket_metric_array(rocket_metrics, "altitude_km", t_s.size)
        fig, axes = plt.subplots(3, 1, figsize=cap_figsize(11, 8), sharex=True)
        axes[0].plot(t_s, q_dyn, color="tab:red", label="dynamic pressure")
        max_q_cfg = None
        for modifier in list(getattr(cfg.rocket, "guidance_modifiers", []) or []):
            params = dict(getattr(modifier, "params", {}) or {})
            if params.get("max_q_pa") is not None:
                max_q_cfg = float(params.get("max_q_pa"))
                break
        if max_q_cfg is not None:
            axes[0].axhline(max_q_cfg, color="black", linestyle="--", label=f"limit {max_q_cfg:.0f} Pa")
        if np.any(np.isfinite(q_dyn)):
            i_q = int(np.nanargmax(np.where(np.isfinite(q_dyn), q_dyn, np.nan)))
            axes[0].axvline(t_s[i_q], color="tab:red", linestyle=":", alpha=0.8)
            axes[0].text(t_s[i_q], q_dyn[i_q], " max Q", fontsize=8, va="bottom")
        axes[0].set_ylabel("q (Pa)")
        axes[0].set_title("Max-Q Throttle Limiting")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="best")

        axes[1].plot(t_s, throttle, color="tab:blue", label="throttle")
        axes[1].set_ylabel("throttle")
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="best")

        axes[2].plot(t_s, mach, color="tab:purple", label="Mach")
        if np.any(np.isfinite(alt_km)):
            ax_alt = axes[2].twinx()
            ax_alt.plot(t_s, alt_km, color="tab:gray", alpha=0.7, label="altitude")
            ax_alt.set_ylabel("altitude (km)")
        axes[2].set_xlabel("Time (s)")
        axes[2].set_ylabel("Mach")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc="best")
        fig.tight_layout()
        p = outdir / "rocket_maxq_throttle.png"
        if mode in ("save", "both"):
            save_oel_figure(fig, p, dpi=int(cfg.outputs.plots.get("dpi", 150)), artifact_id=p.stem)
            out["rocket_maxq_throttle"] = str(p)
        if mode in ("interactive", "both"):
            plt.show(block=False)
        else:
            plt.close(fig)

    if "rocket_tvc_aero_authority" in figure_ids and "rocket" in truth_hist:
        import matplotlib.pyplot as plt

        tvc = _rocket_metric_array(rocket_metrics, "tvc_gimbal_deg", t_s.size, 0.0)
        alpha = _rocket_metric_array(rocket_metrics, "alpha_deg", t_s.size, 0.0)
        beta = _rocket_metric_array(rocket_metrics, "beta_deg", t_s.size, 0.0)
        aero_force = _rocket_metric_array(rocket_metrics, "aero_force_n", t_s.size, 0.0)
        aero_moment = _rocket_metric_array(rocket_metrics, "aero_moment_nm", t_s.size, 0.0)
        q_dyn = _rocket_metric_array(rocket_metrics, "q_dyn_pa", t_s.size, 0.0)
        twr = _rocket_metric_array(rocket_metrics, "thrust_to_weight", t_s.size)

        fig, axes = plt.subplots(4, 1, figsize=cap_figsize(11, 10), sharex=True)
        axes[0].plot(t_s, tvc, color="tab:green", label="TVC gimbal")
        tvc_limit = float(dict(cfg.simulator.dynamics.rocket).get("tvc_max_gimbal_deg", np.nan))
        if np.isfinite(tvc_limit):
            axes[0].axhline(tvc_limit, color="black", linestyle="--", linewidth=0.9)
            axes[0].axhline(-tvc_limit, color="black", linestyle="--", linewidth=0.9)
        axes[0].set_ylabel("deg")
        axes[0].set_title("TVC and Aero Authority")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="best")

        axes[1].plot(t_s, alpha, label="alpha", color="tab:red")
        axes[1].plot(t_s, beta, label="beta", color="tab:purple")
        axes[1].set_ylabel("deg")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="best")

        ax_force = axes[2].twinx()
        l0 = axes[2].plot(t_s, aero_force, label="aero force", color="tab:orange")
        l1 = ax_force.plot(t_s, aero_moment, label="aero moment", color="tab:brown")
        axes[2].set_ylabel("force (N)")
        ax_force.set_ylabel("moment (N-m)")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(l0 + l1, [ln.get_label() for ln in l0 + l1], loc="best")

        ax_twr = axes[3].twinx()
        l2 = axes[3].plot(t_s, q_dyn, label="q", color="tab:blue")
        l3 = ax_twr.plot(t_s, twr, label="T/W", color="tab:gray")
        axes[3].set_xlabel("Time (s)")
        axes[3].set_ylabel("q (Pa)")
        ax_twr.set_ylabel("T/W")
        axes[3].grid(True, alpha=0.3)
        axes[3].legend(l2 + l3, [ln.get_label() for ln in l2 + l3], loc="best")
        fig.tight_layout()
        p = outdir / "rocket_tvc_aero_authority.png"
        if mode in ("save", "both"):
            save_oel_figure(fig, p, dpi=int(cfg.outputs.plots.get("dpi", 150)), artifact_id=p.stem)
            out["rocket_tvc_aero_authority"] = str(p)
        if mode in ("interactive", "both"):
            plt.show(block=False)
        else:
            plt.close(fig)

    if "rocket_insertion_scorecard" in figure_ids and "rocket" in truth_hist:
        import matplotlib.pyplot as plt

        target_alt_km, alt_tol_km, ecc_max = _rocket_target_altitude_cfg(cfg)
        final_alt = _last_finite_value(_rocket_metric_array(rocket_metrics, "altitude_km", t_s.size))
        final_apo = _last_finite_value(_rocket_metric_array(rocket_metrics, "apoapsis_alt_km", t_s.size))
        final_peri = _last_finite_value(_rocket_metric_array(rocket_metrics, "periapsis_alt_km", t_s.size))
        final_ecc = _last_finite_value(_rocket_metric_array(rocket_metrics, "eccentricity", t_s.size))
        prop_frac = _last_finite_value(
            _rocket_metric_array(rocket_metrics, "propellant_remaining_fraction", t_s.size)
        )
        max_q = _max_finite_value(_rocket_metric_array(rocket_metrics, "q_dyn_pa", t_s.size))
        max_alpha = _max_abs_finite_value(_rocket_metric_array(rocket_metrics, "alpha_deg", t_s.size))
        max_tvc = _max_finite_value(_rocket_metric_array(rocket_metrics, "tvc_gimbal_deg", t_s.size))
        max_force = _max_finite_value(_rocket_metric_array(rocket_metrics, "aero_force_n", t_s.size))
        metrics_rows = [
            ("Final altitude", final_alt, "km", target_alt_km, alt_tol_km),
            ("Final apogee", final_apo, "km", target_alt_km, alt_tol_km),
            ("Final perigee", final_peri, "km", target_alt_km, alt_tol_km),
            ("Final eccentricity", final_ecc, "", ecc_max, None),
            ("Propellant remaining", prop_frac, "fraction", None, None),
            ("Max dynamic pressure", max_q, "Pa", None, None),
            ("Max |alpha|", max_alpha, "deg", None, None),
            ("Max TVC gimbal", max_tvc, "deg", float(dict(cfg.simulator.dynamics.rocket).get("tvc_max_gimbal_deg", np.nan)), None),
            ("Max aero force", max_force, "N", None, None),
        ]
        fig, ax = plt.subplots(figsize=cap_figsize(10, 5.8))
        ax.axis("off")
        title = "Rocket Insertion Scorecard"
        if np.isfinite(target_alt_km):
            title += f" (target {target_alt_km:.0f} km)"
        ax.set_title(title, fontsize=14, pad=16)
        table_data = []
        row_colors = []
        for name, value, unit, target, tol in metrics_rows:
            value_txt = "n/a" if not np.isfinite(value) else f"{value:.3g}"
            if unit:
                value_txt = f"{value_txt} {unit}"
            target_txt = ""
            passed = None
            if target is not None and np.isfinite(float(target)):
                if tol is not None and np.isfinite(float(tol)):
                    target_txt = f"{float(target):.3g} +/- {float(tol):.3g}"
                    passed = bool(np.isfinite(value) and abs(value - float(target)) <= float(tol))
                elif name == "Final eccentricity":
                    target_txt = f"<= {float(target):.3g}"
                    passed = bool(np.isfinite(value) and value <= float(target))
                elif "TVC" in name:
                    target_txt = f"<= {float(target):.3g}"
                    passed = bool(np.isfinite(value) and value <= float(target))
            status = "OK" if passed is True else ("Check" if passed is False else "")
            table_data.append([name, value_txt, target_txt, status])
            row_colors.append("#eaf6ea" if passed is True else ("#fdeaea" if passed is False else "#f7f7f7"))
        table = ax.table(
            cellText=table_data,
            colLabels=["Metric", "Value", "Target / Limit", "Status"],
            loc="center",
            cellLoc="left",
            colLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.35)
        for row_idx, color in enumerate(row_colors, start=1):
            for col_idx in range(4):
                table[(row_idx, col_idx)].set_facecolor(color)
        for col_idx in range(4):
            table[(0, col_idx)].set_facecolor("#d9e8f5")
            table[(0, col_idx)].set_text_props(weight="bold")
        fig.tight_layout()
        p = outdir / "rocket_insertion_scorecard.png"
        if mode in ("save", "both"):
            save_oel_figure(fig, p, dpi=int(cfg.outputs.plots.get("dpi", 150)), artifact_id=p.stem)
            out["rocket_insertion_scorecard"] = str(p)
        if mode in ("interactive", "both"):
            plt.show(block=False)
        else:
            plt.close(fig)

    return out
