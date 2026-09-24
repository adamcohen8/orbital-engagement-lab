"""Strict configuration for the optional, Earth/ECI ground knowledge layer."""

from copy import deepcopy

import numpy as np

MAX_PREDICTION_STEPS = 10_000


def normalize_config(raw):
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("ground_segment must be a mapping")
    cfg = deepcopy(raw)
    allowed = {
        "enabled",
        "cadence_s",
        "latency_s",
        "stale_after_s",
        "prediction_step_s",
        "process_noise_diag",
        "priors",
        "outages",
    }
    if set(cfg) - allowed:
        raise ValueError(f"ground_segment unknown fields: {sorted(set(cfg) - allowed)}")
    if type(cfg.get("enabled", False)) is not bool:
        raise ValueError("ground_segment.enabled must be boolean")
    for key, default in (("cadence_s", 1.0), ("latency_s", 0.0), ("stale_after_s", 10.0), ("prediction_step_s", 1.0)):
        value = cfg.get(key, default)
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not np.isfinite(value)
            or value < 0
            or (key != "latency_s" and value == 0)
        ):
            raise ValueError(
                f"ground_segment.{key} must be finite and {'nonnegative' if key == 'latency_s' else 'positive'}"
            )
        cfg[key] = float(value)
    q = np.asarray(cfg.get("process_noise_diag", [0.0] * 6), dtype=float)
    if q.shape != (6,) or not np.all(np.isfinite(q)) or np.any(q < 0):
        raise ValueError("ground_segment.process_noise_diag needs six finite nonnegative variances per second")
    cfg["process_noise_diag"] = q.tolist()
    priors = cfg.setdefault("priors", {})
    if not isinstance(priors, dict):
        raise ValueError("ground_segment.priors must be a mapping")
    for oid, prior in priors.items():
        if not isinstance(prior, dict) or set(prior) != {"state", "covariance", "epoch_s"}:
            raise ValueError(f"ground_segment.priors.{oid} requires state, covariance, epoch_s")
        x, p = np.asarray(prior["state"], dtype=float), np.asarray(prior["covariance"], dtype=float)
        t = float(prior["epoch_s"])
        if x.shape != (6,) or not np.all(np.isfinite(x)) or np.linalg.norm(x[:3]) == 0:
            raise ValueError("ground_segment prior state must be finite ECI [km, km/s]")
        if (
            p.shape != (6, 6)
            or not np.all(np.isfinite(p))
            or not np.allclose(p, p.T, atol=1e-12, rtol=0)
            or np.min(np.linalg.eigvalsh(p)) < -1e-12
        ):
            raise ValueError("ground_segment prior covariance must be symmetric positive semidefinite 6x6")
        if not np.isfinite(t) or t > 0:
            raise ValueError("ground_segment prior epoch must be finite and <= 0")
    outages = cfg.setdefault("outages", [])
    if not isinstance(outages, list):
        raise ValueError("ground_segment.outages must be a list")
    for row in outages:
        if not isinstance(row, dict) or set(row) != {"station_id", "start_s", "end_s", "services"}:
            raise ValueError("ground_segment outage requires station_id, start_s, end_s, services")
        a, b = float(row["start_s"]), float(row["end_s"])
        if not np.isfinite(a) or not np.isfinite(b) or a < 0 or b <= a:
            raise ValueError("ground_segment outage needs 0 <= start_s < end_s")
        if (
            not isinstance(row["services"], list)
            or not row["services"]
            or set(row["services"]) - {"tracking", "downlink", "uplink"}
        ):
            raise ValueError("ground_segment outage services must be tracking, downlink, or uplink")
    return cfg


def validate_scenario(cfg):
    ground = cfg.ground_segment
    if not ground.get("enabled", False):
        return
    if not cfg.ground_stations:
        raise ValueError("ground_segment requires ground_stations")
    orbit = dict(cfg.simulator.dynamics.get("orbit", {}))
    if orbit.get("model", "two_body") != "two_body" or any(
        o.enabled and o.kind != "satellite" for o in cfg.objects.values()
    ):
        raise ValueError("ground_segment currently requires Earth-centered satellite scenarios")
    ids = {oid for oid, o in cfg.objects.items() if o.enabled}
    if set(ground["priors"]) - ids:
        raise ValueError("ground_segment prior references unknown object")
    stations = {s.id for s in cfg.ground_stations}
    if any(o["station_id"] not in stations for o in ground["outages"]):
        raise ValueError("ground_segment outage references unknown station")
    dt = float(cfg.simulator.dt_s)
    oldest_prior = min((float(p["epoch_s"]) for p in ground["priors"].values()), default=0.0)
    if (dt - oldest_prior) / ground["prediction_step_s"] > MAX_PREDICTION_STEPS:
        raise ValueError("ground_segment prediction_step_s requires too many steps from the prior or across one simulation step")
    cadences = [ground["cadence_s"]] + [
        float(s.measurements.get("update_cadence_s", 1.0)) for s in cfg.ground_stations if s.measurements.get("enabled")
    ]
    if any(c < dt or not np.isclose(c / dt, round(c / dt), atol=1e-9, rtol=0) for c in cadences):
        raise ValueError("ground_segment and station measurement cadences must be integer multiples of simulator.dt_s")
