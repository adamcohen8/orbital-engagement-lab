from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import expm

from sim.dynamics.orbit.elements import rv_to_coe_eci
from sim.dynamics.orbit.environment import EARTH_J2, EARTH_MU_KM3_S2, EARTH_RADIUS_KM

RELATIVE_LINEAR_MODELS = {"hcw", "ss_j2"}


def normalize_relative_linear_model(value: str) -> str:
    raw = str(value or "hcw").strip().lower().replace("-", "_")
    aliases = {
        "cw": "hcw",
        "clohessy_wiltshire": "hcw",
        "hill_clohessy_wiltshire": "hcw",
        "relative_hcw": "hcw",
        "ss": "ss_j2",
        "ssj2": "ss_j2",
        "schweighart_sedwick": "ss_j2",
        "schweighart_sedwick_j2": "ss_j2",
        "relative_ss_j2": "ss_j2",
    }
    model = aliases.get(raw, raw)
    if model not in RELATIVE_LINEAR_MODELS:
        valid = ", ".join(sorted(RELATIVE_LINEAR_MODELS))
        raise ValueError(f"Unsupported linear relative dynamics model {value!r}. Valid options: {valid}")
    return model


@dataclass(frozen=True)
class RelativeLinearDynamics:
    """Constant-coefficient rectangular-RIC relative dynamics.

    ``hcw`` is the classical circular-chief two-body model. ``ss_j2`` is the
    homogeneous, chief-centered Schweighart-Sedwick averaged-J2 model. The
    latter uses the nodal-drift-corrected cross-track frequency and is intended
    for close formations about circular or near-circular Earth orbits.

    OEL's state is deputy relative to chief in instantaneous rectangular RIC,
    ordered ``[R, I, C, Rdot, Idot, Cdot]`` in km and km/s. The periodic forcing
    terms sometimes printed with the original SS equations describe motion
    relative to an unperturbed reference orbit. They are deliberately excluded
    here because OEL already centers the state on the propagated chief.
    """

    model: str
    mean_motion_rad_s: float
    reference_radius_km: float | None = None
    reference_inclination_rad: float | None = None
    mu_km3_s2: float = EARTH_MU_KM3_S2
    j2: float = EARTH_J2
    earth_radius_km: float = EARTH_RADIUS_KM
    reference_eccentricity: float | None = None
    maximum_supported_eccentricity: float = 0.01

    def __post_init__(self) -> None:
        model = normalize_relative_linear_model(self.model)
        object.__setattr__(self, "model", model)
        for name in ("mean_motion_rad_s", "mu_km3_s2", "earth_radius_km"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
            object.__setattr__(self, name, value)
        if not np.isfinite(self.j2) or self.j2 < 0.0:
            raise ValueError("j2 must be finite and nonnegative.")
        if not np.isfinite(self.maximum_supported_eccentricity) or not (
            0.0 <= self.maximum_supported_eccentricity < 1.0
        ):
            raise ValueError("maximum_supported_eccentricity must be finite and in [0, 1).")
        if self.reference_eccentricity is not None:
            eccentricity = float(self.reference_eccentricity)
            if not np.isfinite(eccentricity) or eccentricity < 0.0:
                raise ValueError("reference_eccentricity must be finite and nonnegative.")
            if model == "ss_j2" and eccentricity > self.maximum_supported_eccentricity:
                raise ValueError(
                    "ss_j2 requires a circular or near-circular chief: "
                    f"eccentricity {eccentricity:g} exceeds {self.maximum_supported_eccentricity:g}."
                )
            object.__setattr__(self, "reference_eccentricity", eccentricity)
        if model == "ss_j2":
            if self.reference_radius_km is None or self.reference_inclination_rad is None:
                raise ValueError("ss_j2 requires reference_radius_km and reference_inclination_rad.")
            radius = float(self.reference_radius_km)
            inclination = float(self.reference_inclination_rad)
            if not np.isfinite(radius) or radius <= self.earth_radius_km:
                raise ValueError("ss_j2 reference_radius_km must be above the Earth reference radius.")
            if not np.isfinite(inclination) or not 0.0 <= inclination <= np.pi:
                raise ValueError("ss_j2 reference_inclination_rad must be in [0, pi].")
            object.__setattr__(self, "reference_radius_km", radius)
            object.__setattr__(self, "reference_inclination_rad", inclination)
            if 1.0 + self.ss_s <= 0.0:
                raise ValueError("ss_j2 parameters produce a non-real mean-motion correction.")

    @classmethod
    def hcw(cls, mean_motion_rad_s: float) -> RelativeLinearDynamics:
        return cls(model="hcw", mean_motion_rad_s=float(mean_motion_rad_s))

    @classmethod
    def ss_j2_from_chief_state(
        cls,
        chief_state_eci_km_s: np.ndarray,
        *,
        mu_km3_s2: float = EARTH_MU_KM3_S2,
        j2: float = EARTH_J2,
        earth_radius_km: float = EARTH_RADIUS_KM,
        maximum_supported_eccentricity: float = 0.01,
    ) -> RelativeLinearDynamics:
        state = np.asarray(chief_state_eci_km_s, dtype=float).reshape(6)
        elements = rv_to_coe_eci(state[:3], state[3:], mu_km3_s2=float(mu_km3_s2))
        mean_motion = float(np.sqrt(float(mu_km3_s2) / elements.a_km**3))
        return cls(
            model="ss_j2",
            mean_motion_rad_s=mean_motion,
            reference_radius_km=float(elements.a_km),
            reference_inclination_rad=float(np.deg2rad(elements.inc_deg)),
            mu_km3_s2=float(mu_km3_s2),
            j2=float(j2),
            earth_radius_km=float(earth_radius_km),
            reference_eccentricity=float(elements.ecc),
            maximum_supported_eccentricity=float(maximum_supported_eccentricity),
        )

    @property
    def ss_s(self) -> float:
        if self.model != "ss_j2":
            return 0.0
        ratio_sq = (float(self.earth_radius_km) / float(self.reference_radius_km)) ** 2
        inclination = float(self.reference_inclination_rad)
        return float(3.0 * self.j2 * ratio_sq * (1.0 + 3.0 * np.cos(2.0 * inclination)) / 8.0)

    @property
    def ss_c(self) -> float:
        return float(np.sqrt(1.0 + self.ss_s))

    @property
    def cross_track_frequency_rad_s(self) -> float:
        n = float(self.mean_motion_rad_s)
        if self.model == "hcw":
            return n
        ratio_sq = (float(self.earth_radius_km) / float(self.reference_radius_km)) ** 2
        inclination = float(self.reference_inclination_rad)
        # Schweighart-Sedwick nodal-drift correction (their k coefficient).
        return float(n * self.ss_c + 1.5 * self.j2 * n * ratio_sq * np.cos(inclination) ** 2)

    def system_matrix(self) -> np.ndarray:
        n = float(self.mean_motion_rad_s)
        c = self.ss_c
        q = self.cross_track_frequency_rad_s
        return np.array(
            [
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [(5.0 * c * c - 2.0) * n * n, 0.0, 0.0, 0.0, 2.0 * n * c, 0.0],
                [0.0, 0.0, 0.0, -2.0 * n * c, 0.0, 0.0],
                [0.0, 0.0, -q * q, 0.0, 0.0, 0.0],
            ],
            dtype=float,
        )

    @staticmethod
    def input_matrix() -> np.ndarray:
        return np.vstack((np.zeros((3, 3), dtype=float), np.eye(3, dtype=float)))

    def state_transition_matrix(self, dt_s: float) -> np.ndarray:
        dt = float(dt_s)
        if not np.isfinite(dt) or dt < 0.0:
            raise ValueError("dt_s must be finite and nonnegative.")
        if self.model == "hcw":
            return _hcw_state_transition_matrix(self.mean_motion_rad_s, dt)
        return np.asarray(expm(self.system_matrix() * dt), dtype=float)

    def discrete_matrices(self, dt_s: float) -> tuple[np.ndarray, np.ndarray]:
        dt = float(dt_s)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt_s must be finite and positive.")
        augmented = np.zeros((9, 9), dtype=float)
        augmented[:6, :6] = self.system_matrix()
        augmented[:6, 6:] = self.input_matrix()
        transition = expm(augmented * dt)
        return np.asarray(transition[:6, :6], dtype=float), np.asarray(transition[:6, 6:], dtype=float)

    def propagate(self, state_ric: np.ndarray, dt_s: float) -> np.ndarray:
        state = np.asarray(state_ric, dtype=float).reshape(6)
        return self.state_transition_matrix(float(dt_s)) @ state

    def metadata(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "state_basis": "rectangular_RIC_deputy_relative_to_chief",
            "state_units": ["km", "km", "km", "km/s", "km/s", "km/s"],
            "mean_motion_rad_s": float(self.mean_motion_rad_s),
            "constant_coefficient": True,
        }
        if self.model == "ss_j2":
            payload.update(
                {
                    "reference_semantics": "mean_near_circular_chief_centered_instantaneous_RIC",
                    "reference_element_source": "caller_values_or_osculating_state_used_as_mean_proxy",
                    "reference_radius_km": float(self.reference_radius_km),
                    "reference_inclination_rad": float(self.reference_inclination_rad),
                    "reference_eccentricity": self.reference_eccentricity,
                    "maximum_supported_eccentricity": float(self.maximum_supported_eccentricity),
                    "j2": float(self.j2),
                    "earth_radius_km": float(self.earth_radius_km),
                    "ss_s": self.ss_s,
                    "ss_c": self.ss_c,
                    "cross_track_frequency_rad_s": self.cross_track_frequency_rad_s,
                    "cross_track_policy": "nodal_drift_corrected_homogeneous_frequency",
                    "forcing_policy": "unperturbed_reference_periodic_forcing_excluded_for_chief_centered_RIC",
                }
            )
        return payload


def _hcw_state_transition_matrix(mean_motion_rad_s: float, dt_s: float) -> np.ndarray:
    n = float(mean_motion_rad_s)
    dt = float(dt_s)
    nt = n * dt
    c = float(np.cos(nt))
    s = float(np.sin(nt))
    return np.array(
        [
            [4.0 - 3.0 * c, 0.0, 0.0, s / n, 2.0 * (1.0 - c) / n, 0.0],
            [6.0 * (s - nt), 1.0, 0.0, -2.0 * (1.0 - c) / n, (4.0 * s - 3.0 * nt) / n, 0.0],
            [0.0, 0.0, c, 0.0, 0.0, s / n],
            [3.0 * n * s, 0.0, 0.0, c, 2.0 * s, 0.0],
            [-6.0 * n * (1.0 - c), 0.0, 0.0, -2.0 * s, 4.0 * c - 3.0, 0.0],
            [0.0, 0.0, -n * s, 0.0, 0.0, c],
        ],
        dtype=float,
    )
