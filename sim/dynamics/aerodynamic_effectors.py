"""Stage-consistent aerodynamics from realized variable-geometry device state."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping

import numpy as np

from sim.dynamics.coupled_satellite import CoupledSatelliteState, MassProperties, StageEffects
from sim.utils.quaternion import quaternion_to_dcm_bn


@dataclass(frozen=True, slots=True)
class AerodynamicSurfaceGeometry:
    actuator_id: str
    reference_area_m2: float
    drag_coefficient: float
    lift_coefficient: float
    center_of_pressure_body_m: tuple[float, float, float]
    minimum_position: float = 0.0
    maximum_position: float = 1.0
    surface_mass_kg: float = 0.0

    def __post_init__(self) -> None:
        if not self.actuator_id.strip():
            raise ValueError("aerodynamic surface actuator_id must be non-empty")
        for name, value in (
            ("reference_area_m2", self.reference_area_m2),
            ("drag_coefficient", self.drag_coefficient),
            ("surface_mass_kg", self.surface_mass_kg),
        ):
            if not isfinite(float(value)) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not isfinite(float(self.lift_coefficient)):
            raise ValueError("lift_coefficient must be finite")
        center = np.asarray(self.center_of_pressure_body_m, dtype=float)
        if center.size != 3 or not np.all(np.isfinite(center)):
            raise ValueError("center_of_pressure_body_m must contain three finite values")
        if not isfinite(float(self.minimum_position)) or not isfinite(float(self.maximum_position)):
            raise ValueError("surface position limits must be finite")
        if self.maximum_position <= self.minimum_position:
            raise ValueError("surface maximum_position must exceed minimum_position")

    def deployment_fraction(self, position: float) -> float:
        return float(
            np.clip(
                (float(position) - self.minimum_position) / (self.maximum_position - self.minimum_position),
                0.0,
                1.0,
            )
        )


@dataclass(frozen=True, slots=True)
class VariableGeometryAerodynamicsConfig:
    surfaces: tuple[AerodynamicSurfaceGeometry, ...]
    bank_actuator_id: str | None = None
    base_drag_area_m2: float = 0.0
    base_drag_coefficient: float = 2.2
    center_of_mass_body_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        ids = [surface.actuator_id for surface in self.surfaces]
        if not self.surfaces or len(ids) != len(set(ids)):
            raise ValueError("aerodynamic surfaces must be non-empty with unique actuator IDs")
        if self.base_drag_area_m2 < 0.0 or self.base_drag_coefficient < 0.0:
            raise ValueError("base aerodynamic values must be nonnegative")
        center = np.asarray(self.center_of_mass_body_m, dtype=float)
        if center.size != 3 or not np.all(np.isfinite(center)):
            raise ValueError("center_of_mass_body_m must contain three finite values")


@dataclass(frozen=True, slots=True)
class AerodynamicRealization:
    drag_area_m2: float
    lift_area_m2: float
    bank_angle_rad: float
    force_eci_n: tuple[float, float, float]
    torque_body_n_m: tuple[float, float, float]
    device_positions: tuple[tuple[str, float], ...]


class VariableGeometryAerodynamics:
    """Evaluate realized device geometry at every coupled integrator stage."""

    def __init__(self, config: VariableGeometryAerodynamicsConfig) -> None:
        self.config = config

    def evaluate(
        self,
        state: CoupledSatelliteState,
        *,
        density_kg_m3: float,
        atmosphere_velocity_eci_m_s: tuple[float, float, float] = (0.0, 0.0, 0.0),
        device_positions: Mapping[str, float],
    ) -> tuple[StageEffects, AerodynamicRealization]:
        density = float(density_kg_m3)
        if not isfinite(density) or density < 0.0:
            raise ValueError("density_kg_m3 must be finite and nonnegative")
        velocity_eci_m_s = np.asarray(state.velocity_eci_km_s, dtype=float) * 1.0e3
        relative_velocity = velocity_eci_m_s - np.asarray(atmosphere_velocity_eci_m_s, dtype=float)
        speed = float(np.linalg.norm(relative_velocity))
        device_position_items = tuple(sorted((str(key), float(value)) for key, value in device_positions.items()))
        drag_area, lift_area = self._realized_areas(device_positions)
        if speed <= 0.0 or density == 0.0:
            zero = (0.0, 0.0, 0.0)
            realization = AerodynamicRealization(
                drag_area,
                lift_area,
                self._bank_angle(device_positions),
                zero,
                zero,
                device_position_items,
            )
            return StageEffects(), realization
        velocity_hat = relative_velocity / speed
        dynamic_pressure = 0.5 * density * speed * speed
        drag_coefficient_area = self.config.base_drag_area_m2 * self.config.base_drag_coefficient
        lift_coefficient_area = 0.0
        weighted_center = np.zeros(3)
        weighted_area = 0.0
        drag_area = self.config.base_drag_area_m2
        lift_area = 0.0
        for surface in self.config.surfaces:
            deployment = surface.deployment_fraction(device_positions.get(surface.actuator_id, surface.minimum_position))
            area = surface.reference_area_m2 * deployment
            drag_area += area
            lift_area += area
            drag_coefficient_area += area * surface.drag_coefficient
            lift_coefficient_area += area * surface.lift_coefficient
            weighted_center += area * np.asarray(surface.center_of_pressure_body_m, dtype=float)
            weighted_area += area
        drag_force = -dynamic_pressure * drag_coefficient_area * velocity_hat
        bank = self._bank_angle(device_positions)
        lift_reference = _perpendicular_reference(velocity_hat, np.asarray(state.position_eci_km, dtype=float))
        lift_direction = _rotate_about_axis(lift_reference, velocity_hat, bank)
        lift_force = dynamic_pressure * lift_coefficient_area * lift_direction
        force_eci = drag_force + lift_force
        c_bn = quaternion_to_dcm_bn(state.attitude_quat_bn)
        force_body = c_bn @ force_eci
        center_of_pressure = (
            np.asarray(self.config.center_of_mass_body_m, dtype=float)
            if weighted_area <= 0.0
            else weighted_center / weighted_area
        )
        lever = center_of_pressure - np.asarray(self.config.center_of_mass_body_m, dtype=float)
        torque_body = np.cross(lever, force_body)
        realization = AerodynamicRealization(
            float(drag_area),
            float(lift_area),
            bank,
            tuple(float(value) for value in force_eci),
            tuple(float(value) for value in torque_body),
            device_position_items,
        )
        return StageEffects(force_eci_n=force_eci, torque_body_n_m=torque_body), realization

    def mass_properties(
        self,
        base_inertia_body_kg_m2: np.ndarray,
        *,
        device_positions: Mapping[str, float],
    ) -> MassProperties:
        inertia = np.asarray(base_inertia_body_kg_m2, dtype=float).reshape(3, 3).copy()
        center = np.asarray(self.config.center_of_mass_body_m, dtype=float)
        for surface in self.config.surfaces:
            deployment = surface.deployment_fraction(device_positions.get(surface.actuator_id, surface.minimum_position))
            mass = surface.surface_mass_kg
            offset = deployment * (np.asarray(surface.center_of_pressure_body_m, dtype=float) - center)
            inertia += mass * ((offset @ offset) * np.eye(3) - np.outer(offset, offset))
        return MassProperties(inertia)

    def _realized_areas(self, device_positions: Mapping[str, float]) -> tuple[float, float]:
        surface_area = sum(
            surface.reference_area_m2
            * surface.deployment_fraction(device_positions.get(surface.actuator_id, surface.minimum_position))
            for surface in self.config.surfaces
        )
        return self.config.base_drag_area_m2 + surface_area, surface_area

    def _bank_angle(self, device_positions: Mapping[str, float]) -> float:
        if self.config.bank_actuator_id is None:
            return 0.0
        return float(device_positions.get(self.config.bank_actuator_id, 0.0))


def _perpendicular_reference(axis: np.ndarray, position_eci_km: np.ndarray) -> np.ndarray:
    radial = position_eci_km / max(float(np.linalg.norm(position_eci_km)), 1.0e-15)
    projected = radial - float(radial @ axis) * axis
    norm = float(np.linalg.norm(projected))
    if norm <= 1.0e-12:
        candidate = np.array([0.0, 0.0, 1.0])
        projected = candidate - float(candidate @ axis) * axis
        norm = float(np.linalg.norm(projected))
    return projected / norm


def _rotate_about_axis(vector: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    return (
        vector * np.cos(angle)
        + np.cross(axis, vector) * np.sin(angle)
        + axis * float(axis @ vector) * (1.0 - np.cos(angle))
    )
