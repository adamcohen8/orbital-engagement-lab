from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SatellitePreset:
    name: str
    dry_mass_kg: float
    propellant_mass_kg: float
    bus_size_m: tuple[float, float, float]
    inertia_kg_m2: np.ndarray

    @property
    def wet_mass_kg(self) -> float:
        return self.dry_mass_kg + self.propellant_mass_kg


BASIC_SATELLITE = SatellitePreset(
    name="Basic Satellite",
    dry_mass_kg=260.0,
    propellant_mass_kg=40.0,
    bus_size_m=(1.2, 1.0, 1.0),
    inertia_kg_m2=np.diag([120.0, 100.0, 80.0]),
)

CUBESAT_6U = SatellitePreset(
    name="6U CubeSat",
    dry_mass_kg=12.0,
    propellant_mass_kg=0.0,
    bus_size_m=(0.3, 0.2, 0.1),
    inertia_kg_m2=np.diag([0.05, 0.10, 0.13]),
)

SMALLSAT_RPO = SatellitePreset(
    name="SmallSat RPO",
    dry_mass_kg=160.0,
    propellant_mass_kg=40.0,
    bus_size_m=(0.8, 0.7, 0.7),
    inertia_kg_m2=np.diag([20.0, 18.0, 15.0]),
)

TARGET_BUS_PASSIVE = SatellitePreset(
    name="Passive Target Bus",
    dry_mass_kg=500.0,
    propellant_mass_kg=0.0,
    bus_size_m=(2.0, 1.5, 1.5),
    inertia_kg_m2=np.diag([260.0, 220.0, 180.0]),
)

ELECTRIC_PROP_SMALLSAT = SatellitePreset(
    name="Electric Propulsion SmallSat",
    dry_mass_kg=180.0,
    propellant_mass_kg=25.0,
    bus_size_m=(1.0, 0.8, 0.8),
    inertia_kg_m2=np.diag([45.0, 38.0, 30.0]),
)

ADCS_DEMO_SAT = SatellitePreset(
    name="ADCS Demo Satellite",
    dry_mass_kg=120.0,
    propellant_mass_kg=5.0,
    bus_size_m=(0.8, 0.7, 0.6),
    inertia_kg_m2=np.diag([14.0, 12.0, 10.0]),
)
