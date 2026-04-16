# Presets

This folder contains reusable baseline hardware presets for the simulation framework.

Scenario YAML should prefer object preset YAML files under `objects/`. The Python
preset modules remain available for lower-level builders and compatibility.

## Files

- `objects/basic_satellite.yaml`
  - baseline satellite bus specs
  - bottom `-Z` panel chemical thruster
  - reaction-wheel attitude system name
- `objects/basic_two_stage_rocket.yaml`
  - baseline two-stage launch stack specs
- `rockets.py`
  - `BASIC_SSTO_ROCKET`
  - `BASIC_1ST_STAGE`
  - `BASIC_2ND_STAGE`
  - `BASIC_TWO_STAGE_STACK`
- `satellites.py`
  - `BASIC_SATELLITE`
- `thrusters.py`
  - `BASIC_CHEMICAL_BOTTOM_Z`
  - bottom `-Z` panel mount, centerline-aligned with CG (`x=y=0`)
- `attitude_control.py`
  - `BASIC_REACTION_WHEEL_TRIAD`
  - one wheel on each principal axis (`+X`, `+Y`, `+Z`)
- `simulation.py`
  - `build_sim_object_from_presets(...)`: one-call builder that maps satellite + thruster + attitude presets into a ready `SimObject`
  - `build_rocket_vehicle_from_presets(...)`: unified rocket vehicle object from SSTO or staged presets

## Quick Usage

In scenario YAML, point an agent at a preset file and override only the fields
that differ from the baseline:

```yaml
chaser:
  enabled: true
  preset: "../sim/presets/objects/basic_satellite.yaml"
  specs:
    dry_mass_kg: 180.0
    fuel_mass_kg: 25.0
```

Preset paths are resolved relative to the scenario YAML file first. They can
also be absolute paths, repository-relative paths, or names in
`sim/presets/objects` such as `basic_satellite`.

Python builders still work for direct object construction:

```python
from sim.presets import build_sim_object_from_presets

sat = build_sim_object_from_presets(
    object_id="sat_01",
    dt_s=2.0,
    orbit_radius_km=6778.0,
)
```

Enable attitude knowledge estimation in the same builder:

```python
sat = build_sim_object_from_presets(
    object_id="sat_01",
    dt_s=2.0,
    enable_attitude_knowledge=True,
)
```
