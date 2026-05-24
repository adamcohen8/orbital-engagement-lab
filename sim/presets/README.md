# Presets

This folder contains reusable baseline hardware presets for the simulation framework.

Scenario YAML should prefer object preset YAML files under `objects/`. The Python
preset modules remain available for lower-level builders and compatibility.

## Files

- `objects/basic_satellite.yaml`
  - baseline satellite bus specs
  - bottom `-Z` panel chemical thruster
  - reaction-wheel attitude system name
- `objects/cubesat_6u.yaml`
  - small passive/education CubeSat bus
- `objects/smallsat_rpo.yaml`
  - agile RPO smallsat with the public RCS actuator preset
- `objects/target_bus_passive.yaml`
  - larger passive resident-space-object target bus
- `objects/electric_prop_smallsat.yaml`
  - low-thrust high-Isp smallsat with the public electric-propulsion preset
- `objects/adcs_demo_sat.yaml`
  - attitude-control demo bus with magnetorquer-oriented defaults
- `objects/basic_two_stage_rocket.yaml`
  - baseline two-stage launch stack specs
- `rockets.py`
  - `BASIC_SSTO_ROCKET`
  - `BASIC_1ST_STAGE`
  - `BASIC_2ND_STAGE`
  - `BASIC_TWO_STAGE_STACK`
- `satellites.py`
  - `BASIC_SATELLITE`
  - `CUBESAT_6U`
  - `SMALLSAT_RPO`
  - `TARGET_BUS_PASSIVE`
  - `ELECTRIC_PROP_SMALLSAT`
  - `ADCS_DEMO_SAT`
- `thrusters.py`
  - `BASIC_CHEMICAL_BOTTOM_Z`
  - bottom `-Z` panel mount, centerline-aligned with CG (`x=y=0`)
- `attitude_control.py`
  - `BASIC_REACTION_WHEEL_TRIAD`
  - one wheel on each principal axis (`+X`, `+Y`, `+Z`)
- `sim/actuators/presets.py`
  - public actuator-stack presets for RCS, electric propulsion,
    magnetorquers, CMGs, and gimbaled thrusters
  - use with `specs.actuator_preset` or `specs.actuators.preset`
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

Attach a public actuator preset:

```yaml
chaser:
  enabled: true
  specs:
    mass_kg: 250.0
    actuator_preset: BASIC_RCS_6DOF
```

`BASIC_RCS_6DOF` is the reusable six-axis RCS cluster preset. Its thruster
geometry is tested for independent body-frame force and torque authority along
X, Y, and Z, so it is the default choice for RPO controller bring-up that needs
coupled translation and attitude authority.

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
