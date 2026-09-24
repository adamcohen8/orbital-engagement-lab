# Spacecraft thermal and electrical resources

OEL ONP satellites can opt into `specs.thermal` and `specs.power` independently.
These are deterministic simulation-truth models, not received telemetry,
sensors, navigation estimates, or an operator-training interface. Disabled
models leave the mechanical stepping path unchanged.

See [`configs/spacecraft_resources_demo.yaml`](../../configs/spacecraft_resources_demo.yaml)
for a complete runnable configuration. It uses a fixed Sun at 1 AU, a fixed
inertial spacecraft attitude, no SRP acceleration, and a 6000-second LEO orbit.
The fixed Sun is an explicit educational simplification; normal scenarios may
use OEL's epoch-based solar ephemeris instead.

## Configuration

```yaml
specs:
  thermal:
    enabled: true
    initial_temperature_k: 290.0
    heat_capacity_j_k: 80000.0
    solar_absorptivity: 0.3
    infrared_emissivity: 0.8
    area_mode: constant
    projected_area_m2: 1.0
    radiating_area_m2: 4.0
    internal_heat_w: 0.0
    earth_albedo: 0.3
    earth_ir_w_m2: 237.0
    solar_irradiance_w_m2: 1361.0
    earth_quadrature_order: 8
    max_step_s: 1.0
  power:
    enabled: true
    panels:
      - area_m2: 2.0
        normal_body: [1.0, 0.0, 0.0]
        efficiency: 0.3
    baseline_load_w: 150.0
    load_heat_fraction: 1.0
    conversion_efficiency: 0.95
    battery_capacity_wh: 500.0
    initial_soc: 0.6
    max_charge_w: 200.0
    max_discharge_w: 300.0
    charge_efficiency: 0.95
    discharge_efficiency: 0.95
    max_step_s: 1.0
```

Temperatures use kelvin, areas square metres, powers watts, heat capacity J/K,
and battery capacity Wh. Fractions lie in [0,1]; conversion and storage
efficiencies must be greater than zero. Unknown model fields and nonfinite
values are errors. An omitted model is off; parameterized models must explicitly
set `enabled: true`. Battery capacity and charge/discharge limits are required,
not silently unlimited. Panel normals are normalized and expressed in the
spacecraft body frame. OEL's `attitude_quat_bn` rotates inertial vectors to body
coordinates. Disabling attitude dynamics retains the configured orientation.

For a geometry area profile already attached to the spacecraft via
`specs.geometry_profile_path` (or its documented geometry aliases), select
`thermal.area_mode: geometry` and omit `projected_area_m2`. The existing
projected-area interface supplies incident areas toward the Sun and each Earth
quadrature direction. Radiating area is still explicit: projected area is not
surface area. Geometry improves area dependence, not thermal-node resolution.
Profile visibility/self-occlusion limitations still apply.

## Thermal balance and environment

The one-node model integrates

`C dT/dt = Q_solar + Q_albedo + Q_earth_ir + Q_internal - epsilon sigma A_rad T^4`.

Direct solar irradiance scales with inverse squared spacecraft-to-Sun distance.
It reuses OEL's Sun geometry and fractional SRP shadow function even when SRP
acceleration is disabled. `simulator.environment.srp_shadow_model` therefore
also controls resource illumination; `none` explicitly disables Earth eclipse.
OEL's existing conical model approximates partial eclipse; this model does not
upgrade its penumbra fidelity.

Earth is a uniform Lambertian sphere with constant albedo and outgoing IR
exitance. Apparent-disk Gauss-Legendre/azimuth quadrature integrates source
solid angle, receiving projected area, and surface solar incidence. Albedo uses
the visible, sunlit Earth surface and is not multiplied by spacecraft eclipse.
IR persists over the Earth's night side. `earth_quadrature_order` ranges from
4 to 64 and uses twice that many azimuth samples. The defaults are engineering
assumptions, not weather-, geography-, or season-resolved Earth products.

Solar absorptivity applies to direct and reflected shortwave light. IR
emissivity also supplies IR absorptivity under a gray-surface approximation.
Emission uses total exposed radiating area and a negligible deep-space
background. The model has no atmosphere convection, reentry heating coupling,
internal conduction network, heater control, or component-specific temperatures.

Orbit and attitude advance on resource substeps bounded by the smaller enabled
`max_step_s`. Forcing is evaluated at midpoint position and sign-corrected
normalized quaternion interpolation. Temperature uses implicit backward Euler
for the T^4 emission term with a bracketed scalar solve. This positive,
first-order method trades accuracy for stability; reduce `max_step_s` and
increase Earth quadrature order to demonstrate convergence for a study.

## Electrical balance and thermal coupling

Each fixed, one-sided panel generates `S * sunlit_fraction * area *
max(0, normal dot Sun_direction) * cell_efficiency`. Generation reported at the
bus includes conversion efficiency. Panels do not articulate, shadow each
other, or receive albedo-generated electrical power in this version. Cell
efficiency is fixed, with no temperature or degradation dependence.

Generation serves the baseline load first. Surplus charges the battery within
its bus-side charge limit and capacity; unused potential generation is
curtailed. Deficits discharge storage within its bus-side discharge limit and
available energy. Remaining demand becomes `unmet_load_w`. Storage changes by
`charge_w * charge_efficiency - discharge_w / discharge_efficiency`.
At a capacity boundary, rates are interval averages under midpoint forcing.
No bus-voltage/current model, battery aging, brownout, or automatic load
shedding is implied. Operational software must explicitly decide what to do
with a shortfall; existing actuators are not silently inhibited.

The thermal node represents the body **excluding the solar-panel absorbing
surfaces**. Panels are thermally external ideal generators. Include neither
panel area nor panel absorption in the body's thermal geometry. Served load
multiplied by `load_heat_fraction`, conversion losses on harvested power, and
battery charge/discharge losses heat the body. The remaining load fraction is
exported energy. Curtailed generation is unharvested rather than dissipated in
the body. `internal_heat_w` is additional independent heat; do not repeat the
electrical load there. Panel temperature and panel-to-body heat transfer need
a future multi-node model; this is not a whole-spacecraft thermal network.

## API and evidence

- `StateTruth.resource_state` contains the optional physical state and ledger.
- `SimulationSnapshot.spacecraft_resources` exposes per-object truth snapshots.
- Completed payloads contain `spacecraft_resources` histories.
- Saved runs with models enabled write `spacecraft_resources.json` and
  `spacecraft_resources.csv`, including when full-log JSON is disabled.
- Enabled review stores include a `spacecraft_resources` table and schema
  sidecar entries, including compact review mode.

Rows use `time_s` for endpoint state. Rates and illumination describe the last
resource integration interval, identified by `interval_start_s`; they are not
averages over the dashboard or output cadence. The initial row contains state
and zero cumulative energy, with unevaluated rates absent/null. Cumulative
`*_energy_j` fields integrate every resource substep so coarse output does not
lose the energy ledger. Thermal and bus power balance residuals are retained.

Example read-only review query:

```sql
SELECT object_id, time_s, temperature_k, battery_soc,
       solar_generation_w, unmet_load_w, thermal_balance_residual_w
FROM spacecraft_resources ORDER BY object_id, time_s
```

These outputs deliberately do not inject perfect resource knowledge into
flight software or ground estimation. Sensor and downlink models can consume
them through a separately specified observation boundary.

The initial scope is Earth-centered ONP satellite propagation. OGP and CR3BP
resource configurations are rejected. Existing interchange continuation
materializers reject resource-enabled objects rather than silently resetting
temperature and battery state; a versioned resource-state handoff is future work.

## Verification and limits

Focused tests cover analytic thermal equilibrium, constant heating, Earth-disk
solid angle, day/night heating, panel orientation, bounded battery energy,
shortfalls, energy residuals, YAML validation, disabled-path parity, and API /
JSON / CSV / SQLite integration. These establish numerical and integration
behavior for the stated model, not empirical spacecraft qualification.

Physical reference: NASA's [Thermal Control](https://www.nasa.gov/smallsat-institute/sst-soa/thermal-control/)
and [Power](https://www.nasa.gov/smallsat-institute/sst-soa/power-subsystems/)
Small Spacecraft Technology State-of-the-Art chapters.
