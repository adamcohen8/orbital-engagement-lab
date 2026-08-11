# Flight-Software Use-Case Profiles

OEL's use-case profiles are versioned, prebuilt selections layered over the
complete GNC v2 stack boundary. A profile selects one existing complete stack,
a physical hardware family, an onboard task cadence, and a coherent minimum
set of defaults for a recognizable spacecraft application. Mission-specific
targets remain explicit inputs.

Profiles began as productization groundwork rather than qualification claims.
Maturity attaches to the exact profile version and its declared envelope, not
to every profile that uses the same stack. The full product catalog declares 18 exact profiles Supported only when its
matching qualification packets are current. Those private qualification packets
are not part of the public export, so public discovery conservatively reports the
profiles' effective maturity as **Experimental**.

## Selecting a profile

```yaml
objects:
  chaser:
    flight_software:
      profile: fsw.profile.rpo_formation_hold.v1
      params:
        reference_object_id: target
        target_relative_state_ric_m: [0.0, 500.0, 0.0, 0.0, 0.0, 0.0]
```

Validation resolves this to the underlying stack, declared hardware, cadence,
and default parameters. Explicit `params` override profile defaults. A caller
may select another hardware profile or cadence only when the profile declares
that hardware family compatible. Normalized configuration retains both the
profile identity and resolved stack identity so review evidence can distinguish
the use-case product from its implementation engine.

Use the maintained demo as a starting point:

```bash
.venv/bin/python run_simulation.py --config configs/fsw_profile_catalog_demo.yaml --validate-only
.venv/bin/python run_simulation.py --config configs/fsw_profile_catalog_demo.yaml
```

## Initial catalog

| Domain | Profile | Maturity | Groundwork provided |
| --- | --- | --- | --- |
| Baseline | `fsw.profile.coast_monitor.v1` | Experimental* | Passive typed-boundary, anomaly-monitoring, cadence, replay, and review evidence |
| Attitude | `fsw.profile.adcs_commissioning.v1` | Experimental* | Detumble, coarse-Sun recovery, Sun pointing, momentum unloading, and stack-owned FDIR |
| Attitude | `fsw.profile.adcs_nadir_payload.v1` | Experimental* | Nadir pointing with physical reaction wheels and magnetic momentum unloading |
| Attitude | `fsw.profile.adcs_sun_pointing.v1` | Experimental* | Sun-vector acquisition, loss handling, reacquisition, and physical allocation |
| Attitude | `fsw.profile.adcs_target_tracking.v1` | Experimental* | Fixed or exactly-once scheduled inertial-target tracking with loss/reacquisition |
| Orbit | `fsw.profile.orbit_maneuver_execution.v1` | Experimental* | Supplied finite-burn schedule execution |
| Orbit | `fsw.profile.leo_stationkeeping.v1` | Experimental* | Absolute ECI state tracking |
| Orbit | `fsw.profile.orbital_element_maintenance.v1` | Experimental* | Selected classical-element regulation |
| Orbit | `fsw.profile.atmospheric_pass_recovery.v1` | Experimental* | State-driven pass detection and recovery thrust |
| RPO | `fsw.profile.rpo_far_field_rendezvous.v1` | Experimental* | Planned RIC transfer, coast, correction, braking, and cleanup |
| RPO | `fsw.profile.rpo_formation_hold.v1` | Experimental* | Nonzero relative-state hold and formation groundwork |
| RPO | `fsw.profile.rpo_corridor_approach.v1` | Experimental* | V-bar approach and terminal slowdown |
| RPO | `fsw.profile.rpo_waypoint_inspection.v1` | Experimental* | Supplied RIC waypoint inspection path |
| RPO | `fsw.profile.rpo_terminal_proximity.v1` | Experimental* | Terminal closing-rate braking and settling |
| RPO | `fsw.profile.rpo_passive_retreat.v1` | Experimental* | Outward drift acquisition followed by coast |
| RPO | `fsw.profile.rpo_conjunction_response.v1` | Experimental* | Local relative keep-out response and nominal resumption groundwork |
| Low thrust | `fsw.profile.low_thrust_phasing.v1` | Experimental* | Windowed continuous-thrust relative phasing with resource gating |
| Low thrust | `fsw.profile.low_thrust_element_maintenance.v1` | Experimental* | Averaged continuous-thrust semimajor-axis and eccentricity regulation |

The catalog deliberately does not claim complete docking, operational
conjunction assessment, distributed formation consensus, flexible-body
pointing, long-arc low-thrust optimization, or certified safe-mode behavior.
Those require additional stack behavior and evidence rather than a new label.

### Catalog qualification coast-monitor envelope

`fsw.profile.coast_monitor.v1` is supported for deterministic simulation with
`hardware.passive.v1` and task periods from 0.1 through 10 seconds. It samples
typed onboard measurements, emits one diagnostic record per invocation,
records missing batches and duplicate, out-of-order, stale, suspect, invalid,
and unexpected-frame measurements, and never issues actuator commands. Its
qualification covers a six-hour two-body coast, exact snapshot/replay,
three-cadence Controller Bench conformance, and a nine-run seeded orbit/cadence
campaign. The claim does not include safety action, autonomous recovery,
sensor-data repair, hardware real-time behavior, or flight certification.

### Catalog qualification ADCS envelopes

The four exact ADCS profiles use the same rigid-spacecraft qualification
family: diagonal center-of-mass inertia of 12, 10, and 8 kg·m², three
orthogonal reaction wheels, a momentum-dump magnetorquer, 0.05-second attitude
propagation, and onboard task periods from 0.05 through 0.2 seconds. Each
profile has its own physical outcome and three-cadence Controller Bench gate;
the shared seeded campaign varies initial body rate and task cadence while
retaining exact profile identity in the evidence.

Commissioning qualifies configurable detumble hysteresis, coarse-Sun
recovery, nominal Sun pointing, wheel momentum management, and actuator
fallback. Nadir, Sun, and target profiles qualify their own moving or supplied
reference outcomes rather than inheriting commissioning maturity. Sun or
target unavailability enters a rate-damped degraded mode; clearing the typed
fault indication reacquires the reference. Target tracking also accepts
exactly-once scheduled `stack_command` inputs with `operation:
set_target_eci` and SI `target_x_eci_m`, `target_y_eci_m`, and
`target_z_eci_m` fields. Pending commands and deduplication state survive
snapshot/restart.

These claims are simulation-only. Flexible-body motion, payload jitter,
optical blinding, line-of-sight keep-outs, autonomous target acquisition,
target-motion prediction, and flight certification remain excluded.
Electrical energy or power-positive behavior is not claimed without a power
model; qualification instead bounds physical torque realization, saturation,
integrated control effort, wheel momentum, attitude error, and body rate.

### Catalog qualification orbit-operations and RPO envelopes

The four orbit profiles share a maintained 600-second near-Earth J2-plus-drag scenario,
three seeded bounded variations, and onboard task periods from 0.25 through
1.0 seconds. Scheduled maneuver execution proves delivered impulse plus typed
receipt and physical realization, and terminal completion is withheld while
any burn command receipt remains pending or rejected. State stationkeeping and orbital-element
maintenance prove that propagated truth moves toward the supplied reference.
Atmospheric pass recovery uses onboard altitude thresholds and restart-safe
phase state; thrust is inhibited until an observed pass exit and ends after
the requested recovery delta-v. Drag-makeup policy, mean-element targeting,
general maneuver optimization, and atmospheric trajectory optimization remain
outside these exact claims.

The seven RPO profiles share a near-Earth J2 family scenario, bounded seeded
variations, and task periods from 0.1 through 0.5 seconds. Qualification spans
radial, in-track, and cross-track offsets, signed relative velocity, physical
command realization, navigation loss, actuator faults, restart, saturation,
and initially violated retreat and conjunction states. The flagship transfer
provides the deeper far-field acquire/coast/correct/brake/cleanup regression.
Surface-coverage optimization, docking/contact dynamics, distributed
formation coordination, mission-specific keep-out geometry, and operational
conjunction assessment remain excluded.

### Catalog qualification low-thrust envelopes

The two exact low-thrust profiles are supported for deterministic simulation
inside the checked-in six-hour native ONP near-Earth envelope with J2,
exponential-atmosphere drag, ten-second orbit substeps, physical continuous
engines, and task periods from 1 through 20 seconds. The maintained scenario
uses 600-second thrust windows with 300-second active arcs. Exact command
receipts, physical force realization, propellant consumption, thrust duty
cycle, transition count, minimum dwell, gimbal feasibility, and missed-window
telemetry are machine checked.

The phasing case reduces range from 5.000 km to 1.568 km while keeping the
maximum excursion below 5.001 km, final in-track drift below 0.5 m/s, delta-v
below 0.25 m/s, and propellant use below 0.002 kg. Element maintenance uses a
configurable onboard averaging window and must finish closer to its 7000 km
semimajor-axis target than a passive comparison vehicle while containing
eccentricity, delta-v, and propellant. A three-seed campaign varies mass,
initial phase geometry, thrust-window phase, and element averaging, and the
Controller Bench independently covers 1-, 5-, and 20-second task cadences.

Typed resource measurements may inhibit thrust and the stack resumes after
power becomes available. OEL does not automatically derive availability from
eclipse, battery, thermal, or power-system models in this claim; those inputs
remain spacecraft-stack responsibilities. General low-thrust trajectory
optimization, arbitrary orbit raising, mission-specific thermal scheduling,
and flight qualification remain excluded.

## Discovery

```bash
.venv/bin/python -m sim.flight_software list
.venv/bin/python -m sim.flight_software list --kind profile --domain rpo
.venv/bin/python -m sim.flight_software show fsw.profile.rpo_far_field_rendezvous.v1 --json
.venv/bin/python -m sim.flight_software validate
```

The `materialize` command produces a normalized flight-software block when all
mission-specific parameters are supplied:

```bash
.venv/bin/python -m sim.flight_software materialize \
  fsw.profile.rpo_far_field_rendezvous.v1 \
  --params-json '{"reference_object_id":"target"}' \
  --json
```

## Evidence boundary

The public export includes profile selection, validation, materialization, and
runtime behavior, but it does not include the private exact-profile qualification
specifications or packets. Accordingly, `list` and `show` distinguish declared
catalog maturity from effective public maturity and report qualification evidence
as unavailable. The `status` and `qualify` commands are intentionally absent.

`Experimental*` in the table therefore means that the implementation is available
for deterministic simulation, while a Supported evidence claim cannot be audited
from this distribution alone.
