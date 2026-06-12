# Orbital Calculator

The Orbital Calculator is a lightweight public-core helper for quick
back-of-the-napkin orbital mechanics answers.

Run it from the repository root:

```bash
python orbital_calculator.py
```

The calculator opens an interactive terminal menu. First choose a category,
then choose the calculator inside that category. In a normal terminal, use the
arrow keys and Enter to choose. In non-interactive shells, it falls back to
numbered choices. After each result, you can calculate another value and return
to the same category menu.

For simulator-backed mission recovery after an actual OEL run, configure
`analysis.mission_recovery` in scenario YAML. That workflow compares the
simulated assessment-state orbit against the initial orbit and can write
`mission_recovery_summary` / `mission_recovery_elements` review-store tables.

## Categories

### Circular Orbits

- Circular orbit from altitude
- Circular orbit from radius
- Orbit period from altitude
- Orbit period from semi-major axis
- Circular altitude from period
- Geosynchronous orbit altitude
- Escape velocity from altitude

### Elliptical Orbits

- Vis-viva velocity
- Apogee/perigee from semi-major axis and eccentricity
- Semi-major axis/eccentricity from apogee/perigee altitudes
- Velocity at perigee and apogee

### State / Elements Conversion

- RV to robust element report
- Classical COE to RV
- Circular inclined elements to RV
- Equatorial elliptical elements to RV
- Circular equatorial elements to RV

### Transfers And Delta-V

- Hohmann transfer between circular orbits
- Hohmann rendezvous phase angle
- Hohmann rendezvous wait time
- Plane change delta-v
- Inclination change cost from circular altitude
- Combined speed and plane change delta-v
- Mission recovery from in-track impulse

### Sun-Synchronous

- Sun-synchronous inclination from altitude
- J2 secular rates from altitude
- J2 secular rates from semi-major axis

### Phasing

- Phasing drift from altitude change

### Relative Motion / HCW

- HCW natural motion from altitude
- HCW in-track drift estimate

### Eclipse

- Circular-orbit eclipse estimate

### Ground Track

- Ground-track drift from altitude
- Repeat ground-track approximation

### Entry / Reentry

- Entry interface from apogee/perigee

### Atmospheric Drag

- Ballistic coefficient
- Density estimate from altitude
- Drag force and acceleration from altitude
- Drag force and acceleration from density/speed
- Circular-orbit drag decay rate estimate
- Deorbit lifetime range estimate

### Rocket Equation

- Rocket equation delta-v from mass ratio
- Rocket equation mass ratio from delta-v

## Assumptions

All calculations use the public simulator's Earth constants from
`sim.dynamics.orbit.environment`.

Most two-body calculators assume:

- two-body Earth gravity,
- spherical Earth reference radius for altitude calculations,
- impulsive burns for Hohmann transfers and plane changes,
- no drag, J2, SRP, third-body gravity, finite-burn effects, or operational
  targeting constraints.

The sun-synchronous calculator is the exception: it uses a first-order J2 nodal
precession estimate for a circular orbit. It is intended for early intuition,
not mission design.

The J2 secular-rate calculators expose the same first-order perturbation
intuition more directly: RAAN precession, argument of perigee precession, and
the J2 correction to mean anomaly rate. They assume a mean Earth orbit with no
drag, SRP, third-body gravity, or resonance effects.

The Hohmann rendezvous phasing calculators assume coplanar circular two-body
orbits. Phase angle is reported as the target's angle ahead of the chaser in
the prograde direction at transfer departure. The wait-time helper advances
that phase angle using the circular mean-motion difference only.

The mission-recovery calculator assumes an initially circular orbit and an
instantaneous in-track impulse. Positive disturbance delta-v is `+I`/prograde;
negative disturbance delta-v is `-I`/retrograde. It reports:

- the disturbed phasing orbit created by the impulse,
- the equal-and-opposite same-apsis recovery burn needed to restore the
  original circular orbit shape,
- ideal propellant use from the rocket equation,
- a continuous mean-motion slot-lap estimate, and
- the first discrete same-apsis recovery opportunity that returns to the
  original slot within a user-provided angular tolerance.

This is a mission-recovery intuition check, not an operational maneuver plan.
It does not model detection latency, finite burns, thrust pointing limits,
covariance, drag, J2, conjunction constraints, maneuver windows, or command and
control delays.

The phasing calculator reports the approximate along-track drift from changing
the circular-orbit altitude. Positive drift means the phasing orbit moves ahead
of the reference orbit in the two-body mean-motion estimate.

The relative-motion calculators use linear HCW/Clohessy-Wiltshire equations
about a circular chief orbit. They are intended for small relative states and
short intuition checks, not nonlinear rendezvous planning.

The eclipse calculator uses a cylindrical Earth shadow and a fixed beta angle.
It does not model penumbra, solar angular radius, seasonal Sun geometry, or
attitude/power constraints.

The ground-track calculators assume a circular inertial orbit over a spherical
rotating Earth. They ignore inclination effects, J2 nodal regression,
eccentricity, drag, and stationkeeping.

The entry/reentry calculator reports vacuum two-body speed and flight-path
angle at a user-provided interface altitude. It is not an entry heating,
deceleration, skip-out, loads, or survivability estimate.

The atmospheric drag calculators use the public USSA-1976 density approximation
already available in the simulator. The circular-orbit decay-rate calculator is
a local-density estimate:

```text
da/dt ~= -rho / B * sqrt(mu * a)
```

where `B = m / (Cd A)`. It is useful for order-of-magnitude intuition, but it
is not an orbit-lifetime prediction. Real lifetime estimates depend on solar
activity, geomagnetic conditions, attitude history, area changes, eccentricity,
and propagation over changing density.

The deorbit lifetime range estimate integrates that same circular decay-rate
approximation from the initial altitude down to a user-provided deorbit
altitude. It reports low-drag, nominal, and high-drag cases using `0.3x`,
`1.0x`, and `3.0x` density scale factors.

The built-in USSA-1976 density table stops at `1000 km`. Density lookup above
that altitude reports zero density with a warning. Drag decay-rate and deorbit
lifetime range estimates reject initial altitudes above `1000 km` instead of
returning a misleading infinite lifetime.

The rocket equation calculators use the ideal Tsiolkovsky equation with
standard gravity `9.80665 m/s^2`.

The state/elements conversion calculators assume Earth-centered inertial
position and velocity vectors. The RV-to-elements report computes the geometry
directly from the state vector and marks singular classical angles as
undefined. When a classical angle set is invalid, it reports the appropriate
alternate angle instead:

- circular inclined orbits use argument of latitude,
- equatorial elliptical orbits use longitude of perigee,
- circular equatorial orbits use true longitude.

For higher-fidelity propagation, use `run_simulation.py` with a scenario YAML
configuration.

## Importable API

The same formulas are available from Python:

```python
from sim.orbital_calculator import circular_orbit_from_altitude, mission_recovery_from_intrack_impulse

result = circular_orbit_from_altitude(400.0)
print(result.velocity_km_s)
print(result.period_min)

recovery = mission_recovery_from_intrack_impulse(
    reference_altitude_km=400.0,
    disturbance_delta_v_m_s=-5.0,
    spacecraft_mass_kg=100.0,
    isp_s=220.0,
)
print(recovery.recovery_delta_v_m_s)
print(recovery.recovery_propellant_kg)
print(recovery.slot_recovery_time_hr)
```

## Example

```text
Orbital Calculator

What do you want to calculate?

1. Circular Orbits
2. Elliptical Orbits
...

Choose an option number, or q to quit: 1

Circular Orbits

What do you want to calculate?

1. Circular orbit from altitude
2. Circular orbit from radius
...

Choose an option number, or q to go back: 1

Circular orbit from altitude
Altitude above Earth [km]: 400

Results
Altitude                   400.000 km
Orbit radius               6,778.137 km
Circular velocity          7.669 km/s
Orbital period             92.56 min
Mean motion                0.001131 rad/s
Escape velocity at radius  10.845 km/s

Assumptions: Earth two-body gravity, spherical reference radius, no drag/J2/SRP.

Calculate another? [Y/n]: n
Goodbye.
```
