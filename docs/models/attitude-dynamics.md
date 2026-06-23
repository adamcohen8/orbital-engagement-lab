# Attitude Dynamics

This page documents the attitude state propagation model used by OEL satellite
objects when `simulator.dynamics.attitude.enabled` is true. It is a model
reference for simulation review and scenario design, not a flight-qualification
or ADCS hardware-certification claim.

## State, Frames, And Units

OEL stores each object's attitude state in `StateTruth`:

- `attitude_quat_bn`: scalar-first unit quaternion `[q0, q1, q2, q3]`.
- `angular_rate_body_rad_s`: body angular rate in rad/s.
- `position_eci_km` and `velocity_eci_km_s`: translational state in ECI.
- `mass_kg` and `t_s`: mass and simulation time.

The `bn` suffix follows the implementation convention in
`sim.utils.quaternion`: `quaternion_to_dcm_bn(q_bn)` returns the DCM that maps
inertial-frame vectors into the body frame. Disturbance torque calculations use
that DCM as:

```text
v_body = C_bn v_eci
```

Attitude torques are expressed in body axes as N*m. Inertia is a 3x3 body-frame
matrix in kg*m^2. Translational states remain in kilometers and km/s, with
disturbance models converting to SI where needed.

Quaternions are normalized on input/output. Invalid or non-finite quaternions
fall back to `[1, 0, 0, 0]` through the shared quaternion utility.

## Governing Equations

For body angular velocity `w` and body torque `tau`, OEL uses the rigid-body
Euler equation:

```text
I w_dot = tau - w x (I w)
```

The quaternion kinematics are represented by:

```text
q_dot = 0.5 Omega(w) q
```

where `Omega(w)` is implemented in `sim.utils.quaternion.omega_matrix`.

The default propagated update is the exponential-map path in
`sim.dynamics.attitude.rigid_body.propagate_attitude_exponential_map`:

1. Compute `w_dot` from the rigid-body equation.
2. Advance body rate with a first-order step:

   ```text
   w_next = w + dt w_dot
   ```

3. Build a finite quaternion delta from the midpoint body rate:

   ```text
   w_mid = w + 0.5 dt w_dot
   dq = [cos(0.5 |w_mid| dt), axis(w_mid) sin(0.5 |w_mid| dt)]
   q_next = normalize(q (x) dq)
   ```

The right multiplication is intentional: it matches the `Omega(w) @ q`
convention used by the differential form.

There is also an Euler helper,
`propagate_attitude_euler`, but the coupled orbital/attitude dynamics path uses
the exponential-map update.

## Coupled Propagation

The coupled object dynamics live in `sim.dynamics.model.OrbitalAttitudeDynamics`.
Each simulation step:

1. Propagates the translational state through the configured orbit propagator.
2. Forms a midpoint translational state for attitude-dependent environmental
   torque inputs.
3. Substeps attitude with `attitude_substep_s` when configured.
4. At each attitude substep, recomputes disturbance torque from the current
   attitude and adds it to the applied command torque:

   ```text
   tau_total_body = command.torque_body_nm + disturbance_torque_body_nm
   ```

5. Advances quaternion and body rate with the exponential-map propagator.

When attitude propagation is disabled, OEL still carries attitude fields in the
state, but the coupled propagator does not advance them.

## Disturbance Torques

Optional disturbance torques are implemented in
`sim.dynamics.attitude.disturbances.DisturbanceTorqueModel`. They are enabled
from scenario YAML under `simulator.dynamics.attitude.disturbance_torques`.

Supported terms:

- Gravity gradient:

  ```text
  tau_gg = 3 mu / |r|^3 * r_hat_body x (I r_hat_body)
  ```

- Magnetic dipole torque:

  ```text
  tau_mag = m_body x B_body
  ```

  If no magnetic field is supplied in the environment, OEL uses a simple
  centered Earth dipole aligned with inertial +Z.

- Drag torque:

  ```text
  tau_drag = r_cp_body x F_drag_body
  ```

  The force uses atmosphere-relative velocity, configured density model,
  drag coefficient, projected area or geometry profile, and center-of-pressure
  offset.

- Solar-radiation-pressure torque:

  ```text
  tau_srp = r_cp_body x F_srp_body
  ```

  The force uses the configured SRP pressure, `Cr`, sun direction, shadow
  factor, area/facet/geometry profile, and center-of-pressure offset.

Drag and SRP torque can use scalar area, a single facet, multiple facets,
rectangular-prism faces, or a precomputed geometry area profile depending on
object specs and runtime configuration. Geometry profiles are projected-area
lookups, not full ray-traced mesh force models.

## Controllers And Actuators

Attitude controllers are configured with object-level `attitude_control`
pointers. Controllers produce `Command.torque_body_nm` and optional diagnostic
mode flags. The dynamics model treats the resulting command as body torque; it
does not by itself decide controller gains or hardware authority.

Reference controller examples include:

- `sim.control.attitude.zero_torque.ZeroTorqueController`
- `sim.control.attitude.baseline.QuaternionPDController`
- `sim.control.attitude.baseline.ReactionWheelPDController`
- `sim.control.attitude.baseline.SmallAngleLQRController`
- RIC-frame PD/PID/LQR controllers under `sim.control.attitude.ric_*`

When an `AttitudeActuator` stack is configured, command torque can be shaped by
simplified actuator models before entering dynamics. Current attitude actuator
devices include reaction wheels, magnetorquers, thruster pulse quantization,
control-moment-gyro torque caps, and wheel desaturation torque. These actuator
models are engineering/simulation abstractions; they are not detailed hardware
replicas.

## Configuration Knobs

Typical attitude-enabled scenario fields:

```yaml
objects:
  bus:
    specs:
      mass_properties:
        center_of_mass_body_m: [0.0, 0.0, 0.0]
        inertia_kg_m2:
          - [12.0, 0.0, 0.0]
          - [0.0, 10.0, 0.0]
          - [0.0, 0.0, 8.0]
    initial_state:
      attitude_quat_bn: [1.0, 0.0, 0.0, 0.0]
      angular_rate_body_rad_s: [0.0, 0.0, 0.0]
    attitude_control:
      kind: python
      module: sim.control.attitude.zero_torque
      class_name: ZeroTorqueController
      params: {}

simulator:
  dynamics:
    attitude:
      enabled: true
      attitude_substep_s: 0.1
      disturbance_torques:
        gravity_gradient: true
        magnetic: false
        drag: false
        srp: false
```

Important knobs:

- `objects.<id>.initial_state.attitude_quat_bn`: initial scalar-first
  quaternion using the `q_bn` convention.
- `objects.<id>.initial_state.angular_rate_body_rad_s`: initial body rate.
- `objects.<id>.specs.mass_properties.inertia_kg_m2`: finite symmetric
  positive-definite body inertia matrix. Strict validation rejects invalid
  explicitly supplied inertia.
- `objects.<id>.specs.mass_properties.center_of_mass_body_m`: reference origin
  used by geometry/profile torque paths.
- `objects.<id>.specs.aero`: drag/SRP/lift properties such as `drag_area_m2`,
  `reference_area_m2`, `cd`, `cr`, `lift_axis_body`, and
  `cp_offset_body_m`.
- `objects.<id>.specs.geometry.profile_path`: optional projected-area profile
  used for attitude-dependent drag/SRP area and torque.
- `objects.<id>.attitude_control`: controller module/class and parameters.
- `objects.<id>.specs.actuators.attitude`: optional attitude actuator stack
  parameters, when used by a scenario.
- `simulator.dynamics.attitude.enabled`: enable/disable attitude propagation.
- `simulator.dynamics.attitude.attitude_substep_s`: attitude integration
  substep. It must be positive, no larger than `dt_s`, and divide `dt_s`
  cleanly.
- `simulator.dynamics.attitude.disturbance_torques.gravity_gradient`,
  `magnetic`, `drag`, `srp`: disturbance torque toggles.
- `simulator.acceleration.mode`: optional acceleration path for supported
  numeric kernels. The accelerated attitude kernel is intended to preserve the
  same propagated state and guardrail counts as the Python path.

## Implementation Locations

- `sim/dynamics/model.py`: coupled orbit/attitude stepping and attitude
  substep orchestration.
- `sim/dynamics/attitude/rigid_body.py`: rigid-body derivatives,
  exponential-map propagation, Euler helper, and guardrail statistics.
- `sim/dynamics/attitude/disturbances.py`: gravity-gradient, magnetic, drag,
  and SRP torque models.
- `sim/utils/quaternion.py`: quaternion normalization, multiplication,
  body-rate delta, and `C_bn` conversion.
- `sim/acceleration/kernels/attitude.py`: optional accelerated attitude
  propagation kernel.
- `sim/control/attitude/`: built-in attitude controller implementations.
- `sim/actuators/attitude.py`: simplified attitude actuator stack.
- `sim/runtime_support.py`: scenario-to-runtime wiring for attitude dynamics,
  disturbance toggles, controller cadence, mass properties, geometry profiles,
  and actuator configs.

## Validation And Evidence Hooks

Use scenario validation before running new or edited attitude scenarios:

```bash
.venv/bin/python run_simulation.py --config <path> --validate-only
```

Relevant focused tests and harnesses:

```bash
.venv/bin/python -m pytest -q sim/tests/test_attitude_expmap.py
.venv/bin/python -m pytest -q sim/tests/test_attitude_disturbances.py
.venv/bin/python -m pytest -q sim/tests/test_attitude_guardrail_stats.py
```

Private validation harnesses additionally cover propagation, guardrail,
disturbance, reference-convention, and Basilisk sweep contract checks. Basilisk
itself is optional and is not a required OEL dependency. When available, the
Basilisk reference workflow compares selected attitude, disturbance, and
reaction-wheel cases against external histories; when it is not installed, the
contract tests still exercise comparator thresholds and
adapter boundaries.

For completed runs with review output enabled, inspect attitude state evidence
from the review store:

```bash
.venv/bin/python -m sim.review outputs/<scenario_name> \
  --query attitude_state_first_last --json
```

or query `object_state` directly for `quat_w`, `quat_x`, `quat_y`, `quat_z`,
`omega_x_rad_s`, `omega_y_rad_s`, and `omega_z_rad_s`.

## Limitations

- The angular-rate update is first order. The quaternion update uses an
  exponential-map midpoint rate, but this is not a high-order rigid-body
  integrator.
- Disturbance terms are simplified engineering models. They are useful for
  scenario studies and regression evidence, but they are not mission-specific
  environmental qualification.
- Magnetic torque uses a simple centered dipole field unless the environment
  supplies a magnetic field.
- Drag/SRP geometry profiles use projected-area lookup behavior. They do not
  model self-shadowing, articulation, detailed material BRDF/specular response,
  aeroelastic effects, plume interactions, or thermal deformation.
- Controller and actuator models are simulation abstractions. Passing a
  scenario gate does not validate hardware ADCS performance, actuator sizing,
  flight software, jitter, sensor noise, flexible modes, or fault tolerance.
- Basilisk comparisons, when present, are selected reference cases. They do not
  prove broad equivalence across all attitude regimes, spacecraft geometries,
  environments, time steps, or controller schedules.
