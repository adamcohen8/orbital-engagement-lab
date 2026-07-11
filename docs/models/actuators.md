# Actuator Model Reference

Orbital Engagement Lab treats actuators as the layer between controller intent
and the force or torque integrated by the dynamics. Controllers emit the public
`Command` shape:

- `thrust_eci_km_s2`: desired translational acceleration in ECI, km/s^2.
- `torque_body_nm`: desired body torque, N*m, expressed in the body frame.
- `mode_flags`: optional state, device context, and diagnostics.

Configured satellite actuator stacks are built in `sim/runtime_support.py` from
`objects.<id>.specs.actuators` or an actuator preset. Runtime application lives
in `sim/actuators/combined.py`, `sim/actuators/orbital.py`,
`sim/actuators/attitude.py`, and `sim/actuators/faults.py`. The combined stack
applies attitude devices first, then orbital devices, so mount torques from
orbital thrusters are added after attitude-device limiting.

This page is a model reference, not a hardware qualification statement. The
implementations are deterministic engineering approximations intended for
scenario studies, controller bring-up, and validation workflows.

## Frames And Sign Conventions

- Translational commands and applied thrust histories use ECI acceleration in
  km/s^2.
- Body torques use N*m in the spacecraft body frame.
- `attitude_quat_bn` is the body-from-inertial attitude quaternion. The runtime
  converts it to `C_BN`; ECI vectors map to body vectors with `C_BN v_N`.
- Fixed thruster and gimbal direction fields are plume/nozzle directions in the
  body frame. Vehicle force is opposite the plume direction.
- A body-frame thruster at lever arm `r_B` with body-frame force `F_B` adds
  mount torque `tau_B = r_B x F_B`.
- Reaction-wheel motor torque is wheel torque. The body receives the opposite
  torque: `tau_body = -G tau_wheel_net`, where `G` is the 3xN matrix of unit
  wheel axes.

## Orbital Actuator

`OrbitalActuator` accepts a desired ECI acceleration and returns an achievable
ECI acceleration plus any mount torque. Common limiting is:

```text
a_max = min(max_accel_km_s2, max_thrust_n / mass_kg / 1000)
```

when both limits are configured. A zero or unavailable thrust cap produces zero
acceleration. The model also supports acceleration-rate limiting
`max_throttle_rate_km_s2_s`, first-order lag `lag_tau_s`, a velocity impulse
deadband `min_impulse_bit_km_s`, and mass use:

```text
T = mass_kg * ||a_eci|| * 1000
mdot = T / (Isp * g0)
delta_mass_kg = mdot * dt
```

with `g0 = 9.80665 m/s^2`. Propellant availability is enforced by the single-run
runtime after actuator application.

Relevant config keys under `specs.actuators.orbital` include
`max_accel_km_s2`, `max_thrust_n`, `min_impulse_bit_km_s`,
`max_throttle_rate_km_s2_s`, `isp_s`, `lag_tau_s`,
`thruster_direction_body`, `thruster_position_body_m`, and
`couple_to_attitude`.

## RCS Clusters

RCS devices are configured under `orbital.rcs_cluster` with `thrusters`,
`allocation_mode`, `pulse_quantum_s`, `duty_cycle`, `force_weight`, and
`torque_weight`. Each thruster has
`name`, `position_body_m`, `force_direction_body`, `max_thrust_n`,
`min_impulse_bit_n_s`, and `isp_s`.

For each thruster `i`, OEL forms one allocation column from its unit body force
axis `u_i` and mount torque axis `r_i x u_i`:

```text
A = [u_1 ... u_N]
    [r_1 x u_1 ... r_N x u_N]
```

Depending on `allocation_mode`, the target is body force, body torque, or the
stacked six-vector. The allocator uses `scipy.optimize.lsq_linear` to solve a
bounded nonnegative least-squares problem. In combined mode, force rows are
normalized by cluster force capacity and torque rows by cluster torque capacity
before the optional force/torque weights are applied:

```text
min ||A f - target|| subject to 0 <= f_i <= max_thrust_i
```

The achieved body force is rotated back to ECI for propagation. Duty cycle
scales the solved forces. `pulse_quantum_s` rounds on-time to pulse quanta, and
per-thruster `min_impulse_bit_n_s` drops pulses below the minimum impulse bit.
RCS propellant use sums `f_i / (Isp_i g0)`.

Diagnostics include `rcs_thruster_names`, `rcs_thruster_forces_n`,
`rcs_force_body_n`, `rcs_torque_body_nm`, force/torque residual vectors, and
`delta_mass_kg`.

## Electric Propulsion

Electric propulsion is configured under `orbital.electric_propulsion` with
`max_thrust_n`, `isp_s`, `duty_cycle`, optional `max_power_w`, optional
`power_per_newton_w`, and `throttle_time_constant_s`.

The effective thrust cap is:

```text
T_cap = max_thrust_n
T_cap = min(T_cap, max_power_w / power_per_newton_w)  if both power fields apply
T_cap = T_cap * clamp(duty_cycle, 0, 1)
a_cap = T_cap / mass_kg / 1000
```

The requested ECI acceleration is magnitude-clamped to `a_cap`. If
`throttle_time_constant_s > 0`, acceleration follows a first-order response with
`alpha = clamp(dt / tau, 0, 1)`. Diagnostics include
`electric_propulsion_thrust_n`, `electric_propulsion_max_thrust_n`, and
`electric_propulsion_delta_mass_kg`.

## Gimbaled Thrusters

Gimbaled spacecraft thrusters are configured under `orbital.gimbaled_thruster`
with `neutral_direction_body`, optional `position_body_m`,
`max_gimbal_angle_rad` or `max_gimbal_angle_deg`, `max_gimbal_rate_rad_s` or
`max_gimbal_rate_deg_s`, and `response_time_constant_s`.

The model converts the desired ECI acceleration direction to a body-frame force
direction, negates it to a desired plume direction, constrains that direction to
the cone about the neutral plume axis, then slews the current gimbal direction
toward the limited target. The achieved force is:

```text
F_hat_N = C_BN^T (-plume_hat_B)
a_achieved_N = ||a_cmd_N|| F_hat_N
```

The gimbal direction can also add mount torque through the ordinary
`r_B x F_B` thruster coupling. Diagnostics include `gimbal_direction_body`,
`gimbal_angle_rad`, `gimbal_rate_limited`, and `thruster_torque_body_nm`.

## Reaction Wheels

Reaction wheels are configured under `attitude.reaction_wheels` with
`max_torque_nm`, `max_momentum_nms`, optional `wheel_axes_body`, optional
`wheel_inertia_kg_m2`, optional `max_speed_rad_s`,
`torque_time_constant_s`, `viscous_friction_nms`, and
`coulomb_friction_nm`.

For three wheels, omitted axes default to body X, Y, and Z. Other wheel counts
require explicit axes in Python use. Runtime allocation maps desired body torque
to wheel motor torque with:

```text
tau_cmd = -pinv(G) tau_body_cmd
```

or uses `mode_flags["wheel_torque_cmd_nm"]` when provided. Wheel motor torque is
clamped by `max_torque_nm`, optionally first-order lagged, and reduced by
friction:

```text
tau_friction = c_viscous omega + tau_coulomb sign(omega)
tau_net = tau_motor - tau_friction
omega_next = omega + dt * tau_net / J
h_wheel = J omega
tau_body = -G tau_net
```

The model prevents commands from driving farther into momentum or speed
saturation. If inertia is not supplied, OEL infers it from
`max_momentum_nms / max_speed_rad_s` when possible, otherwise it uses a small
fallback inertia. Diagnostics include `rw_torque_cmd_nm`,
`rw_motor_torque_nm`, `rw_net_wheel_torque_nm`,
`rw_body_torque_applied_nm`, `rw_speed_rad_s`,
`rw_momentum_wheels_nms`, and `rw_momentum_body_nms`.

Reaction-wheel sign and momentum behavior have focused OEL tests and optional
Basilisk comparison coverage in `sim/tests/test_attitude_actuator_basilisk_validation.py`.

## Magnetorquers

Magnetorquers are configured under `attitude.magnetorquers` with
`max_dipole_a_m2`, either scalar or length three. The physical path requires
`mode_flags["magnetic_field_body_t"]`, or `magnetic_field_eci_t` plus current
attitude. OEL computes a least-norm dipole request for the desired torque:

```text
m_cmd = (B x tau_desired) / ||B||^2
m = clamp(m_cmd, -m_max, m_max)
tau = m x B
```

When no usable magnetic-field vector is present, the current implementation
returns zero torque and emits `magnetorquer_mode = no_b_field_zero_torque`.
Diagnostics include `magnetic_field_body_t`, `magnetorquer_dipole_cmd_a_m2`,
and `magnetorquer_torque_body_nm`.

The helper `MagnetorquerBdotController` in
`sim/control/attitude/bdot_magnetorquer.py` emits a simple B-dot-style detumble
torque perpendicular to the configured field:

```text
omega_perp = omega - b_hat (omega . b_hat)
tau_cmd = -gain * omega_perp * ||B||^2
```

capped by `max_torque_nm`.

## Control Moment Gyros

Simplified CMGs are configured under `attitude.control_moment_gyros` with
`max_torque_nm`, `momentum_nms`, `gimbal_rate_limit_rad_s`, and
`torque_time_constant_s`. OEL does not model CMG gimbal geometry, singularity
avoidance, or stored-gimbal state. It applies an axis-wise torque cap:

```text
tau_cap = min(abs(max_torque_nm), abs(momentum_nms * gimbal_rate_limit_rad_s))
tau = clamp(tau_cmd, -tau_cap, tau_cap)
```

with optional first-order response. Diagnostics include `cmg_torque_body_nm`
and `cmg_torque_cap_nm`. `CMGSteeringController` applies the same cap around a
base attitude controller.

## Wheel Desaturation

Wheel desaturation is configured under `attitude.wheel_desaturation` with
`momentum_fraction_threshold`, `unload_gain_s_inv`, and
`max_unload_torque_nm`. It requires reaction wheels. The assist computes a
body-frame external unload torque when wheel momentum exceeds the configured
fraction of the momentum-limit norm:

```text
active if ||h_body|| > threshold_fraction * ||h_max||
tau_unload = -unload_gain_s_inv * h_body
||tau_unload|| <= max_unload_torque_nm
```

Diagnostics include `wheel_desaturation_active`,
`wheel_desaturation_torque_body_nm`, and
`wheel_desaturation_momentum_norm_nms` when active. This is a simplified torque
request, not a detailed momentum-management sequence.

## Pulse Torque Thrusters

Pulse torque thrusters are configured under `attitude.thruster_pulse` with
`max_torque_nm` and `pulse_quantum_s`. The model clips each body-torque axis to
the configured limits and scales by rounded pulse duration when
`pulse_quantum_s > 0`.

## Fault And Degradation Layer

`FaultedActuator` can wrap the combined stack with:

- `stuck_off`
- `thrust_scale`
- `torque_scale`
- `thrust_bias_eci_km_s2`
- `torque_bias_body_nm`

The layer is applied after the base actuator. `stuck_off` zeros thrust and
torque. Otherwise, OEL applies scale and bias to the achieved command and emits
fault scale diagnostics.

## Presets And Config Validation

Public presets live in `sim/actuators/presets.py`:

- `BASIC_RCS_6DOF`
- `BASIC_ELECTRIC_PROPULSION`
- `BASIC_MAGNETORQUER_TRIAD`
- `BASIC_CMG_TRIAD`
- `BASIC_GIMBALED_THRUSTER`

Use them through `specs.actuator_preset` or `specs.actuators.preset`. Explicit
fields in `specs.actuators` override preset fields. Strict plugin validation in
`sim/config/plugin_validation.py` checks actuator block shape, unknown keys,
vector lengths, nonnegative limits, duty-cycle ranges, RCS thruster definitions,
and preset names before a run starts. For new or edited scenarios, validate
with:

```bash
.venv/bin/python run_simulation.py --config <path> --validate-only
```

## Logging And Evidence Hooks

Applied acceleration and torque are stored in run histories as
`applied_thrust_by_object` and `applied_torque_by_object`; artifact metadata
labels these as `applied_thrust` and `applied_torque`. Controller debug rows
carry actuator `mode_flags`, including per-device diagnostics listed above.

Single-run summaries include `summary.actuator_diagnostics_summary`. Monte Carlo
payloads aggregate actuator diagnostics under
`aggregate_stats.actuator_diagnostics_by_object`, and AI-report figure data
sources include actuator diagnostics in resource/control summaries.

Useful focused tests and smokes:

- `sim/tests/test_orbital_actuator.py`
- `sim/tests/test_attitude_actuator_devices.py`
- `sim/tests/test_actuator_runtime_integration.py`
- `sim/tests/test_actuator_reporting_integration.py`
- `sim/tests/test_actuator_aware_controllers.py`
- `sim/tests/test_attitude_actuator_basilisk_validation.py`
- `configs/actuator_lab_presets_smoke.yaml`
- `configs/controller_rcs_allocation_smoke.yaml`
- `configs/controller_electric_propulsion_smoke.yaml`
- `configs/controller_gimbaled_thruster_smoke.yaml`
- `configs/controller_magnetorquer_bdot_smoke.yaml`
- `configs/controller_wheel_desaturation_smoke.yaml`
- `configs/controller_cmg_steering_smoke.yaml`

Validation harness routing is documented in the private Validation Operations
guide.
Actuator behavior changes normally map to `--suite orbit_actuator_coupling` for
orbital/mount-force behavior and `--suite actuator_coupling` for attitude
actuator behavior.

## Limitations

- Actuator models are deterministic approximations, not vendor hardware models.
- RCS allocation is bounded nonnegative least squares with simple pulse and duty
  approximations; it does not model valve transients, plume interaction,
  thermal limits, or detailed minimum-on/off timing.
- Electric propulsion uses simple thrust, power, duty-cycle, throttle-lag, and
  ideal propellant-use equations; it does not model power-system dynamics or
  thruster efficiency curves.
- Gimbaled thrusters use cone/rate limiting only; they do not model flexible
  mounts, thrust-vector-control actuators, or misalignment calibration.
- Reaction wheels model axis allocation, torque/momentum/speed limits, simple
  lag, and simple friction; they do not model imbalance, jitter spectra,
  bearing thermal behavior, or detailed electronics.
- Magnetorquers require magnetic-field evidence in `mode_flags`; without it
  they apply zero torque.
- CMGs are axis-wise torque caps, not full CMG steering-law or singularity
  models.
- Wheel desaturation is an external torque assist based on wheel momentum, not
  a complete unload strategy tied to environmental torques or thruster firing
  plans.
- YAML validation currently focuses on the public scalar/three-axis actuator
  shapes used by shipped scenarios and presets; lower-level Python APIs expose
  some more general shapes.
