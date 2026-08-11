# Relative Motion Models

This page summarizes the relative-motion conventions and controller-facing
models implemented in OEL. It is a code-reference page, not an independent
flight-dynamics claim. Use the cited implementation files and scenario
validation artifacts as the source of truth for a specific run.

## Implemented Frames

OEL uses a target-centered RIC frame for relative-orbit setup, control, review
tables, plots, and RPO trainer scoring.

- `R` / radial: along the chief position vector.
- `C` / cross-track: along chief angular momentum, `r_chief x v_chief`.
- `I` / in-track: completes the right-handed triad as `C x R`.

The direction cosine matrix returned by `ric_dcm_ir_from_rv` has columns
`[R_hat, I_hat, C_hat]` and maps RIC components into ECI components. See
`sim/utils/frames.py`.

Unless a field says otherwise, relative states are ordered:

```text
[R, I, C, dR, dI, dC]
```

Position units are kilometers, velocity units are kilometers per second,
accelerations are kilometers per second squared, and times are seconds.

## Rectangular RIC

Rectangular RIC treats the deputy position offset as a Cartesian vector in the
instantaneous chief RIC basis. OEL computes rectangular RIC state from ECI state
with:

```text
dr_eci = r_deputy - r_chief
dv_eci = v_deputy - v_chief
dr_ric = C_ir.T @ dr_eci
dv_ric = C_ir.T @ (dv_eci - omega_ric_eci x dr_eci)
omega_ric_eci = (r_chief x v_chief) / |r_chief|^2
```

The inverse path, `ric_rect_state_to_eci`, adds the rotating-frame velocity term
back before adding the chief ECI state. These transforms are implemented in
`sim/utils/frames.py` and are covered by relative-MPC round-trip tests in
`sim/tests/test_orbit_relative_mpc.py`.

The review store `relative_state` table is rectangular RIC. It stores
`r_radial_km`, `i_intrack_km`, `c_crosstrack_km`,
`v_radial_km_s`, `v_intrack_km_s`, `v_crosstrack_km_s`,
`range_km`, and `range_rate_km_s` for the configured/default deputy-chief
pair. See `sim/reporting/review_store.py` and `docs/review-store.md`.

## Curvilinear RIC

OEL also supports a curvilinear RIC state representation for initialization,
controller input, plotting, and Monte Carlo parameter reporting. The conversion
implemented in `ric_curv_to_rect` uses the chief radius magnitude `r0` and
interprets curvilinear in-track/cross-track position as arc-length-like angles:

```text
r = r0 + R_curv
theta_i = I_curv / r0
theta_c = C_curv / r0

R_rect = r cos(theta_c) cos(theta_i) - r0
I_rect = r cos(theta_c) sin(theta_i)
C_rect = r sin(theta_c)
```

Velocity conversion differentiates those expressions with
`dtheta_i = dI_curv / r0` and `dtheta_c = dC_curv / r0`. The reverse
`ric_rect_to_curv` computes `R_curv = |r0 + dr_rect| - r0`,
`I_curv = r0 atan2(I_rect, r0 + R_rect)`, and a cross-track angle from
`atan2(C_rect, sqrt((r0 + R_rect)^2 + I_rect^2))`.

This is a coordinate transform used by controllers and plots; OEL does not
claim that curvilinear RIC itself is a separate high-fidelity dynamics model.

## Shared HCW And SS-J2 Equations

The HCW-family controllers and transfer utilities use the rectangular RIC
linearized circular-chief model:

```text
R_dot = dR
I_dot = dI
C_dot = dC
dR_dot = 3 n^2 R + 2 n dI + a_R
dI_dot = -2 n dR + a_I
dC_dot = -n^2 C + a_C
```

`n` is `mean_motion_rad_s`. `a_R`, `a_I`, and `a_C` are commanded RIC
accelerations. The same shared dynamics object also supplies homogeneous,
chief-centered Schweighart-Sedwick averaged-J2 coefficients when
`dynamics_model: ss_j2` is selected. SS-J2 requires mean reference radius and
inclination, assumes a near-circular Earth chief, and uses the coplanar nodal-
drift cross-track limit. The periodic terms written for an unperturbed
reference orbit are excluded from OEL's propagated-chief RIC state.

`HCWLQRController` obtains continuous `A` and `B` from that shared object,
computes an exact matrix-exponential zero-order hold using `design_dt_s`, and
solves a discrete LQR. `HCWNoRadialLQRController` uses the same matrices but
only the in-track and cross-track columns of `B`.

`sim/control/orbit/hcw_transfer.py` retains the closed-form HCW compatibility
functions and adds model-agnostic linear STM and rendezvous solves. It raises when the
position-velocity transition block is near singular for the requested transfer
time.

`HCWPDController` and the terminal phase of `RICPDTransferController` apply PD
feedback in rectangular RIC and optionally add the HCW feedforward cancellation
terms visible in the code. `RICPDTransferController` also uses
the shared linear rendezvous solver to choose a guided coast velocity before final
cleanup.

## Controller Assumptions

The runtime belief state passed to relative-orbit controllers is assembled as:

```text
[relative_ric_curv(6), chief_eci_state(6)]
```

from the current deputy and chief truth/estimated states. See
`_relative_orbit_state12` in `sim/runtime_support.py`. Several controllers then
convert the curvilinear state to rectangular RIC before applying their control
law:

- `HCWLQRController`: curvilinear input, rectangular HCW or opt-in SS-J2 LQR feedback,
  acceleration rotated to ECI for thrust.
- `HCWCurvInputRectOutputController`: explicit wrapper showing the same
  curvilinear-input to rectangular-output pipeline.
- `HCWNoRadialLQRController` / `HCWNoRadialManualController`: rectangular HCW
  feedback with radial acceleration forced to zero.
- `HCWPDController`: rectangular RIC PD feedback with optional HCW/SS-J2 terms when
  `mean_motion_rad_s > 0`.
- `RICPDTransferController`: rectangular RIC guided-transfer and terminal PD
  cleanup built around shared HCW/SS-J2 linear transfer math.
- `HCWRelativeOrbitMPCController`: exact-ZOH HCW or opt-in SS-J2 prediction;
  the SS coefficients are refreshed from the current chief state.
- `CurvilinearRICPDController`: feedback error is computed in curvilinear RIC;
  the commanded curvilinear acceleration is mapped to local rectangular RIC by
  a finite-difference position Jacobian before rotation to ECI.
- `RelativeOrbitMPCController`: seeds from a rectangular RIC error, then
  predicts target and chaser ECI states with two-body RK4 over the MPC horizon.

Standalone component controllers command acceleration in ECI through the
shared `Command` interface. The complete GNC v2 `fsw.rpo_reference` stack uses
`RICPDTransferController.guide_relative_state` as a subordinate guidance API,
then owns SI limiting, attitude gating, allocation, typed device commands, and
receipt/realization evidence. Mass/fuel behavior remains a physical runtime
concern outside the flight-software stack.

## Scenario Configuration Knobs

Relative initialization is configured on an object's `initial_state`.

```yaml
initial_state:
  relative_to: "target"
  relative_to_target_ric:
    frame: "rect"   # "rect" or "curv"
    state: [2.0, -8.0, 1.2, 0.0008, -0.0012, 0.0004]
```

Legacy fields `relative_ric_rect` and `relative_ric_curv` are still accepted for
target-relative initialization. `relative_to_target_ric.reference_frame` may
select target-relative or Moon/lunar-relative RIC handling in the runtime path.
Validation requires a finite length-6 state and `frame` equal to `rect` or
`curv`; see `sim/config/plugin_validation.py` and
`_resolve_chaser_relative_ric_init` in `sim/runtime_support.py`.

Common controller parameters include:

- `mean_motion_rad_s`: positive for HCW LQR, no-radial LQR, and RIC_PD
  transfer; non-negative for HCW PD.
- `max_accel_km_s2`: acceleration saturation limit. A value of zero commands
  zero acceleration in controllers that explicitly handle zero saturation.
- `design_dt_s`: discrete design/update timestep for LQR-style controllers.
- `ric_curv_state_slice` and `chief_eci_state_slice`: six-element slices into
  the controller belief vector.
- `state_signs`: length-6 sign convention adjustment before feedback.
- `q_weights`, `r_weights`, `riccati_max_iter`, `riccati_tol`: discrete LQR
  tuning and solve controls.
- `kp`, `kd`, `desired_state_ric`, `desired_state_curv`: PD target and gain
  controls, with names indicating the expected state representation.
- `transfer_time_s`, `burn_time_constant_s`, `correction_interval_s`,
  `velocity_deadband_m_s`, `final_brake_start_s`, `terminal_start_s`, and
  `terminal_range_km`: RIC_PD transfer phase controls.

Plot configuration uses `outputs.plots.reference_object_id` plus figure IDs
such as `trajectory_ric_rect_multi`, `trajectory_ric_curv_2d_multi`,
`rendezvous_summary`, `rendezvous_summary_curvilinear`, and `relative_range`.
See `docs/plotting.md` and `sim/master_outputs.py`.

## Evidence And Validation Hooks

For a queryable single run, enable review output:

```yaml
outputs:
  review:
    enabled: true
    detail: standard
```

Then validate and run the scenario:

```bash
.venv/bin/python run_simulation.py --config <path> --validate-only
.venv/bin/python run_simulation.py --config <path>
```

Useful review queries include:

```sql
SELECT time_s, r_radial_km, i_intrack_km, c_crosstrack_km, range_km
FROM relative_state
ORDER BY time_s;

SELECT time_s, range_km, range_rate_km_s
FROM relative_state
ORDER BY range_km ASC
LIMIT 1;
```

Focused regression coverage lives in:

- `sim/tests/test_orbit_hcw_lqr.py`
- `sim/tests/test_orbit_hcw_lqr_convergence.py`
- `sim/tests/test_orbit_hcw_lqr_no_radial.py`
- `sim/tests/test_orbit_hcw_lqr_curv_variant.py`
- `sim/tests/test_hcw_transfer.py`
- `sim/tests/test_ric_pd_transfer.py`
- `sim/tests/test_orbit_curv_pd.py`
- `sim/tests/test_orbit_relative_mpc.py`
- `sim/tests/test_review_store.py`

The public flagship RIC_PD scenario, validation posture, and limits are
documented in `docs/validation-ric-pd-10km.md`.

## Game And Trainer References

The downloadable RPO trainer derives relative RIC state from the same shared
frame utility in `sim/game/training.py`. The browser preview has its own
teaching/competition physics boundary: tutorial and sandbox use a lightweight
circular-reference Hill-frame model, while arcade mode uses a browser-native
two-body replay engine. It is not the canonical OEL simulator. See
`web/rpo-trainer-preview/docs/physics-contract.md`.

## Limitations

- HCW equations in OEL are the linearized circular-chief model used by the
  controller and transfer utilities. They are not a high-fidelity relative
  dynamics model for eccentric, perturbed, or large-separation regimes.
- `mean_motion_rad_s` is supplied by configuration; controllers do not infer or
  continuously retune it from the chief orbit.
- Curvilinear RIC support is a coordinate representation and conversion path.
  It should not be read as a separate validated propagation model.
- Review-store relative state currently records the primary/default pair from
  the run summary, not arbitrary all-pairs history.
- Plots and trainer displays are inspection and training aids. Scenario
  validation should rely on YAML validation, deterministic runs, review-store
  queries, tests, and documented validation packages.
- Public docs and public-core code do not make operational, flight-safety, or
  high-fidelity validation claims for arbitrary RPO use cases.
