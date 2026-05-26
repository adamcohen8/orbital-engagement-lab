# Controller Naming Conventions

## Purpose

Controller names are product-facing language. They should tell users what a
controller does, what frame or domain it acts in, and how it should be compared
against other controllers. They should not over-emphasize an internal equation,
reference model, or helper method unless that assumption is genuinely part of
the user-facing controller identity.

The `RIC_PD` 10 km RPO controller is the reference example: it uses HCW
relative-motion math internally, but the product-facing controller is a
rectangular-RIC PD transfer controller. Calling it `HCW_PD` would confuse the
frame/control-law identity with an implementation detail.

## Naming Principle

Use names in this order of importance:

1. Product domain or frame: `ric`, `rmoe`, `attitude`, `rocket`, `reentry`.
2. Control law, strategy, or method: `pd`, `lqr`, `if_then`, `mpc`,
   `guided_transfer`.
3. Mission role or scenario family when needed: `transfer`, `rendezvous`,
   `hold`, `survival`, `ascent`, `descent`.
4. Scale, benchmark, or variant when needed: `10km`, `critical`, `robust`,
   `pso`, `mc`.

Implementation math belongs in docs, comments, telemetry fields, and validation
notes unless it is the actual product-facing method. Examples:

- Good: `RIC_PD` for a controller that commands in rectangular RIC coordinates
  using PD feedback, even if it computes an HCW coast arc internally.
- Good: `HCW_LQR` when the controller identity is explicitly an LQR designed
  on HCW dynamics.
- Avoid: `HCW_PD` when HCW is only a helper model and the controller users tune
  and reason about is a RIC-frame PD controller.

## Naming Shapes

Python module paths use lowercase snake case:

```text
sim.control.orbit.ric_pd
sim.control.orbit.rmoe_if_then
sim.control.attitude.reaction_wheel_pd
```

Python classes use domain/method/purpose CamelCase:

```text
RICPDTransferController
RMOEIfThenController
ReactionWheelPDController
RocketTVCTrackingController
```

Controller `mode` strings use lowercase snake case and should align with the
product-facing identity:

```text
ric_pd_transfer
rmoe_if_then
reaction_wheel_pd
rocket_tvc_tracking
```

Scenario IDs, config filenames, output directories, and report/doc slugs should
share the same product-facing stem:

```text
configs/ric_pd_10km_experiment.yaml
configs/ric_pd_10km_experiment_mc.yaml
outputs/flagship_ric_pd_10km
docs/flagship-ric-pd-10km.md
```

Display names may preserve familiar uppercase acronyms:

```text
RIC_PD 10 km RPO
HCW LQR Rendezvous
RMOE If-Then Controller
```

## When Adding A Controller

Before adding or renaming a controller:

1. Choose the product-facing stem before creating files.
2. Confirm whether frame/domain, control law, mission role, and variant are all
   represented clearly.
3. Keep module name, class name, `mode`, scenario IDs, config filenames, output
   directories, docs, tests, and game/training labels aligned.
4. Use implementation-model names in helper functions and explanatory docs
   where they aid trust, but avoid making them the top-level controller brand
   unless users would naturally select or compare the controller by that model.
5. Add a focused regression test that asserts the controller `mode` and at least
   one config path or game/training reference uses the canonical name.

## Renaming Checklist

For controller renames, search and update:

- `sim/control/**`
- `configs/**`
- `sim/game/configs/**`
- `sim/tests/**`
- `README.md`
- `docs/**`
- `examples/**`
- `tools/**`
- generated-output references that appear in maintained docs

Leave historical changelog entries alone when they describe what was true at
the time, but prefer an explicit note if the older name could mislead current
users.
