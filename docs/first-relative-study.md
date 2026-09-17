# Your first relative-motion study

**Question:** does a satellite remain 500 metres radially away from a second
satellite when neither commands thrust?

The supplied study propagates two satellites for ten minutes with ONP two-body
dynamics. It uses exact simulated states, no sensing or estimation, no attitude
model, no perturbations and no control. The radial offset is defined in the
chief's rotating RIC frame; zero initial relative velocity in that frame is
not the same as identical inertial velocities.

## Run and inspect

Follow [installation](installation.md), then create a managed workspace:

```text
oel workspace init path/to/relative-study
```

From that workspace, validate the supplied config, then execute it:

```text
oel sim --config configs/acceptance_relative_coast.yaml --validate-only
oel sim --config configs/acceptance_relative_coast.yaml
```

A source-checkout user can run the same config with
`python run_simulation.py --config ...`. Neither route requires Pro.

Inspect the result:

```text
oel review outputs/acceptance_relative_coast --saved-query run_metadata
oel review outputs/acceptance_relative_coast --query "SELECT time_s, range_km FROM relative_state ORDER BY time_s" --json
```

You should see 121 samples from 0 to 600 seconds, an initial range of 0.5 km,
and a changing range despite no commanded thrust. Compare initial and final
range before drawing conclusions. The changing motion results from the
specified initial relative state and gravity; it is not simulated sensor error.

For an optional figure, use the OEL review plot workflow in
[custom plots](agent-custom-plots.md), selecting `time_s` and `range_km` from
this completed run. Keep the database and source config with any shared figure.

## Explain the result before changing the model

State the initial offset, time horizon, two-body assumption and whether the
range stayed fixed. A successful run means the requested simulation completed;
it does not mean a rendezvous succeeded or prove operational accuracy.

To change the offset or duration, copy the config to a new name, change both
`scenario_name` and `outputs.output_dir`, validate, then run. Do not overwrite
the original output or turn on unrelated fidelity settings. A subsequent
sensor-limited or controlled study is a different question; read
[flight-software observations](flight-software-observations.md) first.
