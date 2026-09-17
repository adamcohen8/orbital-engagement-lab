# Flight-software observations and navigation evidence

A satellite's v2 stack receives typed measurements, computes navigation, and
commands devices. Its runtime adapter generates measurements from simulated
physics. These are separate responsibilities.

## Configure observations and navigation deliberately

```yaml
knowledge:
  targets: [target]
  refresh_rate_s: 2.0
  conditions:
    max_range_km: 10.0
    require_line_of_sight: true
    dropout_prob: 0.0
  sensor_error:
    pos_sigma_km: [0.005, 0.005, 0.005]
    seed: 2501
flight_software:
  stack: fsw.rpo_reference
  hardware_profile: hardware.ideal_wrench.v1
  task_period_s: 0.5
  params:
    reference_object_id: target
    navigation_initialization: cold
    navigation_filter: ekf
```

This is a block to adapt inside a complete scenario, not a standalone run.
Validate the complete YAML before execution. Use the
[scenario YAML guide](scenario-yaml.md) to place this block on a satellite and
define its reference object.

Cadence and noise apply to runtime sensor sampling. `knowledge.conditions`
applies range, body-mounted FOV/solid angle, Earth line-of-sight and dropout
restrictions before target measurements reach onboard logic. The runtime uses
the same detection implementation as the separate observer knowledge path.
Each path has its own sampling schedule and seeded random history; individual
stochastic detections are not promised to coincide. Runtime access uses a
separate random stream so adding a nonrestrictive gate does not perturb noise.
Checkpoint state retains the access random stream and cadence state.

An explicit access restriction also applies with `navigation_initialization:
ideal`. Ideal measurements describe error, not permission to see through a
configured obstruction. Omit restrictions for an intended unrestricted ideal
study. Missing target packets leave recovery/freshness decisions to the stack;
they do not authorize substitution of simulator truth.

`knowledge.estimation` selects the separate observer reporting estimator.
`flight_software.params.navigation_filter` selects onboard orbit navigation.
Setting one does not configure the other. `sample_hold` intentionally holds a
sample; `ekf` propagates its estimate between observations. Neither setting
makes the simulation operationally calibrated.

## Read the quantity you intend to assess

| Quantity | Surface | Meaning |
| --- | --- | --- |
| Physical state | API `truth`, review `object_state` | Simulated physical state at the outer sample |
| Raw observation | `fsw_input_events` | Typed sensor packet with sampling/delivery times and frame |
| Onboard estimate | API `belief`, telemetry topic `oel.navigation_state.v1` | Navigation state already used by the stack; no extra filter propagation during reporting |
| Separate observer track | Knowledge history and detection summaries | Reporting/analysis estimate; not a copy of onboard navigation |

Reference stacks publish navigation telemetry even when optional verbose
status diagnostics are disabled. Its `generated_at` clock identifies the
invocation; `state_epoch_ticks` and `state_age_s` identify a held versus
propagated orbit estimate in that same clock domain. A missing epoch means
unknown, not zero age. The record also names the inertial frame and registry
version. Raw input packets preserve their measurement epochs independently.
Use `outputs.review.detail: full` when you need the complete raw packet
payloads in `fsw_input_events.detail_json`; standard stores retain event
headers and normalized diagnostic fields without those optional JSON details.

Position/velocity telemetry uses metres and metres per second. API belief
keeps km and km/s, followed by available quaternion and angular-rate components.
A stack without published navigation has unavailable belief (NaN components with the existing vector width); the runtime
does not label its latest measurement as an estimate. Custom stacks may emit
the same topic using the documented fields and typed diagnostic envelope.

For a completed review store, inspect the typed scalar fields:

```sql
SELECT object_id, invocation_id, generated_time_ns, field_name, unit,
       value_real, value_integer, value_text
FROM fsw_diagnostic_fields
WHERE topic = 'oel.navigation_state.v1'
ORDER BY object_id, invocation_id, field_index
```

Inspect `review/schema.json` before modifying this query. The API history holds
the latest published estimate at each outer sample. If stack tasks run slower
than outer steps, read telemetry generation time as well as state age; do not
infer that the stack was invoked at every history sample.

Past runs retain their original semantics. Regenerate a run after changing
sensor or navigation settings, and never combine old and new belief histories
as though their meaning were identical.
