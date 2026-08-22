# Directed Link Runtime Monitor Contract

Status: **bounded causal adapter implemented v0.2; engine integration pending**.

Contract identifier: `oel.directed-link-runtime-monitor.v0.2`.

The runtime monitor is authorized for exactly one directed link and one named
consumer. It evaluates only at a declared positive task period after the caller
has committed the required state. Its evaluator must return a typed boolean
physical availability, finite margin, canonical Directed Link Analysis reason,
source-evidence hash, and the frozen link-configuration hash. The monitor
rejects a different configuration hash or an availability/reason mismatch
before it queues an event.

An evaluation made at task boundary `t` becomes eligible for delivery only at
`t + task_period_s`. This prevents a same-boundary result from feeding back
into the state used to calculate it. Delivery to any consumer other than the
declared authorized consumer fails closed. Skipped or off-boundary evaluation
does not silently catch up or interpolate.

The adapter does not itself propagate state, infer attitude, compute an RF
budget, schedule communications, or mutate mission logic. The caller must use
the authoritative Directed Link Analysis equation path. Integration with a
specific ONP mission or flight-software consumer requires a separate scenario
adapter and acceptance test; global Earth coverage remains prohibited from
the runtime loop.
