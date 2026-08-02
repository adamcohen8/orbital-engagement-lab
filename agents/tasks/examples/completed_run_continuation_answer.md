# Completed-run Continuation Answer Example

Status: The source run completed, its final `iss` sample exported
successfully, and the separate ONP continuation validated. The continuation
was not executed.

Commands: I validated and ran the source fixture, used `sim.handoff
export-state --sample final`, materialized a new passive ONP scenario, and ran
`sim.handoff compare-handoff` without a continuation output directory.

Review queries: I queried `run_metadata` for duration, cadence, samples, and
absolute epoch, then selected the final `object_state` row for
`iss` by descending sample index.

Outputs inspected: The source `review/run.sqlite`, completed-run state product,
materialized scenario YAML, handoff manifest, and handoff comparison packet.

Evidence: The product records the source review/config hashes, object ID,
sample index/time, canonical ECI frame, and derived UTC Julian-date epoch. The
manifest records successful validation and `execution_occurred: false`. The
comparison packet reports `equivalent` with zero failed checks.

Conclusion: The selected translational state is ready to seed the separately
named ONP continuation without manual transcription. No continuation run has
occurred.

Limitations: This continues translational state and only a matching complete
state covariance when available. It does not continue controller, estimator,
attitude, or mission-module memory and does not establish operational accuracy.

Next run: After explicit authorization, execute the generated scenario, then
repeat `compare-handoff` with `--run-output-dir` to compare its first review row
with the exported state.
