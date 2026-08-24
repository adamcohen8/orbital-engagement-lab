# Example Answer: OGP-SDP4 Deep-Space Propagation

Status: Validated and completed.

Commands: Validated and ran the checked-in YAML, then used the documented
review queries.

Review queries: `ogp_propagation_contract`, `object_final_state`, and
`object_eci_radius_extrema`.

Outputs inspected: `index.md`, `master_run_summary.json`, and
`review/run.sqlite`.

Evidence: The contract resolved `OGP-SDP4`, `deep_space`, a period above 225
minutes, and separate native/output/history frame fields. Final ECI state and
sampled radius extrema were non-empty.

Conclusion: The checked-in synthetic fixture follows OEL's continuous
deep-space dispatch and produces canonical ECI review rows.

Limitations: This is not a real catalog object and does not prove current or
operational ephemeris accuracy.

Next run: Use another fixed, checksum-valid deep-space input only when its
provenance and intended claim are explicit.
