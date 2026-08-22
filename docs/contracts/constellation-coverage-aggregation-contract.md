# Constellation Coverage Aggregation Contract

Status: **frozen and implemented for the Phase 5 programmatic core v0.2**.

Contract identifier: `oel.constellation-coverage-aggregation.v0.2`.

## Product Boundary

Constellation Coverage Aggregation combines two or more completed global
coverage products. It does not propagate assets or rerun footprint, service,
or RF geometry. Every member must use identical analysis epochs, HEALPix NESTED
order, canonical cell coordinates, and a complete global-Earth disposition.

Members may represent geometric, sensor, rich-service, or communications
coverage, but one aggregate may contain only a single domain disposition.
Communications members must also share one non-empty `service_id`. Every
aggregate declares a non-empty `service_definition_id`, retained in its
identity and summary, so unlike service meanings cannot be mixed silently.
That caller declaration is provenance, not independent proof that the member
models are scientifically interchangeable.

## Frozen Semantics

The configuration declares a unique aggregate ID, a sorted unique set of
member analysis IDs, HEALPix order, service-definition ID, required
multiplicity, and a resource limit. Member order supplied by a caller does not
change the result.

Sparse intervals are expanded only inside the bounded evaluator. At each
sample and cell, multiplicity is the number of available members. The
aggregate is service-qualified when multiplicity is at least the declared
threshold. The resulting boolean mask is reduced with the frozen sampled
interval, dwell, revisit, gap, and censoring rules.

The result retains:

- qualified cell count and fraction by sample;
- maximum multiplicity and active-asset count by sample;
- mean and maximum multiplicity by cell;
- the complete cell-sample multiplicity histogram;
- sparse qualified intervals and per-cell metrics;
- mean and percentiles of finite complete-revisit values;
- maximum finite complete-revisit gap; and
- cells that never meet the declared multiplicity.

Boundary-censored gaps are not promoted to complete maximum-revisit values.

## Evidence and Failure Behavior

Stable artifacts are a manifest, summary JSON, sample CSV, cell CSV, and sparse
interval NPZ. Semantic identity binds sorted member semantic hashes,
configuration, multiplicity, epochs, and intervals.

The evaluator first verifies each source product's completion status,
canonical cells, digest syntax, sparse intervals, sampled counts, dwell,
revisit, and censoring fields. It then fails closed for duplicate or missing
members, mixed domain dispositions, mismatched communications service IDs,
mismatched times, order, cells, unsupported multiplicity, or a resource
estimate beyond the configured envelope.

## Non-Claims

Aggregation does not model terminal contention, inter-satellite routing,
network paths, scheduling, correlated failures, uncertain state, probability,
or steady-state/repeating coverage beyond the supplied horizon. It is an exact
sampled union/overlap calculation over its source products, not an independent
validation of those members.
