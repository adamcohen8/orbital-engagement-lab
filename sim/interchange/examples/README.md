# OEL Interchange Phase 1 Examples

`state_estimate_accepted_current.json` is a synthetic, public-safe Product
Envelope v1 example. It references `source_od_report.json` so read-only
inspection can verify a real source fingerprint without relying on generated
or private evidence.

`validation_fixture_matrix.json` applies explicit quality and freshness
variants to that base product. The matrix covers every v1 quality disposition,
integrity status, and age status. Tests recompute the canonical product ID after
applying each variant.

These files demonstrate contracts only. They do not contain an operational
orbit estimate, do not materialize a scenario, and do not execute OEL.
