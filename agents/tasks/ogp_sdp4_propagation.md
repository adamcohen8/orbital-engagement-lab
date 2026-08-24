# OGP-SDP4 Deep-Space Propagation

Task ID: `ogp_sdp4_propagation`

Example config: `agents/examples/public_agent_ogp_sdp4_propagation.yaml`

Expected output directory: `outputs/agent_examples/public_agent_ogp_sdp4_propagation`

Answer example: `agents/tasks/examples/ogp_sdp4_propagation_answer.md`

Activate an installed OEL environment using the platform-specific instructions
in [`../../docs/installation.md`](../../docs/installation.md) before running
the commands below.

Use the example to validate and run the fixed synthetic GEO-like TLE through
continuous passive OGP. Then inspect these saved-query contracts:

## User Prompt

```text
Propagate a deterministic deep-space TLE with OGP-SDP4, prove the resolved
regime and frames, inspect final state and radius extrema, and state the limits.
```

## Expected Agent Assumptions

- Use the checked-in synthetic fixture unchanged and offline.
- Let `general.model: sgp4` dispatch by period; do not invent `model: sdp4`.
- Treat 225 minutes as the dispatch boundary and verify the resolved result.
- Do not present the synthetic object as current catalog data.

## Commands

```bash
python -m sim.review outputs/agent_examples/public_agent_ogp_sdp4_propagation --saved-query ogp_propagation_contract
python -m sim.review outputs/agent_examples/public_agent_ogp_sdp4_propagation --saved-query object_final_state
python -m sim.review outputs/agent_examples/public_agent_ogp_sdp4_propagation --saved-query object_eci_radius_extrema
```

## Required Review Queries

```sql
SELECT object_id, propagator_family, propagator_name, ogp_regime, orbital_period_min, native_frame, output_frame, state_history_frame FROM object_propagation ORDER BY object_id
```

```sql
SELECT object_id, MIN(radius_km) AS minimum_radius_km, MAX(radius_km) AS maximum_radius_km FROM object_orbital_elements GROUP BY object_id ORDER BY object_id
```

## Expected Answer Shape

- Validation and execution status.
- Resolved family/name, regime, period, and frame contract.
- Final ECI state and sampled radius extrema.
- Synthetic-fixture and operational-accuracy limitations.

## Pass Criteria

- validation and execution complete;
- the propagator contract resolves `OGP-SDP4`, `deep_space`, and a period above
  225 minutes;
- final state and radius extrema are non-empty;
- the answer states that this is a synthetic offline regression fixture, not a
  real catalog object or operational-accuracy demonstration.

## Red Flags

- Configures a nonexistent `general.model: sdp4` field.
- Calls the fixture a real or current catalog object.
- Claims operational accuracy from deterministic self-evidence.
