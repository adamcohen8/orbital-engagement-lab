# Coverage and Link Scenario Analysis

OEL can run evidence-only whole-Earth conical coverage and directed
spacecraft-to-spacecraft or spacecraft-to-ground-station link analysis after a
single ONP or scenario OGP propagation.
This post-processing does not affect commands, modes, or subsequent dynamics.
The deterministic implementation, adapters, contracts, tests, and this
workflow are part of the public core with an experimental support posture.

The scenario must declare an absolute epoch. Directional coverage and
directional terminals require achieved attitude dynamics; OEL fails closed
when that evidence is unavailable. A constant-gain terminal is attitude
independent. OGP products never acquire implied attitude.

Public Scenario YAML with `propagation_method: general` supports only
attitude-independent orbital-analysis endpoints because scenario OGP retains
no achieved attitude history. Directional sensors or terminals must use ONP
with achieved attitude, or a programmatic ECI OGP history paired explicitly
with replay or analytic-ideal attitude. Static initial OGP attitude is never
treated as achieved attitude.

```yaml
simulator:
  initial_jd_utc: 2461041.5

outputs:
  review:
    enabled: true
  orbital_analysis:
    enabled: true
    coverage:
      - analysis_id: satellite_a_global
        source_object_id: satellite_a
        sensor_id: satellite_a.imager
        order: 5
        half_angle_deg: 20.0
        quat_body_from_sensor: [1.0, 0.0, 0.0, 0.0]
        transition_time_tolerance_s: 0.1
        transition_max_iterations: 50
        include_fraction_plot: true
    directed_links:
      - analysis_id: satellite_a_to_b
        link_id: satellite_a_to_b
        tx_object_id: satellite_a
        rx_object_id: satellite_b
        tx_terminal:
          terminal_id: satellite_a.tx
          pattern: {kind: constant, gain_dbi: 6.0}
        rx_terminal:
          terminal_id: satellite_b.rx
          pattern: {kind: constant, gain_dbi: 6.0}
        carrier_frequency_hz: 2200000000.0
        tx_power_w: 10.0
        data_rate_bps: 1000000.0
        system_noise_temperature_k: 300.0
        required_eb_n0_db: 5.0
        transition_time_tolerance_s: 0.1
        transition_max_iterations: 50
        include_margin_plot: true
```

For a configured ground station, replace exactly one object selector with a
ground-station selector. The station's `min_elevation_deg` and `max_range_km`
become link defaults unless the link overrides them. Ground-station terminal
mounting is in the local ENU parent frame:

```yaml
ground_stations:
  - id: colorado_springs
    lat_deg: 38.8339
    lon_deg: -104.8214
    alt_km: 1.84
    min_elevation_deg: 10.0
    max_range_km: 2500.0

outputs:
  orbital_analysis:
    enabled: true
    directed_links:
      - analysis_id: satellite_a_to_colorado_springs
        link_id: satellite_a_to_colorado_springs
        tx_object_id: satellite_a
        rx_ground_station_id: colorado_springs
        tx_terminal:
          terminal_id: satellite_a.tx
          pattern: {kind: constant, gain_dbi: 6.0}
        rx_terminal:
          terminal_id: colorado_springs.rx
          quat_parent_from_terminal: [1.0, 0.0, 0.0, 0.0]
          pattern: {kind: constant, gain_dbi: 30.0}
        carrier_frequency_hz: 2200000000.0
        tx_power_w: 10.0
        data_rate_bps: 1000000.0
        system_noise_temperature_k: 300.0
        required_eb_n0_db: 5.0
```

Each endpoint must declare exactly one of `*_object_id` or
`*_ground_station_id`. Fixed-site-to-fixed-site links are outside this product.

Coverage artifacts are written under
`orbital_analysis/coverage/<analysis_id>/`; link artifacts are written under
`orbital_analysis/directed_links/<analysis_id>/`. When review output is
enabled, the queryable tables are:

- `coverage_summary`, `coverage_samples`, `coverage_intervals`, and
  `coverage_transitions`;
- `link_summary`, `link_samples`, `link_windows`, and `link_transitions`.

In link tables, the legacy column names `tx_object_id` and `rx_object_id` hold
generalized endpoint IDs; a fixed site therefore appears in an
`*_object_id` column. Use the adjacent `tx_endpoint_kind` and
`rx_endpoint_kind` columns to distinguish `spacecraft` from
`fixed_wgs84_site`. Terminal parent frames are not separate authoring fields
in scenario YAML: OEL resolves `body` for spacecraft endpoints and `enu` for
fixed-site endpoints, then retains those resolved values as
`tx_terminal_parent_frame` and `rx_terminal_parent_frame` provenance.

For example:

```sql
SELECT s.link_id, s.tx_object_id, s.rx_object_id,
       SUM(l.available) AS available_samples
FROM link_summary s LEFT JOIN link_samples l USING (analysis_id)
WHERE s.rx_object_id = 'colorado_springs'
GROUP BY s.analysis_id, s.link_id, s.tx_object_id, s.rx_object_id;
```

The standard run summary and `index.md` surface coverage and availability
results directly. Requested native plots are included in the ordinary plot
inventory and each has a sibling `.quality.json` receipt from OEL's strict
plot-quality policy.

Completed ONP/review histories and completed ECI OGP products normalize through
`sim.analysis.history_adapters.AnalysisHistory`. Retained state histories use
cubic Hermite position/velocity interpolation and shortest-arc quaternion
SLERP for event refinement. A caller may provide an arbitrary-epoch state
provider instead. The evidence records the refinement source; the two are not
presented as equivalent propagation claims.

This adapter currently covers the canonical conical sensor and directed-link
products. Rich footprints, communications coverage, constellation aggregation,
tasking, and runtime-causal consumers remain separate integration work.

Validate and run the checked-in public example with:

```bash
python run_simulation.py --config examples/configs/public_coverage_and_link_analysis.yaml --validate-only
python run_simulation.py --config examples/configs/public_coverage_and_link_analysis.yaml
python -m sim.agent_task run coverage_link_review --output-root outputs/agent_tasks
python -m sim.review outputs/examples/public_coverage_and_link_analysis --saved-query coverage_summary
python -m sim.review outputs/examples/public_coverage_and_link_analysis --saved-query directed_link_summary
```

When OEL MCP is connected, `coverage_link_review` is a supported public task
recipe. The MCP review surface also exposes the named queries
`coverage_summary`, `coverage_transition_summary`, `directed_link_summary`, and
`directed_link_windows`, plus the `coverage_fraction` and
`directed_link_margin` plot recipes.

Coverage and directed-link samples are evaluated on the retained scenario
history. Changing `simulator.dt_s` therefore changes propagation/history
sampling as well as analysis cadence; it is not a pure post-processing
resample. Window rows carry start/end censoring and transition/refinement
dispositions so a window spanning the study boundary is not over-read as a
fully observed acquisition or loss.

This workflow is deterministic engineering analysis, not calibrated sensor
performance or operational communications assurance. See the individual
contracts and [programmatic acceptance record](validation-coverage-link-programmatic.md)
for exact supported inputs, evidence, and non-claims.
