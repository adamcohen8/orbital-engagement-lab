# Coverage and Link Scenario Analysis

OEL can run evidence-only whole-Earth conical coverage and directed
object-to-object link analysis after a single ONP or scenario OGP propagation.
This post-processing does not affect commands, modes, or subsequent dynamics.
The deterministic implementation, adapters, contracts, tests, and this
workflow are part of the public core with an experimental support posture.

The scenario must declare an absolute epoch. Directional coverage and
directional terminals require achieved attitude dynamics; OEL fails closed
when that evidence is unavailable. A constant-gain terminal is attitude
independent. OGP products never acquire implied attitude.

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
```

Coverage artifacts are written under
`orbital_analysis/coverage/<analysis_id>/`; link artifacts are written under
`orbital_analysis/directed_links/<analysis_id>/`. When review output is
enabled, the queryable tables are:

- `coverage_summary`, `coverage_samples`, `coverage_intervals`, and
  `coverage_transitions`;
- `link_summary`, `link_samples`, `link_windows`, and `link_transitions`.

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
```

This workflow is deterministic engineering analysis, not calibrated sensor
performance or operational communications assurance. See the individual
contracts and [programmatic acceptance record](validation-coverage-link-programmatic.md)
for exact supported inputs, evidence, and non-claims.
