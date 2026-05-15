# Plotting

Orbital Engagement Lab includes public single-run plotting for quick inspection,
debugging, and documentation artifacts. Campaign, benchmark, optimization, and
sensitivity plots live in the pro layer.

## YAML Usage

```yaml
outputs:
  output_dir: "outputs/my_run"
  mode: "save"
  plots:
    enabled: true
    preset: "rendezvous"
    reference_object_id: "target"
    draw_earth_map: true
    figure_ids:
      - "run_dashboard"
      - "rendezvous_summary"
    dpi: 160
```

`preset` expands to a useful bundle of figure IDs. You can add more
`figure_ids` alongside a preset. `reference_object_id` should name an active
object from the scenario's canonical `objects` map; `target` is only the
convention used by the rendezvous examples.
Set `orbital_elements_object_id` under `outputs.plots` when you want COE figures
to focus on one object; otherwise they overlay all objects with valid state
histories.
Set `draw_earth_map: true` under `outputs.plots` to use a world-map background
for static `ground_track` and `ground_track_multi` figures. The plotter uses
Cartopy when available and otherwise falls back to a lightweight built-in map.

For the flagship plotting scenario used by the checked-in gallery:

```bash
python run_simulation.py --config configs/hcw_pd_10km_experiment.yaml
```

See the [plot gallery](plot-gallery.md) for checked-in examples generated from
that config.

## Plot Presets

- `minimal`: run dashboard
- `orbit`: run dashboard, multi-object ECI trajectory, multi-object ground track, orbital-elements summary
- `rendezvous`: run dashboard, rendezvous summary, RIC projections, relative range, control effort
- `attitude`: run dashboard, quaternion components, body rates, quaternion error
- `estimation`: estimation error norms, component errors, knowledge timeline, sensor access
- `access`: ground-station access and multi-object ground track
- `rocket`: run dashboard, ascent/GNC diagnostics, fuel, orbital elements, timeline, downrange/altitude, max-Q throttle, TVC/aero authority, and insertion scorecard
- `debug`: every public single-run figure ID

## Common Figure IDs

- `run_dashboard`: one-page summary of trajectory, relative motion, thrust, delta-v, and rates
- `rendezvous_summary`: RIC projections, range, relative speed, and RIC components
- `orbital_element_a`: semi-major axis over time
- `orbital_element_ecc`: eccentricity over time
- `orbital_element_inc`: inclination over time
- `orbital_element_raan`: right ascension of ascending node over time
- `orbital_element_argp`: argument of perigee over time
- `orbital_element_true_anomaly`: true anomaly over time
- `orbital_elements_summary`: six-panel COE history, arranged as two columns by three rows
- `orbital_elements_angles`: inclination, RAAN, argument of perigee, and true anomaly on one plot
- `ground_track`: per-object static ground track
- `ground_track_multi`: all-object static ground track
- `control_effort`: thrust components, magnitude, and cumulative delta-v
- `estimation_error`: position and velocity belief error against truth
- `estimation_error_components`: position and velocity component errors
- `sensor_access`: observer-target access timeline, range, and knowledge position error
- `ground_station_access`: station-target access timeline, elevation, and slant range
- `attitude_control_summary`: quaternion tracking error, body-rate norm, thrust magnitude, and thrust-alignment error
- `rocket_ascent_diagnostics`: altitude, speed, dynamic pressure, mass, throttle, and key ascent telemetry
- `rocket_gnc_diagnostics`: pitch/yaw/roll guidance and control diagnostics
- `rocket_orbital_elements`: rocket orbital-element history
- `rocket_fuel_remaining`: remaining propellant history
- `rocket_mission_timeline`: liftoff, pitch-program, max-Q, stage-event, insertion-band, and final-sample timing
- `rocket_downrange_altitude`: launch altitude versus downrange distance, colored by speed when available
- `rocket_maxq_throttle`: dynamic pressure, throttle, Mach, and altitude around max-Q limiting
- `rocket_tvc_aero_authority`: TVC gimbal, alpha/beta, aero loads, dynamic pressure, and thrust-to-weight
- `rocket_insertion_scorecard`: final orbit/resource/control scorecard against configured insertion targets
- `trajectory_eci_multi`: all-object 3D ECI trajectories
- `trajectory_ric_curv_2d_multi`: all-object RIC curvilinear 2D projections
- `relative_range`: pairwise relative range over time
- `control_thrust`: per-object thrust component history

The full list is available in `sim.master_outputs.AVAILABLE_FIGURE_IDS`.

For plots that are not built in, see [Custom Analysis](custom-analysis.md) for
examples that load saved JSON/CSV artifacts and create Matplotlib figures.
Ground-station access is also saved as payload data and summary metrics, so you
can use the custom-analysis path when you need a station-specific or
publication-specific variant.

## Python API

The `sim.plotting` package works from a single-run payload:

```python
from sim.execution import run_simulation_config_file
from sim.plotting import plot_run_dashboard, plot_rendezvous_summary

payload = run_simulation_config_file("configs/hcw_lqr_two_body_perfect.yaml")

fig = plot_run_dashboard(payload, out_path="outputs/dashboard.png", close=True)
fig = plot_rendezvous_summary(payload, out_path="outputs/rendezvous.png", close=True)
```

Available API functions:

- `plot_run_dashboard`
- `plot_rendezvous_summary`
- `plot_control_effort`
- `plot_estimation_error`
- `plot_estimation_error_components`
- `plot_ground_track_from_payload`
- `plot_ground_station_access`
- `plot_orbital_element`
- `plot_orbital_elements_summary`
- `plot_orbital_elements_angles`
- `plot_attitude_control_summary`
- `plot_sensor_access`

## Public And Pro Boundary

Public plotting focuses on understanding one run. Pro plotting focuses on many
runs: Monte Carlo histograms, sensitivity plots, controller-benchmark
comparisons, optimization convergence, campaign dashboards, baseline comparison,
and report packs.
