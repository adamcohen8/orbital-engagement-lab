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
    figure_ids:
      - "run_dashboard"
      - "rendezvous_summary"
    dpi: 160
```

`preset` expands to a useful bundle of figure IDs. You can add more
`figure_ids` alongside a preset.

For a ready-to-run plotting scenario:

```bash
python run_simulation.py --config configs/plotting_rendezvous_demo.yaml
```

See the [plot gallery](plot-gallery.md) for checked-in examples generated from
that config.

## Plot Presets

- `minimal`: run dashboard
- `orbit`: run dashboard, multi-object ECI trajectory, multi-object ground track
- `rendezvous`: run dashboard, rendezvous summary, RIC projections, relative range, control effort
- `attitude`: run dashboard, quaternion components, body rates, quaternion error
- `estimation`: estimation error norms, component errors, knowledge timeline, sensor access
- `rocket`: run dashboard, ascent diagnostics, orbital elements, fuel remaining
- `debug`: every public single-run figure ID

## Common Figure IDs

- `run_dashboard`: one-page summary of trajectory, relative motion, thrust, delta-v, and rates
- `rendezvous_summary`: RIC projections, range, relative speed, and RIC components
- `ground_track`: per-object static ground track
- `ground_track_multi`: all-object static ground track
- `control_effort`: thrust components, magnitude, and cumulative delta-v
- `estimation_error`: position and velocity belief error against truth
- `estimation_error_components`: position and velocity component errors
- `sensor_access`: observer-target access timeline, range, and knowledge position error
- `trajectory_eci_multi`: all-object 3D ECI trajectories
- `trajectory_ric_curv_2d_multi`: all-object RIC curvilinear 2D projections
- `relative_range`: pairwise relative range over time
- `control_thrust`: per-object thrust component history

The full list is available in `sim.master_outputs.AVAILABLE_FIGURE_IDS`.

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
- `plot_sensor_access`

## Public And Pro Boundary

Public plotting focuses on understanding one run. Pro plotting focuses on many
runs: Monte Carlo histograms, sensitivity plots, controller-benchmark
comparisons, optimization convergence, campaign dashboards, baseline comparison,
and report packs.
