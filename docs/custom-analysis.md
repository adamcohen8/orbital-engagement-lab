# Custom Analysis

Orbital Engagement Lab writes JSON, CSV, Markdown, and image artifacts that can
be used outside the built-in plotting system. If a user wants a plot that the
product does not provide, the intended path is:

1. enable the right saved data in YAML,
2. run the scenario or campaign,
3. open the output directory `index.md`, and
4. load the saved artifacts with Python, NumPy, Matplotlib, or another analysis
   tool.

Built-in plots are useful defaults. Saved artifacts are the escape hatch for
custom engineering questions.

## Choose The Right Artifact

For a single run:

- `index.md`: start-here output guide and artifact inventory.
- `master_run_summary.json`: stable summary metrics and artifact maps.
- `master_run_log.json`: full time histories for custom plots.

For Pro campaign workflows:

- `master_monte_carlo_summary.json`: aggregate Monte Carlo campaign payload.
- `master_monte_carlo_analyst_pack.json`: selected details intended for deeper
  campaign review.
- `master_analysis_sensitivity_summary.json`: sensitivity study payload.
- `master_analysis_sensitivity_runs.csv`: one row per sensitivity run.
- `master_analysis_sensitivity_rankings.csv`: ranked parameter effects.

For AI-assisted reports:

- `master_ai_report_input.json`: scoped model-facing packet.
- `master_ai_report_prompt.md`: exact generated prompt.
- `master_ai_report_review_packet.md`: human-readable pre-call review summary.

Use the deterministic simulation and analysis artifacts for custom plotting.
Treat AI report artifacts as audit material for the report workflow, not as the
primary source of simulation truth.

## Save Full Single-Run Data

To make arbitrary time-history plots from a single run, enable the full log:

```yaml
outputs:
  output_dir: "outputs/my_run"
  mode: "save"
  stats:
    enabled: true
    save_json: true
    save_full_log: true
```

After the run, open:

```text
outputs/my_run/index.md
outputs/my_run/master_run_log.json
```

`master_run_summary.json` is enough for summary plots. `master_run_log.json` is
the better source for time-series plots because it contains histories such as
`time_s`, `truth_by_object`, `belief_by_object`, `applied_thrust_by_object`,
`applied_torque_by_object`, `knowledge_by_observer`, and controller debug data
when available.

## Single-Run Example: Altitude Over Time

The truth state convention starts with ECI position and velocity. The first
three truth columns are position in kilometers.

```python
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EARTH_RADIUS_KM = 6378.137

outdir = Path("outputs/plotting_rendezvous_demo")
payload = json.loads((outdir / "master_run_log.json").read_text())

time_s = np.asarray(payload["time_s"], dtype=float)
target_truth = np.asarray(payload["truth_by_object"]["target"], dtype=float)
target_r_eci_km = target_truth[:, 0:3]
target_altitude_km = np.linalg.norm(target_r_eci_km, axis=1) - EARTH_RADIUS_KM

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(time_s / 60.0, target_altitude_km)
ax.set_xlabel("Time (min)")
ax.set_ylabel("Altitude (km)")
ax.set_title("Target Altitude")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(outdir / "custom_target_altitude.png", dpi=160)
```

If `master_run_log.json` is missing, set `outputs.stats.save_full_log: true`
and rerun the scenario.

## Single-Run Example: Chaser-Target Range

For two-object scenarios, relative range can be computed directly from the
truth histories:

```python
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

outdir = Path("outputs/examples/public_rendezvous_closed_loop")
payload = json.loads((outdir / "master_run_log.json").read_text())

time_s = np.asarray(payload["time_s"], dtype=float)
target = np.asarray(payload["truth_by_object"]["target"], dtype=float)
chaser = np.asarray(payload["truth_by_object"]["chaser"], dtype=float)

range_km = np.linalg.norm(chaser[:, 0:3] - target[:, 0:3], axis=1)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(time_s, range_km)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Range (km)")
ax.set_title("Chaser-Target Range")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(outdir / "custom_chaser_target_range.png", dpi=160)
```

This is useful when a built-in figure is close to what you need but not exactly
right for a review or paper.

## Monte Carlo Custom Plots

For Pro Monte Carlo campaigns, make sure aggregate summaries are saved:

```yaml
outputs:
  monte_carlo:
    save_aggregate_summary: true
    save_raw_runs: true
```

`save_aggregate_summary` writes the campaign summary. `save_raw_runs` adds a
run-details artifact when deeper per-run review is needed.

Example histogram from `master_monte_carlo_summary.json`:

```python
import json
from pathlib import Path

import matplotlib.pyplot as plt

outdir = Path("outputs/ai_report_mc_smoke")
summary = json.loads((outdir / "master_monte_carlo_summary.json").read_text())

runs = summary.get("runs", [])
closest_approach_km = [
    float(run["closest_approach_km"])
    for run in runs
    if run.get("closest_approach_km") is not None
]

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(closest_approach_km, bins="auto", edgecolor="black", alpha=0.8)
ax.set_xlabel("Closest Approach (km)")
ax.set_ylabel("Run Count")
ax.set_title("Monte Carlo Closest Approach")
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(outdir / "custom_closest_approach_histogram.png", dpi=160)
```

If the list is empty, the campaign summary did not include the per-run field
you need. Enable raw run details or add the metric to the campaign reporting
payload before rerunning.

## Sensitivity Custom Plots

Sensitivity studies write CSV files for quick external plotting:

```text
master_analysis_sensitivity_runs.csv
master_analysis_sensitivity_rankings.csv
```

Example using only the Python standard library plus Matplotlib:

```python
import csv
from pathlib import Path

import matplotlib.pyplot as plt

outdir = Path("outputs/sensitivity_oaat_demo")
runs_csv = outdir / "master_analysis_sensitivity_runs.csv"

rows = list(csv.DictReader(runs_csv.open(newline="", encoding="utf-8")))

parameter = "simulator.duration_s"
metric = "metric:summary.duration_s"

x = []
y = []
for row in rows:
    if row.get("parameter_path") != parameter:
        continue
    value = row.get("parameter_value")
    metric_value = row.get(metric)
    if value not in (None, "") and metric_value not in (None, ""):
        x.append(float(value))
        y.append(float(metric_value))

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(x, y, marker="o")
ax.set_xlabel(parameter)
ax.set_ylabel(metric)
ax.set_title("Custom Sensitivity Response")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(outdir / "custom_sensitivity_response.png", dpi=160)
```

CSV field names can vary with the configured metrics. Inspect the header row or
open `index.md` and the generated sensitivity report when adapting the example.

## Practical Guidance

- Open `index.md` first. It lists the artifacts actually written for that run.
- Use `master_run_log.json` for arbitrary single-run time histories.
- Use summary JSON for stable top-level metrics.
- Use campaign and sensitivity CSVs when available for spreadsheet-style
  plotting.
- Prefer artifact maps in JSON over hard-coded filenames when writing reusable
  scripts.
- Save custom plots back into the same output directory when they belong to the
  same analysis record.

## Limitations

Saved artifacts reflect what the run was configured to record. If a signal was
not logged, a custom plot cannot reconstruct it after the fact. In that case,
enable the relevant output option, add the metric to the analysis payload, or
extend the simulator logging before rerunning.
