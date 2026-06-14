# First Five Minutes

This path proves the public core works through the primary CLI/YAML workflow
before you try plots, experimental desktop surfaces, or larger scenarios.

## 1. Install

```bash
git clone https://github.com/adamcohen8/orbital-engagement-lab.git
cd orbital-engagement-lab
python3.11 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install ".[dev]"
```

Use Python 3.10 through 3.12. Replace `python3.11` with `python3.10` or
`python3.12` if that is your installed interpreter.

The commands below use `.venv/bin/python` so they work even on systems where
`python` is not on `PATH`. If you activate the virtual environment first,
`python` is equivalent.

## 2. Check Your Environment

```bash
.venv/bin/python run_simulation.py --doctor
```

`WARN` entries for plotting, experimental GUI, or game dependencies are okay
for the first run. Fix any `FAIL` entries before continuing.

## 3. Run The Quickstart

```bash
.venv/bin/python run_simulation.py --quickstart
```

The command runs `configs/quickstart_5min.yaml`, a short deterministic
two-satellite rendezvous scenario. Plots are disabled to keep the first run
fast, headless, and focused on the generated summary artifacts.

Only run scenario YAML files from sources you trust. Scenario configs can point
at importable Python modules/classes for controllers, guidance, mission
strategies, and mission execution modules; loading an untrusted scenario can run
untrusted Python code.

To open the output folder automatically:

```bash
.venv/bin/python run_simulation.py --quickstart --open-output
```

## 4. Open The Start-Here File

The final console output includes:

```text
Start Here : outputs/quickstart_5min/index.md
```

Open that Markdown file first. It lists the run summary and every artifact that
was actually written.

Later, if a scenario includes `ground_stations`, the same output directory will
also include access summaries in `master_run_summary.json` and per-sample access
histories in `master_run_log.json`.

## 5. Try The Next Layer

After the quickstart works, run the flagship 10 km RIC_PD review scenario:

```bash
.venv/bin/python run_simulation.py --config configs/ric_pd_10km_experiment.yaml --validate-only
.venv/bin/python run_simulation.py --config configs/ric_pd_10km_experiment.yaml
```

Open `outputs/flagship_ric_pd_10km/index.md`, then inspect the rendezvous,
control-effort, and thrust-alignment plots. For the full review order, see
[Flagship RIC_PD 10 km Scenario](flagship-ric-pd-10km.md).

For a shorter plotting demo:

```bash
.venv/bin/python run_simulation.py --config configs/plotting_rendezvous_demo.yaml
```

For output inspection, prefer the generated `index.md`, JSON artifacts, plots,
and review-store CLI when a run enables `outputs.review.enabled: true`. The
desktop GUI and Output Review Workbench dynamic plot creator are experimental
desktop surfaces, not the first-five-minute path.
