# First Five Minutes

This path proves the public core works before you try plots, GUI extras, or
larger scenarios.

## 1. Install

```bash
git clone https://github.com/adamcohen8/orbital-engagement-lab.git
cd orbital-engagement-lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install ".[dev]"
```

## 2. Check Your Environment

```bash
python run_simulation.py --doctor
```

`WARN` entries for plotting, GUI, or game dependencies are okay for the first
run. Fix any `FAIL` entries before continuing.

## 3. Run The Quickstart

```bash
python run_simulation.py --quickstart
```

The command runs `configs/quickstart_5min.yaml`, a short deterministic
two-satellite rendezvous scenario. Plots are disabled so the first run does not
depend on Matplotlib.

To open the output folder automatically:

```bash
python run_simulation.py --quickstart --open-output
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

After the quickstart works, generate plots:

```bash
python run_simulation.py --config configs/plotting_rendezvous_demo.yaml
```

Or open the GUI:

```bash
python -m pip install ".[gui]"
python run_gui.py
```

In the GUI, use **Open Quickstart** or **Run Quickstart** from the top bar.
