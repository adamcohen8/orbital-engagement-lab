# Validation Claims

This page states the current public validation posture for Orbital Engagement
Lab. It is intentionally conservative: the public core should be useful and
credible without implying decision-grade mission assurance.

## Current Public Claims

The public repository supports these claims:

- Scenario YAML can be parsed, validated, and executed through the documented
  CLI and Python API entrypoints.
- Deterministic single-run two-body, relative-motion, attitude, sensing,
  estimation, and control examples are covered by unit and regression tests.
- Curated public scenarios produce reproducible JSON, CSV, Markdown, and PNG
  artifacts when run in a supported local Python environment.
- The flagship RIC_PD 10 km scenario demonstrates a closed-loop RPO workflow
  with tuned public controller gains and attitude-gated thrust application under
  the assumptions written in `configs/ric_pd_10km_experiment.yaml`.
- The focused RIC_PD 10 km validation posture documents the flagship scenario's
  intended gates and limitations. The public export can rerun the scenario and
  companion analysis script; the automated validation harness and evidence
  manifests live in the Pro/private workspace.
- The public payload and artifact surfaces are documented in the engine,
  scenario YAML, and payload contracts.

## Explicit Non-Claims

The public repository does not claim:

- flight qualification,
- operational decision authority,
- safety certification,
- validated performance for arbitrary mission envelopes,
- high-fidelity force-model accuracy for all orbit regimes,
- equivalence to STK, FreeFlyer, GMAT, Orekit, Basilisk, MATLAB/Simulink, or any
  program-specific truth model,
- correctness of user-supplied plugin modules, controllers, or scenario YAML.

## Evidence Artifacts

For public users, the first evidence path is:

```bash
.venv/bin/python run_simulation.py --quickstart
.venv/bin/python run_simulation.py --config configs/ric_pd_10km_experiment.yaml
.venv/bin/python examples/python/flagship_analysis.py
```

Review:

- `outputs/quickstart_5min/index.md`
- `outputs/flagship_ric_pd_10km/index.md`
- `outputs/flagship_ric_pd_10km/master_run_summary.json`
- `outputs/flagship_ric_pd_10km/custom_analysis/flagship_metrics.json`

For the focused validation claim and gates, see
[RIC_PD 10 km Validation Package](validation-ric-pd-10km.md).

Private validation harnesses, HPOP comparison workflows, and evidence manifests
exist in the Pro/private workspace. Those artifacts are useful for engineering
review, but they still require independent interpretation, traceability, and
mission-envelope qualification before they can support decision-grade use.

## User Responsibility

Users should independently validate:

- initial conditions and scenario assumptions,
- force-model fidelity,
- controller and actuator limits,
- sensor and estimator assumptions,
- numerical tolerances,
- runtime environment and dependency versions,
- mission-specific safety, legal, and compliance requirements.

When in doubt, treat OEL outputs as simulation evidence for review, not as an
authorization to act.
