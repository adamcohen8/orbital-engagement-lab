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
- The public model-reference docs describe the configured physics equations,
  assumptions, implementation locations, evidence hooks, and limitations needed
  to interpret validation results; see
  [Physics Model Reference](physics-models.md).
- The flagship RIC_PD 10 km scenario demonstrates a closed-loop RPO workflow
  with tuned public controller gains and attitude-gated thrust application under
  the assumptions written in `configs/ric_pd_10km_experiment.yaml`.
- The focused RIC_PD 10 km validation posture documents the flagship scenario's
  intended gates and limitations. The public export can rerun the scenario and
  companion analysis script; the automated validation harness and evidence
  manifests live in the Pro/private workspace.
- The public payload and artifact surfaces are documented in the engine,
  scenario YAML, and payload contracts.
- Selected external-reference validation evidence should be public when the
  reference material, commands, tolerances, and artifacts are redistributable.
  HPOP/MATLAB orbit comparisons and Basilisk attitude dynamics/actuator
  comparisons are the strongest trust signals for OEL, but they support only
  the specific tested cases and not general mission-assurance claims.

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

Some external-reference harnesses, HPOP comparison workflows, Basilisk attitude
reference sweeps, and evidence manifests currently exist in the Pro/private
workspace. Public-safe slices of that evidence should be promoted into the
public trust baseline as they are redacted, reproduced, and tied to bounded
claims. Larger automation, private release evidence, proprietary reference
data, and customer-specific validation packages remain Pro/private.

Model-reference pages are part of that traceability chain. They document what
equations and assumptions a validation result exercised, but they do not replace
tests, external-reference comparisons, or scenario-specific evidence.

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
