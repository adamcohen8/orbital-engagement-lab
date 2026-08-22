# RPO Estimation Comparison

This public validation comparison runs the same eccentric-chief RPO truth arc
with the same azimuth/elevation/range/range-rate measurement cadence and noise
across three estimator paths:

- HCW relative EKF
- TH-integrated relative EKF using numerically integrated Tschauner-Hempel dynamics
- Closed-form YA STM relative EKF for the same eccentric-chief dynamics
- Absolute ECI orbit EKF baseline

The full `rpo_estimation_comparison` harness and its retained evidence are
private release-validation surfaces. Public checkouts can exercise the shipped
estimator behavior with:

```bash
python -m pytest sim/tests/test_relative_hcw_ekf.py sim/tests/test_relative_th_ekf.py
```

The benchmark values below are a bounded historical summary from the private
release evidence; they are not represented as a rerunnable public harness.

The private suite writes:

- `outputs/validation_harness_rpo_estimation_comparison/validation_harness_report.json`
- `outputs/validation_harness_rpo_estimation_comparison/validation_harness_report.md`
- `outputs/validation_harness_rpo_estimation_comparison/validation_estimation_knowledge_summary.md`

Current benchmark envelope:

| Estimator row | Chief orbit | Duration | Measurement cadence | Measurement model |
| --- | --- | ---: | ---: | --- |
| `rpo_estimation_compare_hcw` | `a=9000 km`, `e=0.25` | 1800 s | 10 s | `relative_angles_range_rate` |
| `rpo_estimation_compare_th` | `a=9000 km`, `e=0.25` | 1800 s | 10 s | `relative_angles_range_rate` |
| `rpo_estimation_compare_ya_stm` | `a=9000 km`, `e=0.25` | 1800 s | 10 s | `relative_angles_range_rate` |
| `rpo_estimation_compare_eci_ekf` | `a=9000 km`, `e=0.25` | 1800 s | 10 s | `relative_angles_range_rate` |

The latest local run on 2026-06-28 produced:

| Estimator row | Position RMS km | Velocity RMS km/s | NIS mean | Detection rate |
| --- | ---: | ---: | ---: | ---: |
| HCW relative EKF | 0.0021646067352051952 | 3.0027299876662326e-05 | 16.75977738623135 | 0.2008879023307436 |
| TH-integrated relative EKF | 0.0012271800572850405 | 4.566926412785263e-06 | 4.263492294806171 | 0.2008879023307436 |
| Closed-form YA STM relative EKF | 0.0012271800572323948 | 4.566926411585116e-06 | 4.263492294318839 | 0.2008879023307436 |
| ECI orbit EKF baseline | 0.0011464816975175606 | 5.6432627571356135e-06 | 3.579648059084772 | 0.2008879023307436 |

Interpretation: in this eccentric RPO case, HCW remains bounded but shows larger
model-error signatures between sparse updates. The TH-integrated path improves
relative-state consistency and lands near the ECI orbit EKF baseline while
remaining in the RPO-native rectangular RIC state. The YA row now uses the
closed-form Yamanaka-Ankersen anomaly-domain STM mapped into the same
rectangular RIC km/km/s state, so it should remain numerically close to the
TH-integrated row while avoiding finite-difference STM construction.

This is internal simulator-truth validation, not external tracking-data
calibration or flight qualification.
