# Plot Gallery

These images are generated from the public-safe flagship 10 km RIC_PD RPO
scenario with the OEL dark artifact style:

```bash
python run_simulation.py --config configs/ric_pd_10km_experiment.yaml --resource-profile off
```

The scenario writes run artifacts under `outputs/flagship_ric_pd_10km/`. The
gallery images below are checked-in OEL-styled snapshots from that run so
visitors can see the flagship plotting surface without running the simulator
first. The config's laptop-safe profile may suppress plot generation for quick
local smoke runs; use the command above when intentionally refreshing gallery
images. The live config may generate additional figures as the plotting catalog
grows; when you run the scenario locally, start with
`outputs/flagship_ric_pd_10km/index.md` for the authoritative artifact list and
use this page as a quick visual reference.

## Run Dashboard

![Run dashboard](assets/plots/run_dashboard.png)

## Rendezvous Summary

![Rendezvous summary](assets/plots/rendezvous_summary.png)

## Control Effort

![Control effort](assets/plots/control_effort.png)

## Relative Range

![Relative range](assets/plots/relative_ranges.png)

## RIC Curvilinear Trajectory

![RIC curvilinear trajectory](assets/plots/trajectory_ric_curv_2d_multi.png)

## Integrated Attitude And Burn Gating

### Chaser Attitude

![Chaser attitude](assets/plots/chaser_attitude.png)

### Chaser Quaternion Error

![Chaser quaternion error](assets/plots/chaser_quaternion_error.png)

### Attitude-Control Summary

The current local run also writes `attitude_control_summary.png` when requested
by the active config. It combines quaternion tracking, body-rate, thrust, and
alignment evidence in one review figure.

### Chaser Thrust Alignment Error

![Chaser thrust alignment error](assets/plots/chaser_thrust_alignment_error.png)
