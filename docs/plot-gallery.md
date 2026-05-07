# Plot Gallery

These images are generated from the public-safe flagship 10 km HCW PD RPO
scenario:

```bash
python run_simulation.py --config configs/hcw_pd_10km_experiment.yaml
```

The scenario writes run artifacts under `outputs/flagship_hcw_pd_10km/`. The
gallery images below are checked-in snapshots from that run so visitors can see
the flagship plotting surface without running the simulator first. When you run
the scenario locally, start with `outputs/flagship_hcw_pd_10km/index.md` and use
this page as a quick visual reference.

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

### Chaser Thrust Alignment Error

![Chaser thrust alignment error](assets/plots/chaser_thrust_alignment_error.png)
