# Curated Examples

The product-facing examples are YAML scenario configs. They are meant to be
validated and run through the standard CLI:

```bash
python run_simulation.py --config examples/configs/public_tle_2hr_propagation.yaml --validate-only
python run_simulation.py --config examples/configs/public_tle_2hr_propagation.yaml
```

## Public Configs

- `public_tle_2hr_propagation.yaml`: predict a satellite state history for two hours from TLE lines.
- `public_ground_station_access_from_tle.yaml`: compute ground-station access windows from a TLE.
- `public_closed_loop_rendezvous_lqr.yaml`: run a closed-loop chaser/target rendezvous with HCW LQR.
- `public_orbit_environment_stack.yaml`: inspect perturbation/environment toggles in deterministic propagation.
- `public_attitude_hold_disturbance.yaml`: evaluate attitude hold with initial pointing error and disturbance torque.
- `public_manual_rpo_training.yaml`: launch a manual/game-style RPO scenario with editable player authority.
- `public_rendezvous_closed_loop.yaml`: broader closed-loop rendezvous with attitude pointing, sensing, EKF knowledge, and standard plots.
- `public_manual_engagement.yaml`: manual/game-mode engagement with stabilized attitude, object knowledge, and defensive target logic.

Public configs use the canonical `objects` map. Conventional object IDs such as
`chaser` and `target` are example names, not required engine slots.

Private/Pro examples use `pro_*.yaml` names in the full private workspace and
are not included in the public export.

Older exploratory Python demos live outside the supported public examples
surface.
