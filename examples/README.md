# Curated Examples

The product-facing examples are YAML scenario configs. They are meant to be
validated and run through the standard CLI:

```bash
python run_simulation.py --config examples/configs/public_rendezvous_closed_loop.yaml --validate-only
python run_simulation.py --config examples/configs/public_rendezvous_closed_loop.yaml
```

## Public Configs

- `public_rendezvous_closed_loop.yaml`: closed-loop rendezvous with orbit control, attitude pointing, sensing, EKF knowledge, and standard plots.
- `public_orbit_environment_stack.yaml`: deterministic high-fidelity orbit/environment propagation with perturbations and knowledge tracking.
- `public_manual_engagement.yaml`: manual/game-mode engagement with stabilized attitude, object knowledge, and defensive target logic.

Private/Pro examples are not included in the public export. Older exploratory
Python demos live outside the supported public examples surface.
