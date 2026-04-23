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

## Private Configs

- `pro_controller_benchmark.yaml`: Pro controller benchmark suite that compares rendezvous controller variants and writes leaderboard/report artifacts.
- `pro_monte_carlo_ai_report.yaml`: Pro Monte Carlo campaign with gated reporting, dashboard artifacts, and dry-run AI report packet generation.

Older exploratory Python demos live outside the supported examples surface.
