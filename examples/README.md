# Curated Examples

The product-facing examples are YAML scenario configs. They are meant to be
validated and run through the standard CLI:

```bash
python run_simulation.py --config examples/configs/automation_smoke.yaml --validate-only
python run_simulation.py --config examples/configs/automation_smoke.yaml
```

## Configs

- `automation_smoke.yaml`: small headless smoke scenario for first-run checks.
- `plotting_rendezvous_demo.yaml`: rendezvous scenario that generates standard plots.
- `hcw_lqr_two_body_perfect.yaml`: compact relative-orbit control case.
- `game_mode_basic.yaml`: minimal game/manual-control scenario.
- `simulation_template.yaml`: broad starter template for new scenarios.

Older exploratory Python demos live outside the supported examples surface.
