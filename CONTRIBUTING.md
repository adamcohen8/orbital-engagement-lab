# Contributing

Thanks for taking a look at Orbital Engagement Lab.

## Development Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install ".[dev]"
```

## Checks

Run the focused public checks before opening a pull request:

```bash
python -m pytest -q sim/tests/test_scenario_yaml_config.py sim/tests/test_app_io.py sim/tests/test_api.py sim/tests/test_master_simulator.py
python run_simulation.py --config configs/automation_smoke.yaml --validate-only
```

## Contribution Scope

Good public-core contributions include:

- simulator correctness fixes
- scenario YAML usability improvements
- reference controllers and examples
- documentation and onboarding improvements
- small validation and smoke-test coverage

Product workflows such as controller benchmarking, optimization, Monte Carlo
campaign orchestration, sensitivity studies, and campaign reporting are kept in
the pro layer.

## Style

- Keep changes scoped and testable.
- Prefer existing module patterns over new abstractions.
- Add or update tests when behavior changes.
- Avoid committing generated outputs, local data, or large ephemeris files.
