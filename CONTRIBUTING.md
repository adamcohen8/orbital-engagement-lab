# Contributing

Thanks for taking a look at Orbital Engagement Lab.

## Development Setup

Use Python 3.10 through 3.12. The commands below use Python 3.11; replace
`python3.11` with `python3.10` or `python3.12` if that is your installed
interpreter.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install ".[dev]"
```

## Checks

Run the focused public checks before opening a pull request:

```bash
python -m ruff check <changed-python-files>
python -m ruff format --check <changed-python-files>
python -m pytest -q sim/tests/test_scenario_yaml_config.py sim/tests/test_app_io.py sim/tests/test_api.py sim/tests/test_master_simulator.py
python run_simulation.py --config configs/automation_smoke.yaml --validate-only
```

Ruff is being rolled in incrementally. Use the changed-file commands for normal
PRs until the existing project baseline has been cleaned enough for
`python -m ruff check .` and `python -m ruff format --check .` to become hard
repo-wide gates.

To format intentionally:

```bash
python -m ruff format .
```

To preview local generated artifacts and caches that can be cleaned:

```bash
python tools/clean_local_artifacts.py
```

Apply the cleanup only after reviewing the dry run:

```bash
python tools/clean_local_artifacts.py --apply
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
