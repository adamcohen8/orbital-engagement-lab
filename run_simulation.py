from __future__ import annotations

import argparse
from pathlib import Path

from sim.config import load_simulation_yaml, validate_scenario_plugins
from sim.execution import run_simulation_config_file


def _print_preflight(config_path: str, cfg, errors: list[str]) -> None:
    print("")
    print("=" * 72)
    print("SIMULATION PREFLIGHT")
    print("=" * 72)
    print(f"Config   : {Path(config_path).resolve()}")
    print(f"Scenario : {cfg.scenario_name}")
    print(f"Mode     : Single Run")
    print(f"Timing   : duration={float(cfg.simulator.duration_s):.1f} s, dt={float(cfg.simulator.dt_s):.3f} s")
    if errors:
        print("Status   : INVALID")
        for err in errors:
            print(f"- {err}")
    else:
        print("Status   : OK")
    print("=" * 72)


def _reject_batch_analysis(cfg) -> None:
    if bool(cfg.analysis.enabled) or bool(cfg.monte_carlo.enabled):
        raise SystemExit(
            "Batch analysis is not available in the public core. "
            "Use Orbital Engagement Pro for Monte Carlo, sensitivity, controller-bench, and optimization workflows."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a public-core Orbital Engagement Lab scenario.")
    parser.add_argument("--config", required=True, help="Path to a simulation scenario YAML file.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the scenario config and exit without running the simulator.",
    )
    args = parser.parse_args()

    cfg = load_simulation_yaml(args.config)
    _reject_batch_analysis(cfg)
    errors = list(validate_scenario_plugins(cfg)) if bool(cfg.simulator.plugin_validation.get("strict", True)) else []
    if args.validate_only:
        _print_preflight(args.config, cfg, errors)
        if errors:
            raise SystemExit(1)
        return
    if errors:
        msg = "Plugin validation failed:\n- " + "\n- ".join(errors)
        raise SystemExit(msg)

    out = run_simulation_config_file(args.config)
    run = dict(out.get("run", {}) or {})
    print("")
    print("=" * 72)
    print("SIMULATION COMPLETED")
    print("=" * 72)
    print(f"Scenario : {out.get('scenario_name', run.get('scenario_name', 'unknown'))}")
    print(f"Samples  : {run.get('samples', 0)}")
    print(f"Duration : {float(run.get('duration_s', 0.0)):.1f} s")
    print(f"Output   : {run.get('output_dir', '')}")
    print("=" * 72)


if __name__ == "__main__":
    main()
