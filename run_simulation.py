from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from sim.config import load_simulation_yaml, validate_scenario_plugins
from sim.execution import run_simulation_config_file


QUICKSTART_CONFIG = Path(__file__).resolve().parent / "configs" / "quickstart_5min.yaml"


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


def _check_import(module_name: str) -> tuple[bool, str]:
    code = (
        "import importlib; "
        f"m=importlib.import_module({module_name!r}); "
        "print(getattr(m, '__version__', '') or 'available')"
    )
    try:
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False, timeout=10.0)
    except Exception as exc:
        return False, str(exc)
    detail = (proc.stdout or proc.stderr or "").strip().splitlines()
    message = detail[-1] if detail else f"exit {proc.returncode}"
    return proc.returncode == 0, message


def _print_doctor_report() -> bool:
    checks: list[tuple[str, bool, str, bool]] = [
        ("Python", sys.version_info >= (3, 9), sys.version.split()[0], True),
        ("Quickstart config", QUICKSTART_CONFIG.exists(), str(QUICKSTART_CONFIG), True),
    ]
    for module_name, required in (("yaml", True), ("numpy", True), ("matplotlib", False), ("PySide6", False), ("pygame", False)):
        ok, detail = _check_import(module_name)
        checks.append((module_name, ok, detail, required))
    try:
        output_root = Path("outputs")
        output_root.mkdir(parents=True, exist_ok=True)
        probe = output_root / ".doctor_write_test"
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink()
        checks.append(("Output directory", True, str(output_root), True))
    except Exception as exc:
        checks.append(("Output directory", False, str(exc), True))
    if QUICKSTART_CONFIG.exists():
        try:
            cfg = load_simulation_yaml(str(QUICKSTART_CONFIG))
            errors = list(validate_scenario_plugins(cfg))
            checks.append(("Quickstart validation", not errors, "OK" if not errors else "; ".join(errors[:3]), True))
        except Exception as exc:
            checks.append(("Quickstart validation", False, str(exc), True))
    print("")
    print("=" * 72)
    print("ORBITAL ENGAGEMENT LAB DOCTOR")
    print("=" * 72)
    overall_ok = True
    for label, ok, detail, required in checks:
        status = "OK" if ok else ("WARN" if not required else "FAIL")
        print(f"{label:<22} : {status} - {detail}")
        if required and not ok:
            overall_ok = False
    print("-" * 72)
    print("Ready: run `python run_simulation.py --quickstart`." if overall_ok else "Not ready: fix FAIL items and rerun doctor.")
    print("Optional plotting/GUI/game dependencies may show WARN and are not required for quickstart.")
    print("=" * 72)
    return overall_ok


def _output_index_path(out: dict, cfg) -> str:
    run = dict(out.get("run", {}) or {})
    for candidate in (run.get("output_index_md"), out.get("output_index_md")):
        text = str(candidate or "").strip()
        if text:
            return text
    index_path = Path(str(run.get("output_dir") or cfg.outputs.output_dir)) / "index.md"
    return str(index_path) if index_path.exists() else ""


def _open_output_folder(path_text: str | Path) -> bool:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if path.is_file():
        path = path.parent
    if not path.exists():
        print(f"Open Output: skipped; folder does not exist: {path}")
        return False
    try:
        if sys.platform == "darwin":
            cmd = ["open", str(path)]
        elif os.name == "nt":  # pragma: no cover
            cmd = ["cmd", "/c", "start", "", str(path)]
        else:
            cmd = ["xdg-open", str(path)]
        subprocess.Popen(cmd)
        print(f"Open Output: {path}")
        return True
    except Exception as exc:
        print(f"Open Output: failed to open {path} ({exc})")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a public-core Orbital Engagement Lab scenario.")
    parser.add_argument("--config", default="", help="Path to a simulation scenario YAML file.")
    parser.add_argument("--quickstart", action="store_true", help="Run the bundled five-minute quickstart scenario.")
    parser.add_argument("--doctor", action="store_true", help="Check the local Python environment and quickstart readiness.")
    parser.add_argument("--open-output", action="store_true", help="Open the output folder after a successful simulation run.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the scenario config and exit without running the simulator.",
    )
    args = parser.parse_args()
    if args.doctor:
        if not _print_doctor_report():
            raise SystemExit(1)
        return
    config_path = str(QUICKSTART_CONFIG if args.quickstart else args.config)
    if not config_path:
        raise SystemExit("Provide --config PATH or use --quickstart.")

    cfg = load_simulation_yaml(config_path)
    _reject_batch_analysis(cfg)
    errors = list(validate_scenario_plugins(cfg)) if bool(cfg.simulator.plugin_validation.get("strict", True)) else []
    if args.validate_only:
        _print_preflight(config_path, cfg, errors)
        if errors:
            raise SystemExit(1)
        return
    if errors:
        msg = "Plugin validation failed:\n- " + "\n- ".join(errors)
        raise SystemExit(msg)

    out = run_simulation_config_file(config_path)
    run = dict(out.get("run", {}) or {})
    print("")
    print("=" * 72)
    print("SIMULATION COMPLETED")
    print("=" * 72)
    print(f"Scenario : {out.get('scenario_name', run.get('scenario_name', 'unknown'))}")
    print(f"Samples  : {run.get('samples', 0)}")
    print(f"Duration : {float(run.get('duration_s', 0.0)):.1f} s")
    print(f"Output   : {run.get('output_dir') or cfg.outputs.output_dir}")
    index_path = _output_index_path(out, cfg)
    if index_path:
        print(f"Start Here: {index_path}")
    if args.open_output:
        _open_output_folder(run.get("output_dir") or cfg.outputs.output_dir)
    print("=" * 72)


if __name__ == "__main__":
    main()
