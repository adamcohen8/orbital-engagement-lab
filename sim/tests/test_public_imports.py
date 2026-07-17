from __future__ import annotations

import json
import subprocess
import sys


def test_import_sim_keeps_heavy_feature_modules_lazy() -> None:
    code = """
import json
import sys
import sim

blocked = [
    "sim.control",
    "sim.dynamics",
    "sim.estimation",
    "sim.master_simulator",
    "sim.mission",
    "sim.optimization",
    "sim.sensors",
    "sim.utils.plotting",
    "sim.utils.plotting_capabilities",
    "matplotlib",
    "pygame",
]
print(json.dumps({name: name in sys.modules for name in blocked}, sort_keys=True))
"""
    proc = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, check=True)
    imported = json.loads(proc.stdout)

    assert imported == {name: False for name in imported}


def test_lazy_top_level_exports_preserve_from_sim_import_style() -> None:
    code = """
from sim import HCWLQRController, SimulationConfig, StateTruth

assert SimulationConfig.__name__ == "SimulationConfig"
assert HCWLQRController.__name__ == "HCWLQRController"
assert StateTruth.__name__ == "StateTruth"
"""
    subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, check=True)


def test_single_run_import_keeps_unrelated_od_and_controller_families_lazy() -> None:
    code = """
import json
import sys
import sim.single_run

blocked = [
    "sim.control.orbit.advanced",
    "sim.control.orbit.hcw_mpc",
    "sim.control.attitude.cmg_steering",
    "sim.estimation.ogp_od",
    "sim.estimation.sgp4_od",
    "sim.estimation.slr_od",
    "sim.dynamics.orbit.sgp4",
    "sim.dynamics.orbit.sdp4",
    "sim.acceleration.kernels.orbit",
    "sim.acceleration.kernels.attitude",
    "sim.acceleration.kernels.estimation",
    "numba",
    "scipy",
    "scipy.stats",
]
print(json.dumps({name: name in sys.modules for name in blocked}, sort_keys=True))
"""
    proc = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, check=True)
    imported = json.loads(proc.stdout)

    assert imported == {name: False for name in imported}
