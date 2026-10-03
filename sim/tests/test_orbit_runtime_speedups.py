from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from sim.api import SimulationSession
from sim.performance.suite import (
    _effective_scenario_config,
    _merged_case_profile,
    load_performance_manifest,
)


def test_adaptive_trajectory_fixture_stays_within_approved_rounding_bound(tmp_path: Path) -> None:
    manifest = load_performance_manifest()
    case = next(item for item in manifest.cases if item.name == "adaptive_high_fidelity")
    profile = _merged_case_profile(manifest, case, "full", warmups=0, repeats=1)
    trajectory_cfg, _ = _effective_scenario_config(
        case,
        profile,
        output_dir=tmp_path / "trajectory",
    )
    passive_case = replace(
        case,
        base_overrides={
            **case.base_overrides,
            "objects.satellite.runtime_profile": "flight_software",
            "objects.satellite.flight_software": {
                "stack": "fsw.passive",
                "hardware_profile": "hardware.passive.v1",
                "params": {},
                "task_period_s": trajectory_cfg.scenario.simulator.dt_s,
            },
        },
    )

    passive_cfg, _ = _effective_scenario_config(
        passive_case,
        profile,
        output_dir=tmp_path / "passive",
    )
    trajectory = SimulationSession.from_config(trajectory_cfg).run().truth["satellite"]
    passive = SimulationSession.from_config(passive_cfg).run().truth["satellite"]

    # Compare the same propagation intervals: a 1 s scheduler with a 2 s
    # outer step resets the adaptive solver twice and is a different numerical
    # input. Keep the approved error bounds with matched scheduling inputs.
    np.testing.assert_allclose(trajectory[:, :3], passive[:, :3], rtol=0.0, atol=6.0e-12)
    np.testing.assert_allclose(trajectory[:, 3:6], passive[:, 3:6], rtol=0.0, atol=8.0e-15)
    np.testing.assert_array_equal(trajectory[:, 6:], passive[:, 6:])
