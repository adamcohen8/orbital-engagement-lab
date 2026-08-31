# ruff: noqa: E402
"""Generate, schedule, and replay a public two-asset collection/downlink chain."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.analysis.collection_opportunity import (
    assess_collection_opportunities,
    write_collection_evidence,
)
from sim.analysis.directed_link import (
    DirectedLinkConfig,
    LinkTerminal,
    TerminalPattern,
    evaluate_directed_link,
    fixed_wgs84_site_history,
    spacecraft_endpoint_history,
    write_directed_link_artifacts,
)
from sim.analysis.mission_scheduling_sources import (
    MissionSchedulingSourcePlan,
    build_solve_mission_schedule_from_sources,
    verify_source_built_mission_schedule,
)
from sim.dynamics.orbit.frames import FrameContext
from sim.utils.geodesy import WGS84_A_KM

EPOCH_JD_UTC = 2451545.0


def _write_collection(source_root: Path, *, asset_id: str, name: str) -> Path:
    problem = json.loads(
        (ROOT / "examples/collection/public_equatorial_optical_collection.json").read_text(
            encoding="utf-8"
        )
    )
    problem = copy.deepcopy(problem)
    problem["name"] = name
    problem["spacecraft"]["asset_id"] = asset_id
    problem["resources"] = {"enabled": False}
    evidence = assess_collection_opportunities(problem)
    path = source_root / f"{asset_id.lower()}_collection.json"
    return write_collection_evidence(evidence, path)


def _write_link(
    source_root: Path,
    *,
    asset_id: str,
    station_asset_id: str,
    start_s: float,
    end_s: float,
    link_id: str,
) -> Path:
    context = FrameContext(
        model="simple_gmst",
        jd_utc_start=EPOCH_JD_UTC,
        source="public_mission_scheduling_source_chain",
    )
    times = np.array([start_s, end_s], dtype=float)
    station = fixed_wgs84_site_history(
        asset_id=station_asset_id,
        state_provider_id=f"{station_asset_id}.fixed_wgs84",
        times_s=times,
        geodetic_latitude_deg=0.0,
        longitude_deg=0.0,
        ellipsoidal_height_km=0.0,
        frame_context=context,
    )
    radius_scale = (WGS84_A_KM + 500.0) / WGS84_A_KM
    spacecraft = spacecraft_endpoint_history(
        asset_id=asset_id,
        state_provider_id=f"{asset_id}.analytic_overhead",
        times_s=times,
        positions_eci_km=station.position_eci_km * radius_scale,
        velocities_eci_km_s=station.velocity_eci_km_s * radius_scale,
        attitudes_quat_bn=None,
        attitude_source_kind="not_required",
        attitude_provider_id=None,
    )
    pattern = TerminalPattern(kind="constant", gain_dbi=30.0)
    config = DirectedLinkConfig(
        analysis_id=link_id,
        link_id=link_id,
        tx_terminal=LinkTerminal(
            terminal_id=f"{asset_id}.tx",
            asset_id=asset_id,
            parent_frame="body",
            quat_parent_from_terminal=(1.0, 0.0, 0.0, 0.0),
            pattern=pattern,
        ),
        rx_terminal=LinkTerminal(
            terminal_id=f"{station_asset_id}.rx",
            asset_id=station_asset_id,
            parent_frame="enu",
            quat_parent_from_terminal=(1.0, 0.0, 0.0, 0.0),
            pattern=pattern,
        ),
        carrier_frequency_hz=2.2e9,
        tx_power_w=10.0,
        data_rate_bps=100.0e6,
        system_noise_temperature_k=500.0,
        required_eb_n0_db=-20.0,
        min_fixed_site_elevation_rad=0.0,
    )
    result = evaluate_directed_link(
        config,
        tx_history=spacecraft,
        rx_history=station,
        frame_context=context,
    )
    output = source_root / link_id
    write_directed_link_artifacts(result, output)
    return output


def generate_source_chain(output_root: Path) -> dict[str, object]:
    destination = output_root.expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"Output root must be absent or empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    sources = destination / "generated_sources"
    sources.mkdir()
    collection_a = _write_collection(
        sources,
        asset_id="SAT-A",
        name="public_source_chain_sat_a",
    )
    collection_b = _write_collection(
        sources,
        asset_id="SAT-B",
        name="public_source_chain_sat_b",
    )
    link_a = _write_link(
        sources,
        asset_id="SAT-A",
        station_asset_id="GS-1",
        start_s=85.0,
        end_s=95.0,
        link_id="sat-a-gs1",
    )
    link_b_contended = _write_link(
        sources,
        asset_id="SAT-B",
        station_asset_id="GS-1",
        start_s=88.0,
        end_s=98.0,
        link_id="sat-b-gs1-contended",
    )
    link_b_alternate = _write_link(
        sources,
        asset_id="SAT-B",
        station_asset_id="GS-2",
        start_s=105.0,
        end_s=115.0,
        link_id="sat-b-gs2",
    )
    def relative(path: Path) -> str:
        return str(path.relative_to(destination))

    plan_payload = {
        "schema_version": "oel.mission_scheduling_source_plan.v1",
        "analysis_id": "public_collection_link_source_chain",
        "epoch_jd_utc": EPOCH_JD_UTC,
        "horizon_start_s": 0.0,
        "horizon_end_s": 120.0,
        "assets": [
            {
                "asset_id": asset_id,
                "storage_capacity_bytes": 150.0e6,
                "initial_storage_bytes": 0.0,
                "energy_budget_wh": 10.0,
                "maximum_payload_duty_cycle": 1.0,
                "maximum_slew_rate_rad_s": None,
                "settling_time_s": 2.0,
            }
            for asset_id in ("SAT-A", "SAT-B")
        ],
        "collection_sources": [
            {
                "source_id": "sat-a-collection",
                "path": relative(collection_a),
                "asset_id": "SAT-A",
                "objective_scale": 1.0,
                "energy_cost_wh": 3.0,
            },
            {
                "source_id": "sat-b-collection",
                "path": relative(collection_b),
                "asset_id": "SAT-B",
                "objective_scale": 1.0,
                "energy_cost_wh": 3.0,
            },
        ],
        "link_sources": [
            {
                "source_id": "sat-a-gs1",
                "path": relative(link_a),
                "asset_id": "SAT-A",
                "station_asset_id": "GS-1",
                "station_id": "GS-1",
                "energy_cost_wh": 1.0,
            },
            {
                "source_id": "sat-b-gs1-contended",
                "path": relative(link_b_contended),
                "asset_id": "SAT-B",
                "station_asset_id": "GS-1",
                "station_id": "GS-1",
                "energy_cost_wh": 1.0,
            },
            {
                "source_id": "sat-b-gs2",
                "path": relative(link_b_alternate),
                "asset_id": "SAT-B",
                "station_asset_id": "GS-2",
                "station_id": "GS-2",
                "energy_cost_wh": 1.0,
            },
        ],
        "require_observation_delivery_by_horizon": True,
        "minimum_selected_observations": 2,
        "maximum_candidates": 18,
    }
    plan_path = destination / "source_plan.json"
    plan_path.write_text(
        json.dumps(plan_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    plan = MissionSchedulingSourcePlan.from_mapping(plan_payload)
    artifacts = build_solve_mission_schedule_from_sources(
        plan,
        base_dir=destination,
        output_dir=destination / "evidence",
    )
    verified = verify_source_built_mission_schedule(artifacts.output_dir)
    return {
        **verified,
        "source_plan": str(plan_path),
        "source_manifest": str(artifacts.manifest_json),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = generate_source_chain(args.output_root)
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, indent=2, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
