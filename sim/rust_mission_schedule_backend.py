"""Optional exact native subset search for validated mission schedules."""

from __future__ import annotations

from sim.rust_orbit_backend import _extension


def search(problem, candidates, transition_required, *, epsilon: float) -> tuple[int | None, int]:
    native = getattr(_extension(), "mission_schedule_search", None)
    if native is None:
        raise RuntimeError("Rust mission schedule backend requires mission_schedule_search")
    assets = sorted(problem.assets, key=lambda item: item.asset_id)
    asset_index = {item.asset_id: index for index, item in enumerate(assets)}
    asset_rows = [
        [item.storage_capacity_bytes, item.initial_storage_bytes, item.energy_budget_wh,
         item.maximum_payload_duty_cycle]
        for item in assets
    ]
    kind_code = {"observation": 0, "downlink": 1, "other": 2}
    opportunity_rows = [
        [asset_index[item.asset_id], kind_code[item.kind], item.start_s, item.end_s,
         item.objective_value, item.energy_cost_wh, item.data_volume_bytes,
         item.downlink_capacity_bytes]
        for item in candidates
    ]
    orders = [
        sorted((index for index, item in enumerate(candidates) if item.asset_id == asset.asset_id),
               key=lambda index: (candidates[index].start_s, candidates[index].end_s,
                                  candidates[index].opportunity_id))
        for asset in assets
    ]
    station_conflicts = [0] * len(candidates)
    transition_conflicts = [0] * len(candidates)
    for left_index, left in enumerate(candidates):
        for right_index in range(left_index + 1, len(candidates)):
            right = candidates[right_index]
            if (left.kind == right.kind == "downlink" and left.station_id == right.station_id
                    and left.start_s < right.end_s - epsilon and right.start_s < left.end_s - epsilon):
                station_conflicts[left_index] |= 1 << right_index
            if left.asset_id != right.asset_id:
                continue
            first_index, second_index = sorted((left_index, right_index),
                key=lambda index: (candidates[index].start_s, candidates[index].end_s,
                                   candidates[index].opportunity_id))
            first, second = candidates[first_index], candidates[second_index]
            asset = assets[asset_index[first.asset_id]]
            if (second.start_s < first.end_s - epsilon
                    or second.start_s - first.end_s + epsilon < transition_required(first, second, asset)):
                transition_conflicts[first_index] |= 1 << second_index
    return native(
        asset_rows, opportunity_rows, orders, station_conflicts, transition_conflicts,
        problem.minimum_selected_observations, problem.require_observation_delivery_by_horizon,
        problem.horizon_end_s - problem.horizon_start_s,
    )
