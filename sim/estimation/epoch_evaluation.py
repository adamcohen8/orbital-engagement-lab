from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from sim.scenarios import ScenarioArtifact


def evaluate_artifact_at_epochs(
    artifact: ScenarioArtifact,
    *,
    object_id: str,
    epochs_s: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a validated OEL scenario at exact, monotonically increasing epochs.

    This deliberately uses ``SimulationSession.step`` instead of interpolating a
    fixed-cadence output history. The session continues to use the scenario's
    configured deterministic dynamics and internal force-model substeps; only
    the outer observation epochs are variable.
    """

    epochs = np.asarray(epochs_s, dtype=float).reshape(-1)
    if epochs.size == 0:
        return epochs, np.empty((0, 6), dtype=float)
    if not np.all(np.isfinite(epochs)):
        raise ValueError("observation epochs must be finite.")
    if np.any(epochs < -1.0e-12):
        raise ValueError("observation epochs must be non-negative.")
    if np.any(np.diff(epochs) <= 0.0):
        raise ValueError("observation epochs must be strictly increasing.")

    from sim.api import SimulationSession

    session = SimulationSession.from_config(artifact, history_mode="dynamic")
    initial = session.reset()
    if initial is None:
        raise ValueError("exact-epoch evaluation requires a single-run scenario.")

    states: list[np.ndarray] = []
    current_time = float(initial.time_s)
    for epoch in epochs:
        target = float(epoch)
        if target < current_time - 1.0e-9:
            raise ValueError("observation epochs precede the active simulation epoch.")
        if target > current_time + 1.0e-12:
            snapshot = session.step(target - current_time)
        else:
            snapshot = initial
        current_time = float(snapshot.time_s)
        if abs(current_time - target) > max(1.0e-10, 1.0e-12 * max(abs(target), 1.0)):
            raise ValueError(
                f"scenario ended before requested observation epoch: requested={target}, reached={current_time}."
            )
        if object_id not in snapshot.truth:
            raise ValueError(f"OEL snapshot did not include object_id '{object_id}'.")
        state = np.asarray(snapshot.truth[object_id], dtype=float).reshape(-1)
        if state.size < 6 or not np.all(np.isfinite(state[:6])):
            raise ValueError(f"OEL snapshot for object_id '{object_id}' did not contain a finite Cartesian state.")
        states.append(state[:6].copy())
        initial = snapshot

    return epochs.copy(), np.vstack(states)


def exact_epoch_provenance(epochs_s: Sequence[float] | np.ndarray) -> dict[str, Any]:
    epochs = np.asarray(epochs_s, dtype=float).reshape(-1)
    return {
        "schema_version": 1,
        "method": "simulation_session_variable_step_exact",
        "interpolation_used": False,
        "epoch_count": int(epochs.size),
        "first_epoch_s": float(epochs[0]) if epochs.size else None,
        "last_epoch_s": float(epochs[-1]) if epochs.size else None,
    }
