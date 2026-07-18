# ruff: noqa: F401,F403,F405,I001
from .training_models import *
from .training_geometry import *
from .coaching import *

def _approach_gate_status(gates: tuple[ApproachGateConfig, ...], relative_ric_state: np.ndarray) -> dict[str, tuple[str, ...]]:
    satisfied: list[str] = []
    violated: list[str] = []
    missed: list[str] = []
    required_violated: list[str] = []
    required_missed: list[str] = []
    for gate in gates:
        near = gate.samples_near_gate(relative_ric_state)
        ok = gate.samples_satisfying_gate(relative_ric_state)
        if bool(np.any(ok)):
            satisfied.append(gate.name)
        elif bool(np.any(near)):
            violated.append(gate.name)
            if gate.required:
                required_violated.append(gate.name)
        else:
            missed.append(gate.name)
            if gate.required:
                required_missed.append(gate.name)
    return {
        "satisfied": tuple(satisfied),
        "violated": tuple(violated),
        "missed": tuple(missed),
        "required_violated": tuple(required_violated),
        "required_missed": tuple(required_missed),
    }


def _inspection_gate_status(gates: tuple[InspectionGateConfig, ...], relative_ric_state: np.ndarray) -> dict[str, Any]:
    if not gates:
        return {"satisfied": (), "completed_idx": None}
    rel = np.array(relative_ric_state, dtype=float)
    if rel.ndim == 1:
        rel = rel.reshape(1, -1)
    if rel.shape[1] < 6:
        raise ValueError("relative_ric_state must contain RIC position and velocity.")
    satisfied: list[str] = []
    completed_idx: int | None = None
    for sample_idx in range(rel.shape[0]):
        if len(satisfied) >= len(gates):
            break
        for gate in gates:
            if gate.name in satisfied:
                continue
            current_hits_gate = bool(gate.samples_satisfying_gate(rel[sample_idx : sample_idx + 1])[0])
            segment_hits_gate = bool(
                sample_idx > 0 and gate.segment_satisfies_gate(rel[sample_idx - 1], rel[sample_idx])
            )
            if current_hits_gate or segment_hits_gate:
                satisfied.append(gate.name)
                if len(satisfied) >= len(gates):
                    completed_idx = sample_idx
                    break
    return {"satisfied": tuple(satisfied), "completed_idx": completed_idx}

__all__ = [name for name in globals() if not name.startswith("__")]
