"""Canonical coupling / entanglement analysis for RQM circuit payloads.

This module owns nonlocal circuit analysis for the RQM stack.  It accepts the
public ``rqm-circuits``-style wire shape used by ``quantum-compiler-api`` and Studio, keeps
qualitative gate detection separate from measured state analysis, and uses the
``rqm-entanglement`` basis convention: ``|00>, |01>, |10>, |11>`` with qubit 0
as the left / more-significant qubit.
"""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict

import numpy as np
from numpy.typing import NDArray

from rqm_entanglement.adapters.rqm_core_adapter import su2_from_quaternion_components
from rqm_entanglement.constants import ATOL, CNOT, CZ, I2, ISWAP, SWAP, X, Y, Z
from rqm_entanglement.measures import concurrence_pure, entanglement_entropy_pure
from rqm_entanglement.tensor import local_unitary


class CouplingMetric(TypedDict, total=False):
    """Single measured metric for a qubit pair."""

    pair: tuple[int, int]
    metric_name: str
    value: float
    normalized_value: float | None
    interpretation: str | None


class CouplingAnalysisResult(TypedDict, total=False):
    """Stable coupling-analysis contract shared by API and Studio."""

    mode: str
    provenance: str
    qubit_count: int
    analyzed_pairs: list[tuple[int, int]]
    has_entangling_gates: bool
    entangling_gate_count: int
    entangling_gates_seen: list[str]
    last_entangling_gate: str | None
    is_entangled: bool | None
    pair_metrics: list[CouplingMetric]
    fidelity_preserved: float | None
    notes: list[str]
    limitations: list[str]


class PreservationAnalysisResult(TypedDict):
    """Before / after coupling preservation contract."""

    fidelity_preserved: float | None
    preserved_entanglement_structure: bool | None
    original_coupling: CouplingAnalysisResult
    optimized_coupling: CouplingAnalysisResult
    notes: list[str]
    limitations: list[str]


class CouplingAnalysisOptions(TypedDict, total=False):
    """Optional coupling-analysis controls."""

    pair_filter: list[tuple[int, int]]
    metric_preference: list[str]
    initial_state: str
    allow_qualitative_fallback: bool
    atol: float


class _ParsedInstruction(TypedDict):
    name: str
    targets: list[int]
    controls: NotRequired[list[int]]
    params: NotRequired[dict[str, Any]]


_SUPPORTED_MEASURED_GATES = {
    "i",
    "id",
    "identity",
    "x",
    "y",
    "z",
    "h",
    "s",
    "sdg",
    "sdag",
    "t",
    "tdg",
    "tdag",
    "u1q",
    "rx",
    "ry",
    "rz",
    "cx",
    "cnot",
    "cz",
    "swap",
    "iswap",
}

_TWO_QUBIT_GATES = {"cx", "cnot", "cy", "cz", "swap", "iswap"}
_MEASURED_LIMITATION = (
    "Measured entanglement analysis is currently limited to supported ideal "
    "2-qubit circuits."
)
_QUALITATIVE_NOTE = "Qualitative gate-based coupling detection only."


def _extract_angle(params: dict[str, Any]) -> float | None:
    for key in ("theta", "angle", "radians"):
        value = params.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _parameter_map(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if not isinstance(raw, list):
        return {}

    params: dict[str, Any] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str):
            continue
        params[name] = item.get("value")
    return params


def _quaternion_components(params: dict[str, Any]) -> tuple[float, float, float, float] | None:
    values: list[float] = []
    for name in ("w", "x", "y", "z"):
        value = params.get(name)
        if not isinstance(value, (int, float)):
            return None
        values.append(float(value))
    return values[0], values[1], values[2], values[3]


def _single_qubit_gate_matrix(name: str, params: dict[str, Any]) -> NDArray[np.complex128] | None:
    gate = name.lower()
    if gate in {"i", "id", "identity"}:
        return I2
    if gate == "x":
        return X
    if gate == "y":
        return Y
    if gate == "z":
        return Z
    if gate == "h":
        return np.asarray(
            np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / np.sqrt(2.0),
            dtype=np.complex128,
        )
    if gate == "s":
        return np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=np.complex128)
    if gate in {"sdg", "sdag"}:
        return np.array([[1.0, 0.0], [0.0, -1.0j]], dtype=np.complex128)
    if gate == "t":
        return np.array([[1.0, 0.0], [0.0, np.exp(1.0j * np.pi / 4.0)]], dtype=np.complex128)
    if gate in {"tdg", "tdag"}:
        return np.array([[1.0, 0.0], [0.0, np.exp(-1.0j * np.pi / 4.0)]], dtype=np.complex128)
    if gate == "u1q":
        components = _quaternion_components(params)
        if components is None:
            return None
        try:
            return su2_from_quaternion_components(*components)
        except (ImportError, TypeError, ValueError):
            return None
    if gate in {"rx", "ry", "rz"}:
        theta = _extract_angle(params)
        if theta is None:
            return None
        generator = {"rx": X, "ry": Y, "rz": Z}[gate]
        return np.asarray(
            np.cos(theta / 2.0) * I2 - 1.0j * np.sin(theta / 2.0) * generator,
            dtype=np.complex128,
        )
    return None


def _two_qubit_gate_matrix(name: str, qubits: tuple[int, int]) -> NDArray[np.complex128] | None:
    gate = name.lower()
    left, right = qubits
    if gate in {"cx", "cnot"}:
        if (left, right) == (0, 1):
            return CNOT
        if (left, right) == (1, 0):
            return SWAP @ CNOT @ SWAP
        return None
    if gate == "cz":
        return CZ if {left, right} == {0, 1} else None
    if gate == "swap":
        return SWAP if {left, right} == {0, 1} else None
    if gate == "iswap":
        return ISWAP if {left, right} == {0, 1} else None
    return None


def _target_indices(items: Any) -> list[int] | None:
    if not isinstance(items, list):
        return None
    indices: list[int] = []
    for item in items:
        if isinstance(item, int):
            indices.append(item)
            continue
        if not isinstance(item, dict):
            return None
        index = item.get("index")
        if not isinstance(index, int):
            return None
        indices.append(index)
    return indices


def _parse_instructions(circuit: dict[str, Any]) -> tuple[list[_ParsedInstruction], list[str]]:
    raw = circuit.get("instructions")
    notes: list[str] = []
    parsed: list[_ParsedInstruction] = []
    if not isinstance(raw, list):
        return parsed, ["Circuit payload instructions must be a list."]

    for idx, instruction in enumerate(raw):
        if not isinstance(instruction, dict):
            notes.append(f"Skipping instruction[{idx}] because it is not an object.")
            continue
        gate_obj = instruction.get("gate")
        gate_name = gate_obj.get("name") if isinstance(gate_obj, dict) else instruction.get("name")
        if not isinstance(gate_name, str):
            notes.append(f"Skipping instruction[{idx}] because gate name is missing.")
            continue

        targets = _target_indices(instruction.get("targets"))
        controls = _target_indices(instruction.get("controls")) or []
        if targets is None:
            notes.append(f"Skipping instruction[{idx}] ({gate_name}) due to invalid targets.")
            continue

        parsed.append(
            {
                "name": gate_name.lower(),
                "targets": targets,
                "controls": controls,
                "params": _parameter_map(instruction.get("params")),
            }
        )
    return parsed, notes


def _entangling_qubits(instruction: _ParsedInstruction) -> list[int]:
    name = instruction["name"].lower()
    if name not in _TWO_QUBIT_GATES:
        return []
    controls = instruction.get("controls", [])
    targets = instruction["targets"]
    if controls:
        return controls + targets
    return targets


def _describe_entangling_gate(name: str, qubits: list[int]) -> str:
    label = name.upper()
    if len(qubits) == 2:
        return f"{label} q{qubits[0]}->q{qubits[1]}"
    return f"{label} " + ",".join(f"q{q}" for q in qubits)


def _pairs_from_entangling(
    entangling: list[tuple[str, list[int]]],
    pair_filter: list[tuple[int, int]] | None,
) -> list[tuple[int, int]]:
    filter_set = (
        frozenset((min(a, b), max(a, b)) for a, b in pair_filter)
        if pair_filter is not None
        else None
    )
    seen: set[tuple[int, int]] = set()
    pairs: list[tuple[int, int]] = []
    for _, qubits in entangling:
        if len(qubits) < 2:
            continue
        pair = (min(qubits[0], qubits[1]), max(qubits[0], qubits[1]))
        if filter_set is not None and pair not in filter_set:
            continue
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)
    return pairs


def _initial_state(name: str | None) -> NDArray[np.complex128] | None:
    state_name = name or "zero"
    if state_name in {"zero", "00", "|00>"}:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
    if state_name in {"01", "|01>"}:
        return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.complex128)
    if state_name in {"10", "|10>"}:
        return np.array([0.0, 0.0, 1.0, 0.0], dtype=np.complex128)
    if state_name in {"11", "|11>"}:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.complex128)
    return None


def _instruction_unitary(instruction: _ParsedInstruction) -> NDArray[np.complex128] | None:
    name = instruction["name"].lower()
    targets = instruction["targets"]
    controls = instruction.get("controls", [])
    params = instruction.get("params", {})

    if name not in _SUPPORTED_MEASURED_GATES:
        return None
    if controls:
        if name in {"cx", "cnot"} and len(controls) == 1 and len(targets) == 1:
            return _two_qubit_gate_matrix("cx", (controls[0], targets[0]))
        if name == "cz" and len(controls) == 1 and len(targets) == 1:
            return _two_qubit_gate_matrix("cz", (controls[0], targets[0]))
        return None
    if len(targets) == 1:
        gate = _single_qubit_gate_matrix(name, params)
        if gate is None:
            return None
        return local_unitary(gate, I2) if targets[0] == 0 else local_unitary(I2, gate)
    if len(targets) == 2:
        return _two_qubit_gate_matrix(name, (targets[0], targets[1]))
    return None


def _simulate_final_state(
    instructions: list[_ParsedInstruction],
    initial_state: str | None,
) -> NDArray[np.complex128] | None:
    state = _initial_state(initial_state)
    if state is None:
        return None
    for instruction in instructions:
        if any(q not in (0, 1) for q in instruction["targets"] + instruction.get("controls", [])):
            return None
        unitary = _instruction_unitary(instruction)
        if unitary is None:
            return None
        state = unitary @ state
    return state


def _metric_interpretation(metric_name: str, value: float, atol: float) -> str:
    if metric_name == "concurrence":
        if value <= atol:
            return "separable"
        if value < 0.3:
            return "weakly entangled"
        if value < 0.7:
            return "moderately entangled"
        if value < 1.0 - atol:
            return "strongly entangled"
        return "maximally entangled"
    if value <= atol:
        return "no entanglement"
    if value >= 1.0 - atol:
        return "maximal single-qubit entanglement entropy"
    return f"partial entanglement entropy ({value:.3f} bits)"


def _pair_metrics(
    state: NDArray[np.complex128],
    pairs: list[tuple[int, int]],
    metric_preference: list[str] | None,
    atol: float,
) -> list[CouplingMetric]:
    requested = metric_preference or ["concurrence", "entropy"]
    metrics: list[CouplingMetric] = []
    concurrence = float(np.clip(concurrence_pure(state), 0.0, 1.0))
    entropy = float(np.clip(entanglement_entropy_pure(state, atol=atol), 0.0, 1.0))
    values = {
        "concurrence": concurrence,
        "entropy": entropy,
    }
    for pair in pairs:
        for metric_name in requested:
            value = values.get(metric_name)
            if value is None:
                continue
            rounded = round(value, 10)
            metrics.append(
                {
                    "pair": pair,
                    "metric_name": metric_name,
                    "value": rounded,
                    "normalized_value": rounded,
                    "interpretation": _metric_interpretation(metric_name, rounded, atol),
                }
            )
    return metrics


def _state_fidelity(left: NDArray[np.complex128], right: NDArray[np.complex128]) -> float:
    overlap_magnitude = float(np.abs(np.vdot(left, right)))
    return round(min(1.0, overlap_magnitude**2), 10)


def analyze_circuit_coupling(
    circuit: dict[str, Any],
    options: CouplingAnalysisOptions | None = None,
) -> CouplingAnalysisResult:
    """Analyze coupling and entanglement for an RQM circuit payload."""
    config = options or {}
    atol = float(config.get("atol", ATOL))
    pair_filter = config.get("pair_filter")
    metric_preference = config.get("metric_preference")
    initial_state = config.get("initial_state")
    allow_qualitative_fallback = bool(config.get("allow_qualitative_fallback", True))

    num_qubits = circuit.get("num_qubits")
    if not isinstance(num_qubits, int):
        raise ValueError("Circuit payload is missing integer num_qubits.")

    instructions, notes = _parse_instructions(circuit)
    entangling = [(inst["name"].upper(), _entangling_qubits(inst)) for inst in instructions]
    entangling = [(name, qubits) for name, qubits in entangling if len(qubits) >= 2]
    gate_names_seen = list(dict.fromkeys(name for name, _ in entangling))
    last_gate = _describe_entangling_gate(*entangling[-1]) if entangling else None
    analyzed_pairs = _pairs_from_entangling(entangling, pair_filter)

    qualitative: CouplingAnalysisResult = {
        "mode": "qualitative",
        "provenance": "parser",
        "qubit_count": num_qubits,
        "analyzed_pairs": analyzed_pairs,
        "has_entangling_gates": bool(entangling),
        "entangling_gate_count": len(entangling),
        "entangling_gates_seen": gate_names_seen,
        "last_entangling_gate": last_gate,
        "is_entangled": None,
        "pair_metrics": [],
        "fidelity_preserved": None,
        "notes": notes + [_QUALITATIVE_NOTE],
        "limitations": [_MEASURED_LIMITATION],
    }

    if num_qubits != 2:
        if not allow_qualitative_fallback:
            raise ValueError(
                "Measured entanglement analysis is not available for this circuit "
                "and allow_qualitative_fallback is False."
            )
        return qualitative

    state = _simulate_final_state(instructions, initial_state)
    if state is None:
        if not allow_qualitative_fallback:
            raise ValueError(
                "Measured entanglement analysis is not available for this circuit "
                "and allow_qualitative_fallback is False."
            )
        return qualitative

    concurrence = float(concurrence_pure(state))
    return {
        "mode": "measured",
        "provenance": "rqm-entanglement",
        "qubit_count": num_qubits,
        "analyzed_pairs": analyzed_pairs,
        "has_entangling_gates": bool(entangling),
        "entangling_gate_count": len(entangling),
        "entangling_gates_seen": gate_names_seen,
        "last_entangling_gate": last_gate,
        "is_entangled": concurrence > atol,
        "pair_metrics": _pair_metrics(state, analyzed_pairs, metric_preference, atol),
        "fidelity_preserved": None,
        "notes": notes,
        "limitations": [],
    }


def analyze_optimization_preservation(
    original_circuit: dict[str, Any],
    optimized_circuit: dict[str, Any],
    options: CouplingAnalysisOptions | None = None,
) -> PreservationAnalysisResult:
    """Compare coupling preservation across two RQM circuit payloads."""
    original = analyze_circuit_coupling(original_circuit, options)
    optimized = analyze_circuit_coupling(optimized_circuit, options)

    config = options or {}
    original_state = None
    optimized_state = None
    if original.get("mode") == "measured" and optimized.get("mode") == "measured":
        original_instructions, _ = _parse_instructions(original_circuit)
        optimized_instructions, _ = _parse_instructions(optimized_circuit)
        original_state = _simulate_final_state(original_instructions, config.get("initial_state"))
        optimized_state = _simulate_final_state(optimized_instructions, config.get("initial_state"))

    fidelity_preserved = (
        _state_fidelity(original_state, optimized_state)
        if original_state is not None and optimized_state is not None
        else None
    )
    preserved = None
    if original.get("is_entangled") is not None and optimized.get("is_entangled") is not None:
        preserved = original.get("is_entangled") == optimized.get("is_entangled")

    notes: list[str] = []
    limitations: list[str] = []
    if original.get("mode") != "measured" or optimized.get("mode") != "measured":
        notes.append(
            "Preservation assessment is qualitative only because at least one circuit "
            "does not support measured entanglement analysis."
        )
        limitations.append(_MEASURED_LIMITATION)

    return {
        "fidelity_preserved": fidelity_preserved,
        "preserved_entanglement_structure": preserved,
        "original_coupling": original,
        "optimized_coupling": optimized,
        "notes": notes,
        "limitations": limitations,
    }


__all__ = [
    "CouplingAnalysisOptions",
    "CouplingAnalysisResult",
    "CouplingMetric",
    "PreservationAnalysisResult",
    "analyze_circuit_coupling",
    "analyze_optimization_preservation",
]
