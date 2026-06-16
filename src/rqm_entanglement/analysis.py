"""Stable entanglement analysis API for external integrations.

This module exposes :func:`analyze_entanglement`, a deterministic interface that
accepts either:

- a two-qubit pure state vector (shape ``(4,)``),
- a two-qubit unitary / SU(4) matrix (shape ``(4, 4)``), or
- a sequence of two-qubit gates (each shape ``(4, 4)``).

Primary support is two-qubit analysis. Inputs corresponding to more than two
qubits are handled gracefully by returning a stable result shape with notes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NotRequired, TypeAlias, TypedDict, cast

import numpy as np
from numpy.typing import NDArray

from rqm_entanglement.adapters.rqm_core_adapter import su2_from_quaternion_components
from rqm_entanglement.constants import ATOL, CNOT, CZ, I2, ISWAP, SWAP, X, Y, Z
from rqm_entanglement.measures import (
    concurrence_pure,
    entanglement_entropy_pure,
    von_neumann_entropy,
)
from rqm_entanglement.states import density_matrix, normalize_state, reduced_density_matrix
from rqm_entanglement.tensor import local_unitary
from rqm_entanglement.validation import is_unitary


class EntanglementMetric(TypedDict, total=False):
    """Per-pair metric payload in the stable integration schema."""

    pair: list[int]
    metric_name: str
    metric_value: float
    interpretation: NotRequired[str]


class EntanglementAnalysisResult(TypedDict, total=False):
    """Stable library-level response returned by :func:`analyze_entanglement`."""

    has_entangling_gates: bool
    entangled_pairs: list[EntanglementMetric]
    last_entangling_gate: NotRequired[str]
    fidelity_preserved: float | None
    notes: list[str]


class EntanglementAnalysisOptions(TypedDict, total=False):
    """Optional analysis configuration."""

    atol: float
    include_rqm_correlation: bool
    include_interpretation: bool


@dataclass(frozen=True)
class _MetricBundle:
    concurrence: float
    entropy: float
    mutual_information: float
    rqm_correlation: float | None


_GateSequence: TypeAlias = list[tuple[str, NDArray[np.complex128]]]


def _is_finite_array(arr: NDArray[Any]) -> bool:
    return bool(np.isfinite(arr).all())


def _stable_empty_result(notes: list[str]) -> EntanglementAnalysisResult:
    return {
        "has_entangling_gates": False,
        "entangled_pairs": [],
        "fidelity_preserved": None,
        "notes": notes,
    }


def _sanitize_scalar(
    value: float,
    *,
    lower: float,
    upper: float,
    atol: float,
) -> float:
    if not np.isfinite(value):
        return 0.0
    clipped = float(np.clip(value, lower, upper))
    if abs(clipped) <= atol:
        return 0.0
    if abs(clipped - upper) <= atol:
        return float(upper)
    return clipped


def _metric_interpretation(metric_name: str, value: float) -> str:
    if metric_name == "Mutual Information":
        # For two-qubit pure states this lies in [0, 2].
        normalized = value / 2.0
    else:
        normalized = value

    if normalized < 0.05:
        return "near-zero correlation"
    if normalized < 0.5:
        return "intermediate correlation"
    return "strong quantum correlation"


def _format_metric_entries(
    bundle: _MetricBundle,
    *,
    include_rqm_correlation: bool,
    include_interpretation: bool,
    atol: float,
) -> list[EntanglementMetric]:
    entries: list[tuple[str, float]] = [
        ("Concurrence", _sanitize_scalar(bundle.concurrence, lower=0.0, upper=1.0, atol=atol)),
        ("Entropy", _sanitize_scalar(bundle.entropy, lower=0.0, upper=1.0, atol=atol)),
        (
            "Mutual Information",
            _sanitize_scalar(bundle.mutual_information, lower=0.0, upper=2.0, atol=atol),
        ),
    ]

    if include_rqm_correlation and bundle.rqm_correlation is not None:
        entries.append(
            (
                "RQM Correlation",
                _sanitize_scalar(bundle.rqm_correlation, lower=0.0, upper=1.0, atol=atol),
            )
        )

    result: list[EntanglementMetric] = []
    for metric_name, metric_value in sorted(entries, key=lambda x: x[0]):
        row: EntanglementMetric = {
            "pair": [0, 1],
            "metric_name": metric_name,
            "metric_value": metric_value,
        }
        if include_interpretation:
            row["interpretation"] = _metric_interpretation(metric_name, metric_value)
        result.append(row)
    return result


def _build_probe_states() -> list[NDArray[np.complex128]]:
    # Deterministic product probes probing multiple local basis directions.
    zero = np.array([1.0, 0.0], dtype=np.complex128)
    one = np.array([0.0, 1.0], dtype=np.complex128)
    plus = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    plus_i = np.array([1.0, 1.0j], dtype=np.complex128) / np.sqrt(2.0)

    probes_1q = [zero, one, plus, plus_i]
    probes_2q: list[NDArray[np.complex128]] = []
    for left in probes_1q:
        for right in probes_1q:
            probes_2q.append(np.kron(left, right).astype(np.complex128))
    return probes_2q


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


def _single_qubit_gate_matrix(
    gate_name: str,
    params: dict[str, Any],
) -> NDArray[np.complex128] | None:
    name = gate_name.lower()
    if name in {"i", "id", "identity"}:
        return I2
    if name == "x":
        return X
    if name == "y":
        return Y
    if name == "z":
        return Z
    if name == "h":
        h_matrix = (
            np.array(
                [
                    [1.0, 1.0],
                    [1.0, -1.0],
                ],
                dtype=np.complex128,
            )
            / np.sqrt(2.0)
        )
        return cast(NDArray[np.complex128], h_matrix)
    if name == "s":
        return np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=np.complex128)
    if name in {"sdg", "sdag"}:
        return np.array([[1.0, 0.0], [0.0, -1.0j]], dtype=np.complex128)
    if name == "t":
        return np.array([[1.0, 0.0], [0.0, np.exp(1.0j * np.pi / 4.0)]], dtype=np.complex128)
    if name in {"tdg", "tdag"}:
        return np.array([[1.0, 0.0], [0.0, np.exp(-1.0j * np.pi / 4.0)]], dtype=np.complex128)
    if name == "u1q":
        components = _quaternion_components(params)
        if components is None:
            return None
        try:
            return su2_from_quaternion_components(*components)
        except (ImportError, TypeError, ValueError):
            return None
    if name in {"rx", "ry", "rz"}:
        theta = _extract_angle(params)
        if theta is None:
            return None
        generator = {"rx": X, "ry": Y, "rz": Z}[name]
        rotation = np.cos(theta / 2.0) * I2 - 1.0j * np.sin(theta / 2.0) * generator
        return cast(NDArray[np.complex128], rotation)
    return None


def _two_qubit_gate_matrix(
    gate_name: str,
    targets: Sequence[int],
) -> NDArray[np.complex128] | None:
    name = gate_name.lower()
    if len(targets) != 2:
        return None
    left, right = targets
    if name in {"cx", "cnot"}:
        if (left, right) == (0, 1):
            return CNOT
        if (left, right) == (1, 0):
            return SWAP @ CNOT @ SWAP
        return None
    if name == "cz":
        return CZ if {left, right} == {0, 1} else None
    if name == "swap":
        return SWAP if {left, right} == {0, 1} else None
    if name == "iswap":
        return ISWAP if {left, right} == {0, 1} else None
    return None


def _extract_instruction_targets(
    instruction: dict[str, Any],
) -> list[int] | None:
    return _target_indices(instruction.get("targets"))


def _target_indices(items: Any) -> list[int] | None:
    if not isinstance(items, list):
        return None
    indices: list[int] = []
    for target in items:
        if isinstance(target, int):
            indices.append(target)
            continue
        if not isinstance(target, dict):
            return None
        index = target.get("index")
        if not isinstance(index, int):
            return None
        indices.append(index)
    return indices


def _extract_rqm_circuit_sequence(
    circuit_or_unitary: Any,
) -> tuple[_GateSequence, list[str], bool] | None:
    if not isinstance(circuit_or_unitary, dict):
        return None
    if "instructions" not in circuit_or_unitary:
        return None

    notes: list[str] = []
    num_qubits_raw = circuit_or_unitary.get("num_qubits")
    if not isinstance(num_qubits_raw, int):
        notes.append("Circuit payload is missing integer num_qubits; unable to analyze.")
        return [], notes, False
    if num_qubits_raw > 2:
        notes.append("Circuit has >2 qubits; analysis is currently limited to two qubits.")
        return [], notes, False
    if num_qubits_raw < 2:
        notes.append("Circuit has <2 qubits; two-qubit entanglement metrics are not applicable.")
        return [], notes, False

    instructions = circuit_or_unitary.get("instructions")
    if not isinstance(instructions, list):
        notes.append("Circuit payload instructions must be a list.")
        return [], notes, False

    sequence: _GateSequence = []
    for idx, instruction in enumerate(instructions):
        if not isinstance(instruction, dict):
            notes.append(f"Skipping instruction[{idx}] because it is not an object.")
            continue

        gate_object = instruction.get("gate")
        gate_name_raw = gate_object.get("name") if isinstance(gate_object, dict) else None
        if gate_name_raw is None:
            gate_name_raw = instruction.get("name")
        if not isinstance(gate_name_raw, str):
            notes.append(f"Skipping instruction[{idx}] because gate name is missing.")
            continue
        gate_name = gate_name_raw.lower()

        targets = _extract_instruction_targets(instruction)
        if targets is None:
            notes.append(f"Skipping instruction[{idx}] ({gate_name}) due to invalid targets.")
            continue
        controls = _target_indices(instruction.get("controls")) if "controls" in instruction else []
        if controls is None:
            notes.append(f"Skipping instruction[{idx}] ({gate_name}) due to invalid controls.")
            continue
        if any(target not in (0, 1) for target in targets + controls):
            notes.append(
                f"Skipping instruction[{idx}] ({gate_name}) because targets and controls must be qubits 0 or 1."
            )
            continue

        params_dict = _parameter_map(instruction.get("params"))

        gate_matrix: NDArray[np.complex128] | None
        if controls:
            if gate_name in {"cx", "cnot"} and len(controls) == 1 and len(targets) == 1:
                gate_matrix = _two_qubit_gate_matrix("cx", [controls[0], targets[0]])
            elif gate_name == "cz" and len(controls) == 1 and len(targets) == 1:
                gate_matrix = _two_qubit_gate_matrix("cz", [controls[0], targets[0]])
            else:
                gate_matrix = None
            if gate_matrix is None:
                notes.append(
                    f"Skipping unsupported controlled gate '{gate_name}' in instruction[{idx}]."
                )
                continue
        elif len(targets) == 1:
            gate_1q = _single_qubit_gate_matrix(gate_name, params_dict)
            if gate_1q is None:
                notes.append(
                    f"Skipping unsupported single-qubit gate '{gate_name}' in instruction[{idx}]."
                )
                continue
            gate_matrix = (
                local_unitary(gate_1q, I2)
                if targets[0] == 0
                else local_unitary(I2, gate_1q)
            )
        elif len(targets) == 2:
            gate_matrix = _two_qubit_gate_matrix(gate_name, targets)
            if gate_matrix is None:
                notes.append(
                    f"Skipping unsupported two-qubit gate '{gate_name}' in instruction[{idx}]."
                )
                continue
        else:
            notes.append(
                f"Skipping instruction[{idx}] ({gate_name}); only arity 1 or 2 is supported."
            )
            continue

        sequence.append((gate_name, gate_matrix))

    return sequence, notes, True


def _extract_gate_sequence(circuit_or_unitary: Any) -> _GateSequence | None:
    if isinstance(circuit_or_unitary, np.ndarray):
        return None

    gate_like = circuit_or_unitary
    if hasattr(circuit_or_unitary, "gates"):
        gate_like = getattr(circuit_or_unitary, "gates")

    if not isinstance(gate_like, Sequence):
        return None

    sequence: _GateSequence = []
    for idx, item in enumerate(gate_like):
        if isinstance(item, np.ndarray):
            sequence.append((f"gate[{idx}]", item.astype(np.complex128)))
            continue

        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
            sequence.append((item[0], np.asarray(item[1], dtype=np.complex128)))
            continue

        if isinstance(item, dict):
            name = str(item.get("name", f"gate[{idx}]"))
            maybe_matrix = item.get("unitary", item.get("matrix"))
            if maybe_matrix is None:
                return None
            sequence.append((name, np.asarray(maybe_matrix, dtype=np.complex128)))
            continue

        return None
    return sequence


def _entanglement_metrics_for_state(
    psi: NDArray[Any],
    *,
    atol: float,
    include_rqm_correlation: bool,
) -> _MetricBundle:
    psi_clean = normalize_state(np.asarray(psi, dtype=np.complex128), atol=atol)
    concurrence = concurrence_pure(psi_clean)
    entropy = entanglement_entropy_pure(psi_clean, atol=atol)

    rho = density_matrix(psi_clean)
    rho_a = reduced_density_matrix(rho, subsystem=0)
    rho_b = reduced_density_matrix(rho, subsystem=1)
    s_ab = von_neumann_entropy(rho, atol=atol)
    s_a = von_neumann_entropy(rho_a, atol=atol)
    s_b = von_neumann_entropy(rho_b, atol=atol)
    mutual_information = s_a + s_b - s_ab

    # Experimental additive metric: average of normalized concurrence and entropy.
    rqm_correlation = None
    if include_rqm_correlation:
        rqm_correlation = 0.5 * (
            _sanitize_scalar(concurrence, lower=0.0, upper=1.0, atol=atol)
            + _sanitize_scalar(entropy, lower=0.0, upper=1.0, atol=atol)
        )

    return _MetricBundle(
        concurrence=concurrence,
        entropy=entropy,
        mutual_information=mutual_information,
        rqm_correlation=rqm_correlation,
    )


def _entangling_score_from_unitary(
    U: NDArray[np.complex128],
    probes: Sequence[NDArray[np.complex128]],
    *,
    atol: float,
) -> tuple[float, NDArray[np.complex128]]:
    max_conc = 0.0
    best_state = probes[0]
    for probe in probes:
        evolved = U @ probe
        if not _is_finite_array(evolved):
            continue
        try:
            evolved = normalize_state(evolved, atol=atol)
        except ValueError:
            continue
        c = concurrence_pure(evolved)
        if np.isfinite(c) and c > max_conc:
            max_conc = float(c)
            best_state = evolved
    return max_conc, best_state


def _unitary_fidelity_score(U: NDArray[np.complex128], *, atol: float) -> float:
    residual = np.linalg.norm(U.conj().T @ U - np.eye(4, dtype=np.complex128), ord="fro")
    fidelity = 1.0 - float(residual) / 4.0
    return _sanitize_scalar(fidelity, lower=0.0, upper=1.0, atol=atol)


def analyze_entanglement(
    circuit_or_unitary: Any,
    options: EntanglementAnalysisOptions | None = None,
) -> EntanglementAnalysisResult:
    """Analyze two-qubit entanglement with a stable integration-oriented schema.

    Parameters
    ----------
    circuit_or_unitary:
        Accepted inputs:
        - state vector ``(4,)``;
        - unitary matrix ``(4,4)``;
        - sequence of gate matrices ``[(4,4), ...]`` (optionally named tuples
          ``[(name, matrix), ...]`` or dicts with ``{"name": ..., "unitary": ...}``).
    options:
        Optional settings. Supported keys:
        - ``atol``: numerical tolerance for clamping / zero-testing.
        - ``include_rqm_correlation``: include experimental RQM correlation metric.
        - ``include_interpretation``: add qualitative interpretation strings.
    """
    config = options or {}
    atol = float(config.get("atol", ATOL))
    include_rqm_correlation = bool(config.get("include_rqm_correlation", True))
    include_interpretation = bool(config.get("include_interpretation", True))

    notes: list[str] = []
    probes = _build_probe_states()

    # Case 1: RQM circuit payload.
    rqm_circuit_parse = _extract_rqm_circuit_sequence(circuit_or_unitary)
    gate_sequence: _GateSequence | None = None
    if rqm_circuit_parse is not None:
        parsed_sequence, parse_notes, can_proceed = rqm_circuit_parse
        notes.extend(parse_notes)
        if not can_proceed:
            return _stable_empty_result(notes)
        gate_sequence = parsed_sequence

    # Case 2: generic sequence / circuit-like input.
    if gate_sequence is None:
        gate_sequence = _extract_gate_sequence(circuit_or_unitary)
    if gate_sequence is not None:
        if len(gate_sequence) == 0:
            notes.append("Empty gate sequence provided; no entangling action detected.")
            return {
                "has_entangling_gates": False,
                "entangled_pairs": _format_metric_entries(
                    _MetricBundle(0.0, 0.0, 0.0, 0.0 if include_rqm_correlation else None),
                    include_rqm_correlation=include_rqm_correlation,
                    include_interpretation=include_interpretation,
                    atol=atol,
                ),
                "fidelity_preserved": 1.0,
                "notes": notes,
            }

        has_entangling = False
        last_entangling_gate: str | None = None
        max_score_overall = 0.0
        representative_state = probes[0]
        accumulated = np.eye(4, dtype=np.complex128)

        for gate_name, gate in gate_sequence:
            if gate.shape != (4, 4):
                if gate.ndim == 2 and gate.shape[0] == gate.shape[1] and gate.shape[0] > 4:
                    notes.append(
                        "Detected >2-qubit gate in sequence; analysis is limited to two qubits."
                    )
                    return _stable_empty_result(notes)
                notes.append(f"Skipping invalid gate shape {gate.shape} for {gate_name}.")
                continue
            if not _is_finite_array(gate):
                notes.append(f"Skipping non-finite gate entries for {gate_name}.")
                continue

            score, state = _entangling_score_from_unitary(gate, probes, atol=atol)
            if score > max_score_overall:
                max_score_overall = score
                representative_state = state
            if score > max(atol * 10.0, 1e-10):
                has_entangling = True
                last_entangling_gate = gate_name

            accumulated = gate @ accumulated

        fidelity = _unitary_fidelity_score(accumulated, atol=atol)
        metric_bundle = _entanglement_metrics_for_state(
            representative_state,
            atol=atol,
            include_rqm_correlation=include_rqm_correlation,
        )
        result: EntanglementAnalysisResult = {
            "has_entangling_gates": has_entangling,
            "entangled_pairs": _format_metric_entries(
                metric_bundle,
                include_rqm_correlation=include_rqm_correlation,
                include_interpretation=include_interpretation,
                atol=atol,
            ),
            "fidelity_preserved": fidelity,
            "notes": notes,
        }
        if last_entangling_gate is not None:
            result["last_entangling_gate"] = last_entangling_gate
        return result

    # Case 2: ndarray input (state or unitary).
    arr = np.asarray(circuit_or_unitary)
    if arr.ndim == 1:
        if arr.shape == (4,):
            if not _is_finite_array(arr):
                notes.append("Input state contains non-finite amplitudes; metrics sanitized.")
                return _stable_empty_result(notes)
            metric_bundle = _entanglement_metrics_for_state(
                arr,
                atol=atol,
                include_rqm_correlation=include_rqm_correlation,
            )
            concurrence = _sanitize_scalar(
                metric_bundle.concurrence,
                lower=0.0,
                upper=1.0,
                atol=atol,
            )
            return {
                "has_entangling_gates": bool(concurrence > atol),
                "entangled_pairs": _format_metric_entries(
                    metric_bundle,
                    include_rqm_correlation=include_rqm_correlation,
                    include_interpretation=include_interpretation,
                    atol=atol,
                ),
                "fidelity_preserved": None,
                "notes": notes,
            }

        if arr.shape[0] > 4 and (arr.shape[0] & (arr.shape[0] - 1)) == 0:
            notes.append("Input state appears to include >2 qubits; two-qubit analysis only.")
            return _stable_empty_result(notes)
        notes.append(f"Unsupported state shape {arr.shape}; expected (4,).")
        return _stable_empty_result(notes)

    if arr.ndim == 2:
        if arr.shape == (4, 4):
            U = arr.astype(np.complex128)
            if not _is_finite_array(U):
                notes.append("Input unitary contains non-finite entries; metrics sanitized.")
                return _stable_empty_result(notes)
            if not is_unitary(U, atol=atol):
                notes.append(
                    "Input matrix is not unitary within tolerance; results are best-effort."
                )
            score, representative = _entangling_score_from_unitary(U, probes, atol=atol)
            metric_bundle = _entanglement_metrics_for_state(
                representative,
                atol=atol,
                include_rqm_correlation=include_rqm_correlation,
            )
            has_entangling = bool(score > max(atol * 10.0, 1e-10))
            result = {
                "has_entangling_gates": has_entangling,
                "entangled_pairs": _format_metric_entries(
                    metric_bundle,
                    include_rqm_correlation=include_rqm_correlation,
                    include_interpretation=include_interpretation,
                    atol=atol,
                ),
                "fidelity_preserved": _unitary_fidelity_score(U, atol=atol),
                "notes": notes,
            }
            if has_entangling:
                result["last_entangling_gate"] = "input_unitary"
            return result

        if arr.shape[0] == arr.shape[1] and arr.shape[0] > 4:
            notes.append("Input operator appears to include >2 qubits; two-qubit analysis only.")
            return _stable_empty_result(notes)
        notes.append(f"Unsupported matrix shape {arr.shape}; expected (4,4).")
        return _stable_empty_result(notes)

    notes.append("Unsupported input type; expected state, unitary, or gate sequence.")
    return _stable_empty_result(notes)
