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
from typing import Any, NotRequired, TypeAlias, TypedDict

import numpy as np
from numpy.typing import NDArray

from rqm_entanglement.constants import ATOL
from rqm_entanglement.measures import (
    concurrence_pure,
    entanglement_entropy_pure,
    von_neumann_entropy,
)
from rqm_entanglement.states import density_matrix, normalize_state, reduced_density_matrix
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

    # Case 1: sequence / circuit-like input.
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
