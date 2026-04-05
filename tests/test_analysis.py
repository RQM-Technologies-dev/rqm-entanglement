"""Tests for the stable entanglement analysis API."""

import numpy as np
import pytest

from rqm_entanglement import CNOT, I4, X, analyze_entanglement, apply_unitary, ket00, local_unitary
from rqm_entanglement.states import state_from_amplitudes


def _metric_value(result: dict, metric_name: str) -> float:
    for metric in result["entangled_pairs"]:
        if metric["metric_name"] == metric_name:
            return float(metric["metric_value"])
    raise AssertionError(f"Metric '{metric_name}' not found in result.")


def test_analyze_entanglement_bell_state_high_metrics() -> None:
    psi_bell = state_from_amplitudes(1, 0, 0, 1)
    result = analyze_entanglement(psi_bell)

    assert result["has_entangling_gates"] is True
    assert _metric_value(result, "Concurrence") == pytest.approx(1.0, abs=1e-9)
    assert _metric_value(result, "Entropy") == pytest.approx(1.0, abs=1e-9)
    # For pure Bell states I(A:B) = 2 bits.
    assert _metric_value(result, "Mutual Information") == pytest.approx(2.0, abs=1e-9)


def test_analyze_entanglement_product_state_low_metrics() -> None:
    result = analyze_entanglement(ket00())

    assert result["has_entangling_gates"] is False
    assert _metric_value(result, "Concurrence") == pytest.approx(0.0, abs=1e-12)
    assert _metric_value(result, "Entropy") == pytest.approx(0.0, abs=1e-12)
    assert _metric_value(result, "Mutual Information") == pytest.approx(0.0, abs=1e-12)


def test_analyze_entanglement_partial_state_intermediate_metrics() -> None:
    # psi = cos(theta/2)|00> + sin(theta/2)|11>, concurrence = sin(theta)
    theta = np.pi / 6
    psi = state_from_amplitudes(np.cos(theta / 2), 0, 0, np.sin(theta / 2))

    result = analyze_entanglement(psi)
    expected_c = np.sin(theta)
    # Entropy in bits for Schmidt values [cos(theta/2), sin(theta/2)].
    p = np.cos(theta / 2) ** 2
    expected_s = -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))

    assert result["has_entangling_gates"] is True
    assert _metric_value(result, "Concurrence") == pytest.approx(expected_c, abs=1e-9)
    assert _metric_value(result, "Entropy") == pytest.approx(expected_s, abs=1e-9)
    assert _metric_value(result, "Mutual Information") == pytest.approx(2.0 * expected_s, abs=1e-9)


def test_analyze_entanglement_non_entangling_gate_sequence() -> None:
    # Local-only sequence should be reported as non-entangling.
    local_x = local_unitary(X, X)
    seq = [("first_local", I4), ("second_local", local_x)]

    result = analyze_entanglement(seq)

    assert result["has_entangling_gates"] is False
    assert "last_entangling_gate" not in result
    assert _metric_value(result, "Concurrence") == pytest.approx(0.0, abs=1e-12)


def test_analyze_entanglement_entangling_gate_sequence_tracks_last_gate() -> None:
    psi0 = ket00()
    # H on qubit 0 from RY(pi/2) up to phase. This prepares a Bell pair after CNOT.
    ry_pi_over_2 = np.array(
        [
            [1 / np.sqrt(2), -1 / np.sqrt(2)],
            [1 / np.sqrt(2), 1 / np.sqrt(2)],
        ],
        dtype=np.complex128,
    )
    prep = local_unitary(ry_pi_over_2, np.eye(2, dtype=np.complex128))
    _ = apply_unitary(prep, psi0)  # sanity-check shape/unitarity path

    seq = [("prep_local", prep), ("cx", CNOT)]
    result = analyze_entanglement(seq)
    assert result["has_entangling_gates"] is True
    assert result["last_entangling_gate"] == "cx"


def test_analyze_entanglement_more_than_two_qubits_graceful() -> None:
    psi_3q = np.zeros((8,), dtype=np.complex128)
    psi_3q[0] = 1.0

    result = analyze_entanglement(psi_3q)

    assert result["has_entangling_gates"] is False
    assert result["entangled_pairs"] == []
    assert result["fidelity_preserved"] is None
    assert result["notes"]
    assert any(">2 qubits" in note for note in result["notes"])


def test_analyze_entanglement_deterministic_for_identical_input() -> None:
    psi = state_from_amplitudes(1, 0, 0, 1)
    first = analyze_entanglement(psi)
    second = analyze_entanglement(psi)
    assert first == second

