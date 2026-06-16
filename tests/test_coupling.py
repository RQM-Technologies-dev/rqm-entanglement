"""Tests for the canonical coupling-analysis API."""

from rqm_entanglement import analyze_circuit_coupling, analyze_optimization_preservation


def _bell_payload() -> dict:
    return {
        "schema_version": "0.2",
        "num_qubits": 2,
        "instructions": [
            {"gate": {"name": "h", "arity": 1}, "targets": [{"index": 0}]},
            {
                "gate": {"name": "cx", "arity": 2},
                "targets": [{"index": 1}],
                "controls": [{"index": 0}],
            },
        ],
    }


def test_bell_circuit_is_measured_by_rqm_entanglement() -> None:
    result = analyze_circuit_coupling(_bell_payload())

    assert result["mode"] == "measured"
    assert result["provenance"] == "rqm-entanglement"
    assert result["is_entangled"] is True
    assert result["analyzed_pairs"] == [(0, 1)]
    assert result["entangling_gates_seen"] == ["CX"]
    assert result["pair_metrics"][0]["value"] == 1.0


def test_cnot_on_zero_state_is_measured_but_separable() -> None:
    payload = {
        "schema_version": "0.2",
        "num_qubits": 2,
        "instructions": [
            {
                "gate": {"name": "cx", "arity": 2},
                "targets": [{"index": 1}],
                "controls": [{"index": 0}],
            }
        ],
    }

    result = analyze_circuit_coupling(payload)

    assert result["mode"] == "measured"
    assert result["has_entangling_gates"] is True
    assert result["is_entangled"] is False
    assert [metric["value"] for metric in result["pair_metrics"]] == [0.0, 0.0]


def test_three_qubit_circuit_uses_parser_fallback() -> None:
    payload = {
        "schema_version": "0.2",
        "num_qubits": 3,
        "instructions": [
            {"gate": {"name": "h", "arity": 1}, "targets": [{"index": 0}]},
            {
                "gate": {"name": "cx", "arity": 2},
                "targets": [{"index": 1}],
                "controls": [{"index": 0}],
            },
            {
                "gate": {"name": "cx", "arity": 2},
                "targets": [{"index": 2}],
                "controls": [{"index": 1}],
            },
        ],
    }

    result = analyze_circuit_coupling(payload)

    assert result["mode"] == "qualitative"
    assert result["provenance"] == "parser"
    assert result["is_entangled"] is None
    assert result["pair_metrics"] == []


def test_preservation_compares_measured_final_states() -> None:
    result = analyze_optimization_preservation(_bell_payload(), _bell_payload())

    assert result["fidelity_preserved"] == 1.0
    assert result["preserved_entanglement_structure"] is True
    assert result["original_coupling"]["provenance"] == "rqm-entanglement"
