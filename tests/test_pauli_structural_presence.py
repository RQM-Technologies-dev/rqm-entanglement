import pytest

from rqm_entanglement import analyze_circuit_coupling


def circuit(gate, angle=0.0, cancel=False):
    instruction = {"gate": {"name": gate, "arity": 2},
                   "targets": [{"index": 0}, {"index": 1}],
                   "params": [{"name": "angle", "value": angle}]}
    ops = [instruction]
    if cancel:
        ops.append({**instruction, "params": [{"name": "angle", "value": -angle}]})
    return {"schema_version": "0.2", "num_qubits": 2, "instructions": ops}


@pytest.mark.parametrize("gate", ["rxx", "ryy", "rzz"])
@pytest.mark.parametrize("angle,cancel", [(0, False), (0.7, False), (0.7, True)])
def test_structural_presence_not_state_claim(gate, angle, cancel):
    src = circuit(gate, angle, cancel)
    result = analyze_circuit_coupling(src)
    assert result["mode"] == "qualitative"
    assert result["has_entangling_gates"] is True
    assert result["entangling_gate_count"] == 1 + cancel
    assert result["is_entangled"] is None
    assert result["pair_metrics"] == []
    assert result["fidelity_preserved"] is None
    with pytest.raises(ValueError, match="allow_qualitative_fallback"):
        analyze_circuit_coupling(src, {"allow_qualitative_fallback": False})


def test_supported_product_stays_numerical():
    result = analyze_circuit_coupling(circuit("cz"))
    assert result["mode"] == "measured"
    assert result["has_entangling_gates"] is True
    assert result["is_entangled"] is False
