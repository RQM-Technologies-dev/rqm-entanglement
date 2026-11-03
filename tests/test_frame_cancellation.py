"""Phase-safe, opt-in Cartan composition with conservative KAK fallback."""

from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from rqm_entanglement import (
    AxisHinge,
    CartanRelation,
    QuaternionCartanBlock,
    compose_relations,
    su4,
)

IDENTITY = (1.0, 0.0, 0.0, 0.0)
XQ = (0.0, 1.0, 0.0, 0.0)
TOL = 1e-10


def block(
    coords=(0.1, 0.05, 0.01), left=(IDENTITY, IDENTITY), right=(IDENTITY, IDENTITY), phase=0.0
):
    return QuaternionCartanBlock.from_components(
        left_q0=left[0],
        left_q1=left[1],
        right_q0=right[0],
        right_q1=right[1],
        cartan_a=coords[0],
        cartan_b=coords[1],
        cartan_c=coords[2],
        global_phase=phase,
    )


def matrix(q):
    w, x, y, z = q
    return np.array([[w - 1j * z, -y - 1j * x], [y - 1j * x, w + 1j * z]])


def reference(b):
    expm = pytest.importorskip("scipy.linalg").expm
    # Independent standard complex-matrix exponential, including scalar phase.
    x = np.array([[0, 1], [1, 0]])
    y = np.array([[0, -1j], [1j, 0]])
    z = np.diag([1, -1])
    h = sum(c * np.kron(p, p) for c, p in zip(b.cartan, (x, y, z), strict=True))
    return (
        np.exp(1j * b.global_phase)
        * np.kron(matrix(b.left_q1), matrix(b.left_q0))
        @ expm(1j * h)
        @ np.kron(matrix(b.right_q1), matrix(b.right_q0))
    )


def compose_and_check(a, b, expected_calls, **kwargs):
    expected = reference(a) @ reference(b)
    with patch.object(su4, "decompose_with_qiskit", wraps=su4.decompose_with_qiskit) as kak:
        result = compose_relations(a, b, **kwargs)
    assert kak.call_count == expected_calls
    actual = result.to_unitary()
    assert np.max(np.abs(actual - expected)) <= TOL
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import Operator

    # Actual Qiskit-controlled construction catches discarded global signs/phases.
    before = QuantumCircuit(3)
    after = QuantumCircuit(3)
    before.append(UnitaryGate(expected).control(), [0, 1, 2])
    after.append(UnitaryGate(actual).control(), [0, 1, 2])
    assert np.max(np.abs(Operator(before).data - Operator(after).data)) <= TOL
    return result


def test_default_behavior_still_uses_existing_kak():
    a, b = block(), block()
    compose_and_check(a, b, 1)
    compose_and_check(a, b, 1, prefer_frame_cancellation=False)
    compose_and_check(a, b, 0, prefer_frame_cancellation=True)


@pytest.mark.parametrize("negative_factors", [0, 1, 2, 3])
def test_negative_middle_factors_keep_scalar_phase(negative_factors):
    frames = tuple(XQ if negative_factors & (1 << i) else IDENTITY for i in range(2))
    a = block(right=frames, phase=2.9)
    b = block(left=frames, phase=-2.3)
    compose_and_check(a, b, 0, prefer_frame_cancellation=True)


@pytest.mark.parametrize("delta,expected_calls", [(1e-12, 1), (4e-12, 1), (0.1, 1)])
def test_middle_frame_tolerance_is_conservative(delta, expected_calls):
    q = (float(np.cos(delta / 2)), float(np.sin(delta / 2)), 0.0, 0.0)
    compose_and_check(
        block(), block(left=(q, IDENTITY)), expected_calls, prefer_frame_cancellation=True
    )


@pytest.mark.parametrize(
    "a,expected_calls",
    [
        (np.nextafter(np.pi / 4, -np.inf), 0),
        (np.pi / 4, 0),
        (np.nextafter(np.pi / 4, np.inf), 1),
        (np.pi / 4 + 1e-12, 1),
    ],
)
def test_chamber_gate_has_no_tolerance_slack(a, expected_calls):
    # Input validation permits roundoff near the chamber. Bypass must not expand it.
    compose_and_check(
        block((0, 0, 0)), block((a, 0, 0)), expected_calls, prefer_frame_cancellation=True
    )


def test_summed_coordinates_outside_chamber_fall_back():
    compose_and_check(
        block((0.5, 0.2, 0.1)), block((0.5, 0.2, -0.1)), 1, prefer_frame_cancellation=True
    )


def test_unknown_convention_is_rejected_before_composition():
    with pytest.raises(ValueError, match="unsupported convention_version"):
        replace(block(), convention_version="future-experimental-convention")


def test_final_proof_rejects_injected_phase_fault(monkeypatch):
    a, b = block(), block()
    original = QuaternionCartanBlock.from_components.__func__

    def corrupt(cls, **kwargs):
        kwargs["global_phase"] += 0.125
        return original(cls, **kwargs)

    monkeypatch.setattr(QuaternionCartanBlock, "from_components", classmethod(corrupt))
    compose_and_check(a, b, 1, prefer_frame_cancellation=True)


def test_simple_representations_and_return_types_are_unchanged():
    for a, b in [
        (AxisHinge("xx", 0.2), AxisHinge("xx", 0.1)),
        (AxisHinge("xx", 0.2), CartanRelation(0.1, 0.2, 0.3)),
    ]:
        assert compose_relations(a, b) == compose_relations(a, b, prefer_frame_cancellation=True)
    a, b = block(), AxisHinge("xx", 0.2)
    with patch.object(su4, "decompose_with_qiskit", wraps=su4.decompose_with_qiskit) as kak:
        result = compose_relations(a, b, prefer_frame_cancellation=True)
    assert kak.call_count == 1
    assert np.max(np.abs(result.to_unitary() - reference(a) @ b.to_unitary())) <= TOL


def test_held_out_random_frames_and_noncommuting_controls():
    rng = np.random.default_rng(20261105)

    def q():
        v = rng.normal(size=4)
        return tuple(v / np.linalg.norm(v))

    for _ in range(32):
        q0, q1 = q(), q()
        inverse = tuple((v[0], -v[1], -v[2], -v[3]) for v in (q0, q1))
        a = block(right=(q0, q1), left=(q(), q()), phase=float(rng.uniform(-np.pi, np.pi)))
        b = block(left=inverse, right=(q(), q()), phase=float(rng.uniform(-np.pi, np.pi)))
        compose_and_check(a, b, 0, prefer_frame_cancellation=True)
        changed = block(left=(q(), q()), right=(q(), q()), phase=b.global_phase)
        compose_and_check(a, changed, 1, prefer_frame_cancellation=True)


def test_direct_path_does_not_require_optional_qiskit():
    a, b = block(), block()
    with patch.object(su4, "decompose_with_qiskit", side_effect=ImportError("no qiskit")):
        result = compose_relations(a, b, prefer_frame_cancellation=True)
        assert np.max(np.abs(result.to_unitary() - a.to_unitary() @ b.to_unitary())) <= TOL
        with pytest.raises(ImportError, match="no qiskit"):
            compose_relations(a, b)
