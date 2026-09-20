"""Dense independent oracles for analytic signed Pauli coefficient actions."""

import math

import numpy as np
import pytest

from rqm_entanglement import CartanRelation, QuaternionCartanBlock
from rqm_entanglement.pauli_transfer import apply_pair_coefficients
from rqm_entanglement.relational import _local_matrix, promote_with_local_frames

P = [np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1, -1])]
B = [np.kron(a, b) for a in P for b in P]


@pytest.mark.parametrize("gate", ["rxx", "ryy", "rzz", "cx"])
@pytest.mark.parametrize("angle", [0.0, 1e-12, 0.37, -1.9, math.pi, -math.pi])
def test_pair_action_all_signed_basis_elements(gate, angle):
    if gate == "cx":
        u = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    else:
        g = np.kron(P["ixyz".index(gate[1])], P["ixyz".index(gate[1])])
        u = math.cos(angle / 2) * np.eye(4) - 1j * math.sin(angle / 2) * g
    for k, b in enumerate(B):
        c = np.zeros((4, 4))
        c.flat[k] = 1
        got = apply_pair_coefficients(c, (0, 1), gate, angle)
        matrix = sum(v * p for v, p in zip(got.flat, B))
        np.testing.assert_allclose(matrix, u @ b @ u.conj().T, atol=1e-14)
        # Non-adjacent reversed axes and untouched spectator dimensions.
        tensor = np.stack([c.T, 2 * c.T], axis=1)
        actual = apply_pair_coefficients(tensor, (2, 0), gate, angle)
        np.testing.assert_allclose(actual, np.stack([got.T, 2 * got.T], axis=1), atol=1e-14)


@pytest.mark.parametrize("coords", [(-0.8, -0.4, 0.2), (0.0, 0.0, 0.0), (-math.pi / 2, -0.3, 0.0)])
def test_canonical_promotion_skips_reconstruction_and_kak(monkeypatch, coords):
    relation = CartanRelation(*coords)
    left = (-math.cos(0.2), -math.sin(0.2), 0.0, 0.0)
    right = (math.cos(0.3), 0.0, math.sin(0.3), 0.0)
    expected = np.exp(0.17j) * _local_matrix(left, right) @ relation.to_unitary()
    monkeypatch.setattr(CartanRelation, "to_unitary", lambda *a: pytest.fail("reconstruction"))
    monkeypatch.setattr(QuaternionCartanBlock, "from_unitary", lambda *a: pytest.fail("KAK"))
    got = promote_with_local_frames(relation, left_q0=left, left_q1=right, global_phase=0.17)
    np.testing.assert_allclose(got.to_unitary(), expected, atol=1e-12)


@pytest.mark.parametrize(
    "coords", [(0.8, 0.4, 0.2), (-0.2, -0.8, 0.0), (-math.pi / 2 - 1e-8, 0.0, 0.0)]
)
def test_noncanonical_promotion_retains_fallback(monkeypatch, coords):
    original = QuaternionCartanBlock.from_unitary
    calls = []

    def wrapped(u):
        calls.append(1)
        return original(u)

    monkeypatch.setattr(QuaternionCartanBlock, "from_unitary", wrapped)
    relation = CartanRelation(*coords)
    got = relation.promote()
    assert len(calls) == 1
    np.testing.assert_allclose(got.to_unitary(), relation.to_unitary(), atol=1e-10)
