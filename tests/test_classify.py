"""Tests for rqm_entanglement.classify."""

import numpy as np
import pytest

from rqm_entanglement.classify import (
    is_local_product_operator,
    local_product_factors,
    operator_schmidt_rank,
)
from rqm_entanglement.constants import CNOT, CZ, I2, I4, SWAP, X, Y, Z
from rqm_entanglement.tensor import kron

# ── operator_schmidt_rank ─────────────────────────────────────────────────────

def test_identity_is_local_product():
    assert operator_schmidt_rank(I4) == 1


def test_local_product_rank_one():
    U = kron(X, Y)
    assert operator_schmidt_rank(U) == 1


def test_cnot_is_not_rank_one():
    assert operator_schmidt_rank(CNOT) > 1


def test_swap_is_not_rank_one():
    """SWAP has operator Schmidt rank > 1 (it is a nonlocal operator)."""
    assert operator_schmidt_rank(SWAP) > 1


def test_cz_is_not_rank_one():
    assert operator_schmidt_rank(CZ) > 1


# ── is_local_product_operator ─────────────────────────────────────────────────

def test_i4_is_local_product():
    assert is_local_product_operator(I4)


def test_kron_paulis_is_local_product():
    for A, B in [(X, I2), (I2, Y), (Z, Z), (Y, X)]:
        assert is_local_product_operator(kron(A, B))


def test_cnot_is_not_local_product():
    assert not is_local_product_operator(CNOT)


def test_swap_is_not_local_product():
    assert not is_local_product_operator(SWAP)


# ── local_product_factors ─────────────────────────────────────────────────────

def test_local_product_factors_reconstructs():
    """For U = A ⊗ B, factors A and B reconstruct U up to global phase."""
    A = np.array([[0, 1], [1, 0]], dtype=np.complex128)  # X
    B = np.array([[1, 0], [0, -1]], dtype=np.complex128)  # Z
    U = kron(A, B)
    result = local_product_factors(U)
    assert result is not None
    A_rec, B_rec = result
    # Reconstruct and compare (up to global phase)
    U_rec = kron(A_rec, B_rec)
    # They must be proportional (global phase ambiguity)
    mask = np.abs(U.ravel()) > 1e-10
    ratio = U_rec.ravel()[mask] / U.ravel()[mask]
    np.testing.assert_allclose(ratio / ratio[0], np.ones(len(ratio)), atol=1e-9)


def test_local_product_factors_none_for_cnot():
    assert local_product_factors(CNOT) is None


def test_local_product_factors_identity():
    result = local_product_factors(I4)
    assert result is not None
    A, B = result
    U_rec = kron(A, B)
    np.testing.assert_allclose(np.abs(U_rec), np.eye(4), atol=1e-9)


def test_local_product_factors_wrong_shape():
    with pytest.raises(ValueError):
        local_product_factors(np.eye(2, dtype=np.complex128))  # type: ignore[arg-type]
