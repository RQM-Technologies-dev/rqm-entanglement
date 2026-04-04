"""Tests for rqm_entanglement.tensor."""

import numpy as np
import pytest

from rqm_entanglement.basis import ket00, ket01, ket10, ket11
from rqm_entanglement.constants import CNOT, I2, X
from rqm_entanglement.tensor import apply_unitary, kron, local_unitary


def test_kron_single():
    result = kron(I2)
    np.testing.assert_array_equal(result, I2)


def test_kron_two():
    result = kron(I2, I2)
    assert result.shape == (4, 4)
    np.testing.assert_allclose(result, np.eye(4), atol=1e-12)


def test_kron_three():
    result = kron(I2, I2, I2)
    assert result.shape == (8, 8)


def test_kron_no_args():
    with pytest.raises(ValueError):
        kron()


def test_local_unitary_shape():
    U = local_unitary(I2, I2)
    assert U.shape == (4, 4)
    np.testing.assert_allclose(U, np.eye(4), atol=1e-12)


def test_local_unitary_wrong_shape():
    with pytest.raises(ValueError):
        local_unitary(np.eye(4), I2)  # type: ignore[arg-type]


def test_local_unitary_x_x():
    # X ⊗ X: swaps |00>↔|11>, |01>↔|10>
    XX = local_unitary(X, X)
    np.testing.assert_allclose(XX @ ket00(), ket11(), atol=1e-12)
    np.testing.assert_allclose(XX @ ket01(), ket10(), atol=1e-12)


def test_local_unitary_i2_x_on_ket00_gives_ket01():
    U = local_unitary(I2, X)
    np.testing.assert_allclose(U @ ket00(), ket01(), atol=1e-12)


def test_local_unitary_x_i2_on_ket00_gives_ket10():
    U = local_unitary(X, I2)
    np.testing.assert_allclose(U @ ket00(), ket10(), atol=1e-12)


def test_apply_unitary_identity():
    psi = ket00()
    result = apply_unitary(np.eye(4, dtype=np.complex128), psi)
    np.testing.assert_allclose(result, psi, atol=1e-12)


def test_apply_unitary_cnot():
    # CNOT |10> = |11>
    result = apply_unitary(CNOT, ket10())
    np.testing.assert_allclose(result, ket11(), atol=1e-12)
    # CNOT |00> = |00>
    result2 = apply_unitary(CNOT, ket00())
    np.testing.assert_allclose(result2, ket00(), atol=1e-12)


def test_apply_unitary_wrong_shapes():
    with pytest.raises(ValueError):
        apply_unitary(np.eye(4, dtype=np.complex128), np.array([1, 0], dtype=np.complex128))
    with pytest.raises(ValueError):
        apply_unitary(np.eye(2, dtype=np.complex128), ket00())  # type: ignore[arg-type]


def test_apply_unitary_preserves_norm_for_unitary_input():
    psi = (ket00() + ket11()) / np.sqrt(2)
    U = local_unitary(X, I2)
    out = apply_unitary(U, psi)
    assert np.isclose(np.linalg.norm(out), 1.0, atol=1e-12)
