"""Tests for rqm_entanglement.basis."""

import numpy as np
import pytest

from rqm_entanglement.basis import (
    basis_state,
    computational_basis,
    ket00,
    ket01,
    ket10,
    ket11,
)


def test_ket00():
    psi = ket00()
    assert psi.shape == (4,)
    assert psi.dtype == np.complex128
    np.testing.assert_array_equal(psi, [1, 0, 0, 0])


def test_ket01():
    psi = ket01()
    np.testing.assert_array_equal(psi, [0, 1, 0, 0])


def test_ket10():
    psi = ket10()
    np.testing.assert_array_equal(psi, [0, 0, 1, 0])


def test_ket11():
    psi = ket11()
    np.testing.assert_array_equal(psi, [0, 0, 0, 1])


def test_basis_state_all_labels():
    for label, ref in [("00", ket00()), ("01", ket01()), ("10", ket10()), ("11", ket11())]:
        np.testing.assert_array_equal(basis_state(label), ref)


def test_basis_state_invalid():
    with pytest.raises(ValueError, match="bits must be"):
        basis_state("2")


def test_computational_basis_returns_tuple():
    result = computational_basis()
    assert isinstance(result, tuple)
    assert len(result) == 4
    for v in result:
        assert v.shape == (4,)


def test_basis_states_are_orthonormal():
    vecs = list(computational_basis())
    for i, u in enumerate(vecs):
        for j, v in enumerate(vecs):
            expected = 1.0 if i == j else 0.0
            assert abs(np.dot(u.conj(), v) - expected) < 1e-12
