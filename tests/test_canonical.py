"""Tests for rqm_entanglement.canonical."""

import numpy as np
import pytest

from rqm_entanglement.canonical import (
    canonical_entangler,
    xx_rotation,
    yy_rotation,
    zz_rotation,
)
from rqm_entanglement.validation import is_su4, is_unitary


@pytest.mark.parametrize("theta", [0.0, np.pi / 4, np.pi / 2, np.pi, 2 * np.pi])
def test_xx_rotation_unitary(theta: float):
    U = xx_rotation(theta)
    assert U.shape == (4, 4)
    assert is_unitary(U), f"xx_rotation({theta}) is not unitary"


@pytest.mark.parametrize("theta", [0.0, np.pi / 4, np.pi / 2, np.pi, 2 * np.pi])
def test_yy_rotation_unitary(theta: float):
    assert is_unitary(yy_rotation(theta))


@pytest.mark.parametrize("theta", [0.0, np.pi / 4, np.pi / 2, np.pi, 2 * np.pi])
def test_zz_rotation_unitary(theta: float):
    assert is_unitary(zz_rotation(theta))


def test_xx_rotation_zero_is_identity():
    np.testing.assert_allclose(xx_rotation(0.0), np.eye(4), atol=1e-12)


def test_yy_rotation_zero_is_identity():
    np.testing.assert_allclose(yy_rotation(0.0), np.eye(4), atol=1e-12)


def test_zz_rotation_zero_is_identity():
    np.testing.assert_allclose(zz_rotation(0.0), np.eye(4), atol=1e-12)


def test_canonical_entangler_is_unitary():
    U = canonical_entangler(np.pi / 4, np.pi / 4, np.pi / 4)
    assert is_unitary(U)


def test_canonical_entangler_zero_is_identity():
    U = canonical_entangler(0.0, 0.0, 0.0)
    np.testing.assert_allclose(U, np.eye(4), atol=1e-12)


def test_canonical_entangler_cnot_like():
    """canonical_entangler(π/2, π/2, π/2) is locally equivalent to CNOT.

    We do not assert exact equality (that would require a Cartan decomposition),
    but we verify the result is unitary and has the expected operator Schmidt rank.
    """
    from rqm_entanglement.classify import operator_schmidt_rank

    U = canonical_entangler(np.pi / 2, np.pi / 2, np.pi / 2)
    assert is_unitary(U)
    rank = operator_schmidt_rank(U)
    assert rank > 1  # nonlocal


def test_xx_rotation_half_pi_structure():
    """xx_rotation(π/2) = (1/√2)(I4 - i XX)."""
    from rqm_entanglement.constants import I4, XX

    U = xx_rotation(np.pi / 2)
    expected = (1 / np.sqrt(2)) * (I4 - 1j * XX)
    np.testing.assert_allclose(U, expected, atol=1e-12)


@pytest.mark.parametrize("theta", [0.0, np.pi / 7, np.pi / 3, -np.pi / 5])
def test_canonical_entangler_zz_reduction(theta: float):
    U = canonical_entangler(0.0, 0.0, theta)
    np.testing.assert_allclose(U, zz_rotation(theta), atol=1e-12)


def test_canonical_entangler_is_su4():
    U = canonical_entangler(np.pi / 7, np.pi / 5, np.pi / 3)
    assert is_unitary(U)
    assert np.isclose(np.linalg.det(U), 1.0, atol=1e-12)
    assert is_su4(U)
