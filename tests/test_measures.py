"""Tests for rqm_entanglement.measures."""

import numpy as np
import pytest

from rqm_entanglement.basis import ket00, ket01, ket10, ket11
from rqm_entanglement.measures import (
    concurrence_pure,
    entanglement_entropy_pure,
    is_separable_pure,
    schmidt_values_pure,
    von_neumann_entropy,
)
from rqm_entanglement.states import state_from_amplitudes

# ── Bell state fixture ────────────────────────────────────────────────────────

@pytest.fixture()
def bell_phi_plus() -> np.ndarray:
    """Return |Φ+> = (|00> + |11>) / √2."""
    return state_from_amplitudes(1, 0, 0, 1)  # normalized by default


# ── concurrence_pure ─────────────────────────────────────────────────────────

def test_concurrence_product_state_zero():
    assert concurrence_pure(ket00()) == pytest.approx(0.0, abs=1e-12)


def test_concurrence_bell_state_one(bell_phi_plus: np.ndarray):
    assert concurrence_pure(bell_phi_plus) == pytest.approx(1.0, abs=1e-9)


def test_concurrence_partial_entanglement():
    # a00=cos(θ/2), a11=sin(θ/2) → C = sin(θ)
    theta = np.pi / 6  # C = sin(π/6) = 0.5
    psi = state_from_amplitudes(np.cos(theta / 2), 0, 0, np.sin(theta / 2))
    assert concurrence_pure(psi) == pytest.approx(np.sin(theta), abs=1e-9)


# ── is_separable_pure ────────────────────────────────────────────────────────

def test_is_separable_product_states():
    for ket in (ket00(), ket01(), ket10(), ket11()):
        assert is_separable_pure(ket)


def test_not_separable_bell(bell_phi_plus: np.ndarray):
    assert not is_separable_pure(bell_phi_plus)


# ── schmidt_values_pure ──────────────────────────────────────────────────────

def test_schmidt_values_product_state():
    sv = schmidt_values_pure(ket00())
    assert sv.shape == (2,)
    # For a product state, one Schmidt value is 1 and the other is 0
    np.testing.assert_allclose(sorted(sv)[::-1], [1.0, 0.0], atol=1e-12)


def test_schmidt_values_bell_state(bell_phi_plus: np.ndarray):
    sv = schmidt_values_pure(bell_phi_plus)
    np.testing.assert_allclose(sorted(sv)[::-1], [1 / np.sqrt(2), 1 / np.sqrt(2)], atol=1e-9)


# ── von_neumann_entropy ──────────────────────────────────────────────────────

def test_von_neumann_entropy_pure_state_is_zero():
    """For a pure state ρ = |ψ><ψ|, entropy = 0."""
    from rqm_entanglement.states import density_matrix

    rho = density_matrix(ket00())
    assert von_neumann_entropy(rho) == pytest.approx(0.0, abs=1e-9)


def test_von_neumann_entropy_maximally_mixed():
    """For ρ = I/2, entropy = 1 bit."""
    rho = np.eye(2, dtype=np.complex128) / 2
    assert von_neumann_entropy(rho) == pytest.approx(1.0, abs=1e-9)


def test_von_neumann_entropy_invalid_shape():
    with pytest.raises(ValueError):
        von_neumann_entropy(np.eye(3, dtype=np.complex128))


# ── entanglement_entropy_pure ────────────────────────────────────────────────

def test_entanglement_entropy_product_is_zero():
    assert entanglement_entropy_pure(ket00()) == pytest.approx(0.0, abs=1e-9)


def test_entanglement_entropy_bell_is_one(bell_phi_plus: np.ndarray):
    s0 = entanglement_entropy_pure(bell_phi_plus, subsystem=0)
    s1 = entanglement_entropy_pure(bell_phi_plus, subsystem=1)
    assert s0 == pytest.approx(1.0, abs=1e-9)
    assert s1 == pytest.approx(1.0, abs=1e-9)


def test_entanglement_entropy_readme_example():
    """Replicate the README Bell-state example end-to-end."""
    from rqm_entanglement import (
        CNOT,
        I2,
        Y,
        apply_unitary,
        local_unitary,
    )

    def ry(theta: float) -> np.ndarray:
        return np.cos(theta / 2) * I2 - 1j * np.sin(theta / 2) * Y

    psi0 = ket00()
    U = CNOT @ local_unitary(ry(np.pi / 2), I2)
    psi = apply_unitary(U, psi0)

    assert concurrence_pure(psi) == pytest.approx(1.0, abs=1e-9)
    assert entanglement_entropy_pure(psi) == pytest.approx(1.0, abs=1e-9)
