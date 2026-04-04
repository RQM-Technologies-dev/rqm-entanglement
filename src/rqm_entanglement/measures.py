"""Pure-state entanglement measures and von Neumann entropy."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from rqm_entanglement.constants import ATOL
from rqm_entanglement.states import density_matrix, reduced_density_matrix
from rqm_entanglement.validation import assert_state_vector


def concurrence_pure(psi: NDArray) -> float:  # type: ignore[type-arg]
    """Return the concurrence of a pure two-qubit state.

    For ψ = [a00, a01, a10, a11]^T::

        C(ψ) = 2 |a00·a11 - a01·a10|

    C = 0 iff the state is product; C = 1 iff maximally entangled.
    """
    assert_state_vector(psi)
    p = psi.astype(np.complex128)
    return float(2.0 * abs(p[0] * p[3] - p[1] * p[2]))


def is_separable_pure(psi: NDArray, atol: float = ATOL) -> bool:  # type: ignore[type-arg]
    """Return ``True`` if *psi* is a product (separable) pure state."""
    return concurrence_pure(psi) <= atol


def schmidt_values_pure(psi: NDArray) -> NDArray[np.float64]:  # type: ignore[type-arg]
    """Return the Schmidt coefficients of a pure two-qubit state.

    Reshape the amplitude vector to a (2,2) matrix M where
    M[i,j] = ⟨ij|ψ⟩, then return the singular values of M in
    descending order.  The entanglement entropy equals
    -Σ σ² log₂(σ²).
    """
    assert_state_vector(psi)
    M = psi.astype(np.complex128).reshape(2, 2)
    return np.linalg.svd(M, compute_uv=False)  # type: ignore[return-value]


def von_neumann_entropy(rho: NDArray, atol: float = ATOL) -> float:  # type: ignore[type-arg]
    """Return the von Neumann entropy of *rho* in bits: S = -Tr(ρ log₂ ρ).

    Tiny negative eigenvalues caused by floating-point noise are clipped to
    zero before computing the entropy.
    """
    if rho.shape not in ((2, 2), (4, 4)):
        raise ValueError(f"rho must be (2,2) or (4,4), got {rho.shape}")
    eigvals = np.linalg.eigvalsh(rho.astype(np.complex128))
    eigvals = np.clip(eigvals.real, 0.0, None)
    # Keep only nonzero eigenvalues (0 log 0 = 0 by convention)
    nonzero = eigvals[eigvals > atol]
    if nonzero.size == 0:
        return 0.0
    return float(-np.sum(nonzero * np.log2(nonzero)))


def entanglement_entropy_pure(
    psi: NDArray,  # type: ignore[type-arg]
    subsystem: int = 0,
    atol: float = ATOL,
) -> float:
    """Return the entanglement entropy S(ρ_A) in bits for a pure state *psi*.

    Computes the reduced density matrix of *subsystem* and returns its
    von Neumann entropy.
    """
    assert_state_vector(psi)
    rho = density_matrix(psi)
    rho_reduced = reduced_density_matrix(rho, subsystem)
    return von_neumann_entropy(rho_reduced, atol=atol)
