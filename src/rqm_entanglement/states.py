"""State-vector and density-matrix helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from rqm_entanglement.constants import ATOL
from rqm_entanglement.validation import assert_state_vector


def normalize_state(psi: NDArray, atol: float = ATOL) -> NDArray[np.complex128]:  # type: ignore[type-arg]
    """Return *psi* divided by its norm.

    Raises ``ValueError`` if the norm is below *atol* (zero-vector).
    """
    assert_state_vector(psi)
    norm = float(np.linalg.norm(psi))
    if norm < atol:
        raise ValueError("Cannot normalize a zero-amplitude state vector.")
    return (psi / norm).astype(np.complex128)


def state_from_amplitudes(
    a00: complex,
    a01: complex,
    a10: complex,
    a11: complex,
    normalize: bool = True,
) -> NDArray[np.complex128]:
    """Build a (4,) state vector from four amplitudes.

    Ordering: [a00, a01, a10, a11] corresponding to |00>,|01>,|10>,|11>.
    If *normalize* is ``True`` (default), the vector is L2-normalized.
    """
    psi = np.array([a00, a01, a10, a11], dtype=np.complex128)
    if normalize:
        return normalize_state(psi)
    return psi


def density_matrix(psi: NDArray) -> NDArray[np.complex128]:  # type: ignore[type-arg]
    """Return the (4,4) density matrix ρ = |ψ><ψ|."""
    assert_state_vector(psi)
    psi_c = psi.astype(np.complex128)
    return np.outer(psi_c, psi_c.conj())


def reduced_density_matrix(
    state_or_rho: NDArray,  # type: ignore[type-arg]
    subsystem: int,
) -> NDArray[np.complex128]:
    """Return the (2,2) reduced density matrix after tracing out one qubit.

    Parameters
    ----------
    state_or_rho:
        Either a (4,) pure state vector or a (4,4) density matrix.
    subsystem:
        0 → keep qubit 0 (trace out qubit 1).
        1 → keep qubit 1 (trace out qubit 0).
    """
    if subsystem not in (0, 1):
        raise ValueError(f"subsystem must be 0 or 1, got {subsystem}")

    if state_or_rho.shape == (4,):
        rho = density_matrix(state_or_rho)
    elif state_or_rho.shape == (4, 4):
        rho = state_or_rho.astype(np.complex128)
    else:
        raise ValueError(
            f"state_or_rho must have shape (4,) or (4,4), got {state_or_rho.shape}"
        )

    # Reshape to (qubit0, qubit1, qubit0', qubit1') tensor
    rho_tensor = rho.reshape(2, 2, 2, 2)

    if subsystem == 0:
        # Trace out qubit 1: sum over qubit1 index
        # rho_A[i,j] = sum_k rho[i,k,j,k]
        return np.asarray(np.einsum("ikjk->ij", rho_tensor), dtype=np.complex128)
    else:
        # Trace out qubit 0: sum over qubit0 index
        # rho_B[k,l] = sum_i rho[i,k,i,l]
        return np.asarray(np.einsum("ikil->kl", rho_tensor), dtype=np.complex128)
