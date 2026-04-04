"""Shape and algebraic validation helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from rqm_entanglement.constants import ATOL, RTOL


def assert_state_vector(psi: NDArray) -> None:  # type: ignore[type-arg]
    """Raise ``ValueError`` unless *psi* is a (4,) complex array."""
    if psi.shape != (4,):
        raise ValueError(f"State vector must have shape (4,), got {psi.shape}")


def assert_two_qubit_operator(U: NDArray) -> None:  # type: ignore[type-arg]
    """Raise ``ValueError`` unless *U* is a (4,4) array."""
    if U.shape != (4, 4):
        raise ValueError(f"Two-qubit operator must have shape (4,4), got {U.shape}")


def assert_single_qubit_operator(U: NDArray) -> None:  # type: ignore[type-arg]
    """Raise ``ValueError`` unless *U* is a (2,2) array."""
    if U.shape != (2, 2):
        raise ValueError(f"Single-qubit operator must have shape (2,2), got {U.shape}")


def is_unitary(
    U: NDArray,  # type: ignore[type-arg]
    atol: float = ATOL,
    rtol: float = RTOL,
) -> bool:
    """Return ``True`` if *U* is unitary: U† U ≈ I."""
    n = U.shape[0]
    if U.shape != (n, n):
        return False
    product = U.conj().T @ U
    return bool(np.allclose(product, np.eye(n, dtype=np.complex128), atol=atol, rtol=rtol))


def is_su4(
    U: NDArray,  # type: ignore[type-arg]
    atol: float = ATOL,
    rtol: float = RTOL,
) -> bool:
    """Return ``True`` if *U* is a 4×4 special-unitary matrix (det ≈ 1)."""
    if U.shape != (4, 4):
        return False
    if not is_unitary(U, atol=atol, rtol=rtol):
        return False
    det = np.linalg.det(U)
    return bool(np.isclose(det, 1.0, atol=atol, rtol=rtol))


def normalize_global_phase(
    U: NDArray,  # type: ignore[type-arg]
    atol: float = ATOL,
) -> NDArray:  # type: ignore[type-arg]
    """Remove global phase so that the first non-negligible element is real positive.

    This makes phase-equivalent matrices compare equal after normalization.
    """
    flat = U.ravel()
    for val in flat:
        if abs(val) > atol:
            phase = val / abs(val)
            return np.asarray(U / phase, dtype=np.complex128)
    return np.asarray(U, dtype=np.complex128)
