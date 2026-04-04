"""Operator Schmidt decomposition and local-product classification.

The operator Schmidt rank characterizes how "entangled" a two-qubit operator
is as a superoperator.  Rank 1 means the operator is a local product A ⊗ B.

Note on SWAP
------------
SWAP has operator Schmidt rank > 1 (it is a nonlocal operator), but it maps
every product state to another product state.  This package does *not* expose
a generic ``is_entangling_gate`` function, because classifying SWAP and similar
operators requires a full KAK / Cartan decomposition that is not implemented
in v0.1.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from rqm_entanglement.constants import ATOL
from rqm_entanglement.validation import assert_two_qubit_operator


def _reshuffled(U: NDArray) -> NDArray[np.complex128]:  # type: ignore[type-arg]
    """Return the operator-Schmidt reshuffled (4,4) matrix.

    For U with index structure U_(ij, kl), reshape to (i, j, k, l),
    permute to (i, k, j, l), then reshape to (4, 4).
    """
    M = U.astype(np.complex128).reshape(2, 2, 2, 2)
    # (i,j,k,l) -> (i,k,j,l)
    M = M.transpose(0, 2, 1, 3)
    return M.reshape(4, 4)


def operator_schmidt_rank(U: NDArray, atol: float = ATOL) -> int:  # type: ignore[type-arg]
    """Return the operator Schmidt rank of a (4,4) two-qubit operator.

    The rank equals the number of nonzero singular values of the reshuffled
    matrix.  Rank 1 iff *U* is a local product A ⊗ B.
    """
    assert_two_qubit_operator(U)
    singular_values = np.linalg.svd(_reshuffled(U), compute_uv=False)
    return int(np.sum(singular_values > atol))


def is_local_product_operator(U: NDArray, atol: float = ATOL) -> bool:  # type: ignore[type-arg]
    """Return ``True`` if *U* is (numerically) a local product operator A ⊗ B."""
    return operator_schmidt_rank(U, atol=atol) == 1


def local_product_factors(
    U: NDArray,  # type: ignore[type-arg]
    atol: float = ATOL,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]] | None:
    """Return ``(A, B)`` if U ≈ A ⊗ B, or ``None`` if *U* is not a local product.

    The factors are recovered from the leading singular vectors of the
    reshuffled matrix.  They are defined only up to a scalar phase:
    if ``(A, B)`` is a solution, so is ``(e^{iφ} A, e^{-iφ} B)`` for any φ.

    Returns ``None`` if the operator Schmidt rank is not 1.
    """
    assert_two_qubit_operator(U)
    R = _reshuffled(U)
    Ul, s, Vt = np.linalg.svd(R)

    if s[0] < atol:
        return None
    # Check that second singular value is negligible → rank 1
    if len(s) > 1 and s[1] > atol:
        return None

    # Leading left singular vector → A (reshaped to 2×2)
    # Leading right singular vector → B^T (reshaped to 2×2)
    scale = np.sqrt(s[0])
    A = (Ul[:, 0] * scale).reshape(2, 2).astype(np.complex128)
    B = (Vt[0, :] * scale).reshape(2, 2).astype(np.complex128)
    return A, B
