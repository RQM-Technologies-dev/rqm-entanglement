"""Tensor-product helpers for two-qubit operators and state vectors."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from rqm_entanglement.constants import ATOL
from rqm_entanglement.validation import (
    assert_single_qubit_operator,
    assert_state_vector,
    assert_two_qubit_operator,
)


def kron(*ops: NDArray) -> NDArray:  # type: ignore[type-arg]
    """Return the Kronecker (tensor) product of one or more matrices.

    ``kron(A, B, C)`` computes A ⊗ B ⊗ C left-to-right.
    """
    if not ops:
        raise ValueError("kron requires at least one operand")
    result: NDArray = ops[0].astype(np.complex128)  # type: ignore[type-arg]
    for op in ops[1:]:
        result = np.kron(result, op.astype(np.complex128))
    return result


def local_unitary(U1: NDArray, U2: NDArray) -> NDArray:  # type: ignore[type-arg]
    """Return U1 ⊗ U2, the local two-qubit unitary acting on qubits 0 and 1.

    Both *U1* and *U2* must be (2,2) matrices.
    """
    assert_single_qubit_operator(U1)
    assert_single_qubit_operator(U2)
    return np.kron(U1.astype(np.complex128), U2.astype(np.complex128))


def apply_unitary(U: NDArray, psi: NDArray) -> NDArray:  # type: ignore[type-arg]
    """Apply a (4,4) unitary *U* to a (4,) state vector *psi*.

    The result is *not* renormalized unless numerical drift pushes the norm
    outside ``ATOL`` of 1.0, in which case a ``ValueError`` is raised to
    surface the issue rather than silently hiding it.
    """
    assert_two_qubit_operator(U)
    assert_state_vector(psi)
    result: NDArray = U.astype(np.complex128) @ psi.astype(np.complex128)  # type: ignore[type-arg]
    norm = float(np.linalg.norm(result))
    if abs(norm - 1.0) > ATOL * 1e3:
        raise ValueError(
            f"apply_unitary produced a state with norm {norm:.6g}; "
            "check that U is unitary and psi is normalized."
        )
    return result
