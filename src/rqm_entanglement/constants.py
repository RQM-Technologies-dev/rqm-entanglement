"""Pauli matrices, two-qubit identity, standard two-qubit gates, and tolerances.

All arrays have dtype ``np.complex128``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# ──────────────────────────────────────────────────────────────────────────────
# Tolerances
# ──────────────────────────────────────────────────────────────────────────────

ATOL: float = 1e-9
RTOL: float = 1e-9

# ──────────────────────────────────────────────────────────────────────────────
# Identities
# ──────────────────────────────────────────────────────────────────────────────

I2: NDArray[np.complex128] = np.eye(2, dtype=np.complex128)
I4: NDArray[np.complex128] = np.eye(4, dtype=np.complex128)

# ──────────────────────────────────────────────────────────────────────────────
# Single-qubit Paulis
# ──────────────────────────────────────────────────────────────────────────────

X: NDArray[np.complex128] = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y: NDArray[np.complex128] = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z: NDArray[np.complex128] = np.array([[1, 0], [0, -1]], dtype=np.complex128)

# ──────────────────────────────────────────────────────────────────────────────
# Two-qubit tensor-product Paulis
# ──────────────────────────────────────────────────────────────────────────────

XX: NDArray[np.complex128] = np.kron(X, X).astype(np.complex128)
YY: NDArray[np.complex128] = np.kron(Y, Y).astype(np.complex128)
ZZ: NDArray[np.complex128] = np.kron(Z, Z).astype(np.complex128)

# ──────────────────────────────────────────────────────────────────────────────
# Standard two-qubit gates  (computational basis |00>, |01>, |10>, |11>)
# qubit 0 is the more-significant index
# ──────────────────────────────────────────────────────────────────────────────

# CNOT: control = qubit 0, target = qubit 1
CNOT: NDArray[np.complex128] = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ],
    dtype=np.complex128,
)

# CZ: control = qubit 0, target = qubit 1
CZ: NDArray[np.complex128] = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, -1],
    ],
    dtype=np.complex128,
)

# SWAP
SWAP: NDArray[np.complex128] = np.array(
    [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.complex128,
)

# iSWAP
ISWAP: NDArray[np.complex128] = np.array(
    [
        [1, 0, 0, 0],
        [0, 0, 1j, 0],
        [0, 1j, 0, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.complex128,
)
