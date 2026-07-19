"""Optional Qiskit Weyl-decomposition authority.

Qiskit is imported only inside adapter functions so the core package remains
usable for reconstruction, serialization, hashing, and block classification
without the optional dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

QISKIT_FACTOR_ORDER = "qiskit-Klr:kron(q1=Kl,q0=Kr);circuit-little-endian"


@dataclass(frozen=True)
class QiskitWeylResult:
    """Qiskit Weyl coordinates and local U(2) factors in RQM qubit order."""

    a: float
    b: float
    c: float
    global_phase: float
    left_q0_matrix: NDArray[np.complex128]
    left_q1_matrix: NDArray[np.complex128]
    right_q0_matrix: NDArray[np.complex128]
    right_q1_matrix: NDArray[np.complex128]


def decompose_with_qiskit(unitary: NDArray[np.complex128]) -> QiskitWeylResult:
    """Decompose a finite 4x4 unitary with Qiskit's public Weyl API."""
    try:
        from qiskit.synthesis import TwoQubitWeylDecomposition  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "SU(4) decomposition requires the optional 'qiskit' dependency; "
            "install rqm-entanglement[qiskit]."
        ) from exc

    value = np.asarray(unitary, dtype=np.complex128)
    decomposition = TwoQubitWeylDecomposition(value, fidelity=None)
    # Qiskit's matrix formula is kron(K*l, K*r).  Under Qiskit's
    # little-endian circuit convention, the right factor acts on q0.
    return QiskitWeylResult(
        a=float(decomposition.a),
        b=float(decomposition.b),
        c=float(decomposition.c),
        global_phase=float(decomposition.global_phase),
        left_q0_matrix=np.asarray(decomposition.K1r, dtype=np.complex128),
        left_q1_matrix=np.asarray(decomposition.K1l, dtype=np.complex128),
        right_q0_matrix=np.asarray(decomposition.K2r, dtype=np.complex128),
        right_q1_matrix=np.asarray(decomposition.K2l, dtype=np.complex128),
    )
