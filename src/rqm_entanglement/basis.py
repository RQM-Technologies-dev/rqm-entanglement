"""Computational-basis two-qubit state vectors.

Basis ordering: |00>, |01>, |10>, |11>
Qubit 0 is the more-significant (left) index.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def ket00() -> NDArray[np.complex128]:
    """Return |00> = [1, 0, 0, 0]^T."""
    return np.array([1, 0, 0, 0], dtype=np.complex128)


def ket01() -> NDArray[np.complex128]:
    """Return |01> = [0, 1, 0, 0]^T."""
    return np.array([0, 1, 0, 0], dtype=np.complex128)


def ket10() -> NDArray[np.complex128]:
    """Return |10> = [0, 0, 1, 0]^T."""
    return np.array([0, 0, 1, 0], dtype=np.complex128)


def ket11() -> NDArray[np.complex128]:
    """Return |11> = [0, 0, 0, 1]^T."""
    return np.array([0, 0, 0, 1], dtype=np.complex128)


def basis_state(bits: str) -> NDArray[np.complex128]:
    """Return the computational basis state for *bits* in {'00','01','10','11'}.

    >>> basis_state('10')
    array([0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j])
    """
    _map = {"00": ket00, "01": ket01, "10": ket10, "11": ket11}
    if bits not in _map:
        raise ValueError(f"bits must be one of {list(_map)!r}, got {bits!r}")
    return _map[bits]()


def computational_basis() -> tuple[
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.complex128],
]:
    """Return (|00>, |01>, |10>, |11>) as a 4-tuple of state vectors."""
    return ket00(), ket01(), ket10(), ket11()
