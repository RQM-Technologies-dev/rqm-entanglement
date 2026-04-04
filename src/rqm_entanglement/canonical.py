"""Canonical two-qubit entangling gate via commuting Pauli-pair rotations.

The canonical family is::

    U_ent(c1, c2, c3) = exp[-i/2 (c1 XX + c2 YY + c3 ZZ)]

Because XX, YY, ZZ mutually commute, the matrix exponential factors exactly::

    U_ent = xx_rotation(c1) @ yy_rotation(c2) @ zz_rotation(c3)

Each factor is computed analytically::

    exp[-i θ/2 P] = cos(θ/2) I4 - i sin(θ/2) P     for P ∈ {XX, YY, ZZ}
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from rqm_entanglement.constants import I4, XX, YY, ZZ


def xx_rotation(theta: float) -> NDArray[np.complex128]:
    """Return exp[-i θ/2 XX] = cos(θ/2) I4 - i sin(θ/2) XX."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.asarray(c * I4 - 1j * s * XX, dtype=np.complex128)


def yy_rotation(theta: float) -> NDArray[np.complex128]:
    """Return exp[-i θ/2 YY] = cos(θ/2) I4 - i sin(θ/2) YY."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.asarray(c * I4 - 1j * s * YY, dtype=np.complex128)


def zz_rotation(theta: float) -> NDArray[np.complex128]:
    """Return exp[-i θ/2 ZZ] = cos(θ/2) I4 - i sin(θ/2) ZZ."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.asarray(c * I4 - 1j * s * ZZ, dtype=np.complex128)


def canonical_entangler(c1: float, c2: float, c3: float) -> NDArray[np.complex128]:
    """Return U_ent(c1, c2, c3) = exp[-i/2 (c1 XX + c2 YY + c3 ZZ)].

    Computed as ``xx_rotation(c1) @ yy_rotation(c2) @ zz_rotation(c3)``.
    Valid because XX, YY, ZZ mutually commute.
    """
    return xx_rotation(c1) @ yy_rotation(c2) @ zz_rotation(c3)
