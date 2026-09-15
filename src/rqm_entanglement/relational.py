"""Adaptive relational representations for two-qubit RQM compilation.

The compiler rule is simple: keep the lowest-dimensional representation that
preserves the proven structure, and promote only when an operation leaves that
representation closed.

This module does not replace standard complex quantum mechanics.  Every
representation materializes exactly to the existing two-qubit unitary/state
semantics, so the compressed forms are compiler IRs with a conventional
verification path.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from rqm_entanglement.adapters.rqm_core_adapter import QuaternionTuple
from rqm_entanglement.canonical import canonical_entangler, xx_rotation, yy_rotation, zz_rotation
from rqm_entanglement.su4 import QuaternionCartanBlock, rotation_to_weyl_coordinates

Axis = Literal["xx", "yy", "zz"]
IDENTITY_QUATERNION: QuaternionTuple = (1.0, 0.0, 0.0, 0.0)
_TOL = 1e-12


class RelationalLevel(str, Enum):
    """Ordered conceptual levels used by the adaptive relational IR."""

    BELL = "bell"
    AXIS_HINGE = "axis_hinge"
    CARTAN_RELATION = "cartan_relation"
    QUATERNION_CARTAN = "quaternion_cartan"


@dataclass(frozen=True)
class BellHinge:
    """Compressed Bell-sector state representation.

    ``parity`` is the ZZ eigenvalue and ``phase`` is the XX eigenvalue.
    The four sign pairs identify Phi+/Phi-/Psi+/Psi- respectively.  The
    maximally-entangled hinge angle is fixed at pi/2.
    """

    parity: Literal[-1, 1]
    phase: Literal[-1, 1]
    theta: float = math.pi / 2.0

    def __post_init__(self) -> None:
        if self.parity not in (-1, 1) or self.phase not in (-1, 1):
            raise ValueError("BellHinge parity and phase must each be ±1")
        if not math.isclose(self.theta, math.pi / 2.0, abs_tol=_TOL, rel_tol=0.0):
            raise ValueError("BellHinge is the maximally-entangled theta=pi/2 sector")

    @property
    def name(self) -> str:
        return {
            (1, 1): "phi_plus",
            (1, -1): "phi_minus",
            (-1, 1): "psi_plus",
            (-1, -1): "psi_minus",
        }[(self.parity, self.phase)]

    def to_state(self) -> NDArray[np.complex128]:
        """Materialize the conventional Bell state in |00>,|01>,|10>,|11> order."""
        sign = float(self.phase)
        norm = 1.0 / math.sqrt(2.0)
        if self.parity == 1:
            return np.array([norm, 0.0, 0.0, sign * norm], dtype=np.complex128)
        return np.array([0.0, norm, sign * norm, 0.0], dtype=np.complex128)

    def apply_local_pauli(self, pauli: Literal["i", "x", "z", "xz"]) -> "BellHinge":
        """Update the Bell symmetry sector without expanding the state vector.

        Applying X to either one qubit flips parity.  Applying Z to either one
        qubit flips the XX phase sector.  XZ flips both.
        """
        if pauli == "i":
            return self
        if pauli == "x":
            return BellHinge(-self.parity, self.phase)
        if pauli == "z":
            return BellHinge(self.parity, -self.phase)
        if pauli == "xz":
            return BellHinge(-self.parity, -self.phase)
        raise ValueError(f"unsupported Bell-sector local Pauli: {pauli}")


@dataclass(frozen=True)
class AxisHinge:
    """One-axis nonlocal interaction exp[-i theta/2 P⊗P]."""

    axis: Axis
    theta: float

    def __post_init__(self) -> None:
        if self.axis not in ("xx", "yy", "zz"):
            raise ValueError("axis must be one of: xx, yy, zz")
        if not math.isfinite(self.theta):
            raise ValueError("theta must be finite")

    def to_unitary(self) -> NDArray[np.complex128]:
        if self.axis == "xx":
            return xx_rotation(self.theta)
        if self.axis == "yy":
            return yy_rotation(self.theta)
        return zz_rotation(self.theta)

    def compose_same_axis(self, other: "AxisHinge") -> "AxisHinge":
        if self.axis != other.axis:
            raise ValueError("different hinge axes require promotion to CartanRelation")
        return AxisHinge(self.axis, self.theta + other.theta)

    def promote(self) -> "CartanRelation":
        coords = {"xx": (self.theta, 0.0, 0.0), "yy": (0.0, self.theta, 0.0), "zz": (0.0, 0.0, self.theta)}
        return CartanRelation(*coords[self.axis])


@dataclass(frozen=True)
class CartanRelation:
    """Three commuting nonlocal hinge coordinates in rotation convention."""

    c1: float = 0.0
    c2: float = 0.0
    c3: float = 0.0

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) for v in (self.c1, self.c2, self.c3)):
            raise ValueError("Cartan coordinates must be finite")

    def to_unitary(self) -> NDArray[np.complex128]:
        return canonical_entangler(self.c1, self.c2, self.c3)

    def compose(self, other: "CartanRelation") -> "CartanRelation":
        """Compose aligned Cartan cores exactly by coordinate addition."""
        return CartanRelation(self.c1 + other.c1, self.c2 + other.c2, self.c3 + other.c3)

    def minimize(self, *, atol: float = _TOL) -> AxisHinge | "CartanRelation":
        active = [("xx", self.c1), ("yy", self.c2), ("zz", self.c3)]
        nonzero = [(axis, value) for axis, value in active if abs(value) > atol]
        if len(nonzero) == 1:
            axis, value = nonzero[0]
            return AxisHinge(axis, value)  # type: ignore[arg-type]
        return self

    def promote(self) -> QuaternionCartanBlock:
        a, b, c = rotation_to_weyl_coordinates(self.c1, self.c2, self.c3)
        return QuaternionCartanBlock.from_components(
            left_q0=IDENTITY_QUATERNION,
            left_q1=IDENTITY_QUATERNION,
            cartan_a=a,
            cartan_b=b,
            cartan_c=c,
            right_q0=IDENTITY_QUATERNION,
            right_q1=IDENTITY_QUATERNION,
            global_phase=0.0,
        )


RelationalOperator = AxisHinge | CartanRelation | QuaternionCartanBlock


def relational_level(value: BellHinge | RelationalOperator) -> RelationalLevel:
    if isinstance(value, BellHinge):
        return RelationalLevel.BELL
    if isinstance(value, AxisHinge):
        return RelationalLevel.AXIS_HINGE
    if isinstance(value, CartanRelation):
        return RelationalLevel.CARTAN_RELATION
    if isinstance(value, QuaternionCartanBlock):
        return RelationalLevel.QUATERNION_CARTAN
    raise TypeError(f"unsupported relational representation: {type(value)!r}")


def promote_with_local_frames(
    relation: AxisHinge | CartanRelation,
    *,
    left_q0: QuaternionTuple = IDENTITY_QUATERNION,
    left_q1: QuaternionTuple = IDENTITY_QUATERNION,
    right_q0: QuaternionTuple = IDENTITY_QUATERNION,
    right_q1: QuaternionTuple = IDENTITY_QUATERNION,
    global_phase: float = 0.0,
) -> QuaternionCartanBlock:
    """Promote a nonlocal relation when independent local frames are required."""
    cartan = relation.promote() if isinstance(relation, AxisHinge) else relation
    if isinstance(cartan, AxisHinge):  # defensive; AxisHinge.promote returns CartanRelation
        cartan = cartan.promote()
    a, b, c = rotation_to_weyl_coordinates(cartan.c1, cartan.c2, cartan.c3)
    return QuaternionCartanBlock.from_components(
        left_q0=left_q0,
        left_q1=left_q1,
        cartan_a=a,
        cartan_b=b,
        cartan_c=c,
        right_q0=right_q0,
        right_q1=right_q1,
        global_phase=global_phase,
    )


def compose_relations(left: RelationalOperator, right: RelationalOperator) -> RelationalOperator:
    """Compose while staying in the smallest representation known to be closed.

    Same-axis hinges stay one-dimensional.  Different axis hinges promote to
    the commuting three-coordinate Cartan relation.  Aligned Cartan relations
    remain Cartan.  Any composition involving local quaternion shells is
    recanonicalized through ``QuaternionCartanBlock.from_unitary``; that exact
    general path uses the repository's optional Qiskit Weyl authority.
    """
    if isinstance(left, AxisHinge) and isinstance(right, AxisHinge):
        if left.axis == right.axis:
            return left.compose_same_axis(right)
        return left.promote().compose(right.promote()).minimize()

    if isinstance(left, (AxisHinge, CartanRelation)) and isinstance(right, (AxisHinge, CartanRelation)):
        lcartan = left.promote() if isinstance(left, AxisHinge) else left
        rcartan = right.promote() if isinstance(right, AxisHinge) else right
        return lcartan.compose(rcartan).minimize()

    lmat = left.to_unitary() if hasattr(left, "to_unitary") else left.to_unitary()
    rmat = right.to_unitary() if hasattr(right, "to_unitary") else right.to_unitary()
    # Function composition convention: left after right.
    return QuaternionCartanBlock.from_unitary(np.asarray(lmat) @ np.asarray(rmat))


def compress_unitary(unitary: NDArray[np.complex128], *, atol: float = _TOL) -> RelationalOperator:
    """Decompose an arbitrary two-qubit unitary and demote when structure allows.

    General decomposition uses ``QuaternionCartanBlock.from_unitary``.  If all
    local quaternion shells are identity (within ``atol``), the result is
    reduced to CartanRelation and, when only one coordinate is active, further
    to AxisHinge.
    """
    block = QuaternionCartanBlock.from_unitary(np.asarray(unitary, dtype=np.complex128))

    def identity(q: QuaternionTuple) -> bool:
        return bool(np.allclose(np.asarray(q, dtype=float), np.asarray(IDENTITY_QUATERNION), atol=atol, rtol=0.0))

    if not all(identity(q) for q in (block.left_q0, block.left_q1, block.right_q0, block.right_q1)):
        return block
    if abs(block.global_phase) > atol:
        return block

    # Stored Weyl coordinates exp[i(aXX+bYY+cZZ)] map to rotation coordinates -2(a,b,c).
    relation = CartanRelation(-2.0 * block.cartan_a, -2.0 * block.cartan_b, -2.0 * block.cartan_c)
    return relation.minimize(atol=atol)
