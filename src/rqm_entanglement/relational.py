"""Adaptive relational representations for two-qubit RQM compilation.

The compiler rule is simple: keep the lowest-dimensional representation that
preserves the proven structure, and promote only when an operation leaves that
representation closed.

This module does not replace standard complex quantum mechanics. Every
representation materializes exactly to the existing two-qubit unitary/state
semantics, so the compressed forms are compiler IRs with a conventional
verification path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from rqm_entanglement.adapters.rqm_core_adapter import (
    QuaternionTuple,
    quaternion_to_su2_matrix,
)
from rqm_entanglement.canonical import (
    canonical_entangler,
    xx_rotation,
    yy_rotation,
    zz_rotation,
)
from rqm_entanglement.su4 import QuaternionCartanBlock

Axis = Literal["xx", "yy", "zz"]
IDENTITY_QUATERNION: QuaternionTuple = (1.0, 0.0, 0.0, 0.0)
_TOL = 1e-12


class RelationalLevel(StrEnum):
    """Ordered conceptual levels used by the adaptive relational IR."""

    BELL = "bell"
    AXIS_HINGE = "axis_hinge"
    CARTAN_RELATION = "cartan_relation"
    QUATERNION_CARTAN = "quaternion_cartan"


@dataclass(frozen=True)
class BellHinge:
    """Compressed Bell-sector state representation."""

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
        """Materialize the conventional Bell state."""
        sign = float(self.phase)
        norm = 1.0 / math.sqrt(2.0)
        if self.parity == 1:
            return np.array([norm, 0.0, 0.0, sign * norm], dtype=np.complex128)
        return np.array([0.0, norm, sign * norm, 0.0], dtype=np.complex128)

    def apply_local_pauli(self, pauli: Literal["i", "x", "z", "xz"]) -> BellHinge:
        """Update the Bell symmetry sector without expanding the state vector."""
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

    def compose_same_axis(self, other: AxisHinge) -> AxisHinge:
        if self.axis != other.axis:
            raise ValueError("different hinge axes require promotion to CartanRelation")
        return AxisHinge(self.axis, self.theta + other.theta)

    def promote(self) -> CartanRelation:
        coords: dict[Axis, tuple[float, float, float]] = {
            "xx": (self.theta, 0.0, 0.0),
            "yy": (0.0, self.theta, 0.0),
            "zz": (0.0, 0.0, self.theta),
        }
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

    def compose(self, other: CartanRelation) -> CartanRelation:
        """Compose aligned Cartan cores exactly by coordinate addition."""
        return CartanRelation(
            self.c1 + other.c1,
            self.c2 + other.c2,
            self.c3 + other.c3,
        )

    def minimize(self, *, atol: float = _TOL) -> AxisHinge | CartanRelation:
        active: tuple[tuple[Axis, float], ...] = (
            ("xx", self.c1),
            ("yy", self.c2),
            ("zz", self.c3),
        )
        nonzero = [(axis, value) for axis, value in active if abs(value) > atol]
        if len(nonzero) == 1:
            axis, value = nonzero[0]
            return AxisHinge(axis, value)
        return self

    def promote(self) -> QuaternionCartanBlock:
        """Promote through exact SU(4) recanonicalization into the Weyl chamber."""
        return QuaternionCartanBlock.from_unitary(self.to_unitary())


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


def _local_matrix(q0: QuaternionTuple, q1: QuaternionTuple) -> NDArray[np.complex128]:
    return np.asarray(
        np.kron(quaternion_to_su2_matrix(q1), quaternion_to_su2_matrix(q0)),
        dtype=np.complex128,
    )


def promote_with_local_frames(
    relation: AxisHinge | CartanRelation,
    *,
    left_q0: QuaternionTuple = IDENTITY_QUATERNION,
    left_q1: QuaternionTuple = IDENTITY_QUATERNION,
    right_q0: QuaternionTuple = IDENTITY_QUATERNION,
    right_q1: QuaternionTuple = IDENTITY_QUATERNION,
    global_phase: float = 0.0,
) -> QuaternionCartanBlock:
    """Promote a relation plus local frames via exact SU(4) recanonicalization."""
    cartan = relation.promote() if isinstance(relation, AxisHinge) else relation
    unitary = (
        np.exp(1j * global_phase)
        * _local_matrix(left_q0, left_q1)
        @ cartan.to_unitary()
        @ _local_matrix(right_q0, right_q1)
    )
    return QuaternionCartanBlock.from_unitary(unitary)


def compose_relations(
    left: RelationalOperator,
    right: RelationalOperator,
) -> RelationalOperator:
    """Compose while staying in the smallest representation known to be closed."""
    if isinstance(left, AxisHinge) and isinstance(right, AxisHinge):
        if left.axis == right.axis:
            return left.compose_same_axis(right)
        return left.promote().compose(right.promote()).minimize()

    simple = (AxisHinge, CartanRelation)
    if isinstance(left, simple) and isinstance(right, simple):
        lcartan = left.promote() if isinstance(left, AxisHinge) else left
        rcartan = right.promote() if isinstance(right, AxisHinge) else right
        return lcartan.compose(rcartan).minimize()

    return QuaternionCartanBlock.from_unitary(left.to_unitary() @ right.to_unitary())


def compress_unitary(
    unitary: NDArray[np.complex128],
    *,
    atol: float = _TOL,
) -> RelationalOperator:
    """Decompose an arbitrary unitary and demote when proven structure allows."""
    block = QuaternionCartanBlock.from_unitary(
        np.asarray(unitary, dtype=np.complex128)
    )

    def identity(quaternion: QuaternionTuple) -> bool:
        return bool(
            np.allclose(
                np.asarray(quaternion, dtype=float),
                np.asarray(IDENTITY_QUATERNION),
                atol=atol,
                rtol=0.0,
            )
        )

    quaternions = (
        block.left_q0,
        block.left_q1,
        block.right_q0,
        block.right_q1,
    )
    if not all(identity(quaternion) for quaternion in quaternions):
        return block
    if abs(block.global_phase) > atol:
        return block

    relation = CartanRelation(
        -2.0 * block.cartan_a,
        -2.0 * block.cartan_b,
        -2.0 * block.cartan_c,
    )
    return relation.minimize(atol=atol)
