"""Quaternion-Cartan representation and deterministic SU(4) classification.

The stored Weyl coordinates use the EXP-012/Qiskit convention
``exp[i(a XX + b YY + c ZZ)]``.  The repository's public rotation helpers use
``exp[-i/2(c1 XX + c2 YY + c3 ZZ)]``.  Exact conversion helpers are provided;
the two conventions are never mixed implicitly.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rqm_entanglement.adapters.qiskit_weyl import (
    QISKIT_FACTOR_ORDER,
    decompose_with_qiskit,
)
from rqm_entanglement.adapters.rqm_core_adapter import (
    QuaternionTuple,
    canonicalize_quaternion_sign_with_phase,
    normalize_local_su2_factor,
    quaternion_to_su2_matrix,
)
from rqm_entanglement.canonical import canonical_entangler
from rqm_entanglement.classify import operator_schmidt_rank
from rqm_entanglement.constants import I4
from rqm_entanglement.validation import is_unitary, normalize_global_phase

CONVENTION_VERSION = "rqm-su4q-v1-exp+iabc-" + QISKIT_FACTOR_ORDER
WEYL_TOLERANCE = 1e-10
RECONSTRUCTION_TOLERANCE = 1e-10
FINGERPRINT_QUANTUM = 1e-9
FULL_HASH_QUANTUM = 1e-13
_PHASE_TOLERANCE = 1e-14
_BINARY_MAGIC = b"RQSU"
_BINARY_VERSION = 1


def normalize_global_phase_value(phase: float) -> float:
    """Normalize a finite phase to the documented interval ``[-pi, pi)``."""
    if not math.isfinite(phase):
        raise ValueError("global phase must be finite")
    wrapped = (float(phase) + math.pi) % (2.0 * math.pi) - math.pi
    if abs(wrapped) <= _PHASE_TOLERANCE:
        return 0.0
    if abs(wrapped + math.pi) <= _PHASE_TOLERANCE:
        return -math.pi
    return wrapped


def weyl_to_rotation_coordinates(a: float, b: float, c: float) -> tuple[float, float, float]:
    """Convert ``exp[i(aXX+bYY+cZZ)]`` to ``exp[-i/2(c1XX+c2YY+c3ZZ)]``."""
    return -2.0 * float(a), -2.0 * float(b), -2.0 * float(c)


def rotation_to_weyl_coordinates(
    c1: float, c2: float, c3: float
) -> tuple[float, float, float]:
    """Convert rotation coordinates to the stored EXP-012/Qiskit convention."""
    return -0.5 * float(c1), -0.5 * float(c2), -0.5 * float(c3)


def cartan_core_from_weyl(a: float, b: float, c: float) -> NDArray[np.complex128]:
    """Return ``exp[i(aXX+bYY+cZZ)]`` through the existing canonical entangler."""
    return canonical_entangler(*weyl_to_rotation_coordinates(a, b, c))


def in_weyl_chamber(
    coordinates: Iterable[float], *, tolerance: float = WEYL_TOLERANCE
) -> bool:
    """Return whether coordinates lie in Qiskit's canonical Weyl chamber."""
    values = tuple(float(value) for value in coordinates)
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        return False
    a, b, c = values
    return (
        a <= math.pi / 4.0 + tolerance
        and a >= b - tolerance
        and b >= abs(c) - tolerance
        and a >= -tolerance
        and b >= -tolerance
    )


def phase_aligned_operator_error(
    reference: NDArray[np.complex128], candidate: NDArray[np.complex128]
) -> tuple[float, float]:
    """Return maximum entry error after optimal global-phase alignment."""
    left = np.asarray(reference, dtype=np.complex128)
    right = np.asarray(candidate, dtype=np.complex128)
    overlap = np.trace(right.conj().T @ left)
    phase = float(np.angle(overlap)) if abs(overlap) > 0.0 else 0.0
    aligned = right * np.exp(1j * phase)
    return float(np.max(np.abs(left - aligned))), phase


def _finite_tuple(values: Iterable[float], length: int) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != length or not np.all(np.isfinite(result)):
        raise ValueError(f"expected {length} finite float values")
    return tuple(0.0 if value == 0.0 else value for value in result)


def _encode_text(value: str | None) -> bytes:
    payload = ("" if value is None else value).encode("utf-8")
    if len(payload) > 65535:
        raise ValueError("metadata field is too long")
    return struct.pack("<H", len(payload)) + payload


def _decode_text(payload: bytes, offset: int) -> tuple[str, int]:
    if offset + 2 > len(payload):
        raise ValueError("truncated metadata length")
    length = struct.unpack_from("<H", payload, offset)[0]
    offset += 2
    if offset + length > len(payload):
        raise ValueError("truncated metadata")
    return payload[offset : offset + length].decode("utf-8"), offset + length


@dataclass(frozen=True)
class QuaternionCartanBlock:
    """Immutable local-quaternion shells plus canonical Weyl coordinates."""

    left_q0: QuaternionTuple
    left_q1: QuaternionTuple
    cartan_a: float
    cartan_b: float
    cartan_c: float
    right_q0: QuaternionTuple
    right_q1: QuaternionTuple
    global_phase: float
    convention_version: str = CONVENTION_VERSION
    source_hash: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "left_q0", _finite_tuple(self.left_q0, 4))
        object.__setattr__(self, "left_q1", _finite_tuple(self.left_q1, 4))
        object.__setattr__(self, "right_q0", _finite_tuple(self.right_q0, 4))
        object.__setattr__(self, "right_q1", _finite_tuple(self.right_q1, 4))
        object.__setattr__(self, "cartan_a", float(self.cartan_a))
        object.__setattr__(self, "cartan_b", float(self.cartan_b))
        object.__setattr__(self, "cartan_c", float(self.cartan_c))
        object.__setattr__(self, "global_phase", normalize_global_phase_value(self.global_phase))
        if not isinstance(self.convention_version, str) or not self.convention_version:
            raise ValueError("convention_version is required")
        if self.source_hash is not None and not isinstance(self.source_hash, str):
            raise ValueError("source_hash must be text or None")
        result = self.validate()
        if not result["valid"]:
            raise ValueError("; ".join(result["failures"]))

    @property
    def cartan(self) -> tuple[float, float, float]:
        """Return stored ``(a,b,c)`` Weyl coordinates."""
        return self.cartan_a, self.cartan_b, self.cartan_c

    @classmethod
    def from_components(
        cls,
        *,
        left_q0: Iterable[float],
        left_q1: Iterable[float],
        cartan_a: float,
        cartan_b: float,
        cartan_c: float,
        right_q0: Iterable[float],
        right_q1: Iterable[float],
        global_phase: float = 0.0,
        convention_version: str = CONVENTION_VERSION,
        source_hash: str | None = None,
    ) -> QuaternionCartanBlock:
        """Build a canonical block while preserving exact sign/phase semantics."""
        phase = float(global_phase)
        quaternions: list[QuaternionTuple] = []
        for quaternion in (left_q0, left_q1, right_q0, right_q1):
            canonical, phase, _ = canonicalize_quaternion_sign_with_phase(quaternion, phase)
            quaternions.append(canonical)
        return cls(
            left_q0=quaternions[0],
            left_q1=quaternions[1],
            cartan_a=cartan_a,
            cartan_b=cartan_b,
            cartan_c=cartan_c,
            right_q0=quaternions[2],
            right_q1=quaternions[3],
            global_phase=phase,
            convention_version=convention_version,
            source_hash=source_hash,
        )

    @classmethod
    def from_unitary(
        cls,
        unitary: NDArray[np.complex128],
        *,
        source_hash: str | None = None,
    ) -> QuaternionCartanBlock:
        """Decompose a finite unitary through the optional Qiskit adapter."""
        value = np.asarray(unitary, dtype=np.complex128)
        if value.shape != (4, 4) or not np.all(np.isfinite(value)):
            raise ValueError("source unitary must be a finite 4x4 matrix")
        if not is_unitary(value, atol=1e-10, rtol=0.0):
            residual = float(np.max(np.abs(value.conj().T @ value - I4)))
            raise ValueError(f"source matrix is not unitary: {residual}")
        authority = decompose_with_qiskit(value)
        quaternions: list[QuaternionTuple] = []
        phase = authority.global_phase
        for matrix in (
            authority.left_q0_matrix,
            authority.left_q1_matrix,
            authority.right_q0_matrix,
            authority.right_q1_matrix,
        ):
            quaternion, removed_phase, _, _ = normalize_local_su2_factor(matrix)
            quaternions.append(quaternion)
            phase += removed_phase
        return cls(
            left_q0=quaternions[0],
            left_q1=quaternions[1],
            cartan_a=authority.a,
            cartan_b=authority.b,
            cartan_c=authority.c,
            right_q0=quaternions[2],
            right_q1=quaternions[3],
            global_phase=phase,
            convention_version=CONVENTION_VERSION,
            source_hash=source_hash,
        )

    def to_unitary(self) -> NDArray[np.complex128]:
        """Reconstruct the represented U(4) matrix in ``|00>,|01>,|10>,|11>`` order."""
        left = np.kron(
            quaternion_to_su2_matrix(self.left_q1),
            quaternion_to_su2_matrix(self.left_q0),
        )
        right = np.kron(
            quaternion_to_su2_matrix(self.right_q1),
            quaternion_to_su2_matrix(self.right_q0),
        )
        return np.asarray(
            np.exp(1j * self.global_phase)
            * left
            @ cartan_core_from_weyl(*self.cartan)
            @ right,
            dtype=np.complex128,
        )

    def validate(self) -> dict[str, Any]:
        """Return deterministic validation details without mutating the block."""
        failures: list[str] = []
        norms: dict[str, float] = {}
        for name in ("left_q0", "left_q1", "right_q0", "right_q1"):
            values = np.asarray(getattr(self, name), dtype=np.float64)
            norm = float(np.linalg.norm(values))
            norms[name] = norm
            if not np.all(np.isfinite(values)) or abs(norm - 1.0) > 1e-12:
                failures.append(f"{name} is not a finite unit quaternion")
                continue
            canonical, _, flipped = canonicalize_quaternion_sign_with_phase(values, 0.0)
            if flipped or not np.allclose(values, canonical, atol=1e-15, rtol=0.0):
                failures.append(f"{name} violates the canonical sign convention")
        if not in_weyl_chamber(self.cartan):
            failures.append("Cartan coordinates are outside the Weyl chamber")
        if self.convention_version != CONVENTION_VERSION:
            failures.append("unsupported convention_version")
        if normalize_global_phase_value(self.global_phase) != self.global_phase:
            failures.append("global phase is not normalized")
        return {
            "valid": not failures,
            "failures": failures,
            "quaternion_norms": norms,
            "weyl_chamber_valid": in_weyl_chamber(self.cartan),
            "convention_version": self.convention_version,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned JSON-compatible representation."""
        return {
            "left_q0": list(self.left_q0),
            "left_q1": list(self.left_q1),
            "cartan_a": self.cartan_a,
            "cartan_b": self.cartan_b,
            "cartan_c": self.cartan_c,
            "right_q0": list(self.right_q0),
            "right_q1": list(self.right_q1),
            "global_phase": self.global_phase,
            "convention_version": self.convention_version,
            "source_hash": self.source_hash,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> QuaternionCartanBlock:
        """Decode and validate a JSON-compatible representation."""
        if not isinstance(payload, dict):
            raise ValueError("quaternion-Cartan payload must be an object")
        return cls(
            left_q0=payload["left_q0"],
            left_q1=payload["left_q1"],
            cartan_a=payload["cartan_a"],
            cartan_b=payload["cartan_b"],
            cartan_c=payload["cartan_c"],
            right_q0=payload["right_q0"],
            right_q1=payload["right_q1"],
            global_phase=payload["global_phase"],
            convention_version=payload["convention_version"],
            source_hash=payload.get("source_hash"),
        )

    def _numeric_values(self) -> tuple[float, ...]:
        return (
            *self.left_q0,
            *self.left_q1,
            *self.cartan,
            *self.right_q0,
            *self.right_q1,
            self.global_phase,
        )

    def to_binary(self) -> bytes:
        """Return deterministic little-endian binary serialization."""
        return (
            _BINARY_MAGIC
            + struct.pack("<B20d", _BINARY_VERSION, *self._numeric_values())
            + _encode_text(self.convention_version)
            + _encode_text(self.source_hash)
        )

    @classmethod
    def from_binary(cls, payload: bytes) -> QuaternionCartanBlock:
        """Decode deterministic binary serialization and reject trailing data."""
        numeric_size = struct.calcsize("<B20d")
        if len(payload) < 4 + numeric_size or payload[:4] != _BINARY_MAGIC:
            raise ValueError("invalid quaternion-Cartan binary header")
        unpacked = struct.unpack_from("<B20d", payload, 4)
        if unpacked[0] != _BINARY_VERSION:
            raise ValueError("unsupported quaternion-Cartan binary version")
        values = unpacked[1:]
        offset = 4 + numeric_size
        convention, offset = _decode_text(payload, offset)
        source_hash, offset = _decode_text(payload, offset)
        if offset != len(payload):
            raise ValueError("unexpected trailing quaternion-Cartan binary data")
        return cls(
            left_q0=values[0:4],
            left_q1=values[4:8],
            cartan_a=values[8],
            cartan_b=values[9],
            cartan_c=values[10],
            right_q0=values[11:15],
            right_q1=values[15:19],
            global_phase=values[19],
            convention_version=convention,
            source_hash=source_hash or None,
        )

    def full_canonical_hash(self) -> str:
        """Hash the full unitary after deterministic global-phase normalization."""
        normalized = normalize_global_phase(self.to_unitary(), atol=1e-15)
        values = np.concatenate((normalized.real.ravel(), normalized.imag.ravel()))
        quantized = [int(round(float(value) / FULL_HASH_QUANTUM)) for value in values]
        encoded = json.dumps(
            {"version": 1, "quantum": FULL_HASH_QUANTUM, "matrix": quantized},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def nonlocal_fingerprint(self) -> str:
        """Hash tolerance-quantized Weyl coordinates for local-equivalence matching."""
        quantized = [int(round(value / FINGERPRINT_QUANTUM)) for value in self.cartan]
        encoded = json.dumps(
            {
                "convention_version": self.convention_version,
                "fingerprint_quantum": FINGERPRINT_QUANTUM,
                "cartan_quantized": quantized,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SU4Classification:
    """Typed deterministic Weyl classification result."""

    cartan: tuple[float, float, float]
    weyl_chamber_valid: bool
    nonlocal_fingerprint: str
    full_canonical_hash: str
    operator_schmidt_rank: int
    class_label: str
    tolerance: float
    convention_version: str
    nonlocal_operator: bool
    entangling_gate: bool
    perfect_entangler: bool
    swap_like: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible classification payload."""
        return {
            "cartan": list(self.cartan),
            "weyl_chamber_valid": self.weyl_chamber_valid,
            "nonlocal_fingerprint": self.nonlocal_fingerprint,
            "full_canonical_hash": self.full_canonical_hash,
            "operator_schmidt_rank": self.operator_schmidt_rank,
            "class_label": self.class_label,
            "tolerance": self.tolerance,
            "convention_version": self.convention_version,
            "nonlocal_operator": self.nonlocal_operator,
            "entangling_gate": self.entangling_gate,
            "perfect_entangler": self.perfect_entangler,
            "swap_like": self.swap_like,
        }


_LANDMARKS: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("local_identity", (0.0, 0.0, 0.0)),
    ("cnot_cz_class", (math.pi / 4.0, 0.0, 0.0)),
    ("iswap_class", (math.pi / 4.0, math.pi / 4.0, 0.0)),
    ("sqrt_iswap_class", (math.pi / 8.0, math.pi / 8.0, 0.0)),
    ("swap_class", (math.pi / 4.0, math.pi / 4.0, math.pi / 4.0)),
    ("sqrt_swap_class", (math.pi / 8.0, math.pi / 8.0, math.pi / 8.0)),
    ("b_gate_class", (math.pi / 4.0, math.pi / 8.0, 0.0)),
)


def _class_label(coordinates: tuple[float, float, float], tolerance: float) -> str:
    for label, landmark in _LANDMARKS:
        if np.allclose(coordinates, landmark, atol=tolerance, rtol=0.0):
            return label
        # The SWAP edge can be returned with signed c by the canonical authority.
        if label in {"swap_class", "sqrt_swap_class"} and np.allclose(
            (coordinates[0], coordinates[1], abs(coordinates[2])),
            landmark,
            atol=tolerance,
            rtol=0.0,
        ):
            return label
    return "generic_interior"


def classify_su4(
    unitary_or_block: NDArray[np.complex128] | QuaternionCartanBlock,
    *,
    tolerance: float = WEYL_TOLERANCE,
) -> SU4Classification:
    """Classify a block or decompose and classify a finite two-qubit unitary."""
    block = (
        unitary_or_block
        if isinstance(unitary_or_block, QuaternionCartanBlock)
        else QuaternionCartanBlock.from_unitary(unitary_or_block)
    )
    label = _class_label(block.cartan, tolerance)
    rank = operator_schmidt_rank(block.to_unitary(), atol=tolerance)
    nonlocal_operator = rank > 1
    swap_like = label == "swap_class"
    entangling_gate = nonlocal_operator and not swap_like
    a, b, c = block.cartan
    perfect_entangler = bool(
        entangling_gate
        and a + b >= math.pi / 4.0 - tolerance
        and b + c <= math.pi / 4.0 + tolerance
    )
    return SU4Classification(
        cartan=block.cartan,
        weyl_chamber_valid=in_weyl_chamber(block.cartan, tolerance=tolerance),
        nonlocal_fingerprint=block.nonlocal_fingerprint(),
        full_canonical_hash=block.full_canonical_hash(),
        operator_schmidt_rank=rank,
        class_label=label,
        tolerance=float(tolerance),
        convention_version=block.convention_version,
        nonlocal_operator=nonlocal_operator,
        entangling_gate=entangling_gate,
        perfect_entangler=perfect_entangler,
        swap_like=swap_like,
    )


def decompose_su4(unitary: NDArray[np.complex128]) -> QuaternionCartanBlock:
    """Return the canonical production quaternion-Cartan block."""
    return QuaternionCartanBlock.from_unitary(unitary)


def reconstruct_su4(block: QuaternionCartanBlock) -> NDArray[np.complex128]:
    """Reconstruct a validated quaternion-Cartan block."""
    if not isinstance(block, QuaternionCartanBlock):
        raise TypeError("block must be a QuaternionCartanBlock")
    return block.to_unitary()


def nonlocal_fingerprint(
    unitary_or_block: NDArray[np.complex128] | QuaternionCartanBlock,
) -> str:
    """Return the stable local-equivalence fingerprint."""
    block = (
        unitary_or_block
        if isinstance(unitary_or_block, QuaternionCartanBlock)
        else QuaternionCartanBlock.from_unitary(unitary_or_block)
    )
    return block.nonlocal_fingerprint()


def are_locally_equivalent(
    a: NDArray[np.complex128] | QuaternionCartanBlock,
    b: NDArray[np.complex128] | QuaternionCartanBlock,
    *,
    tolerance: float = WEYL_TOLERANCE,
) -> bool:
    """Return whether two operators share Weyl coordinates within tolerance."""
    left = a if isinstance(a, QuaternionCartanBlock) else QuaternionCartanBlock.from_unitary(a)
    right = b if isinstance(b, QuaternionCartanBlock) else QuaternionCartanBlock.from_unitary(b)
    return bool(np.allclose(left.cartan, right.cartan, atol=tolerance, rtol=0.0))
