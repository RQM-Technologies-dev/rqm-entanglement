from __future__ import annotations

import builtins
import json
import math

import numpy as np
import pytest

from rqm_entanglement import (
    CNOT,
    CZ,
    ISWAP,
    SWAP,
    QuaternionCartanBlock,
    are_locally_equivalent,
    canonical_entangler,
    cartan_core_from_weyl,
    classify_su4,
    decompose_su4,
    phase_aligned_operator_error,
    quaternion_to_su2_matrix,
    rotation_to_weyl_coordinates,
    weyl_to_rotation_coordinates,
)


def _random_su4(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    q, r = np.linalg.qr(raw)
    phases = np.diag(r)
    q = q @ np.diag(np.conj(phases) / np.abs(phases))
    return q / np.linalg.det(q) ** 0.25


def _unit_quaternion(values: tuple[float, float, float, float]) -> tuple[float, ...]:
    array = np.asarray(values, dtype=np.float64)
    return tuple(array / np.linalg.norm(array))


@pytest.mark.parametrize(
    ("unitary", "label"),
    [
        (np.eye(4), "local_identity"),
        (CNOT, "cnot_cz_class"),
        (CZ, "cnot_cz_class"),
        (ISWAP, "iswap_class"),
        (cartan_core_from_weyl(math.pi / 8, math.pi / 8, 0.0), "sqrt_iswap_class"),
        (SWAP, "swap_class"),
        (
            cartan_core_from_weyl(math.pi / 8, math.pi / 8, math.pi / 8),
            "sqrt_swap_class",
        ),
        (cartan_core_from_weyl(math.pi / 4, math.pi / 8, 0.0), "b_gate_class"),
        (cartan_core_from_weyl(0.63, 0.29, -0.17), "generic_interior"),
    ],
)
def test_weyl_landmarks(unitary: np.ndarray, label: str) -> None:
    block = decompose_su4(unitary)
    error, _ = phase_aligned_operator_error(unitary, block.to_unitary())
    assert error <= 1e-10
    assert classify_su4(block).class_label == label


def test_rotation_and_weyl_conventions_are_exact() -> None:
    coordinates = (0.31, 0.17, -0.09)
    rotations = weyl_to_rotation_coordinates(*coordinates)
    assert rotation_to_weyl_coordinates(*rotations) == coordinates
    assert np.allclose(
        cartan_core_from_weyl(*coordinates),
        canonical_entangler(*rotations),
        atol=1e-14,
    )


def test_asymmetric_qiskit_factor_ordering() -> None:
    factors = tuple(
        _unit_quaternion(values)
        for values in (
            (0.91, 0.11, -0.27, 0.29),
            (0.71, -0.41, 0.19, 0.53),
            (0.66, 0.43, 0.37, -0.49),
            (0.84, -0.23, 0.44, 0.21),
        )
    )
    source = (
        np.exp(0.317j)
        * np.kron(quaternion_to_su2_matrix(factors[1]), quaternion_to_su2_matrix(factors[0]))
        @ cartan_core_from_weyl(0.63, 0.29, -0.17)
        @ np.kron(quaternion_to_su2_matrix(factors[3]), quaternion_to_su2_matrix(factors[2]))
    )
    block = decompose_su4(source)
    correct, _ = phase_aligned_operator_error(source, block.to_unitary())
    swapped = (
        np.exp(1j * block.global_phase)
        * np.kron(
            quaternion_to_su2_matrix(block.left_q0),
            quaternion_to_su2_matrix(block.left_q1),
        )
        @ cartan_core_from_weyl(*block.cartan)
        @ np.kron(
            quaternion_to_su2_matrix(block.right_q0),
            quaternion_to_su2_matrix(block.right_q1),
        )
    )
    swapped_error, _ = phase_aligned_operator_error(source, swapped)
    assert correct <= 1e-12
    assert swapped_error >= 1e-3


def test_json_binary_hash_and_phase_round_trips() -> None:
    source = _random_su4(12012)
    block = decompose_su4(source)
    from_json = QuaternionCartanBlock.from_dict(json.loads(json.dumps(block.to_dict())))
    from_binary = QuaternionCartanBlock.from_binary(block.to_binary())
    phase_equivalent = decompose_su4(np.exp(2j * math.pi) * source)
    assert block.full_canonical_hash() == from_json.full_canonical_hash()
    assert block.full_canonical_hash() == from_binary.full_canonical_hash()
    assert block.full_canonical_hash() == phase_equivalent.full_canonical_hash()
    assert block.nonlocal_fingerprint() == from_binary.nonlocal_fingerprint()


def test_local_equivalence_fingerprint_and_full_hash_distinction() -> None:
    identity_q = (1.0, 0.0, 0.0, 0.0)
    left = _unit_quaternion((0.73, -0.31, 0.21, 0.55))
    right = _unit_quaternion((0.61, 0.44, -0.33, 0.57))
    base = QuaternionCartanBlock.from_components(
        left_q0=identity_q,
        left_q1=identity_q,
        cartan_a=0.51,
        cartan_b=0.24,
        cartan_c=-0.11,
        right_q0=identity_q,
        right_q1=identity_q,
    )
    orbit = QuaternionCartanBlock.from_components(
        left_q0=left,
        left_q1=right,
        cartan_a=0.51,
        cartan_b=0.24,
        cartan_c=-0.11,
        right_q0=identity_q,
        right_q1=identity_q,
    )
    assert are_locally_equivalent(base, orbit)
    assert base.nonlocal_fingerprint() == orbit.nonlocal_fingerprint()
    assert base.full_canonical_hash() != orbit.full_canonical_hash()


def test_quaternion_w_near_zero_sign_boundary_preserves_semantics() -> None:
    epsilon = 1e-16
    raw = (-epsilon, -1.0, 0.0, 0.0)
    identity_q = (1.0, 0.0, 0.0, 0.0)
    original = (
        np.kron(np.eye(2), quaternion_to_su2_matrix(raw))
        @ cartan_core_from_weyl(0.4, 0.2, 0.1)
    )
    block = QuaternionCartanBlock.from_components(
        left_q0=raw,
        left_q1=identity_q,
        cartan_a=0.4,
        cartan_b=0.2,
        cartan_c=0.1,
        right_q0=identity_q,
        right_q1=identity_q,
    )
    assert block.left_q0[1] > 0.0
    assert np.max(np.abs(original - block.to_unitary())) <= 1e-12


@pytest.mark.parametrize("phase", [math.pi - 1e-15, -math.pi + 1e-15, 3 * math.pi])
def test_global_phase_boundaries_round_trip(phase: float) -> None:
    identity_q = (1.0, 0.0, 0.0, 0.0)
    block = QuaternionCartanBlock.from_components(
        left_q0=identity_q,
        left_q1=identity_q,
        cartan_a=0.0,
        cartan_b=0.0,
        cartan_c=0.0,
        right_q0=identity_q,
        right_q1=identity_q,
        global_phase=phase,
    )
    assert -math.pi <= block.global_phase < math.pi
    assert QuaternionCartanBlock.from_binary(block.to_binary()).global_phase == block.global_phase


def test_deterministic_random_su4_acceptance_metrics() -> None:
    max_operator_error = 0.0
    max_probability_error = 0.0
    max_cartan_error = 0.0
    probe = np.array([0.31, -0.22j, 0.47 + 0.11j, -0.17], dtype=np.complex128)
    probe /= np.linalg.norm(probe)
    for seed in range(12):
        source = _random_su4(seed)
        block = decompose_su4(source)
        reconstructed = block.to_unitary()
        operator_error, phase = phase_aligned_operator_error(source, reconstructed)
        probability_error = float(
            np.max(np.abs(np.abs(source @ probe) ** 2 - np.abs(reconstructed @ probe) ** 2))
        )
        recovered = decompose_su4(reconstructed)
        cartan_error = float(np.max(np.abs(np.asarray(block.cartan) - recovered.cartan)))
        assert math.isfinite(phase)
        assert np.all(np.isfinite(reconstructed))
        max_operator_error = max(max_operator_error, operator_error)
        max_probability_error = max(max_probability_error, probability_error)
        max_cartan_error = max(max_cartan_error, cartan_error)
    assert max_operator_error <= 1e-10
    assert max_probability_error <= 1e-10
    assert max_cartan_error <= 1e-9


def test_invalid_inputs_and_optional_qiskit_absence(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="not unitary"):
        decompose_su4(np.ones((4, 4), dtype=np.complex128))
    with pytest.raises(ValueError, match="binary header"):
        QuaternionCartanBlock.from_binary(b"not-a-block")

    real_import = builtins.__import__

    def blocked_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "qiskit.synthesis" or name.startswith("qiskit.synthesis."):
            raise ImportError("blocked for optional-dependency test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(ImportError, match="optional 'qiskit'"):
        decompose_su4(np.eye(4, dtype=np.complex128))


def test_existing_block_features_work_without_qiskit_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def blocked_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "qiskit" or name.startswith("qiskit."):
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    block = QuaternionCartanBlock.from_components(
        left_q0=(1.0, 0.0, 0.0, 0.0),
        left_q1=(1.0, 0.0, 0.0, 0.0),
        cartan_a=0.4,
        cartan_b=0.2,
        cartan_c=0.1,
        right_q0=(1.0, 0.0, 0.0, 0.0),
        right_q1=(1.0, 0.0, 0.0, 0.0),
    )
    assert QuaternionCartanBlock.from_dict(block.to_dict()) == block
    assert classify_su4(block).weyl_chamber_valid
