"""rqm_entanglement – two-qubit / nonlocal layer of the RQM stack.

Public API
----------
Constants & tolerances:
    ATOL, RTOL, I2, I4, X, Y, Z, XX, YY, ZZ
    CNOT, CZ, SWAP, ISWAP

Basis states:
    ket00, ket01, ket10, ket11, basis_state, computational_basis

Tensor helpers:
    kron, local_unitary, apply_unitary

State helpers:
    normalize_state, state_from_amplitudes, density_matrix, reduced_density_matrix

Canonical entanglers:
    xx_rotation, yy_rotation, zz_rotation, canonical_entangler

Measures:
    concurrence_pure, is_separable_pure, schmidt_values_pure
    von_neumann_entropy, entanglement_entropy_pure

Classification:
    operator_schmidt_rank, is_local_product_operator, local_product_factors

Adapter:
    rqm_core_available, local_from_quaternions, su2_from_quaternion_components
"""

from rqm_entanglement.adapters.rqm_core_adapter import (
    canonicalize_quaternion_sign_with_phase,
    local_from_quaternions,
    normalize_local_su2_factor,
    quaternion_to_su2_matrix,
    rqm_core_available,
    su2_from_quaternion_components,
    su2_matrix_to_quaternion,
)
from rqm_entanglement.analysis import (
    EntanglementAnalysisOptions,
    EntanglementAnalysisResult,
    EntanglementMetric,
    analyze_entanglement,
)
from rqm_entanglement.basis import (
    basis_state,
    computational_basis,
    ket00,
    ket01,
    ket10,
    ket11,
)
from rqm_entanglement.canonical import (
    canonical_entangler,
    xx_rotation,
    yy_rotation,
    zz_rotation,
)
from rqm_entanglement.classify import (
    is_local_product_operator,
    local_product_factors,
    operator_schmidt_rank,
)
from rqm_entanglement.constants import (
    ATOL,
    CNOT,
    CZ,
    I2,
    I4,
    ISWAP,
    RTOL,
    SWAP,
    XX,
    YY,
    ZZ,
    X,
    Y,
    Z,
)
from rqm_entanglement.coupling import (
    CouplingAnalysisOptions,
    CouplingAnalysisResult,
    CouplingMetric,
    PreservationAnalysisResult,
    analyze_circuit_coupling,
    analyze_optimization_preservation,
)
from rqm_entanglement.measures import (
    concurrence_pure,
    entanglement_entropy_pure,
    is_separable_pure,
    schmidt_values_pure,
    von_neumann_entropy,
)
from rqm_entanglement.states import (
    density_matrix,
    normalize_state,
    reduced_density_matrix,
    state_from_amplitudes,
)
from rqm_entanglement.su4 import (
    CONVENTION_VERSION,
    QuaternionCartanBlock,
    SU4Classification,
    VerifiedSU4Decomposition,
    are_locally_equivalent,
    cartan_core_from_weyl,
    classify_su4,
    decompose_su4,
    decompose_su4_verified,
    in_weyl_chamber,
    nonlocal_fingerprint,
    phase_aligned_operator_error,
    reconstruct_su4,
    rotation_to_weyl_coordinates,
    weyl_to_rotation_coordinates,
)
from rqm_entanglement.tensor import apply_unitary, kron, local_unitary
from rqm_entanglement.validation import (
    assert_single_qubit_operator,
    assert_state_vector,
    assert_two_qubit_operator,
    is_su4,
    is_unitary,
    normalize_global_phase,
)

__all__ = [
    # tolerances
    "ATOL",
    "RTOL",
    # constants
    "I2",
    "I4",
    "X",
    "Y",
    "Z",
    "XX",
    "YY",
    "ZZ",
    "CNOT",
    "CZ",
    "SWAP",
    "ISWAP",
    # basis
    "ket00",
    "ket01",
    "ket10",
    "ket11",
    "basis_state",
    "computational_basis",
    # tensor
    "kron",
    "local_unitary",
    "apply_unitary",
    # states
    "normalize_state",
    "state_from_amplitudes",
    "density_matrix",
    "reduced_density_matrix",
    # canonical
    "xx_rotation",
    "yy_rotation",
    "zz_rotation",
    "canonical_entangler",
    "cartan_core_from_weyl",
    "weyl_to_rotation_coordinates",
    "rotation_to_weyl_coordinates",
    # arbitrary SU(4)
    "CONVENTION_VERSION",
    "QuaternionCartanBlock",
    "SU4Classification",
    "VerifiedSU4Decomposition",
    "decompose_su4",
    "decompose_su4_verified",
    "reconstruct_su4",
    "classify_su4",
    "are_locally_equivalent",
    "nonlocal_fingerprint",
    "in_weyl_chamber",
    "phase_aligned_operator_error",
    # measures
    "concurrence_pure",
    "is_separable_pure",
    "schmidt_values_pure",
    "von_neumann_entropy",
    "entanglement_entropy_pure",
    # classification
    "operator_schmidt_rank",
    "is_local_product_operator",
    "local_product_factors",
    # analysis API
    "EntanglementMetric",
    "EntanglementAnalysisResult",
    "EntanglementAnalysisOptions",
    "analyze_entanglement",
    "CouplingMetric",
    "CouplingAnalysisResult",
    "CouplingAnalysisOptions",
    "PreservationAnalysisResult",
    "analyze_circuit_coupling",
    "analyze_optimization_preservation",
    # validation
    "assert_state_vector",
    "assert_two_qubit_operator",
    "assert_single_qubit_operator",
    "is_unitary",
    "is_su4",
    "normalize_global_phase",
    # adapter
    "rqm_core_available",
    "local_from_quaternions",
    "su2_from_quaternion_components",
    "su2_matrix_to_quaternion",
    "quaternion_to_su2_matrix",
    "normalize_local_su2_factor",
    "canonicalize_quaternion_sign_with_phase",
]
