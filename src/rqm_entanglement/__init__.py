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
    rqm_core_available, local_from_quaternions
"""

from rqm_entanglement.adapters.rqm_core_adapter import (
    local_from_quaternions,
    rqm_core_available,
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
from rqm_entanglement.coupling import (
    CouplingAnalysisOptions,
    CouplingAnalysisResult,
    CouplingMetric,
    PreservationAnalysisResult,
    analyze_circuit_coupling,
    analyze_optimization_preservation,
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
]
