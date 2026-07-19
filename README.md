# rqm-entanglement

`rqm-entanglement` is the two-qubit / nonlocal layer of the RQM quantum stack.
It provides tensor-product state and operator helpers, arbitrary two-qubit
quaternion–Cartan decomposition, Weyl classification and local-equivalence
fingerprints, the canonical commuting nonlocal gate family, pure-state
entanglement measures, and operator Schmidt-rank classification. `rqm-core`
remains the authority for canonical quaternion/SU(2) conversion.

It uses ordinary complex tensor products and standard entanglement measures.
Local quaternion adapters provide equivalent `SU(2)` coordinates; they do not
define native quaternionic composite mechanics. See
[RQM_TECHNICAL_CANON_V2.md](RQM_TECHNICAL_CANON_V2.md).

---

## Architecture boundary with `rqm-core`

| Layer | Owns |
|---|---|
| `rqm-core` | quaternion math, SU(2), single-qubit geometry, quat→2×2 unitary mapping |
| `rqm-entanglement` | two-qubit tensor structure, arbitrary SU(4) quaternion–Cartan decomposition, Weyl classification, nonlocal fingerprints, entanglement analysis |

All `rqm-core` integration lives in `src/rqm_entanglement/adapters/rqm_core_adapter.py`.
That adapter is the only place Entanglement imports Core, and the measured
circuit analyzers use it for canonical `u1q` quaternion gates emitted by
`rqm-compiler`.

---

## Basis ordering

Computational basis: **|00⟩, |01⟩, |10⟩, |11⟩**  
Qubit 0 is the more-significant (left) index.  
CNOT: control = qubit 0, target = qubit 1.

The verified optional-Qiskit factor order is
`qiskit-Klr:kron(q1=Kl,q0=Kr);circuit-little-endian`.

---

## Canonical entangler

```
U_ent(c1, c2, c3) = exp[-i/2 (c1 XX + c2 YY + c3 ZZ)]
```

Because XX, YY, ZZ mutually commute, this is computed analytically as

```
U_ent = xx_rotation(c1) @ yy_rotation(c2) @ zz_rotation(c3)
```

where `exp[-i θ/2 P] = cos(θ/2) I4 - i sin(θ/2) P` for P ∈ {XX, YY, ZZ}.

---

## Quaternion–Cartan SU(4)

Version 0.2 promotes the tested EXP-012 representation:

```python
from rqm_entanglement import QuaternionCartanBlock, classify_su4

block = QuaternionCartanBlock.from_unitary(unitary)  # requires [qiskit]
assert block.validate()["valid"]
reconstructed = block.to_unitary()
classification = classify_su4(block)
```

Stored Weyl coordinates use `exp[i(a XX + b YY + c ZZ)]`. Existing
`canonical_entangler(c1,c2,c3)` uses
`exp[-i/2(c1 XX + c2 YY + c3 ZZ)]`; use `weyl_to_rotation_coordinates` and
`rotation_to_weyl_coordinates` for the exact conversion.

Reconstruction, serialization, hashes, classification of an existing block,
and canonical-entangler construction do not import Qiskit. Generic arbitrary
SU(4) decomposition uses the optional public Qiskit Weyl authority:

```bash
pip install -e ".[qiskit]"
```

The classifier distinguishes nonlocal operator, entangling-gate,
perfect-entangler, and SWAP-like status. SWAP remains truthfully nonlocal while
not being labeled as a gate that entangles product inputs.

---

## Quick start

```python
import numpy as np
from rqm_entanglement import (
    I2, Y, CNOT,
    ket00, local_unitary, apply_unitary,
    concurrence_pure, entanglement_entropy_pure
)

def ry(theta: float) -> np.ndarray:
    return np.cos(theta / 2) * I2 - 1j * np.sin(theta / 2) * Y

psi0 = ket00()
U = CNOT @ local_unitary(ry(np.pi / 2), I2)
psi = apply_unitary(U, psi0)

print(concurrence_pure(psi))           # ~1.0
print(entanglement_entropy_pure(psi))  # ~1.0
```

---

## Stable entanglement analysis API (for rqm-api / Studio)

For integration points like `rqm-api` endpoint `/v1/entanglement/analyze`,
use the stable exported function:

```python
from rqm_entanglement import analyze_entanglement
```

```python
from rqm_entanglement import CNOT, I2, Y, ket00, local_unitary, apply_unitary, analyze_entanglement
import numpy as np

def ry(theta: float) -> np.ndarray:
    return np.cos(theta / 2) * I2 - 1j * np.sin(theta / 2) * Y

psi0 = ket00()
psi_bell = apply_unitary(CNOT @ local_unitary(ry(np.pi / 2), I2), psi0)

result = analyze_entanglement(psi_bell)
print(result)
```

Result schema:

```python
{
  "has_entangling_gates": bool,
  "entangled_pairs": [
    {
      "pair": [0, 1],
      "metric_name": str,
      "metric_value": float,
      "interpretation": str,  # optional
    }
  ],
  "last_entangling_gate": str,  # optional
  "fidelity_preserved": float | None,
  "notes": [str],
}
```

Example (Bell-state-like input):

```python
{
  "has_entangling_gates": True,
  "entangled_pairs": [
    {"pair": [0, 1], "metric_name": "Concurrence", "metric_value": 1.0},
    {"pair": [0, 1], "metric_name": "Entropy", "metric_value": 1.0},
    {"pair": [0, 1], "metric_name": "Mutual Information", "metric_value": 2.0},
    {"pair": [0, 1], "metric_name": "RQM Correlation", "metric_value": 1.0},
  ],
  "fidelity_preserved": None,
  "notes": [],
}
```

Input forms accepted by `analyze_entanglement`:

- pure state vector `(4,)`
- unitary / SU(4) matrix `(4,4)`
- gate sequence (list of `(4,4)` matrices), including:
  - unnamed gates: `[U0, U1, ...]`
  - named gates: `[("gate_name", U), ...]`
  - dict-like entries: `[{"name": "...", "unitary": U}, ...]`
- RQM circuit payload (Studio/API contract style), e.g.:
  ```json
  {
    "schema_version": "0.1",
    "num_qubits": 2,
    "instructions": [
      {
        "gate": { "name": "h", "arity": 1 },
        "targets": [{ "index": 0, "type": "qubit" }]
      },
      {
        "gate": { "name": "cx", "arity": 2 },
        "targets": [
          { "index": 0, "type": "qubit" },
          { "index": 1, "type": "qubit" }
        ]
      }
    ]
  }
  ```
  Supported contract gate names currently include:
  - single-qubit: `i`, `x`, `y`, `z`, `h`, `s`, `sdg`, `t`, `tdg`, `rx`, `ry`, `rz`, `u1q`
  - two-qubit: `cx`/`cnot`, `cz`, `swap`, `iswap`
  Unsupported instructions are skipped with explanatory `notes` while preserving
  the stable result schema.

For a finite unitary `(4,4)` input, the result additively includes `su4` with
Cartan coordinates, Weyl class, both hashes, operator Schmidt rank, local
quaternion shells, global phase, reconstruction error, and convention version.
When the optional Qiskit dependency is absent, all legacy fields remain
available and `notes` explains why decomposition was omitted.

---

## Metric formulas and conventions

Basis ordering is always `|00>, |01>, |10>, |11>` with qubit 0 as the
more-significant (left) index.

For a pure state
`|psi> = [a00, a01, a10, a11]^T`:

- **Concurrence**
  - `C = 2 * |a00*a11 - a01*a10|`
  - Range: `[0, 1]`
  - `0` for product states, `1` for maximally entangled states
- **Entanglement Entropy**
  - `S(rho_A) = -Tr(rho_A log2 rho_A)` (bits), where `rho_A` is reduced density
    matrix of either qubit for pure two-qubit states
  - Range: `[0, 1]`
- **Mutual Information**
  - `I(A:B) = S(rho_A) + S(rho_B) - S(rho_AB)` (bits)
  - For pure two-qubit states, `S(rho_AB)=0`, so `I(A:B)=2*S(rho_A)` and range
    is `[0, 2]`
- **RQM Correlation** (optional software summary metric, not new physics)
  - `0.5 * (clamped_concurrence + clamped_entropy)`
  - Range: `[0, 1]`

Numerical conventions:

- tiny negative eigenvalues from floating-point roundoff are clipped to `0`
- output metrics are clamped to physically valid ranges
- non-finite values (`NaN`/`Inf`) are sanitized to stable values with notes
- metric ordering is deterministic for identical input

---

## Scope and current limits

- Primary support: **2-qubit SU(4)/state analysis**
- Inputs that appear to represent `>2` qubits are handled gracefully:
  - no crash
  - stable output schema
  - explanatory `notes`
- Full generic multi-qubit entanglement analysis is out of scope.

The tested quaternion–Cartan representation is standard complex quantum
mechanics. It is not unique quantum information or native quaternionic
composite mechanics, and it carries no general synthesis, runtime, or IBM
hardware superiority claim. See
[`docs/EXP012_PROMOTION_PROVENANCE.md`](docs/EXP012_PROMOTION_PROVENANCE.md).

---

## Installation

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```
