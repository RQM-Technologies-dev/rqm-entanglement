# rqm-entanglement

`rqm-entanglement` is the two-qubit / nonlocal layer of the RQM quantum stack.
It provides tensor-product state and operator helpers, the canonical commuting
nonlocal gate family, pure-state entanglement measures (concurrence, Schmidt
decomposition, von Neumann entropy), and operator Schmidt-rank classification —
all with a minimal numpy-only runtime.

---

## Architecture boundary with `rqm-core`

| Layer | Owns |
|---|---|
| `rqm-core` | quaternion math, SU(2), single-qubit geometry, quat→2×2 unitary mapping |
| `rqm-entanglement` | two-qubit tensor structure, canonical nonlocal generators, entanglement measures, operator classification |

All `rqm-core` integration lives in `src/rqm_entanglement/adapters/rqm_core_adapter.py`
and is fully optional.  If `rqm-core` is not installed the adapter raises a
clear `ImportError` only when its functions are called.

---

## Basis ordering

Computational basis: **|00⟩, |01⟩, |10⟩, |11⟩**  
Qubit 0 is the more-significant (left) index.  
CNOT: control = qubit 0, target = qubit 1.

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

## v0.1 scope – mathematical honesty

This release does **not** implement a full KAK / Cartan decomposition or a
generic arbitrary-SU(4) classifier.  It implements:

- operator Schmidt rank and local-product detection
- pure-state concurrence and separability
- entanglement entropy via partial trace and von Neumann entropy

**SWAP note**: SWAP has operator Schmidt rank > 1 (it is a nonlocal operator),
but it maps every product state to another product state.  This package does
*not* label SWAP as a generic "entangling gate"; doing so correctly requires
a full Cartan analysis that is deferred to a later release.

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

## Installation

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

