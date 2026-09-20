# rqm-compiler 0.4 Nonlocal-Math Ownership Boundary

This document freezes the ownership boundary between `rqm-entanglement` and
`rqm-compiler 0.4`.

## Canonical owner: rqm-entanglement

`rqm-entanglement` is the source of truth for:

- `BellHinge`;
- `AxisHinge`;
- `CartanRelation`;
- `QuaternionCartanBlock`;
- exact XX/YY/ZZ pair rotations;
- Cartan-core construction;
- promotion/demotion between relational levels;
- composition of relational operators;
- arbitrary SU(4) decomposition/reconstruction;
- Weyl-chamber coordinates, classes, fingerprints and local equivalence;
- phase-aligned reconstruction error for two-qubit unitaries.

These are mathematical semantics, not compiler policy.

## Compiler-owned responsibilities

`rqm-compiler` may:

- recognize when circuit operations prove membership in a relational level;
- decide when to keep, promote or materialize a representation;
- choose query/readout strategies;
- account for `C_R`, `C_Q`, topology and contraction metrics;
- perform exact observable propagation using the public relational semantics;
- build local execution/contraction plans;
- fall back conservatively when a specialized route is not proven.

It must not define a second canonical Cartan decomposition, Weyl convention,
`AxisHinge` semantics, or `QuaternionCartanBlock` reconstruction.

## 0.4 audit

The 0.4 compiler audit confirmed that its SU(4) extraction path imports
`QuaternionCartanBlock`, classification and reconstruction checks from this
package.

As part of the audit, the compiler's direct formulas for `RXX`, `RYY`, and
`RZZ` matrix construction were removed from `su4_blocks.py`; it now
delegates those canonical pair rotations to `rqm-entanglement`.

Compiler-side sparse Pauli conjugation and topology-aware contraction remain
compiler/query algorithms. They are not alternative definitions of Cartan or
two-qubit unitary semantics.

## Dependency direction

```text
rqm-core
   |
   v
rqm-entanglement
   |
   v
rqm-compiler
```

`rqm-entanglement` must not depend on `rqm-compiler`. This keeps the
mathematics independently testable and prevents compiler policy from becoming
part of the canonical nonlocal model.

## 0.4 acceptance criteria

Before `rqm-compiler 0.4.0` release:

1. `rqm-entanglement` tests pass independently;
2. compiler relational tests pass against the package public API;
3. no compiler source defines a competing `AxisHinge`, `CartanRelation` or
   `QuaternionCartanBlock` type;
4. canonical pair rotations and SU(4)/Weyl mathematics are delegated here;
5. changes to nonlocal conventions require an `rqm-entanglement` release and
   downstream compatibility qualification.
