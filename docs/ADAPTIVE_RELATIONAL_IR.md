# Adaptive Relational IR

`rqm-entanglement` uses an adaptive compiler representation for two-qubit structure:

> **Use the smallest representation justified by proven structure; promote only when an operation leaves that representation closed.**

This is a compiler representation policy, not a claim that standard tensor-product quantum mechanics has been replaced. Every relational form has an exact conventional materialization path and can be checked against the existing state-vector / SU(4) implementation.

## Representation ladder

| Level | Representation | Meaning | Closed under |
|---|---|---|---|
| Bell sector | `BellHinge(parity, phase)` | Maximally entangled Bell state identified by `ZZ` and `XX` eigenvalue signs | local single-qubit Pauli sector changes |
| One-axis relation | `AxisHinge(axis, theta)` | `exp[-i theta/2 P⊗P]`, P in X/Y/Z | repeated interactions on the same axis |
| Cartan relation | `CartanRelation(c1,c2,c3)` | commuting `XX`, `YY`, `ZZ` nonlocal core | aligned Cartan-core composition |
| Quaternion-Cartan | `QuaternionCartanBlock` | local quaternion frames around a canonical nonlocal core | arbitrary two-qubit unitary representation after recanonicalization |
| Matrix escape hatch | existing `(4,4)` complex unitary | conventional backend / verification representation | all supported two-qubit unitary operations |

The ladder is intentionally asymmetric: Bell compression is a **state** representation, while the remaining levels are **operator** representations.

## Bell hinge

The Bell basis is represented by two relational eigenvalues:

```text
Phi+ = BellHinge(+1, +1)
Phi- = BellHinge(+1, -1)
Psi+ = BellHinge(-1, +1)
Psi- = BellHinge(-1, -1)
```

`parity` is the `ZZ` eigenvalue and `phase` is the `XX` eigenvalue. All four are the same maximally engaged Bell hinge (`theta = pi/2`) in different symmetry sectors. `to_state()` materializes the standard complex Bell vector.

## Promotion rules

1. Repeated same-axis hinges add angles and remain `AxisHinge`.
2. Different canonical axes promote to `CartanRelation`, where the commuting coordinates add exactly.
3. Independent local quaternion frames promote the relation to `QuaternionCartanBlock`.
4. Composition that mixes local frames and nonlocal cores may require recanonicalization through the exact SU(4) path.
5. After canonicalization, `compress_unitary()` attempts to demote back to a lower representation when the structure proves it is safe.

The intended compiler behavior is therefore:

```text
recognize -> compress -> propagate geometrically -> promote only if necessary
          -> recanonicalize when necessary -> compress again
```

## Scientific boundary

The relational IR is lossless only for the structure it claims to encode. It must never infer Bell, axis-hinge, or Cartan structure without proving the corresponding invariant. `QuaternionCartanBlock` and the conventional matrix representation remain the general two-qubit authorities.

The feature should be evaluated by exact reconstruction error and compiler cost, not by asserting new physics. Any future native quaternionic composite mechanics must be separately derived and experimentally/theoretically validated.
