# EXP-012 production promotion provenance

- Source repository: `RQM-Technologies-dev/rqm-experiments`
- Experiment: EXP-012 — Quaternion–Cartan SU(4)
- Source commit: `615de9f5143ed603ca732b33a287e3e4f654c4c9`
- Claim: CLM-015
- Verdict: `promising_but_unresolved`
- Promotion date: 2026-07-18
- Destination version: `rqm-entanglement 0.2.0`

## Validated source evidence

EXP-012 recorded maximum phase-aligned operator error
`2.7698284490535387e-14`, maximum fixed-input probability error
`1.6209256159527285e-14`, maximum Cartan-coordinate error
`5.551115123125783e-16`, zero invalid Weyl-chamber records, zero binary/JSON
round-trip failures, zero canonical-hash failures, and four of four
local-equivalence fingerprint checks passing.

The verified factor order is
`qiskit-Klr:kron(q1=Kl,q0=Kr);circuit-little-endian`.  The computational basis
is `|00>, |01>, |10>, |11>` and qubit 0 is the left/more-significant RQM
circuit index.

## Promoted capability and boundary

This package now owns the tested immutable quaternion–Cartan representation,
exact reconstruction, optional Qiskit Weyl decomposition, deterministic
serialization/hashing, Weyl classification, and local-equivalence
fingerprints.  It continues to use `rqm-core` as the authority for quaternion
algebra and local SU(2) conversion.  It has no runtime dependency on
`rqm-experiments`.

The stored Cartan coordinates use `exp[i(a XX + b YY + c ZZ)]`.  Existing RQM
rotation helpers use `exp[-i/2(c1 XX + c2 YY + c3 ZZ)]`; public exact conversion
helpers connect those conventions.

This is standard complex quantum mechanics and introduces no new physics,
unique quantum information, or native quaternionic composite mechanics.  The
promotion does not claim general synthesis, runtime, hardware, or performance
superiority.  In particular, EXP-012 did not show the direct RQ synthesis path
beating the strongest matrix-KAK/Qiskit implementation.  Raw IBM counts,
credentials, hardware manifests, and experiment workflow code were not
promoted.
