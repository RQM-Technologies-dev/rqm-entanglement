# Changelog

## [0.2.2](https://github.com/RQM-Technologies-dev/rqm-entanglement/compare/v0.2.1...v0.2.2) (2026-09-20)


### Features

* add adaptive relational IR ([401b5f4](https://github.com/RQM-Technologies-dev/rqm-entanglement/commit/401b5f44180ddb1a0232c4d247097236c7315da3))


### Bug Fixes

* qualify entanglement 0.2.2 dependency package ([#16](https://github.com/RQM-Technologies-dev/rqm-entanglement/issues/16)) ([3e09e26](https://github.com/RQM-Technologies-dev/rqm-entanglement/commit/3e09e26125362210771e7596262beb5317ebae9e))


### Documentation

* clarify compiler 0.4 relational ownership ([4b14294](https://github.com/RQM-Technologies-dev/rqm-entanglement/commit/4b14294b75da9031b8428dbbf9033db562a5fef3))
* freeze compiler 0.4 nonlocal math boundary ([00ab4ca](https://github.com/RQM-Technologies-dev/rqm-entanglement/commit/00ab4ca5298698ea45e162d399c3d2bf0189a960))
* name the canonical entanglement API consumer ([#12](https://github.com/RQM-Technologies-dev/rqm-entanglement/issues/12)) ([de368d0](https://github.com/RQM-Technologies-dev/rqm-entanglement/commit/de368d0d30741f8c1f0ed9d483246b6f24e81afa))

## 0.2.2 — release candidate

- Package the analytic real Pauli-coefficient transfer module required by
  rqm-compiler 0.3.7.
- Include canonical Cartan promotion without redundant reconstruction while
  retaining decomposition for noncanonical coordinates.
- Apply formatting and type annotations required by existing release checks;
  no additional runtime features are introduced during qualification.

## [0.2.1](https://github.com/RQM-Technologies-dev/rqm-entanglement/compare/v0.2.0...v0.2.1) (2026-07-29)


### Bug Fixes

* dispatch protected publication by repository ([#11](https://github.com/RQM-Technologies-dev/rqm-entanglement/issues/11)) ([871fab6](https://github.com/RQM-Technologies-dev/rqm-entanglement/commit/871fab601d3d52be5c332a2f3656aa46f66ee04a))
* dispatch release pull request CI reliably ([#8](https://github.com/RQM-Technologies-dev/rqm-entanglement/issues/8)) ([5d4852f](https://github.com/RQM-Technologies-dev/rqm-entanglement/commit/5d4852f33fcc7e0ae416926e0668a74bf618c406))
* keep generated releases verifiable ([#10](https://github.com/RQM-Technologies-dev/rqm-entanglement/issues/10)) ([a432f64](https://github.com/RQM-Technologies-dev/rqm-entanglement/commit/a432f64d9800cb8cf6f51af1bd1ed38b5113f97a))

## 0.2.0 — 2026-07-18

- Add the immutable `QuaternionCartanBlock` production representation.
- Add optional-Qiskit arbitrary SU(4) decomposition and exact reconstruction.
- Add Weyl classification, local-equivalence fingerprints, and canonical hashes.
- Extend `analyze_entanglement` additively for finite 4x4 unitary inputs.
- Freeze EXP-012 provenance, tensor ordering, sign, phase, and claims boundaries.
