"""Optional integration adapter for ``rqm-core``.

This module is the *only* place that imports from ``rqm-core``.  The rest of
this package works without it.

If ``rqm-core`` is not installed, :func:`rqm_core_available` returns
``False`` and adapter helpers raise ``ImportError`` with a clear message.
"""

from __future__ import annotations

import importlib
import math
import pkgutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

QuaternionTuple = tuple[float, float, float, float]
_FLOAT_TOL = 1e-12
_SIGN_TOL = 1e-14


def rqm_core_available() -> bool:
    """Return ``True`` if ``rqm-core`` (``rqm_core``) can be imported."""
    try:
        _import_rqm_core()
        return True
    except ImportError:
        return False


def _candidate_workspace_roots() -> list[Path]:
    """Return deterministic candidate roots that may contain an ``rqm-core`` repo."""
    here = Path(__file__).resolve()
    candidates: list[Path] = []
    for parent in here.parents:
        candidates.extend((parent, parent.parent))
    return list(dict.fromkeys(candidates))


def _maybe_add_rqm_core_src_to_syspath() -> bool:
    """Add ``rqm-core/src`` from common workspace layouts, if present."""
    repo_names = ("rqm-core", "rqm_core")
    for root in _candidate_workspace_roots():
        for repo_name in repo_names:
            src_dir = root / repo_name / "src"
            init_file = src_dir / "rqm_core" / "__init__.py"
            if not init_file.is_file():
                continue
            src_text = str(src_dir)
            if src_text not in sys.path:
                sys.path.insert(0, src_text)
            return True
    return False


def _import_rqm_core() -> Any:
    """Import ``rqm_core``, including workspace-local sibling discovery fallback."""
    try:
        return importlib.import_module("rqm_core")
    except ImportError as first_error:
        added = _maybe_add_rqm_core_src_to_syspath()
        if added:
            try:
                return importlib.import_module("rqm_core")
            except ImportError as retry_error:
                raise ImportError(
                    "rqm_core import failed even after adding a workspace-local "
                    "rqm-core/src path. Install `rqm-core` or ensure the sibling "
                    "checkout contains src/rqm_core/__init__.py."
                ) from retry_error
        raise ImportError(
            "rqm_core is not installed and no workspace-local sibling checkout was "
            "found (tried common layouts like ../rqm-core and ../../rqm-core). "
            "Install `rqm-core` or place a sibling checkout with src/rqm_core."
        ) from first_error


def _get_su2_from_quaternion() -> Any:
    """Locate the canonical quaternion → SU(2) function in rqm_core.

    Inspects the installed package to find the right callable rather than
    hard-coding a private import path.
    """
    rqm_core = _import_rqm_core()

    # Probe for the public API – common names used in rqm-core
    candidates = [
        "quaternion_to_su2",
        "quat_to_su2",
        "su2_from_quaternion",
        "to_unitary",
    ]
    for name in candidates:
        fn = getattr(rqm_core, name, None)
        if callable(fn):
            return fn

    # Fall back: search sub-modules
    pkg_path = getattr(rqm_core, "__path__", None)
    if pkg_path is not None:
        for module_info in pkgutil.walk_packages(pkg_path, prefix="rqm_core."):
            try:
                mod = importlib.import_module(module_info.name)
            except Exception:  # noqa: BLE001
                continue
            for name in candidates:
                fn = getattr(mod, name, None)
                if callable(fn):
                    return fn

    raise ImportError(
        "Could not locate a quaternion→SU(2) function in rqm_core. "
        "Looked for public names first (quaternion_to_su2, then fallbacks) "
        "and submodule exports. Expected one of: " + ", ".join(candidates)
    )


def su2_from_quaternion_components(
    w: float,
    x: float,
    y: float,
    z: float,
) -> NDArray[np.complex128]:
    """Return rqm-core's canonical SU(2) matrix for quaternion components."""
    rqm_core = _import_rqm_core()
    quaternion_cls = getattr(rqm_core, "Quaternion", None)
    q = quaternion_cls(w, x, y, z) if callable(quaternion_cls) else (w, x, y, z)
    su2_from_quat = _get_su2_from_quaternion()
    matrix: NDArray[np.complex128] = np.asarray(su2_from_quat(q), dtype=np.complex128)
    if matrix.shape != (2, 2):
        raise ValueError(
            "rqm_core quaternion conversion returned malformed matrix: "
            f"expected shape (2, 2), got {matrix.shape}"
        )
    return matrix


def _quaternion_tuple(value: Any) -> QuaternionTuple:
    """Return a normalized finite quaternion tuple using ``rqm-core``."""
    if all(hasattr(value, name) for name in ("w", "x", "y", "z")):
        values = tuple(float(getattr(value, name)) for name in ("w", "x", "y", "z"))
    else:
        values = tuple(float(component) for component in value)
    if len(values) != 4 or not np.all(np.isfinite(values)):
        raise ValueError("a quaternion requires exactly four finite values")
    rqm_core = _import_rqm_core()
    quaternion_cls = getattr(rqm_core, "Quaternion", None)
    if not callable(quaternion_cls):
        raise ImportError("rqm_core does not expose its canonical Quaternion type")
    quaternion = quaternion_cls(*values)
    if quaternion.norm() <= _FLOAT_TOL:
        raise ValueError("zero quaternion is invalid")
    normalized = quaternion.normalize()
    result = (normalized.w, normalized.x, normalized.y, normalized.z)
    return tuple(0.0 if abs(component) <= 1e-16 else float(component) for component in result)  # type: ignore[return-value]


def quaternion_to_su2_matrix(quaternion: Any) -> NDArray[np.complex128]:
    """Convert four quaternion components through ``rqm-core``'s SU(2) authority."""
    return su2_from_quaternion_components(*_quaternion_tuple(quaternion))


def su2_matrix_to_quaternion(matrix: NDArray[np.complex128]) -> QuaternionTuple:
    """Convert an SU(2) matrix through ``rqm-core`` and return finite components."""
    value = np.asarray(matrix, dtype=np.complex128)
    if value.shape != (2, 2) or not np.all(np.isfinite(value)):
        raise ValueError("SU(2) matrix must be finite and have shape (2,2)")
    rqm_core = _import_rqm_core()
    converter = getattr(rqm_core, "su2_to_quaternion", None)
    if not callable(converter):
        raise ImportError("rqm_core does not expose su2_to_quaternion")
    return _quaternion_tuple(converter(value))


def canonicalize_quaternion_sign_with_phase(
    quaternion: Any,
    global_phase: float,
    *,
    atol: float = _SIGN_TOL,
) -> tuple[QuaternionTuple, float, bool]:
    """Freeze the quaternion sign and compensate the enclosing U(4) phase.

    The first component whose magnitude exceeds ``atol`` is made positive.  This
    implements ``w > 0`` with the first nonzero ``x/y/z`` component as the
    boundary tie-breaker.  A sign flip changes an SU(2) factor by ``-1``, so the
    enclosing global phase is reduced by pi to preserve exact U(4) semantics.
    """
    values = np.asarray(_quaternion_tuple(quaternion), dtype=np.float64)
    pivot = next((float(item) for item in values if abs(float(item)) > atol), 0.0)
    flipped = pivot < 0.0
    phase = float(global_phase)
    if not math.isfinite(phase):
        raise ValueError("global phase must be finite")
    if flipped:
        values = -values
        phase -= math.pi
    values[np.abs(values) <= 1e-16] = 0.0
    return tuple(float(item) for item in values), phase, flipped  # type: ignore[return-value]


def normalize_local_su2_factor(
    matrix: NDArray[np.complex128],
    *,
    atol: float = _FLOAT_TOL,
) -> tuple[QuaternionTuple, float, bool, float]:
    """Remove a U(2) determinant phase and encode the SU(2) factor as a quaternion.

    Returns ``(quaternion, removed_phase, sign_flipped, reconstruction_error)``.
    The returned ``removed_phase`` already includes sign-canonicalization
    compensation and can be added directly to an enclosing U(4) phase.
    """
    value = np.asarray(matrix, dtype=np.complex128)
    if value.shape != (2, 2) or not np.all(np.isfinite(value)):
        raise ValueError("local factor must be a finite 2x2 matrix")
    unitarity_error = float(
        np.max(np.abs(value.conj().T @ value - np.eye(2, dtype=np.complex128)))
    )
    if unitarity_error > atol:
        raise ValueError(f"local factor is not unitary: {unitarity_error}")
    determinant = np.linalg.det(value)
    if abs(abs(determinant) - 1.0) > atol:
        raise ValueError("local determinant does not have unit modulus")
    removed_phase = 0.5 * float(np.angle(determinant))
    special = value * np.exp(-1j * removed_phase)
    if abs(np.linalg.det(special) - 1.0) > 1e-10:
        special = -special
        removed_phase += math.pi
    quaternion = su2_matrix_to_quaternion(special)
    quaternion, removed_phase, flipped = canonicalize_quaternion_sign_with_phase(
        quaternion, removed_phase
    )
    reconstructed = quaternion_to_su2_matrix(quaternion)
    target = -special if flipped else special
    error = float(np.max(np.abs(reconstructed - target)))
    if error > 1e-10:
        raise ValueError(f"quaternion/SU(2) conversion mismatch: {error}")
    return quaternion, removed_phase, flipped, error


def local_from_quaternions(q1: Any, q2: Any) -> NDArray[np.complex128]:
    """Return U1 ⊗ U2 where U1, U2 are SU(2) matrices from quaternions q1, q2.

    Delegates quaternion → SU(2) conversion entirely to ``rqm-core``.
    Raises ``ImportError`` if ``rqm-core`` is not available.

    Parameters
    ----------
    q1, q2:
        Quaternion objects (or array-like) accepted by rqm_core's canonical
        quaternion-to-SU(2) function.

    Returns
    -------
    NDArray of shape (4, 4) and dtype complex128.
    """
    su2_from_quat = _get_su2_from_quaternion()
    U1: NDArray[np.complex128] = np.asarray(su2_from_quat(q1), dtype=np.complex128)
    U2: NDArray[np.complex128] = np.asarray(su2_from_quat(q2), dtype=np.complex128)
    if U1.shape != (2, 2):
        raise ValueError(
            "rqm_core quaternion conversion returned malformed matrix for q1: "
            f"expected shape (2, 2), got {U1.shape}"
        )
    if U2.shape != (2, 2):
        raise ValueError(
            "rqm_core quaternion conversion returned malformed matrix for q2: "
            f"expected shape (2, 2), got {U2.shape}"
        )
    return np.asarray(np.kron(U1, U2), dtype=np.complex128)
