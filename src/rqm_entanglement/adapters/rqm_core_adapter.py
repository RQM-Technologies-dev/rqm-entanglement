"""Optional integration adapter for ``rqm-core``.

This module is the *only* place that imports from ``rqm-core``.  The rest of
this package works without it.

If ``rqm-core`` is not installed, :func:`rqm_core_available` returns
``False`` and :func:`local_from_quaternions` raises ``ImportError`` with a
clear message.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


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
