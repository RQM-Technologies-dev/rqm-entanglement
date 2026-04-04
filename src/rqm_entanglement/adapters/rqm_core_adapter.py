"""Optional integration adapter for ``rqm-core``.

This module is the *only* place that imports from ``rqm-core``.  The rest of
this package works without it.

If ``rqm-core`` is not installed, :func:`rqm_core_available` returns
``False`` and :func:`local_from_quaternions` raises ``ImportError`` with a
clear message.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def rqm_core_available() -> bool:
    """Return ``True`` if ``rqm-core`` (``rqm_core``) can be imported."""
    try:
        import importlib

        importlib.import_module("rqm_core")
        return True
    except ImportError:
        return False


def _get_su2_from_quaternion() -> Any:
    """Locate the canonical quaternion → SU(2) function in rqm_core.

    Inspects the installed package to find the right callable rather than
    hard-coding a private import path.
    """
    try:
        import rqm_core  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "rqm-core is not installed or not importable.  "
            "Install it (e.g. `pip install rqm-core`) or add it to sys.path "
            "before calling adapter functions."
        ) from exc

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
    import importlib
    import pkgutil

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
        "Could not locate a quaternion→SU(2) function in rqm_core.  "
        "Expected one of: " + ", ".join(candidates)
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
    return np.asarray(np.kron(U1, U2), dtype=np.complex128)
