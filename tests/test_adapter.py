"""Tests for rqm_entanglement.adapters.rqm_core_adapter."""

import numpy as np
import pytest

from rqm_entanglement.adapters.rqm_core_adapter import (
    _get_su2_from_quaternion,
    _import_rqm_core,
    local_from_quaternions,
    rqm_core_available,
)
from rqm_entanglement.validation import is_unitary


def test_rqm_core_available_returns_bool():
    result = rqm_core_available()
    assert isinstance(result, bool)


def test_local_from_quaternions_raises_when_unavailable():
    """If rqm-core is not installed, local_from_quaternions must raise ImportError."""
    if rqm_core_available():
        pytest.skip("rqm-core is available; skipping unavailability test")
    with pytest.raises(ImportError, match="rqm-core"):
        local_from_quaternions(None, None)


def test_local_from_quaternions_when_available():
    if not rqm_core_available():
        pytest.skip("rqm-core is unavailable after adapter discovery")

    rqm_core = _import_rqm_core()
    su2_from_quaternion = _get_su2_from_quaternion()

    q1 = None
    q2 = None

    quaternion_cls = getattr(rqm_core, "Quaternion", None)
    if quaternion_cls is not None:
        q1 = quaternion_cls(1.0, 0.0, 0.0, 0.0)
        q2 = quaternion_cls(0.0, 1.0, 0.0, 0.0)
    else:
        candidate_pairs = [
            ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)),
            ([1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]),
            (np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0, 0.0])),
        ]
        for maybe_q1, maybe_q2 in candidate_pairs:
            try:
                su2_from_quaternion(maybe_q1)
                su2_from_quaternion(maybe_q2)
                q1, q2 = maybe_q1, maybe_q2
                break
            except Exception:  # noqa: BLE001
                continue

    if q1 is None or q2 is None:
        pytest.skip("rqm-core is available, but no conservative quaternion input form was accepted")

    U = local_from_quaternions(q1, q2)
    assert U.shape == (4, 4)
    assert U.dtype == np.complex128
    assert is_unitary(U)
