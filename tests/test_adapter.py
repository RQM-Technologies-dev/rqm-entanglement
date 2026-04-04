"""Tests for rqm_entanglement.adapters.rqm_core_adapter."""

import pytest

from rqm_entanglement.adapters.rqm_core_adapter import (
    local_from_quaternions,
    rqm_core_available,
)


def test_rqm_core_available_returns_bool():
    result = rqm_core_available()
    assert isinstance(result, bool)


def test_local_from_quaternions_raises_when_unavailable():
    """If rqm-core is not installed, local_from_quaternions must raise ImportError."""
    if rqm_core_available():
        pytest.skip("rqm-core is available; skipping unavailability test")
    with pytest.raises(ImportError, match="rqm-core"):
        local_from_quaternions(None, None)
