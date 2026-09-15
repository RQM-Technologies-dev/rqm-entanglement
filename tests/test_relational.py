import math

import numpy as np

from rqm_entanglement import (
    AxisHinge,
    BellHinge,
    CartanRelation,
    RelationalLevel,
    compose_relations,
    concurrence_pure,
    entanglement_entropy_pure,
    relational_level,
)


def test_bell_hinge_materializes_all_four_bell_sectors():
    expected = {
        (1, 1): np.array([1, 0, 0, 1], dtype=complex) / math.sqrt(2),
        (1, -1): np.array([1, 0, 0, -1], dtype=complex) / math.sqrt(2),
        (-1, 1): np.array([0, 1, 1, 0], dtype=complex) / math.sqrt(2),
        (-1, -1): np.array([0, 1, -1, 0], dtype=complex) / math.sqrt(2),
    }
    for sector, state in expected.items():
        hinge = BellHinge(*sector)
        assert np.allclose(hinge.to_state(), state)
        assert math.isclose(concurrence_pure(hinge.to_state()), 1.0)
        assert math.isclose(entanglement_entropy_pure(hinge.to_state()), 1.0)
        assert relational_level(hinge) is RelationalLevel.BELL


def test_bell_local_paulis_update_only_relational_sector():
    phi_plus = BellHinge(1, 1)
    assert phi_plus.apply_local_pauli("z") == BellHinge(1, -1)
    assert phi_plus.apply_local_pauli("x") == BellHinge(-1, 1)
    assert phi_plus.apply_local_pauli("xz") == BellHinge(-1, -1)


def test_same_axis_hinges_compose_without_promotion():
    left = AxisHinge("xx", math.pi / 8)
    right = AxisHinge("xx", math.pi / 8)
    result = compose_relations(left, right)
    assert isinstance(result, AxisHinge)
    assert result.axis == "xx"
    assert math.isclose(result.theta, math.pi / 4)
    assert np.allclose(result.to_unitary(), left.to_unitary() @ right.to_unitary())


def test_different_commuting_axes_promote_to_cartan_relation():
    xx = AxisHinge("xx", 0.2)
    zz = AxisHinge("zz", -0.3)
    result = compose_relations(xx, zz)
    assert isinstance(result, CartanRelation)
    assert np.allclose((result.c1, result.c2, result.c3), (0.2, 0.0, -0.3))
    assert np.allclose(result.to_unitary(), xx.to_unitary() @ zz.to_unitary())


def test_cartan_minimizes_back_to_axis_hinge():
    relation = CartanRelation(0.0, 0.37, 0.0)
    result = relation.minimize()
    assert result == AxisHinge("yy", 0.37)


def test_axis_promotion_preserves_exact_unitary():
    axis = AxisHinge("zz", 0.41)
    cartan = axis.promote()
    block = cartan.promote()
    assert np.allclose(axis.to_unitary(), cartan.to_unitary())
    assert np.allclose(axis.to_unitary(), block.to_unitary())
