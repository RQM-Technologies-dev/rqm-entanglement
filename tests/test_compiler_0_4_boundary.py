import tomllib
from pathlib import Path

import numpy as np

from rqm_entanglement import (
    AxisHinge,
    CartanRelation,
    QuaternionCartanBlock,
    compose_relations,
    xx_rotation,
    yy_rotation,
    zz_rotation,
)

ROOT=Path(__file__).resolve().parents[1]

def test_entanglement_package_does_not_depend_on_compiler():
    data=tomllib.loads((ROOT/"pyproject.toml").read_text())
    deps=data["project"]["dependencies"]
    assert not any("rqm-compiler" in d for d in deps)

def test_relational_types_are_canonical_public_api():
    a=AxisHinge("xx",0.2)
    c=CartanRelation(0.2,0.0,0.0)
    assert np.allclose(a.to_unitary(),c.to_unitary(),atol=1e-12)
    assert isinstance(c.promote(),QuaternionCartanBlock)

def test_canonical_pair_rotations_match_hinges():
    for axis,fn in (("xx",xx_rotation),("yy",yy_rotation),("zz",zz_rotation)):
        h=AxisHinge(axis,0.37)
        assert np.allclose(h.to_unitary(),fn(0.37),atol=1e-12)

def test_relational_composition_stays_owned_here():
    out=compose_relations(AxisHinge("xx",0.1),AxisHinge("zz",0.2))
    assert isinstance(out,CartanRelation)
