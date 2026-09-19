"""Analytic real Pauli-coefficient actions for AxisHinge and CX operations.

Coefficient tensors index I,X,Y,Z on each axis. No dense quantum gate matrix
is constructed. These are forward U rho U† actions, including signed operators.
"""
import math
import numpy as np

# Pauli products are discrete algebra, not numerically decomposed gate matrices.
_M={('X','Y'):(1j,'Z'),('Y','Z'):(1j,'X'),('Z','X'):(1j,'Y'),
    ('Y','X'):(-1j,'Z'),('Z','Y'):(-1j,'X'),('X','Z'):(-1j,'Y')}
_LABELS='IXYZ'

def _mul(a,b):
    if a=='I':return 1,b
    if b=='I':return 1,a
    if a==b:return 1,'I'
    return _M[a,b]


def _rotation_table(axis):
    source=[];target=[];sign=[]
    for i,a in enumerate(_LABELS):
        for j,b in enumerate(_LABELS):
            if sum(x!='I' and x!=axis for x in (a,b))%2:
                pa,aa=_mul(axis,a);pb,bb=_mul(axis,b)
                source.append(4*i+j);target.append(4*_LABELS.index(aa)+_LABELS.index(bb))
                sign.append(float((-1j*pa*pb).real))
    return np.array(source),np.array(target),np.array(sign)

_ROT={a:_rotation_table(a) for a in 'XYZ'}


def _cx_table():
    # Images of X and Z generators, with Y=iXZ; first coordinate is control.
    generators={('X',0):'XX',('Z',0):'ZI',('X',1):'IX',('Z',1):'ZZ'}
    target=[];sign=[]
    for a in _LABELS:
        for b in _LABELS:
            result='II';phase=1+0j
            for q,label in enumerate((a,b)):
                kinds=() if label=='I' else ('X','Z') if label=='Y' else (label,)
                if label=='Y':phase*=1j
                for kind in kinds:
                    mapped=generators[kind,q];p,x=_mul(result[0],mapped[0]);r,y=_mul(result[1],mapped[1])
                    phase*=p*r;result=x+y
            target.append(4*_LABELS.index(result[0])+_LABELS.index(result[1]));sign.append(float(phase.real))
    return np.array(target),np.array(sign)

_CX=_cx_table()


def apply_pair_coefficients(coefficients, axes, gate, angle=0.0):
    """Apply a hinge or CX to two coefficient axes; CX axes are control,target."""
    coefficients=np.asarray(coefficients,dtype=float)
    axes=list(axes);order=axes+[q for q in range(coefficients.ndim) if q not in axes]
    moved=coefficients.transpose(order);flat=moved.reshape(16,-1)
    if gate=='cx':
        target,sign=_CX;out=np.empty_like(flat);out[target]=sign[:,None]*flat
    elif gate in ('rxx','ryy','rzz'):
        source,target,sign=_ROT[gate[1].upper()]
        out=flat.copy();out[source]=math.cos(angle)*flat[source]
        out[target]+=math.sin(angle)*sign[:,None]*flat[source]
    else:raise ValueError(f'unsupported analytic pair gate {gate}')
    return out.reshape(moved.shape).transpose(np.argsort(order))
